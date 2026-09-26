"""Jev adapter for the TypeSafe Decisions API."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from benchmark import contracts
from benchmark.canonical import digest_value

from .base import BaseRunner, renormalize_choice_probabilities

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIRECT_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
ADAPTER_VERSION = "1"
MAX_ATTEMPTS = 3
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120
RETRY_BACKOFF_BASE = 0.1
RETRY_BACKOFF_MAX = 1.0
MODEL_PATTERN = re.compile(r"^jev-[0-9]+\.[0-9]+\.[0-9]+$")
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionError,
    TimeoutError,
)
_REDACTED_KEY_PARTS = (
    "authorization",
    "apikey",
    "accesskey",
    "privatekey",
    "token",
    "secret",
    "password",
    "credential",
)
_REDACTED_RESPONSE_KEYS = frozenset(
    {
        "body",
        "raw_body",
        "response_body",
        "raw_response",
        "response_text",
    }
)
_SENSITIVE_TOKEN_USAGE_PARTS = (
    "auth",
    "access",
    "secret",
    "apikey",
    "api",
    "password",
    "credential",
    "privatekey",
    "bearer",
    "refresh",
    "session",
    "client",
)


class _ExceptionCapture:
    def __init__(self) -> None:
        self.error: Exception | None = None

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if isinstance(exception, Exception):
            self.error = exception
            return True
        return False


def _response_error(message: str) -> RuntimeError:
    return RuntimeError(message)


def _load_environment() -> None:
    capture = _ExceptionCapture()
    with capture:
        load_dotenv(dotenv_path=REPOSITORY_ROOT / ".env", override=False)
    if capture.error is not None:
        raise _response_error("Jev environment configuration could not be loaded") from None


def _validate_model(value: object) -> str:
    if not isinstance(value, str) or MODEL_PATTERN.fullmatch(value) is None:
        raise ValueError("Jev model must be a bare model ID matching jev-<major>.<minor>.<patch>")
    return value


def _validate_base_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Jev base URL must be a safe HTTPS URL")
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError("Jev base URL must be a safe HTTPS URL")
    if "?" in value or "#" in value:
        raise ValueError("Jev base URL must not contain a query or fragment")
    try:
        parsed = urlparse(value)
        username = parsed.username
        password = parsed.password
        hostname = parsed.hostname
        _ = parsed.port
    except (TypeError, ValueError):
        raise ValueError("Jev base URL must be a safe HTTPS URL") from None
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or hostname is None
        or username is not None
        or password is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc.endswith(":")
    ):
        raise ValueError("Jev base URL must be a safe HTTPS URL without credentials")
    return value


def _numeric_value(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError("Jev answer probability must be numeric")


_CONTRACT_ANSWER_FIELDS: dict[str, tuple[str, ...]] = {
    "noul": ("type", "noul"),
    "choice": ("type", "choice", "probabilities", "confidence"),
    "score": ("type", "score", "confidence"),
}


def _projected_answer(answer: Mapping[str, Any], answer_type: str) -> dict[str, Any]:
    """Keep only the fields the benchmark contract declares for this answer type.

    Jev returns vendor extras, notably ``legend`` and ``probabilities`` on score answers. The
    contract is exact, so the adapter drops what the contract does not declare rather than
    loosening validation for every runner.
    """
    fields = _CONTRACT_ANSWER_FIELDS[answer_type]
    missing = [field for field in fields if field not in answer]
    if missing:
        raise ValueError(f"Jev {answer_type} answer is missing declared field(s): {missing}")
    return {field: deepcopy(answer[field]) for field in fields}


def _normalized_answer(value: object) -> object:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    answer = deepcopy(dict(value))
    answer_type = answer.get("type")
    if answer_type == "boolean":
        if "boolean" not in answer:
            return answer
        return {"type": "noul", "noul": _numeric_value(answer["boolean"])}
    if answer_type == "noul":
        if "probability" in answer:
            answer["noul"] = answer.pop("probability")
        if "noul" in answer:
            answer["noul"] = _numeric_value(answer["noul"])
        return _projected_answer(answer, "noul")
    if answer_type == "choice":
        if "choice" not in answer:
            for alias in ("label", "value"):
                if alias in answer:
                    answer["choice"] = answer.pop(alias)
                    break
        if "probabilities" not in answer:
            for alias in ("probs", "probability_map"):
                if alias in answer:
                    answer["probabilities"] = answer.pop(alias)
                    break
        if "confidence" in answer and isinstance(answer["confidence"], (int, float)):
            answer["confidence"] = _numeric_value(answer["confidence"])
        if isinstance(answer.get("probabilities"), Mapping):
            answer["probabilities"] = renormalize_choice_probabilities(
                answer["probabilities"], "Jev choice answer"
            )
        return _projected_answer(answer, "choice")
    if answer_type == "score":
        if "score" not in answer:
            for alias in ("value", "rating"):
                if alias in answer:
                    answer["score"] = answer.pop(alias)
                    break
        if "confidence" in answer and isinstance(answer["confidence"], (int, float)):
            answer["confidence"] = _numeric_value(answer["confidence"])
        return _projected_answer(answer, "score")
    return answer


def _redact_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _is_sensitive_key(value: object) -> bool:
    normalized = _redact_key(value)
    if normalized.endswith("tokens") and not any(
        part in normalized for part in _SENSITIVE_TOKEN_USAGE_PARTS
    ):
        return False
    return any(part in normalized for part in _REDACTED_KEY_PARTS)


def _redact_response(value: object, api_key: str) -> object:
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, child in value.items():
            key_text = str(key)
            if _is_sensitive_key(key) or _redact_key(key) in _REDACTED_RESPONSE_KEYS:
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = _redact_response(child, api_key)
        return redacted
    if isinstance(value, list):
        return [_redact_response(child, api_key) for child in value]
    if isinstance(value, tuple):
        return [_redact_response(child, api_key) for child in value]
    if isinstance(value, str) and api_key and api_key in value:
        return value.replace(api_key, "[REDACTED]")
    return deepcopy(value)


def _response_questions(
    questions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for qid, question in questions.items():
        if not isinstance(question, Mapping):
            raise _response_error("Jev request questions were malformed")
        copied_question = deepcopy(dict(question))
        copied_question["qid"] = qid
        result.append(copied_question)
    return result


class JevRunner(BaseRunner):
    name = "jev"

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        _load_environment()
        selected_model = model if model is not None else os.environ.get("JEV_MODEL", DEFAULT_MODEL)
        selected_base_url = (
            base_url if base_url is not None else os.environ.get("JEV_BASE_URL", DIRECT_URL)
        )
        self.model = _validate_model(selected_model)
        self.base_url = _validate_base_url(selected_base_url)
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise RuntimeError("TYPESAFE_API_KEY is required")
        self.api_key = api_key
        self.name = self.model
        self.revision = "unknown"
        self.serving = "api"
        self.device = "unknown"
        self.adapter_version = ADAPTER_VERSION
        self._metadata = {
            "runner": self.name,
            "model": self.model,
            "revision": self.revision,
            "serving": self.serving,
            "device": self.device,
            "adapter_version": self.adapter_version,
            "base_url": self.base_url,
        }
        self.session: requests.Session = requests.Session()

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(RETRY_BACKOFF_BASE * (2**attempt), RETRY_BACKOFF_MAX)
        time.sleep(delay)

    def _retry_failure(self, attempt: int, status: int | None = None) -> None:
        if attempt == MAX_ATTEMPTS - 1:
            suffix = f" with HTTP {status}" if status is not None else ""
            raise _response_error(f"Jev request failed after three attempts{suffix}") from None
        self._sleep_before_retry(attempt)

    def _request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        for attempt in range(MAX_ATTEMPTS):
            response: Any
            capture = _ExceptionCapture()
            with capture:
                response = self.session.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=deepcopy(dict(payload)),
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )
            error = capture.error
            if error is not None:
                if isinstance(error, _RETRYABLE_EXCEPTIONS):
                    self._retry_failure(attempt)
                    continue
                if isinstance(error, requests.exceptions.HTTPError):
                    status = self._status_code(getattr(error, "response", None))
                    if status is not None and self._retryable_status(status):
                        self._retry_failure(attempt, status)
                        continue
                    if status is not None:
                        raise RuntimeError(f"Jev request failed with HTTP {status}") from None
                raise _response_error("Jev request failed") from None

            status = self._status_code(response)
            if status is not None and self._retryable_status(status):
                self._retry_failure(attempt, status)
                continue
            status_capture = _ExceptionCapture()
            with status_capture:
                raise_for_status = getattr(response, "raise_for_status", None)
                if callable(raise_for_status):
                    raise_for_status()
            error = status_capture.error
            if error is not None:
                if isinstance(error, requests.exceptions.HTTPError):
                    error_status = self._status_code(getattr(error, "response", None)) or status
                    if error_status is not None and self._retryable_status(error_status):
                        self._retry_failure(attempt, error_status)
                        continue
                    if error_status is not None:
                        raise RuntimeError(f"Jev request failed with HTTP {error_status}") from None
                elif isinstance(error, _RETRYABLE_EXCEPTIONS):
                    self._retry_failure(attempt)
                    continue
                raise _response_error("Jev request failed") from None
            if status is not None and not 200 <= status < 300:
                raise RuntimeError(f"Jev request failed with HTTP {status}")
            json_capture = _ExceptionCapture()
            with json_capture:
                data = response.json()
            if json_capture.error is not None:
                raise _response_error("Jev response was malformed") from None
            if not isinstance(data, Mapping):
                raise _response_error("Jev response was malformed")

            return data
        raise RuntimeError("Jev request failed after three attempts")

    @staticmethod
    def _status_code(response: object) -> int | None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    @staticmethod
    def _retryable_status(status: int) -> bool:
        return status == 429 or 500 <= status <= 599

    def _validate_response_answers(
        self,
        questions: Mapping[str, Any],
        answers: Mapping[Any, Any],
    ) -> None:
        try:
            self._validate_answer_ids(questions, answers)
        except (TypeError, ValueError):
            raise RuntimeError("Jev response answer IDs did not match questions") from None
        validation_questions = _response_questions(questions)
        validation_capture = _ExceptionCapture()
        with validation_capture:
            contracts.validate_answers(validation_questions, answers)
        if validation_capture.error is not None:
            raise _response_error("Jev response answers failed validation") from None

    def _normalize_response(
        self,
        data: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> dict[str, Any]:
        returned_model = data.get("model")
        if not isinstance(returned_model, str) or returned_model != self.model:
            raise RuntimeError("Jev returned model did not match requested model")
        if "answers" in data:
            raw_answers = data["answers"]
        else:
            raw_answers = {
                key: value for key, value in data.items() if key not in {"model", "usage"}
            }
        if not isinstance(raw_answers, Mapping):
            raise _response_error("Jev response answers were malformed")
        answer_capture = _ExceptionCapture()
        with answer_capture:
            answers = {qid: _normalized_answer(answer) for qid, answer in raw_answers.items()}
        if answer_capture.error is not None:
            raise _response_error("Jev response answers were malformed") from None
        self._validate_response_answers(questions, answers)
        usage = data.get("usage", {})
        if usage is None:
            usage = {}
        if not isinstance(usage, Mapping):
            raise _response_error("Jev response usage was malformed")
        digest_capture = _ExceptionCapture()
        with digest_capture:
            digest = digest_value(_redact_response(data, self.api_key))
        if digest_capture.error is not None:
            raise _response_error("Jev response digest could not be computed") from None
        return {
            "answers": deepcopy(answers),
            "_usage": deepcopy(dict(usage)),
            "_raw_model": returned_model,
            "_response_digest": digest,
        }

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        payload = {"model": self.model, "state": copied_state, "questions": copied_questions}
        data = self._request(payload)
        return self._normalize_response(data, copied_questions)

    def info(self) -> dict[str, Any]:
        return dict(super().info())
