from __future__ import annotations

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeVar, cast
from unittest.mock import Mock

import pytest
import requests

import runners.jev_runner as _jev_module
from runners.jev_runner import JevRunner

jev_module = cast(Any, _jev_module)
FixtureFunction = TypeVar("FixtureFunction", bound=Callable[..., Any])


def fixture(*, autouse: bool = False) -> Callable[[FixtureFunction], FixtureFunction]:
    return cast(Callable[[FixtureFunction], FixtureFunction], pytest.fixture(autouse=autouse))


BASE_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
TEST_KEY = "test-key-not-a-secret"


def _forbidden_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("network access is forbidden")


@fixture(autouse=True)
def do_not_read_real_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jev_module, "load_dotenv", lambda **_: False, raising=False)
    monkeypatch.setattr(jev_module.requests, "post", _forbidden_network)
    monkeypatch.setattr(jev_module.requests.sessions.Session, "post", _forbidden_network)
    monkeypatch.setattr(jev_module.requests.sessions.Session, "request", _forbidden_network)
    monkeypatch.setattr(jev_module.requests.sessions.Session, "send", _forbidden_network)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)
    monkeypatch.delenv("JEV_BASE_URL", raising=False)


def make_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str = MODEL,
    base_url: str = BASE_URL,
) -> JevRunner:
    monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)
    return JevRunner(model=model, base_url=base_url)


def make_response(
    payload: object,
    *,
    status_code: int = 200,
    raise_error: bool = False,
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    if raise_error:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None
    return response


def noul_questions() -> dict[str, dict[str, object]]:
    return {"intent": {"type": "noul"}}


def valid_questions() -> dict[str, dict[str, object]]:
    return {
        "intent": {"type": "noul"},
        "choice": {
            "type": "choice",
            "criteria": {"yes": "Yes", "no": "No"},
        },
        "score": {
            "type": "score",
            "criteria": ["low", "high"],
            "max_score": 1,
        },
    }


def valid_payload(
    *,
    model: str = MODEL,
    usage: object | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "answers": {
            "intent": {"type": "boolean", "boolean": True},
            "choice": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.123456789, "no": 0.876543211},
                "confidence": 0.123456789,
            },
            "score": {"type": "score", "score": 1, "confidence": 0.75},
        },
        "usage": usage if usage is not None else {"input_tokens": 3, "output_tokens": 1},
    }


def test_normalizes_vendor_answer_aliases_and_top_level_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    questions = {
        "choice": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}},
        "score": {"type": "score", "criteria": ["low", "high"], "max_score": 1},
    }
    response = make_response(
        {
            "model": MODEL,
            "choice": {
                "type": "choice",
                "label": "yes",
                "probs": {"yes": 0.25, "no": 0.75},
                "confidence": 0.25,
            },
            "score": {"type": "score", "value": 1, "confidence": 0.75},
        }
    )
    runner.session.post = Mock(return_value=response)

    result = runner.predict({"message": "hello"}, questions)

    assert result["answers"]["choice"]["probabilities"] == {"yes": 0.25, "no": 0.75}
    assert result["answers"]["score"] == {"type": "score", "score": 1, "confidence": 0.75}


def test_loads_repository_root_dotenv_before_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = Mock(return_value=False)
    monkeypatch.setattr(jev_module, "load_dotenv", load)
    monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)

    JevRunner()

    load.assert_called_once_with(
        dotenv_path=jev_module.REPOSITORY_ROOT / ".env",
        override=False,
    )


def test_dotenv_errors_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    load = Mock(side_effect=RuntimeError(f"environment contained {TEST_KEY}"))
    monkeypatch.setattr(jev_module, "load_dotenv", load)
    monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)

    with pytest.raises(RuntimeError, match="environment") as error:
        JevRunner()

    assert TEST_KEY not in str(error.value)


def test_default_model_and_base_url_come_from_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)
    monkeypatch.setenv("JEV_MODEL", MODEL)
    monkeypatch.setenv("JEV_BASE_URL", BASE_URL)

    runner = JevRunner()

    assert runner.model == MODEL
    assert runner.name == MODEL
    assert runner.base_url == BASE_URL
    assert runner.info()["runner"] == MODEL
    assert runner.info()["model"] == MODEL
    assert {"revision", "serving", "device", "adapter_version"} <= set(runner.info())


def test_defaults_apply_when_environment_values_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)

    runner = JevRunner()

    assert runner.model == MODEL
    assert runner.base_url == BASE_URL


def test_prefixed_or_unsafe_model_ids_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for model in (
        "typesafe:jev-1.13.0",
        "jev-1.13",
        "jev-1.13.0-extra",
        "https://jev-1.13.0",
        "jev-1.13.0?token=credential",
    ):
        monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)

        with pytest.raises(ValueError, match="bare model ID"):
            JevRunner(model=model)


def test_missing_key_is_rejected_without_constructing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock(side_effect=AssertionError("session must not be constructed"))
    monkeypatch.setattr(jev_module.requests, "Session", session)

    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevRunner()


def test_unsafe_base_urls_are_rejected_before_session_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for base_url in (
        "http://api.typesafe.ai/v1/systemone",
        "https://user:password@api.typesafe.ai/v1/systemone",
        "https://api.typesafe.ai/v1/systemone?token=value",
        "https://api.typesafe.ai/v1/systemone#fragment",
        "https://api.typesafe.ai/v1/systemone?",
        "https:///v1/systemone",
        "https://api.typesafe.ai:bad/v1/systemone",
    ):
        monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)
        session = Mock(side_effect=AssertionError("session must not be constructed"))
        monkeypatch.setattr(jev_module.requests, "Session", session)

        with pytest.raises(ValueError):
            JevRunner(base_url=base_url)


def test_runner_uses_one_session_and_separate_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()
    monkeypatch.setenv("TYPESAFE_API_KEY", TEST_KEY)
    monkeypatch.setattr(jev_module.requests, "post", Mock(side_effect=AssertionError("no network")))
    monkeypatch.setattr(jev_module.requests, "Session", Mock(return_value=session))
    response = make_response({"model": MODEL, "answers": {"intent": {"type": "noul", "noul": 0.4}}})
    session.post.return_value = response

    runner = JevRunner()
    runner.predict({"message": "hello"}, noul_questions())

    assert runner.session is session
    assert session.post.call_count == 1
    timeout = session.post.call_args.kwargs["timeout"]
    assert isinstance(timeout, tuple)
    assert len(timeout) == 2
    assert timeout[0] != timeout[1]
    headers = session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == f"Bearer {TEST_KEY}"


def test_network_guard_blocks_unoverridden_session_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)

    with pytest.raises(AssertionError, match="network access is forbidden"):
        runner.session.post(BASE_URL)
    with pytest.raises(AssertionError, match="network access is forbidden"):
        runner.session.request("POST", BASE_URL)
    with pytest.raises(AssertionError, match="network access is forbidden"):
        jev_module.requests.post(BASE_URL)
    with pytest.raises(AssertionError, match="network access is forbidden"):
        runner.session.send(cast(Any, None))

    instance_response = Mock()
    runner.session.post = Mock(return_value=instance_response)
    assert runner.session.post(BASE_URL) is instance_response


def test_returned_model_must_match_requested_model_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    response = make_response(valid_payload(model="jev-1.12.0"))
    post = Mock(return_value=response)
    runner.session.post = post

    with pytest.raises(RuntimeError, match="returned model"):
        runner.predict({"message": "hello"}, valid_questions())

    assert post.call_count == 1


def test_retry_stops_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    post = Mock(side_effect=TimeoutError("timeout"))
    runner.session.post = post
    sleep = Mock()
    monkeypatch.setattr(jev_module.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="three attempts"):
        runner.predict({"message": "hello"}, noul_questions())

    assert post.call_count == 3
    assert sleep.call_count == 2


def test_retry_attempts_receive_pristine_payloads_and_preserve_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    state = {"message": "hello", "nested": {"value": 1}}
    questions = valid_questions()
    original_state = deepcopy(state)
    original_questions = deepcopy(questions)
    calls: list[dict[str, Any]] = []

    def post(*args: object, **kwargs: object) -> Mock:
        request = kwargs["json"]
        assert isinstance(request, dict)
        calls.append(deepcopy(request))
        if len(calls) == 1:
            request["state"]["adapter_mutation"] = True
            request["questions"]["intent"]["adapter_mutation"] = True
            raise requests.exceptions.Timeout("timeout")
        return make_response(valid_payload())

    runner.session.post = post
    monkeypatch.setattr(jev_module.time, "sleep", Mock())

    result = runner.predict(state, questions)

    expected_request = {"model": MODEL, "state": original_state, "questions": original_questions}
    assert calls == [expected_request, expected_request]
    assert state == original_state
    assert questions == original_questions
    assert result["answers"]["intent"] == {"type": "noul", "noul": 1.0}


def test_retryable_statuses_retry_then_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for status_code in (429, 500, 503):
        runner = make_runner(monkeypatch)
        retry_response = make_response({}, status_code=status_code)
        success_response = make_response(
            {"model": MODEL, "answers": {"intent": {"type": "noul", "noul": 0.4}}}
        )
        post = Mock(side_effect=[retry_response, success_response])
        runner.session.post = post
        sleep = Mock()
        monkeypatch.setattr(jev_module.time, "sleep", sleep)

        result: dict[str, Any] = runner.predict({"message": "hello"}, noul_questions())

        assert post.call_count == 2
        assert sleep.call_count == 1
        assert result["answers"] == {"intent": {"type": "noul", "noul": 0.4}}


def test_non_retryable_statuses_fail_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for status_code in (400, 401, 404, 422):
        runner = make_runner(monkeypatch)
        response = make_response(
            {"error": "body must not be exposed"}, status_code=status_code, raise_error=True
        )
        post = Mock(return_value=response)
        runner.session.post = post
        sleep = Mock()
        monkeypatch.setattr(jev_module.time, "sleep", sleep)

        with pytest.raises(RuntimeError, match=f"HTTP {status_code}") as error:
            runner.predict({"message": "hello"}, noul_questions())

        assert post.call_count == 1
        assert sleep.call_count == 0
        assert "body must not be exposed" not in str(error.value)


def test_malformed_response_is_not_retried_and_error_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    response = make_response({"model": MODEL, "answers": {"intent": {"type": "noul", "noul": 2}}})
    post = Mock(return_value=response)
    runner.session.post = post

    with pytest.raises(RuntimeError, match="response") as error:
        runner.predict({"message": "hello"}, noul_questions())

    assert post.call_count == 1
    assert TEST_KEY not in str(error.value)


def test_malformed_json_is_not_retried_and_does_not_expose_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    response = make_response({})
    response.json.side_effect = ValueError(f"raw body contained {TEST_KEY}")
    post = Mock(return_value=response)
    runner.session.post = post

    with pytest.raises(RuntimeError, match="response") as error:
        runner.predict({"message": "hello"}, noul_questions())

    assert post.call_count == 1
    assert TEST_KEY not in str(error.value)


def test_missing_or_extra_answer_ids_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for answer_ids in (set(), {"intent", "extra"}):
        runner = make_runner(monkeypatch)
        answers = {qid: {"type": "noul", "noul": 0.5} for qid in answer_ids}
        response = make_response({"model": MODEL, "answers": answers})
        post = Mock(return_value=response)
        runner.session.post = post

        with pytest.raises(RuntimeError, match="answer IDs"):
            runner.predict({"message": "hello"}, noul_questions())

        assert post.call_count == 1


def test_invalid_normalized_answers_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    response = make_response(
        {
            "model": MODEL,
            "answers": {
                "intent": {"type": "noul", "noul": 0.5},
                "choice": {
                    "type": "choice",
                    "choice": "yes",
                    "probabilities": {"yes": 0.2},
                    "confidence": 0.2,
                },
            },
        }
    )
    runner.session.post = Mock(return_value=response)

    with pytest.raises(RuntimeError, match="response"):
        runner.predict({"message": "hello"}, valid_questions())


def test_non_mapping_questions_fail_closed_with_matching_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    response = make_response({"model": MODEL, "answers": {"intent": {"type": "noul", "noul": 0.5}}})
    post = Mock(return_value=response)
    runner.session.post = post

    with pytest.raises(RuntimeError, match="question"):
        runner.predict({"message": "hello"}, {"intent": "not-a-mapping"})

    assert post.call_count == 1


def test_usage_returned_model_and_full_probability_map_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    usage: dict[str, Any] = {
        "input_tokens": 11,
        "output_tokens": 2,
        "nested": {"provider_total": 13},
    }
    response = make_response(valid_payload(usage=usage))
    runner.session.post = Mock(return_value=response)

    result: dict[str, Any] = runner.predict({"message": "hello"}, valid_questions())

    assert result["answers"]["choice"]["probabilities"] == {
        "yes": 0.123456789,
        "no": 0.876543211,
    }
    assert result["_usage"] == usage
    assert result["_raw_model"] == MODEL
    result["_usage"]["nested"]["provider_total"] = 99
    assert usage["nested"]["provider_total"] == 13


def test_rounded_choice_probabilities_are_resummed_onto_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    questions = {
        "choice": {
            "type": "choice",
            "criteria": {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"},
        }
    }
    payload = {
        "model": MODEL,
        "answers": {
            "choice": {
                "type": "choice",
                "choice": "c",
                # Transport rounding: the raw values sum to 0.9999.
                "probabilities": {
                    "a": 0.1,
                    "b": 0.2,
                    "c": 0.2999,
                    "d": 0.2,
                    "e": 0.2,
                },
                "confidence": 0.2999,
            }
        },
    }
    assert abs(sum(payload["answers"]["choice"]["probabilities"].values()) - 1.0) > 1e-6
    runner.session.post = Mock(return_value=make_response(payload))

    result = runner.predict({"message": "hello"}, questions)

    projected = result["answers"]["choice"]["probabilities"]
    assert set(projected) == {"a", "b", "c", "d", "e"}
    assert abs(sum(projected.values()) - 1.0) <= 1e-12
    assert max(projected, key=lambda label: projected[label]) == "c"
    assert result["answers"]["choice"]["choice"] == "c"


def test_impossible_choice_probabilities_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = {"choice": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}}
    for probabilities in ({}, {"yes": 0.0, "no": 0.0}, {"yes": -1.0, "no": 2.0}):
        runner = make_runner(monkeypatch)
        payload = {
            "model": MODEL,
            "answers": {
                "choice": {
                    "type": "choice",
                    "choice": "yes",
                    "probabilities": probabilities,
                    "confidence": 0.5,
                }
            },
        }
        runner.session.post = Mock(return_value=make_response(payload))

        with pytest.raises(RuntimeError, match="response"):
            runner.predict({"message": "hello"}, questions)


def test_score_vendor_extras_are_projected_onto_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    questions = {
        "score": {"type": "score", "criteria": ["low", "mid", "high"], "max_score": 2}
    }
    payload = {
        "model": MODEL,
        "answers": {
            "score": {
                "type": "score",
                "score": 1,
                "confidence": 0.5,
                "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
            }
        },
    }
    runner.session.post = Mock(return_value=make_response(payload))

    result = runner.predict({"message": "hello"}, questions)

    assert result["answers"]["score"] == {"type": "score", "score": 1, "confidence": 0.5}


def test_noul_vendor_extras_are_projected_onto_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    payload = {
        "model": MODEL,
        "answers": {
            "intent": {"type": "noul", "noul": 0.75, "confidence": 0.75, "explanation": "x"}
        },
    }
    runner.session.post = Mock(return_value=make_response(payload))

    result = runner.predict({"message": "hello"}, noul_questions())

    assert result["answers"]["intent"] == {"type": "noul", "noul": 0.75}


def test_answers_missing_declared_contract_fields_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        ({"type": "noul"}, noul_questions()),
        (
            {"type": "score", "score": 1},
            {"score": {"type": "score", "criteria": ["low", "high"], "max_score": 1}},
        ),
        (
            {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.5, "no": 0.5}},
            {"choice": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}},
        ),
    )
    for answer, questions in cases:
        runner = make_runner(monkeypatch)
        runner.session.post = Mock(
            return_value=make_response({"model": MODEL, "answers": dict.fromkeys(questions, answer)})
        )

        with pytest.raises(RuntimeError, match="response"):
            runner.predict({"message": "hello"}, questions)


def test_response_digest_redacts_secrets_and_tracks_material_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    payload = valid_payload()
    payload["secret"] = TEST_KEY
    payload["headers"] = {"Authorization": f"Bearer {TEST_KEY}"}

    def response_digest(response_payload: dict[str, Any]) -> str:
        runner.session.post = Mock(return_value=make_response(response_payload))
        result = runner.predict({"message": "hello"}, valid_questions())
        return cast(str, result["_response_digest"])

    baseline = response_digest(payload)
    secret_variant = deepcopy(payload)
    secret_variant["secret"] = "different-secret-value"
    secret_variant["headers"] = {"Authorization": "Bearer different-authorization"}
    material_variant = deepcopy(payload)
    material_variant["answers"]["score"]["score"] = 0

    assert re.fullmatch(r"[0-9a-f]{64}", baseline)
    assert response_digest(secret_variant) == baseline
    assert response_digest(material_variant) != baseline


def test_response_digest_preserves_token_usage_and_redacts_secret_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    payload = valid_payload(
        usage={
            "input_tokens": 3,
            "output_tokens": 1,
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
            "cached_tokens": 0,
        }
    )
    payload["auth_token"] = "auth-secret"
    payload["access_token"] = "access-secret"
    payload["secret_token"] = "secret-secret"

    def response_digest(response_payload: dict[str, Any]) -> str:
        runner.session.post = Mock(return_value=make_response(response_payload))
        result = runner.predict({"message": "hello"}, valid_questions())
        return cast(str, result["_response_digest"])

    baseline = response_digest(payload)
    usage_variant = deepcopy(payload)
    usage_variant["usage"] = {
        "input_tokens": 4,
        "output_tokens": 2,
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 9,
        "cached_tokens": 1,
    }
    secret_variant = deepcopy(payload)
    secret_variant["auth_token"] = "different-auth-secret"
    secret_variant["access_token"] = "different-access-secret"
    secret_variant["secret_token"] = "different-secret-secret"

    assert re.fullmatch(r"[0-9a-f]{64}", baseline)
    assert response_digest(usage_variant) != baseline
    assert response_digest(secret_variant) == baseline


def test_predict_does_not_mutate_state_questions_or_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    state = {"message": "hello", "nested": {"value": 1}}
    questions = valid_questions()
    original_state = deepcopy(state)
    original_questions = deepcopy(questions)
    payload = valid_payload()

    def post(*args: object, **kwargs: object) -> Mock:
        request = kwargs["json"]
        assert isinstance(request, dict)
        request["state"]["adapter_mutation"] = True
        request["questions"]["intent"]["adapter_mutation"] = True
        return make_response(payload)

    runner.session.post = post
    result: dict[str, Any] = runner.predict(state, questions)

    assert state == original_state
    assert questions == original_questions
    assert result["answers"]["intent"] == {"type": "noul", "noul": 1.0}
    assert payload["answers"]["intent"] == {"type": "boolean", "boolean": True}


def test_unexpected_transport_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    post = Mock(side_effect=Exception(f"raw response body contained {TEST_KEY}"))
    runner.session.post = post
    monkeypatch.setattr(jev_module.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="request") as error:
        runner.predict({"message": "hello"}, noul_questions())

    assert post.call_count == 1
    assert TEST_KEY not in str(error.value)


def test_retry_error_does_not_include_key_or_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(monkeypatch)
    post = Mock(side_effect=requests.exceptions.ConnectionError(f"Authorization: {TEST_KEY}"))
    runner.session.post = post
    monkeypatch.setattr(jev_module.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="three attempts") as error:
        runner.predict({"message": "hello"}, noul_questions())

    assert TEST_KEY not in str(error.value)
    assert "Authorization" not in str(error.value)
