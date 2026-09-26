"""Laya adapter: local open-weights inference via the laya package."""

import os
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .base import BaseRunner, renormalize_choice_probabilities

ADAPTER_VERSION = "1"
DEFAULT_REPO = "convaiinnovations/laya"
DEFAULT_CHECKPOINT = "root"

laya: Any | None = None

_REVISION_ATTRIBUTES = (
    "revision",
    "checkpoint_revision",
    "model_revision",
    "commit",
    "commit_hash",
    "_commit_hash",
)
_DEVICE_ATTRIBUTES = ("device",)
_CHECKPOINT_ATTRIBUTES = ("checkpoint", "checkpoint_name")
_REPOSITORY_ATTRIBUTES = (
    "resolved_repo",
    "model_repo",
    "_name_or_path",
    "name_or_path",
    "repository",
    "repo",
    "model_id",
)
_MODEL_ATTRIBUTES = ("resolved_model", "model_id", "model_name")
_PACKAGE_VERSION_ATTRIBUTES = ("__version__", "version", "package_version")
_HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_ANSWER_FIELDS: dict[str, tuple[str, ...]] = {
    "choice": ("type", "choice", "probabilities", "confidence"),
    "score": ("type", "score", "confidence"),
    "noul": ("type", "noul"),
}


def _get_laya() -> Any:
    global laya
    if laya is None:
        import laya as package

        laya = package
    return laya


def _text(value: object) -> str | None:
    if value is None or isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return None
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _attribute_value(value: object, names: tuple[str, ...]) -> str | None:
    for name in names:
        candidate = getattr(value, name, None)
        text = _text(candidate)
        if text is not None:
            return text
    if isinstance(value, Mapping):
        for name in names:
            text = _text(value.get(name))
            if text is not None:
                return text
    return None


def _identity_values(agent: Any) -> tuple[object, ...]:
    model = getattr(agent, "model", None)
    config = getattr(model, "config", None)
    return (
        agent,
        model,
        config,
        getattr(agent, "cfg", None),
        getattr(config, "model_config", None),
    )


def _required_identity_value(agent: Any, names: tuple[str, ...], label: str) -> str:
    for value in _identity_values(agent):
        resolved = _attribute_value(value, names)
        if resolved is not None:
            return resolved
    raise ValueError(f"Laya identity missing {label}")


def _is_local_model_reference(repo: str) -> bool:
    return repo.startswith(("/", "./", "../")) or os.path.isabs(repo)


def _cached_snapshot_revision(repo: str, subfolder: str | None = None) -> str | None:
    """Read the exact commit already resolved in the local Hugging Face cache."""
    if not repo or _is_local_model_reference(repo):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    filename = f"{subfolder}/rl_agent_config.json" if subfolder else "rl_agent_config.json"
    try:
        cached = try_to_load_from_cache(repo, filename)
    except (OSError, ValueError):
        return None
    if not isinstance(cached, str) or not cached:
        return None
    commit = Path(cached).parent.name
    if _HF_COMMIT_PATTERN.fullmatch(commit) is None:
        return None
    return commit


def _resolved_revision(agent: Any, requested_repo: str | None, subfolder: str | None) -> str:
    for value in _identity_values(agent):
        resolved = _attribute_value(value, _REVISION_ATTRIBUTES)
        if resolved is not None:
            return resolved
    if requested_repo is not None:
        cached = _cached_snapshot_revision(requested_repo, subfolder)
        if cached is not None:
            return cached
    raise ValueError("Laya identity missing revision")


def _device_name(agent: Any) -> str:
    return _required_identity_value(agent, _DEVICE_ATTRIBUTES, "device")


def _checkpoint_name(agent: Any, requested_checkpoint: str | None) -> str:
    for value in _identity_values(agent):
        resolved = _attribute_value(value, _CHECKPOINT_ATTRIBUTES)
        if resolved is not None:
            return resolved
    if requested_checkpoint is not None:
        return requested_checkpoint
    raise ValueError("Laya identity missing checkpoint")


def _resolved_repository(agent: Any) -> str | None:
    values = _identity_values(agent)
    for name in _REPOSITORY_ATTRIBUTES:
        for value in values:
            resolved = _attribute_value(value, (name,))
            if resolved is not None:
                return resolved
    return None


def _resolved_model(agent: Any) -> str | None:
    values = _identity_values(agent)
    for name in _MODEL_ATTRIBUTES:
        for value in values:
            resolved = _attribute_value(value, (name,))
            if resolved is not None:
                return resolved
    return None


def _package_version(package: Any) -> str:
    version = _attribute_value(package, _PACKAGE_VERSION_ATTRIBUTES)
    if version is None:
        raise ValueError("Laya identity missing package version")
    return version


def _renormalized_probabilities(probabilities: object, qid: str) -> dict[str, float]:
    return renormalize_choice_probabilities(probabilities, qid)


def normalize_answers(
    questions: Mapping[str, Any], answers: Mapping[Any, Any]
) -> dict[str, dict[str, Any]]:
    """Project Laya's richer answer payloads onto the benchmark answer contract.

    Laya returns vendor extras (``action`` on every type, plus ``legend``/``probabilities`` on
    score and ``confidence`` on noul). The benchmark contract is exact, so the adapter keeps
    only the declared fields and never invents a value it did not receive.
    """
    normalized: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        if not isinstance(question, Mapping):
            raise TypeError(f"{qid}: question must be a mapping")
        question_type = question.get("type")
        if not isinstance(question_type, str) or question_type not in _ANSWER_FIELDS:
            raise ValueError(f"{qid}: question type is not a supported Laya answer type")
        answer = answers[qid]
        if not isinstance(answer, Mapping):
            raise TypeError(f"{qid}: Laya answer must be a mapping")
        if answer.get("type") != question_type:
            raise ValueError(f"{qid}: Laya answer type does not match the declared question type")
        fields = _ANSWER_FIELDS[question_type]
        missing = [field for field in fields if field not in answer]
        if missing:
            raise ValueError(f"{qid}: Laya answer is missing declared field(s): {missing}")
        projected = {field: deepcopy(answer[field]) for field in fields}
        if question_type == "choice":
            projected["probabilities"] = _renormalized_probabilities(
                projected["probabilities"], qid
            )
            criteria = question.get("criteria")
            if isinstance(criteria, Mapping) and set(projected["probabilities"]) != set(criteria):
                raise ValueError(f"{qid}: Laya probability labels do not match the question criteria")
        normalized[qid] = projected
    return normalized


def _required_text(value: object, label: str) -> str:
    text = _text(value)
    if text is None:
        raise ValueError(f"Laya identity missing {label}")
    return text


def _build_laya_metadata(
    requested_repo: str | None,
    agent: Any,
    runner: str,
    serving: str,
    package: Any | None = None,
    *,
    requested_checkpoint: str | None = None,
    checkpoint_subfolder: str | None = None,
) -> dict[str, Any]:
    package_obj = _get_laya() if package is None else package
    resolved_repo = _resolved_repository(agent)
    if requested_repo is None:
        if resolved_repo is None:
            raise ValueError("Router identity missing resolved repository")
        requested_text = None
    else:
        requested_text = _required_text(requested_repo, "requested repository")
    revision = _resolved_revision(agent, requested_repo, checkpoint_subfolder)
    device = _device_name(agent)
    checkpoint = _checkpoint_name(agent, requested_checkpoint)
    package_version = _package_version(package_obj)
    if requested_checkpoint is not None and checkpoint != requested_checkpoint:
        raise ValueError(
            f"Laya checkpoint identity does not match requested checkpoint {requested_checkpoint!r}"
        )
    resolved_model = _resolved_model(agent)
    return {
        "runner": runner,
        "model": requested_text if requested_text is not None else resolved_repo,
        "revision": revision,
        "serving": serving,
        "device": device,
        "adapter_version": ADAPTER_VERSION,
        "repo": requested_text,
        "requested_repo": requested_text,
        "resolved_repo": resolved_repo,
        "resolved_model": resolved_model,
        "checkpoint": checkpoint,
        "package_version": package_version,
        "requested_checkpoint": requested_checkpoint,
        "resolved": {
            "repo": resolved_repo,
            "model": resolved_model,
            "revision": revision,
            "device": device,
            "checkpoint": checkpoint,
            "package_version": package_version,
        },
    }


class LayaRunner(BaseRunner):
    name = "laya"

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        *,
        package: Any | None = None,
        subfolder: str | None = None,
    ) -> None:
        requested_repo = _required_text(repo, "requested repository")
        package_obj = _get_laya() if package is None else package
        loader = getattr(package_obj, "load", None)
        if not callable(loader):
            raise TypeError("Laya package does not expose a callable load()")
        checkpoint = (
            _required_text(subfolder, "requested checkpoint")
            if subfolder is not None
            else DEFAULT_CHECKPOINT
        )
        load_kwargs: dict[str, Any] = {} if subfolder is None else {"subfolder": checkpoint}
        self.repo = requested_repo
        self.agent = loader(requested_repo, **load_kwargs)
        self._metadata = _build_laya_metadata(
            self.repo,
            self.agent,
            self.name,
            "local",
            package=package_obj,
            requested_checkpoint=checkpoint,
            checkpoint_subfolder=subfolder,
        )

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        result = self.agent.predict(copied_state, copied_questions)
        if not isinstance(result, Mapping):
            raise TypeError("Laya adapter result must be a mapping")
        snapshot = deepcopy(dict(result))
        answers = self._validate_answer_ids(questions, snapshot.get("answers"))
        output: dict[str, Any] = {"answers": normalize_answers(questions, answers)}
        routing, _, _ = self._routing_metadata(snapshot)
        if routing is not None:
            output["_routing"] = deepcopy(dict(routing))
        optional_fields = {
            "_usage": ("usage", "_usage"),
            "_raw_model": ("model", "_raw_model", "raw_model"),
        }
        for target, sources in optional_fields.items():
            for source in sources:
                if source in snapshot:
                    output[target] = deepcopy(snapshot[source])
                    break
        return output

    def info(self) -> dict[str, Any]:
        return super().info()
