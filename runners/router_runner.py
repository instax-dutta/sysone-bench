"""Router adapter: laya.Router dispatching per request."""

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .base import VALID_PHASES, BaseRunner
from .laya_runner import _build_laya_metadata, _get_laya, _text, normalize_answers

PRELOADED_CHECKPOINTS = ("english", "multilingual")
ROUTE_REASON_COUNTS_KEY = "route_reason_counts"
ROUTE_COUNTS_BY_PHASE_KEY = "route_counts_by_phase"
ROUTE_REASONS_BY_PHASE_KEY = "route_reasons_by_phase"
ROUTE_REASON_COUNTS_BY_PHASE_KEY = "route_reason_counts_by_phase"
ROUTE_REASON_COLLECTIONS_KEY = "route_reason_collections"


class RouterRunner(BaseRunner):
    name = "laya-router"

    def __init__(self, *, repo: str | None = None, package: Any | None = None) -> None:
        package_obj = _get_laya() if package is None else package
        router_class = getattr(package_obj, "Router", None)
        if not callable(router_class):
            raise TypeError("Laya package does not expose a callable Router")

        self.router = router_class(preload=False, max_loaded=2)
        self.router.preload(list(PRELOADED_CHECKPOINTS))
        self._preloaded = self._reported_preloaded()
        self._max_loaded = int(getattr(self.router, "max_loaded", 2))

        self.route_counts: Counter[str] = Counter()
        self.route_reasons: dict[str, str] = {}
        self._route_reason_counts: dict[str, Counter[str]] = {}
        self._route_reason_collections: dict[str, set[str]] = {}
        self._route_counts_by_phase: dict[str, Counter[str]] = {
            phase: Counter() for phase in VALID_PHASES
        }
        self._route_reasons_by_phase: dict[str, dict[str, set[str]]] = {
            phase: {} for phase in VALID_PHASES
        }
        self._route_reason_counts_by_phase: dict[str, dict[str, Counter[str]]] = {
            phase: {} for phase in VALID_PHASES
        }

        requested_repo = self._requested_repo(repo)
        agents = getattr(self.router, "agents", None)
        if not isinstance(agents, Mapping):
            agents = getattr(self.router, "_agents", None)
        if not isinstance(agents, Mapping):
            raise TypeError("Router identity requires a mapping of preloaded checkpoint agents")

        self._checkpoint_identities: dict[str, dict[str, Any]] = {}
        for checkpoint in self._preloaded:
            agent = agents.get(checkpoint)
            if agent is None:
                raise ValueError(
                    f"Router identity missing preloaded checkpoint agent: {checkpoint}"
                )
            self._checkpoint_identities[checkpoint] = _build_laya_metadata(
                requested_repo,
                agent,
                self.name,
                "local-router",
                package=package_obj,
                requested_checkpoint=checkpoint,
                checkpoint_subfolder=checkpoint,
            )
            if self._checkpoint_identities[checkpoint].get("resolved_repo") is None:
                raise ValueError(
                    f"Router identity missing resolved repository for checkpoint: {checkpoint}"
                )

        revisions = [
            f"{checkpoint}:{self._checkpoint_identities[checkpoint]['revision']}"
            for checkpoint in sorted(self._checkpoint_identities)
        ]
        devices = sorted(
            {
                str(self._checkpoint_identities[checkpoint]["device"])
                for checkpoint in self._checkpoint_identities
            }
        )
        package_versions = {
            str(self._checkpoint_identities[checkpoint]["package_version"])
            for checkpoint in self._checkpoint_identities
        }
        if len(package_versions) != 1:
            raise ValueError("Router identity has inconsistent package versions")
        repositories = {
            identity["resolved_repo"]
            for identity in self._checkpoint_identities.values()
            if identity.get("resolved_repo") is not None
        }
        if len(repositories) == 1:
            repository = next(iter(repositories))
        else:
            repository = None
        self._metadata = {
            "runner": self.name,
            "model": self.name,
            "revision": ",".join(revisions),
            "serving": "local-router",
            "device": ",".join(devices),
            "adapter_version": "1",
            "identity_scope": "per-checkpoint",
            "requested_repo": requested_repo,
            "repo": requested_repo if requested_repo is not None else repository,
            "resolved_repo": repository,
            "package_version": next(iter(package_versions)),
            "preloaded": list(self._preloaded),
            "max_loaded": self._max_loaded,
        }

    def _reported_preloaded(self) -> list[str]:
        reported = getattr(self.router, "preloaded", None)
        if reported is None:
            return list(PRELOADED_CHECKPOINTS)
        if not isinstance(reported, (list, tuple, set, frozenset)):
            raise TypeError("Router preloaded checkpoints must be a sequence of names")
        names = [_text(name) for name in reported]
        if not names or any(name is None for name in names):
            raise ValueError("Router preloaded checkpoints must contain non-empty names")
        if isinstance(reported, (set, frozenset)):
            return sorted(name for name in names if name is not None)
        return [name for name in names if name is not None]

    def _requested_repo(self, repo: str | None) -> str | None:
        if repo is not None:
            requested = _text(repo)
            if requested is None:
                raise ValueError("Router repository must be a non-empty string")
            return requested
        for name in ("requested_repo", "repo", "repository", "model_repo"):
            requested = _text(getattr(self.router, name, None))
            if requested is not None:
                return requested
        return None

    def _record_route(self, phase: str, model: str, reasons: tuple[str, ...]) -> None:
        phase_counts = self._route_counts_by_phase[phase]
        phase_counts[model] += 1
        self.route_counts[model] += 1

        phase_reasons = self._route_reasons_by_phase[phase].setdefault(model, set())
        phase_reasons.update(reasons)
        aggregate_reasons = self._route_reason_collections.setdefault(model, set())
        aggregate_reasons.update(reasons)
        self.route_reasons[model] = min(aggregate_reasons)

        phase_reason_counts = self._route_reason_counts_by_phase[phase].setdefault(model, Counter())
        aggregate_reason_counts = self._route_reason_counts.setdefault(model, Counter())
        for reason in reasons:
            phase_reason_counts[reason] += 1
            aggregate_reason_counts[reason] += 1

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        phase_key = self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        result = self.router.predict(copied_state, copied_questions)
        if not isinstance(result, Mapping):
            raise TypeError("Router adapter result must be a mapping")
        snapshot = deepcopy(dict(result))
        answers = self._validate_answer_ids(questions, snapshot.get("answers"))
        routing, model, reasons = self._routing_metadata(snapshot)

        output: dict[str, Any] = {"answers": normalize_answers(questions, answers)}
        if routing is not None and model is not None:
            self._record_route(phase_key, model, reasons)
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
        info = super().info()
        route_reasons_by_phase = {
            phase: {
                model: sorted(reasons)
                for model, reasons in sorted(self._route_reasons_by_phase[phase].items())
            }
            for phase in VALID_PHASES
        }
        route_reason_counts_by_phase = {
            phase: {
                model: dict(sorted(counts.items()))
                for model, counts in sorted(self._route_reason_counts_by_phase[phase].items())
            }
            for phase in VALID_PHASES
        }
        route_reason_counts = {
            model: dict(sorted(counts.items()))
            for model, counts in sorted(self._route_reason_counts.items())
        }
        route_reason_collections = {
            model: sorted(reasons)
            for model, reasons in sorted(self._route_reason_collections.items())
        }
        checkpoint_identities = {
            checkpoint: deepcopy(self._checkpoint_identities[checkpoint])
            for checkpoint in sorted(self._checkpoint_identities)
        }
        info.update(
            {
                "preloaded": list(self._preloaded),
                "max_loaded": self._max_loaded,
                "route_counts": {
                    model: self.route_counts[model] for model in sorted(self.route_counts)
                },
                "route_reasons": {
                    model: self.route_reasons[model] for model in sorted(self.route_reasons)
                },
                ROUTE_REASON_COUNTS_KEY: route_reason_counts,
                ROUTE_REASON_COLLECTIONS_KEY: route_reason_collections,
                ROUTE_COUNTS_BY_PHASE_KEY: {
                    phase: dict(sorted(self._route_counts_by_phase[phase].items()))
                    for phase in VALID_PHASES
                },
                ROUTE_REASONS_BY_PHASE_KEY: route_reasons_by_phase,
                ROUTE_REASON_COUNTS_BY_PHASE_KEY: route_reason_counts_by_phase,
                "checkpoint_identities": checkpoint_identities,
                "identities": deepcopy(checkpoint_identities),
            }
        )
        return info
