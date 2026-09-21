"""Router adapter: laya.Router dispatching per request (recommended production path).
Preloads english + multilingual only; typed-decisions stays lazy (never picked by
language routing in these suites). Records per-call routing decisions for the report.
"""
import copy
from collections import Counter
from .base import BaseRunner


class RouterRunner(BaseRunner):
    name = "laya-router"

    def __init__(self):
        from laya import Router
        self.router = Router(preload=False, max_loaded=2)
        self.router.preload(["english", "multilingual"])
        self.route_counts = Counter()
        self.route_reasons = {}

    def predict(self, state, questions):
        res = self.router.predict(copy.deepcopy(state), copy.deepcopy(questions))
        routing = res.get("routing", {})
        model = routing.get("model", "?")
        self.route_counts[model] += 1
        self.route_reasons.setdefault(model, routing.get("reason", ""))
        return {"answers": res["answers"], "_routing": routing}

    def info(self):
        return {"runner": self.name, "serving": "local-router",
                "preloaded": ["english", "multilingual"], "max_loaded": 2,
                "route_counts": dict(self.route_counts),
                "route_reasons": self.route_reasons}
