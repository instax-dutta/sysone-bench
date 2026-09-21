"""Laya adapter: local open-weights inference via the laya package."""
import copy
import laya
from .base import BaseRunner


class LayaRunner(BaseRunner):
    name = "laya"

    def __init__(self, repo="convaiinnovations/laya"):
        self.agent = laya.load(repo)

    def predict(self, state, questions):
        return self.agent.predict(copy.deepcopy(state), copy.deepcopy(questions))

    def info(self):
        return {"runner": self.name, "repo": "convaiinnovations/laya",
                "checkpoint": "english", "serving": "local"}
