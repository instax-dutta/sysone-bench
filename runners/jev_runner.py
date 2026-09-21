"""Jev adapter: TypeSafe Decisions API. Needs TYPESAFE_API_KEY in env.

Endpoint shape per TypeSafe docs + OpenRouter decisions API:
  POST {base}/v1/systemone  (direct)  or  POST .../api/alpha/decisions (OpenRouter)
  body: {"model": <pinned id>, "state": {...}, "questions": {...}}
Jev answers already match the laya shape (choice/noul/score); this adapter only
normalizes field names. Pin a versioned id, never jev-latest, and record it.
"""
import copy
import os
import requests
from .base import BaseRunner

DIRECT_URL = "https://api.typesafe.ai/v1/systemone"


class JevRunner(BaseRunner):
    name = "jev"

    def __init__(self, model=None, base_url=None):
        self.api_key = os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Set TYPESAFE_API_KEY in env (see .env.example).")
        self.model = model or os.environ.get("JEV_MODEL", "jev-1.13.0")
        self.base_url = base_url or os.environ.get("JEV_BASE_URL", DIRECT_URL)
        mid = self.model.split(":")[-1]
        self.name = mid if mid.startswith("jev") else f"jev-{mid}"

    def predict(self, state, questions):
        resp = requests.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "state": copy.deepcopy(state),
                  "questions": copy.deepcopy(questions)},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answers = data.get("answers", data)
        norm = {}
        for qid, a in answers.items():
            a = dict(a)
            # normalize vendor variants to laya answer shape
            if a.get("type") == "boolean" and "boolean" in a:
                a = {"type": "noul", "noul": float(a["boolean"])}
            elif a.get("type") == "noul" and "probability" in a and "noul" not in a:
                a["noul"] = float(a.pop("probability"))
            norm[qid] = a
        return {"answers": norm, "_raw_model": data.get("model", self.model),
                "_usage": data.get("usage", {})}

    def info(self):
        return {"runner": self.name, "model": self.model,
                "base_url": self.base_url, "serving": "api"}
