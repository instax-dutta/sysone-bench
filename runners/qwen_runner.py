"""Qwen-PCD adapter: stock mlx-community/Qwen2.5-1.5B-Instruct-4bit evaluated with
the vendored parallel constrained decoding engine (runners/pcd, Apache 2.0, from
drinkmoonshine/parallel-constrained-decoding, same code shipped as
harshatheg/Qwen-2.5-1B-RLCD, which carries no fine-tuned weights).

Mapping (documented, not identical): laya choice -> enum, noul -> boolean,
score -> enum over the rubric levels (argmax level reported as score).
State dicts render as 'key: value' lines. Same states, same order, seed 42.
"""
import copy

from .base import BaseRunner
from runners.pcd import engine_mlx
from runners.pcd.schema import StructuredSchema
from runners.pcd.engine import run_parallel_generation

MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def state_to_context(state):
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    return "\n".join(str(x) for x in state)


def questions_to_schema(questions):
    schema = {}
    for qid, q in questions.items():
        t = q["type"]
        if t == "choice":
            crit = q.get("criteria", {})
            desc = q.get("instructions", "") + "".join(
                f"; {k}: {v}" for k, v in crit.items() if v)
            schema[qid] = {"type": "enum", "description": desc,
                           "choices": list(crit.keys())}
        elif t == "noul":
            schema[qid] = {"type": "boolean",
                           "description": q.get("instructions", "")}
        elif t == "score":
            levels = list(q.get("criteria", []))
            schema[qid] = {"type": "enum",
                           "description": q.get("instructions", "") +
                           " Answer with exactly one level.",
                           "choices": levels, "_score_levels": True}
    return schema


class QwenRunner(BaseRunner):
    name = "qwen-pcd-15b"

    def __init__(self):
        engine_mlx.MODEL_ID = MODEL_ID
        engine_mlx.get_engine()  # load once, fail fast

    def predict(self, state, questions):
        questions = copy.deepcopy(questions)
        schema = questions_to_schema(questions)
        clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in schema.items()}
        res = run_parallel_generation(state_to_context(state),
                                      StructuredSchema(clean))
        tele = res.get("field_telemetry", {})
        answers = {}
        for qid, q in questions.items():
            f = tele[qid]
            t = q["type"]
            if t == "choice":
                probs = {c["choice"]: round(float(c["probability"]), 4)
                         for c in f["top_choices"]}
                answers[qid] = {"type": "choice", "choice": f["value"],
                                "probabilities": probs,
                                "confidence": round(float(f["confidence"]), 4)}
            elif t == "noul":
                p_true = float(f["confidence"]) if str(f["value"]).lower() == "true" \
                    else 1.0 - float(f["confidence"])
                answers[qid] = {"type": "noul", "noul": round(p_true, 4)}
            else:  # score asked as enum over levels; argmax level becomes score
                levels = list(q.get("criteria", []))
                idx = levels.index(f["value"]) if f["value"] in levels else 0
                answers[qid] = {"type": "score", "score": float(idx),
                                "confidence": round(float(f["confidence"]), 4)}
        return {"answers": answers}

    def info(self):
        return {"runner": self.name, "model": MODEL_ID, "serving": "local-mlx",
                "engine": "vendored parallel-constrained-decoding",
                "score_mapping": "enum-over-levels"}
