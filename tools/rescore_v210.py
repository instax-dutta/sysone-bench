#!/usr/bin/env python3
"""Re-score every stored run against the corrected manifest.

No inference, no GPU. Each prediction row already carries `answers`, and the
orchestrator's own rule (benchmark/orchestrator.py::_score_rows) is that a
decision is correct when the answer matches `expected`. So correcting the gold
is arithmetic over data we already have.

Writes before/after per suite for every run, so the change is auditable.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE = REPO / "datasets" / "v2"
MANIFEST = BASE / "manifest.v2.1.0.jsonl"
OLD_MANIFEST = BASE / "manifest.jsonl"
RAW = pathlib.Path(os.environ.get("SYSONE_RESULTS_ROOT", REPO / "results" / "raw"))
OUT = pathlib.Path(os.environ.get("SYSONE_RESCORE_OUT", REPO / "results" / "raw" / "_rescored"))


def load_gold(path: pathlib.Path) -> dict[str, dict]:
    return {c["case_id"]: c for c in map(json.loads, path.read_text().splitlines())}


# Reuse the orchestrator's own rule rather than reimplementing it. The first
# version of this script hand-rolled the comparison and reproduced 0 of 50
# stored summaries, because score questions use floor(score + 0.5) rounding.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from benchmark.orchestrator import _answer_is_correct  # noqa: E402


def _correct(question: dict, answer, expected) -> bool:
    return _answer_is_correct(question, answer, expected)


def score(pred_path: pathlib.Path, gold: dict[str, dict]) -> dict | None:
    per_suite: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    rows = 0
    for line in pred_path.read_text().splitlines():
        r = json.loads(line)
        cid = r["case_id"]
        g = gold.get(cid)
        if g is None or r.get("split") != "evaluation" or g["split"] != "evaluation":
            continue
        rows += 1
        qmap = {q["qid"]: q for q in g["questions"]}
        exp, ans = g["expected"], r.get("answers", {})
        for qid in r["question_ids"]:
            q = qmap.get(qid)
            if q is None:
                continue
            slot = per_suite[g["suite_id"]]
            slot[1] += 1
            a = ans.get(qid)
            if a is not None and _correct(q, a, exp.get(qid)):
                slot[0] += 1
    if rows == 0:
        return None
    suites = {s: {"accuracy": (c / t if t else None), "decisions": t}
              for s, (c, t) in per_suite.items()}
    total_correct = sum(c for c, _ in per_suite.values())
    total_decisions = sum(t for _, t in per_suite.values())
    return {
        "cases": rows,
        "suites": suites,
        "decisions": total_decisions,
        "correct": total_correct,
        # Pooled, exactly as _score_rows does. NOT the mean of suite rates.
        "accuracy": (total_correct / total_decisions if total_decisions else None),
    }


def main() -> int:
    gold_new, gold_old = load_gold(MANIFEST), load_gold(OLD_MANIFEST)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(RAW.rglob("predictions.jsonl"))
    print(f"  manifest rows : {len(gold_new)}")
    print(f"  runs found    : {len(runs)}\n")

    report, skipped = {}, []
    for p in runs:
        run = p.parent.name
        try:
            before, after = score(p, gold_old), score(p, gold_new)
        except Exception as e:  # noqa: BLE001
            print(f"  {run:34} FAILED {type(e).__name__}: {e}")
            continue
        if before is None or after is None:
            skipped.append(run)
            continue
        report[run] = {"before": before, "after": after}
        d = (after["accuracy"] or 0) - (before["accuracy"] or 0)
        print(f"  {run:34} {before['accuracy']:.4f} -> {after['accuracy']:.4f}  {d:+.4f}")

    (OUT / "rescore.json").write_text(json.dumps(report, indent=1))
    print(f"\n  scored {len(report)} runs, skipped {len(skipped)}")
    if skipped:
        print(f"  skipped: {skipped}")
    print(f"  wrote {OUT/'rescore.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())