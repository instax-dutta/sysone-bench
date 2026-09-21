"""Run suites against one or more models. Usage:
  USE_TF=0 ./.venv/bin/python run.py --models laya
  TYPESAFE_API_KEY=... ./.venv/bin/python run.py --models laya,jev
Writes results/run_<name>_<ts>.json (append-only).
"""
import argparse
import copy
import hashlib
import json
import statistics
import time
from datetime import datetime

import numpy as np

import laya
from datasets.cases import SUITES, SEED
from runners.laya_runner import LayaRunner


def noul_prob(a):
    return float(a.get("noul", a.get("probability", a.get("boolean", 0.0))))


def question_hash(questions):
    return hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:12]


def run_suite(runner, questions, cases):
    lat, correct, confs, y_true, y_prob, per_q, rows = [], [], [], [], [], {}, []
    for state, expected in cases:
        t0 = time.perf_counter()
        res = runner.predict(state, questions)
        dt = (time.perf_counter() - t0) * 1000
        lat.append(dt)
        for qid, exp in expected.items():
            a = res["answers"][qid]
            if a["type"] == "choice":
                ok = int(a["choice"] == exp)
                c = float(a["confidence"])
            else:
                p = noul_prob(a)
                ok = int(int(p >= 0.5) == exp)
                c = max(p, 1 - p)
                y_true.append(exp)
                y_prob.append(p)
            correct.append(ok)
            confs.append(c)
            per_q.setdefault(qid, []).append(ok)
            rows.append({"qid": qid, "ok": ok, "conf": round(c, 3)})
    acc = float(np.mean(correct))
    ece = float(laya.ece_score(np.array(confs), np.array(correct)))
    brier = float(np.mean([(p - t) ** 2 for p, t in zip(y_prob, y_true)])) if y_prob else None
    return {"states": len(cases), "decisions": len(correct), "accuracy": round(acc, 4),
            "ece": round(ece, 4), "brier_noul": round(brier, 4) if brier is not None else None,
            "ms_per_call_mean": round(statistics.mean(lat), 1),
            "ms_per_call_p50": round(statistics.median(lat), 1),
            "ms_per_call_p95": round(float(np.percentile(lat, 95)), 1),
            "per_question_acc": {k: round(float(np.mean(v)), 4) for k, v in per_q.items()},
            "rows": rows}


def speed_scaling(runner):
    base = {"message": "My payment failed twice, error 500, need help urgently."}
    tq = laya.triage_questions()
    keys = list(tq.keys())
    out = {}
    for nq in [1, 5, 10, 20]:
        qs = {f"q{i}_{keys[i % len(keys)]}": copy.deepcopy(tq[keys[i % len(keys)]])
              for i in range(nq)}
        for _ in range(2):
            runner.predict(base, qs)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            runner.predict(base, qs)
            ts.append((time.perf_counter() - t0) * 1000)
        out[str(nq)] = {"ms_per_call_p50": round(statistics.median(ts), 1),
                        "ms_per_q": round(statistics.median(ts) / nq, 1)}
    return out


def build_runner(name):
    if name == "laya":
        return LayaRunner()
    if name == "jev":
        from runners.jev_runner import JevRunner
        return JevRunner()
    raise ValueError(f"unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="laya")
    args = ap.parse_args()
    for model in [m.strip() for m in args.models.split(",")]:
        runner = build_runner(model)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = {"meta": {"runner": runner.name, "seed": SEED, "timestamp": ts,
                        **runner.info(), "question_hash": {}},
               "suites": {}}
        for suite, (qfn_name, cases) in SUITES.items():
            questions = getattr(laya, qfn_name)()
            out["meta"]["question_hash"][suite] = question_hash(questions)
            if suite == "triage":  # warmup
                runner.predict({"message": "hello"}, questions)
            out["suites"][suite] = run_suite(runner, questions, cases)
            s = out["suites"][suite]
            print(f"[{runner.name}] {suite}: acc={s['accuracy']} ece={s['ece']} "
                  f"p50={s['ms_per_call_p50']}ms", flush=True)
        out["speed_scaling"] = speed_scaling(runner)
        rows = [r for s in out["suites"].values() for r in s["rows"]]
        gating = {}
        for thr in [0.5, 0.7, 0.85, 0.95]:
            sel = [r for r in rows if r["conf"] >= thr]
            gating[str(thr)] = {"coverage": round(len(sel) / len(rows), 3),
                                "acc": round(float(np.mean([r["ok"] for r in sel])), 4) if sel else None,
                                "n": len(sel)}
        out["gating"] = gating
        path = f"results/run_{runner.name}_{ts}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{runner.name}] saved {path}")


if __name__ == "__main__":
    main()
