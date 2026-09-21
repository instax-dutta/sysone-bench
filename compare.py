"""Head-to-head compare of two run files. Usage:
  ./.venv/bin/python compare.py results/run_laya_<ts>.json results/run_jev-<v>_<ts>.json
Checks question hashes match, then prints accuracy / ECE / latency deltas.
"""
import json
import math
import sys


def ci(acc, n, z=1.96):
    se = math.sqrt(acc * (1 - acc) / n)
    return round(max(0, acc - z * se), 3), round(min(1, acc + z * se), 3)


def main():
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    qa, qb = a["meta"]["question_hash"], b["meta"]["question_hash"]
    print(f"A={a['meta']['runner']}  B={b['meta']['runner']}")
    print(f"questions identical: {qa == qb} ({qa})")
    print(f"A model: {a['meta']}  B model: {b['meta']}")
    for suite in a["suites"]:
        sa, sb = a["suites"][suite], b["suites"][suite]
        loa, hia = ci(sa["accuracy"], sa["decisions"])
        lob, hib = ci(sb["accuracy"], sb["decisions"])
        d = round(sb["accuracy"] - sa["accuracy"], 3)
        print(f"\n{suite}: A acc={sa['accuracy']} (95CI {loa}-{hia}, n={sa['decisions']}) "
              f"ECE={sa['ece']} p50={sa['ms_per_call_p50']}ms")
        print(f"{' ' * len(suite)}  B acc={sb['accuracy']} (95CI {lob}-{hib}, n={sb['decisions']}) "
              f"ECE={sb['ece']} p50={sb['ms_per_call_p50']}ms  delta={d:+}")
        for q in sa["per_question_acc"]:
            print(f"  {q}: A={sa['per_question_acc'][q]} B={sb['per_question_acc'].get(q, '?')}")
    print(f"\ngating A={a['gating']}")
    print(f"gating B={b['gating']}")
    out = f"results/compare_{a['meta']['runner']}_vs_{b['meta']['runner']}.json"
    json.dump({"a": sys.argv[1], "b": sys.argv[2], "question_hash_match": qa == qb,
               "suites": {s: {"a": a["suites"][s], "b": b["suites"][s]} for s in a["suites"]}},
              open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
