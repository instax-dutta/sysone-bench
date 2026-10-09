#!/usr/bin/env python3
"""Write v2.1.0 run records beside the v2.0.0 ones.

The v2.0.0 run directories are append-only records and are never touched. For
each run this writes a NEW directory suffixed -v210 containing:

  metadata.json   copied, with dataset_version and manifest digests repointed
  summary.json    recomputed against the corrected gold
  RESCORE.md      what changed for this run, and why

It deliberately does NOT copy predictions.jsonl. The answers are unchanged and
already stored under the v2.0.0 run; duplicating ~90MB of identical predictions
would make the record heavier without adding evidence. The v2.1.0 record names
the run it was derived from.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
RESCORE = REPO / "rescore.json"
MANIFEST = REPO / "datasets" / "v2" / "manifest.v2.1.0.jsonl"
MANIFEST_SHA = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
OLD_SHA = "a938cc2483a592dc84e0d5baac12594491bcaa5b4ceb6b7c3b0def71b36297bd"


def write_json_exclusive(path: pathlib.Path, payload: object) -> None:
    """Append-only: never overwrite an existing record."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def count_deferrals(src: pathlib.Path) -> dict | None:
    """Count projected deferrals from the stored predictions, or None if absent.

    A run whose predictions carry no per-decision status is simply not a
    deferring runner, so we record nothing rather than recording a zero that
    would read as a measured fact about every adapter.
    """
    pred = src / "predictions.jsonl"
    if not pred.exists():
        return None
    deferred = projected = total = 0
    for line in pred.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "evaluation":
            continue
        raw = row.get("_raw_model")
        if not isinstance(raw, dict):
            return None
        for dec in raw.get("decisions", []):
            if not isinstance(dec, dict):
                continue
            if "status" not in dec and "projected_to_argmax" not in dec:
                return None
            total += 1
            if dec.get("status") == "deferred":
                deferred += 1
            if dec.get("projected_to_argmax"):
                projected += 1
    if not total:
        return None
    return {
        "evaluation_decisions": total,
        "deferred": deferred,
        "deferred_share": deferred / total,
        "projected_to_argmax": projected,
    }


def main() -> int:
    rescore = json.loads(RESCORE.read_text())
    made = skipped = 0

    for run, v in sorted(rescore.items()):
        src = None
        for cand in RAW.rglob(run):
            if (cand / "metadata.json").exists() and not cand.name.endswith("-v210"):
                src = cand
                break
        if src is None:
            print(f"  {run:32} no source run dir")
            skipped += 1
            continue

        dest = src.with_name(src.name + "-v210")
        if dest.exists():
            print(f"  {run:32} already exists")
            skipped += 1
            continue

        dest.mkdir(parents=True)
        meta = json.loads((src / "metadata.json").read_text())

        meta["dataset_version"] = "2.1.0"
        man = dict(meta.get("manifest") or {})
        man["previous_checksum_digest"] = OLD_SHA
        man["checksum_digest"] = MANIFEST_SHA
        meta["manifest"] = man
        meta["derived_from_run_id"] = meta.get("run_id")
        meta["run_id"] = dest.name
        meta["rescored"] = True
        deferrals = count_deferrals(src)
        if deferrals:
            meta["deferrals"] = deferrals
            meta["deferrals_note"] = (
                "The runner declined these decisions. This adapter has no abstention type in the "
                "decision contract, so each declined decision was answered at the argmax option and "
                "is counted above as a confident correct-or-incorrect answer the runner did not "
                "choose. The overall accuracy therefore reflects the adapter's projection, not the "
                "runner's own coverage."
            )
        meta["rescored_reason"] = (
            "Gold labels corrected for agnews, banking77_12, mnli and sst5 by adopting the "
            "public source label, and for emotion by per-case review. Answers, prompts, "
            "criteria and state are unchanged, so no inference was re-run."
        )
        write_json_exclusive(dest / "metadata.json", meta)

        after = v["after"]
        write_json_exclusive(dest / "summary.json", {
            "schema_version": 2,
            "metric_schema_version": 2,
            "dataset_version": "2.1.0",
            "seed": 42,
            "split": "evaluation",
            "cases": after["cases"],
            "decisions": after["decisions"],
            "accuracy": after["accuracy"],
            "suites": {s: {"accuracy": sv["accuracy"], "decisions": sv["decisions"]}
                       for s, sv in sorted(after["suites"].items())},
        })

        after_suites = v["after"]["suites"]
        # The auditor only accepts a run directory that carries a checksums file,
        # so seal the v2.1.0 record the same way the original was sealed.
        lines_for_hash = []
        for name in ("metadata.json", "summary.json"):
            lines_for_hash.append(
                f"{hashlib.sha256((dest / name).read_bytes()).hexdigest()}  {name}")
        (dest / "checksums.sha256").write_text("\n".join(lines_for_hash) + "\n")

        moved = []
        for suite, b in sorted(v["before"]["suites"].items()):
            a_val = after_suites.get(suite, {}).get("accuracy")
            if a_val != b["accuracy"]:
                moved.append((suite, b["accuracy"], a_val))
        d = v["after"]["accuracy"] - v["before"]["accuracy"]
        lines = [
            f"# {meta['run_id']}",
            "",
            f"Re-scored from `{meta['derived_from_run_id']}` against manifest "
            f"`{MANIFEST_SHA[:16]}`, dataset version 2.1.0.",
            "",
            f"Overall accuracy **{v['before']['accuracy']:.4f}** -> **{v['after']['accuracy']:.4f}** ({d:+.4f})",
            "",
            "No model was re-run. The stored answers are unchanged; only the gold moved.",
            "",
            "## Per-suite change",
            "",
            "| suite | before | after | change |",
            "|---|---|---|---|",
        ]
        lines += [f"| {s} | {b:.4f} | {a:.4f} | {a - b:+.4f} |" for s, b, a in moved]
        if not moved:
            lines.append("| (none) | | | |")
        (dest / "RESCORE.md").write_text("\n".join(lines) + "\n")
        made += 1

    print(f"\n  wrote {made} v2.1.0 records, skipped {skipped}")
    print(f"  manifest sha256: {MANIFEST_SHA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())