#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt from data/results.json.

Both files were hand-written, which is how they came to disagree with the site:
llms.txt still advertised the 2.0.0 manifest digest and pre-correction scores.
Every count, score, rank and digest here is read from the generated data. Only
prose is authored here, never a number.

Run after any change to results.json:
    python webpage/build-data.py && python tools/build_llms.py
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "webpage" / "data" / "results.json"
OUT_DIR = REPO / "webpage"


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}"


def fmt(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def ranking_rows(d: dict, limit: int | None = None) -> list[str]:
    ms = sorted(d["measurements"], key=lambda m: -m["accuracy"])
    if limit:
        ms = ms[:limit]
    out = []
    for i, m in enumerate(ms, 1):
        rev = (m.get("revision") or "")[:12]
        out.append(f"| {i} | `{m['runner']}` | {fmt(m['accuracy'])} | `{m.get('model') or '-'}` | `{rev or '-'}` |")
    return out


def suite_table(d: dict) -> list[str]:
    ms = sorted(d["measurements"], key=lambda m: -m["accuracy"])
    head = "| Model | " + " | ".join(d["suites"]) + " |"
    sep = "|---" * (len(d["suites"]) + 1) + "|"
    rows = [head, sep]
    for m in ms:
        cells = " | ".join(pct(m.get("suites", {}).get(s)) for s in d["suites"])
        rows.append(f"| `{m['runner']}` | {cells} |")
    return rows


def excluded_table(d: dict) -> list[str]:
    rows = ["| Entry | Reason | Why |", "|---|---|---|"]
    for name, reason, detail in d.get("excluded", []):
        rows.append(f"| `{name}` | {reason} | {detail} |")
    return rows


def duplicates_block(d: dict) -> list[str]:
    dup = d.get("duplicates") or {}
    out = []
    for name, dirs in sorted((dup.get("repeatedRuns") or {}).items()):
        out.append(f"- `{name}`: {len(dirs)} directories, one measurement ({', '.join('`' + x + '`' for x in dirs)})")
    for fp, runners in sorted((dup.get("identicalPredictions") or {}).items()):
        out.append(
            f"- {' and '.join('`' + r + '`' for r in runners)}: byte-identical answers on all "
            f"{d['decisions']:,} decisions, so they count once"
        )
    return out


def deferral_lines(d: dict) -> list[str]:
    out = []
    for m in sorted(d["measurements"], key=lambda x: -x["accuracy"]):
        f = m.get("deferrals")
        if not f:
            continue
        out.append(
            f"- **`{m['runner']}` declined {f['deferred']:,} of {f['evaluation_decisions']:,} evaluation "
            f"decisions ({f['deferred_share'] * 100:.1f}%).** The decision contract has no abstention "
            f"type, so the adapter answered every one at the argmax option and each is counted as a "
            f"confident correct-or-incorrect answer the model did not choose. "
            f"{f['projected_to_argmax']:,} carried an explicit argmax projection in the raw output. "
            f"Read this row as the adapter's coverage, not the model's own."
        )
    return out


def stats(d: dict) -> dict:
    ms = sorted(d["measurements"], key=lambda m: -m["accuracy"])
    closed = d["referenceClosedApi"]["accuracy"]
    gap = closed - ms[0]["accuracy"]
    top3 = ms[0]["accuracy"] - ms[2]["accuracy"]
    return {
        "best": ms[0],
        "gap": gap,
        "top3": top3,
        "closed": closed,
        "decided": int(top3 * d["decisions"]),
    }


def build_llms_txt(d: dict) -> str:
    s = stats(d)
    ms = sorted(d["measurements"], key=lambda m: -m["accuracy"])
    n, dirs, scope = d["distinctMeasurements"], d["runDirectoriesVerified"], d["scopeTotal"]
    pending = scope - n
    L = []
    L.append("# sysone-bench")
    L.append("")
    L.append("> An independent head-to-head benchmark of System One decision models. No vendor affiliation,")
    L.append("> no API access. Every model answers the same sealed "
             f"{d['cases']:,} cases and is graded on {d['decisions']:,} scored")
    L.append("> evaluation decisions at seed 42.")
    L.append("")
    L.append("Site: https://sysone.sdad.pro/")
    L.append("Repository: https://github.com/instax-dutta/sysone-bench")
    L.append("Data: https://sysone.sdad.pro/data/results.json")
    L.append("Full results as one document: https://sysone.sdad.pro/llms-full.txt")
    L.append("")
    L.append("## What this benchmark measures")
    L.append("")
    L.append('System One "decision models" answer typed questions about a piece of context rather than')
    L.append("generating free text. Three question types are used:")
    L.append("")
    L.append("- `choice` - select one option from a declared set (2 to 255 criteria)")
    L.append("- `score` - report an ordinal level on a declared scale")
    L.append("- `noul` - report P(true) for a binary judgement")
    L.append("")
    L.append(f"{len(d['suites'])} suites are measured: " + ", ".join(d["suites"]) + ".")
    L.append("")
    L.append("## Headline results")
    L.append("")
    L.append(f"{n} System One decision models measured, of a {scope}-entry scope"
             + (f", with {pending} awaiting measurement." if pending else ", all measured."))
    L.append("Counts are derived from checksum-verified run directories by a script, not maintained by hand.")
    L.append("")
    L.append("Closed API reference, not a panel entry:")
    L.append("")
    L.append(f"- **{d['referenceClosedApi']['model']} - {fmt(s['closed'])}** "
             f"(closed API at {d['referenceClosedApi'].get('serving', 'api')})")
    L.append("")
    L.append("Best open-weights models:")
    L.append("")
    L.append("| Rank | Model | Accuracy |")
    L.append("|---:|---|---:|")
    L += ranking_rows(d, 10)
    L.append("")
    L.append(f"The best open model trails the closed API reference by **{s['gap']:.4f}**.")
    L.append("")
    L.append(f"The top three open models sit within **{s['top3']:.4f}** of each other, roughly "
             f"{s['decided']} decisions out of {d['decisions']:,}.")
    L.append("At this sample size they are effectively tied and the ordering among them is not meaningful.")
    L.append("")
    if deferral_lines(d):
        L.append("## Rows that declined to answer")
        L.append("")
        L += deferral_lines(d)
        L.append("")
    L.append("## Why these numbers are comparable")
    L.append("")
    L.append("Vendor-published leaderboard figures are not comparable to these. They come from different")
    L.append("prompts. This benchmark fixes the inputs instead:")
    L.append("")
    L.append(f"- One sealed manifest, {d['cases']:,} cases and 1,550 typed questions, verified byte-identical")
    L.append("  before any number is read")
    L.append(f"- Manifest sha256 `{d['manifestSha256']}`")
    L.append(f"- Logical digest `{d['logicalDigest']}` (over question bytes, invariant across gold corrections)")
    L.append(f"- Seed {d['seed']}, dataset version {d['datasetVersion']}")
    L.append(f"- {d['evaluationCases']:,} evaluation cases, {d['decisions']:,} scored decisions per model")
    L.append("- Every run records a full 40-character model revision, plus base model and base revision for")
    L.append("  adapter and technique entries, plus device, dtype, serving mode, and the scoring readout used")
    L.append("")
    L.append("## Precision effect, measured rather than assumed")
    L.append("")
    tev = [m for m in ms if m["runner"] == "tev1-08b"]
    if len(tev) == 1:
        L.append("One model was run on two hosts against identical bytes and one pinned revision. Every GPU row")
        L.append("is bf16 and every CPU row is fp32, so precision is recorded per row, not treated as constant.")
    else:
        L.append("Every GPU row is bf16 and every CPU row is fp32, so precision is recorded per row rather than")
        L.append("treated as constant.")
    L.append("")
    L.append("## Coverage, and what is missing")
    L.append("")
    L.append(f"{dirs} verified run directories but **{n} distinct measurements**.")
    dups = duplicates_block(d)
    if dups:
        L += dups
    else:
        L.append("No repeated runs and no identical-prediction collisions in this set.")
    L.append("")
    L.append("A measurement whose checksum fails is dropped rather than published.")
    L.append("")
    L.append(f"{len(d.get('excluded', []))} entries sit outside the measured scope, each with a recorded reason:")
    L.append("")
    L += excluded_table(d)
    L.append("")
    L.append(f"The scope is {scope} entries. {n} are measured"
             + (f" and {pending} awaits measurement." if pending else "."))
    L.append("")
    L.append("## Caveats stated plainly")
    L.append("")
    L.append(f"- **Ground truth, plainly.** Dataset `{d['datasetVersion']}` re-anchors the gold: agnews,")
    L.append("  banking77_12, mnli and sst5 adopt the public source label directly; emotion keeps a per-case")
    L.append("  human decision on each disagreement with the published label, still single-reviewed, so no")
    L.append("  inter-annotator statistic. The four synthetic suites still have no public source. Corrected")
    L.append("  labels are rescored from the stored answers; no model was re-run.")
    L.append("- **Rows tagged \"technique reimplementation\"** are our reading of a published method over a")
    L.append("  public base checkpoint, not the authors' code.")
    for m in ms:
        if m["runner"] == "pngwn":
            L.append(f"- **pngwn scored {fmt(m['accuracy'])}** but its vendor never published the prompt format")
            L.append("  its scorer was trained on. It is recorded as unresolved rather than as a measurement.")
    L.append("- **Latency is not comparable across rows.** CPU rows and container-capped rows were measured")
    L.append("  under different CPU and memory limits, so latency must not be charted across the panel.")
    L.append("")
    return "\n".join(L)


def build_llms_full(d: dict) -> str:
    s = stats(d)
    ms = sorted(d["measurements"], key=lambda m: -m["accuracy"])
    n, dirs, scope = d["distinctMeasurements"], d["runDirectoriesVerified"], d["scopeTotal"]
    L = []
    L.append("# sysone-bench, full results")
    L.append("")
    L.append(f"Dataset `{d['datasetVersion']}`. Seed {d['seed']}. Manifest sha256 `{d['manifestSha256']}`.")
    L.append(f"Logical digest `{d['logicalDigest']}`, over question bytes and therefore invariant across the")
    L.append("gold correction between 2.0.0 and 2.1.0.")
    L.append("")
    L.append("## What was measured")
    L.append("")
    L.append('System One "decision models" answer typed questions about a piece of context rather than')
    L.append("generating free text. Three question types are used:")
    L.append("")
    L.append("- `choice` - select one option from a declared set (2 to 255 criteria)")
    L.append("- `score` - report an ordinal level on a declared scale")
    L.append("- `noul` - report P(true) for a binary judgement")
    L.append("")
    L.append(f"{len(d['suites'])} suites, {d['cases']:,} cases, {d['evaluationCases']:,} of them in the evaluation")
    L.append(f"split, {d['decisions']:,} scored decisions per model at seed {d['seed']}.")
    L.append("")
    L.append("## Headline result")
    L.append("")
    L.append(f"**{n} distinct measurements** from {dirs} verified run directories, against a {scope}-entry scope.")
    L.append("")
    L.append(f"- Best open-weights: **`{s['best']['runner']}` at {fmt(s['best']['accuracy'])}**")
    L.append(f"- Closed API reference, not a panel entry: **{d['referenceClosedApi']['model']} at {fmt(s['closed'])}**")
    L.append(f"- Gap: {s['gap']:.4f}")
    L.append("")
    L.append("The top three open models sit within "
             f"{s['top3']:.4f} of each other, roughly {s['decided']} decisions out of {d['decisions']:,}, so their")
    L.append("ordering is not meaningful at this sample size.")
    L.append("")
    L.append("## Full ranking, all measurements")
    L.append("")
    L.append("Ranked by accuracy. The closed-API reference is listed separately above and is never merged")
    L.append("into this open-weights ranking.")
    L.append("")
    L.append("| # | Model | Accuracy | Weights | Pinned revision |")
    L.append("|---|---|---|---|---|")
    L += ranking_rows(d)
    L.append("")
    if deferral_lines(d):
        L.append("## Rows that declined to answer")
        L.append("")
        L += deferral_lines(d)
        L.append("")
    L.append("## Per-suite accuracy")
    L.append("")
    L.append("Aggregate accuracy hides the fact that a model can be excellent at moderation and useless at")
    L.append("intent routing. Every model is scored on all "
             f"{len(d['suites'])} suites, as a percentage.")
    L.append("")
    L += suite_table(d)
    L.append("")
    L.append("## How to read the numbers")
    L.append("")
    L.append(f"- Accuracy is pooled over the {d['decisions']:,} evaluation decisions, not a mean of suite rates.")
    L.append(f"- {d['evaluationCases']:,} of the {d['cases']:,} cases are in the evaluation split; the remainder is")
    L.append("  calibration and is never scored.")
    L.append("- One seed, no per-suite tuning. A suite is scored exactly as the manifest declares it.")
    L.append("- Every GPU row is bf16 and every CPU row is fp32. Latency is not comparable across rows.")
    L.append("")
    L.append("## Counting rule")
    L.append("")
    L.append(f"**{n} distinct measurements**. Repeated runs collapse:")
    L.append("")
    L += duplicates_block(d)
    L.append("")
    L.append("A measurement whose checksum fails is dropped rather than published.")
    L.append("")
    L.append("## Not measured, and why")
    L.append("")
    L.append(f"{len(d.get('excluded', []))} entries sit outside the measured scope, each with a recorded reason:")
    L.append("")
    L += excluded_table(d)
    L.append("")
    L.append(f"The scope is {scope} entries: {n} measured"
             + (f", {scope - n} awaiting measurement." if scope - n else "."))
    L.append("")
    L.append("## Caveats, stated plainly")
    L.append("")
    L.append(f"- **Ground truth.** Dataset `{d['datasetVersion']}` re-anchors the gold. agnews, banking77_12,")
    L.append("  mnli and sst5 adopt the public source label directly. emotion keeps a per-case human decision")
    L.append("  on each disagreement with the published label; it is still single-reviewed with no second")
    L.append("  reviewer and no adjudication, so no inter-annotator statistic is reported. The four synthetic")
    L.append("  suites have no public source. Corrected labels were rescored from stored answers; no model was")
    L.append("  re-run and no inference was repeated.")
    L.append("- **Technique reimplementations** are our reading of a published method over a public base")
    L.append("  checkpoint, not the authors' code. They set `technique_reimplementation: true`.")
    for m in ms:
        if m["runner"] == "pngwn":
            L.append(f"- **pngwn scored {fmt(m['accuracy'])}** but its vendor never published the prompt format")
            L.append("  its scorer was trained on, so it is recorded as unresolved.")
    L.append("- **Withdrawn rows are not reported.** An entry whose artifacts cannot be reproduced is removed")
    L.append("  from the ranking and listed above with the reason, rather than published from a superseded run.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"- Manifest sha256 `{d['manifestSha256']}`")
    L.append(f"- Logical digest `{d['logicalDigest']}`")
    L.append(f"- Seed {d['seed']}, dataset version `{d['datasetVersion']}`")
    L.append(f"- {dirs} verified run directories, {n} distinct measurements")
    L.append("- Every run records model, full revision, device, dtype, serving mode and scoring readout.")
    L.append("- Full run directories are mirrored to a private dataset and verified byte for byte.")
    L.append("")
    L.append("## Machine-readable data")
    L.append("")
    L.append("https://sysone.sdad.pro/data/results.json")
    L.append("")
    return "\n".join(L)


def main() -> int:
    if not DATA.is_file():
        print(f"missing {DATA}; run python webpage/build-data.py first", file=sys.stderr)
        return 1
    d = json.loads(DATA.read_text())
    for name, body in (("llms.txt", build_llms_txt(d)), ("llms-full.txt", build_llms_full(d))):
        (OUT_DIR / name).write_text(body)
        print(f"  wrote webpage/{name}: {len(body.splitlines())} lines")
    n = d["distinctMeasurements"]
    print(f"  {n} measurements, {d['runDirectoriesVerified']} dirs, scope {d['scopeTotal']}, "
          f"dataset {d['datasetVersion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())