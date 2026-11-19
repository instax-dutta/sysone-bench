#!/usr/bin/env python3
"""Render webpage/index.html from template.html and generated results data.

The hosted site has no build step, so the citation-worthy headline facts are baked into the served
HTML at publish time. app.js still hydrates the full interactive report from the same JSON file.
Run this after webpage/build-data.py, and commit both the template and the rendered page.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data" / "results.json").read_text(encoding="utf-8"))
TEMPLATE = (HERE / "template.html").read_text(encoding="utf-8")

best = DATA["measurements"][0]
ref = DATA["referenceClosedApi"]
counts = {
    "measurements": DATA["distinctMeasurements"],
    "dirs": DATA["runDirectoriesVerified"],
    "decisions": DATA["decisions"],
    "suites": len(DATA["suites"]),
    "scope": DATA["scopeTotal"],
    "unmeasured": DATA["scopeTotal"] - DATA["distinctMeasurements"],
    "cases": DATA["cases"],
}
values = {
    "best_score": f"{best['accuracy']:.4f}",
    "best_name": best["runner"],
    "gap": f"{ref['accuracy'] - best['accuracy']:.4f}",
    "closed_name": f"{ref['model']} ({ref['accuracy']:.4f})",
    "version": DATA["datasetVersion"],
    "generated_at": DATA.get("generatedAt", ""),
    "manifest": DATA["manifestSha256"],
    "logical": DATA["logicalDigest"],
    "eval_cases": f"{DATA['evaluationCases']:,}",
    "seed": DATA["seed"],
}
values.update({f"counts.{key}": f"{value:,}" for key, value in counts.items()})


def f4(value: object) -> str:
    return f"{float(value):.4f}" if value is not None else "—"


def best_by_task() -> str:
    items = []
    for suite in DATA["suites"]:
        scored = [
            (m["runner"], m["suites"][suite])
            for m in DATA["measurements"]
            if m["suites"].get(suite) is not None
        ]
        top = max(score for _, score in scored)
        tied = [runner for runner, score in scored if score >= top - 0.005]
        items.append(
            '<div class="task"><p class="task-name">'
            + escape(suite.replace("_", " "))
            + "</p><p class=\"task-lead\">"
            + ", ".join(escape(runner) for runner in tied)
            + "</p><p class=\"task-score\">"
            + f4(top)
            + "</p></div>"
        )
    return '<div class="task-grid">' + "".join(items) + "</div>"


def faq() -> str:
    rows = [
        (
            "What does sysone-bench measure?",
            "Typed decisions: multiple choice, ordinal scores and binary judgements. "
            "Every model answers the same sealed cases from one manifest at seed "
            f"{DATA['seed']}.",
        ),
        (
            "Which open model scores highest?",
            f"{best['runner']} at {f4(best['accuracy'])}. It trails the closed-API "
            f"reference {ref['model']} at {f4(ref['accuracy'])} by "
            f"{f4(ref['accuracy'] - best['accuracy'])}.",
        ),
        (
            "How many models and decisions?",
            f"{counts['measurements']} measurements, each on {counts['decisions']} evaluation "
            f"decisions across {counts['suites']} suites, from a {counts['scope']}-entry scope.",
        ),
        (
            "Which manifest were they scored against?",
            f"Dataset {DATA['datasetVersion']}, raw SHA-256 {DATA['manifestSha256']}.",
        ),
        (
            "Why not use vendor-published scores?",
            "They come from different prompts and different harnesses. Two numbers from "
            "different prompts are not a comparison, so no vendor-published figure is reused here.",
        ),
        (
            "Can I inspect a complete run?",
            "Yes. One full run directory with its checksum manifest is published at "
            '<a href="public-artifacts/seb-9b-t4/">public-artifacts/seb-9b-t4/</a>.',
        ),
    ]
    return "".join(
        '<div class="answer"><h3>' + escape(q) + "</h3><p>" + a + "</p></div>"
        for q, a in rows
    )


def suite_table() -> str:
    head = "".join(f"<th scope=\"col\">{escape(s.replace('_', ' '))}</th>" for s in DATA["suites"])
    rows = []
    for m in DATA["measurements"]:
        cells = "".join(f"<td>{f4(m['suites'].get(s))}</td>" for s in DATA["suites"])
        rows.append(f"<tr><th scope=\"row\">{escape(m['runner'])}</th>{cells}</tr>")
    return (
        '<table class="sr-only" id="suite-data-table"><caption>Accuracy for every measured model in every suite.</caption>'
        f"<thead><tr><th scope=\"col\">Model</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def precision_table() -> str:
    rows = "".join(
        "<tr><th scope=\"row\">"
        + escape(str(r.get("runId", "")))
        + "</th><td>"
        + escape(str(r.get("device", "")))
        + "</td><td>"
        + escape(str(r.get("dtype") or "default"))
        + "</td><td>"
        + f4(r.get("accuracy"))
        + "</td></tr>"
        for r in DATA.get("precision", {}).get("tev1-08b", [])
    )
    return (
        '<table class="sr-only" id="precision-data-table"><caption>tev1-08b measured on two hosts.</caption>'
        '<thead><tr><th scope="col">Run</th><th scope="col">Device</th><th scope="col">Dtype</th>'
        f'<th scope="col">Accuracy</th></tr></thead><tbody>{rows}</tbody></table>'
    )


def coverage_table() -> str:
    rows = "".join(
        f"<tr><th scope=\"row\">{label}</th><td>{value}</td></tr>"
        for label, value in (
            ("Verified run directories", DATA["runDirectoriesVerified"]),
            ("Distinct measurements", DATA["distinctMeasurements"]),
            ("Scope entries", DATA["scopeTotal"]),
        )
    )
    return (
        '<table class="sr-only" id="coverage-data-table"><caption>Coverage counts.</caption>'
        '<thead><tr><th scope="col">Measure</th><th scope="col">Count</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )


values["best_by_task"] = best_by_task()
values["faq"] = faq()
values["suite_table"] = suite_table()
values["precision_table"] = precision_table()
values["coverage_table"] = coverage_table()


def replace(match: re.Match[str]) -> str:
    key = match.group(1)
    if key not in values:
        raise KeyError(f"unknown template token: {key}")
    return str(values[key])


rendered = re.sub(r"\{\{([a-z_.]+)\}\}", replace, TEMPLATE)
if "{{" in rendered:
    raise ValueError("unreplaced template token")
(HERE / "index.html").write_text(rendered, encoding="utf-8")
print(f"wrote {HERE / 'index.html'} : {len(values)} tokens filled")
