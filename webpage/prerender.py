#!/usr/bin/env python3
"""Render webpage/index.html from template.html and generated results data.

The hosted site has no build step, so the citation-worthy headline facts are baked into the served
HTML at publish time. app.js still hydrates the full interactive report from the same JSON file.
Run this after webpage/build-data.py, and commit both the template and the rendered page.
"""

from __future__ import annotations

import json
import re
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
