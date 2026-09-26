# Sysone-bench v2 Statistics, Reports, and Benchmark Graphics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute task-specific metrics, paired clustered uncertainty, raw and fitted calibration, risk-coverage, and produce seven deterministic Qwen 3.5-style image sheets with source data, reports, and release audits.

**Architecture:** Statistics and plotting consume only validated v2 comparison JSON. Reports and figures share a single figure-source-data layer, so every visual value and caption can be traced to a run ID, comparison ID, and dataset manifest digest. No plotting module imports model adapters or performs network calls.

**Tech Stack:** Python 3.12, NumPy, pandas, matplotlib with the Agg backend, Pillow for image checks, pytest, Ruff, mypy, and GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-24-sysone-bench-v2-design.md`

**Depends on:** `docs/superpowers/plans/2026-09-24-sysone-bench-v2-core.md`, `docs/superpowers/plans/2026-09-24-sysone-bench-v2-data.md`, and `docs/superpowers/plans/2026-09-24-sysone-bench-v2-runners.md`

## Global Constraints

- Never calculate a cross-suite composite score.
- Use 20,000 paired state-cluster bootstrap replicates and 20,000 paired state-cluster permutation tests with seed `42`.
- Use half-up nearest-level rounding for score correctness.
- Use top-label probability for choice gating and P(true) for Noul gating.
- Do not plot score outputs as probability ECE or risk-coverage.
- Use a white chart canvas even though brainstorming pages use dark mode.
- Use the fixed palette: Laya `#009E73`, Jev `#0072B2`, Qwen-PCD `#E69F00`, Router `#CC79A7`.
- Every graph is generated from validated source data and emits PNG, SVG, PDF, CSV, caption, and figure-manifest records.
- Historical reports and result JSONs remain available under archive paths.
- Do not commit, push, or publish automatically.

## File Map

- Create `benchmark/metrics.py` - task-specific decision scoring and suite summaries.
- Create `benchmark/statistics.py` - paired bootstrap, permutation tests, Holm adjustment, and calibrators.
- Create `benchmark/figure_data.py` - immutable rows consumed by every figure.
- Create `benchmark/graphics.py` - Qwen-style figure rendering and output manifests.
- Create `benchmark/reporting.py` - canonical REPORT and FEEDBACK_REPORT generation.
- Create `benchmark/release_audit.py` - legacy hash, source-data, secret, and artifact checks.
- Create `graphs/v2/AGENTS.md` - graph generation and publication contract.
- Create `reports/AGENTS.md` - report/archive ownership.
- Create `tests/test_metrics.py` - golden metric tests.
- Create `tests/test_statistics.py` - deterministic statistical tests.
- Create `tests/test_graphics.py` - rendering, dimensions, and source-equality tests.
- Create `tests/test_reports.py` - report value and archive tests.
- Create `tests/test_release_audit.py` - legacy and secret checks.
- Create `assets/fonts/` - pinned Inter font files and OFL license.
- Create `tools/fetch_inter_font.py` - reproducible font download and checksum verification.
- Update `README.md`, `REPORT.md`, `FEEDBACK_REPORT.md`, `MACHINES.md`, `PLAN.md`, and `AGENTS.md` after validated results exist.
- Create `THIRD_PARTY_NOTICES.md` - PCD, fonts, model references, and dataset notices.

- Create `tests/fixtures.py` - deterministic comparison, figure, report, and audit fixtures.

---

### Task 0: Define deterministic release test fixtures

**Files:**
- Create: `tests/fixtures.py`

**Interfaces:**
- Produces `comparison_fixture() -> dict[str, object]`.
- Produces `figure_fixture() -> dict[str, object]`.
- Produces `write_fake_release(tmp_path: Path, extra_file: str | None = None, content: str | None = None) -> None`.
- Produces `fake_comparison_path(tmp_path: Path) -> Path`.

- [ ] **Step 1: Add release fixture data**

```python
import json
from pathlib import Path


def comparison_fixture() -> dict[str, object]:
    suites = ["triage", "guardrails", "moderation", "agnews", "emotion", "banking77_12", "mnli", "sst5", "multilingual_intent"]
    return {
        "schema_version": 2,
        "dataset": {"version": "2.0.0", "manifest_digest": "a" * 64, "evaluation_cases": 952, "evaluation_decisions": 1240},
        "runs": {"laya": {"run_id": "laya-test", "checksum": "b" * 64, "model": "fake-laya"}, "jev": {"run_id": "jev-test", "checksum": "c" * 64, "model": "jev-1.13.0"}, "qwen-pcd": {"run_id": "qwen-test", "checksum": "d" * 64, "model": "fake-qwen"}},
        "suites": {suite: {"laya": 0.80, "jev": 0.88, "qwen-pcd": 0.75, "decisions": 10, "delta": 0.08, "ci_low": 0.01, "ci_high": 0.15, "p_value": 0.03} for suite in suites},
        "calibration": {"choice": {"ece_raw": 0.1, "ece_fitted": 0.05}, "noul": {"ece_raw": 0.08, "ece_fitted": 0.04}},
        "efficiency": {"laya": {"p50_ms": 600, "p95_ms": 900, "throughput": 1.2}, "jev": {"p50_ms": 900, "p95_ms": 1200, "throughput": 0.8}, "qwen-pcd": {"p50_ms": 1000, "p95_ms": 1800, "throughput": 0.6}},
        "multilingual": {"Hindi": {"laya": 0.3, "jev": 1.0, "qwen-pcd": 0.8}, "Spanish": {"laya": 0.3, "jev": 1.0, "qwen-pcd": 0.8}},
        "router": {"english_delta": 0.0, "multilingual_delta": 0.4, "route_counts": {"english": 500, "multilingual": 100}},
    }


def figure_fixture() -> dict[str, object]:
    comparison = comparison_fixture()
    rows = []
    for model in ("laya", "jev", "qwen-pcd"):
        for suite in comparison["suites"]:
            rows.append({"suite_id": suite, "model": model, "value": comparison["suites"][suite][model], "decisions": 10, "run_id": comparison["runs"][model]["run_id"], "checksum": comparison["runs"][model]["checksum"], "manifest_digest": comparison["dataset"]["manifest_digest"]})
    return {"accuracy": rows, "paired_effects": rows, "calibration": [], "risk_coverage": [], "efficiency": [], "multilingual": [], "router": []}


def write_fake_release(tmp_path: Path, extra_file: str | None = None, content: str | None = None) -> None:
    (tmp_path / "README.md").write_text("current", encoding="utf-8")
    (tmp_path / "REPORT.md").write_text("1,240 evaluation decisions", encoding="utf-8")
    if extra_file is not None:
        (tmp_path / extra_file).write_text(content or "", encoding="utf-8")


def fake_comparison_path(tmp_path: Path) -> Path:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison_fixture()), encoding="utf-8")
    return path
```

- [ ] **Step 2: Run the fixture import test**

Run: `uv run python -c "import sys; sys.path.insert(0, 'tests'); from fixtures import comparison_fixture; assert len(comparison_fixture()['suites']) == 9"`

Expected: PASS after the fixture module is created.

---

### Task 1: Implement task-specific metric functions

**Files:**
- Create: `benchmark/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces `decision_correct(question_type: str, expected: Any, answer: Mapping[str, Any]) -> bool`.
- Produces `score_level(prediction: float) -> int`.
- Produces `score_correct(expected: int, prediction: float) -> bool`.
- Produces `summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]`.
- Produces `macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> float`.

- [ ] **Step 1: Write failing golden metric tests**

```python
import math


def test_choice_requires_exact_label():
    assert decision_correct("choice", "yes", {"type": "choice", "choice": "yes"})
    assert not decision_correct("choice", "yes", {"type": "choice", "choice": "no"})


def test_noul_uses_half_threshold():
    assert decision_correct("noul", 1, {"type": "noul", "noul": 0.5})
    assert not decision_correct("noul", 1, {"type": "noul", "noul": 0.4999})


def test_score_uses_half_up_nearest_level():
    assert score_level(0.5) == 1
    assert score_level(1.5) == 2
    assert score_correct(2, 1.6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -q`

Expected: FAIL because `benchmark.metrics` does not exist.

- [ ] **Step 3: Implement correctness and score rounding**

Use:

```python
import math
from collections.abc import Mapping
from typing import Any


def score_level(prediction: float) -> int:
    if prediction < 0 or not math.isfinite(prediction):
        raise ValueError("score must be finite and nonnegative")
    return math.floor(prediction + 0.5)


def score_correct(expected: int, prediction: float) -> bool:
    return score_level(prediction) == expected


def decision_correct(question_type: str, expected: object, answer: Mapping[str, object]) -> bool:
    if question_type == "choice":
        return answer["choice"] == expected
    if question_type == "noul":
        return int(float(answer["noul"]) >= 0.5) == int(expected)
    if question_type == "score":
        return score_level(float(answer["score"])) == int(expected)
    raise ValueError(f"unsupported question type: {question_type}")
```

- [ ] **Step 4: Implement suite summaries**

Compute exact accuracy, macro-F1, balanced accuracy, per-class recall, Brier, NLL, raw ECE, and fitted ECE for choice; add AUROC and AUPRC for Noul; add MAE, RMSE, nearest-level accuracy, within-one accuracy, Spearman correlation, and ordinal confusion for score. Return `None` for metrics that do not apply to a primitive.

- [ ] **Step 5: Run golden tests**

Run: `uv run pytest tests/test_metrics.py -q`

Expected: PASS.

---

### Task 2: Implement paired clustered statistics and calibration

**Files:**
- Create: `benchmark/statistics.py`
- Create: `tests/test_statistics.py`

**Interfaces:**
- Produces `paired_cluster_bootstrap(a_rows, b_rows, cluster_ids, replicates=20000, seed=42) -> dict[str, float]`.
- Produces `paired_cluster_permutation(a_rows, b_rows, cluster_ids, replicates=20000, seed=42) -> dict[str, float]`.
- Produces `holm_adjust(p_values: Sequence[float]) -> list[float]`.
- Produces `fit_temperature(probabilities: Sequence[float], labels: Sequence[int]) -> float`.
- Produces `apply_temperature(probabilities: Sequence[float], temperature: float) -> list[float]`.

- [ ] **Step 1: Write failing deterministic tests**

```python
def test_bootstrap_is_deterministic():
    first = paired_cluster_bootstrap([0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], replicates=1000, seed=42)
    second = paired_cluster_bootstrap([0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], replicates=1000, seed=42)
    assert first == second


def test_cluster_resampling_keeps_questions_together():
    result = paired_cluster_bootstrap([0, 0, 1, 1], [1, 1, 1, 1], [10, 10, 11, 11], replicates=1000, seed=42)
    assert result["point_delta"] == 0.5


def test_holm_adjustment_is_monotone():
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_statistics.py -q`

Expected: FAIL because `benchmark.statistics` does not exist.

- [ ] **Step 3: Implement state-cluster bootstrap**

Group row indices by `cluster_ids`, sample the same cluster IDs with replacement for both model arrays, compute the row-size-weighted `b_rows - a_rows` decision-level delta for each replicate, and return `point_delta`, `ci_low`, `ci_high`, `replicates`, and `seed`.

- [ ] **Step 4: Implement paired permutation and Holm correction**

For each replicate, independently flip the sign of each state cluster's paired difference, compute the absolute aggregate, and return a two-sided p-value. Sort p-values for Holm adjustment, enforce monotonicity, and restore original order.

- [ ] **Step 5: Implement temperature fitting and evaluation**

Fit a scalar temperature on calibration probabilities with stable log-space optimization, record the fitted value, and apply it only to evaluation probabilities. Preserve raw probabilities and fitted probabilities separately.

- [ ] **Step 6: Run statistics tests**

Run: `uv run pytest tests/test_statistics.py -q`

Expected: PASS.

---

### Task 3: Build immutable figure source data

**Files:**
- Create: `benchmark/figure_data.py`
- Create: `tests/test_figure_data.py`

**Interfaces:**
- Produces `build_figure_data(comparison: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]`.
- Produces `write_figure_data(data: Mapping[str, Any], output_root: Path) -> Path`.
- Produces `validate_figure_data(data: Mapping[str, Any], comparison: Mapping[str, Any]) -> None`.

- [ ] **Step 1: Write failing source-equality tests**

```python
def test_figure_data_contains_every_suite_and_model():
    data = build_figure_data(comparison_fixture())
    assert set(data["accuracy"]) == {"laya", "jev", "qwen-pcd"}
    assert {row["suite_id"] for row in data["accuracy"]} == set(comparison_fixture()["suites"])


def test_figure_data_rejects_missing_run_hash():
    data = build_figure_data(comparison_fixture())
    data["accuracy"][0].pop("run_id")
    with pytest.raises(ValueError, match="run_id"):
        validate_figure_data(data, comparison_fixture())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_figure_data.py -q`

Expected: FAIL because `benchmark.figure_data` does not exist.

- [ ] **Step 3: Implement figure rows**

Create rows for `accuracy`, `paired_effects`, `calibration`, `risk_coverage`, `efficiency`, `multilingual`, and `router`. Every row must contain source run IDs, source checksums, dataset digest, split, metric definition, sample count, and the plotted values.

- [ ] **Step 4: Write CSV and JSON outputs**

Write one CSV per figure family under `graphs/v2/data/` and one `figure-source.json` with the complete row set. Sort rows by suite and model before writing so output is deterministic.

- [ ] **Step 5: Run figure-data tests**

Run: `uv run pytest tests/test_figure_data.py -q`

Expected: PASS.

---

### Task 4: Fetch and verify the Inter font assets

**Files:**
- Create: `tools/fetch_inter_font.py`
- Create: `assets/fonts/Inter-Regular.ttf`
- Create: `assets/fonts/Inter-SemiBold.ttf`
- Create: `assets/fonts/OFL.txt`
- Test: `tests/test_font_assets.py`

**Interfaces:**
- Produces `verify_font_assets(font_root: Path) -> dict[str, str]`.
- Produces exact SHA-256 values for both font files in `assets/fonts/checksums.json`.

- [ ] **Step 1: Write failing font tests**

```python
def test_font_files_and_license_exist():
    verify_font_assets(Path("assets/fonts"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_font_assets.py -q`

Expected: FAIL because the font assets do not exist.

- [ ] **Step 3: Implement the pinned downloader**

Download `https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip`, extract only `Inter-Regular.ttf` and `Inter-SemiBold.ttf`, copy the accompanying OFL license, calculate SHA-256, and write `checksums.json`. The script must verify checksums on every run and never download a different filename silently.

- [ ] **Step 4: Run the font test**

Run: `uv run pytest tests/test_font_assets.py -q`

Expected: PASS.

---

### Task 5: Implement the seven Qwen-style figure renderers

**Files:**
- Create: `benchmark/graphics.py`
- Create: `tests/test_graphics.py`
- Create: `graphs/v2/AGENTS.md`

**Interfaces:**
- Produces `render_release(data: Mapping[str, Any], output_root: Path) -> dict[str, Any]`.
- Produces `render_figure(figure_id: str, rows: Sequence[Mapping[str, Any]], output_root: Path) -> dict[str, Any]`.
- Produces a figure manifest with PNG, SVG, PDF, CSV, caption, source run IDs, and checksums.

- [ ] **Step 1: Write failing rendering tests**

```python
from PIL import Image


def test_accuracy_figure_is_3200_by_1800(tmp_path):
    result = render_release(figure_fixture(), tmp_path)
    with Image.open(result["figures"]["01-accuracy-by-suite"]["png"]) as image:
        assert image.size == (3200, 1800)


def test_graph_manifest_has_all_artifacts(tmp_path):
    result = render_release(figure_fixture(), tmp_path)
    for figure in result["figures"].values():
        assert Path(figure["png"]).exists()
        assert Path(figure["svg"]).exists()
        assert Path(figure["pdf"]).exists()
        assert Path(figure["csv"]).exists()
        assert Path(figure["caption"]).exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `MPLBACKEND=Agg uv run pytest tests/test_graphics.py -q`

Expected: FAIL because the graphics module does not exist.

- [ ] **Step 3: Implement the shared style**

Use matplotlib `Agg`, fixed `svg.hashsalt`, fixed metadata, white backgrounds, 3200×1800 PNG output, Inter fonts, direct labels, thin gridlines, and the approved palette. Set accuracy limits to `[0, 100]` and never use a hidden truncated axis.

- [ ] **Step 4: Implement the seven figure IDs**

- `01-accuracy-by-suite`: 3×3 bars.
- `02-paired-effects`: 3×3 delta intervals around zero.
- `03-calibration`: 2×2 reliability panels.
- `04-risk-coverage`: 2×2 answer/state coverage panels.
- `05-efficiency`: 2×2 p50, p95, throughput, resource-use panels.
- `06-multilingual`: five language panels.
- `07-router-deployment`: English parity and routed multilingual gain.

Every panel receives a title, metric subtitle, direct labels, legend, sample size, uncertainty method, run IDs, dataset version, and manifest hash.

- [ ] **Step 5: Add graph AGENTS rules**

Document source-data equality, no model imports, fixed palette, no radar charts, no logos, output formats, and the rule that a figure cannot be committed without its CSV, caption, and manifest entry.

- [ ] **Step 6: Run rendering tests**

Run: `MPLBACKEND=Agg uv run pytest tests/test_graphics.py -q`

Expected: PASS.

---

### Task 6: Generate canonical reports from validated data

**Files:**
- Create: `benchmark/reporting.py`
- Create: `tests/test_reports.py`
- Create: `reports/AGENTS.md`
- Create: `reports/archive/`
- Modify: `README.md`
- Modify: `REPORT.md`
- Modify: `FEEDBACK_REPORT.md`
- Modify: `MACHINES.md`
- Modify: `PLAN.md`

**Interfaces:**
- Produces `render_report(comparison: Mapping[str, Any], output_root: Path) -> Path`.
- Produces `render_feedback_report(comparison: Mapping[str, Any], output_root: Path) -> Path`.
- Produces `archive_existing_reports(repo_root: Path, archive_root: Path) -> list[Path]`.

- [ ] **Step 1: Write failing report tests**

```python
def test_report_uses_validated_values_not_legacy_values(tmp_path):
    path = render_report(comparison_fixture(), tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "1,240" in text
    assert "0.888" not in text


def test_archive_does_not_delete_existing_report(tmp_path):
    source = tmp_path / "REPORT.md"
    source.write_text("historical", encoding="utf-8")
    paths = archive_existing_reports(tmp_path, tmp_path / "reports/archive")
    assert source.exists()
    assert paths
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_reports.py -q`

Expected: FAIL because reporting functions do not exist.

- [ ] **Step 3: Implement report generation**

Generate the canonical report from the same figure source rows and comparison JSON. Include methods, dataset counts, model identities, run protocol, paired effects, calibration, gating, efficiency, provenance, limitations, and artifact links. Generate the feedback report from final values without historical mixed versions.

- [ ] **Step 4: Archive existing reports**

Copy current `REPORT.md`, `FEEDBACK_REPORT.md`, and `MACHINES.md` into dated files under `reports/archive/` before replacing root files. Never delete historical content.

- [ ] **Step 5: Update the README and plan**

Embed the seven current images, link source CSVs and figure manifests, document local and the shared host commands, state the narrowed manifest-identity claim, and mark the approved plan phases.

- [ ] **Step 6: Add third-party notices**

Record PCD upstream URL and revision, local backend changes, Inter font license, model references, dataset citations and licenses, and data-egress behavior.

- [ ] **Step 7: Run report tests**

Run: `uv run pytest tests/test_reports.py -q`

Expected: PASS.

---

### Task 7: Add release audit and final CI checks

**Files:**
- Create: `benchmark/release_audit.py`
- Create: `tests/test_release_audit.py`
- Create: `THIRD_PARTY_NOTICES.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `AGENTS.md`
- Modify: `datasets/AGENTS.md`
- Modify: `runners/AGENTS.md`
- Modify: `results/AGENTS.md`

**Interfaces:**
- Produces `audit_release(repo_root: Path, comparison_path: Path) -> dict[str, Any]`.
- Raises `ReleaseAuditError` for legacy hash changes, missing source data, secret patterns, stale report values, or untracked graph artifacts.

- [ ] **Step 1: Write failing audit tests**

```python
def test_audit_rejects_secret_pattern(tmp_path):
    write_fake_release(tmp_path, extra_file="leak.txt", content="TYPESAFE_API_KEY=secret")
    with pytest.raises(ReleaseAuditError, match="secret"):
        audit_release(tmp_path, fake_comparison_path(tmp_path))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_release_audit.py -q`

Expected: FAIL because the audit module does not exist.

- [ ] **Step 3: Implement the audit**

Check legacy result hashes, JSON parseability, manifest digests, comparison source hashes, graph source equality, report numeric equality, tracked-secret patterns, and required artifact existence. Exclude `.env`, `.venv`, `.superpowers`, and other ignored private paths from content scans.

- [ ] **Step 4: Update DOX contracts**

Add `graphs/AGENTS.md`, `reports/AGENTS.md`, and `ops/AGENTS.md`; update root Child DOX Index and the dataset, runner, and results contracts. Record the stable dark-mode visual companion preference.

- [ ] **Step 5: Run the complete quality suite**

Run:

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run mypy benchmark
MPLBACKEND=Agg uv run pytest
uv run python -m benchmark.release_audit
```

Expected: all checks pass and the release audit reports `PASS`.

- [ ] **Step 6: Verify the worktree and legacy records**

Run:

```bash
git diff --check
git status --short --branch
git diff -- results/
```

Expected: no historical result changes, no secret files tracked, and only intended v2 source, dataset, report, graph, DOX, and test changes.

## Completion Gate

The release track is complete when all metrics have golden tests, clustered intervals are deterministic, seven figures render with complete source artifacts, reports contain only v2 facts, third-party notices are present, the audit passes, CI passes on Python 3.11 and 3.12, and no legacy result or secret is modified.
