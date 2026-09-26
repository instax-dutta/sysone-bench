"""Assemble the multi-model comparison document and figure source data from sealed runs.

This module is the reporting boundary. It reads finished, checksum-verified run directories and
the sealed manifest, recomputes every published number from the raw predictions, cross-checks the
recomputation against each run's own sealed summary, and only then emits a comparison document for
``benchmark.figure_data``.

Nothing here calls a model. Nothing here writes into a run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchmark import figure_data, metrics, statistics
from benchmark.canonical import canonical_json

MODEL_LABELS = ("laya", "jev", "qwen-pcd")
PRIMARY_CONTRAST = ("laya", "jev")
REPLICATES = 20_000
SEED = 42
_PROBABILITY_PRIMITIVES = ("choice", "noul")


@dataclass(frozen=True)
class RunFacts:
    """One finished run, with its own checksum identity and raw prediction rows."""

    label: str
    run_id: str
    checksum: str
    run_dir: Path
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def verified_run_checksum(run_dir: Path) -> str:
    """Verify every artifact listed in ``checksums.sha256`` and return the manifest digest."""
    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be a Path")
    checksum_path = run_dir / "checksums.sha256"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise ValueError(f"run {run_dir.name} has no regular checksums.sha256")
    entries: list[tuple[str, str]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"run {run_dir.name} has a malformed checksum line")
        entries.append((parts[1], parts[0]))
    if not entries:
        raise ValueError(f"run {run_dir.name} has an empty checksum manifest")
    for relative, expected in entries:
        target = run_dir / relative
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"run {run_dir.name} is missing regular artifact {relative}")
        observed = _file_digest(target)
        if observed != expected:
            raise ValueError(f"run {run_dir.name} artifact {relative} fails its checksum")
    return hashlib.sha256(checksum_path.read_bytes()).hexdigest()


def load_run(label: str, run_dir: Path) -> RunFacts:
    """Load one sealed run, refusing any run whose artifacts do not match their checksums."""
    if label not in MODEL_LABELS:
        raise ValueError(f"unknown model label {label!r}")
    checksum = verified_run_checksum(run_dir)
    metadata = _read_json_object(run_dir / "metadata.json", "run metadata")
    summary = _read_json_object(run_dir / "summary.json", "run summary")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or run_id != run_dir.name:
        raise ValueError(f"run {run_dir.name} metadata run_id does not match its directory")
    rows: list[Mapping[str, Any]] = []
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"run {run_dir.name} prediction row must be an object")
            rows.append(cast(Mapping[str, Any], value))
    return RunFacts(
        label=label,
        run_id=run_id,
        checksum=checksum,
        run_dir=run_dir,
        rows=tuple(rows),
        summary=summary,
        metadata=metadata,
    )


def load_manifest(manifest_path: Path) -> dict[str, Mapping[str, Any]]:
    """Index the sealed manifest by case ID."""
    records: dict[str, Mapping[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("manifest record must be an object")
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("manifest record case_id must be a non-empty string")
            if case_id in records:
                raise ValueError(f"manifest repeats case {case_id}")
            records[case_id] = cast(Mapping[str, Any], record)
    if not records:
        raise ValueError("manifest is empty")
    return records


def decision_correctness(
    run: RunFacts, manifest: Mapping[str, Mapping[str, Any]]
) -> dict[tuple[str, str], bool]:
    """Recompute per-decision correctness from raw predictions using the public metric rule."""
    result: dict[tuple[str, str], bool] = {}
    for row in run.rows:
        case_id = cast(str, row["case_id"])
        record = manifest.get(case_id)
        if record is None:
            raise ValueError(f"run {run.run_id} case {case_id} is not in the manifest")
        questions = {cast(str, question["qid"]): question for question in record["questions"]}
        expected = cast(Mapping[str, Any], row["expected"])
        answers = cast(Mapping[str, Any], row["answers"])
        for qid in cast(Sequence[str], row["question_ids"]):
            question = questions.get(qid)
            if question is None:
                raise ValueError(f"run {run.run_id} references unknown question {qid}")
            result[(case_id, qid)] = metrics.decision_correct(
                cast(str, question["type"]), expected[qid], answers[qid]
            )
    return result


def _evaluation_rows(run: RunFacts) -> list[Mapping[str, Any]]:
    rows = [row for row in run.rows if row.get("phase") == "evaluation"]
    if not rows:
        raise ValueError(f"run {run.run_id} has no evaluation rows")
    return rows


def suite_table(
    run: RunFacts,
    correctness: Mapping[tuple[str, str], bool],
) -> dict[str, dict[str, Any]]:
    """Per-suite evaluation accuracy, decision count, and latency, recomputed from raw rows."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in _evaluation_rows(run):
        buckets.setdefault(cast(str, row["suite_id"]), []).append(row)
    table: dict[str, dict[str, Any]] = {}
    for suite_id, rows in buckets.items():
        decisions = [
            correctness[(cast(str, row["case_id"]), qid)]
            for row in rows
            for qid in cast(Sequence[str], row["question_ids"])
        ]
        if not decisions:
            raise ValueError(f"run {run.run_id} suite {suite_id} has no evaluation decisions")
        latencies = sorted(float(row["latency_seconds"]) for row in rows)
        table[suite_id] = {
            "accuracy": math.fsum(1.0 for value in decisions if value) / len(decisions),
            "cases": len(rows),
            "decisions": len(decisions),
            "p50_seconds": _percentile(latencies, 0.5),
            "p95_seconds": _percentile(latencies, 0.95),
        }
    return table


def verify_against_sealed_summary(
    run: RunFacts, table: Mapping[str, Mapping[str, Any]]
) -> None:
    """Fail unless the recomputed numbers reproduce the run's own sealed summary exactly."""
    sealed = run.summary
    if "accuracy" not in sealed:
        raise ValueError(f"run {run.run_id} summary has no evaluation accuracy")
    if not isinstance(sealed["accuracy"], (int, float)) or isinstance(sealed["accuracy"], bool):
        raise TypeError(f"run {run.run_id} summary accuracy must be a number")
    recomputed_cases = sum(int(entry["cases"]) for entry in table.values())
    recomputed_decisions = sum(int(entry["decisions"]) for entry in table.values())
    if recomputed_cases != sealed.get("cases"):
        raise ValueError(f"run {run.run_id} case count disagrees with its sealed summary")
    if recomputed_decisions != sealed.get("decisions"):
        raise ValueError(f"run {run.run_id} decision count disagrees with its sealed summary")
    sealed_suites = cast(Mapping[str, Mapping[str, Any]], sealed.get("suites", {}))
    if set(sealed_suites) != set(table):
        raise ValueError(f"run {run.run_id} suite set disagrees with its sealed summary")
    for suite_id, entry in table.items():
        sealed_suite = sealed_suites[suite_id]
        if sealed_suite.get("accuracy") != entry["accuracy"]:
            raise ValueError(
                f"run {run.run_id} suite {suite_id} accuracy disagrees with its sealed summary"
            )
        if sealed_suite.get("cases") != entry["cases"]:
            raise ValueError(
                f"run {run.run_id} suite {suite_id} case count disagrees with its sealed summary"
            )
        if sealed_suite.get("decisions") != entry["decisions"]:
            raise ValueError(
                f"run {run.run_id} suite {suite_id} decision count disagrees with its sealed summary"
            )


def _decision_keys(
    suite_id: str,
    run: RunFacts,
    manifest: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for row in _evaluation_rows(run):
        if row["suite_id"] != suite_id:
            continue
        case_id = cast(str, row["case_id"])
        for qid in cast(Sequence[str], row["question_ids"]):
            keys.append((case_id, qid))
    if not keys:
        raise ValueError(f"run {run.run_id} suite {suite_id} has no evaluation decisions")
    return keys


def paired_contrast(
    low: RunFacts,
    high: RunFacts,
    low_correctness: Mapping[tuple[str, str], bool],
    high_correctness: Mapping[tuple[str, str], bool],
    manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Paired state-cluster bootstrap intervals and permutation tests per suite.

    The low-label run is the baseline, so every value is ``high - low``. Decisions are clustered
    by case ID because several questions in one case share a state and are not independent.
    """
    if low.label not in MODEL_LABELS or high.label not in MODEL_LABELS:
        raise ValueError("contrast labels must be known models")
    suites = sorted(
        {row["suite_id"] for row in _evaluation_rows(low)}
        | {row["suite_id"] for row in _evaluation_rows(high)}
    )
    contrast: dict[str, dict[str, Any]] = {}
    for suite_id in suites:
        low_keys = _decision_keys(suite_id, low, manifest)
        high_keys = _decision_keys(suite_id, high, manifest)
        if low_keys != high_keys:
            raise ValueError(f"suite {suite_id} decision keys differ between runs")
        for key in low_keys:
            if key not in low_correctness or key not in high_correctness:
                raise ValueError(
                    f"suite {suite_id} correctness is missing decision {key[0]}/{key[1]}"
                )
        clusters = [f"{case_id}::{qid}" for case_id, qid in low_keys]
        low_rows = [1.0 if low_correctness[key] else 0.0 for key in low_keys]
        high_rows = [1.0 if high_correctness[key] else 0.0 for key in high_keys]
        bootstrap = statistics.paired_cluster_bootstrap(
            low_rows, high_rows, clusters, replicates=REPLICATES, seed=SEED
        )
        permutation = statistics.paired_cluster_permutation(
            low_rows, high_rows, clusters, replicates=REPLICATES, seed=SEED
        )
        contrast[suite_id] = {
            "delta": bootstrap["point_delta"],
            "ci_low": bootstrap["ci_low"],
            "ci_high": bootstrap["ci_high"],
            "p_value": permutation["p_value"],
        }
    adjusted = statistics.holm_adjust(
        [contrast[suite_id]["p_value"] for suite_id in sorted(contrast)]
    )
    for suite_id, value in zip(sorted(contrast), adjusted, strict=True):
        contrast[suite_id]["p_value_holm"] = value
    return contrast


def _calibration_rows(
    runs: Mapping[str, RunFacts],
) -> list[dict[str, Any]]:
    """One record per model, primitive, and calibration metric, attributed to its own run."""
    records: list[dict[str, Any]] = []
    for label in MODEL_LABELS:
        run = runs[label]
        summary = metrics.summarize_predictions(list(run.rows))
        counts = _primitive_counts(run)
        for primitive in _PROBABILITY_PRIMITIVES:
            values = cast(Mapping[str, Any], summary[primitive])
            for metric in ("ece_raw", "ece_fitted"):
                value = values.get(metric)
                if value is None:
                    continue
                records.append(
                    {
                        "primitive": primitive,
                        "metric": metric,
                        "value": value,
                        "model": label,
                        "run_id": run.run_id,
                        "checksum": run.checksum,
                        "split": "evaluation",
                        "unit": "fraction",
                        "metric_definition": _calibration_definition(primitive, metric),
                        "sample_count": counts[primitive],
                    }
                )
    return records


def _primitive_counts(run: RunFacts) -> dict[str, int]:
    """Evaluation decision count per question type, for calibration sample sizes."""
    counts: dict[str, int] = {}
    for row in _evaluation_rows(run):
        questions = {cast(str, question["qid"]): question for question in row["questions"]}
        for qid in cast(Sequence[str], row["question_ids"]):
            question_type = cast(str, questions[qid]["type"])
            counts[question_type] = counts.get(question_type, 0) + 1
    return counts


def _calibration_definition(primitive: str, metric: str) -> str:
    if metric == "ece_raw":
        return f"Expected calibration error for {primitive} using raw evaluation probabilities"
    return (
        f"Expected calibration error for {primitive} using calibration-fitted evaluation "
        "probabilities"
    )


def _efficiency_block(
    tables: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, float]]:
    block: dict[str, dict[str, float]] = {}
    for label in MODEL_LABELS:
        latencies = sorted(
            float(entry["p50_seconds"]) for entry in tables[label].values()
        )
        p95 = sorted(float(entry["p95_seconds"]) for entry in tables[label].values())
        block[label] = {
            "p50_ms": round(_percentile(latencies, 0.5) * 1000.0, 3),
            "p95_ms": round(_percentile(p95, 0.95) * 1000.0, 3),
        }
    return block


def _multilingual_block(
    runs: Mapping[str, RunFacts],
    manifest: Mapping[str, Mapping[str, Any]],
    correctness: Mapping[str, Mapping[tuple[str, str], bool]],
) -> dict[str, dict[str, float]]:
    """Per-language evaluation accuracy, accumulated as counts then divided once."""
    counts: dict[str, dict[str, list[int]]] = {}
    for label in MODEL_LABELS:
        for row in _evaluation_rows(runs[label]):
            if row["suite_id"] != "multilingual_intent":
                continue
            case_id = cast(str, row["case_id"])
            language = manifest[case_id]["state"].get("language")
            if not isinstance(language, str) or not language:
                raise ValueError(f"case {case_id} has no language for the multilingual breakdown")
            bucket = counts.setdefault(language, {}).setdefault(label, [0, 0])
            for qid in cast(Sequence[str], row["question_ids"]):
                bucket[0] += 1
                bucket[1] += 1 if correctness[label][(case_id, qid)] else 0
    block: dict[str, dict[str, float]] = {}
    for language in sorted(counts):
        block[language] = {
            label: correct / total for label, (total, correct) in sorted(counts[language].items())
        }
    return block


def build_comparison(
    runs: Mapping[str, RunFacts],
    manifest: Mapping[str, Mapping[str, Any]],
    correctness: Mapping[str, Mapping[tuple[str, str], bool]],
    tables: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the three-model comparison document consumed by ``build_figure_data``."""
    manifest_digests = {run.metadata.get("manifest_digest") for run in runs.values()}
    if len(manifest_digests) != 1:
        raise ValueError("runs were produced against different manifests")
    dataset_versions = {run.metadata.get("dataset_version") for run in runs.values()}
    if len(dataset_versions) != 1:
        raise ValueError("runs report different dataset versions")
    manifest_digest = cast(str, next(iter(manifest_digests)))
    dataset_version = cast(str, next(iter(dataset_versions)))
    run_block: dict[str, Any] = {}
    for label in MODEL_LABELS:
        run = runs[label]
        identity = cast(Mapping[str, Any], run.metadata.get("model", {}))
        run_block[label] = {
            "run_id": run.run_id,
            "checksum": run.checksum,
            "model": identity.get("model", run.run_id),
            "revision": identity.get("revision", "unknown"),
            "serving": identity.get("serving", "unknown"),
        }
    contrast = paired_contrast(
        runs[PRIMARY_CONTRAST[0]],
        runs[PRIMARY_CONTRAST[1]],
        correctness[PRIMARY_CONTRAST[0]],
        correctness[PRIMARY_CONTRAST[1]],
        manifest,
    )
    suites: dict[str, Any] = {}
    for suite_id in sorted(tables[PRIMARY_CONTRAST[0]]):
        entry: dict[str, Any] = {
            "decisions": tables[PRIMARY_CONTRAST[0]][suite_id]["decisions"],
        }
        for label in MODEL_LABELS:
            entry[label] = tables[label][suite_id]["accuracy"]
        entry.update(contrast[suite_id])
        suites[suite_id] = entry
    comparison: dict[str, Any] = {
        "schema_version": 2,
        "dataset": {
            "version": dataset_version,
            "manifest_digest": manifest_digest,
            "evaluation_cases": runs[PRIMARY_CONTRAST[0]].summary["cases"],
            "evaluation_decisions": runs[PRIMARY_CONTRAST[0]].summary["decisions"],
        },
        "runs": run_block,
        "suites": suites,
        "calibration": _calibration_rows(runs),
        "efficiency": _efficiency_block(tables),
        "multilingual": _multilingual_block(runs, manifest, correctness),
    }
    canonical_json(comparison)
    return comparison


def write_report(
    runs: Mapping[str, RunFacts],
    manifest_path: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Recompute, cross-check, and write the comparison document plus figure source data.

    The report directory must not already exist, so a published report is never partially
    replaced and a failed attempt leaves nothing behind.
    """
    manifest = load_manifest(manifest_path)
    correctness: dict[str, dict[tuple[str, str], bool]] = {}
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for label in MODEL_LABELS:
        correctness[label] = decision_correctness(runs[label], manifest)
        tables[label] = suite_table(runs[label], correctness[label])
        verify_against_sealed_summary(runs[label], tables[label])
    comparison = build_comparison(runs, manifest, correctness, tables)
    data = figure_data.build_figure_data(comparison)
    if output_root.exists():
        raise FileExistsError(f"report output root already exists: {output_root.name}")
    report_path = output_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_source = figure_data.write_figure_data(data, output_root / "figures")
    with report_path.open("x", encoding="utf-8") as file:
        file.write(canonical_json(comparison).decode("utf-8") + "\n")
    return {"comparison": report_path, "figure_source": figure_source}


def figure_families() -> Iterable[str]:
    return figure_data.FIGURE_FAMILIES


def main(argv: Sequence[str] | None = None) -> int:
    """Build a report directory from three sealed runs.

    Prints only the two artifact paths on success. Any failure prints a redacted reason and
    returns non-zero without writing a partial report.
    """
    parser = argparse.ArgumentParser(
        prog="benchmark.report", description="Assemble the multi-model report from sealed runs"
    )
    parser.add_argument("--laya", type=Path, required=True, help="Laya run directory")
    parser.add_argument("--jev", type=Path, required=True, help="Jev run directory")
    parser.add_argument("--qwen", type=Path, required=True, help="Qwen PCD run directory")
    parser.add_argument("--manifest", type=Path, required=True, help="sealed manifest path")
    parser.add_argument("--output-root", type=Path, required=True, help="new report directory")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return 0 if error.code is None else int(error.code)
    try:
        runs = {
            "laya": load_run("laya", args.laya),
            "jev": load_run("jev", args.jev),
            "qwen-pcd": load_run("qwen-pcd", args.qwen),
        }
        written = write_report(runs, args.manifest, args.output_root)
    except FileExistsError:
        print("report failed: output root already exists", file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"report failed: {error}", file=sys.stderr)
        return 1
    for name in ("comparison", "figure_source"):
        print(f"{name}: {written[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
