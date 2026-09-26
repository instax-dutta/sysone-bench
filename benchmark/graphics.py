"""Render the benchmark figures from the figure source data.

Every figure is produced from ``figure-source.json`` only, so a chart can never disagree with the
rows that were validated for release. Rendering is deterministic: no randomness, no hidden state,
and no figure reads a run directory directly.

Design rules enforced here, from the binding decisions in the design spec: no radar charts, no
3D, no composite cross-suite score, and no truncated accuracy axes. Suite accuracy therefore always
starts at zero, and the per-suite view is a small-multiples grid rather than one averaged bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

MODEL_ORDER = ("jev", "laya", "qwen-pcd")
MODEL_LABELS = {
    "jev": "Jev 1.13.0",
    "laya": "Laya 0.3.11",
    "qwen-pcd": "Qwen2.5-1.5B PCD",
}
MODEL_COLORS = {
    "jev": "#3ddc97",
    "laya": "#5aa9ff",
    "qwen-pcd": "#ffb454",
}
BACKGROUND = "#0f1115"
PANEL = "#171a21"
INK = "#e8eaed"
MUTED = "#9aa3b2"
GRID = "#262b36"
ACCENT = "#ff6b6b"
FIGURE_DPI = 160
# Matplotlib stamps PNGs with a creation time and a software string by default, which would
# make two renders of identical data differ byte for byte. Dropping both keeps figures
# reproducible from figure-source.json, which is the point of rendering from it.
_PNG_METADATA = {"Software": None, "Creation Time": None}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": FIGURE_DPI,
        }
    )


def _load(figure_source: Path) -> dict[str, list[dict[str, Any]]]:
    value = json.loads(figure_source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("figure source must be a JSON object")
    data = cast(dict[str, list[dict[str, Any]]], value)
    for family, rows in data.items():
        if not isinstance(rows, list):
            raise TypeError(f"figure source {family} must be a list")
    return data


def _family(
    data: Mapping[str, Sequence[Mapping[str, Any]]], family: str
) -> list[Mapping[str, Any]]:
    rows = data.get(family)
    if rows is None:
        raise ValueError(f"figure source is missing the {family} family")
    return [cast(Mapping[str, Any], row) for row in rows]


def _axes_list(axes: Any) -> list[Any]:
    """Normalize a single Axes, a nested ndarray, or a flat list into a flat list of Axes."""
    if isinstance(axes, plt.Axes):
        return [axes]
    return [axes.flat[index] for index in range(axes.size)]


def _by_suite(
    rows: Sequence[Mapping[str, Any]], key: str = "model"
) -> dict[str, dict[str, Mapping[str, Any]]]:
    table: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        table.setdefault(cast(str, row["suite_id"]), {})[cast(str, row[key])] = row
    return table


def _format_value(value: float) -> str:
    """Label small magnitudes with enough precision to be readable."""
    magnitude = abs(value)
    if magnitude == 0.0:
        return "0"
    if magnitude < 0.001:
        return f"{value:.5f}".rstrip("0")
    if magnitude < 0.01:
        return f"{value:.4f}"
    if magnitude < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _annotate_bars(axis: Any, bars: Any, values: Sequence[float]) -> None:
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            _format_value(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=MUTED,
        )


def suite_accuracy_figure(
    data: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Figure:
    """Small multiples: one panel per suite, three equal model series, untruncated accuracy axis."""
    table = _by_suite(_family(data, "accuracy"))
    suites = sorted(table)
    columns = 3
    rows = (len(suites) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 3.2 * rows))
    flat = _axes_list(axes)
    for index, suite in enumerate(suites):
        axis = flat[index]
        present = [model for model in MODEL_ORDER if model in table[suite]]
        values = [float(table[suite][model]["value"]) for model in present]
        bars = axis.bar(
            range(len(present)),
            values,
            color=[MODEL_COLORS[model] for model in present],
            width=0.62,
        )
        _annotate_bars(axis, bars, values)
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels([MODEL_LABELS[model].split(" ")[0] for model in present])
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.set_title(suite.replace("_", " "))
        axis.grid(axis="y", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        decisions = table[suite][present[0]]["sample_count"]
        axis.annotate(
            f"n={decisions} decisions",
            xy=(0.0, -0.22),
            xycoords="axes fraction",
            fontsize=7,
            color=MUTED,
        )
    for index in range(len(suites), len(flat)):
        flat[index].axis("off")
    figure.suptitle(
        "Evaluation accuracy by suite, identical manifest for every model",
        fontsize=13,
        color=INK,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def paired_effects_figure(
    data: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Figure:
    """Jev minus Laya paired accuracy delta with 95% state-cluster bootstrap intervals."""
    rows = _family(data, "paired_effects")
    if not rows:
        raise ValueError("paired effects family has no rows to plot")
    ordered = sorted(rows, key=lambda row: (float(row["value"]), str(row["suite_id"])))
    labels = [str(row["suite_id"]).replace("_", " ") for row in ordered]
    values = [float(row["value"]) for row in ordered]
    lows = [float(row["plotted_values"].get("ci_low", value)) for row, value in zip(ordered, values, strict=True)]
    highs = [float(row["plotted_values"].get("ci_high", value)) for row, value in zip(ordered, values, strict=True)]
    holm = [row.get("p_value_holm") for row in ordered]
    figure, axis = plt.subplots(figsize=(9.5, 0.62 * len(ordered) + 2.4))
    positions = list(range(len(ordered)))
    errors = [
        [max(0.0, value - low) for value, low in zip(values, lows, strict=True)],
        [max(0.0, high - value) for value, high in zip(values, highs, strict=True)],
    ]
    colors = [ACCENT if p is not None and float(p) >= 0.05 else MODEL_COLORS["jev"] for p in holm]
    axis.errorbar(
        values,
        positions,
        xerr=errors,
        fmt="none",
        ecolor=GRID,
        elinewidth=6,
        capsize=0,
        zorder=1,
    )
    axis.scatter(values, positions, s=70, c=colors, zorder=2, edgecolors=BACKGROUND, linewidths=1.2)
    axis.axvline(0.0, color=MUTED, linewidth=1.0, linestyle="--")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Jev accuracy minus Laya accuracy (fraction)")
    axis.set_xlim(-0.05, max(0.6, max(values) + 0.1))
    axis.grid(axis="x", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for position, row, value in zip(positions, ordered, values, strict=True):
        plotted = row["plotted_values"]
        p_value = plotted.get("p_value")
        holm_value = plotted.get("p_value_holm", p_value)
        suffix = "" if p_value is None else f", Holm p={float(holm_value):.4f}"
        axis.annotate(
            f"{value:+.3f} [{float(plotted.get('ci_low', value)):+.3f}, "
            f"{float(plotted.get('ci_high', value)):+.3f}]{suffix}",
            xy=(value, position),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=MUTED,
        )
    figure.suptitle(
        "Paired difference where the same case is answered by both models",
        fontsize=12,
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 0.82, 0.96))
    return figure


def calibration_figure(data: Mapping[str, Sequence[Mapping[str, Any]]]) -> Figure:
    """Expected calibration error per model and primitive, raw and calibration-fitted."""
    rows = _family(data, "calibration")
    primitives = sorted({cast(str, row["primitive"]) for row in rows})
    metrics = ("ece_raw", "ece_fitted")
    figure, axes = plt.subplots(1, len(primitives), figsize=(5.6 * len(primitives), 4.0))
    flat = _axes_list(axes)
    for axis, primitive in zip(flat, primitives, strict=True):
        width = 0.26
        offsets = (0, 1, 2)
        for offset, model in zip(offsets, MODEL_ORDER, strict=True):
            heights: list[float] = []
            for metric in metrics:
                match = next(
                    (
                        row
                        for row in rows
                        if row["model"] == model
                        and row["primitive"] == primitive
                        and row["metric"] == metric
                    ),
                    None,
                )
                heights.append(float(match["value"]) if match is not None else 0.0)
            positions = [index + (offset - 1) * width for index in range(len(metrics))]
            bars = axis.bar(
                positions,
                heights,
                width=width,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model] if primitive == primitives[0] else None,
            )
            _annotate_bars(axis, bars, heights)
        axis.set_xticks(range(len(metrics)))
        axis.set_xticklabels(["raw", "calibration fitted"])
        axis.set_title(f"{primitive} questions")
        axis.set_ylabel("expected calibration error")
        axis.grid(axis="y", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Calibration on the evaluation split, lower is better", fontsize=12, color=INK)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    return figure


def efficiency_figure(data: Mapping[str, Sequence[Mapping[str, Any]]]) -> Figure:
    """Median and 95th-percentile end-to-end latency per model, log scale for the 40x spread."""
    rows = _family(data, "efficiency")
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        table.setdefault(cast(str, row["model"]), {})[cast(str, row["metric"])] = float(row["value"])
    metrics = ("p50_ms", "p95_ms")
    figure, axis = plt.subplots(figsize=(8.6, 4.4))
    width = 0.26
    for offset, model in zip((0, 1, 2), MODEL_ORDER, strict=True):
        heights = [table.get(model, {}).get(metric, 0.0) for metric in metrics]
        positions = [index + (offset - 1) * width for index in range(len(metrics))]
        bars = axis.bar(
            positions,
            heights,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        for bar, value in zip(bars, heights, strict=True):
            axis.annotate(
                f"{value:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=MUTED,
            )
    axis.set_yscale("log")
    axis.set_xticks(range(len(metrics)))
    axis.set_xticklabels(["median (p50)", "95th percentile (p95)"])
    axis.set_ylabel("latency per case (ms, log scale)")
    axis.set_title("End-to-end latency per case, including model and decoding time")
    axis.grid(axis="y", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    return figure


def multilingual_figure(data: Mapping[str, Sequence[Mapping[str, Any]]]) -> Figure:
    """Per-language accuracy for the multilingual intent suite."""
    rows = _family(data, "multilingual")
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        table.setdefault(cast(str, row.get("language", "all")), {})[
            cast(str, row["model"])
        ] = float(row["value"])
    languages = sorted(table)
    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    width = 0.26
    for offset, model in zip((0, 1, 2), MODEL_ORDER, strict=True):
        heights = [table[language].get(model, 0.0) for language in languages]
        positions = [index + (offset - 1) * width for index in range(len(languages))]
        bars = axis.bar(
            positions,
            heights,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        _annotate_bars(axis, bars, heights)
    axis.set_xticks(range(len(languages)))
    axis.set_xticklabels(languages)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_ylabel("accuracy")
    axis.set_title("Multilingual intent accuracy by language", pad=16)
    axis.grid(axis="y", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure


FIGURES = {
    "suite-accuracy.png": suite_accuracy_figure,
    "paired-jev-minus-laya.png": paired_effects_figure,
    "calibration-ece.png": calibration_figure,
    "latency.png": efficiency_figure,
    "multilingual-accuracy.png": multilingual_figure,
}


def render_all(figure_source: Path, output_root: Path) -> dict[str, Path]:
    """Render every figure from the figure source data into a fresh output directory.

    Every figure is built before any file is written, so a failure part way through leaves no
    output directory behind, and the directory is only created once all figures exist in memory.
    """
    if not isinstance(figure_source, Path) or not isinstance(output_root, Path):
        raise TypeError("figure_source and output_root must be Path values")
    _style()
    data = _load(figure_source)
    figures = {name: builder(data) for name, builder in FIGURES.items()}
    output_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, figure in figures.items():
        target = output_root / name
        if target.exists():
            for open_figure in figures.values():
                plt.close(open_figure)
            raise FileExistsError(f"refusing to overwrite existing figure {target}")
        figure.savefig(target, format="png", bbox_inches="tight", metadata=_PNG_METADATA)
        plt.close(figure)
        written[name] = target
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Render every figure from a figure source file.

    Prints only the written figure paths. Failures print a redacted reason and return
    non-zero, and an existing figure is never overwritten.
    """
    parser = argparse.ArgumentParser(
        prog="benchmark.graphics", description="Render figures from figure source data"
    )
    parser.add_argument("--figure-source", type=Path, required=True, help="figure-source.json")
    parser.add_argument("--output-root", type=Path, required=True, help="new figure directory")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return 0 if error.code is None else int(error.code)
    try:
        written = render_all(args.figure_source, args.output_root)
    except FileExistsError:
        print("graphics failed: a figure already exists in the output directory", file=sys.stderr)
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"graphics failed: {error}", file=sys.stderr)
        return 1
    for name in sorted(written):
        print(f"{name}: {written[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
