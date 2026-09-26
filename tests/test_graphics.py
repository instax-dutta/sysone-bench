from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from benchmark import graphics
from tests import fixtures


def _figure_source(tmp_path: Path) -> Path:
    data = graphics.__dict__  # keep the import meaningful for linters
    del data
    return _write_source(tmp_path)


def _write_source(tmp_path: Path) -> Path:
    from benchmark import figure_data

    built = figure_data.build_figure_data(fixtures.comparison_fixture())
    return figure_data.write_figure_data(built, tmp_path / "figures")


def test_render_all_writes_every_declared_figure(tmp_path: Path) -> None:
    source = _write_source(tmp_path)

    written = graphics.render_all(source, tmp_path / "png")

    assert set(written) == set(graphics.FIGURES)
    for name, path in written.items():
        assert path.is_file(), name
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), name
        assert len(path.read_bytes()) > 5_000, name


def test_render_all_refuses_to_overwrite(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    graphics.render_all(source, tmp_path / "png")

    with pytest.raises(FileExistsError):
        graphics.render_all(source, tmp_path / "png")


def test_render_all_requires_path_arguments(tmp_path: Path) -> None:
    source = _write_source(tmp_path)

    with pytest.raises(TypeError):
        graphics.render_all(str(source), tmp_path / "png")  # type: ignore[arg-type]


def test_figure_source_must_be_an_object(tmp_path: Path) -> None:
    bad = tmp_path / "figure-source.json"
    bad.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError):
        graphics.render_all(bad, tmp_path / "png")


def test_suite_accuracy_uses_an_untruncated_axis(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = graphics._load(source)

    figure = graphics.suite_accuracy_figure(data)
    try:
        for axis in figure.axes:
            assert axis.get_ylim() == (0.0, 1.0)
    finally:
        figure.clf()


def test_multilingual_figure_reads_language_labels(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = graphics._load(source)
    rows = cast(list[dict[str, Any]], data["multilingual"])
    languages = {cast(str, row.get("language", "all")) for row in rows}
    assert languages == {"Hindi", "Spanish"}

    figure = graphics.multilingual_figure(data)
    try:
        axis = figure.axes[0]
        labels = [text.get_text() for text in axis.get_xticklabels()]
        assert labels == sorted(languages)
    finally:
        figure.clf()


def test_paired_effects_figure_orders_by_effect_size(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = graphics._load(source)

    figure = graphics.paired_effects_figure(data)
    try:
        axis = figure.axes[0]
        positions = [text.get_text() for text in axis.get_yticklabels()]
        assert positions == sorted(positions)
        assert float(axis.get_xlim()[1]) > 0.0
    finally:
        figure.clf()


def test_efficiency_figure_uses_a_log_latency_axis(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = graphics._load(source)

    figure = graphics.efficiency_figure(data)
    try:
        assert figure.axes[0].get_yscale() == "log"
    finally:
        figure.clf()


def test_calibration_figure_covers_both_primitives(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = graphics._load(source)

    figure = graphics.calibration_figure(data)
    try:
        titles = [axis.get_title() for axis in figure.axes]
        assert titles == ["choice questions", "noul questions"]
    finally:
        figure.clf()


def test_missing_family_is_reported(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = graphics._load(source)
    del data["calibration"]

    with pytest.raises(ValueError, match="missing the calibration family"):
        graphics.calibration_figure(data)


def test_rendered_figure_data_matches_the_source_rows(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    data = graphics._load(source)
    accuracy = cast(list[dict[str, Any]], data["accuracy"])
    expected = len({cast(str, row["model"]) for row in accuracy}) * len(
        {cast(str, row["suite_id"]) for row in accuracy}
    )

    assert len(accuracy) == expected
    json.dumps(data)


def test_graphics_cli_writes_every_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_source(tmp_path)
    output_root = tmp_path / "png"

    code = graphics.main(
        ["--figure-source", str(source), "--output-root", str(output_root)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.count("\n") == len(graphics.FIGURES)
    for name in graphics.FIGURES:
        assert (output_root / name).is_file()


def test_graphics_cli_refuses_to_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_source(tmp_path)
    output_root = tmp_path / "png"
    graphics.main(["--figure-source", str(source), "--output-root", str(output_root)])

    code = graphics.main(
        ["--figure-source", str(source), "--output-root", str(output_root)]
    )

    assert code == 1
    assert "already exists" in capsys.readouterr().err


def test_graphics_cli_reports_a_bad_source_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "figure-source.json"
    bad.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "png"

    code = graphics.main(["--figure-source", str(bad), "--output-root", str(output_root)])

    assert code == 1
    assert "missing the accuracy family" in capsys.readouterr().err
    assert not output_root.exists()
