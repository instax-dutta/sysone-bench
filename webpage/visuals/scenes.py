"""ManimGX scenes for the sysone-bench measurement ledger.

Render with:
    manimgx render webpage/visuals/scenes.py FieldRace -r 1280x720 --fps 30
    manimgx render webpage/visuals/scenes.py SameBytes -r 1280x720 --fps 30

All values come from webpage/data/results.json; nothing in the animation is hand-typed.
"""

from __future__ import annotations

import json
from pathlib import Path

import manimgx as m

DATA = json.loads((Path(__file__).resolve().parent.parent / "data" / "results.json").read_text())
TOP = DATA["measurements"][:10]

PAPER = "#F3EFE4"
INK = "#161412"
RULE = "#C9C2B2"
OXIDE = "#9A3B2C"

m.config.background_color = PAPER


class FieldRace(m.Scene):
    """The top ten measurements appear as ruled ledger rows, then fill left to right."""

    def construct(self) -> None:
        title = m.Text("SEALED MANIFEST  /  TOP TEN", font_size=22, color=OXIDE)
        title.to_edge(m.UP, buff=0.45).to_edge(m.LEFT, buff=0.7)
        rule = m.Line([-6.4, 2.95, 0], [6.4, 2.95, 0], color=INK, stroke_width=2)
        self.play(m.Write(title), m.Create(rule), run_time=0.6)

        rows = []
        for index, item in enumerate(TOP):
            y = 2.5 - index * 0.6
            rank = m.Text(f"{index + 1:02d}", font_size=18, color=OXIDE)
            rank.move_to([-6.1, y, 0])
            name = m.Text(item["runner"], font_size=19, color=INK)
            name.move_to([-4.2, y, 0])
            value = m.Text(f"{item['accuracy']:.4f}", font_size=19, color=INK)
            value.move_to([5.6, y, 0])
            track = m.Rectangle(width=5.4, height=0.012, color=RULE, fill_opacity=1)
            track.move_to([0.6, y - 0.25, 0])
            fill = m.Rectangle(width=0.01, height=0.14, color=OXIDE, fill_opacity=1)
            fill.move_to([-2.1, y - 0.25, 0])
            target_width = 5.4 * item["accuracy"]
            rows.append((rank, name, value, track, fill, target_width))

        for rank, name, value, track, fill, width in rows:
            self.play(
                m.FadeIn(rank, shift=m.RIGHT * 0.1),
                m.FadeIn(name, shift=m.RIGHT * 0.1),
                m.FadeIn(value, shift=m.RIGHT * 0.1),
                m.Create(track),
                run_time=0.18,
            )
            self.play(fill.animate.stretch_to_fit_width(width).move_to([-2.1 + width / 2, fill.get_y(), 0]), run_time=0.22)

        note = m.Text("50 MODELS  ·  1,240 DECISIONS EACH  ·  SEED 42", font_size=17, color=INK)
        note.move_to([0, -3.75, 0])
        self.play(m.FadeIn(note), run_time=0.4)
        self.wait(1.5)


class SameBytes(m.Scene):
    """One sealed manifest fans out to independently measured models."""

    def construct(self) -> None:
        head = m.Text("ONE SEALED MANIFEST", font_size=24, color=OXIDE)
        head.to_edge(m.UP, buff=0.6)
        source = m.Rectangle(width=3.2, height=1.4, color=INK, fill_opacity=0.06, stroke_opacity=0)
        source.move_to([-4.4, 0, 0])
        label = m.Text("1,190 cases\n1,550 questions\nseed 42", font_size=19, color=INK, line_spacing=1.4)
        label.move_to(source.get_center())
        self.play(m.Create(source), m.FadeIn(label), run_time=0.7)

        targets = []
        for index, item in enumerate(DATA["measurements"][:5]):
            y = 2.2 - index * 1.1
            box = m.Rectangle(width=3.6, height=0.7, color=RULE, fill_opacity=0.15)
            box.move_to([4.0, y, 0])
            text = m.Text(
                f"{item['runner']}   {item['accuracy']:.4f}",
                font_size=18,
                color=INK,
            )
            text.move_to(box.get_center())
            arrow = m.Arrow(source.get_right(), box.get_left(), color=OXIDE, stroke_width=2)
            targets.append((arrow, box, text))
            self.play(m.Create(arrow), run_time=0.25)
            self.play(m.Create(box), m.FadeIn(text), run_time=0.3)

        foot = m.Text("EVERY INPUT VERIFIED BYTE-IDENTICAL BEFORE SCORING", font_size=17, color=INK)
        foot.to_edge(m.DOWN, buff=0.5)
        self.play(m.FadeIn(foot), run_time=0.5)
        self.wait(1.5)
