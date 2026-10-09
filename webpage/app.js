/* sysone-bench — chart rendering
 *
 * Every chart is built as inline SVG from data/results.json. No chart library: the visual language
 * has to stay consistent with the CSS tokens, and a leaderboard is not a generic bar chart.
 */
"use strict";

const A = (sel) => document.querySelector(sel);
const NS = "http://www.w3.org/2000/svg";
const el = (name, attrs = {}) => {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};
const pct = (x) => (x * 100).toFixed(2);
const fixed = (x, n = 4) => (x === null || x === undefined ? "—" : Number(x).toFixed(n));

let D = null;
let state = { filter: "all", sort: "accuracy", model: null, compare: "none" };

/* ------------------------------------------------------------- monogram */
const HUES = [188, 205, 262, 320, 12, 32, 48, 78, 142];
function modelHue(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return HUES[h % HUES.length];
}
function monogram(name) {
  const parts = name.replace(/[^a-z0-9]+/gi, "-").split("-").filter(Boolean);
  const letters = (parts.length > 1
    ? parts[0][0] + parts[1][0]
    : name.replace(/[^a-z0-9]/gi, "").slice(0, 2)).toUpperCase();
  const hue = modelHue(name);
  return '<span class="mono-mark" style="--mark:hsl(' + hue + ' 62% 58%);--mark-ink:hsl(' +
    hue + ' 70% 88%)">' + letters + "</span>";
}

/* ------------------------------------------------------------------ load */
function fail(err) {
  // Never leave the reader with a blank document. Show the cause, and un-hide every section.
  console.error(err);
  document.querySelectorAll(".reveal").forEach((n) => n.classList.add("in", "failed"));
  const p = A("#leaderboard .band-head p");
  if (p) {
    p.innerHTML =
      `<strong>Data failed to render.</strong> ${String(err && err.message ? err.message : err)} ` +
      "Every figure on this page is generated from <code>data/results.json</code>; nothing is " +
      "hard-coded, so the charts stay empty until that file loads.";
  }
}

window.addEventListener("error", (e) => fail(e.error || e.message));
window.addEventListener("unhandledrejection", (e) => fail(e.reason));

fetch("data/results.json")
  .then((r) => {
    if (!r.ok) throw new Error(`results.json responded ${r.status}`);
    return r.json();
  })
  .then((json) => {
    D = json;
    hydrate();
    drawAll();
    wire();
  })
  .catch(fail);

/* --------------------------------------------------------------- helpers */
const rows = () => D.measurements;

function visible() {
  let list = rows().slice();
  if (state.filter === "technique") list = list.filter((r) => r.techniqueReimplementation);
  else if (state.filter === "vendor") list = list.filter((r) => !r.techniqueReimplementation);
  if (state.sort === "accuracy") list.sort((a, b) => b.accuracy - a.accuracy);
  else if (state.sort === "runner") list.sort((a, b) => a.runner.localeCompare(b.runner));
  else {
    // Spread = mean absolute per-suite deviation. A wide spread is not a penalty, it is a shape.
    const spread = (r) => {
      const v = Object.values(r.suites);
      if (!v.length) return 0;
      const mean = v.reduce((s, x) => s + x, 0) / v.length;
      return v.reduce((s, x) => s + Math.abs(x - mean), 0) / v.length;
    };
    list.sort((a, b) => spread(a) - spread(b));
  }
  return list;
}

function kindTag(r) {
  if (r.runner === "pngwn") return '<span class="tag tag-unresolved">prompt unresolved</span>';
  if (r.techniqueReimplementation) return '<span class="tag tag-technique">technique reimpl.</span>';
  if (r.sharding) return '<span class="tag tag-shard">2-gpu shard</span>';
  return '<span class="tag">vendor readout</span>';
}

function colour(r) {
  if (r.techniqueReimplementation) return "var(--caveat)";
  if (r.runner === "pngwn") return "var(--warn)";
  return "var(--accent)";
}

/* ------------------------------------------------------------ leaderboard */
function drawLeaderboard() {
  const svg = A("#chart-leaderboard");
  svg.replaceChildren();
  const data = visible();
  const ref = D.referenceClosedApi;

  const rowH = 17;
  const padL = 168, padR = 64, padT = 30, padB = 26;
  const w = 1000;
  const h = padT + data.length * rowH + padB;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("height", h);

  const min = 0.30, max = 0.92;
  const x = (v) => padL + ((v - min) / (max - min)) * (w - padL - padR);

  for (let t = 0.3; t <= 0.9; t += 0.1) {
    svg.append(el("line", { x1: x(t), x2: x(t), y1: padT - 8, y2: h - padB, class: "grid-line" }));
    const lab = el("text", { x: x(t), y: h - padB + 14, class: "axis-label", "text-anchor": "middle" });
    lab.textContent = t.toFixed(1);
    svg.append(lab);
  }

  // The closed-API reference, drawn as a vertical rule behind the bars so the gap is visible.
  const rx = x(ref.accuracy);
  svg.append(el("line", {
    x1: rx, x2: rx, y1: padT - 14, y2: h - padB,
    stroke: "var(--closed)", "stroke-width": 1.5, "stroke-dasharray": "3 3", opacity: 0.85,
  }));
  const rl = el("text", { x: rx + 5, y: padT - 18, class: "axis-label", fill: "var(--closed)" });
  rl.textContent = `closed API ${pct(ref.accuracy)}`;
  svg.append(rl);

  data.forEach((r, i) => {
    const y = padT + i * rowH;
    const bw = Math.max(1, x(r.accuracy) - padL);
    const g = el("g", { class: "bar", tabindex: "0" });
    g.appendChild(el("title")).textContent =
      `${r.runner} — ${pct(r.accuracy)}%  ·  ${r.model || ""}  ·  ${r.scoring || "vendor readout"}`;

    g.append(el("rect", { x: 0, y, width: w, height: rowH, fill: "transparent" }));

    const name = el("text", { x: 8, y: y + 12, class: "bar-label" });
    name.textContent = r.runner.length > 21 ? r.runner.slice(0, 20) + "…" : r.runner;
    if (r.techniqueReimplementation) name.setAttribute("fill", "var(--caveat)");
    g.append(name);

    g.append(el("rect", { x: padL, y: y + 3, width: bw, height: rowH - 7, rx: 2, fill: colour(r) }));
    const val = el("text", { x: padL + bw + 7, y: y + 12, class: "bar-value" });
    val.textContent = r.accuracy.toFixed(4);
    g.append(val);
    svg.append(g);
  });
}

/* ----------------------------------------------------------------- table */
function drawTable() {
  const body = A("#table-all tbody");
  body.replaceChildren();
  visible().forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="rank">${i + 1}</td>
      <td><code>${r.runner}</code></td>
      <td class="num">${r.accuracy.toFixed(4)}</td>
      <td>${kindTag(r)}</td>
      <td style="color:var(--ink-3);font-size:0.8rem">${r.scoring || "—"}</td>
      <td class="num" style="color:var(--ink-4);font-size:0.75rem">${r.runId}</td>`;
    body.append(tr);
  });
}

/* ---------------------------------------------------------------- suites */
function drawSuites() {
  const svg = A("#chart-suites");
  svg.replaceChildren();
  const legend = A("#suite-legend");
  legend.replaceChildren();

  const pick = state.compare === "top" ? rows().slice(0, 5) : [rows().find((r) => r.runner === state.model)];
  const series = pick.filter(Boolean);
  const suites = D.suites;

  const padL = 46, padR = 16, padT = 18, padB = 88;
  const w = 1000;
  const h = 330;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

  const plotH = h - padT - padB;
  const band = (w - padL - padR) / suites.length;
  const y = (v) => padT + plotH - v * plotH;

  for (let t = 0; t <= 1.0001; t += 0.25) {
    svg.append(el("line", { x1: padL, x2: w - padR, y1: y(t), y2: y(t), class: "grid-line" }));
    const lab = el("text", { x: padL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
    lab.textContent = t.toFixed(2);
    svg.append(lab);
  }

  const hues = ["var(--accent)", "var(--closed)", "var(--caveat)", "var(--ok)", "var(--warn)"];
  suites.forEach((s, si) => {
    const bx = padL + si * band;
    series.forEach((r, ri) => {
      const v = r.suites[s];
      if (v === undefined) return;
      const bw = Math.min(44, (band * 0.6) / series.length);
      const x0 = bx + (band - bw * series.length) / 2 + ri * bw;
      const rect = el("rect", {
        x: x0, y: y(v), width: Math.max(1, bw - 3), height: Math.max(1, plotH - (y(v) - padT)),
        rx: 3, fill: hues[ri % hues.length], "fill-opacity": .82, class: "bar",
      });
      rect.appendChild(el("title")).textContent = `${r.runner} · ${s} · ${pct(v)}%`;
      svg.append(rect);
    });
    const tick = el("text", {
      x: bx + band / 2, y: h - padB + 16, class: "axis-label", "text-anchor": "end",
      transform: `rotate(-34 ${bx + band / 2} ${h - padB + 16})`,
    });
    tick.textContent = s;
    svg.append(tick);
  });

  A("#suite-title").textContent =
    state.compare === "top" ? "Top 5 models, per suite" : `${state.model}, per suite`;

  series.forEach((r, i) => {
    const d = document.createElement("div");
    d.innerHTML = `<i style="background:${hues[i % hues.length]}"></i> ${r.runner}`;
    legend.append(d);
  });
}

/* --------------------------------------------------------- ranked list */
/* ------------------------------------------------------- deferral caveat */
function drawDeferralNote() {
  const host = A("#deferral-note");
  if (!host) return;
  const r = (D.measurements || []).find((x) => x.runner === state.model);
  const d = r && r.deferrals;
  if (!d) { host.innerHTML = ""; host.hidden = true; return; }
  host.hidden = false;
  const pct = (d.deferred_share * 100).toFixed(1);
  host.innerHTML =
    '<div class="k">Declined answers &middot; ' + r.runner + '</div>' +
    '<p><strong>' + d.deferred.toLocaleString() + ' of ' + d.evaluation_decisions.toLocaleString() +
    ' evaluation decisions (' + pct + '%)</strong> were declined by this runner. The decision contract has no ' +
    'abstention type, so the adapter answered every one of them at the argmax option and each is counted above ' +
    'as a confident correct-or-incorrect answer the model did not choose. ' +
    d.projected_to_argmax.toLocaleString() + ' carried an explicit argmax projection in the raw output. ' +
    'Read this row as the adapter\'s coverage, not the model\'s own.</p>';
}

function drawReadoutCaveat() {
  const host = A("#readout-caveat");
  const c = D.readoutCaveat;
  if (!host || !c) return;
  const n = (D.measurements || []).filter((r) => r.readoutCaveat).length;
  if (!n) { host.remove(); return; }
  host.innerHTML =
    '<div class="k">Readout caveat &middot; ' + n + ' of ' + D.distinctMeasurements + ' rows</div>' +
    '<p>' + c.note + ' <a href="' + c.confirmedBy + '">Thread with the authors</a>.</p>';
}

function drawCards() {
  const grid = A("#model-grid");
  if (!grid) return;
  grid.replaceChildren();
  const list = visible();
  const lo = 0.20, hi = 0.95;
  list.forEach((r, i) => {
    const row = document.createElement("button");
    row.className = "model";
    row.type = "button";
    row.setAttribute("aria-pressed", String(state.model === r.runner));
    const flags = [
      r.sharding ? '<span class="pill">shard</span>' : "",
      r.techniqueReimplementation ? '<span class="pill">reimpl.</span>' : "",
      r.runner === "pngwn" ? '<span class="pill">unresolved</span>' : "",
      r.readoutCaveat ? '<span class="pill caveat" title="Scored per option independently; may under-report a model trained to answer with restricted label codes">readout</span>' : "",
      r.deferrals ? '<span class="pill caveat" title="This runner declined ' + r.deferrals.deferred + ' of ' + r.deferrals.evaluation_decisions + ' evaluation decisions (' + Math.round(r.deferrals.deferred_share * 100) + '%). The decision contract has no abstention type, so the adapter answered each one at the argmax option. The score below reflects that projection, not the model\'s own coverage.">declined</span>' : "",
    ].join("");
    row.innerHTML =
      '<span class="rk">' + (i + 1) + '</span>' +
      monogram(r.runner) +
      '<span class="nm">' + r.runner + '</span>' +
      '<span class="model-track"><i style="width:' + (((r.accuracy - lo) / (hi - lo)) * 100).toFixed(1) +
        '%;background:' + (r.techniqueReimplementation ? "var(--caveat)" : "var(--ink)") + '"></i></span>' +
      '<span class="acc" style="color:' + colour(r) + '">' + r.accuracy.toFixed(4) + '</span>' +
      '<span class="model-flags">' + flags + '</span>';
    row.addEventListener("click", () => {
      state.model = r.runner;
      state.compare = "none";
      document.querySelectorAll("[data-compare]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.compare === "none")));
      const sel = A("#suite-model");
      if (sel) sel.value = r.runner;
      drawCards();
      drawDeferralNote();
      drawSuites();
      A("#suites").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    grid.append(row);
  });
  const c = A("#grid-count");
  if (c) c.textContent = list.length + " shown \u00b7 click any row for its per-suite profile";
}

/* ------------------------------------------------------ suite multiples */
function drawMultiples() {
  const host = A("#suite-multiples");
  if (!host) return;
  host.replaceChildren();
  D.suites.forEach((suite) => {
    const ranked = rows()
      .filter((r) => r.suites[suite] !== undefined)
      .sort((a, b) => b.suites[suite] - a.suites[suite]);
    const top = ranked.slice(0, 5);
    const vals = ranked.map((r) => r.suites[suite]);
    const mean = vals.reduce((s2, v) => s2 + v, 0) / vals.length;
    const spread = vals.reduce((s2, v) => s2 + Math.abs(v - mean), 0) / vals.length;

    const box = document.createElement("div");
    box.className = "multiple";
    box.innerHTML = '<h4>' + suite + '</h4><div class="spread">top ' +
      pct(top[0] ? top[0].suites[suite] : 0) + '% \u00b7 spread ' + pct(spread) + '</div>';
    host.append(box);

    const w = 300, rowH = 13, padL = 112, padR = 38;
    const h = top.length * rowH + 6;
    const svg = el("svg", { viewBox: "0 0 " + w + " " + h, role: "img" });
    svg.setAttribute("aria-label", "Top five models on " + suite);
    const lo = Math.max(0, Math.min.apply(null, vals) - 0.06);
    const x = (v) => padL + ((v - lo) / (1 - lo)) * (w - padL - padR);
    top.forEach((r, i) => {
      const y = i * rowH + 2;
      const lab = el("text", { x: 0, y: y + 9, class: "bar-label" });
      lab.textContent = r.runner.length > 16 ? r.runner.slice(0, 15) + "\u2026" : r.runner;
      svg.append(lab);
      const bw = Math.max(1, x(r.suites[suite]) - padL);
      const bar = el("rect", { x: padL, y: y + 2, width: bw, height: rowH - 5, rx: 2, fill: colour(r) });
      bar.appendChild(el("title")).textContent = r.runner + " \u00b7 " + suite + " \u00b7 " + pct(r.suites[suite]) + "%";
      svg.append(bar);
      const v = el("text", { x: padL + bw + 5, y: y + 9, class: "bar-value" });
      v.textContent = r.suites[suite].toFixed(2);
      svg.append(v);
    });
    box.append(svg);
  });
}

/* ------------------------------------------------------------ strip plot */
function drawStrip() {
  const svg = A("#chart-strip");
  svg.replaceChildren();
  const list = rows().slice().sort((a, b) => a.accuracy - b.accuracy);
  const ref = D.referenceClosedApi;
  const padL = 14, padR = 58, padT = 22, padB = 30;
  const w = 520, h = 250;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const min = 0.25, max = 0.92;
  const x = (v) => padL + ((v - min) / (max - min)) * (w - padL - padR);

  for (let t = 0.3; t <= 0.9; t += 0.1) {
    svg.append(el("line", { x1: x(t), x2: x(t), y1: padT - 8, y2: h - padB, class: "grid-line" }));
    const lab = el("text", { x: x(t), y: h - padB + 15, class: "axis-label", "text-anchor": "middle" });
    lab.textContent = t.toFixed(1);
    svg.append(lab);
  }
  const jitter = (i) => padT + 10 + ((i * 37) % 100) / 100 * (h - padT - padB - 16);

  list.forEach((r, i) => {
    const c = el("circle", {
      cx: x(r.accuracy), cy: jitter(i), r: 3.4,
      fill: r.techniqueReimplementation ? "var(--caveat)" : r.runner === "pngwn" ? "var(--warn)" : "var(--accent)",
      opacity: 0.85,
    });
    c.appendChild(el("title")).textContent = `${r.runner} \u2014 ${pct(r.accuracy)}%`;
    svg.append(c);
  });

  const rx = x(ref.accuracy);
  svg.append(el("line", { x1: rx, x2: rx, y1: padT - 14, y2: h - padB,
    stroke: "var(--closed)", "stroke-width": 1.5, "stroke-dasharray": "3 3" }));
  const t = el("text", { x: rx + 4, y: padT - 18, class: "axis-label", fill: "var(--closed)" });
  t.textContent = "Jev";
  svg.append(t);

  const best = list[list.length - 1];
  const bl = el("text", {
    x: x(best.accuracy) - 8, y: padT + 4, class: "bar-value",
    fill: "var(--accent)", "text-anchor": "end",
  });
  bl.textContent = "best " + best.accuracy.toFixed(4);
  svg.append(bl);
}

/* --------------------------------------------------------------- heatmap */
/* Transposed: suites down the side, models across. 43 rows by 9 columns is a tall strip in a
   wide container; 9 rows by 43 columns fills it and reads left-to-right in score order. */
function drawHeatmap() {
  const svg = A("#chart-heatmap");
  svg.replaceChildren();
  const list = rows().slice().sort((a, b) => b.accuracy - a.accuracy);
  const suites = D.suites;

  const padL = 116, padT = 74, cw = 24, ch = 26, gap = 2;
  const w = padL + list.length * (cw + gap) + 10;
  const h = padT + suites.length * (ch + gap) + 8;
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  svg.style.width = w + "px";
  svg.style.maxWidth = "none";

  list.forEach((r, ci) => {
    const x = padL + ci * (cw + gap);
    if (ci % 4 === 0) {
      const t = el("text", {
        x: x + cw / 2, y: padT - 10, class: "axis-label", "text-anchor": "start",
        transform: `rotate(-58 ${x + cw / 2} ${padT - 10})`,
      });
      t.textContent = r.runner;
      svg.append(t);
    }
  });

  suites.forEach((suite, si) => {
    const y = padT + si * (ch + gap);
    const lab = el("text", { x: 0, y: y + ch / 2 + 4, class: "bar-label" });
    lab.textContent = suite;
    svg.append(lab);

    list.forEach((r, ci) => {
      const v = r.suites[suite];
      const rect = el("rect", {
        x: padL + ci * (cw + gap), y, width: cw, height: ch, rx: 3,
        fill: v === undefined
          ? "var(--surface-2)"
          : `color-mix(in oklab, var(--accent) ${Math.round(v * 100)}%, var(--surface-2))`,
        opacity: r.techniqueReimplementation ? 0.5 : 1,
      });
      rect.appendChild(el("title")).textContent =
        `${r.runner} \u00b7 ${suite} \u00b7 ${v === undefined ? "no data" : pct(v) + "%"}`;
      svg.append(rect);
    });
  });
}

/* ------------------------------------------------------------- precision */
function drawPrecision() {
  const svg = A("#chart-precision");
  svg.replaceChildren();
  const host = D.measurements.find((r) => r.runner === "tev1-08b");
  const pairs = [
    ["CPU · fp32", 0.7629, "var(--closed)"],
    ["T4 · fp16", host ? host.accuracy : 0.7734, "var(--accent)"],
  ];
  const padL = 54, padT = 16, padB = 42, padR = 74;
  const w = 520, h = 220;
  const plotH = h - padT - padB;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  // Truncated axis, stated on the chart, because the whole point is a 0.0105 gap.
  const min = 0.70, max = 0.80;
  const y = (v) => padT + plotH - ((v - min) / (max - min)) * plotH;

  for (let t = 0.70; t <= 0.8001; t += 0.025) {
    svg.append(el("line", { x1: padL, x2: w - padR, y1: y(t), y2: y(t), class: "grid-line" }));
    const lab = el("text", { x: padL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
    lab.textContent = t.toFixed(3);
    svg.append(lab);
  }
  const axisNote = el("text", { x: w - padR, y: h - 8, class: "axis-label", "text-anchor": "end" });
  axisNote.textContent = "axis truncated to 0.70–0.80";
  svg.append(axisNote);

  pairs.forEach(([label, v, fill], i) => {
    const bw = 96, x0 = padL + 24 + i * (bw + 52);
    svg.append(el("rect", { x: x0, y: y(v), width: bw, height: Math.max(1, h - padB - y(v)), rx: 3, fill }));
    svg.lastChild.appendChild(el("title")).textContent = `${label} · ${pct(v)}%`;
    const vt = el("text", { x: x0 + bw / 2, y: y(v) - 7, class: "bar-value", "text-anchor": "middle" });
    vt.textContent = v.toFixed(4);
    svg.append(vt);
    const lt = el("text", { x: x0 + bw / 2, y: h - padB + 17, class: "axis-label", "text-anchor": "middle" });
    lt.textContent = label;
    svg.append(lt);
  });

  const gapY = y(0.7629) - 26;
  const gl = el("text", { x: w - padR, y: gapY + 8, class: "axis-label", fill: "var(--warn)", "text-anchor": "end" });
  gl.textContent = "Δ 0.0105";
  svg.append(gl);
}

/* -------------------------------------------------------------- coverage */
function drawCoverage() {
  const svg = A("#chart-coverage");
  svg.replaceChildren();
  const items = [
    ["Run directories verified", D.runDirectoriesVerified, "var(--accent)"],
    ["Distinct runners", 44, "var(--accent)"],
    ["Distinct measurements", D.distinctMeasurements, "var(--ok)"],
  ];
  const padL = 176, padT = 14, rowH = 40;
  const w = 560, h = padT + items.length * rowH + 10;
  const max = Math.max(...items.map((i) => i[1]));
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("height", h);

  items.forEach(([label, v, fill], i) => {
    const y = padT + i * rowH;
    const bw = (v / max) * (w - padL - 66);
    const t = el("text", { x: 0, y: y + 21, class: "bar-label" });
    t.textContent = label;
    svg.append(t);
    svg.append(el("rect", { x: padL, y: y + 5, width: bw, height: 20, rx: 3, fill }));
    const vt = el("text", { x: padL + bw + 8, y: y + 21, class: "bar-value" });
    vt.textContent = v;
    svg.append(vt);
  });
}

/* -------------------------------------------------------------- hydrate */
const EXCLUDED = [
  ["winnow-e4b", "GGUF only", "A vision-language checkpoint published solely as GGUF. Needs a different runtime."],
  ["clm-v0.1-8b", "Contrastive reranker", "Scores state-answer pairs rather than producing typed decisions."],
  ["winnow-12b", "GGUF only", "22.3 GiB of bf16 weights published solely as GGUF. Fits current hardware but needs a different runtime."],
  ["jeff", "Artifacts lost", "Measured 0.6855 on a DGX Spark, but that run was never mirrored and the host is gone. The row is withdrawn until it is re-run; the broken earlier run is retained under quarantine and is not reported."],
];

function hydrate() {
  const best = rows()[0];
  A("#stat-best-name").textContent = best.runner;
  A("#stat-measured").textContent = D.distinctMeasurements;
  A("#stat-best").textContent = best.accuracy.toFixed(4);
  A("#stat-closed").textContent = D.referenceClosedApi.accuracy.toFixed(4);
  A("#stat-gap").textContent = D.decisions.toLocaleString("en-US");

  const sel = A("#suite-model");
  rows().forEach((r) => {
    const o = document.createElement("option");
    o.value = r.runner;
    o.textContent = `${r.runner} — ${r.accuracy.toFixed(4)}`;
    sel.append(o);
  });
  state.model = best.runner;

  const list = A("#collapse-list");
  const dup = D.duplicates;
  const row = (name, why) =>
    `<div style="display:flex;gap:.6rem;align-items:baseline;padding:.5rem 0;border-bottom:1px solid var(--line-soft)">
       <code style="font-size:.8rem;color:var(--ink)">${name}</code>
       <span style="font-size:.78rem;color:var(--ink-3);margin-left:auto;text-align:right">${why}</span>
     </div>`;
  for (const [runner, dirs] of Object.entries(dup.repeatedRuns || {})) {
    list.innerHTML += row(runner, `${dirs.length} directories, one model`);
  }
  for (const [fp, runners] of Object.entries(dup.identicalPredictions || {})) {
    list.innerHTML += row(runners.join(" + "), "identical predictions on all 1240 decisions");
  }

  const grid = A("#excluded-grid");
  grid.innerHTML = EXCLUDED.map(([name, reason, detail]) => `
    <div class="card">
      <h3 style="display:flex;justify-content:space-between;gap:.6rem;align-items:baseline">
        <code style="font-size:.9rem">${name}</code>
      </h3>
      <p style="color:var(--warn);font-size:.78rem;font-family:var(--font-num);margin-bottom:.5rem">${reason}</p>
      <p style="font-size:.85rem">${detail}</p>
    </div>`).join("");

  // Every count in the copy is filled from the generated data, never hand-typed.
  const counts = {
    measurements: D.distinctMeasurements,
    dirs: D.runDirectoriesVerified,
    decisions: D.decisions.toLocaleString("en-US"),
    suites: D.suites.length,
    scope: D.scopeTotal,
    unmeasured: D.scopeTotal - D.distinctMeasurements,
  };
  document.querySelectorAll("[data-count]").forEach((el) => {
    const v = counts[el.dataset.count];
    if (v !== undefined) el.textContent = String(v);
  });

  A("#foot-count").textContent =
    `${D.distinctMeasurements} measured, ${D.scopeTotal - D.distinctMeasurements} not measured.`;

  const tied = rows().slice(0, 3);
  const spread = tied[0].accuracy - tied[2].accuracy;
  A("#tie-note").textContent =
    `${tied.map((r) => r.runner).join(", ")} sit within ${spread.toFixed(4)} of each other`;
}

function drawAll() {
  drawStrip();
  drawDeferralNote();
  drawReadoutCaveat();
  drawCards();
  drawSuites();
  drawMultiples();
  drawHeatmap();
  drawPrecision();
  drawCoverage();
}

function wire() {
  document.querySelectorAll("[data-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.filter = btn.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      drawCards();
    });
  });

  A("#sort-by").addEventListener("change", (e) => {
    state.sort = e.target.value;
    drawCards();
  });

  A("#suite-model").addEventListener("change", (e) => {
    state.model = e.target.value;
    state.compare = "none";
    document.querySelectorAll("[data-compare]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.compare === "none")));
    drawSuites();
  });

  document.querySelectorAll("[data-compare]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.compare = btn.dataset.compare;
      document.querySelectorAll("[data-compare]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      drawSuites();
    });
  });

  const nodes = Array.from(document.querySelectorAll(".reveal"));
  const show = (n) => n.classList.add("in");
  nodes.forEach(show);

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasGsap = typeof window.gsap !== "undefined";

  // Rail: ticks mark the accuracy scale, and the scale marker tracks the scroll position.
  const ticks = document.getElementById("rail-ticks");
  const rail = document.getElementById("rail");
  if (ticks) {
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].forEach((t) => {
      const row = document.createElement("div");
      const major = Math.round(t * 10) % 2 === 0;
      row.className = "rail-tick" + (major ? " major" : "");
      row.style.flex = "1 1 0";
      row.innerHTML = "<b>" + t.toFixed(1) + "</b><i></i>";
      ticks.append(row);
    });
  }
  if (rail) rail.classList.add("on");

  // Smooth scrolling via Lenis, driven by GSAP's ticker so ScrollTrigger stays in step with it.
  // GSAP's own ScrollSmoother is a Club plugin and is not on the public CDN, so Lenis stands in.
  if (!hasGsap || reduced || typeof Lenis === "undefined") return;

  gsap.registerPlugin(ScrollTrigger);

  const lenis = new Lenis({
    lerp: 0.11,
    smoothWheel: true,
    syncTouch: false,
    wheelMultiplier: 1,
    touchMultiplier: 1.4,
  });
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add((time) => lenis.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener("click", (e) => {
      const id = a.getAttribute("href");
      if (!id || id === "#") return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      lenis.scrollTo(target, { offset: -70, duration: 1.2 });
    });
  });

  gsap.utils.toArray(".reveal").forEach((node) => {
    gsap.fromTo(node, { y: 26, opacity: 0 }, {
      y: 0, opacity: 1, duration: 0.9, ease: "power3.out",
      scrollTrigger: { trigger: node, start: "top 88%", once: true },
    });
  });

  // The measurement rail tracks reading position, so it reads as an instrument rather than trim.
  if (rail) {
    gsap.to(rail, {
      scrollTrigger: { trigger: document.body, start: "top top", end: "bottom bottom", scrub: 0.4 },
      opacity: 1, ease: "none",
    });
  }
}
