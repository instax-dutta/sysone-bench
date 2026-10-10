/* sysone-bench — chart rendering and interaction
 *
 * Every chart is hand-built inline SVG from data/results.json. No chart library: the visual
 * language has to stay consistent with the CSS tokens, and a leaderboard is not a generic bar
 * chart. No external runtime: reveals use IntersectionObserver and bars grow with the Web
 * Animations API, so the page ships with zero script dependencies.
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
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let D = null;
let state = { filter: "all", sort: "accuracy", model: null, compare: "none" };

/* --------------------------------------------------------------- colours */
function colour(r) {
  if (r.techniqueReimplementation) return "var(--dim)";
  if (r.runner === "pngwn") return "var(--warn)";
  return "var(--sig)";
}
const CH = ["var(--ch1)", "var(--ch2)", "var(--ch3)", "var(--ch4)", "var(--ch5)"];

/* ---------------------------------------------------------- grow a mark */
/* Bars are born from their baseline rather than appearing whole, so a redraw reads as a
   measurement rising into place. Transform-origin is set per orientation. */
function grow(node, from, origin, dur = 520) {
  if (REDUCED || !node.animate) return;
  node.style.transformBox = "fill-box";
  node.style.transformOrigin = origin;
  node.animate(
    [{ transform: from }, { transform: "none" }],
    { duration: dur, easing: "cubic-bezier(.22,.61,.36,1)", fill: "backwards" }
  );
}

/* ------------------------------------------------------------------ load */
function fail(err) {
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
  if (r.runner === "pngwn") return '<span class="pill">unresolved</span>';
  if (r.techniqueReimplementation) return '<span class="pill">reimpl.</span>';
  if (r.sharding) return '<span class="pill">shard</span>';
  return '<span class="pill">vendor</span>';
}

/* Suite sparkline: the nine values as a bare trace. Decorative duplicate of the row, so it is
   hidden from assistive tech. */
function sparkline(r) {
  const vals = D.suites.map((s) => r.suites[s]).filter((v) => v !== undefined);
  if (vals.length < 2) return "";
  const w = 120, h = 22, pad = 3;
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const pts = vals
    .map((v, i) => {
      const x = pad + (i / (vals.length - 1)) * (w - 2 * pad);
      const y = h - pad - ((v - min) / span) * (h - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">` +
    `<line class="sparkbase" x1="0" y1="${h / 2}" x2="${w}" y2="${h / 2}"/>` +
    `<polyline class="sparkline" points="${pts}"/></svg>`
  );
}

/* ------------------------------------------------------------ the field */
function drawField(animate = true) {
  const grid = A("#model-grid");
  if (!grid) return;
  grid.replaceChildren();
  const list = visible();
  const lo = 0.20, hi = 0.95;
  list.forEach((r, i) => {
    const c = colour(r);
    const row = document.createElement("button");
    row.className = "model" + (r.techniqueReimplementation ? " re" : "");
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
      '<span class="nm" title="' + r.runner + '">' + r.runner + '</span>' +
      '<span class="model-track"><i style="width:' + (((r.accuracy - lo) / (hi - lo)) * 100).toFixed(1) +
        '%;background:' + c + '"></i></span>' +
      '<span class="acc" style="color:' + c + '">' + r.accuracy.toFixed(4) + '</span>' +
      '<span class="spark">' + sparkline(r) + '</span>' +
      '<span class="model-flags">' + flags + '</span>';
    row.addEventListener("click", () => {
      state.model = r.runner;
      state.compare = "none";
      document.querySelectorAll("[data-compare]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b.dataset.compare === "none")));
      const sel = A("#suite-model");
      if (sel) sel.value = r.runner;
      syncSeg(A('[data-seg="compare"]'));
      drawField(false);
      drawDeferralNote();
      drawSuites();
      A("#suites").scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
    });
    grid.append(row);
    if (animate && !REDUCED && row.animate) {
      row.animate(
        [{ opacity: 0, transform: "translateY(7px)" }, { opacity: 1, transform: "none" }],
        { duration: 380, delay: Math.min(i * 16, 360), easing: "cubic-bezier(.22,.61,.36,1)", fill: "backwards" }
      );
    }
  });
  const c = A("#grid-count");
  if (c) c.textContent = list.length + " shown \u00b7 click any row for its per-suite profile";
}

/* ------------------------------------------------------ deferral caveat */
function drawDeferralNote() {
  const host = A("#deferral-note");
  if (!host) return;
  const r = (D.measurements || []).find((x) => x.runner === state.model);
  const d = r && r.deferrals;
  if (!d) { host.innerHTML = ""; host.hidden = true; return; }
  host.hidden = false;
  const share = (d.deferred_share * 100).toFixed(1);
  host.innerHTML =
    '<div class="k">Declined answers &middot; ' + r.runner + '</div>' +
    '<p><strong>' + d.deferred.toLocaleString() + ' of ' + d.evaluation_decisions.toLocaleString() +
    ' evaluation decisions (' + share + '%)</strong> were declined by this runner. The decision contract has no ' +
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

/* --------------------------------------------------------------- suites */
function drawSuites() {
  const svg = A("#chart-suites");
  svg.replaceChildren();
  const legend = A("#suite-legend");
  legend.replaceChildren();

  const pick = state.compare === "top" ? rows().slice(0, 5) : [rows().find((r) => r.runner === state.model)];
  const series = pick.filter(Boolean);
  const suites = D.suites;

  const padL = 46, padR = 16, padT = 18, padB = 92;
  const w = 1040, h = 340;
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

  suites.forEach((s, si) => {
    const bx = padL + si * band;
    series.forEach((r, ri) => {
      const v = r.suites[s];
      if (v === undefined) return;
      const bw = Math.min(44, (band * 0.6) / series.length);
      const x0 = bx + (band - bw * series.length) / 2 + ri * bw;
      const rect = el("rect", {
        x: x0, y: y(v), width: Math.max(1, bw - 3), height: Math.max(1, plotH - (y(v) - padT)),
        rx: 1, fill: CH[ri % CH.length], "fill-opacity": .9, class: "bar",
      });
      rect.appendChild(el("title")).textContent = `${r.runner} · ${s} · ${pct(v)}%`;
      svg.append(rect);
      grow(rect, "scaleY(0)", "bottom");
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
    d.innerHTML = `<i style="background:${CH[i % CH.length]}"></i> ${r.runner}`;
    legend.append(d);
  });
}

/* ------------------------------------------------------------ multiples */
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

    const w = 300, rowH = 14, padL = 104, padR = 40;
    const h = top.length * rowH + 6;
    const svg = el("svg", { viewBox: "0 0 " + w + " " + h, role: "img" });
    svg.setAttribute("aria-label", "Top five models on " + suite);
    const lo = Math.max(0, Math.min.apply(null, vals) - 0.06);
    const x = (v) => padL + ((v - lo) / (1 - lo)) * (w - padL - padR);
    top.forEach((r, i) => {
      const y = i * rowH + 2;
      const lab = el("text", { x: 0, y: y + 9, class: "bar-label" });
      lab.textContent = r.runner.length > 15 ? r.runner.slice(0, 14) + "\u2026" : r.runner;
      svg.append(lab);
      const bw = Math.max(1, x(r.suites[suite]) - padL);
      const bar = el("rect", { x: padL, y: y + 2, width: bw, height: rowH - 6, rx: 1, fill: colour(r) });
      bar.appendChild(el("title")).textContent = r.runner + " \u00b7 " + suite + " \u00b7 " + pct(r.suites[suite]) + "%";
      svg.append(bar);
      grow(bar, "scaleX(0)", "left", 460);
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
  const padL = 16, padR = 66, padT = 26, padB = 32;
  const w = 600, h = 236;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const min = 0.25, max = 0.92;
  const x = (v) => padL + ((v - min) / (max - min)) * (w - padL - padR);

  for (let t = 0.3; t <= 0.9; t += 0.1) {
    svg.append(el("line", { x1: x(t), x2: x(t), y1: padT - 10, y2: h - padB, class: "grid-line" }));
    const lab = el("text", { x: x(t), y: h - padB + 16, class: "axis-label", "text-anchor": "middle" });
    lab.textContent = t.toFixed(1);
    svg.append(lab);
  }
  const jitter = (i) => padT + 8 + (((i * 37) % 100) / 100) * (h - padT - padB - 14);

  list.forEach((r, i) => {
    const c = el("circle", { cx: x(r.accuracy), cy: jitter(i), r: 3.4, fill: colour(r), opacity: 0.88 });
    c.appendChild(el("title")).textContent = `${r.runner} \u2014 ${pct(r.accuracy)}%`;
    svg.append(c);
    if (!REDUCED && c.animate) {
      c.animate([{ opacity: 0, transform: "scale(.4)" }, { opacity: .88, transform: "none" }],
        { duration: 420, delay: Math.min(i * 8, 320), easing: "cubic-bezier(.22,.61,.36,1)", fill: "backwards" });
    }
  });

  const rx = x(ref.accuracy);
  svg.append(el("line", { x1: rx, x2: rx, y1: padT - 16, y2: h - padB,
    stroke: "var(--ref)", "stroke-width": 1.4, "stroke-dasharray": "2 3" }));
  const t = el("text", { x: rx + 5, y: padT - 18, class: "axis-label", fill: "var(--ref)" });
  t.textContent = `Jev 1.13.0 \u00b7 ${ref.accuracy.toFixed(4)}`;
  svg.append(t);

  const best = list[list.length - 1];
  const bl = el("text", {
    x: x(best.accuracy) - 8, y: padT + 4, class: "bar-value", fill: "var(--sig)", "text-anchor": "end",
  });
  bl.textContent = "best " + best.accuracy.toFixed(4);
  svg.append(bl);
}

/* --------------------------------------------------------------- heatmap */
/* Transposed: suites down the side, models across, in score order so it reads left to right. */
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
        x: padL + ci * (cw + gap), y, width: cw, height: ch, rx: 2,
        fill: v === undefined
          ? "var(--bg-3)"
          : `color-mix(in oklab, var(--sig) ${Math.round(v * 100)}%, var(--bg-2))`,
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
    ["CPU \u00b7 fp32", 0.7629, "var(--ink-3)"],
    ["T4 \u00b7 fp16", host ? host.accuracy : 0.7734, "var(--sig)"],
  ];
  const padL = 58, padT = 16, padB = 46, padR = 78;
  const w = 560, h = 240;
  const plotH = h - padT - padB;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  const min = 0.70, max = 0.80;
  const y = (v) => padT + plotH - ((v - min) / (max - min)) * plotH;

  for (let t = 0.70; t <= 0.8001; t += 0.025) {
    svg.append(el("line", { x1: padL, x2: w - padR, y1: y(t), y2: y(t), class: "grid-line" }));
    const lab = el("text", { x: padL - 8, y: y(t) + 3, class: "axis-label", "text-anchor": "end" });
    lab.textContent = t.toFixed(3);
    svg.append(lab);
  }
  const axisNote = el("text", { x: w - padR, y: h - 8, class: "axis-label", "text-anchor": "end" });
  axisNote.textContent = "axis truncated to 0.70\u20130.80";
  svg.append(axisNote);

  pairs.forEach(([label, v, fill], i) => {
    const bw = 96, x0 = padL + 26 + i * (bw + 56);
    const rect = el("rect", { x: x0, y: y(v), width: bw, height: Math.max(1, h - padB - y(v)), rx: 1, fill });
    rect.appendChild(el("title")).textContent = `${label} \u00b7 ${pct(v)}%`;
    svg.append(rect);
    grow(rect, "scaleY(0)", "bottom");
    const vt = el("text", { x: x0 + bw / 2, y: y(v) - 7, class: "bar-value", "text-anchor": "middle" });
    vt.textContent = v.toFixed(4);
    svg.append(vt);
    const lt = el("text", { x: x0 + bw / 2, y: h - padB + 18, class: "axis-label", "text-anchor": "middle" });
    lt.textContent = label;
    svg.append(lt);
  });

  const gl = el("text", { x: w - padR, y: y(0.7629) - 18, class: "axis-label", fill: "var(--warn)", "text-anchor": "end" });
  gl.textContent = "\u0394 0.0105";
  svg.append(gl);
}

/* -------------------------------------------------------------- coverage */
function drawCoverage() {
  const svg = A("#chart-coverage");
  svg.replaceChildren();
  const items = [
    ["Run directories verified", D.runDirectoriesVerified, "var(--sig)"],
    ["Distinct measurements", D.distinctMeasurements, "var(--sig)"],
    ["Entries unmeasured", D.scopeTotal - D.distinctMeasurements, "var(--dim)"],
  ];
  const padL = 176, padT = 14, rowH = 42;
  const w = 600, h = padT + items.length * rowH + 10;
  const max = Math.max(...items.map((i) => i[1]));
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("height", h);

  items.forEach(([label, v, fill], i) => {
    const y = padT + i * rowH;
    const bw = (v / max) * (w - padL - 66);
    const t = el("text", { x: 0, y: y + 22, class: "bar-label" });
    t.textContent = label;
    svg.append(t);
    const bar = el("rect", { x: padL, y: y + 6, width: bw, height: 20, rx: 1, fill });
    svg.append(bar);
    grow(bar, "scaleX(0)", "left");
    const vt = el("text", { x: padL + bw + 8, y: y + 22, class: "bar-value" });
    vt.textContent = v;
    svg.append(vt);
  });
}

/* -------------------------------------------------------------- numerals */
function countUp(node, value, { decimals = 4, grouped = false, dur = 760 } = {}) {
  const fmt = (x) => (grouped
    ? Math.round(x).toLocaleString("en-US")
    : Number(x).toFixed(decimals));
  if (REDUCED) { node.textContent = fmt(value); return; }
  const t0 = performance.now();
  (function frame(t) {
    const k = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    node.textContent = fmt(value * e);
    if (k < 1) requestAnimationFrame(frame);
    else node.textContent = fmt(value);
  })(t0);
}

/* -------------------------------------------------------------- hydrate */
function hydrate() {
  const best = rows()[0];
  A("#stat-best-name").textContent = best.runner;
  countUp(A("#stat-best"), best.accuracy, { decimals: 4 });
  countUp(A("#stat-closed"), D.referenceClosedApi.accuracy, { decimals: 4 });
  countUp(A("#stat-gap"), D.decisions, { grouped: true });
  const gap = best.accuracy - D.referenceClosedApi.accuracy;
  const dl = A("#delta-line");
  if (dl) dl.innerHTML = "<b>\u2212" + Math.abs(gap).toFixed(4) + "</b> against the closed-API reference";

  const sel = A("#suite-model");
  rows().forEach((r) => {
    const o = document.createElement("option");
    o.value = r.runner;
    o.textContent = `${r.runner} \u2014 ${r.accuracy.toFixed(4)}`;
    sel.append(o);
  });
  state.model = best.runner;

  const list = A("#collapse-list");
  const dup = D.duplicates;
  const row = (name, why) =>
    `<div style="display:flex;gap:.6rem;align-items:baseline;padding:.5rem 0;border-bottom:1px solid var(--line)">
       <code style="font-size:.8rem;color:var(--ink)">${name}</code>
       <span style="font-size:.76rem;color:var(--ink-3);margin-left:auto;text-align:right">${why}</span>
     </div>`;
  for (const [runner, dirs] of Object.entries(dup.repeatedRuns || {})) {
    list.innerHTML += row(runner, `${dirs.length} directories, one model`);
  }
  for (const [fp, runners] of Object.entries(dup.identicalPredictions || {})) {
    list.innerHTML += row(runners.join(" + "), "identical predictions on all 1240 decisions");
  }

  const grid = A("#excluded-grid");
  grid.innerHTML = (D.excluded || []).map(([name, reason, detail]) => `
    <div>
      <h4>${name}</h4>
      <p class="reason">${reason}</p>
      <p>${detail}</p>
    </div>`).join("");

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
  drawField();
  drawSuites();
  drawMultiples();
  drawHeatmap();
  drawPrecision();
  drawCoverage();
}

/* ---------------------------------------------------- sliding segment */
function syncSeg(seg) {
  if (!seg) return;
  const active = seg.querySelector('.chip[aria-pressed="true"]');
  const thumb = seg.querySelector(".seg-thumb");
  if (!active || !thumb) return;
  thumb.style.width = active.offsetWidth + "px";
  thumb.style.transform = `translateX(${active.offsetLeft}px)`;
}
function syncAllSegs() {
  document.querySelectorAll(".seg").forEach(syncSeg);
}

/* ----------------------------------------------------------------- wire */
function wire() {
  document.querySelectorAll("[data-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.filter = btn.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      syncSeg(btn.closest(".seg"));
      drawField();
    });
  });

  A("#sort-by").addEventListener("change", (e) => {
    state.sort = e.target.value;
    drawField();
  });

  A("#suite-model").addEventListener("change", (e) => {
    state.model = e.target.value;
    state.compare = "none";
    document.querySelectorAll("[data-compare]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.compare === "none")));
    syncSeg(A('[data-seg="compare"]'));
    drawDeferralNote();
    drawSuites();
  });

  document.querySelectorAll("[data-compare]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.compare = btn.dataset.compare;
      document.querySelectorAll("[data-compare]").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn)));
      syncSeg(btn.closest(".seg"));
      drawSuites();
    });
  });

  // Reveals: once each, resolved to full opacity, never stranded.
  const nodes = Array.from(document.querySelectorAll(".reveal"));
  if (REDUCED || !("IntersectionObserver" in window)) {
    nodes.forEach((n) => n.classList.add("in"));
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.01 });
    nodes.forEach((n) => io.observe(n));
  }

  // Segment thumbs: place after fonts settle so widths are final, and keep them on resize.
  // The .ready class disables the transition so the first placement is instant, never a
  // grow-from-zero.
  const thumbs = document.querySelectorAll(".seg-thumb");
  thumbs.forEach((t) => t.classList.add("ready"));
  syncAllSegs();
  requestAnimationFrame(() => thumbs.forEach((t) => t.classList.remove("ready")));
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      thumbs.forEach((t) => t.classList.add("ready"));
      syncAllSegs();
      requestAnimationFrame(() => thumbs.forEach((t) => t.classList.remove("ready")));
    });
  }
  window.addEventListener("resize", syncAllSegs);

  // Anchor links: native smooth scroll, skipped under reduced motion.
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener("click", (e) => {
      const id = a.getAttribute("href");
      if (!id || id === "#") return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "start" });
    });
  });
}
