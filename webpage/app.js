/* sysone-bench — every figure comes from data/results.json. No build step. */
(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const fmt = (a) => (a == null ? "—" : Number(a).toFixed(4));
  const pct = (a) => (a == null ? 0 : Math.max(0, Math.min(1, Number(a))));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";

  function svg(w, h) {
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `0 0 ${w} ${h}`);
    s.setAttribute("width", w);
    s.setAttribute("height", h);
    s.setAttribute("role", "img");
    return s;
  }
  function mount(host, w, h) {
    host.replaceChildren();
    host.setAttribute("viewBox", `0 0 ${w} ${h}`);
    host.setAttribute("width", w);
    host.setAttribute("height", h);
    return host;
  }
  function text(parent, x, y, value, size = 11, anchor = "start", fill = "#161412") {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.setAttribute("font-size", size); t.setAttribute("text-anchor", anchor);
    t.setAttribute("fill", fill);
    t.textContent = value;
    parent.appendChild(t);
    return t;
  }
  function rect(parent, x, y, w, h, fill, opacity = 1) {
    const r = document.createElementNS(NS, "rect");
    r.setAttribute("x", x); r.setAttribute("y", y);
    r.setAttribute("width", Math.max(0, w)); r.setAttribute("height", Math.max(0, h));
    if (fill) r.setAttribute("fill", fill);
    r.setAttribute("fill-opacity", opacity);
    parent.appendChild(r);
    return r;
  }
  function line(parent, x1, y1, x2, y2, color = "#c9c2b2") {
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1);
    l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", color); l.setAttribute("stroke-width", 1);
    parent.appendChild(l);
    return l;
  }
  function fillCounts(d) {
    const counts = {
      measurements: d.distinctMeasurements,
      dirs: d.runDirectoriesVerified,
      decisions: d.decisions,
      suites: d.suites.length,
      scope: d.scopeTotal,
      unmeasured: d.scopeTotal - d.distinctMeasurements,
      cases: d.cases,
    };
    $$("[data-count]").forEach((node) => {
      const v = counts[node.dataset.count];
      if (v != null) node.textContent = v.toLocaleString("en-US");
    });
    $("#edition-version").textContent = d.datasetVersion;
    $("#as-of").textContent = d.generatedAt || "";
    $("#hash-manifest").textContent = d.manifestSha256;
    $("#hash-logical").textContent = d.logicalDigest;
    $("#hash-eval").textContent = d.evaluationCases.toLocaleString("en-US");
    $("#hash-seed").textContent = d.seed;
  }

  function countUp(node, target) {
    // The number is a measurement: never show an intermediate value that could be read as data.
    node.textContent = target.toFixed(4);
  }

  function fillHero(d) {
    const best = d.measurements[0];
    const ref = d.referenceClosedApi;
    countUp($("#best-score"), best.accuracy);
    $("#best-name").textContent = best.runner;
    $("#gap-value").textContent = (ref.accuracy - best.accuracy).toFixed(4);
    $("#closed-name").textContent = `${ref.model} (${fmt(ref.accuracy)})`;
    const proof = [
      ["Best open", fmt(best.accuracy), best.runner],
      ["Closed reference", fmt(ref.accuracy), ref.model],
      ["Decisions each", d.decisions.toLocaleString("en-US"), "one evaluation split"],
      ["Suites", d.suites.length, "one metric each"],
      ["Scope", `${d.distinctMeasurements}/${d.scopeTotal}`, "all run here"],
    ];
    $("#proof").innerHTML = proof.map(([k, v, s]) =>
      `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd><small>${esc(s)}</small></div>`).join("");
    $("#proof").setAttribute("role", "list");
  }

  function sparkline(row) {
    const s = svg(116, 22);
    s.classList.add("spark");
    data.suites.forEach((suite, i) => {
      const v = pct(row.suites[suite]);
      const bar = rect(s, i * 13, 22 - v * 22, 9, v * 22, "#9a3b2c", v ? 0.8 : 0.12);
      bar.classList.add("bar");
    });
    return s;
  }

  let data;
  function renderLedger() {
    const filter = $(".chip[data-filter][aria-pressed='true']")?.dataset.filter || "all";
    const sort = $("#sort-by").value;
    let rows = data.measurements.filter((m) =>
      filter === "all" ? true : filter === "technique" ? m.techniqueReimplementation || m.readoutCaveat : !m.techniqueReimplementation && !m.readoutCaveat);
    if (sort === "runner") rows = [...rows].sort((a, b) => a.runner.localeCompare(b.runner));
    else if (sort === "spread") rows = [...rows].sort((a, b) => {
      const av = data.suites.map((s) => pct(a.suites[s])), bv = data.suites.map((s) => pct(b.suites[s]));
      return (Math.max(...bv) - Math.min(...bv)) - (Math.max(...av) - Math.min(...av));
    });
    const host = $("#ledger");
    host.replaceChildren();
    const head = document.createElement("div");
    head.className = "ledger-row head";
    head.innerHTML = "<span>#</span><span>Model</span><span class='score'>Accuracy</span><span>Nine suites</span><span class='flags'>Provenance</span>";
    host.appendChild(head);
    rows.forEach((m, i) => {
      const row = document.createElement("div");
      row.className = "ledger-row";
      row.dataset.i = String(i);
      const flags = [m.techniqueReimplementation ? "technique" : null, m.readoutCaveat ? "readout caveat" : null, m.deferrals ? "deferred" : null, m.sharding ? "2-GPU shard" : null].filter(Boolean);
      row.innerHTML = `<span class="rank">${String(i + 1).padStart(2, "0")}</span><span class="who">${esc(m.runner)}</span><span class="score">${fmt(m.accuracy)}</span><span></span><span class="flags">${esc(flags.join(" · "))}</span>`;
      row.children[3].appendChild(sparkline(m));
      row.title = `${m.runner} · ${m.runId} · ${m.device || "?"} / ${m.dtype || "?"} · ${m.scoring || "vendor readout"}`;
      host.appendChild(row);
    });
    $("#ledger-count").textContent = `${rows.length} of ${data.measurements.length} measurements shown`;
  }

  function suiteChart() {
    const host = $("#suite-chart");
    const compareTop = $(".chip[data-compare][aria-pressed='true']")?.dataset.compare === "top";
    const chosen = $("#model-select").value;
    const series = compareTop ? data.measurements.slice(0, 5) : [data.measurements.find((m) => m.runner === chosen) || data.measurements[0]];
    const W = 980, H = 300, left = 40, top = 20, bottom = 54, plotH = H - top - bottom;
    const s = mount(host, W, H);
    s.setAttribute("aria-label", compareTop ? "Top five models by suite" : `Per-suite accuracy for ${chosen}`);
    for (let i = 0; i <= 10; i += 2) {
      const y = top + plotH * (1 - i / 10);
      line(s, left, y, W - 12, y, i === 0 ? "#161412" : "#ded7c8");
      text(s, left - 8, y + 4, (i / 10).toFixed(1), 10, "end", "#5d574d");
    }
    const colors = ["#9a3b2c", "#161412", "#b07a55", "#6d6458", "#c9c2b2"];
    const group = (W - left - 20) / data.suites.length;
    data.suites.forEach((suite, si) => {
      const usable = group - 16;
      const bw = usable / series.length;
      series.forEach((m, mi) => {
        const v = pct(m.suites[suite]);
        const h = plotH * v;
        const x = left + si * group + 8 + mi * bw;
        const bar = rect(s, x, top + plotH - h, Math.max(2, bw - 2), h, colors[mi], 0.92);
        bar.appendChild(document.createElementNS(NS, "title")).textContent = `${m.runner} · ${suite}: ${fmt(m.suites[suite])}`;
      });
      const label = text(s, left + si * group + group / 2, H - 30, suite.replace("_", " "), 10, "middle", "#161412");
      label.setAttribute("transform", `rotate(-28 ${left + si * group + group / 2} ${H - 30})`);
    });
  }

  function heatmap() {
    const host = $("#heatmap");
    const labelW = 190, cellW = 56, rowH = 13, headH = 72;
    const W = labelW + data.suites.length * cellW;
    const H = headH + data.measurements.length * rowH + 8;
    const s = mount(host, W, H);
    data.suites.forEach((suite, i) => {
      const x = labelW + i * cellW + cellW / 2;
      const t = text(s, x, headH - 12, suite.replace("_", " "), 9, "end", "#5d574d");
      t.setAttribute("transform", `rotate(-45 ${x} ${headH - 12})`);
    });
    data.measurements.forEach((m, r) => {
      const y = headH + r * rowH;
      text(s, 0, y + 9, m.runner, 8.5, "start", "#161412");
      data.suites.forEach((suite, i) => {
        const v = pct(m.suites[suite]);
        const shade = v >= 0.9 ? 0.95 : v >= 0.8 ? 0.72 : v >= 0.7 ? 0.5 : v >= 0.5 ? 0.3 : 0.15;
        rect(s, labelW + i * cellW, y, cellW - 3, rowH - 1, m.techniqueReimplementation ? "#8b8377" : "#9a3b2c", shade);
      });
    });
  }

  function precision() {
    const rows = data.precision?.["tev1-08b"] || [];
    if (rows.length < 2) return;
    const host = $("#precision-chart");
    const W = 480, H = 180, s = mount(host, W, H);
    const max = 1, base = H - 40;
    line(s, 30, base, W - 10, base, "#161412");
    rows.forEach((r, i) => {
      const h = (base - 20) * pct(r.accuracy) / max;
      const x = 70 + i * 170;
      const bar = rect(s, x, base - h, 90, h, "#9a3b2c", 0.9);
      text(s, x + 45, base - h - 8, fmt(r.accuracy), 16, "middle");
      text(s, x + 45, base + 18, `${r.device} · ${r.dtype || "default"}`, 10, "middle", "#5d574d");
      text(s, x + 45, base + 32, r.runId, 9, "middle", "#5d574d");
    });
    const gap = Math.abs(rows[0].accuracy - rows[1].accuracy);
    $("#precision-note").innerHTML = `<p><strong>${gap.toFixed(4)} apart.</strong> Precision at this scale is small but not zero, so it is recorded per row rather than assumed constant. The two runs use one pinned revision and identical bytes.</p>`;
  }

  function coverage() {
    const host = $("#coverage-chart");
    const pairs = [["Verified run directories", data.runDirectoriesVerified], ["Distinct measurements", data.distinctMeasurements]];
    const W = 560, H = 140, barX = 250, barMax = 250, s = mount(host, W, H);
    pairs.forEach(([label, value], i) => {
      const y = 16 + i * 56;
      text(s, 0, y + 14, label, 11, "start");
      const w = barMax * value / Math.max(...pairs.map((p) => p[1]));
      rect(s, barX, y, w, 18, "#9a3b2c", 0.85);
      text(s, barX + w + 10, y + 14, String(value), 13, "start");
    });
    const list = $("#collapse-list");
    const rep = data.duplicates.repeatedRuns || {}, ids = data.duplicates.identicalPredictions || {};
    const parts = [];
    Object.entries(rep).forEach(([runner, runs]) => parts.push(`<div class="exclusion"><b>${esc(runner)}</b>${esc(runs.join(" · "))} — one retried or re-hosted run, counted once.</div>`));
    Object.values(ids).forEach((runners) => parts.push(`<div class="exclusion"><b>${esc(runners.join(" / "))}</b>Byte-identical on all evaluation decisions — one measurement.</div>`));
    list.innerHTML = parts.join("");
    $("#exclusions").innerHTML = (data.excluded || []).map(([name, kind, why]) =>
      `<div class="exclusion"><b>${esc(name)} · ${esc(kind)}</b>${esc(why)}</div>`).join("");
  }

  function notes() {
    const d = data;
    const def = d.measurements.find((m) => m.deferrals);
    if (def) {
      $("#deferral-note").hidden = false;
      $("#deferral-note").innerHTML = `<p><strong>${esc(def.runner)} declined ${def.deferrals.deferred} of ${def.deferrals.evaluation_decisions} decisions.</strong> The contract has no abstention, so each was answered at the argmax and counted as a confident answer. Its accuracy is the adapter's projection, not its own coverage.</p>`;
    }
    if (d.readoutCaveat) {
      $("#readout-note").hidden = false;
      $("#readout-note").innerHTML = `<p><strong>Readout caveat.</strong> ${esc(d.readoutCaveat.note)}</p>`;
    }
    const top = d.measurements.slice(0, 3).map((m) => m.runner).join(", ");
    const spread = d.measurements[0].accuracy - d.measurements[4].accuracy;
    $("#tie-note").innerHTML = `<p><strong>Do not read rank inside the top band as a stable lead.</strong> ${esc(top)} and the next two are within ${spread.toFixed(4)} of each other at this sample size.</p>`;
    $("#gold-note").innerHTML = `<p><strong>Ground truth, plainly.</strong> Version 2.0.0 used one human reviewer who corrected an AI draft of every answer under protocol <code>human-reviewed-ai-assisted-v1</code>: no second reviewer, no adjudication and no inter-annotator statistic. Version 2.1.0 re-anchors four public suites on their source labels and keeps a per-case human decision on the emotion disagreements. Emotion remains single-reviewed.</p>`;
  }

  function observe() {
    const targets = $$(".band-head, .chart, .panel, .note, .method-grid article, .ledger, .ledger-row");
    if (reduced || !("IntersectionObserver" in window)) { targets.forEach((n) => n.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        if (e.target.classList.contains("ledger-row")) {
          e.target.style.animationDelay = `${(Number(e.target.dataset.i || 0) % 12) * 22}ms`;
        }
        e.target.classList.add("in");
        io.unobserve(e.target);
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    targets.forEach((n) => { n.classList.add("reveal"); io.observe(n); });
  }

  async function init() {
    try {
      const response = await fetch("data/results.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      data = await response.json();
    } catch (error) {
      document.body.insertAdjacentHTML("afterbegin", `<div class="note">Could not load the measurement data: ${esc(error.message)}.</div>`);
      return;
    }
    fillCounts(data);
    fillHero(data);
    renderLedger();
    const select = $("#model-select");
    select.innerHTML = data.measurements.map((m) => `<option value="${esc(m.runner)}">${esc(m.runner)} · ${fmt(m.accuracy)}</option>`).join("");
    suiteChart(); heatmap(); precision(); coverage(); notes(); observe();
    const film = $("#field-film");
    if (film) {
      const revealFilm = () => {
        if (film.currentTime > 1) film.parentElement.classList.add("ready");
      };
      film.addEventListener("timeupdate", revealFilm);
      film.addEventListener("loadeddata", revealFilm);
      film.addEventListener("error", () => film.parentElement.classList.add("no-video"));
      if (!film.canPlayType("video/webm") && !film.canPlayType("video/mp4")) {
        film.parentElement.classList.add("no-video");
      }
      setTimeout(() => {
        if (film.readyState < 2) film.parentElement.classList.add("no-video");
      }, 1800);
    }
    if (data.scopeTotal - data.distinctMeasurements === 0) {
      $("#coverage-title").textContent = "Every entry in scope has a recorded measurement";
    }
    $$("[data-filter]").forEach((b) => b.addEventListener("click", () => {
      $$("[data-filter]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); renderLedger();
    }));
    $$("[data-compare]").forEach((b) => b.addEventListener("click", () => {
      $$("[data-compare]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      $("#model-select").disabled = b.dataset.compare === "top"; suiteChart();
    }));
    $("#sort-by").addEventListener("change", renderLedger);
    select.addEventListener("change", suiteChart);
  }
  document.addEventListener("DOMContentLoaded", init);
})();
