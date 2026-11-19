# webpage

Static site for sysone-bench. No framework, no build step. Every push to `master` that touches this
directory deploys the folder through `.github/workflows/deploy-pages.yml` to GitHub Pages.

Preview locally with `python3 -m http.server 8899` from this directory.

Custom domain: `sysone.sdad.pro`. The committed `CNAME` file attaches it once DNS points at the
GitHub Pages address `instax-dutta.github.io`.

## Regenerating the data

Every number on the site is rendered from `data/results.json`, and that file is generated from
checksum-verified run directories. Nothing is typed by hand.

    python webpage/build-data.py
    python webpage/prerender.py

It calls `ops/panel_coverage.py` for the authoritative counts, so the site's headline number and the
repository's coverage number cannot drift apart. If a run's checksum fails, the measurement is
dropped rather than shown.

## Files

| File | Purpose |
|---|---|
| `template.html` | Source page with `{{tokens}}`; edit this, not the rendered file |
| `prerender.py` | Fills headline facts from `data/results.json` into `index.html` |
| `index.html` | Rendered page with static headline facts and JSON-LD (dataset, publisher, FAQ) |
| `style.css` | Editorial design system: paper, ink, hairlines. One oxblood accent |
| `app.js` | Hand-built SVG charts, ranked field list, filters, reveal-on-scroll |
| `data/results.json` | Generated. Do not edit by hand |
| `llms.txt` | Plain-language summary for AI answer engines |
| `llms-full.txt` | Generated. Every result and caveat in one self-contained document |
| `robots.txt` | Explicitly permits GPTBot, ClaudeBot, PerplexityBot and others |
| `sitemap.xml` | Single-URL sitemap |
| `CNAME` | Custom domain attachment for GitHub Pages |
| `visuals/scenes.py` | ManimGX scenes for the animated ledger and manifest diagrams |
| `assets/` | Rendered MP4 animations and their settled poster frames |
| `redesign/` | Approved direction, comparison page, screenshots and review evidence |
| `favicon.svg` | Icon mark: three ranked bars in amber on graphite. Source of truth for the icon |
| `icon-512.png` | Raster of `favicon.svg` at 512, for manifest and social fallbacks |
| `apple-touch-icon.png` | Full-bleed 180 square. iOS applies its own mask, so no pre-rounded corners |
| `og.png` | 1200x630 social card |

## Social image and icons

`og.png` is rendered by a headless-Chromium pass, using the same graphite, bone, amber and cyan as the
site. It deliberately prints no volatile fact: no model count, no best open score, no ranking. A social
card is a static PNG, so any of those would be a false claim the moment a new model is measured. It
carries the stable method facts only - sealed case count, scored decisions per model, closed-API
reference, seed - and lists the nine suites as a ruled list rather than as bars, because equal-length
bars would assert a magnitude that does not exist.

Regenerate it if the method framing changes, using these constraints:

1. Build a 1200x630 HTML card in the site palette that carries only stable method facts: sealed case
   count, scored decisions per model, the closed-API reference, seed, and the nine suites as a ruled
   list. Never read the model count, best open score or ranking into it.
2. Screenshot it at 1200x630 with the viewport matched, and assert `scrollHeight <= 630`,
   `scrollWidth <= 1200`, and that the footer sits above the fold. An earlier pass silently clipped the
   stats and the footer.
3. `apple-touch-icon.png` and `icon-512.png` are the same mark rendered at 180 and 512, full-bleed with
   no pre-rounded corners. A fixed-size wrapper leaves the mark in the corner of a larger canvas.

All four files are committed assets. Serving the site still requires no build step.

## Local preview

    cd webpage && python3 -m http.server 8899

Then open `http://127.0.0.1:8899`. A server is required, not `file://`, because `app.js` fetches
`data/results.json`.

## Design

A calibration lab. Graphite bench, bone ink, hairline rules, no cards, no shadows, no glow.
Space Grotesk for display and prose, IBM Plex Mono for every number. Sodium-amber for open weights,
cold cyan reserved for the closed-API reference, grey for technique reimplementations, red for
unresolved. The numbers are the graphic. No external script runtime: reveals use `IntersectionObserver`
and bars grow with the Web Animations API.

## Charts

Six, all inline SVG built by `app.js`:

1. **Field** - all 50 measurements as a ranked list with hairline bars and a nine-suite sparkline per
   row, plus a strip plot in the readout
2. **Per-suite** - grouped bars for one model, or the top five
3. **Heatmap** - 49 models against 9 suites, horizontally scrollable
4. **Precision** - tev1-08b on CPU fp32 against T4 fp16, on a truncated axis that says so
5. **Coverage** - verified directories, distinct measurements and unmeasured entries

Axes that do not start at zero state it on the chart.
