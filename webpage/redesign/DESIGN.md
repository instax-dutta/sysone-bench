# Website redesign 01: Measurement ledger

## Direction

Selected: candidate 02, **Measurement ledger** (see `style-explorer.html`).

North star: reading an archival measurement ledger. The site is a citable instrument record, not a
product launch. Paper ground, ink typography, one oxide-red accent, hairline rules, and numerals set
large enough to be the graphic.

Why over the alternatives: the current calibration-bench (baseline) is already dark and amber; the
instrument candidate is a more literal version of the same idea. A warm paper ledger makes the 50
measurements read as evidence, gives the numbers maximum contrast, and is visually distinct from
the generic dark developer-tool template. It also carries a long ruled table and machine-readable
caveats without needing decoration.

## Tone translated into decisions

- Quiet, precise, archival: one accent at under 10% of any viewport; no glow, no gradient, no 3D.
- Dense but legible: 12-column ruled grid, 1px rules, mono aligned numerals, generous line height
  for prose.
- Serif display (Georgia / Iowan Old Style stack) for the few sentences; mono for every number and
  label; system sans only for controls.
- One memorable moment: the top result and its nine suite values arrive as a Typeset animation
  rendered with ManimGX, then the static ruled field takes over.

## Hero skeleton

User comes to read the headline measurement and scan the field. The first screen is the current best
open result, the closed reference, and the top of the ruled field. No slogan-only hero, no feature
cards. The animated ledger strip sits in the first screen and loops silently.

## Sections

1. Masthead: one line of identity, two links, dataset version.
2. Readout: best open score, closed reference, scope counts, animated manimgx strip.
3. The field: all 50 measurements, filter/sort, provenance flags, per-suite sparkline.
4. Suite evidence: per-suite bars, top-five compare, heatmap.
5. Precision and ties: direct CPU/T4 comparison, gap caveats.
6. Coverage: derived counts, collapsed duplicates, exclusions.
7. Method: sealed manifest, revisions, provenance, checksums, gold-truth caveat.
8. Data and AI: results.json, llms files, JSON-LD, reproduce commands.
9. Footer: scope and provenance.

## Deliberately removed

Marketing adjectives, decorative glow, the "hero headline then features" rhythm, hand-typed counts,
and any claim of byte-identical model-internal prompts. Counts and scores come from
`data/results.json` at render time in the browser.

## Verification

- `node --check webpage/app.js`
- local static server at 127.0.0.1:8899, desktop 1280x900 and mobile 390x844
- no horizontal page overflow; animations respect `prefers-reduced-motion`
- JSON-LD and GEO files regenerate from data; no volatile fact in share imagery
