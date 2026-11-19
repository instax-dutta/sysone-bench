# GEO Audit Report: sysone-bench

**Audit date:** 2026-11-19
**URL:** https://instax-dutta.github.io/sysone-bench/ (canonical target: https://sysone.sdad.pro/)
**Business type:** Publisher / independent research dataset
**Pages analyzed:** 1 HTML page plus `robots.txt`, `sitemap.xml`, `llms.txt`, `llms-full.txt`, `data/results.json` and response headers

---

## Executive summary

**Overall GEO Score: 70/100 (Fair)**

The redesigned site is technically open to answer engines: robots allows every major AI crawler,
the sitemap is fresh, `llms.txt` and `llms-full.txt` are linked from the page, and JSON-LD is valid.
Trust signals are unusually strong because ground-truth limitations, deferrals and readout caveats
are published rather than hidden. The gap is not crawling, it is the shape of the entity: there is no
independent scholarly citation, the private results mirror cannot be indexed, and the headline facts
were previously available only to JavaScript-executing clients.

### Score breakdown

| Category | Score | Weight | Weighted score |
|---|---:|---:|---:|
| AI Citability | 74 | 25% | 18.5 |
| Brand Authority | 48 | 20% | 9.6 |
| Content E-E-A-T | 84 | 20% | 16.8 |
| Technical GEO | 80 | 15% | 12.0 |
| Schema and structured data | 80 | 10% | 8.0 |
| Platform optimization | 55 | 10% | 5.5 |
| **Overall GEO Score** | | | **70.4 / 100** |

Platform Optimization is derived from the Brand Authority evidence: genuine directory listings exist
(systemonemodels.org, jevdirectory.org, awesomejev.cc, jevlibrary.dev, metatext.io, madewithlaya.com)
but there is no Wikipedia entry, paper, press coverage or active social footprint.

---

## Critical issues

None. The site is indexable, returns 200, and carries structured data.

## High-priority issues

1. **Generated copy contradicted the published data.** `llms.txt` and `llms-full.txt` claimed every
   GPU row was bf16 and every CPU row fp32. `results.json` contains fp16, fp32 and vendor-managed GPU
   rows. Fixed: the generator now states that dtype is recorded per row.
2. **A quoted caveat number was wrong.** `READOUT_CAVEAT.note` said `0.7292` for lev on banking77_12;
   the published row is `0.7917`. Fixed: the note is now formatted from the measured value.
3. **pngwn was counted and excluded at once.** The llms files called pngwn "unresolved rather than a
   measurement" while it appeared as measurement 50. Fixed: it stays in the count with an
   unresolved-readout flag.
4. **"No API access" conflicted with the closed-API reference score.** Fixed to "no privileged access
   to model internals"; locally run inputs remain byte-identical and that claim is now qualified.
5. **Headline numbers existed only in JavaScript.** Raw crawlers saw empty placeholders. Fixed with
   `webpage/template.html` and `webpage/prerender.py`; the Pages workflow bakes counts, best score,
   checksum and as-of date into `index.html` at publish time.
6. **Reproducibility relies on a private mirror.** One complete run directory, including its
   checksum manifest, should be published, or access to the private mirror must be documented.
   Not fixed: the mirror is deliberately private under the project's storage contract.

## Medium-priority issues

- No visible as-of date, now fixed via `generatedAt` in `results.json`, the generated llms files,
  the masthead and JSON-LD `dateModified`.
- No named maintainer or conflict-of-interest disclosure, now addressed with a Person node, a
  maintainer link and a no-vendor-funding note in the footer.
- No visible sources/licence/citation panel, now added to the data section.
- FAQ JSON-LD overstated byte-identical inputs across the closed API, now qualified.
- Canonical URLs still point at `sysone.sdad.pro` while DNS is being switched to GitHub Pages.
  This resolves when the user points the domain at `instax-dutta.github.io`.
- Third-party directories still cache the older 49/50 and kev-4b headline figures. The first-party
  README and generated files are now correct; cached third parties need a refresh request.
- Entity-name collision with the unrelated `systemonebench.dev` project, now disambiguated in the
  page's sources note.
- GitHub Pages cannot emit custom response headers. Missing `X-Content-Type-Options`, CSP,
  Referrer-Policy and Permissions-Policy are a platform constraint; a `meta` referrer policy is set.

## Low-priority issues

- llms tables show 12-character revision prefixes; a sentence now points to full 40-character
  revisions in `data/results.json`.
- `Dataset` lacked `variableMeasured`, dates, `inLanguage`, `sameAs` and a described `DataDownload`;
  all added to JSON-LD.
- Two images lacked intrinsic dimensions; `width`/`height` added to reduce CLS.
- No public DOI or Zenodo record. Not fixed; recommendation below.

---

## Category deep dives

### AI Citability (74/100)

Citable strengths: claim-first prose, explicit sample sizes, named protocols, per-run provenance, and
long caveat passages that can be quoted verbatim. The strongest single block is the ground-truth
disclosure. The main weaknesses were factual contradictions in generated copy (fixed) and the
absence of a public, crawlable copy of the two-host precision rows (they are now published under
`precision` in `results.json` and described in the llms files).

### Brand Authority (48/100)

The repo, a public HF discussion on `convaiinnovations/laya` and several independent directories are
real discovery signals. The HF results dataset is private (HTTP 401), no scholarly citation was
found, and the similarly named `systemonebench.dev` is a different project. The entity name is used
consistently across title, Open Graph, JSON-LD and llms files.

### Content E-E-A-T (84/100)

Ground-truth limitations, single-reviewer status, no adjudication, deferrals and readout caveats are
all published. Weaknesses: no named maintainer until this pass, no public artifact bundle, and no
primary-source links for the four public-label suites until this pass.

### Technical GEO (80/100)

robots.txt explicitly allows GPTBot, OAI-SearchBot, ChatGPT-User, ClaudeBot, Claude-User,
anthropic-ai, PerplexityBot, Google-Extended, Applebot-Extended, Bytespider, CCBot,
meta-externalagent, cohere-ai and JSONBot. Sitemap is fresh and declared. Canonical, Open Graph and
unlimited-snippet directives are present. JavaScript-only rendering was the significant gap and is
now mitigated by prerendering. GitHub Pages constrains response headers.

### Schema and structured data (80/100 before fixes)

Valid JSON-LD graph with WebSite, Organization, Person, Dataset and FAQPage. Volatile scores and
counts are correctly excluded. `variableMeasured`, dates, language, dataset `sameAs`, maintainer and
described download were missing and have been added.

### Platform optimization (55/100)

Directory listings exist but authority is weak (small view and engagement counts). No Wikipedia,
paper, press or active social channel. Private HF dataset contributes no crawlable signal.

---

## Quick wins implemented

1. Corrected the dtype claim, the lev caveat number and pngwn's status in the generators.
2. Added `generatedAt` and an as-of line to the data, llms files, masthead and JSON-LD.
3. Prerendered headline facts into `index.html` through `webpage/prerender.py` and the Pages workflow.
4. Added `variableMeasured`, dates, `inLanguage`, Person, `sameAs` and a described DataDownload to JSON-LD.
5. Added static caveat text in every injected note container, plus sources, licence, COI and
   disambiguation text.
6. Added intrinsic dimensions to both images and a meta referrer policy.

## Remaining 30-day plan

### Week 1
- [ ] Point `sysone.sdad.pro` (apex A/AAAA and `www` CNAME) at GitHub Pages and confirm HTTPS.
- [ ] Re-run this audit against the custom domain and verify canonical and JSON-LD @ids match.

### Week 2
- [ ] Publish one complete run directory with its checksum manifest as a public artifact.
- [ ] Ask the independent directories to refresh their cached headline figures.

### Week 3
- [ ] Create a citable identifier (Zenodo/DOI) and a short methods note with the manifest digest.

### Week 4
- [ ] Publish a public manifest/results HF Space or dataset while keeping raw GPU runs private.
- [ ] Seek one external citation and add one authoritative social or professional profile link.

---

## Appendix: sources analyzed

| Source | Purpose | Issues |
|---|---|---|
| `/` | Homepage HTML, metadata, JSON-LD, copy | CSR placeholders, caveat containers, schema completeness |
| `/robots.txt` | AI crawler access | none |
| `/sitemap.xml` | Indexing and freshness | domain cutover pending |
| `/llms.txt`, `/llms-full.txt` | Answer-engine corpus | factual contradictions, fixed |
| `/data/results.json` | Single source of numbers | no generatedAt before this pass, fixed |
| response headers | Hosting and security | custom headers unavailable on GitHub Pages |
