# Purpose
- Owns durable design specifications and implementation plans for sysone-bench.

# Ownership
- `superpowers/specs/` contains approved design documents.
- Implementation plans and their execution records belong here when they affect durable project contracts.
- `decision-index-panel-coverage.md` records the expanded panel scope: measured, out-of-scope, and unreachable entries.
- `session-handoff-2026-10-03.md` records what was verified, what was lost to session death, and the exact resume steps for the 2x T4 work.
- `panel-hardware-requirements.md` records per-model VRAM estimates, the sharding result, and the acceptance checklist for contributed measurements.
- `GEO-AUDIT-REPORT.md` records the post-redesign GEO audit, its scores and the implemented fixes.
- `decision-index-panel-coverage.md` carries the measured results table; its counts come from `ops/panel_coverage.py` and must not be edited by hand.

# Local Contracts
- Design documents state approved scope, invariants, data flow, verification, and acceptance criteria.
- Documents must distinguish approved decisions from execution inputs and unresolved blockers.
- Historical benchmark records remain under `results/` and are not rewritten by documentation work.

# Work Guidance
- Use dated, descriptive filenames.
- Keep specifications operational and free of placeholders.
- Update the nearest DOX index when adding a durable documentation boundary.

# Verification
- Markdown files contain no unresolved placeholders or malformed code fences.
- Links and referenced repository paths are checked before closeout.

# Child DOX Index
- `superpowers/` - brainstorming specifications and implementation plans.
