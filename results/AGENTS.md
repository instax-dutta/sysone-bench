# Purpose
- Append-only store of run outputs and head-to-head comparisons.

# Ownership
- Result files are records. Never edit or overwrite one; write a new file.

# Local Contracts
- Run file: `run_<runner-name>_<YYYYMMDD-HHMMSS>.json` with keys
  `meta`, `suites`, `speed_scaling`, `gating`.
- `meta` must include: runner name, model/version pin, timestamp, device,
  seed, question-source hash (to prove byte-identical questions).
- `compare.py` output: `compare_<a>_vs_<b>.json`. Never hand-edit.

# Work Guidance
- Empty until first `run.py` execution.

# Verification
- Every run file validates with: `./.venv/bin/python -c "import json; json.load(open('results/<file>.json'))"`.

# Child DOX Index
- None.
