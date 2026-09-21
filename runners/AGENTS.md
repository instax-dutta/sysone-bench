# Purpose
- Owns the runner interface: the contract every model adapter implements.
- Root `run.py` talks only to this interface, never to model SDKs directly.

# Ownership
- Interface changes require updating every adapter (`laya_runner.py`, `jev_runner.py`) in the same edit.

# Local Contracts
- `BaseRunner.predict(state, questions) -> dict` returns `{"answers": {qid: answer}}` where each answer is
  `{"type": "choice"|"noul"|"score", ...}` matching the laya answer shape:
  - choice: `{"type": "choice", "choice": <label>, "probabilities": {...}, "confidence": <0-1>}`
  - noul: `{"type": "noul", "noul": <P(true) 0-1>}`
  - score: `{"type": "score", "score": <float>, "confidence": <0-1>}`
- `BaseRunner.name` is the run-file identifier (`laya`, `jev-1.13.0`, ...).
- `BaseRunner.info()` returns metadata dict recorded into the result file
  (model id, version pin, device, notes).
- Runners must not mutate the input questions dict.
- Secrets: only `jev_runner.py` may read `TYPESAFE_API_KEY` from env. Never log it.

# Work Guidance
- Normalize vendor responses to the laya answer shape at the adapter edge, so
  `run.py` and `compare.py` stay model-agnostic.

# Verification
- `./.venv/bin/python -c "from runners.laya_runner import LayaRunner; print(LayaRunner.name)"` prints `laya`.

# Child DOX Index
- None. `base.py`, `laya_runner.py`, `jev_runner.py`, `router_runner.py` all live under this contract.
