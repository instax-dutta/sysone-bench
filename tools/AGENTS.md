# Purpose
- Owns thin command-line entry points that execute repository workflows outside package import paths.

# Ownership
- `build_dataset_v2.py` owns the clean-environment v2 dataset build entry point.
- `build_manifest_v210.py`, `rescore_v210.py` and `write_runs_v210.py` own the 2.1.0 gold correction and
  its successor records. `rescore_v210.py` scores stored answers and needs no GPU; `write_runs_v210.py`
  writes the `-v210` directories beside the runs they supersede and records any `deferrals` block it
  finds in the predictions.
- `build_llms.py` renders `webpage/llms.txt` and `webpage/llms-full.txt` from the generated site data.
- The entry point owns external `datasets` import validation and local v2 package-path extension; it does not own dataset labels or source metadata.

# Local Contracts
- Import external `datasets` before adding the repository root to `sys.path`.
- Reject a local `datasets/__init__.py` resolution and extend the external package path explicitly.
- Pass locked wrapper, split, config, and revision metadata through the builder.
- Do not run production downloads or batch builds locally; execute them on the approved remote worker.
- Resolve paths relative to the repository root, never a hardcoded home directory. `results/AGENTS.md`
  forbids hardcoded home paths in scripts; honour `SYSONE_RESULTS_ROOT` where a results root applies.
- Never write a count, score, digest or rank into a generated file by hand. `build_llms.py` exists
  because hand-written GEO copy drifted away from the data it described.

# Work Guidance
- Keep CLI behavior thin and delegate construction and serialization to `datasets.v2`.
- Never print credentials, access model APIs, or modify result artifacts from a data build command.

# Verification
- `uv run --extra build pytest tests/test_dataset_counts.py -q`
- `uv run ruff check tools/build_dataset_v2.py`
- `uv run mypy tools/build_dataset_v2.py`
- `python tools/build_llms.py` twice and confirm byte-identical output
- `python webpage/build-data.py && python tools/build_llms.py` after any change to results data

# Child DOX Index
- None.
