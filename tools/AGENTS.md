# Purpose
- Owns thin command-line entry points that execute repository workflows outside package import paths.

# Ownership
- `build_dataset_v2.py` owns the clean-environment v2 dataset build entry point.
- The entry point owns external `datasets` import validation and local v2 package-path extension; it does not own dataset labels or source metadata.

# Local Contracts
- Import external `datasets` before adding the repository root to `sys.path`.
- Reject a local `datasets/__init__.py` resolution and extend the external package path explicitly.
- Pass locked wrapper, split, config, and revision metadata through the builder.
- Do not run production downloads or batch builds locally; execute them on the approved remote worker.

# Work Guidance
- Keep CLI behavior thin and delegate construction and serialization to `datasets.v2`.
- Never print credentials, access model APIs, or modify result artifacts from a data build command.

# Verification
- `uv run --extra build pytest tests/test_dataset_counts.py -q`
- `uv run ruff check tools/build_dataset_v2.py`
- `uv run mypy tools/build_dataset_v2.py`

# Child DOX Index
- None.
