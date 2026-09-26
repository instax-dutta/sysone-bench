# Purpose
- Owns deterministic tests and test-only fixtures for sysone-bench.

# Ownership
- Tests validate public runner, benchmark, storage, data, and operations contracts without owning production behavior.
- Test fixtures must not load model weights, call real inference, contact the network, or read the local secret environment.

# Local Contracts
- New behavior follows test-first red-green-refactor evidence.
- Tests use fakes, temporary paths, mocked subprocess or SSH boundaries, fake Docker and launcher behavior, immutable cidfile/ID cleanup handoffs, complete fixed-path preflight snapshots, real collected postflight directory state, separate workspace/run roots, owner-marker replacement scenarios, and offline commands.
- Release figure fixtures emit the complete validated source-row shape, including aligned source identities, context-derived `source_kind` roles, dataset and uncertainty metadata, numeric plotted values, and explicit empty families; tests must not weaken writer validation to accommodate stale fixtures.
- Tests must not modify dataset contents, published result JSONs, remote hosts, or existing containers.
- Secret tests use placeholders only and assert that credentials stay out of argv, logs, files, and exception text.

# Work Guidance
- Keep assertions focused on observable behavior and fail closed on malformed or unsafe inputs.
- Task 5 tests may verify the exact Task 6 command and checksum handoff through static or fake boundaries, but must never perform real Task 6 orchestration or model execution.
- Update the nearest owning contract when a test boundary gains a durable rule.

# Verification
- `uv run --offline pytest tests/test_pelican_preflight.py -q`
- `uv run --offline pytest tests/test_statistics.py -q`
- `uv run --offline pytest tests/test_figure_data.py -q`
- `bash -n ops/pelican/run_open_model.sh ops/pelican/cleanup.sh`
- Scoped Ruff, Ruff format, and mypy checks cover new Python files.
- `git diff --check` and result-artifact immutability checks run at closeout.

# Child DOX Index
- No child DOX files yet.
