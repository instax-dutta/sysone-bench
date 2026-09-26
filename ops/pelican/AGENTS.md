# Purpose
- Owns safe, isolated execution on the shared `tejes@pelican` worker.

# Ownership
- `preflight.py` owns read-only host capacity and capability checks.
- `run_open_model.sh` owns one open-model container and its run directory.
- `remote_jev.py` and `remote_jev_entrypoint.py` own one-shot Jev credential transport.
- `cleanup.sh` removes only a container carrying this run's ownership label.
- `Dockerfile` defines the Python 3.12 CPU runtime image.

# Local Contracts
- The worker contract is 4 CPUs, 12 GiB memory, no exposed ports, `nice 19`, idle I/O, a dedicated workspace, and a dedicated model cache.
- no-global-install: dependencies belong in the image or its virtual environment, never in a host-wide installation.
- Preflight is read-only and requires a complete snapshot: process CPU affinity must include CPUs 0-3, load1/cpu must be at most 0.50, load5/cpu at most 0.75, memory at least 16 GiB, disk at least 100 GiB on the fixed workspace root, AVX2, Python 3.12, a ready dedicated workspace root, a known run-root state, an absent owned run path before creation, a valid post-run owner marker, an absent owned container, an owned non-symlink model cache, a fixed sealed manifest with a valid checksum, and no read errors. A known absent run root may be created only after the gate; symlinked, occupied, and unknown roots fail.
- Preflight distinguishes a Docker `No such object` response from daemon or permission errors and treats all unknown filesystem or container states as failures.
- `collect_snapshot()` reports an existing owned run directory as `run_path_state="ready"`; `allow_owned` accepts only that state with a valid owner marker and rejects absent, symlink, unknown, unowned, and non-directory states.
- The launcher creates only its own run directory and container, starts neither until preflight passes, and never mounts the Docker socket or publishes ports.
- After creating a run directory, the launcher exclusively creates `.sysone-owner`, revalidates fixed-root ownership and path shape, and repeats that validation before Docker and after model execution; marker mismatch, symlink replacement, and unowned paths fail closed.
- Ordinary symlink and ownership replacement windows are closed by the marker checks. A same-user process that atomically swaps a path between the final shell check and Docker's bind mount cannot be eliminated by POSIX shell; this residual race is documented and never treated as safe.
- The fixed workspace root is `/home/tejes/sysone-bench-v2`; the fixed run root is its direct child `/home/tejes/sysone-bench-v2/runs`, and run paths are direct children of that run root.
- Preflight disk availability is measured on the fixed workspace root, not a broader host directory.
- The launcher rejects root invocation, derives the container UID/GID from fixed host `id` paths, and passes those numeric identities to `docker run --user`; it mounts the sealed manifest and checksum read-only.
- The fixed project image is `sysone-bench-v2:pelican-cpu`; build it with `docker build --file ops/pelican/Dockerfile --tag sysone-bench-v2:pelican-cpu .`.
- The image mounts the owned run directory only at `/results/<run_id>`; application code remains in the image at `/workspace`. The in-container output root basename therefore equals the requested run ID.
- The image CPU build uses exact Torch `2.14.0+cpu` from `https://download.pytorch.org/whl/cpu`, the frozen `uv.lock` with the `qwen` extra for other runtime packages, a filter limited to Torch/CUDA/NVIDIA/Triton lines, and a build-time CUDA rejection guard. The filtered closure must retain Transformers. The checked-in PyPI lock's Torch dependency includes CUDA packages, so the CPU index path must not be removed or replaced with the default PyPI Torch resolution.
- The launcher invokes `benchmark.orchestrator` with the fixed model, sealed manifest, checksum, owned `/results/<run_id>` run directory, and run ID. The exact `.sysone-owner` marker is the sole pre-existing-directory exception; it allows the four model artifacts and defers checksum finalization to the launcher only when the directory basename equals `run_id` and no required v2 artifact exists.
- The launcher runs `/usr/bin/nice -n 19 /usr/bin/ionice -c 3` around the in-container model process, not around the Docker client.
- After successful model execution, the launcher records the post-run snapshot, rejects symlinked artifacts, and exclusively finalizes `checksums.sha256` through `benchmark.storage.write_checksums`. Failed model executions never receive a success checksum.
- Existing containers and workloads are never stopped, restarted, reconfigured, or removed; existing containers remain untouched.
- Cleanup accepts only the immutable container ID recorded in the owned run directory's cidfile, revalidates the observed ID plus ownership and run labels, and removes only that ID. A missing cidfile, invalid ID, inspection failure, label mismatch, or name collision does nothing; cleanup never resolves or removes by container name.
- The launcher captures the cidfile ID after Docker returns and retains it for the trap. If a signal arrives before Docker writes the cidfile, the trap safely does nothing because no immutable ID handoff exists; this residual timing limitation must remain documented.
- A missing or failed model command is a failed run; the launcher must not create a successful result.
- `TYPESAFE_API_KEY` is read only from the local gitignored `.env`, held in memory, and sent as one SSH stdin line to the fixed `tejes@pelican` destination and mandated `/home/tejes/sysone-bench-v2/ops/remote_jev_entrypoint.py` path. The public sender accepts no destination or credential-file overrides. It is never placed in argv, logs, files, results, or exception text.
- The remote receiver removes inherited `TYPESAFE_API_KEY` values from its copied environment before adding the one stdin value, passes the key only in the child environment, clears that child environment after the one-shot process, and does not persist it.
- The default remote Jev command writes below `/home/tejes/sysone-bench-v2/results/v2/runs`, supplies a UTC-and-UUID safe run ID, and writes a normal v2 checksum. Any explicit receiver arguments are treated as the complete orchestrator command and are never combined with the default.
- The fixed SSH transport timeout is `SSH_TIMEOUT_SECONDS = 21600` seconds for bounded full-suite execution.

# Work Guidance
- Keep preflight checks before directory or container creation where a collision can be detected.
- Use quoted argument arrays and fixed run identifiers; never use shell evaluation.
- Keep model execution and metric logic in Task 6 orchestration. This boundary owns only preflight, the owned run directory, final postflight checksuming, container isolation, and cleanup; it fails closed when any required artifact is absent.
- Do not add model weights, result JSONs, dataset artifacts, or local secrets to the image context.

# Verification
- `uv run --offline pytest tests/test_pelican_preflight.py -q`
- `bash -n ops/pelican/run_open_model.sh ops/pelican/cleanup.sh`
- `uv run --offline ruff check ops/pelican/*.py ops/remote_jev_entrypoint.py tests/test_pelican_preflight.py`
- `uv run --offline ruff format --check ops/pelican/*.py ops/remote_jev_entrypoint.py tests/test_pelican_preflight.py`
- `uv run --offline mypy --allow-subclassing-any --follow-imports=skip --ignore-missing-imports ops/pelican/*.py ops/remote_jev_entrypoint.py tests/test_pelican_preflight.py`
- Dockerfile, `.dockerignore`, and YAML checks are static only; do not start Docker or contact pelican from development tests.

# Child DOX Index
- No child DOX files yet.
