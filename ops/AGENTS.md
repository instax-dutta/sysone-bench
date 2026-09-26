# Purpose
- Owns isolated execution helpers for approved remote workers.

# Ownership
- `pelican/` owns the read-only capacity gate, open-model launcher, Jev SSH boundary, image definition, and cleanup.
- `remote_jev_entrypoint.py` is the executable deployment wrapper for the mandated remote receiver path.
- Remote worker policy is fail-closed and does not authorize changes to shared workloads.

# Local Contracts
- Remote execution must not install packages globally on the worker.
- New work is limited to resources named and owned by the current run.
- Secret material is transient and never enters logs, files, argv, results, or diagnostics.
- The SSH client uses only `tejes@pelican`, the mandated remote receiver path, and a sanitized environment with inherited credentials removed.

# Work Guidance
- Keep host inspection separate from execution and cleanup.
- Preserve the nearest child contract when editing `pelican/`.

# Verification
- Run child-scoped tests and syntax checks without contacting the worker.

# Child DOX Index
- `pelican/` - shared-host isolation, capacity gates, and one-shot secret transport.
