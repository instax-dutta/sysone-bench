# DOX framework
- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract
- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing
1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing
Every meaningful change requires a DOX pass before the task is done.
Update the closest owning AGENTS.md when a change affects:
- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy
- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape
- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style
- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout
1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences
- Never use emojis
- Use dashes, not em dashes
- No preamble, no recap, no closing pleasantries in responses
- Lead with the next action; number multi-step tasks; one concrete next action at the end
- Never run tests, builds, dataset generation, model jobs, or other batch commands on the local machine; run all execution on the remote benchmark host over ssh
- On the remote host, never modify unrelated processes, containers, files, or workloads; restrict work to the sysone-bench checkout and dedicated run/cache roots unless the user explicitly requests otherwise

## Project: sysone-bench
- First independent head-to-head benchmark of System One decision models: Laya (open weights) vs Jev (closed API), with Qwen PCD as a third open series
- Rule: every model answers byte-identical states and questions from one sealed manifest, fixed seed, same run. No vendor-published cross-comparison.
- Published v2.0.0 result: Jev 0.9065, Laya 0.6863, Qwen PCD 0.6048 over 1,240 evaluation decisions. Report at `results/v2/report-20260926/`.
- Never commit API keys. The Jev key is streamed over stdin and never written to disk in this repo or on the run host.
- Result JSONs are append-only records: never overwrite a published run, write a new file.
- The v2 ground truth used `human-reviewed-ai-assisted-v1`: one human reviewer corrected an AI draft, with no second reviewer and no adjudication. Do not describe it as two-reviewer or adjudicated.

## Child DOX Index
- `datasets/` - benchmark cases and ground truth. Owns case schema, labeling rules.
- `runners/` - model adapters (Laya, Jev). Owns the runner interface contract.
- `results/` - run outputs, comparisons, and published reports. Owns result schema, append-only rule, and the report directory contract.
- `docs/` - design specifications and implementation plans. Owns durable documentation boundaries.
- `benchmark/` - v2 core contracts, canonical identity, and append-only storage. Owns the core package contract.
- `ops/` - isolated remote execution, worker safety, and transient secret transport. Owns the remote execution boundary and its configuration contract.
- `tests/` - deterministic fake-only regression tests and fixtures. Owns test isolation and evidence.
- `tools/` - clean-environment command-line entry points. Owns external-package import protection.
- Root owns: `run.py`, `compare.py`, `PLAN.md`, `README.md`, env setup.
