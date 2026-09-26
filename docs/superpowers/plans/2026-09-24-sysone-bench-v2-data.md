# Sysone-bench v2 Dataset and Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provenance-complete 1,190-case dataset v2, generate blind human-review packets, record agreement and adjudication, and seal a full SHA-256 manifest before any model run.

**Architecture:** Dataset construction lives under `datasets/v2/` and uses the canonical contracts from the core plan. A standalone build entry point is executed outside the repository root so the local `datasets` package cannot shadow Hugging Face Datasets. Human labels are JSON exports from self-contained HTML packets and are never inferred from model output.

**Tech Stack:** Python 3.12, Hugging Face Datasets in the `build` extra, standard-library JSON/HTML generation, pytest, and the canonical manifest API.

**Spec:** `docs/superpowers/specs/2026-09-24-sysone-bench-v2-design.md`

**Depends on:** `docs/superpowers/plans/2026-09-24-sysone-bench-v2-core.md`

## Global Constraints

- Preserve `datasets/cases.py`, `datasets/public_cases.json`, and all legacy result records.
- Final counts are exactly 1,190 cases, 1,550 decisions, 238 calibration cases, and 952 evaluation cases.
- Use deterministic stratified splitting with seed `42`.
- Every public case stores source dataset, wrapper, resolved revision, split, row ID, source label, local modification, license, citation, and checksum.
- Human reviewers never receive existing labels, source labels, model names, or prior scores.
- Every multilingual case needs native-speaker review and a second independent intent label.
- No model run starts until the manifest is sealed.
- Never commit, push, or publish automatically.

## File Map

- Create `datasets/v2/__init__.py` - dataset v2 package marker.
- Create `datasets/v2/sources.py` - source registry and resolved revision metadata.
- Create `datasets/v2/build_manifest.py` - deterministic public-source and curated case construction.
- Create `datasets/v2/labeling.py` - blind packet, agreement, and adjudication generation.
- Create `datasets/v2/validate.py` - count, balance, split, provenance, and duplicate gates.
- Create `tools/build_dataset_v2.py` - clean-environment build entry point executed outside the repository root.
- Create `datasets/v2/sources.lock.json` - resolved public dataset revisions and licenses.
- Create `datasets/v2/manifest.jsonl` - generated sealed case records.
- Create `datasets/v2/provenance.json` - source and local-modification registry.
- Create `datasets/v2/manifest.sha256` - full manifest digest.
- Create `datasets/v2/labels/reviewer_a.json` - Reviewer A export after human completion.
- Create `datasets/v2/labels/reviewer_b.json` - Reviewer B export after human completion.
- Create `datasets/v2/labels/adjudication.json` - final adjudicated labels after human completion.
- Create `datasets/v2/labels/agreement.json` - generated agreement report.
- Create `datasets/v2/labeling/reviewer_a.html` - blind Reviewer A packet.
- Create `datasets/v2/labeling/reviewer_b.html` - blind Reviewer B packet.
- Create `datasets/v2/labeling/adjudication.html` - disagreement-only packet.
- Create `datasets/v2/AGENTS.md` - dataset v2 ownership and sealing rules.
- Create `tests/test_dataset_counts.py` - exact suite and split counts.
- Create `tests/test_dataset_provenance.py` - provenance and source lock validation.
- Create `tests/test_labeling_packets.py` - no-label leakage and export schema tests.
- Create `tests/test_dataset_duplicates.py` - exact and normalized duplicate gates.

- Create `tests/fixtures.py` - deterministic small synthetic records and packet helpers used by data tests.

---

### Task 0: Define deterministic dataset test fixtures

**Files:**
- Create: `tests/fixtures.py`

**Interfaces:**
- Produces `load_test_records() -> list[dict[str, object]]`.
- Produces `load_sources() -> dict[str, object]`.
- Produces `write_test_packet(tmp_path: Path) -> Path`.
- Produces `seal_test_manifest(tmp_path: Path, order: list[str]) -> str`.

- [ ] **Step 1: Add the fixture data**

```python
import hashlib
import json
from pathlib import Path

SUITE_COUNTS = {
    "triage": 60,
    "guardrails": 60,
    "moderation": 60,
    "agnews": 200,
    "emotion": 240,
    "banking77_12": 120,
    "mnli": 150,
    "sst5": 150,
    "multilingual_intent": 150,
}
SUITE_DECISIONS = {"triage": 4, "guardrails": 2, "moderation": 3}
CALIBRATION_CASES = {"triage": 12, "guardrails": 12, "moderation": 12, "agnews": 40, "emotion": 48, "banking77_12": 24, "mnli": 30, "sst5": 30, "multilingual_intent": 30}


def load_sources() -> dict[str, object]:
    return {
        "agnews": {"canonical": "fancyzhx/ag_news", "wrapper": "fancyzhx/ag_news", "revision": "resolved-commit", "split": "test", "license": "test-license", "citation": "test-citation"},
        "emotion": {"canonical": "dair-ai/emotion", "wrapper": "dair-ai/emotion", "revision": "resolved-commit", "split": "test", "license": "test-license", "citation": "test-citation"},
        "banking77_12": {"canonical": "PolyAI/banking77", "wrapper": "mteb/banking77", "revision": "resolved-commit", "split": "test", "license": "test-license", "citation": "test-citation"},
        "mnli": {"canonical": "nyu-mll/glue", "wrapper": "nyu-mll/glue", "revision": "resolved-commit", "split": "validation_matched", "license": "test-license", "citation": "test-citation"},
        "sst5": {"canonical": "SetFit/sst5", "wrapper": "SetFit/sst5", "revision": "resolved-commit", "split": "test", "license": "test-license", "citation": "test-citation"},
    }


def load_test_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for suite_id, count in SUITE_COUNTS.items():
        decision_count = SUITE_DECISIONS.get(suite_id, 1)
        calibration_count = CALIBRATION_CASES[suite_id]
        for index in range(count):
            question_ids = [f"q{number}" for number in range(decision_count)]
            question_objects = [{"qid": qid, "type": "noul", "instructions": "fixture question"} for qid in question_ids]
            expected = {qid: index % 2 for qid in question_ids}
            records.append({
                "schema_version": 2,
                "dataset_version": "2.0.0",
                "suite_id": suite_id,
                "case_id": f"{suite_id}-{index:04d}",
                "order_index": index,
                "split": "calibration" if index < calibration_count else "evaluation",
                "state": {"text": f"{suite_id} fixture {index}"},
                "questions": question_objects,
                "expected": expected,
                "provenance": {
                    "source_id": suite_id,
                    "canonical": load_sources().get(suite_id, {}).get("canonical", "curated"),
                    "wrapper": load_sources().get(suite_id, {}).get("wrapper", "curated"),
                    "revision": load_sources().get(suite_id, {}).get("revision", "curated-v2"),
                    "split": load_sources().get(suite_id, {}).get("split", "curated"),
                    "row_id": index,
                    "source_label": "fixture",
                    "local_modification": "synthetic fixture",
                    "license": load_sources().get(suite_id, {}).get("license", "curated"),
                    "citation": load_sources().get(suite_id, {}).get("citation", "curated"),
                    "checksum": hashlib.sha256(f"{suite_id}:{index}".encode()).hexdigest(),
                },
                "label_provenance": "fixture-label",
            })
    return records


def write_test_packet(tmp_path: Path) -> Path:
    from datasets.v2.labeling import write_reviewer_packet
    output = tmp_path / "packet.html"
    write_reviewer_packet(load_test_records()[:2], output, "A")
    return output


def seal_test_manifest(tmp_path: Path, order: list[str]) -> str:
    from benchmark.manifest import manifest_digest
    records = [{"suite_id": "fixture", "case_id": "fixture-0001", "order_index": 0, "split": "evaluation", "state": {}, "questions": [{"qid": qid, "type": "noul", "instructions": "fixture question"} for qid in order], "expected": {qid: 1 for qid in order}, "provenance_id": "fixture"}]
    return manifest_digest(records)
```

- [ ] **Step 2: Run the fixture import test**

Run: `uv run --extra build python -c "import sys; sys.path.insert(0, 'tests'); from fixtures import load_test_records; assert len(load_test_records()) == 1190"`

Expected: PASS after the fixture module is created.

---

### Task 1: Define source registry and deterministic suite sampling

**Files:**
- Create: `datasets/v2/sources.py`
- Create: `tests/test_source_registry.py`
- Create: `datasets/v2/sources.lock.json`

**Interfaces:**
- Produces `source_record(source_id: str) -> Mapping[str, Any]`.
- Produces `derive_seed(suite_id: str, base_seed: int = 42) -> int`.
- Produces `sample_rows(rows: Sequence[Mapping[str, Any]], count: int, suite_id: str) -> list[Mapping[str, Any]]`.

- [ ] **Step 1: Write failing source tests**

```python
from datasets.v2.sources import derive_seed, sample_rows


def test_suite_seed_is_stable_and_distinct():
    assert derive_seed("agnews") == derive_seed("agnews")
    assert derive_seed("agnews") != derive_seed("emotion")


def test_sampling_is_deterministic():
    rows = [{"id": index} for index in range(20)]
    assert sample_rows(rows, 5, "agnews") == sample_rows(rows, 5, "agnews")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra build pytest tests/test_source_registry.py -q`

Expected: FAIL because `datasets.v2.sources` does not exist.

- [ ] **Step 3: Implement suite-specific seed derivation**

Use:

```python
import hashlib
import random
from collections.abc import Mapping, Sequence
from typing import Any


def derive_seed(suite_id: str, base_seed: int = 42) -> int:
    digest = hashlib.sha256(f"{base_seed}:{suite_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sample_rows(rows: Sequence[Mapping[str, Any]], count: int, suite_id: str) -> list[Mapping[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError("sample count is outside the source row count")
    generator = random.Random(derive_seed(suite_id))
    return [rows[index] for index in generator.sample(range(len(rows)), count)]
```

- [ ] **Step 4: Define and resolve source records**

Register these source IDs and canonical names:

```python
SOURCES = {
    "agnews": {
        "canonical": "fancyzhx/ag_news",
        "wrapper": "fancyzhx/ag_news",
        "split": "test",
    },
    "emotion": {
        "canonical": "dair-ai/emotion",
        "wrapper": "dair-ai/emotion",
        "split": "test",
    },
    "banking77": {
        "canonical": "PolyAI/banking77",
        "wrapper": "mteb/banking77",
        "split": "test",
    },
    "mnli": {
        "canonical": "nyu-mll/glue",
        "wrapper": "nyu-mll/glue",
        "config": "mnli",
        "split": "validation_matched",
    },
    "sst5": {
        "canonical": "SetFit/sst5",
        "wrapper": "SetFit/sst5",
        "split": "test",
    },
}
```

Resolve the current commit for each wrapper with the Hugging Face Hub API, record the resolved commit and license in `sources.lock.json`, and fail the build if a source cannot be resolved. Do not leave `main` as the reproducibility revision.

- [ ] **Step 5: Run source tests**

Run: `uv run --extra build pytest tests/test_source_registry.py -q`

Expected: PASS.

---

### Task 2: Build the exact v2 case inventory

**Files:**
- Create: `datasets/v2/build_manifest.py`
- Create: `tools/build_dataset_v2.py`
- Create: `tests/test_dataset_counts.py`

**Interfaces:**
- Produces `build_records(output_root: Path, sources: Mapping[str, Any]) -> list[dict[str, Any]]`.
- Produces `write_manifest(records: Sequence[Mapping[str, Any]], output_root: Path) -> str`.
- CLI: `python tools/build_dataset_v2.py --output datasets/v2`.

- [ ] **Step 1: Write failing count tests**

```python
from collections import Counter


def test_v2_counts_match_the_approved_spec():
    expected = {
        "triage": 60,
        "guardrails": 60,
        "moderation": 60,
        "agnews": 200,
        "emotion": 240,
        "banking77_12": 120,
        "mnli": 150,
        "sst5": 150,
        "multilingual_intent": 150,
    }
    records = load_test_records()
    assert Counter(record["suite_id"] for record in records) == expected
    assert len(records) == 1190
    assert sum(len(record["expected"]) for record in records) == 1550
```

- [ ] **Step 2: Run the count test to verify it fails**

Run: `uv run --extra build pytest tests/test_dataset_counts.py -q`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement deterministic suite construction**

Build the following exact inventories:

- Triage: 12 cases for each of five intents, with binary labels balanced 30/30.
- Guardrails: 30 positive and 30 negative cases with both labels independently adjudicated.
- Moderation: 30 toxic-positive and 30 toxic-negative cases, with balanced threat and spam coverage and allowed overlap.
- AG News: 50 cases per topic.
- Emotion: 40 cases per emotion.
- Banking77: 10 cases for each of 12 intents.
- MNLI: 50 cases per relation.
- SST-5: 30 cases per level.
- Multilingual: five languages, six intents including `other`, and five cases per language-intent pair.

Use suite-specific seeds from Task 1. Keep original row IDs and source labels in provenance fields. Generate a stable `case_id` with the format `<suite_id>-<four-digit order>`.

- [ ] **Step 4: Implement the clean-environment build entry point**

`tools/build_dataset_v2.py` must import external `datasets` before adding the repository root to `sys.path`. It must assert that the imported package is not the local `datasets/__init__.py`, and it must invoke `build_records` from `datasets.v2.build_manifest`.

The documented command is:

```bash
cd /tmp
/absolute/path/to/sysone-bench/.venv/bin/python /absolute/path/to/sysone-bench/tools/build_dataset_v2.py --output /absolute/path/to/sysone-bench/datasets/v2
```

- [ ] **Step 5: Run the count test**

Run: `uv run --extra build pytest tests/test_dataset_counts.py -q`

Expected: PASS after the generated test fixture is built.

---

### Task 3: Add split assignment, provenance, and duplicate gates

**Files:**
- Create: `datasets/v2/validate.py`
- Modify: `datasets/v2/build_manifest.py`
- Create: `tests/test_dataset_provenance.py`
- Create: `tests/test_dataset_duplicates.py`

**Interfaces:**
- Produces `assign_splits(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]`.
- Produces `validate_dataset(records: Sequence[Mapping[str, Any]], sources: Mapping[str, Any]) -> dict[str, Any]`.
- Produces `normalized_state(value: object) -> str`.

- [ ] **Step 1: Write failing split and provenance tests**

```python
def test_split_counts_are_exact():
    summary = validate_dataset(load_test_records(), load_sources())
    assert summary["calibration_cases"] == 238
    assert summary["evaluation_cases"] == 952
    assert summary["calibration_decisions"] == 310
    assert summary["evaluation_decisions"] == 1240


def test_public_case_requires_source_revision():
    record = load_test_records()[0]
    record["provenance"]["revision"] = "main"
    with pytest.raises(ValueError, match="resolved revision"):
        validate_dataset([record], load_sources())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra build pytest tests/test_dataset_provenance.py -q`

Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Implement deterministic stratified splitting**

Assign calibration cases with the following exact per-suite counts:

```python
CALIBRATION_CASES = {
    "triage": 12,
    "guardrails": 12,
    "moderation": 12,
    "agnews": 40,
    "emotion": 48,
    "banking77_12": 24,
    "mnli": 30,
    "sst5": 30,
    "multilingual_intent": 30,
}
```

Use `derive_seed(f"split:{suite_id}")`, sort candidates by `case_id`, and select within each intended class or language-intent stratum. Assert that every calibration and evaluation ID is disjoint and that the totals are exact.

- [ ] **Step 4: Implement provenance and duplicate validation**

Require every public record to contain canonical source, wrapper, resolved revision, split, original row ID, source label, local modification, license, citation, and checksum. Normalize states with `json.dumps(..., sort_keys=True, ensure_ascii=False)` and reject exact normalized duplicates. Record near-duplicate clusters using `difflib.SequenceMatcher` at threshold `0.78` without silently deleting cases.

- [ ] **Step 5: Run provenance and duplicate tests**

Run: `uv run --extra build pytest tests/test_dataset_provenance.py tests/test_dataset_duplicates.py -q`

Expected: PASS.

---

### Task 4: Generate blind human-label packets

**Files:**
- Create: `datasets/v2/labeling.py`
- Create: `tests/test_labeling_packets.py`
- Create: `datasets/v2/AGENTS.md`

**Interfaces:**
- Produces `write_reviewer_packet(records: Sequence[Mapping[str, Any]], output: Path, reviewer_code: str) -> None`.
- Produces `validate_reviewer_export(payload: Mapping[str, Any], expected_case_ids: set[str]) -> None`.
- Produces `agreement_report(reviewer_a: Mapping[str, Any], reviewer_b: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]`.

- [ ] **Step 1: Write failing packet-leakage tests**

```python
from pathlib import Path


def test_packet_does_not_include_expected_or_source_labels(tmp_path: Path):
    path = write_test_packet(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "expected" not in text
    assert "source_label" not in text
    assert "Laya" not in text
    assert "Jev" not in text


def test_export_rejects_missing_case():
    with pytest.raises(ValueError, match="case"):
        validate_reviewer_export({"schema_version": 2, "labels": {}}, {"triage-0001"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra build pytest tests/test_labeling_packets.py -q`

Expected: FAIL because the packet generator does not exist.

- [ ] **Step 3: Implement self-contained packet generation**

Generate an HTML document containing only case ID, state, ordered questions, allowed answers, and neutral instructions. Use a fixed seeded order, local draft storage, keyboard navigation, and a download button that exports:

```json
{
  "schema_version": 2,
  "reviewer_code": "A",
  "labels": {
    "triage-0001": {"intent": "refund", "is_urgent": 1}
  }
}
```

Validate exact case-ID coverage, answer types, legal option labels, Boolean values, and score ranges on export.

- [ ] **Step 4: Implement agreement and disagreement output**

Compute raw agreement, nominal Cohen's kappa, quadratic weighted kappa for score labels, per-class confusion matrices, and per-language multilingual agreement. Generate an adjudication packet containing only disagreements and both reviewer labels.

- [ ] **Step 5: Add dataset v2 AGENTS rules**

Document that model names, old labels, and source labels are forbidden in blind packets; that the dataset cannot be sealed with unresolved disagreements; and that calibration cases are never used for headline accuracy.

- [ ] **Step 6: Run packet tests**

Run: `uv run --extra build pytest tests/test_labeling_packets.py -q`

Expected: PASS.

---

### Task 5: Add human export validation and seal the manifest

**Files:**
- Modify: `datasets/v2/labeling.py`
- Create: `tests/test_dataset_seal.py`
- Create: `datasets/v2/provenance.json`
- Create: `datasets/v2/manifest.sha256`

**Interfaces:**
- Produces `merge_adjudication(records: Sequence[Mapping[str, Any]], reviewer_a: Mapping[str, Any], reviewer_b: Mapping[str, Any], adjudication: Mapping[str, Any]) -> list[dict[str, Any]]`.
- Produces `seal_manifest(output_root: Path) -> str`.

- [ ] **Step 1: Write failing seal tests**

```python
def test_seal_requires_two_complete_exports_and_adjudication(tmp_path):
    with pytest.raises(ValueError, match="reviewer"):
        seal_manifest(tmp_path)


def test_manifest_digest_changes_when_question_order_changes(tmp_path):
    first = seal_test_manifest(tmp_path, order=["a", "b"])
    second = seal_test_manifest(tmp_path, order=["b", "a"])
    assert first != second
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra build pytest tests/test_dataset_seal.py -q`

Expected: FAIL because sealing is not implemented.

- [ ] **Step 3: Implement merge and seal**

Validate both exports against the exact manifest case IDs, require an adjudication entry for every disagreement, write the final expected values into the manifest records, run the complete dataset validator, write `provenance.json`, and compute the full manifest digest with `benchmark.manifest.manifest_digest`.

Write `manifest.sha256` as `<digest>  manifest.jsonl`.

- [ ] **Step 4: Run all dataset checks**

Run:

```bash
uv run --extra build pytest tests/test_dataset_counts.py tests/test_dataset_provenance.py tests/test_dataset_duplicates.py tests/test_labeling_packets.py tests/test_dataset_seal.py -q
```

Expected: PASS.

### Task 6: Execute the human review handoff

**Files:**
- Modify: `datasets/v2/labels/reviewer_a.json`
- Modify: `datasets/v2/labels/reviewer_b.json`
- Modify: `datasets/v2/labels/adjudication.json`
- Create: `datasets/v2/labels/agreement.json`
- Create: `datasets/v2/manifest.sha256`

**Interfaces:**
- Consumes `reviewer_a.html`, `reviewer_b.html`, and `adjudication.html`.
- Produces the two reviewer exports, the adjudication export, the agreement report, and the sealed manifest digest.

- [ ] **Step 1: Generate the blind packets**

Run:

```bash
uv run --extra build python tools/build_dataset_v2.py --output datasets/v2
uv run --extra build python -m datasets.v2.labeling --records datasets/v2/manifest.jsonl --output datasets/v2/labeling
```

Expected: Reviewer A and Reviewer B packets contain the same 1,190 case IDs in the same seeded order and contain no expected or source labels.

- [ ] **Step 2: Obtain both independent exports**

Give `reviewer_a.html` and `reviewer_b.html` to the two human reviewers. Save their downloaded files as `datasets/v2/labels/reviewer_a.json` and `datasets/v2/labels/reviewer_b.json`. Do not edit either export by hand.

- [ ] **Step 3: Generate agreement and adjudication packets**

Run:

```bash
uv run --extra build python -m datasets.v2.labeling --agreement \
  --reviewer-a datasets/v2/labels/reviewer_a.json \
  --reviewer-b datasets/v2/labels/reviewer_b.json \
  --output datasets/v2/labeling/adjudication.html \
  --report datasets/v2/labels/agreement.json
```

Expected: the agreement report includes raw agreement, nominal kappa, weighted score kappa, per-class confusion, per-language agreement, and every disagreement.

- [ ] **Step 4: Apply human adjudication**

Complete `datasets/v2/labels/adjudication.json` with the final label, reason, and adjudicator code for every disagreement. Do not resolve a disagreement by majority vote among model outputs.

- [ ] **Step 5: Seal the manifest**

Run:

```bash
uv run --extra build python -m datasets.v2.labeling --seal \
  --records datasets/v2/manifest.jsonl \
  --reviewer-a datasets/v2/labels/reviewer_a.json \
  --reviewer-b datasets/v2/labels/reviewer_b.json \
  --adjudication datasets/v2/labels/adjudication.json \
  --output datasets/v2
```

Expected: all dataset gates pass and `manifest.sha256` is written. A missing human export or unresolved disagreement must fail the command.

---

## Completion Gate

The dataset track is complete when both human exports and adjudication are present, all counts and splits match the approved specification, every public record has resolved provenance, no blind packet leaks labels, duplicate gates pass, and `manifest.sha256` reproduces from a clean environment.
