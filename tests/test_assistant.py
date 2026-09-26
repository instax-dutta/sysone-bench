from __future__ import annotations

import json
from typing import Any

import pytest

from datasets.v2.assistant import (
    CerebrasClient,
    OpenCodeClient,
    build_assistant_cases,
    build_assistant_html,
    build_suggestion_request,
    load_draft_answers,
    prefill_suggestions,
    validate_assistant_answer,
)


def _record() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_version": "2.0.0",
        "suite_id": "fixture",
        "case_id": "case-1",
        "order_index": 0,
        "split": "evaluation",
        "state": {"text": "A sample state", "language": "en"},
        "questions": [
            {
                "qid": "choice",
                "type": "choice",
                "instructions": "Choose one.",
                "criteria": {"yes": "Yes", "no": "No"},
            },
            {"qid": "flag", "type": "noul", "instructions": "Answer true or false."},
            {
                "qid": "score",
                "type": "score",
                "instructions": "Choose a level.",
                "criteria": ["low", "high"],
                "max_score": 1,
            },
        ],
        "expected": {"choice": "yes", "flag": 1, "score": 1},
        "provenance_id": "secret-provenance",
        "label_provenance": "secret-label",
    }


def test_build_assistant_cases_removes_labels_and_builds_options() -> None:
    cases = build_assistant_cases([_record()])

    assert len(cases) == 1
    case = cases[0]
    assert case["case_id"] == "case-1"
    assert "expected" not in case
    assert "provenance_id" not in case
    assert "label_provenance" not in case
    assert [question["qid"] for question in case["questions"]] == ["choice", "flag", "score"]
    assert case["questions"][0]["options"] == [
        {"value": "yes", "label": "yes", "description": "Yes"},
        {"value": "no", "label": "no", "description": "No"},
    ]
    assert case["questions"][1]["options"] == [
        {"value": 0, "label": "0", "description": "0"},
        {"value": 1, "label": "1", "description": "1"},
    ]


def test_validate_assistant_answer_enforces_question_domain() -> None:
    choice = {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}
    noul = {"type": "noul"}
    score = {"type": "score", "max_score": 2}

    assert validate_assistant_answer(choice, "yes") == "yes"
    assert validate_assistant_answer(noul, 1) == 1
    assert validate_assistant_answer(score, 2) == 2
    for question, answer in ((choice, "maybe"), (noul, 2), (score, 3), (score, True)):
        with pytest.raises((TypeError, ValueError)):
            validate_assistant_answer(question, answer)


def test_suggestion_request_contains_no_labels_and_repass_feedback() -> None:
    case = build_assistant_cases([_record()])[0]
    question = case["questions"][0]
    request = build_suggestion_request(
        question,
        case["state"],
        previous_answer="no",
        feedback="The previous answer was wrong; reconsider independently.",
    )

    encoded = json.dumps(request)
    assert "expected" not in encoded
    assert "provenance" not in encoded
    assert "The previous answer was wrong" in encoded
    assert request["response_format"] == {"type": "json_object"}


def test_cerebras_client_parses_valid_structured_answer() -> None:
    requests: list[Any] = []

    def transport(request: Any) -> dict[str, Any]:
        requests.append(request)
        return {
            "choices": [
                {"message": {"content": json.dumps({"answer": "no"})}},
            ]
        }

    client = CerebrasClient("test-key", model="gpt-oss-120b", transport=transport)
    result = client.suggest(
        {"qid": "choice", "type": "choice", "criteria": {"yes": "Yes", "no": "No"}},
        {"text": "sample"},
    )

    assert result == {"answer": "no", "model": "gpt-oss-120b"}
    request = requests[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.get_header("User-agent") == "sysone-bench-assistant/1.0"
    payload = json.loads(request.data)
    assert payload["model"] == "gpt-oss-120b"
    assert payload["response_format"] == {"type": "json_object"}


def test_cerebras_client_uses_configured_base_url_and_json_response() -> None:
    requests: list[Any] = []

    def transport(request: Any) -> dict[str, Any]:
        requests.append(request)
        return {"choices": [{"message": {"content": json.dumps({"answer": 1})}}]}

    client = CerebrasClient(
        "test-key",
        base_url="https://example.test/v1",
        transport=transport,
    )
    client.suggest({"qid": "flag", "type": "noul"}, {})

    request = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://example.test/v1/chat/completions"
    assert payload["stream"] is False


def test_opencode_client_parses_mimo_json_events() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str]) -> str:
        commands.append(command)
        return "\n".join(
            [
                json.dumps({"type": "text", "part": {"text": '{"answer":"no"}'}}),
                json.dumps({"type": "step_finish", "part": {"reason": "stop"}}),
            ]
        )

    client = OpenCodeClient(model="opencode/mimo-v2.6-flash-free", runner=runner)
    result = client.suggest(
        {"qid": "choice", "type": "choice", "criteria": {"yes": "Yes", "no": "No"}},
        {"text": "sample"},
    )

    assert result == {"answer": "no", "model": "opencode/mimo-v2.6-flash-free"}
    assert commands[0][0:5] == ["opencode", "run", "--pure", "--format", "json"]
    assert "expected" not in commands[0][-1]


def test_cerebras_client_rejects_invalid_structured_answer() -> None:
    def transport(request: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": json.dumps({"answer": 3})}}]}

    client = CerebrasClient("test-key", transport=transport)

    with pytest.raises(ValueError, match="answer"):
        client.suggest({"qid": "score", "type": "score", "max_score": 2}, {})


def test_assistant_html_contains_dark_approve_repass_controls() -> None:
    html = build_assistant_html(build_assistant_cases([_record()]), "gpt-oss-120b")

    assert "Approve" in html
    assert "Re-pass" in html
    assert "assistant_confirmed.json" in html
    assert "official labels" in html
    assert "color-scheme: dark" in html
    assert "expected" not in html
    assert "/api/suggest" in html


def test_assistant_case_order_is_deterministic() -> None:
    first = build_assistant_cases([_record()])
    second = build_assistant_cases([_record()])

    assert first == second


def test_prefill_suggestions_checkpoints_legal_draft_entries(tmp_path: Any) -> None:
    calls: list[tuple[str, str]] = []

    class FakeClient:
        model = "fake-model"

        def suggest(self, question: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
            calls.append((question["qid"], state["text"]))
            value = (
                "yes" if question["type"] == "choice" else 1 if question["type"] == "noul" else 0
            )
            return {"answer": value, "model": self.model}

    output = tmp_path / "assistant_prefill.json"
    result = prefill_suggestions([_record()], FakeClient(), output, checkpoint_every=1)

    assert result["total"] == 3
    assert result["filled"] == 3
    assert result["failed"] == 0
    assert len(calls) == 3
    answers = load_draft_answers(output)
    assert answers["case-1::choice"] == "yes"
    assert answers["case-1::flag"] == 1
    assert "expected" not in output.read_text(encoding="utf-8")


def test_prefill_suggestions_resumes_without_repeating_completed_items(tmp_path: Any) -> None:
    calls: list[str] = []

    class FakeClient:
        model = "fake-model"

        def suggest(self, question: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
            calls.append(question["qid"])
            return {"answer": "yes" if question["type"] == "choice" else 0, "model": self.model}

    output = tmp_path / "assistant_prefill.json"
    prefill_suggestions([_record()], FakeClient(), output, limit=1)
    result = prefill_suggestions([_record()], FakeClient(), output)

    assert result["filled"] == 3
    assert calls == ["choice", "flag", "score"]
    assert len(load_draft_answers(output)) == 3


def test_assistant_html_embeds_prefilled_answers() -> None:
    cases = build_assistant_cases([_record()])
    html = build_assistant_html(cases, "fake-model", {"case-1::choice": "yes"})

    assert "initial_answers" in html
    assert "case-1::choice" in html
