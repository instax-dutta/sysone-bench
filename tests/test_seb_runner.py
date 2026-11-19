"""Offline checks for the seb-9b option-key readout. No weights, no network."""

from __future__ import annotations

import json

import pytest

from runners.vendor.seb import (
    CHOICE_KEYS,
    SYSTEM_PROMPT,
    Seb9BRunner,
    _key_ids,
    _options,
    _project,
    _prompt,
)


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert messages[0]["content"] == SYSTEM_PROMPT
        return "|".join(message["content"] for message in messages)

    def encode(self, key, *, add_special_tokens=False):
        return [ord(key[0])]


def test_options_follow_the_model_card_key_scheme() -> None:
    choice = _options(
        {"type": "choice", "criteria": {"refund": "money back", "other": "anything else"}}, "q1"
    )
    assert choice == {"A": "refund: money back", "B": "other: anything else"}
    assert _options({"type": "noul", "instructions": "Refund?"}, "q2") == {"Y": "yes", "N": "no"}
    score = _options({"type": "score", "criteria": ["low", "mid", "high"]}, "q3")
    assert score == {"0": "low", "1": "mid", "2": "high"}
    with pytest.raises(ValueError, match="exceed"):
        _options({"type": "choice", "criteria": {str(i): i for i in range(21)}}, "q4")
    with pytest.raises(ValueError, match="exceed"):
        _options({"type": "score", "criteria": list(range(6))}, "q5")


def test_prompt_carries_state_question_and_keys_with_thinking_off() -> None:
    question = {"type": "choice", "instructions": "What next?", "criteria": {"a": "A", "b": "B"}}
    prompt = _prompt(_Tokenizer(), {"message": "hi"}, question, {"A": "a", "B": "b"})
    assert prompt.startswith(SYSTEM_PROMPT + "|")
    payload = json.loads(prompt.split("|", 1)[1])
    assert payload["state"] == {"message": "hi"}
    assert payload["question"] == "What next?"
    assert payload["options"] == {"A": "a", "B": "b"}


def test_key_ids_require_single_token_keys() -> None:
    assert _key_ids(_Tokenizer(), {"Y": "yes", "N": "no"}, "q") == {"Y": 89, "N": 78}

    class _Split(_Tokenizer):
        def encode(self, key, *, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(ValueError, match="not one"):
        _key_ids(_Split(), {"Y": "yes"}, "q")


def test_projection_maps_keys_back_to_contract_answers() -> None:
    question = {"type": "choice", "criteria": {"refund": "money back", "other": "else"}}
    answer = _project("choice", question, {"A": "refund", "B": "other"},
                      {"A": 0.7, "B": 0.3}, "q")
    assert answer["choice"] == "refund"
    assert answer["probabilities"] == {"refund": 0.7, "other": 0.3}
    assert answer["confidence"] == 0.7
    assert _project("noul", {}, {"Y": "yes", "N": "no"}, {"Y": 0.9, "N": 0.1}, "q") == {
        "type": "noul", "noul": 0.9
    }
    score = _project("score", {}, {"0": "low", "1": "high"}, {"0": 0.2, "1": 0.8}, "q")
    assert score == {"type": "score", "score": 1.0, "confidence": 0.8}


def test_identity_is_pinned_to_full_revisions() -> None:
    assert Seb9BRunner.name == "seb-9b"
    assert len(Seb9BRunner.default_revision) == 40
    assert len(Seb9BRunner.base_revision) == 40
    assert Seb9BRunner.license_name == "apache-2.0"
    assert len(CHOICE_KEYS) == 20
