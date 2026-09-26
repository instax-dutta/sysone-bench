from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from .labeling import _packet_records, _read_manifest_jsonl

_DEFAULT_MODEL = "gpt-oss-120b"
_DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
_MAX_REQUEST_BYTES = 1_000_000
_MISSING = object()


def _chat_completions_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("assistant base URL must be a nonempty string")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("assistant base URL must be an HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("assistant base URL must not contain a query or fragment")
    return base_url.rstrip("/") + "/chat/completions"


def _question_type(question: Mapping[str, Any]) -> str:
    value = question.get("type")
    if not isinstance(value, str) or value not in {"choice", "noul", "score"}:
        raise ValueError("question type must be choice, noul, or score")
    return value


def _score_value(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("score answer must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise ValueError("score answer must be an integer")


def _question_options(question: Mapping[str, Any]) -> list[dict[str, Any]]:
    question_type = _question_type(question)
    if question_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("choice question criteria must be a nonempty object")
        return [
            {"value": value, "label": str(value), "description": str(description)}
            for value, description in criteria.items()
        ]
    if question_type == "noul":
        return [
            {"value": value, "label": str(value), "description": str(value)} for value in (0, 1)
        ]
    maximum = question.get("max_score")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise TypeError("score question max_score must be numeric")
    if not math.isfinite(float(maximum)) or float(maximum) < 0 or not float(maximum).is_integer():
        raise ValueError("score question max_score must be a nonnegative integer")
    maximum_int = int(maximum)
    values = list(range(maximum_int + 1))
    return [{"value": value, "label": str(value), "description": str(value)} for value in values]


def build_assistant_cases(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    packets = _packet_records(records)
    cases: list[dict[str, Any]] = []
    for packet in packets:
        questions: list[dict[str, Any]] = []
        for raw_question in packet["questions"]:
            question = dict(raw_question)
            question["options"] = _question_options(question)
            questions.append(question)
        cases.append(
            {
                "case_id": packet["case_id"],
                "state": packet["state"],
                "questions": questions,
            }
        )
    return cases


def validate_assistant_answer(question: Mapping[str, Any], answer: object) -> str | int:
    question_type = _question_type(question)
    if question_type == "choice":
        criteria = question.get("criteria")
        if (
            not isinstance(answer, str)
            or not isinstance(criteria, Mapping)
            or answer not in criteria
        ):
            raise ValueError("answer is not an allowed choice label")
        return answer
    if question_type == "noul":
        if isinstance(answer, bool) or not isinstance(answer, int) or answer not in (0, 1):
            raise ValueError("noul answer must be integer 0 or 1")
        return answer
    value = _score_value(answer)
    maximum = question.get("max_score")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise TypeError("score question max_score must be numeric")
    if value < 0 or value > int(maximum):
        raise ValueError("score answer is outside the allowed range")
    return value


def build_suggestion_request(
    question: Mapping[str, Any],
    state: Mapping[str, Any],
    previous_answer: object = _MISSING,
    feedback: str | None = None,
) -> dict[str, Any]:
    question_type = _question_type(question)
    question_payload = {
        "qid": question.get("qid"),
        "type": question_type,
        "instructions": question.get("instructions", ""),
        "options": _question_options(question),
    }
    user_payload: dict[str, Any] = {"state": dict(state), "question": question_payload}
    if previous_answer is not _MISSING:
        user_payload["previous_answer"] = previous_answer
    if feedback is not None:
        user_payload["feedback"] = feedback
    system = (
        "You assist a human reviewer with a classification dataset. "
        "Choose exactly one answer from the supplied options. "
        "Use only the state and question supplied. "
        "Return only a JSON object with one field named answer. "
        "Do not include explanations, markdown, confidence, or extra fields."
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 80,
        "stream": False,
    }


def _default_transport(request: urllib.request.Request) -> Mapping[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Cerebras request failed") from error
    if not isinstance(value, Mapping):
        raise TypeError("Cerebras returned an invalid response")
    return value


class CerebrasClient:
    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        transport: Callable[[urllib.request.Request], Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("CEREBRAS_API_KEY is required")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Cerebras model must be a nonempty string")
        self._api_key = api_key
        self.model = model
        self._endpoint = _chat_completions_url(base_url)
        self._transport = transport or _default_transport

    def suggest(
        self,
        question: Mapping[str, Any],
        state: Mapping[str, Any],
        previous_answer: object = _MISSING,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        payload = build_suggestion_request(question, state, previous_answer, feedback)
        payload["model"] = self.model
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sysone-bench-assistant/1.0",
            },
            method="POST",
        )
        try:
            response = self._transport(request)
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError("Cerebras request failed") from error
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Cerebras returned no answer")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise TypeError("Cerebras returned an invalid answer")
        message = first.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise TypeError("Cerebras returned an invalid answer")
        try:
            parsed = json.loads(cast(str, message["content"]))
        except json.JSONDecodeError as error:
            raise RuntimeError("Cerebras returned invalid JSON") from error
        if not isinstance(parsed, Mapping) or "answer" not in parsed:
            raise RuntimeError("Cerebras response has no answer field")
        answer = validate_assistant_answer(question, parsed["answer"])
        return {"answer": answer, "model": self.model}


def _opencode_output_text(output: str) -> str:
    parts: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "text" or not isinstance(event.get("part"), Mapping):
            continue
        text = event["part"].get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _opencode_json_object(output: str) -> Mapping[str, Any]:
    text = _opencode_output_text(output)
    start = text.find("{")
    if start < 0:
        raise RuntimeError("OpenCode returned no JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise RuntimeError("OpenCode returned invalid JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError("OpenCode returned an invalid JSON value")
    return value


class OpenCodeClient:
    def __init__(
        self,
        model: str = "opencode/mimo-v2.6-flash-free",
        runner: Callable[[list[str]], str] | None = None,
        timeout: int = 240,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenCode model must be a nonempty string")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            raise ValueError("OpenCode timeout must be a positive integer")
        self.model = model
        self._timeout = timeout
        self._runner = runner or self._run_command

    def _run_command(self, command: list[str]) -> str:
        try:
            completed = subprocess.run(
                command,
                cwd=Path(tempfile.gettempdir()),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("OpenCode request failed") from error
        if completed.returncode != 0:
            raise RuntimeError("OpenCode request failed")
        return completed.stdout

    def suggest(
        self,
        question: Mapping[str, Any],
        state: Mapping[str, Any],
        previous_answer: object = _MISSING,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        request = build_suggestion_request(question, state, previous_answer, feedback)
        messages = request["messages"]
        prompt = (
            f"{messages[0]['content']}\n"
            f"Input:\n{messages[1]['content']}\n"
            "Return only the JSON object."
        )
        command = ["opencode", "run", "--pure", "--format", "json", "-m", self.model, prompt]
        try:
            output = self._runner(command)
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError("OpenCode request failed") from error
        parsed = _opencode_json_object(output)
        if "answer" not in parsed:
            raise RuntimeError("OpenCode response has no answer field")
        answer = validate_assistant_answer(question, parsed["answer"])
        return {"answer": answer, "model": self.model}


def _draft_key(case_id: str, qid: str) -> str:
    return f"{case_id}::{qid}"


def _load_draft_entries(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("assistant draft is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError("assistant draft must be an object")
    if value.get("source") != "cerebras-assistant" or value.get("official_labels") is not False:
        raise ValueError("assistant draft has an invalid source marker")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise TypeError("assistant draft entries must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise TypeError(f"assistant draft entry {index} must be an object")
        case_id = raw.get("case_id")
        qid = raw.get("qid")
        if not isinstance(case_id, str) or not case_id or not isinstance(qid, str) or not qid:
            raise ValueError(f"assistant draft entry {index} identity is invalid")
        key = _draft_key(case_id, qid)
        if key in result:
            raise ValueError("assistant draft contains duplicate entries")
        result[key] = dict(raw)
    return result


def load_draft_answers(path: Path) -> dict[str, object]:
    return {
        key: entry["answer"]
        for key, entry in _load_draft_entries(path).items()
        if "answer" in entry
    }


def _write_draft(path: Path, entries: Mapping[str, Mapping[str, Any]], model: str) -> None:
    if path.name in {"reviewer_a.json", "reviewer_b.json", "adjudication.json", "manifest.sha256"}:
        raise ValueError("assistant output cannot use an official label filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": "cerebras-assistant",
        "official_labels": False,
        "model": model,
        "entries": [entries[key] for key in sorted(entries)],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prefill_one(
    client: Any,
    case: Mapping[str, Any],
    question: Mapping[str, Any],
    retries: int,
) -> tuple[str, dict[str, Any] | None]:
    key = _draft_key(str(case["case_id"]), str(question["qid"]))
    for attempt in range(retries):
        try:
            result = client.suggest(question, case["state"])
            if not isinstance(result, Mapping) or "answer" not in result:
                raise ValueError("provider returned no answer")
            answer = validate_assistant_answer(question, result["answer"])
            return key, {
                "answer": answer,
                "case_id": case["case_id"],
                "model": str(result.get("model", getattr(client, "model", "unknown"))),
                "qid": question["qid"],
                "question_type": question["type"],
                "status": "prefilled",
            }
        except (RuntimeError, TypeError, ValueError):
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    return key, None


def prefill_suggestions(
    records: Sequence[Mapping[str, Any]],
    client: Any,
    output_path: Path,
    *,
    workers: int = 1,
    checkpoint_every: int = 10,
    limit: int | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be a Path")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
        raise ValueError("retries must be a positive integer")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a nonnegative integer")
    cases = build_assistant_cases(records)
    items = [(case, question) for case in cases for question in case["questions"]]
    question_map = {(case["case_id"], question["qid"]): question for case, question in items}
    entries = _load_draft_entries(output_path)
    item_keys = {_draft_key(str(case["case_id"]), str(question["qid"])) for case, question in items}
    unknown = set(entries) - item_keys
    if unknown:
        raise ValueError("assistant draft contains entries outside the manifest")
    for key, entry in entries.items():
        case_id, qid = key.split("::", 1)
        question = question_map[(case_id, qid)]
        validate_assistant_answer(question, entry.get("answer"))
    pending = [
        (case, question)
        for case, question in items
        if _draft_key(str(case["case_id"]), str(question["qid"])) not in entries
    ]
    if limit is not None:
        pending = pending[:limit]
    results: Iterator[tuple[str, dict[str, Any] | None]]
    if workers == 1:
        results = (_prefill_one(client, item[0], item[1], retries) for item in pending)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(
            lambda item: _prefill_one(client, item[0], item[1], retries), pending
        )
    attempted = 0
    failed = 0
    try:
        for key, draft_entry in results:
            attempted += 1
            if draft_entry is None:
                failed += 1
            else:
                entries[key] = draft_entry
            if attempted % checkpoint_every == 0:
                _write_draft(output_path, entries, str(getattr(client, "model", "unknown")))
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    _write_draft(output_path, entries, str(getattr(client, "model", "unknown")))
    return {
        "total": len(items),
        "filled": len(entries),
        "attempted": attempted,
        "failed": failed,
        "path": str(output_path),
    }


def _html_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build_assistant_html(
    cases: Sequence[Mapping[str, Any]],
    model: str,
    initial_answers: Mapping[str, object] | None = None,
) -> str:
    data = _html_script(
        {
            "cases": list(cases),
            "initial_answers": dict(initial_answers or {}),
            "model": model,
        }
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label assistant</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; --bg: #0b1220; --surface: #111827; --raised: #172033; --border: #334155; --text: #e5e7eb; --muted: #a8b3c4; --accent: #60a5fa; --strong: #2563eb; --focus: #fbbf24; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; min-height: 100vh; color: var(--text); background: var(--bg); }}
header, main, footer {{ width: min(52rem, 100%); margin: 0 auto; padding: 1rem; }}
header {{ border-bottom: 1px solid var(--border); }}
h1, h2 {{ margin: 0 0 .6rem; }}
p {{ color: var(--muted); }}
.card {{ margin-top: 1rem; padding: 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: .5rem; }}
#question {{ font-size: 1.15rem; line-height: 1.45; }}
#state {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: .8rem; color: var(--text); background: var(--raised); border-radius: .35rem; }}
#options {{ display: grid; gap: .55rem; margin: 1rem 0; }}
label {{ display: block; padding: .7rem; background: var(--raised); border: 1px solid var(--border); border-radius: .35rem; cursor: pointer; }}
label:has(input:checked) {{ border-color: var(--accent); background: #16305a; }}
input {{ accent-color: var(--accent); margin-right: .55rem; }}
button {{ margin: .25rem; padding: .6rem .9rem; color: var(--text); background: var(--raised); border: 1px solid var(--border); border-radius: .35rem; cursor: pointer; }}
button:hover {{ background: var(--strong); border-color: var(--accent); }}
button:focus-visible, input:focus-visible {{ outline: 3px solid var(--focus); outline-offset: 2px; }}
button:disabled {{ cursor: not-allowed; opacity: .45; }}
#status {{ min-height: 1.5rem; }}
#progress {{ color: var(--accent); font-weight: 600; }}
</style>
</head>
<body>
<header>
<h1>Label assistant</h1>
<p>Draft only. This page stores approved suggestions separately and never writes official labels.</p>
<div id="progress"></div>
</header>
<main>
<section class="card">
<div id="question"></div>
<pre id="state"></pre>
<div id="options"></div>
<div id="status" role="status" aria-live="polite"></div>
</section>
<footer>
<button id="back" type="button">Back</button>
<button id="restart" type="button">Restart</button>
<button id="repass" type="button">Re-pass</button>
<button id="approve" type="button">Approve</button>
<button id="download" type="button">Download assistant draft</button>
</footer>
</main>
<script id="assistant-data" type="application/json">{data}</script>
<script>
(function () {{
  const data = JSON.parse(document.getElementById("assistant-data").textContent);
  const items = [];
  data.cases.forEach(function (item) {{
    item.questions.forEach(function (question) {{
      items.push({{case: item, question: question}});
    }});
  }});
  const answerKey = "cerebras-assistant-answers-v1";
  const approvedKey = "cerebras-assistant-approved-v1";
  const answers = Object.assign({{}}, data.initial_answers || {{}}, JSON.parse(localStorage.getItem(answerKey) || "{{}}"));
  const approved = JSON.parse(localStorage.getItem(approvedKey) || "{{}}");
  const question = document.getElementById("question");
  const state = document.getElementById("state");
  const options = document.getElementById("options");
  const status = document.getElementById("status");
  const progress = document.getElementById("progress");
  const back = document.getElementById("back");
  const restart = document.getElementById("restart");
  const repass = document.getElementById("repass");
  const approve = document.getElementById("approve");
  const download = document.getElementById("download");
  let index = Number(localStorage.getItem("cerebras-assistant-index-v1") || 0);
  let busy = false;

  function key(item) {{
    return item.case.case_id + "::" + item.question.qid;
  }}

  function save() {{
    localStorage.setItem(answerKey, JSON.stringify(answers));
    localStorage.setItem(approvedKey, JSON.stringify(approved));
    localStorage.setItem("cerebras-assistant-index-v1", String(index));
  }}

  function currentAnswer() {{
    return answers[key(items[index])];
  }}

  function render() {{
    if (index < 0 || index >= items.length) {{
      question.textContent = "All questions approved";
      state.textContent = "Download the assistant draft when you are ready.";
      options.replaceChildren();
      progress.textContent = items.length + " of " + items.length;
      status.textContent = "";
      back.disabled = true;
      repass.disabled = true;
      approve.disabled = true;
      return;
    }}
    const item = items[index];
    const answer = currentAnswer();
    question.textContent = item.question.instructions;
    state.textContent = JSON.stringify(item.case.state, null, 2);
    options.replaceChildren();
    item.question.options.forEach(function (option) {{
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "assistant-answer";
      input.value = String(option.value);
      input.checked = answer !== undefined && String(answer) === String(option.value);
      input.addEventListener("change", function () {{
        answers[key(item)] = option.value;
        save();
        status.textContent = "Suggestion selected. Approve or re-pass.";
      }});
      label.appendChild(input);
      label.appendChild(document.createTextNode(" " + option.label + " - " + option.description));
      options.appendChild(label);
    }});
    progress.textContent = "Question " + (index + 1) + " of " + items.length;
    status.textContent = answer === undefined ? "Getting a suggestion..." : "Suggestion ready.";
    back.disabled = index === 0 || busy;
    repass.disabled = busy;
    approve.disabled = busy || answer === undefined;
  }}

  async function suggest(repassing) {{
    if (busy || index < 0 || index >= items.length) return;
    busy = true;
    status.textContent = repassing ? "Re-passing..." : "Getting GPT suggestion...";
    render();
    const item = items[index];
    const previous = currentAnswer();
    try {{
      const response = await fetch("/api/suggest", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          case_id: item.case.case_id,
          qid: item.question.qid,
          previous_answer: previous === undefined ? null : previous,
          feedback: repassing ? "The previous suggestion was not correct. Re-evaluate independently and choose a different answer when appropriate." : null
        }})
      }});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Suggestion failed");
      answers[key(item)] = payload.answer;
      save();
      status.textContent = "Suggestion ready.";
    }} catch (error) {{
      status.textContent = error instanceof Error ? error.message : "Suggestion failed";
    }} finally {{
      busy = false;
      render();
    }}
  }}

  restart.addEventListener("click", function () {{
    Object.keys(answers).forEach(function (key) {{ delete answers[key]; }});
    Object.keys(approved).forEach(function (key) {{ delete approved[key]; }});
    index = 0;
    save();
    render();
  }});
  back.addEventListener("click", function () {{
    if (index > 0) {{ index -= 1; save(); render(); if (currentAnswer() === undefined) suggest(false); }}
  }});
  repass.addEventListener("click", function () {{ suggest(true); }});
  approve.addEventListener("click", function () {{
    const answer = currentAnswer();
    if (answer === undefined) return;
    const item = items[index];
    approved[key(item)] = {{case_id: item.case.case_id, qid: item.question.qid, answer: answer, question_type: item.question.type, model: data.model}};
    index += 1;
    save();
    render();
    if (index < items.length && currentAnswer() === undefined) suggest(false);
  }});
  download.addEventListener("click", function () {{
    const payload = {{schema_version: 1, source: "cerebras-assistant", official_labels: false, model: data.model, entries: Object.values(approved)}};
    const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{type: "application/json"}});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "assistant_confirmed.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }});
  render();
  if (index < items.length && currentAnswer() === undefined) suggest(false);
}}());
</script>
</body>
</html>
"""


class _AssistantServer(ThreadingHTTPServer):
    cases: dict[str, dict[str, Any]]
    questions: dict[tuple[str, str], dict[str, Any]]
    html: str
    client: Any


class _AssistantHandler(BaseHTTPRequestHandler):
    server: _AssistantServer

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", self.server.html.encode("utf-8"))
            return
        if path == "/health":
            self._json(200, {"status": "ok", "questions": len(self.server.questions)})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/suggest":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid request"})
            return
        if length < 0 or length > _MAX_REQUEST_BYTES:
            self._json(413, {"error": "request too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._json(400, {"error": "invalid JSON"})
            return
        if not isinstance(payload, Mapping):
            self._json(400, {"error": "invalid request"})
            return
        case_id = payload.get("case_id")
        qid = payload.get("qid")
        if not isinstance(case_id, str) or not isinstance(qid, str):
            self._json(400, {"error": "case_id and qid are required"})
            return
        case = self.server.cases.get(case_id)
        question = self.server.questions.get((case_id, qid))
        if case is None or question is None:
            self._json(404, {"error": "question not found"})
            return
        previous = payload.get("previous_answer")
        feedback = payload.get("feedback")
        if feedback is not None and not isinstance(feedback, str):
            self._json(400, {"error": "feedback must be text"})
            return
        try:
            result = self.server.client.suggest(
                question,
                case["state"],
                previous_answer=previous,
                feedback=feedback,
            )
        except RuntimeError:
            self._json(503, {"error": "assistant provider is unavailable"})
            return
        except (TypeError, ValueError) as error:
            self._json(502, {"error": str(error)})
            return
        self._json(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    records: Sequence[Mapping[str, Any]],
    client: Any,
    port: int = 8768,
    draft_path: Path | None = None,
) -> ThreadingHTTPServer:
    cases = build_assistant_cases(records)
    case_map = {case["case_id"]: case for case in cases}
    question_map = {
        (case["case_id"], question["qid"]): question
        for case in cases
        for question in case["questions"]
    }
    initial_answers = load_draft_answers(draft_path) if draft_path is not None else {}
    server = _AssistantServer(("127.0.0.1", port), _AssistantHandler)
    server.cases = case_map
    server.questions = question_map
    server.html = build_assistant_html(cases, client.model, initial_answers)
    server.client = client
    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datasets.v2.assistant")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--provider",
        choices=("opencode", "http"),
        default=os.environ.get("ASSISTANT_PROVIDER", "opencode"),
    )
    parser.add_argument("--model", default=os.environ.get("ASSISTANT_MODEL"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "ASSISTANT_BASE_URL",
            os.environ.get("CEREBRAS_BASE_URL", _DEFAULT_BASE_URL),
        ),
    )
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--prefill-workers", type=int, default=4)
    parser.add_argument("--prefill-limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    records = _read_manifest_jsonl(args.manifest)
    model = args.model or (
        "opencode/mimo-v2.6-flash-free" if args.provider == "opencode" else _DEFAULT_MODEL
    )
    try:
        client: Any
        if args.provider == "opencode":
            client = OpenCodeClient(model=model)
        else:
            client = CerebrasClient(
                os.environ.get("CEREBRAS_API_KEY", ""),
                model=model,
                base_url=args.base_url,
            )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    draft_path = args.draft
    if args.prefill:
        if draft_path is None:
            draft_path = args.manifest.with_name("assistant_prefill.json")
        result = prefill_suggestions(
            records,
            client,
            draft_path,
            workers=args.prefill_workers,
            limit=args.prefill_limit,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.prefill_only:
            return 0
    server = create_server(records, client, args.port, draft_path)
    print(f"assistant: http://127.0.0.1:{args.port}/", flush=True)
    print(f"model: {client.model}", flush=True)
    if draft_path is not None:
        print(f"draft: {draft_path}", flush=True)
    print("official labels: disabled", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
