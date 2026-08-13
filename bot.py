"""Data-analyst Telegram bot for TDS Project 1.

Key fixes:
- gpt-5.6-sol is called with reasoning_effort=none when using Chat Completions tools.
- Removed the unsafe hardcoded Mexico/WHO answers.
- Correct WHO GHO sex dimension: SEX_BTSX, not BTSX.
- Validates tool output and retries bad queries instead of silently accepting empty data.
- Uses the latest user message plus recent chat context, and always emits one JSON object.
"""

import contextlib
import io
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("AIPIPE_TOKEN", "")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Put the compatible model first. AIPipe's gpt-5.6-sol rejects tool calls unless
# reasoning_effort is explicitly set to none on the Chat Completions endpoint.
MODELS = [m.strip() for m in os.environ.get(
    "MODELS", "gpt-5.6-sol,gpt-5.5,gpt-4o-mini"
).split(",") if m.strip()]

MAX_AGENT_STEPS = 6
PY_TIMEOUT = 60
REQUEST_TIMEOUT = 180
ANSWER_BUDGET = 210

_log_lock = threading.Lock()
_hist_lock = threading.Lock()
_histories: dict[int, list[dict[str, str]]] = {}


def log_event(**fields: Any) -> None:
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_python(code: str) -> str:
    """Run analyst code in a bounded thread and return stdout/stderr."""
    out = io.StringIO()
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, {"__name__": "__main__"})
            result["ok"] = True
        except Exception:
            result["ok"] = False
            traceback.print_exc(file=out)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(PY_TIMEOUT)
    if thread.is_alive():
        return f"ERROR: code timed out after {PY_TIMEOUT}s"
    text = out.getvalue()
    if not text:
        return "ERROR: code printed no output. Add print() and retry."
    return text[-12000:]


TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run Python for data analysis. requests, pandas, numpy, bs4, "
            "openpyxl and lxml are installed; network access is available. "
            "Always call response.raise_for_status() and always print structured results."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}]

SYSTEM_PROMPT = r"""You are an expert data analyst answering Telegram questions.

Use the LATEST user message as the task; earlier messages are context for multi-turn questions.
If the question embeds data or names a public dataset, use run_python to fetch and compute the answer. Never guess a value that can be computed. Always call raise_for_status() after requests.

Your final response must be exactly one JSON object with exactly these top-level keys:
{"answer": <shape requested by the user>, "log_url": "LOG_URL"}
Replace LOG_URL only with the literal placeholder LOG_URL. The server replaces it with the public URL. Do not use markdown or prose outside the object. Preserve the requested answer keys, nesting, types, rounding, and ordering when possible.

For WHO GHO indicator WHOSIS_000001, use https://ghoapi.azureedge.net/api/WHOSIS_000001. Rows are in response['value']; filter both sexes with Dim1 == 'SEX_BTSX' (not 'BTSX'), use SpatialDim ISO3 codes, TimeDim years, and NumericValue. If an OData filter returns zero rows, fetch the endpoint and filter locally instead of inventing an answer.

Do not use any hardcoded country answer. In particular, never answer Mexico unless the actual requested countries and computed data justify it.
"""


def chat_completion(messages: list[dict[str, Any]], model: str, use_tools: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages}
    if use_tools:
        body["tools"] = TOOLS
    # This is the exact compatibility fix indicated by the Render log.
    if model == "gpt-5.6-sol":
        body["reasoning_effort"] = "none"
    elif not model.startswith("gpt-5"):
        body["temperature"] = 0

    response = requests.post(
        f"{MODEL_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "tds-data-analyst-bot/2.0",
        },
        json=body,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code} from {model}: {response.text[:800]}")
    payload = response.json()
    return payload["choices"][0]["message"]


def completion_with_fallback(messages: list[dict[str, Any]], use_tools: bool, chat_id: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for model in MODELS:
        started = time.monotonic()
        try:
            msg = chat_completion(messages, model, use_tools)
            log_event(event="llm_success", model=model, chat_id=chat_id,
                      elapsed_ms=round((time.monotonic() - started) * 1000))
            return msg
        except Exception as exc:
            last_error = exc
            log_event(event="llm_model_failed", model=model, chat_id=chat_id, error=str(exc))
    raise last_error or RuntimeError("no models configured")


def extract_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        return None
    return None


def solve(chat_id: int, question: str) -> str:
    log_event(event="question", chat_id=chat_id, text=question)
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        del history[:-20]
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    deadline = time.monotonic() + ANSWER_BUDGET
    final_text = ""
    for step in range(MAX_AGENT_STEPS):
        timed_out = time.monotonic() >= deadline
        if timed_out:
            messages.append({"role": "user", "content": "Return your best final JSON now. No more tools."})
        try:
            msg = completion_with_fallback(messages, use_tools=not timed_out, chat_id=chat_id)
        except Exception as exc:
            log_event(event="llm_error_final", chat_id=chat_id, error=str(exc))
            break

        tool_calls = msg.get("tool_calls") or []
        if tool_calls and not timed_out:
            messages.append(msg)
            for tool_call in tool_calls:
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                    code = args.get("code", "")
                except Exception:
                    code = ""
                if not code:
                    output = "ERROR: missing Python code"
                else:
                    log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:5000])
                    output = run_python(code)
                log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:5000])
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": output})
            continue
        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text) or {"answer": final_text.strip()[:2000] or "unable to determine"}
    if "answer" not in obj:
        obj = {"answer": obj}
    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    log_event(event="answer", chat_id=chat_id, reply=reply)
    return reply


def tg(method: str, **params: Any) -> dict[str, Any]:
    response = requests.post(f"{TG_API}/{method}", json=params, timeout=65)
    response.raise_for_status()
    return response.json()


def handle_update(update: dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message")
    if not message:
        return
    text = message.get("text") or message.get("caption") or ""
    if not text:
        return
    chat_id = message["chat"]["id"]
    try:
        reply = solve(chat_id, text)
    except Exception:
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL}, separators=(",", ":"))
    tg("sendMessage", chat_id=chat_id, text=reply)


def poll_loop() -> None:
    log_event(event="startup", base_url=BASE_URL, models=MODELS)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            response = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            )
            response.raise_for_status()
            for update in response.json().get("result", []):
                offset = update["update_id"] + 1
                pool.submit(handle_update, update)
        except Exception as exc:
            log_event(event="poll_error", error=str(exc))
            time.sleep(5)


def keepwarm_loop() -> None:
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


app = FastAPI()


@app.on_event("startup")
def start() -> None:
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "models": MODELS, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}
