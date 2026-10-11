"""Phase 3: prove an OpenAI-compatible endpoint supports everything the agent
needs (E1-E12). Uses app/llm_client.py's own normalization code, so this
script exercises the exact code path the app uses.

Usage:
    # llama.cpp / vLLM
    uv run python scripts/endpoint_check.py --base-url http://127.0.0.1:8080/v1 \
        --model qwen3.6-35b-a3b --reps 3 --label my-llamacpp

    # Ollama (needs llm.thinking_control=reasoning_effort; it ignores chat_template_kwargs)
    GM_LLM__THINKING_CONTROL=reasoning_effort python scripts/endpoint_check.py \
        --base-url http://127.0.0.1:11434/v1 --model qwen3.6:35b --reps 3 --label my-ollama

Read the JSON report, not just the PASS line: E3 asserts only that <think> tags do not leak
into content, which a server emitting NO reasoning satisfies trivially. The value that tells
you reasoning was actually captured is checks.E3.samples[].meta.reasoning_source -- "field" or
"inline_tags" mean captured, "none" means silently lost.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `from app...` when run as a script

from app.llm_client import LLMClient  # noqa: E402
from app.settings import REPO_ROOT, load_settings  # noqa: E402

SEARCH_VAULT_TOOL = {
    "type": "function",
    "function": {
        "name": "search_vault",
        "description": "Search the institutional document vault.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
    },
}

UPDATE_COMPLIANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_compliance",
        "description": "Update the compliance matrix for one requirement.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "item": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string", "enum": ["met", "partial", "missing"]},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "status", "evidence"],
                }
            },
            "required": ["item"],
        },
    },
}


@dataclass
class CheckOutcome:
    ok: bool
    detail: str
    sample: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _validate_tool_call(result, tool_schema: dict[str, Any]) -> CheckOutcome:
    if len(result.tool_calls) != 1:
        return CheckOutcome(False, f"expected exactly 1 tool call, got {len(result.tool_calls)}")
    call = result.tool_calls[0]
    if call.name != tool_schema["function"]["name"]:
        return CheckOutcome(False, f"wrong tool name: {call.name}")
    if call.arguments is None:
        return CheckOutcome(False, f"args did not parse as JSON: {call.parse_error}", call.arguments_raw)
    try:
        jsonschema.validate(call.arguments, tool_schema["function"]["parameters"])
    except jsonschema.ValidationError as exc:
        return CheckOutcome(False, f"args failed schema validation: {exc.message}", call.arguments_raw)
    return CheckOutcome(True, f"tool_call={call.name}({call.arguments})", call.arguments_raw)


def check_e1(client: LLMClient, model: str) -> CheckOutcome:
    models = client.list_models()
    ok = model in models
    return CheckOutcome(ok, f"models={models}", json.dumps(models)[:2000])


def check_e2(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat("drafter", [{"role": "user", "content": "Reply with the single word: ready"}])
    ok = bool(result.content.strip())
    return CheckOutcome(ok, f"content={result.content!r}", result.content[:2000])


def check_e3(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat(
        "orchestrator", [{"role": "user", "content": "Briefly think about 7+5, then answer with just the number."}]
    )
    ok = "<think>" not in result.content and "</think>" not in result.content
    return CheckOutcome(
        ok,
        f"reasoning_source={result.reasoning_source}, content={result.content!r}",
        result.content[:2000],
        meta={"reasoning_source": result.reasoning_source},
    )


def check_e4(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat("drafter", [{"role": "user", "content": "What is 2+2? Answer with just the number."}])
    ok = not result.reasoning
    return CheckOutcome(ok, f"reasoning={result.reasoning!r}", result.content[:2000])


def check_e5(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat(
        "orchestrator",
        [{"role": "user", "content": "Use the search_vault tool to search for 'prior outcomes'. You must call the tool."}],
        tools=[SEARCH_VAULT_TOOL],
    )
    return _validate_tool_call(result, SEARCH_VAULT_TOOL)


def check_e6(client: LLMClient, model: str) -> CheckOutcome:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Use the search_vault tool to search for 'prior outcomes'. You must call the tool."}
    ]
    result = client.chat("orchestrator", messages, tools=[SEARCH_VAULT_TOOL])
    if not result.tool_calls:
        return CheckOutcome(False, "no tool call made on first turn")
    call = result.tool_calls[0]
    messages.append(
        {
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments_raw}}],
        }
    )
    messages.append(
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps({"hits": [{"doc_id": "ZX-4471", "snippet": "token ZX-4471 present"}]})}
    )
    final = client.chat("orchestrator", messages, tools=[SEARCH_VAULT_TOOL])
    ok = "ZX-4471" in final.content
    return CheckOutcome(ok, f"final_content={final.content!r}", final.content[:2000])


def check_e7(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat(
        "orchestrator",
        [
            {
                "role": "user",
                "content": "Call update_compliance to mark requirement R-1 as status 'met' with evidence ['doc-a','doc-b']. You must call the tool.",
            }
        ],
        tools=[UPDATE_COMPLIANCE_TOOL],
    )
    return _validate_tool_call(result, UPDATE_COMPLIANCE_TOOL)


def check_e8(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat(
        "orchestrator",
        [{"role": "user", "content": "Use the search_vault tool to search for 'prior outcomes'. You must call the tool."}],
        tools=[SEARCH_VAULT_TOOL],
        stream=True,
    )
    return _validate_tool_call(result, SEARCH_VAULT_TOOL)


def check_e9(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat(
        "orchestrator",
        [{"role": "user", "content": "What is 2+2? Just answer with the number, no tools needed."}],
        tools=[SEARCH_VAULT_TOOL],
    )
    ok = len(result.tool_calls) == 0
    return CheckOutcome(ok, f"tool_calls={len(result.tool_calls)}, content={result.content!r}", result.content[:2000])


def check_e10(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat("drafter", [{"role": "user", "content": "Say hello in one short sentence."}])
    ok = not result.usage.estimated
    detail = f"prompt_tokens={result.usage.prompt_tokens}, completion_tokens={result.usage.completion_tokens}, estimated={result.usage.estimated}"
    return CheckOutcome(ok, detail, detail)


def check_e11(client: LLMClient, model: str) -> CheckOutcome:
    result = client.chat(
        "drafter",
        [{"role": "user", "content": "Write a detailed 400-word overview of workforce development in rural community colleges."}],
        stream=True,
    )
    ok = result.usage.completion_tokens > 0 and result.timings.total_s > 0
    tps = result.timings.decode_tps
    tps_str = f"{tps:.2f}" if tps else "0"
    detail = (
        f"completion_tokens={result.usage.completion_tokens}, total_s={result.timings.total_s:.2f}, "
        f"decode_tps={tps_str}, ttft_s={result.timings.ttft_s}"
    )
    return CheckOutcome(ok, detail, detail, meta={"decode_tps": tps, "ttft_s": result.timings.ttft_s})


def check_e12(client: LLMClient, model: str) -> CheckOutcome:
    filler = ("The quarterly report discusses regional workforce trends and " * 40).strip()
    long_doc = "\n".join(filler for _ in range(30))
    planted = "\n\nThe secret code word is ZX-9981.\n\n"
    prompt = long_doc + planted + long_doc + "\n\nWhat is the secret code word mentioned above? Reply with just the code word."
    result = client.chat("drafter", [{"role": "user", "content": prompt}])
    ok = "ZX-9981" in result.content
    detail = f"prompt_tokens={result.usage.prompt_tokens}, content={result.content!r}"
    return CheckOutcome(ok, detail, result.content[:2000])


CHECKS: list[tuple[str, str, bool, Callable[[LLMClient, str], CheckOutcome]]] = [
    ("E1", "Models list", True, check_e1),
    ("E2", "Plain completion", True, check_e2),
    ("E3", "Thinking separation", True, check_e3),
    ("E4", "Thinking off", True, check_e4),
    ("E5", "Single tool call", True, check_e5),
    ("E6", "Tool round trip", True, check_e6),
    ("E7", "Nested object args", True, check_e7),
    ("E8", "Streaming tool call", True, check_e8),
    ("E9", "No spurious tool call", True, check_e9),
    ("E10", "Usage fields", False, check_e10),
    ("E11", "Decode speed", False, check_e11),
    ("E12", "Long context", False, check_e12),
]

_STRICT_3_OF_3 = {"E1", "E2", "E4"}


def _min_required(check_id: str, reps: int) -> int:
    if check_id in _STRICT_3_OF_3:
        return reps
    return max(1, reps - 1)


def _server_version(base_url: str, model: str) -> str:
    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            timeout=30,
        )
        return resp.json().get("system_fingerprint", "unknown")
    except (httpx.HTTPError, ValueError, KeyError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    settings = load_settings()
    settings.llm.base_url = args.base_url
    settings.llm.model = args.model
    client = LLMClient(settings)

    report: dict[str, Any] = {
        "base_url": args.base_url,
        "model": args.model,
        "label": args.label,
        "server_version": _server_version(args.base_url, args.model),
        "reps": args.reps,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    overall_ok = True
    print(f"{'ID':<4} {'Name':<24} {'Req':<5} {'Pass/Reps':<10} {'Result'}")
    print("-" * 70)

    for check_id, name, required, fn in CHECKS:
        outcomes: list[CheckOutcome] = []
        for _ in range(args.reps):
            try:
                outcomes.append(fn(client, args.model))
            except Exception as exc:  # noqa: BLE001 -- a check failure must not abort the run
                outcomes.append(CheckOutcome(False, f"exception: {exc}"))
        passes = sum(1 for o in outcomes if o.ok)
        min_req = _min_required(check_id, args.reps)
        check_passed = (not required) or passes >= min_req
        if required and not check_passed:
            overall_ok = False

        print(f"{check_id:<4} {name:<24} {'yes' if required else 'no':<5} {passes}/{args.reps:<8} {'PASS' if check_passed else 'FAIL'}")

        report["checks"][check_id] = {
            "name": name,
            "required": required,
            "passes": passes,
            "reps": args.reps,
            "min_required": min_req,
            "passed": check_passed,
            "samples": [{"ok": o.ok, "detail": o.detail, "sample": o.sample[:2000], "meta": o.meta} for o in outcomes],
        }

    report["overall_passed"] = overall_ok

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"endpoint_check_{args.label}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
