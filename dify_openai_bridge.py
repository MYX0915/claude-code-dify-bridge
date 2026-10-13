#!/usr/bin/env python3
"""Expose a small OpenAI-compatible facade for a Dify chat application.

The Dify application API is /v1/chat-messages, while LiteLLM expects an
OpenAI-compatible /v1/chat/completions provider. This adapter keeps the Dify
application key local and converts the request and response formats.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


DEFAULT_CONFIG = str(Path(__file__).resolve().with_name("xcg_ai_dify_agent.json"))
CONFIG_PATH = Path(os.environ.get("DIFY_CONFIG_JSON", DEFAULT_CONFIG))
MODEL_SELECTOR = os.environ.get("DIFY_APP_MODEL", "DeepSeek-v4-pro")
LOCAL_API_KEY = os.environ.get("DIFY_BRIDGE_API_KEY", "sk-dify-local-bridge")
DEFAULT_USER = os.environ.get("DIFY_USER", "claude-code")
DIFY_BASE_URL_OVERRIDE = os.environ.get("DIFY_BASE_URL")
DIFY_APP_API_KEY = os.environ.get("DIFY_APP_API_KEY")
DIFY_PARAMETER_CHECK_INTERVAL = max(60, int(os.environ.get("DIFY_PARAMETER_CHECK_INTERVAL", "3600")))
logger = logging.getLogger("dify_bridge")
logging.basicConfig(
    level=os.environ.get("DIFY_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s dify_bridge %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_dify_config() -> tuple[str, str, str]:
    rows: list[dict[str, Any]] = []
    if CONFIG_PATH.exists():
        rows = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    row = next((x for x in rows if x.get("model") == MODEL_SELECTOR), None)
    if row is None:
        row = next((x for x in rows if x.get("name") == MODEL_SELECTOR), None)
    base_url = DIFY_BASE_URL_OVERRIDE or (row or {}).get("base_url")
    api_key = DIFY_APP_API_KEY or (row or {}).get("api_key")
    if not base_url or not api_key:
        raise RuntimeError(
            "缺少 Dify 配置：请设置 DIFY_BASE_URL 和 DIFY_APP_API_KEY，"
            f"或提供包含模型 {MODEL_SELECTOR} 的本地配置文件 {CONFIG_PATH}"
        )
    return str(base_url).rstrip("/"), str(api_key), (row or {}).get("model", MODEL_SELECTOR)


DIFY_BASE_URL, DIFY_API_KEY, DIFY_MODEL = load_dify_config()
DIFY_PARAMETER_KEYS: set[str] = set()
DIFY_PARAMETER_TASK: asyncio.Task[None] | None = None
app = FastAPI(title="Dify OpenAI Bridge")


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                elif item.get("type") in {"tool_result", "function_call_output"}:
                    parts.append(f"[TOOL_RESULT]\n{content_to_text(item.get('content') or item.get('output'))}")
        return "\n".join(parts)
    return "" if content is None else str(content)


def messages_to_query(messages: list[dict[str, Any]]) -> str:
    # Dify owns the application's system prompt. Preserve the Claude/LiteLLM
    # conversation as plain text so the Dify chat app can answer with context.
    blocks: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        text = content_to_text(msg.get("content"))
        if text:
            blocks.append(f"[{role}]\n{text}")
    return "\n\n".join(blocks) or "你好"


def parse_dsml_tool_calls(answer: str) -> list[dict[str, Any]]:
    """Parse the DSML tool markup emitted by some DeepSeek deployments."""
    normalized = answer.replace("｜｜DSML｜｜", "").replace("||DSML||", "")
    normalized = re.sub(r"<\|\s*\|\s*DSML\s*\|\s*\|\s*[^>]*>", "", normalized, flags=re.I)
    normalized = re.sub(r"</\|\s*\|\s*DSML\s*\|\s*\|\s*[^>]*>", "", normalized, flags=re.I)
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<invoke\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</invoke>", normalized, re.S | re.I):
        arguments: dict[str, Any] = {}
        for parameter in re.finditer(
            r"<parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</parameter>",
            match.group(2),
            re.S | re.I,
        ):
            value = html.unescape(parameter.group(2)).strip()
            parameter_name = re.sub(r"(?<!^)([A-Z])", r"_\1", parameter.group(1)).lower()
            try:
                arguments[parameter_name] = json.loads(value)
            except json.JSONDecodeError:
                arguments[parameter_name] = value
        calls.append({"name": match.group(1), "arguments": arguments, "call_id": "call_" + uuid.uuid4().hex})
    for match in re.finditer(r"<ToolUse\s+id=[\"']([^\"']+)[\"']\s+path=[\"']([^\"']+)[\"']\s*/?>", normalized, re.S | re.I):
        tool_name = "Read" if match.group(1).lower() in {"read_file", "read"} else match.group(1)
        calls.append({"name": tool_name, "arguments": {"file_path": html.unescape(match.group(2))}, "call_id": "call_" + uuid.uuid4().hex})
    for match in re.finditer(r"<tool_call\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</tool_call>", normalized, re.S | re.I):
        raw_arguments = html.unescape(match.group(2)).strip()
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            logger.warning("无法解析工具参数，工具名=%s", match.group(1))
            arguments = {"raw": raw_arguments}
        if isinstance(arguments, dict):
            arguments = {
                re.sub(r"(?<!^)([A-Z])", r"_\1", str(key)).lower(): value
                for key, value in arguments.items()
            }
        else:
            arguments = {"input": arguments}
        calls.append({"name": match.group(1), "arguments": arguments, "call_id": "call_" + uuid.uuid4().hex})
    if ("<invoke" in normalized.lower() or "<tooluse" in normalized.lower() or "<tool_call" in normalized.lower()) and not calls:
        logger.warning("检测到未知工具标记格式，未能转换为工具调用")
    if calls:
        logger.info("已解析工具调用：%s", ", ".join(call["name"] for call in calls))
    return calls


def responses_input_to_query(payload: dict[str, Any]) -> str:
    """Convert the newer OpenAI Responses input shape to Dify plain text."""
    instructions = payload.get("instructions")
    items = payload.get("input")
    if isinstance(items, str):
        messages = [{"role": "user", "content": items}]
    elif isinstance(items, list):
        messages = []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}
                    )
                messages.append({"role": item.get("role", "user"), "content": content})
            elif isinstance(item, dict) and item.get("type") in {"input_text", "output_text", "text"}:
                messages.append({"role": "user", "content": item.get("text", "")})
            elif isinstance(item, dict) and item.get("type") == "function_call_output":
                messages.append({"role": "tool", "content": item.get("output", "")})
        if not messages:
            messages = [{"role": "user", "content": str(items)}]
    else:
        messages = []
    query = messages_to_query(messages)
    if instructions:
        query = f"[SYSTEM]\n{content_to_text(instructions)}\n\n{query}"
    return query


def request_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def extract_effort(payload: dict[str, Any]) -> str | None:
    candidates = [
        payload.get("effort"),
        payload.get("reasoning_effort"),
        (payload.get("reasoning") or {}).get("effort") if isinstance(payload.get("reasoning"), dict) else None,
        (payload.get("metadata") or {}).get("effort") if isinstance(payload.get("metadata"), dict) else None,
    ]
    for value in candidates:
        if value:
            return str(value)
    return None


async def call_dify(query: str, user: str, conversation_id: str = "") -> dict[str, Any]:
    body = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "conversation_id": conversation_id,
        "user": user,
    }
    started = time.monotonic()
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "claude-code-dify-bridge/1.0",
    }
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for attempt in range(3):
            try:
                response = await client.post(
                    f"{DIFY_BASE_URL}/chat-messages",
                    headers=headers,
                    json=body,
                )
            except httpx.RequestError as exc:
                if attempt == 2:
                    logger.exception("Dify 请求失败，user=%s", user)
                    return {"_error": 502, "_body": "Dify upstream request failed"}
                logger.warning("Dify 请求网络错误，将重试：%s", type(exc).__name__)
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code < 400:
                logger.info(
                    "dify_request status=%s attempt=%s duration_ms=%d",
                    response.status_code,
                    attempt + 1,
                    int((time.monotonic() - started) * 1000),
                )
                return response.json()
            if response.status_code in {408, 429} or response.status_code >= 500:
                if attempt < 2:
                    logger.warning("Dify 返回 %s，将重试（第 %d 次）", response.status_code, attempt + 1)
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
            logger.warning("Dify 返回错误：status=%s，user=%s", response.status_code, user)
            return {"_error": response.status_code, "_body": response.text[:2000]}
    return {"_error": 502, "_body": "Dify upstream request failed"}


async def refresh_dify_parameters() -> None:
    global DIFY_PARAMETER_KEYS
    try:
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.get(
                f"{DIFY_BASE_URL}/parameters",
                headers={
                    "Authorization": f"Bearer {DIFY_API_KEY}",
                    "Accept": "application/json",
                    "User-Agent": "claude-code-dify-bridge/1.0",
                },
            )
        if response.status_code >= 400:
            logger.warning("Dify 参数探测失败：status=%s", response.status_code)
            return
        data = response.json()
        exposed = data.get("user_input_form", []) if isinstance(data, dict) else []
        DIFY_PARAMETER_KEYS = {
            str(item.get("variable") or item.get("name"))
            for item in exposed
            if isinstance(item, dict) and (item.get("variable") or item.get("name"))
        }
        logger.info("Dify 参数探测完成：可用参数数量=%d", len(DIFY_PARAMETER_KEYS))
    except (httpx.RequestError, ValueError) as exc:
        logger.warning("Dify 参数探测异常：%s", type(exc).__name__)


async def poll_dify_parameters() -> None:
    while True:
        await refresh_dify_parameters()
        await asyncio.sleep(DIFY_PARAMETER_CHECK_INTERVAL)


@app.on_event("startup")
async def start_parameter_monitor() -> None:
    global DIFY_PARAMETER_TASK
    DIFY_PARAMETER_TASK = asyncio.create_task(poll_dify_parameters())


@app.on_event("shutdown")
async def stop_parameter_monitor() -> None:
    if DIFY_PARAMETER_TASK is not None:
        DIFY_PARAMETER_TASK.cancel()
        await asyncio.gather(DIFY_PARAMETER_TASK, return_exceptions=True)


def openai_response(answer: str, model: str, rid: str) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": rid,
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "dify_model": DIFY_MODEL}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "created": now, "owned_by": "dify"},
            {"id": "claude-sonnet-4-5", "object": "model", "created": now, "owned_by": "dify"},
            {"id": "sales-assistant", "object": "model", "created": now, "owned_by": "dify"},
        ],
    }


def check_local_key(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    return auth == f"Bearer {LOCAL_API_KEY}"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not check_local_key(request):
        return JSONResponse({"error": {"message": "invalid bridge key", "type": "auth_error"}}, status_code=401)
    payload = await request.json()
    if effort := extract_effort(payload):
        logger.info("收到 Claude Effort=%s；Dify 应用未暴露对应推理参数，保持原始模型配置", effort)
    messages = payload.get("messages") or []
    query = messages_to_query(messages)
    user = str(payload.get("user") or DEFAULT_USER)
    result = await call_dify(query, user, str(payload.get("conversation_id") or ""))
    if "_error" in result:
        return JSONResponse({"error": {"message": result["_body"], "type": "dify_error"}}, status_code=result["_error"])
    answer = str(result.get("answer") or "")
    rid = request_id()
    tool_calls = parse_dsml_tool_calls(answer)
    if tool_calls:
        return JSONResponse({
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", "sales-assistant")),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call["call_id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
                    } for call in tool_calls],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })
    if payload.get("stream"):
        async def stream() -> AsyncIterator[bytes]:
            chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": payload.get("model", "sales-assistant"), "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
            done = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": payload.get("model", "sales-assistant"), "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")
    return JSONResponse(openai_response(answer, str(payload.get("model", "sales-assistant")), rid))


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    if not check_local_key(request):
        return JSONResponse({"type": "error", "error": {"type": "authentication_error", "message": "invalid bridge key"}}, status_code=401)
    payload = await request.json()
    if effort := extract_effort(payload):
        logger.info("收到 Claude Effort=%s；Dify 应用未暴露对应推理参数，保持原始模型配置", effort)
    query = messages_to_query(payload.get("messages") or [])
    user = str(payload.get("metadata", {}).get("user_id") or DEFAULT_USER)
    result = await call_dify(query, user)
    if "_error" in result:
        return JSONResponse({"type": "error", "error": {"type": "api_error", "message": result["_body"]}}, status_code=result["_error"])
    answer = str(result.get("answer") or "")
    mid = "msg_" + uuid.uuid4().hex
    tool_calls = parse_dsml_tool_calls(answer)
    if tool_calls:
        return JSONResponse({
            "id": mid,
            "type": "message",
            "role": "assistant",
            "model": payload.get("model", "sales-assistant"),
            "content": [{"type": "tool_use", "id": call["call_id"], "name": call["name"], "input": call["arguments"]} for call in tool_calls],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })
    return JSONResponse({"id": mid, "type": "message", "role": "assistant", "model": payload.get("model", "sales-assistant"), "content": [{"type": "text", "text": answer}], "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}})


@app.post("/v1/responses")
async def responses(request: Request):
    """Accept OpenAI Responses requests emitted by recent Claude Code/LiteLLM."""
    if not check_local_key(request):
        return JSONResponse({"error": {"message": "invalid bridge key", "type": "auth_error"}}, status_code=401)
    payload = await request.json()
    if effort := extract_effort(payload):
        logger.info("收到 Claude Effort=%s；Dify 应用未暴露对应推理参数，保持原始模型配置", effort)
    query = responses_input_to_query(payload)
    user = str(payload.get("metadata", {}).get("user_id") or payload.get("user") or DEFAULT_USER)
    result = await call_dify(query, user)
    if "_error" in result:
        return JSONResponse({"error": {"message": result["_body"], "type": "dify_error"}}, status_code=result["_error"])
    answer = str(result.get("answer") or "")
    rid = "resp_" + uuid.uuid4().hex
    model = str(payload.get("model", "sales-assistant"))
    tool_calls = parse_dsml_tool_calls(answer)
    if tool_calls:
        return JSONResponse({
            "id": rid,
            "object": "response",
            "created_at": int(time.time()),
            "model": model,
            "status": "completed",
            "output": [{
                "id": call["call_id"],
                "type": "function_call",
                "status": "completed",
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
            } for call in tool_calls],
            "output_text": "",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        })
    return JSONResponse({
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [{
            "id": "msg_" + uuid.uuid4().hex,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": answer, "annotations": []}],
        }],
        "output_text": answer,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DIFY_BRIDGE_PORT", "4010")))
