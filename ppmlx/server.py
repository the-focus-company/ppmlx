from __future__ import annotations
import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

log = logging.getLogger("ppmlx.server")

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ppmlx import __version__

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    try:
        from ppmlx.db import get_db
        db = get_db()
        db.init()
        app.state.db = db
    except ImportError:
        app.state.db = None

    try:
        from ppmlx.config import load_config
        cfg = load_config()
        interval = cfg.logging.snapshot_interval_seconds
    except ImportError:
        interval = 60

    snapshot_task = asyncio.create_task(_snapshot_loop(interval))

    yield

    snapshot_task.cancel()
    try:
        await snapshot_task
    except asyncio.CancelledError:
        pass

    try:
        if app.state.db:
            app.state.db.flush()
            app.state.db.close()
    except Exception:
        pass


app = FastAPI(
    title="ppmlx",
    version=__version__,
    description="Ollama-style OpenAI-compatible LLM API for Apple Silicon",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _snapshot_loop(interval_seconds: int) -> None:
    """Periodically log system snapshots to the database."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            from ppmlx.memory import get_system_ram_gb
            from ppmlx.db import get_db
            from ppmlx.engine import get_engine
            ram_total = get_system_ram_gb()
            loaded = get_engine().list_loaded()
            uptime = int(time.time() - _start_time)
            get_db().log_system_snapshot(
                memory_total_gb=ram_total,
                memory_used_gb=0.0,  # hard to get used memory without psutil
                loaded_models=loaded,
                uptime_seconds=uptime,
            )
        except Exception:
            pass


def _route_engine(repo_id: str, has_images: bool) -> str:
    """Determine which engine to use: 'text', 'vision', or 'embed'."""
    try:
        from ppmlx.models import is_vision_model, is_embed_model
        if is_embed_model(repo_id):
            return "embed"
        if has_images and is_vision_model(repo_id):
            return "vision"
    except ImportError:
        pass
    return "text"


def _has_images(messages: list[dict]) -> bool:
    """Check if any message contains an image_url content part."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _log_request(_, **kwargs) -> None:
    """Log a request to the DB (best-effort)."""
    try:
        from ppmlx.db import get_db
        get_db().log_request(**kwargs)
    except Exception:
        pass


def _merge_system_messages(messages: list[dict]) -> list[dict]:
    """Merge all system messages into a single one at the start.

    Many models (e.g. Qwen) only accept one system message and require it
    to be the first message.  When clients like Codex send both an
    ``instructions`` system prompt *and* developer-role messages (also
    mapped to system), we merge them so the model template doesn't error.
    """
    system_parts: list[str] = []
    other: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            if c:
                system_parts.append(c)
        else:
            other.append(msg)
    if not system_parts:
        return messages
    merged = [{"role": "system", "content": "\n\n".join(system_parts)}]
    merged.extend(other)
    return merged


# ── Tool-call parsing ───────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# JSON format: <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
_TC_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# XML-like format: <function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>
_TC_XML_FUNC_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>", re.DOTALL
)
_TC_XML_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL
)


def _parse_tool_call_body(body: str) -> dict | None:
    """Parse the content inside a ``<tool_call>`` block.

    Supports two formats:
    1. JSON: ``{"name": "fn", "arguments": {"key": "val"}}``
    2. XML-like (Qwen3.5): ``<function=fn><parameter=k>v</parameter></function>``
    """
    body = body.strip()

    # Try JSON first
    jm = _TC_JSON_RE.search(body)
    if jm:
        try:
            data = json.loads(jm.group(0))
            name = data.get("name", "")
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, dict):
                args = json.dumps(args)
            elif not isinstance(args, str):
                args = json.dumps(args)
            return {"name": name, "arguments": args}
        except (json.JSONDecodeError, ValueError):
            pass

    # Try XML-like format
    fm = _TC_XML_FUNC_RE.search(body)
    if fm:
        name = fm.group(1)
        func_body = fm.group(2)
        params = {}
        for pm in _TC_XML_PARAM_RE.finditer(func_body):
            params[pm.group(1)] = pm.group(2).strip()
        return {"name": name, "arguments": json.dumps(params)}

    return None


def _parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract ``<tool_call>`` blocks from model output.

    Returns *(remaining_text, tool_calls)* where each tool call is
    ``{"name": "...", "arguments": "..."}`` (arguments as a JSON string).
    """
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        tc = _parse_tool_call_body(m.group(1))
        if tc:
            calls.append(tc)
    remaining = _TOOL_CALL_RE.sub("", text).strip()
    return remaining, calls


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    try:
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    try:
        from ppmlx.memory import get_system_ram_gb
        ram_gb = get_system_ram_gb()
    except Exception:
        ram_gb = 0.0

    registry_info = {}
    try:
        from ppmlx.registry import registry_meta
        registry_info = registry_meta()
    except Exception:
        pass

    return {
        "status": "ok",
        "version": __version__,
        "loaded_models": loaded,
        "uptime_seconds": int(time.time() - _start_time),
        "system": {
            "memory_total_gb": round(ram_gb, 1),
        },
        "registry": registry_info,
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint — returns JSON stats from the DB."""
    try:
        from ppmlx.db import get_db
        stats = get_db().get_stats()
    except Exception:
        stats = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}

    try:
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    return {**stats, "loaded_models": loaded}


def _model_metadata(model_id: str) -> dict:
    """Build rich model metadata for Codex / OpenAI-compatible clients."""
    meta = {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ppmlx",
        "slug": model_id,
    }
    # Try to pull extra info from registry
    try:
        from ppmlx.registry import registry_lookup
        entry = registry_lookup(model_id)
        if entry:
            meta["owned_by"] = entry.get("lab", "ppmlx")
    except Exception:
        entry = None
    return meta


@app.get("/v1/models")
async def list_models():
    """List available models (local + aliases)."""
    try:
        from ppmlx.models import list_local_models, all_aliases
        local = list_local_models()
        aliases = all_aliases()

        data = []
        seen = set()

        for m in local:
            mid = m.get("alias") or m.get("repo_id", "unknown")
            if mid not in seen:
                seen.add(mid)
                data.append(_model_metadata(mid))

        for alias in aliases:
            if alias not in seen:
                seen.add(alias)
                data.append(_model_metadata(alias))

        return {"object": "list", "data": data, "models": data}
    except ImportError:
        return {"object": "list", "data": [], "models": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    body = await request.json()

    # Append request summary for debugging (best-effort)
    try:
        import pathlib
        _cc_log = pathlib.Path("/tmp/ppmlx_chatcompletions_debug.jsonl")
        log_entry = {
            "ts": time.time(),
            "model": body.get("model"),
            "stream": body.get("stream"),
            "tools": len(body.get("tools") or []),
            "tool_names": [t.get("function", {}).get("name") or t.get("name")
                           for t in (body.get("tools") or [])],
            "messages_count": len(body.get("messages", [])),
        }
        with open(_cc_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    model_name = body.get("model", "")
    messages = body.get("messages", [])
    # Normalize OpenAI "developer" role to "system" for compatibility
    for msg in messages:
        if msg.get("role") == "developer":
            msg["role"] = "system"
    messages = _merge_system_messages(messages)
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    max_tokens = body.get("max_tokens", 2048)
    stop = body.get("stop")
    seed = body.get("seed")
    repetition_penalty = body.get("repetition_penalty")
    tools = body.get("tools") or None

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    has_imgs = _has_images(messages)
    engine_type = _route_engine(repo_id, has_imgs)

    request_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    start_ts = time.time()
    created = int(start_ts)

    if stream:
        return _stream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts, tools,
        )
    else:
        return await _nonstream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts, tools,
        )


def _stream_chat(
    request_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, stop, seed,
    repetition_penalty, request, start_ts, tools=None,
):
    """Return streaming SSE response."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        first_token_ts = None
        is_first_chunk = True
        full_text = ""
        try:
            if engine_type == "text":
                from ppmlx.engine import get_engine
                engine = get_engine()
                gen = engine.stream_generate(
                    repo_id, messages,
                    temperature=0.7 if temperature is None else temperature,
                    top_p=1.0 if top_p is None else top_p,
                    max_tokens=2048 if max_tokens is None else max_tokens,
                    seed=seed,
                    tools=tools,
                )
                async for chunk in _async_iter_sync_gen(gen):
                    full_text += chunk
                    if first_token_ts is None:
                        first_token_ts = time.time()
                    if is_first_chunk:
                        delta = {"role": "assistant", "content": chunk}
                        is_first_chunk = False
                    else:
                        delta = {"content": chunk}
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            elif engine_type == "vision":
                from ppmlx.engine_vlm import get_vision_engine
                engine = get_vision_engine()
                text, _, _ = engine.generate(repo_id, messages, max_tokens=1024 if max_tokens is None else max_tokens)
                full_text = text
                if first_token_ts is None:
                    first_token_ts = time.time()
                delta = {"role": "assistant", "content": text} if is_first_chunk else {"content": text}
                is_first_chunk = False
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            err = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(err)}\n\n"

        # Parse tool calls if tools were provided
        _, tool_calls = _parse_tool_calls(full_text) if tools else ("", [])

        if tool_calls:
            # Emit tool_calls in streaming format
            tc_list = []
            for i, tc in enumerate(tool_calls):
                call_id = "call_" + uuid.uuid4().hex[:24]
                tc_list.append({
                    "index": i,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })
            delta = {"tool_calls": tc_list}
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

        total_dur = (time.time() - start_ts) * 1000
        ttft = (first_token_ts - start_ts) * 1000 if first_token_ts else None
        _log_request(
            request,
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model_alias=model_name,
            model_repo=repo_id,
            stream=True,
            total_duration_ms=total_dur,
            time_to_first_token_ms=ttft,
            messages_count=len(messages),
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_chat(
    request_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, stop, seed,
    repetition_penalty, request, start_ts, tools=None,
):
    """Return non-streaming JSON response."""
    try:
        if engine_type == "text":
            from ppmlx.engine import get_engine
            engine = get_engine()
            text, reasoning, prompt_tokens, completion_tokens = engine.generate(
                repo_id, messages,
                temperature=0.7 if temperature is None else temperature,
                top_p=1.0 if top_p is None else top_p,
                max_tokens=2048 if max_tokens is None else max_tokens,
                seed=seed,
                repetition_penalty=repetition_penalty,
                tools=tools,
            )
        elif engine_type == "vision":
            from ppmlx.engine_vlm import get_vision_engine
            engine = get_vision_engine()
            text, prompt_tokens, completion_tokens = engine.generate(repo_id, messages)
            reasoning = None
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is an embedding model.")
    except HTTPException:
        raise
    except Exception as e:
        _log_request(
            request,
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model_alias=model_name,
            model_repo=repo_id,
            stream=False,
            status="error",
            error_message=str(e),
        )
        raise HTTPException(status_code=503, detail=str(e))

    total_dur = (time.time() - start_ts) * 1000

    # Parse tool calls if tools were provided
    remaining_text, tool_calls = _parse_tool_calls(text) if tools else (text, [])

    message: dict = {"role": "assistant", "content": remaining_text or None}
    if reasoning:
        message["reasoning"] = reasoning

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
        message["tool_calls"] = [
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ]

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    _log_request(
        request,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        model_alias=model_name,
        model_repo=repo_id,
        stream=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        total_duration_ms=total_dur,
        messages_count=len(messages),
    )

    return JSONResponse(response)


@app.post("/v1/completions")
async def completions(request: Request):
    """OpenAI-compatible text completions endpoint."""
    body = await request.json()
    model_name = body.get("model", "")
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 2048)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    messages = [{"role": "user", "content": prompt}]

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    request_id = "cmpl-" + uuid.uuid4().hex[:12]
    created = int(time.time())
    start_ts = time.time()

    try:
        from ppmlx.engine import get_engine
        engine = get_engine()
        text, reasoning, prompt_tokens, completion_tokens = engine.generate(
            repo_id, messages,
            temperature=0.7 if temperature is None else temperature,
            max_tokens=2048 if max_tokens is None else max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return JSONResponse({
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint."""
    body = await request.json()
    model_name = body.get("model", "")
    input_text = body.get("input", "")

    if isinstance(input_text, str):
        texts = [input_text]
    else:
        texts = list(input_text)

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    try:
        from ppmlx.engine_embed import get_embed_engine
        engine = get_embed_engine()
        vectors = engine.encode(repo_id, texts)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    data = [{"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(vectors)]
    total_tokens = sum(len(t.split()) for t in texts)

    _log_request(
        request,
        request_id="embed-" + uuid.uuid4().hex[:8],
        endpoint="/v1/embeddings",
        model_alias=model_name,
        model_repo=repo_id,
        stream=False,
        prompt_tokens=total_tokens,
        total_tokens=total_tokens,
    )

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── OpenAI Responses API (/v1/responses) ─────────────────────────────


def _responses_input_to_messages(input_data) -> list[dict]:
    """Convert Responses API 'input' field to chat messages list."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if isinstance(input_data, list):
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                content = item.get("content", "")
                # Handle content array (e.g. [{type: "input_text", text: "..."}])
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text", "")
                            if t:
                                text_parts.append(t)
                    content = "\n".join(text_parts) if text_parts else ""
                messages.append({"role": role, "content": content})
        return messages
    return [{"role": "user", "content": str(input_data)}]


@app.post("/v1/responses")
async def responses(request: Request):
    """OpenAI Responses API endpoint (used by Codex and newer OpenAI tools)."""
    body = await request.json()

    # Append each request to a debug log (best-effort)
    try:
        import pathlib
        _log_path = pathlib.Path("/tmp/ppmlx_responses_debug.jsonl")
        input_data_raw = body.get("input", "")
        # Summarize input for logging
        if isinstance(input_data_raw, list):
            input_summary = []
            for item in input_data_raw:
                role = item.get("role", "?")
                content = item.get("content", "")
                if isinstance(content, list):
                    text = " ".join(p.get("text", "")[:200] for p in content if isinstance(p, dict))
                elif isinstance(content, str):
                    text = content[:200]
                else:
                    text = str(content)[:200]
                input_summary.append({"role": role, "text": text[:200]})
        else:
            input_summary = str(input_data_raw)[:200]
        log_entry = {
            "ts": time.time(),
            "model": body.get("model"),
            "stream": body.get("stream"),
            "tools": len(body.get("tools") or []),
            "input_summary": input_summary,
            "instructions_len": len(body.get("instructions", "") or ""),
        }
        with open(_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    model_name = body.get("model", "")
    input_data = body.get("input", "")
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    max_tokens = body.get("max_output_tokens") or body.get("max_tokens", 4096)
    # instructions field acts as a system prompt
    instructions = body.get("instructions")

    tools = body.get("tools") or None
    log.info("POST /v1/responses model=%s stream=%s tools=%d",
             model_name, stream, len(tools) if tools else 0)

    messages = _responses_input_to_messages(input_data)
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    # Merge consecutive system messages into one (some models only allow a
    # single system message at the start, e.g. Qwen).
    messages = _merge_system_messages(messages)

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    has_imgs = _has_images(messages)
    engine_type = _route_engine(repo_id, has_imgs)

    resp_id = "resp_" + uuid.uuid4().hex[:24]
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if stream:
        return _stream_responses(
            resp_id, msg_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, request, tools,
        )
    else:
        return await _nonstream_responses(
            resp_id, msg_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, request, tools,
        )


def _sse(event: str, data: dict) -> str:
    # Codex requires a "type" field matching the event name in each SSE payload.
    if "type" not in data:
        data = {**data, "type": event}
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_SENTINEL = object()


async def _async_iter_sync_gen(sync_gen, loop=None):
    """Yield items from a blocking sync generator without blocking the event loop.

    Runs the generator in a background thread and bridges items to the async
    world via an :class:`asyncio.Queue`.
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
            for item in sync_gen:
                loop.call_soon_threadsafe(q.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

    import threading
    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        item = await q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _stream_responses(
    resp_id, msg_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, request,
    tools=None,
):
    """Streaming SSE for the Responses API."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        log.info("responses stream start model=%s tools=%d msgs=%d",
                 model_name, len(tools) if tools else 0, len(messages))

        # response.created
        resp_obj = {
            "id": resp_id,
            "object": "response",
            "created_at": created,
            "model": model_name,
            "status": "in_progress",
            "output": [],
            "usage": None,
        }
        yield _sse("response.created", resp_obj)

        # Always open a text message item upfront so the client sees
        # output_item.added immediately (prevents timeout while model loads).
        msg_item = {
            "type": "message",
            "id": msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        yield _sse("response.output_item.added", {
            "output_index": 0,
            "item": msg_item,
        })
        yield _sse("response.content_part.added", {
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })

        full_text = ""
        start_ts = time.time()

        try:
            if engine_type == "text":
                from ppmlx.engine import get_engine
                engine = get_engine()
                # Always stream — keeps the SSE connection alive and
                # prevents Codex from timing out during generation.
                gen = engine.stream_generate(
                    repo_id, messages,
                    temperature=0.7 if temperature is None else temperature,
                    top_p=1.0 if top_p is None else top_p,
                    max_tokens=4096 if max_tokens is None else max_tokens,
                    tools=tools,
                )
                async for chunk in _async_iter_sync_gen(gen):
                    full_text += chunk
                    # Stream text deltas to keep connection alive.
                    yield _sse("response.output_text.delta", {
                        "output_index": 0,
                        "content_index": 0,
                        "delta": chunk,
                    })
            elif engine_type == "vision":
                from ppmlx.engine_vlm import get_vision_engine
                engine = get_vision_engine()
                text, _, _ = engine.generate(
                    repo_id, messages,
                    max_tokens=4096 if max_tokens is None else max_tokens,
                )
                full_text = text
                yield _sse("response.output_text.delta", {
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                })
        except Exception as e:
            log.exception("responses stream error")
            yield _sse("error", {
                "type": "server_error",
                "message": str(e),
            })
            return

        gen_dur = time.time() - start_ts
        log.info("responses generation done in %.1fs, %d chars", gen_dur, len(full_text))

        # Parse tool calls from the model output
        remaining_text, tool_calls = _parse_tool_calls(full_text)
        log.info("parsed %d tool_calls, remaining_text=%d chars",
                 len(tool_calls), len(remaining_text))

        # ── Close the text message ───────────────────────────────────
        yield _sse("response.output_text.done", {
            "output_index": 0,
            "content_index": 0,
            "text": remaining_text,
        })
        yield _sse("response.content_part.done", {
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": remaining_text, "annotations": []},
        })
        done_msg = {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
        }
        yield _sse("response.output_item.done", {
            "output_index": 0,
            "item": done_msg,
        })

        output_items: list[dict] = [done_msg]
        output_idx = 1

        # ── Emit function call items ─────────────────────────────────
        for tc in tool_calls:
            fc_id = "fc_" + uuid.uuid4().hex[:24]
            call_id = "call_" + uuid.uuid4().hex[:24]
            fc_item = {
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": tc["name"],
                "arguments": "",
                "status": "in_progress",
            }
            yield _sse("response.output_item.added", {
                "output_index": output_idx,
                "item": fc_item,
            })
            yield _sse("response.function_call_arguments.delta", {
                "output_index": output_idx,
                "delta": tc["arguments"],
            })
            yield _sse("response.function_call_arguments.done", {
                "output_index": output_idx,
                "arguments": tc["arguments"],
            })
            done_fc = {**fc_item, "arguments": tc["arguments"], "status": "completed"}
            yield _sse("response.output_item.done", {
                "output_index": output_idx,
                "item": done_fc,
            })
            output_items.append(done_fc)
            output_idx += 1

        # ── Wrap up ──────────────────────────────────────────────────
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages if isinstance(m.get("content"), str))
        completion_tokens = len(full_text.split())
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        resp_obj["status"] = "completed"
        resp_obj["output"] = output_items
        resp_obj["usage"] = usage
        yield _sse("response.completed", resp_obj)
        log.info("responses stream completed, %d output items", len(output_items))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_responses(
    resp_id, msg_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, request,
    tools=None,
):
    """Non-streaming Responses API."""
    try:
        if engine_type == "text":
            from ppmlx.engine import get_engine
            engine = get_engine()
            text, reasoning, prompt_tokens, completion_tokens = engine.generate(
                repo_id, messages,
                temperature=0.7 if temperature is None else temperature,
                top_p=1.0 if top_p is None else top_p,
                max_tokens=4096 if max_tokens is None else max_tokens,
                tools=tools,
            )
        elif engine_type == "vision":
            from ppmlx.engine_vlm import get_vision_engine
            engine = get_vision_engine()
            text, prompt_tokens, completion_tokens = engine.generate(repo_id, messages)
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is an embedding model.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    remaining_text, tool_calls = _parse_tool_calls(text)

    output: list[dict] = []
    if remaining_text:
        output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
        })
    for tc in tool_calls:
        output.append({
            "type": "function_call",
            "id": "fc_" + uuid.uuid4().hex[:24],
            "call_id": "call_" + uuid.uuid4().hex[:24],
            "name": tc["name"],
            "arguments": tc["arguments"],
            "status": "completed",
        })
    # Fallback: if no tool calls and no remaining text, emit original text
    if not output:
        output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })

    return JSONResponse({
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "model": model_name,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


# ── WebSocket transport for Responses API (Codex) ────────────────────


@app.websocket("/v1/responses")
async def responses_ws(websocket: WebSocket):
    """WebSocket transport for the Responses API (used by Codex CLI)."""
    await websocket.accept()
    try:
        while True:
            body = await websocket.receive_json()
            msg_type = body.get("type", "")
            if msg_type != "response.create":
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "invalid_request", "message": f"Unknown message type: {msg_type}"},
                })
                continue

            # Extract fields from the response.create message
            response_body = body.get("response", body)
            model_name = response_body.get("model", body.get("model", ""))
            input_data = response_body.get("input", body.get("input", ""))
            temperature = response_body.get("temperature", body.get("temperature", 0.7))
            top_p = response_body.get("top_p", body.get("top_p", 1.0))
            max_tokens = (
                response_body.get("max_output_tokens")
                or body.get("max_output_tokens")
                or response_body.get("max_tokens")
                or body.get("max_tokens", 4096)
            )
            instructions = response_body.get("instructions", body.get("instructions"))
            tools = response_body.get("tools", body.get("tools")) or None

            messages = _responses_input_to_messages(input_data)
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            messages = _merge_system_messages(messages)

            try:
                from ppmlx.models import resolve_alias
                repo_id = resolve_alias(model_name)
            except Exception:
                repo_id = model_name

            has_imgs = _has_images(messages)
            engine_type = _route_engine(repo_id, has_imgs)

            resp_id = "resp_" + uuid.uuid4().hex[:24]
            msg_id = "msg_" + uuid.uuid4().hex[:24]
            created = int(time.time())

            resp_obj = {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "model": model_name,
                "status": "in_progress",
                "output": [],
                "usage": None,
            }
            await websocket.send_json({"type": "response.created", **resp_obj})

            full_text = ""
            try:
                if engine_type == "text":
                    from ppmlx.engine import get_engine
                    engine = get_engine()
                    if tools:
                        text, _, _, _ = engine.generate(
                            repo_id, messages,
                            temperature=0.7 if temperature is None else temperature,
                            top_p=1.0 if top_p is None else top_p,
                            max_tokens=4096 if max_tokens is None else max_tokens,
                            tools=tools,
                        )
                        full_text = text
                    else:
                        for chunk in engine.stream_generate(
                            repo_id, messages,
                            temperature=0.7 if temperature is None else temperature,
                            top_p=1.0 if top_p is None else top_p,
                            max_tokens=4096 if max_tokens is None else max_tokens,
                        ):
                            full_text += chunk
                elif engine_type == "vision":
                    from ppmlx.engine_vlm import get_vision_engine
                    engine = get_vision_engine()
                    text, _, _ = engine.generate(
                        repo_id, messages,
                        max_tokens=4096 if max_tokens is None else max_tokens,
                    )
                    full_text = text
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "server_error", "message": str(e)},
                })
                continue

            remaining_text, tool_calls = _parse_tool_calls(full_text)
            output_items: list[dict] = []
            output_idx = 0

            # Text message
            if remaining_text:
                msg_item = {
                    "type": "message", "id": msg_id,
                    "status": "in_progress", "role": "assistant", "content": [],
                }
                await websocket.send_json({
                    "type": "response.output_item.added",
                    "output_index": output_idx, "item": msg_item,
                })
                await websocket.send_json({
                    "type": "response.content_part.added",
                    "output_index": output_idx, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                await websocket.send_json({
                    "type": "response.output_text.delta",
                    "output_index": output_idx, "content_index": 0,
                    "delta": remaining_text,
                })
                await websocket.send_json({
                    "type": "response.output_text.done",
                    "output_index": output_idx, "content_index": 0,
                    "text": remaining_text,
                })
                await websocket.send_json({
                    "type": "response.content_part.done",
                    "output_index": output_idx, "content_index": 0,
                    "part": {"type": "output_text", "text": remaining_text, "annotations": []},
                })
                done_msg = {
                    "type": "message", "id": msg_id, "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
                }
                await websocket.send_json({
                    "type": "response.output_item.done",
                    "output_index": output_idx, "item": done_msg,
                })
                output_items.append(done_msg)
                output_idx += 1

            # Function calls
            for tc in tool_calls:
                fc_id = "fc_" + uuid.uuid4().hex[:24]
                call_id = "call_" + uuid.uuid4().hex[:24]
                fc_item = {
                    "type": "function_call", "id": fc_id, "call_id": call_id,
                    "name": tc["name"], "arguments": "", "status": "in_progress",
                }
                await websocket.send_json({
                    "type": "response.output_item.added",
                    "output_index": output_idx, "item": fc_item,
                })
                await websocket.send_json({
                    "type": "response.function_call_arguments.delta",
                    "output_index": output_idx, "delta": tc["arguments"],
                })
                await websocket.send_json({
                    "type": "response.function_call_arguments.done",
                    "output_index": output_idx, "arguments": tc["arguments"],
                })
                done_fc = {**fc_item, "arguments": tc["arguments"], "status": "completed"}
                await websocket.send_json({
                    "type": "response.output_item.done",
                    "output_index": output_idx, "item": done_fc,
                })
                output_items.append(done_fc)
                output_idx += 1

            prompt_tokens = sum(
                len(m.get("content", "").split()) for m in messages
                if isinstance(m.get("content"), str)
            )
            completion_tokens = len(full_text.split())
            usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            resp_obj["status"] = "completed"
            resp_obj["output"] = output_items or [{
                "type": "message", "id": msg_id, "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": full_text, "annotations": []}],
            }]
            resp_obj["usage"] = usage
            await websocket.send_json({"type": "response.completed", **resp_obj})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
