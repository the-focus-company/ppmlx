"""Framework compatibility test runner and matrix generator.

Tests that ppmlx's OpenAI-compatible API works correctly with popular
AI frameworks by validating the exact HTTP contract each framework expects.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


# ── Result types ──────────────────────────────────────────────────────

@dataclass
class CompatTestResult:
    """Result of a single compatibility test."""
    framework: str
    feature: str
    passed: bool
    duration_ms: float
    error: str | None = None


@dataclass
class CompatMatrix:
    """Full compatibility matrix across frameworks and features."""
    results: list[CompatTestResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def add(self, result: CompatTestResult) -> None:
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def frameworks(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.results:
            seen.setdefault(r.framework, None)
        return list(seen)

    @property
    def features(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.results:
            seen.setdefault(r.feature, None)
        return list(seen)

    def get(self, framework: str, feature: str) -> CompatTestResult | None:
        for r in self.results:
            if r.framework == framework and r.feature == feature:
                return r
        return None

    def to_markdown(self) -> str:
        """Render the matrix as a markdown table."""
        frameworks = self.frameworks
        features = self.features
        if not frameworks or not features:
            return "_No compatibility tests ran._"

        lines: list[str] = []
        lines.append("# ppmlx Framework Compatibility Matrix")
        lines.append("")

        # Summary
        lines.append(f"**{self.passed}/{self.total} tests passed**")
        lines.append("")

        # Header row
        header = "| Feature |"
        sep = "| --- |"
        for fw in frameworks:
            header += f" {fw} |"
            sep += " :---: |"
        lines.append(header)
        lines.append(sep)

        # Data rows
        for feat in features:
            row = f"| {feat} |"
            for fw in frameworks:
                result = self.get(fw, feat)
                if result is None:
                    row += " - |"
                elif result.passed:
                    row += " PASS |"
                else:
                    row += " FAIL |"
            lines.append(row)

        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
            },
            "results": [
                {
                    "framework": r.framework,
                    "feature": r.feature,
                    "passed": r.passed,
                    "duration_ms": round(r.duration_ms, 2),
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# ── Mock setup (shared by tests and CLI) ─────────────────────────────

_MOCKED_MODULES = [
    "ppmlx.engine", "ppmlx.engine_vlm", "ppmlx.engine_embed",
    "ppmlx.models", "ppmlx.db", "ppmlx.config", "ppmlx.memory",
    "ppmlx.schema",
]


def setup_mocks() -> None:
    """Install mock modules and engines for running compat tests without a GPU."""
    for mod in _MOCKED_MODULES:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    mock_engine = MagicMock()
    mock_engine.generate.return_value = ("Hello from ppmlx!", None, 10, 5)
    mock_engine.stream_generate.return_value = iter(["Hello", " from", " ppmlx!"])
    mock_engine.list_loaded.return_value = []
    sys.modules["ppmlx.engine"].get_engine = MagicMock(return_value=mock_engine)

    mock_embed = MagicMock()
    mock_embed.encode = MagicMock(
        side_effect=lambda repo, texts: [[0.1, 0.2, 0.3] for _ in texts]
    )
    sys.modules["ppmlx.engine_embed"].get_embed_engine = MagicMock(
        return_value=mock_embed
    )

    sys.modules["ppmlx.models"].resolve_alias = MagicMock(side_effect=lambda x: x)
    sys.modules["ppmlx.models"].list_local_models = MagicMock(return_value=[])
    sys.modules["ppmlx.models"].all_aliases = MagicMock(return_value={})
    sys.modules["ppmlx.models"].is_vision_model = MagicMock(return_value=False)
    sys.modules["ppmlx.models"].is_embed_model = MagicMock(return_value=False)

    mock_db = MagicMock()
    mock_db.get_stats.return_value = {
        "total_requests": 0, "avg_duration_ms": None, "by_model": [],
    }
    sys.modules["ppmlx.db"].get_db = MagicMock(return_value=mock_db)
    sys.modules["ppmlx.memory"].get_system_ram_gb = MagicMock(return_value=16.0)

    mock_cfg = MagicMock()
    mock_cfg.logging.snapshot_interval_seconds = 60
    sys.modules["ppmlx.config"].load_config = MagicMock(return_value=mock_cfg)


# ── Helpers ──────────────────────────────────────────────────────────

def parse_sse_chunks(response) -> tuple[list[dict], bool]:
    """Parse SSE lines from a streaming response.

    Returns (data_chunks, saw_done) where data_chunks is a list of
    parsed JSON objects and saw_done indicates whether [DONE] was seen.
    """
    chunks: list[dict] = []
    saw_done = False
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            if payload == "[DONE]":
                saw_done = True
            else:
                chunks.append(json.loads(payload))
    return chunks, saw_done


# ── Test runner ───────────────────────────────────────────────────────

def _refresh_stream_mock() -> None:
    """Reset the mock stream_generate iterator so streaming tests work.

    Mock iterators are consumed after one use.  This must be called
    before every test that exercises the streaming endpoint.
    """
    engine_mod = sys.modules.get("ppmlx.engine")
    if engine_mod is None:
        return
    get_engine = getattr(engine_mod, "get_engine", None)
    if get_engine is None:
        return
    mock_engine = get_engine()
    mock_engine.stream_generate.return_value = iter(["Hello", " from", " ppmlx!"])


def _run_test(
    framework: str,
    feature: str,
    fn: callable,
    client: TestClient,
) -> CompatTestResult:
    """Run a single compatibility test and capture the result."""
    _refresh_stream_mock()
    start = time.time()
    try:
        fn(client)
        duration = (time.time() - start) * 1000
        return CompatTestResult(
            framework=framework, feature=feature,
            passed=True, duration_ms=duration,
        )
    except Exception as exc:
        duration = (time.time() - start) * 1000
        return CompatTestResult(
            framework=framework, feature=feature,
            passed=False, duration_ms=duration, error=str(exc),
        )


# ── OpenAI SDK tests ─────────────────────────────────────────────────

def _test_openai_chat_completion(client: TestClient) -> None:
    """Test the request/response contract that openai.ChatCompletion expects."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    })
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert "id" in data
    assert "created" in data
    assert "model" in data
    assert "choices" in data and len(data["choices"]) > 0
    choice = data["choices"][0]
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    assert "content" in choice["message"]
    assert "finish_reason" in choice
    assert "usage" in data
    usage = data["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def _test_openai_chat_streaming(client: TestClient) -> None:
    """Test SSE streaming format that openai SDK expects."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    })
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"

    chunks, saw_done = parse_sse_chunks(resp)
    assert len(chunks) >= 1, "Expected at least one chunk"
    assert saw_done, "Expected [DONE] sentinel"

    first = chunks[0]
    assert first["object"] == "chat.completion.chunk"
    assert "id" in first
    assert "choices" in first
    assert "delta" in first["choices"][0]

    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] is not None


def _test_openai_models_list(client: TestClient) -> None:
    """Test GET /v1/models returns OpenAI-compatible model list."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)


def _test_openai_embeddings(client: TestClient) -> None:
    """Test embeddings endpoint for OpenAI SDK contract."""
    resp = client.post("/v1/embeddings", json={
        "model": "test-embed",
        "input": "Hello world",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert "data" in data and len(data["data"]) > 0
    emb = data["data"][0]
    assert emb["object"] == "embedding"
    assert isinstance(emb["embedding"], list)
    assert emb["index"] == 0
    assert "usage" in data


def _test_openai_completions(client: TestClient) -> None:
    """Test text completion endpoint contract."""
    resp = client.post("/v1/completions", json={
        "model": "test-model",
        "prompt": "Once upon a time",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert "choices" in data and len(data["choices"]) > 0
    assert "text" in data["choices"][0]
    assert "usage" in data


def _test_openai_function_calling(client: TestClient) -> None:
    """Test that function/tool calling response shape matches OpenAI spec."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }],
    })
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["object"] == "chat.completion"
    # The response should have a valid structure even if no tool_calls generated
    choice = data["choices"][0]
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"


# ── LangChain tests ──────────────────────────────────────────────────

def _test_langchain_chat(client: TestClient) -> None:
    """Test the exact request format LangChain's ChatOpenAI sends."""
    # LangChain sends temperature, top_p, n, stream, max_tokens
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.7,
        "top_p": 1.0,
        "stream": False,
        "max_tokens": 256,
    })
    assert resp.status_code == 200
    data = resp.json()
    # LangChain checks these fields
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(data["choices"][0]["message"]["content"], str)


def _test_langchain_streaming(client: TestClient) -> None:
    """Test LangChain streaming pattern: reads SSE lines."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "temperature": 0.0,
    })
    assert resp.status_code == 200

    chunks, _ = parse_sse_chunks(resp)
    content_parts = [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if chunk["choices"][0].get("delta", {}).get("content")
    ]
    assert len(content_parts) > 0, "Expected content in stream"


def _test_langchain_embeddings(client: TestClient) -> None:
    """Test LangChain OpenAIEmbeddings pattern."""
    # LangChain sends input as a list of strings
    resp = client.post("/v1/embeddings", json={
        "model": "test-embed",
        "input": ["Hello", "World"],
        "encoding_format": "float",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    for item in data["data"]:
        assert item["object"] == "embedding"
        assert isinstance(item["embedding"], list)


def _test_langchain_tool_calling(client: TestClient) -> None:
    """Test LangChain bind_tools pattern."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Get weather in NYC"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }],
        "tool_choice": "auto",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data


# ── httpx / requests pattern tests ───────────────────────────────────

def _test_httpx_json_post(client: TestClient) -> None:
    """Test raw httpx-style JSON POST."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    data = resp.json()
    assert "choices" in data


def _test_httpx_streaming(client: TestClient) -> None:
    """Test raw httpx streaming pattern (reading line-by-line)."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    })
    assert resp.status_code == 200

    chunks, saw_done = parse_sse_chunks(resp)
    assert len(chunks) > 0, "Expected data chunks"
    assert saw_done, "Expected [DONE] sentinel"
    for chunk in chunks:
        assert "choices" in chunk


def _test_httpx_health_check(client: TestClient) -> None:
    """Test health endpoint (common pattern for readiness probes)."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def _test_httpx_bearer_auth(client: TestClient) -> None:
    """Test that Bearer auth header is accepted (any key works)."""
    resp = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer any-key-works"},
    )
    assert resp.status_code == 200


def _test_httpx_batch_embeddings(client: TestClient) -> None:
    """Test batch embeddings with a list of inputs."""
    resp = client.post("/v1/embeddings", json={
        "model": "test-embed",
        "input": ["first", "second", "third"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 3
    for i, item in enumerate(data["data"]):
        assert item["index"] == i


# ── Edge case tests ──────────────────────────────────────────────────

def _test_edge_empty_messages(client: TestClient) -> None:
    """Test that empty message content is handled gracefully."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": ""}],
    })
    # Should not crash; 200 or a well-formed error
    assert resp.status_code in (200, 400, 422)


def _test_edge_missing_model(client: TestClient) -> None:
    """Test request with empty model field."""
    resp = client.post("/v1/chat/completions", json={
        "model": "",
        "messages": [{"role": "user", "content": "Hi"}],
    })
    # Should not crash
    assert resp.status_code in (200, 400, 404, 422, 503)


def _test_edge_multi_turn(client: TestClient) -> None:
    """Test multi-turn conversation (system + user + assistant + user)."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["role"] == "assistant"


def _test_edge_extra_fields_ignored(client: TestClient) -> None:
    """Test that unknown fields in request body are ignored gracefully."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "unknown_field": True,
        "custom_param": 42,
    })
    assert resp.status_code == 200


def _test_edge_sequential_requests(client: TestClient) -> None:
    """Test that multiple sequential requests work (simulates concurrency)."""
    for _ in range(5):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 200


def _test_edge_large_message(client: TestClient) -> None:
    """Test handling of a large message payload."""
    long_content = "word " * 1000  # ~5000 chars
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": long_content}],
    })
    assert resp.status_code in (200, 503)


# ── Test registry ────────────────────────────────────────────────────

COMPAT_TESTS: list[tuple[str, str, callable]] = [
    # OpenAI SDK
    ("OpenAI SDK", "Chat Completion", _test_openai_chat_completion),
    ("OpenAI SDK", "Streaming", _test_openai_chat_streaming),
    ("OpenAI SDK", "Model List", _test_openai_models_list),
    ("OpenAI SDK", "Embeddings", _test_openai_embeddings),
    ("OpenAI SDK", "Text Completion", _test_openai_completions),
    ("OpenAI SDK", "Function Calling", _test_openai_function_calling),
    # LangChain
    ("LangChain", "Chat Completion", _test_langchain_chat),
    ("LangChain", "Streaming", _test_langchain_streaming),
    ("LangChain", "Embeddings", _test_langchain_embeddings),
    ("LangChain", "Tool Calling", _test_langchain_tool_calling),
    # httpx / requests
    ("httpx/requests", "JSON POST", _test_httpx_json_post),
    ("httpx/requests", "Streaming", _test_httpx_streaming),
    ("httpx/requests", "Health Check", _test_httpx_health_check),
    ("httpx/requests", "Bearer Auth", _test_httpx_bearer_auth),
    ("httpx/requests", "Batch Embeddings", _test_httpx_batch_embeddings),
    # Edge cases
    ("Edge Cases", "Empty Messages", _test_edge_empty_messages),
    ("Edge Cases", "Missing Model", _test_edge_missing_model),
    ("Edge Cases", "Multi-turn", _test_edge_multi_turn),
    ("Edge Cases", "Extra Fields Ignored", _test_edge_extra_fields_ignored),
    ("Edge Cases", "Sequential Requests", _test_edge_sequential_requests),
    ("Edge Cases", "Large Message", _test_edge_large_message),
]


def run_compat_suite(client: TestClient) -> CompatMatrix:
    """Run the full compatibility test suite against a test client."""
    matrix = CompatMatrix()
    for framework, feature, fn in COMPAT_TESTS:
        result = _run_test(framework, feature, fn, client)
        matrix.add(result)
    return matrix
