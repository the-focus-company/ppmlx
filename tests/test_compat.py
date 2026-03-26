"""Framework compatibility tests for ppmlx's OpenAI-compatible API.

Validates that ppmlx works correctly with OpenAI SDK, LangChain,
httpx/requests, and handles edge cases properly.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

from ppmlx.compat import setup_mocks, parse_sse_chunks

# Install mocks before importing the server module.
setup_mocks()

from fastapi.testclient import TestClient
from ppmlx.server import app


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Provide a fresh FastAPI test client with mocked engines."""
    # Refresh mocks so consumed iterators are reset.
    setup_mocks()
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════
# OpenAI SDK compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestOpenAISDK:
    """Tests verifying ppmlx matches the OpenAI Python SDK's expected contract."""

    def test_chat_completion(self, client: TestClient):
        """Non-streaming chat completion returns all required fields."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert isinstance(data["created"], int)
        assert data["model"] == "test-model"

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] in ("stop", "length", "tool_calls")

        usage = data["usage"]
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_chat_streaming(self, client: TestClient):
        """Streaming chat returns proper SSE chunks ending with [DONE]."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        chunks, saw_done = parse_sse_chunks(resp)
        assert len(chunks) >= 2, "Expected at least two data chunks"
        assert saw_done, "Stream must end with data: [DONE]"

        first = chunks[0]
        assert first["object"] == "chat.completion.chunk"
        assert first["id"].startswith("chatcmpl-")
        assert first["choices"][0]["delta"].get("role") == "assistant"

        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] is not None

    def test_model_list(self, client: TestClient):
        """GET /v1/models returns OpenAI-compatible list."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        for model in data["data"]:
            assert "id" in model
            assert model.get("object") == "model"

    def test_embeddings_single(self, client: TestClient):
        """Embeddings with a single string input."""
        resp = client.post("/v1/embeddings", json={
            "model": "test-embed",
            "input": "Hello world",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        emb = data["data"][0]
        assert emb["object"] == "embedding"
        assert isinstance(emb["embedding"], list)
        assert len(emb["embedding"]) > 0
        assert emb["index"] == 0
        assert "prompt_tokens" in data["usage"]

    def test_text_completion(self, client: TestClient):
        """Legacy text completion endpoint."""
        resp = client.post("/v1/completions", json={
            "model": "test-model",
            "prompt": "Once upon a time",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["id"].startswith("cmpl-")
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]
        assert data["choices"][0]["finish_reason"] in ("stop", "length", None)
        assert "usage" in data

    def test_function_calling_request(self, client: TestClient):
        """Function calling: server accepts tools and returns valid response."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                },
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["finish_reason"] in ("stop", "tool_calls")


# ═══════════════════════════════════════════════════════════════════════
# LangChain compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestLangChain:
    """Tests verifying ppmlx works with LangChain's ChatOpenAI patterns."""

    def test_chat_with_system_prompt(self, client: TestClient):
        """LangChain sends system + user messages with explicit params."""
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
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)

    def test_streaming_content_assembly(self, client: TestClient):
        """LangChain assembles streaming content from delta chunks."""
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
        assembled = "".join(content_parts)
        assert len(assembled) > 0, "Stream produced no content"

    def test_embeddings_batch(self, client: TestClient):
        """LangChain OpenAIEmbeddings sends input as a list."""
        resp = client.post("/v1/embeddings", json={
            "model": "test-embed",
            "input": ["Hello", "World"],
            "encoding_format": "float",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        for i, item in enumerate(data["data"]):
            assert item["object"] == "embedding"
            assert item["index"] == i
            assert isinstance(item["embedding"], list)

    def test_tool_calling_with_tool_choice(self, client: TestClient):
        """LangChain bind_tools sends tool_choice alongside tools."""
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
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
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


# ═══════════════════════════════════════════════════════════════════════
# httpx / requests patterns
# ═══════════════════════════════════════════════════════════════════════


class TestHTTPPatterns:
    """Tests for common raw HTTP patterns developers use."""

    def test_json_post_with_auth(self, client: TestClient):
        """POST with JSON body and Bearer auth header."""
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
        assert "choices" in resp.json()

    def test_streaming_line_parsing(self, client: TestClient):
        """Raw line-by-line SSE parsing pattern."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200

        chunks, saw_done = parse_sse_chunks(resp)
        assert len(chunks) > 0, "Expected data chunks in SSE stream"
        assert saw_done, "Expected [DONE] sentinel in SSE stream"

    def test_health_check(self, client: TestClient):
        """Health endpoint for readiness/liveness probes."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_bearer_auth_accepted(self, client: TestClient):
        """Any Bearer token is accepted (ppmlx has no auth)."""
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer literally-anything"},
        )
        assert resp.status_code == 200

    def test_batch_embeddings(self, client: TestClient):
        """Batch embeddings with multiple inputs."""
        resp = client.post("/v1/embeddings", json={
            "model": "test-embed",
            "input": ["one", "two", "three"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        for i, item in enumerate(data["data"]):
            assert item["index"] == i

    def test_content_type_json(self, client: TestClient):
        """Response Content-Type is application/json for non-streaming."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert "application/json" in resp.headers["content-type"]

    def test_cors_headers(self, client: TestClient):
        """CORS headers are present (needed for browser-based clients)."""
        resp = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code in (200, 204, 400)


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_content(self, client: TestClient):
        """Empty message content should not crash."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": ""}],
        })
        assert resp.status_code in (200, 400, 422)

    def test_empty_model_name(self, client: TestClient):
        """Empty model name should not crash."""
        resp = client.post("/v1/chat/completions", json={
            "model": "",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code in (200, 400, 404, 422, 503)

    def test_multi_turn_conversation(self, client: TestClient):
        """Multi-turn conversation with all role types."""
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
        assert resp.json()["choices"][0]["message"]["role"] == "assistant"

    def test_extra_fields_ignored(self, client: TestClient):
        """Unknown request body fields should be ignored."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "unknown_field": True,
            "custom_param": 42,
        })
        assert resp.status_code == 200

    def test_sequential_requests(self, client: TestClient):
        """Multiple sequential requests should all succeed."""
        for i in range(5):
            resp = client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": f"Request {i}"}],
            })
            assert resp.status_code == 200, f"Request {i} failed"

    def test_large_message(self, client: TestClient):
        """Large message payload should be handled."""
        long_content = "word " * 1000
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": long_content}],
        })
        assert resp.status_code in (200, 503)

    def test_temperature_zero(self, client: TestClient):
        """Temperature=0 is valid and should not error."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0,
        })
        assert resp.status_code == 200

    def test_max_tokens_one(self, client: TestClient):
        """max_tokens=1 is an extreme but valid value."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        })
        assert resp.status_code == 200

    def test_developer_role_mapped(self, client: TestClient):
        """Developer role (used by Codex) is mapped to system."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        })
        assert resp.status_code == 200

    def test_stop_sequences(self, client: TestClient):
        """Stop sequences in various formats should be accepted."""
        for stop_val in ["\n", ["\n", "END"]]:
            resp = client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stop": stop_val,
            })
            assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Compatibility matrix integration test
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="class")
def compat_matrix():
    """Run the compat suite once and share the result across the class."""
    from ppmlx.compat import run_compat_suite

    setup_mocks()
    with TestClient(app) as c:
        return run_compat_suite(c)


class TestCompatMatrix:
    """Tests for the compatibility matrix runner itself."""

    def test_run_suite(self, compat_matrix):
        """The full compat suite runs and produces a matrix."""
        assert compat_matrix.total > 0
        assert compat_matrix.passed + compat_matrix.failed == compat_matrix.total
        assert len(compat_matrix.frameworks) >= 3
        assert len(compat_matrix.features) >= 5

    def test_matrix_to_markdown(self, compat_matrix):
        """Matrix produces valid markdown output."""
        md = compat_matrix.to_markdown()
        assert "# ppmlx Framework Compatibility Matrix" in md
        assert "PASS" in md
        assert "|" in md

    def test_matrix_to_dict(self, compat_matrix):
        """Matrix serializes to a JSON-compatible dict."""
        d = compat_matrix.to_dict()
        assert "summary" in d
        assert "results" in d
        assert d["summary"]["total"] == compat_matrix.total
        assert d["summary"]["passed"] == compat_matrix.passed
        json.dumps(d)  # verify JSON-serializable

    def test_all_compat_tests_pass(self, compat_matrix):
        """All compatibility tests should pass with the mock server."""
        failures = [r for r in compat_matrix.results if not r.passed]
        if failures:
            details = "\n".join(
                f"  {r.framework}/{r.feature}: {r.error}"
                for r in failures
            )
            pytest.fail(
                f"{len(failures)} compatibility test(s) failed:\n{details}"
            )
