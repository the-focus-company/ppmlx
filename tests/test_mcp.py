"""Tests for ppmlx.mcp -- MCP server tools."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ppmlx.mcp import mcp_server, generate, list_models, embed


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def test_tools_registered():
    """All three tools are registered on the MCP server."""
    tool_names = set(mcp_server._tool_manager._tools.keys())
    assert "generate" in tool_names
    assert "list_models" in tool_names
    assert "embed" in tool_names


# ---------------------------------------------------------------------------
# generate tool
# ---------------------------------------------------------------------------

@patch("ppmlx.models.resolve_alias", return_value="mlx-community/Meta-Llama-3-8B-Instruct-4bit")
@patch("ppmlx.engine.get_engine")
def test_generate_returns_text(mock_get_engine, mock_resolve):
    """generate() returns the text produced by the engine."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = ("Hello world!", None, 10, 5)
    mock_get_engine.return_value = mock_engine

    result = generate(
        model="llama3",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.5,
        max_tokens=100,
    )

    assert result == "Hello world!"
    mock_resolve.assert_called_once_with("llama3")
    mock_engine.generate.assert_called_once_with(
        repo_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.5,
        max_tokens=100,
    )


@patch("ppmlx.models.resolve_alias", return_value="mlx-community/some-model")
@patch("ppmlx.engine.get_engine")
def test_generate_defaults(mock_get_engine, mock_resolve):
    """generate() uses sensible defaults for temperature and max_tokens."""
    mock_engine = MagicMock()
    mock_engine.generate.return_value = ("Reply", None, 8, 3)
    mock_get_engine.return_value = mock_engine

    result = generate(
        model="some-model",
        messages=[{"role": "user", "content": "Test"}],
    )

    assert result == "Reply"
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] is None


# ---------------------------------------------------------------------------
# list_models tool
# ---------------------------------------------------------------------------

@patch("ppmlx.models.list_local_models")
@patch("ppmlx.models.all_aliases")
def test_list_models_returns_json(mock_all_aliases, mock_local):
    """list_models() returns a JSON array of model info dicts."""
    mock_all_aliases.return_value = {
        "llama3": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "qwen3.5:4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
    }
    mock_local.return_value = [
        {"repo_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit"},
    ]

    result = list_models()
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) == 2

    # Should be sorted by alias
    assert data[0]["alias"] == "llama3"
    assert data[0]["downloaded"] is True
    assert data[1]["alias"] == "qwen3.5:4b"
    assert data[1]["downloaded"] is False


@patch("ppmlx.models.list_local_models")
@patch("ppmlx.models.all_aliases")
def test_list_models_empty(mock_all_aliases, mock_local):
    """list_models() returns an empty array when no models are available."""
    mock_all_aliases.return_value = {}
    mock_local.return_value = []

    result = list_models()
    data = json.loads(result)
    assert data == []


# ---------------------------------------------------------------------------
# embed tool
# ---------------------------------------------------------------------------

@patch("ppmlx.models.resolve_alias", return_value="mlx-community/nomic-embed-text-v1.5")
@patch("ppmlx.engine_embed.get_embed_engine")
def test_embed_returns_vector(mock_get_embed, mock_resolve):
    """embed() returns a JSON array of floats."""
    mock_engine = MagicMock()
    mock_engine.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_get_embed.return_value = mock_engine

    result = embed(model="nomic-embed", text="Hello world")
    vec = json.loads(result)

    assert vec == [0.1, 0.2, 0.3, 0.4]
    mock_resolve.assert_called_once_with("nomic-embed")
    mock_engine.encode.assert_called_once_with(
        repo_id="mlx-community/nomic-embed-text-v1.5",
        texts=["Hello world"],
    )


# ---------------------------------------------------------------------------
# run_mcp / CLI integration
# ---------------------------------------------------------------------------

@patch("ppmlx.mcp.mcp_server")
@patch("ppmlx.models.resolve_alias", return_value="mlx-community/some-model")
@patch("ppmlx.engine.get_engine")
def test_run_mcp_preloads_model(mock_get_engine, mock_resolve, mock_server):
    """run_mcp() pre-loads the specified model, then starts the server."""
    from ppmlx.mcp import run_mcp

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    run_mcp(model="some-model")

    mock_resolve.assert_called_once_with("some-model")
    mock_engine.load.assert_called_once_with("mlx-community/some-model")
    mock_server.run.assert_called_once_with(transport="stdio")


@patch("ppmlx.mcp.mcp_server")
def test_run_mcp_no_model(mock_server):
    """run_mcp() without a model just starts the server."""
    from ppmlx.mcp import run_mcp

    run_mcp(model=None)

    mock_server.run.assert_called_once_with(transport="stdio")
