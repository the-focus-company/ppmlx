"""Tests for the built-in web chat playground."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from ppmlx.server import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Route tests ──────────────────────────────────────────────────────


def test_playground_returns_200(client):
    """GET /ui returns 200 with HTML content."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_playground_contains_required_elements(client):
    """The HTML includes the essential UI elements."""
    response = client.get("/ui")
    html = response.text
    assert "<title>ppmlx Playground</title>" in html
    assert 'id="chat-input"' in html
    assert 'id="model-select"' in html
    assert 'id="send-btn"' in html
    assert 'id="chat-area"' in html


def test_playground_has_js_functions(client):
    """The HTML includes the key JS functions for chat functionality."""
    response = client.get("/ui")
    html = response.text
    assert "function sendMessage()" in html
    assert "function newConversation()" in html
    assert "function loadModels()" in html
    assert "function toggleTheme()" in html
    assert "function exportConversation()" in html


def test_playground_calls_api_endpoint(client):
    """The JS references the correct API endpoints."""
    response = client.get("/ui")
    html = response.text
    assert "/v1/chat/completions" in html
    assert "/v1/models" in html


def test_playground_has_streaming_support(client):
    """The HTML/JS includes streaming (SSE) support."""
    response = client.get("/ui")
    html = response.text
    assert "stream: true" in html
    assert "data: [DONE]" in html or "[DONE]" in html


def test_playground_has_theme_toggle(client):
    """The UI supports dark/light theme switching."""
    response = client.get("/ui")
    html = response.text
    assert 'data-theme' in html
    assert "ppmlx_theme" in html


def test_playground_has_conversation_management(client):
    """The UI supports conversation management."""
    response = client.get("/ui")
    html = response.text
    assert "ppmlx_convs" in html
    assert "function clearConversation()" in html


# ── CLI command tests ────────────────────────────────────────────────

def test_ui_command_exists():
    """The 'ui' command is registered in the CLI app."""
    from typer.testing import CliRunner
    from ppmlx.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "playground" in result.output.lower() or "browser" in result.output.lower()


def test_ui_command_has_options():
    """The 'ui' command accepts --host, --port, --no-browser."""
    from typer.testing import CliRunner
    from ppmlx.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--no-browser" in result.output


# ── Playground module unit tests ─────────────────────────────────────

def test_playground_html_is_valid():
    """The HTML template is a non-empty string."""
    from ppmlx.playground import _PLAYGROUND_HTML
    assert isinstance(_PLAYGROUND_HTML, str)
    assert len(_PLAYGROUND_HTML) > 1000
    assert _PLAYGROUND_HTML.strip().startswith("<!DOCTYPE html>")
    assert _PLAYGROUND_HTML.strip().endswith("</html>")
