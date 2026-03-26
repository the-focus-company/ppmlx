"""MCP (Model Context Protocol) server for ppmlx.

Exposes local MLX models as tools so that Claude Code, Cursor, and other
AI IDEs can use them via the stdio transport.

Usage:
    ppmlx mcp              # start MCP server on stdio
    ppmlx mcp --model llama3   # pre-load a model
"""
from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

log = logging.getLogger("ppmlx.mcp")

mcp_server = FastMCP(
    name="ppmlx",
    instructions=(
        "ppmlx MCP server — run LLMs locally on Apple Silicon via MLX. "
        "Use the generate tool for chat completions, list_models to discover "
        "available models, and embed for text embeddings."
    ),
)


def _resolve(model: str) -> str:
    """Resolve a model alias to a HuggingFace repo ID."""
    from ppmlx.models import resolve_alias

    return resolve_alias(model)


@mcp_server.tool(
    name="generate",
    description=(
        "Generate a chat completion using a local MLX model. "
        "Returns the model's text response."
    ),
)
def generate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Run chat completion on a local MLX model.

    Args:
        model: Model name or alias (e.g. "llama3", "qwen3.5:4b").
        messages: Chat messages as [{"role": "user", "content": "Hello"}].
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate (None = model default).

    Returns:
        The generated text response.
    """
    from ppmlx.engine import get_engine

    repo_id = _resolve(model)
    engine = get_engine()
    text, _reasoning, _prompt_tokens, _completion_tokens = engine.generate(
        repo_id=repo_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return text


@mcp_server.tool(
    name="list_models",
    description="List available models (both local and from the registry).",
)
def list_models() -> str:
    """List available models.

    Returns:
        JSON array of model objects with alias, repo_id, and downloaded status.
    """
    from ppmlx.models import all_aliases, list_local_models

    aliases = all_aliases()
    local = {m["repo_id"] for m in list_local_models()}

    models = []
    for alias, repo_id in sorted(aliases.items()):
        models.append(
            {
                "alias": alias,
                "repo_id": repo_id,
                "downloaded": repo_id in local,
            }
        )
    return json.dumps(models, indent=2)


@mcp_server.tool(
    name="embed",
    description="Generate text embeddings using a local MLX embedding model.",
)
def embed(model: str, text: str) -> str:
    """Generate an embedding vector for the given text.

    Args:
        model: Embedding model name or alias (e.g. "nomic-embed").
        text: The text to embed.

    Returns:
        JSON array of floats representing the embedding vector.
    """
    from ppmlx.engine_embed import get_embed_engine

    repo_id = _resolve(model)
    engine = get_embed_engine()
    vectors = engine.encode(repo_id=repo_id, texts=[text])
    return json.dumps(vectors[0])


def run_mcp(model: str | None = None) -> None:
    """Run the MCP server on stdio, optionally pre-loading a model."""
    if model:
        from ppmlx.engine import get_engine

        repo_id = _resolve(model)
        log.info("Pre-loading model: %s (%s)", model, repo_id)
        get_engine().load(repo_id)

    mcp_server.run(transport="stdio")
