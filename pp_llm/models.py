from __future__ import annotations
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any


# ── Built-in alias map (exact HuggingFace repo IDs from spec) ──────────

DEFAULT_ALIASES: dict[str, str] = {
    # Qwen 3.5 (Feb-Mar 2026)
    "qwen3.5:0.8b":       "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
    "qwen3.5:2b":         "mlx-community/Qwen3.5-2B-MLX-4bit",
    "qwen3.5:4b":         "mlx-community/Qwen3.5-4B-MLX-4bit",
    "qwen3.5:9b":         "mlx-community/Qwen3.5-9B-MLX-4bit",
    "qwen3.5:27b":        "mlx-community/Qwen3.5-27B-4bit",
    "qwen3.5:35b-a3b":    "mlx-community/Qwen3.5-35B-A3B-4bit",
    "qwen3.5:122b-a10b":  "mlx-community/Qwen3.5-122B-A10B-4bit",
    # Qwen 3
    "qwen3:4b":           "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "qwen3:8b":           "mlx-community/Qwen3-8B-4bit",
    "qwen3:14b":          "mlx-community/Qwen3-14B-4bit",
    "qwen3:32b":          "mlx-community/Qwen3-32B-4bit",
    "qwen3:30b-a3b":      "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "qwen3-coder:80b-a3b":"mlx-community/Qwen3-Coder-Next-80B-A3B-4bit",
    # Gemma 3
    "gemma3:4b":          "mlx-community/gemma-3-4b-it-4bit",
    "gemma3:4b-text":     "mlx-community/gemma-3-text-4b-it-4bit",
    "gemma3:12b":         "mlx-community/gemma-3-12b-it-4bit",
    "gemma3:27b":         "mlx-community/gemma-3-27b-it-qat-4bit",
    # GPT-OSS
    "gpt-oss:20b":        "mlx-community/gpt-oss-20b-4bit",
    "gpt-oss:120b":       "mlx-community/gpt-oss-120b-4bit",
    # DeepSeek R1
    "deepseek-r1:8b":     "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit",
    "deepseek-r1:14b":    "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    "deepseek-r1:32b":    "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
    # Mistral
    "mistral:7b":         "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mistral-small:24b":  "mlx-community/Mistral-Small-3.1-Text-24B-Instruct-2503-8bit",
    "devstral:24b":       "mlx-community/Devstral-Small-24B-4bit",
    # Phi-4
    "phi4:mini":          "mlx-community/Phi-4-mini-instruct-4bit",
    # Embedding models
    "embed:all-minilm":   "mlx-community/all-MiniLM-L6-v2-4bit",
    "embed:qwen3-0.6b":   "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "embed:qwen3-4b":     "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    "embed:qwen3-8b":     "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
}

# Patterns for routing
_VISION_INDICATORS = ["-VL-", "-vlm", "Qwen3.5-", "gemma-3-"]
_TEXT_ONLY_INDICATORS = ["-text-", "-Text-", "OptiQ"]
_EMBED_PREFIXES = ("embed:",)


class ModelNotFoundError(Exception):
    pass


def _get_pp_llm_dir() -> Path:
    try:
        from pp_llm.config import get_pp_llm_dir
        return get_pp_llm_dir()
    except ImportError:
        return Path.home() / ".pp-llm"


def _get_models_dir() -> Path:
    d = _get_pp_llm_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_aliases_file() -> Path:
    return _get_pp_llm_dir() / "aliases.json"


def load_user_aliases() -> dict[str, str]:
    """Load user-defined aliases from ~/.pp-llm/aliases.json."""
    p = _get_aliases_file()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_user_alias(name: str, repo_id: str) -> None:
    """Add or update a user alias."""
    aliases = load_user_aliases()
    aliases[name] = repo_id
    p = _get_aliases_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(aliases, indent=2))


def remove_user_alias(name: str) -> bool:
    """Remove a user alias. Returns True if it existed."""
    aliases = load_user_aliases()
    if name in aliases:
        del aliases[name]
        _get_aliases_file().write_text(json.dumps(aliases, indent=2))
        return True
    return False


def all_aliases() -> dict[str, str]:
    """Return merged dict: user aliases override defaults."""
    merged = dict(DEFAULT_ALIASES)
    merged.update(load_user_aliases())
    return merged


def resolve_alias(name: str) -> str:
    """
    Resolve a model name to a HuggingFace repo ID.

    1. If name contains '/' -> direct repo ID, return as-is
    2. Check user aliases
    3. Check DEFAULT_ALIASES
    4. Prefix match (e.g. 'qwen3.5' -> smallest qwen3.5 variant)
    5. Raise ModelNotFoundError with helpful message
    """
    if "/" in name:
        return name

    user = load_user_aliases()
    if name in user:
        return user[name]
    if name in DEFAULT_ALIASES:
        return DEFAULT_ALIASES[name]

    # Prefix match: find smallest variant
    matches = [(k, v) for k, v in DEFAULT_ALIASES.items() if k.startswith(name + ":") or k == name]
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    available = sorted({**DEFAULT_ALIASES, **user}.keys())
    raise ModelNotFoundError(
        f"Unknown model: '{name}'\n"
        f"Available aliases: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}\n"
        f"Or use a HuggingFace repo ID directly (e.g. 'mlx-community/model-name')."
    )


def is_vision_model(repo_id: str) -> bool:
    """Detect if a model should be loaded with mlx-vlm."""
    if any(ind in repo_id for ind in _TEXT_ONLY_INDICATORS):
        return False
    return any(ind in repo_id for ind in _VISION_INDICATORS)


def is_embed_model(alias_or_repo: str) -> bool:
    """Detect if a model is an embedding model."""
    if alias_or_repo.startswith(_EMBED_PREFIXES):
        return True
    lower = alias_or_repo.lower()
    return any(p in lower for p in ["embed", "embedding", "minilm", "bge-", "nomic-"])


def repo_to_local_name(repo_id: str) -> str:
    """Convert 'org/repo' -> 'org--repo' for local directory name."""
    return repo_id.replace("/", "--")


def _get_repo_size(repo_id: str, token: str | None = None) -> int | None:
    """Return total download size in bytes by listing the repo tree. Returns None on failure."""
    try:
        from huggingface_hub import list_repo_tree
        return sum(
            getattr(f, "size", 0) or 0
            for f in list_repo_tree(repo_id, token=token, recursive=True)
        )
    except Exception:
        return None


def download_model(alias_or_repo: str, token: str | None = None) -> Path:
    """
    Download a model from HuggingFace Hub with a uv-style Rich progress bar.
    Returns local path.
    """
    from rich.progress import (
        Progress, BarColumn, DownloadColumn,
        TransferSpeedColumn, TimeRemainingColumn, TextColumn,
    )

    repo_id = resolve_alias(alias_or_repo)
    local_name = repo_to_local_name(repo_id)
    local_path = _get_models_dir() / local_name

    if local_path.exists() and any(local_path.iterdir()):
        return local_path

    local_path.mkdir(parents=True, exist_ok=True)
    total = _get_repo_size(repo_id, token)

    stop = threading.Event()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=1,
        transient=False,
    ) as progress:
        task = progress.add_task(f"↓ {alias_or_repo}", total=total)

        def _monitor() -> None:
            while not stop.is_set():
                try:
                    sz = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
                    progress.update(task, completed=sz)
                except Exception:
                    pass
                stop.wait(1.0)

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()

        # Suppress huggingface_hub's own tqdm bars while our progress bar is active
        prev_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                token=token,
                ignore_patterns=["*.md", "*.txt", "original/*"],
            )
        except Exception as e:
            shutil.rmtree(local_path, ignore_errors=True)
            raise ModelNotFoundError(f"Failed to download '{repo_id}': {e}") from e
        finally:
            if prev_tqdm is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev_tqdm
            stop.set()
            t.join(timeout=2)
            if total:
                progress.update(task, completed=total)

    return local_path


def get_model_path(alias_or_repo: str) -> Path | None:
    """Return local path if model exists, else None."""
    try:
        repo_id = resolve_alias(alias_or_repo)
    except ModelNotFoundError:
        repo_id = alias_or_repo

    local_name = repo_to_local_name(repo_id)
    p = _get_models_dir() / local_name
    if p.exists() and any(p.iterdir()):
        return p
    return None


def list_local_models() -> list[dict[str, Any]]:
    """List all locally downloaded models."""
    models_dir = _get_models_dir()
    result = []
    if not models_dir.exists():
        return result

    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024 ** 3)
        repo_id = d.name.replace("--", "/", 1)
        aliases_for_model = [k for k, v in all_aliases().items() if v == repo_id]
        result.append({
            "name": d.name,
            "repo_id": repo_id,
            "alias": aliases_for_model[0] if aliases_for_model else repo_id,
            "size_gb": round(size_gb, 2),
            "path": d,
        })
    return result


def remove_model(alias_or_repo: str) -> bool:
    """Remove a locally downloaded model. Returns True if removed."""
    path = get_model_path(alias_or_repo)
    if path is None:
        return False
    shutil.rmtree(path)
    return True
