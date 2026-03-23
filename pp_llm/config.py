from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6767
    cors: bool = True
    max_loaded_models: int = 2


@dataclass
class DefaultsConfig:
    model: str = "qwen3.5:0.8b"
    embed_model: str = "embed:all-minilm"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048


@dataclass
class LoggingConfig:
    enabled: bool = True
    snapshot_interval_seconds: int = 60


@dataclass
class MemoryConfig:
    wired_limit_mb: int = 0


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def get_pp_llm_dir() -> Path:
    """Return ~/.pp-llm, creating it if needed."""
    d = Path.home() / ".pp-llm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_bool(v: str) -> bool:
    return v.lower() not in ("0", "false", "no")


def load_config(cli_overrides: dict[str, Any] | None = None) -> Config:
    """Load config with priority: CLI overrides > env vars > TOML file > defaults."""
    cfg = Config()
    toml_path = Path.home() / ".pp-llm" / "config.toml"
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        _apply_toml(cfg, data)
    except Exception:
        pass
    _apply_env(cfg)
    if cli_overrides:
        _apply_cli(cfg, cli_overrides)
    return cfg


def _apply_toml(cfg: Config, data: dict) -> None:
    if "server" in data:
        s = data["server"]
        if "host" in s: cfg.server.host = str(s["host"])
        if "port" in s: cfg.server.port = int(s["port"])
        if "cors" in s: cfg.server.cors = bool(s["cors"])
        if "max_loaded_models" in s: cfg.server.max_loaded_models = int(s["max_loaded_models"])
    if "defaults" in data:
        d = data["defaults"]
        if "model" in d: cfg.defaults.model = str(d["model"])
        if "embed_model" in d: cfg.defaults.embed_model = str(d["embed_model"])
        if "temperature" in d: cfg.defaults.temperature = float(d["temperature"])
        if "top_p" in d: cfg.defaults.top_p = float(d["top_p"])
        if "max_tokens" in d: cfg.defaults.max_tokens = int(d["max_tokens"])
    if "logging" in data:
        lg = data["logging"]
        if "enabled" in lg: cfg.logging.enabled = bool(lg["enabled"])
        if "snapshot_interval_seconds" in lg:
            cfg.logging.snapshot_interval_seconds = int(lg["snapshot_interval_seconds"])
    if "memory" in data:
        m = data["memory"]
        if "wired_limit_mb" in m: cfg.memory.wired_limit_mb = int(m["wired_limit_mb"])


def _apply_env(cfg: Config) -> None:
    mapping = {
        "PP_LLM_HOST": ("server", "host", str),
        "PP_LLM_PORT": ("server", "port", int),
        "PP_LLM_CORS": ("server", "cors", _parse_bool),
        "PP_LLM_MAX_LOADED_MODELS": ("server", "max_loaded_models", int),
        "PP_LLM_DEFAULT_MODEL": ("defaults", "model", str),
        "PP_LLM_DEFAULT_EMBED_MODEL": ("defaults", "embed_model", str),
        "PP_LLM_TEMP": ("defaults", "temperature", float),
        "PP_LLM_TOP_P": ("defaults", "top_p", float),
        "PP_LLM_MAX_TOKENS": ("defaults", "max_tokens", int),
        "PP_LLM_LOG_ENABLED": ("logging", "enabled", _parse_bool),
        "PP_LLM_LOG_SNAPSHOT_INTERVAL": ("logging", "snapshot_interval_seconds", int),
        "PP_LLM_MEMORY_WIRED_LIMIT": ("memory", "wired_limit_mb", int),
    }
    for env_key, (section, attr, coerce) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(getattr(cfg, section), attr, coerce(val))
            except (ValueError, AttributeError):
                pass


def _apply_cli(cfg: Config, overrides: dict) -> None:
    for key, val in overrides.items():
        if val is None:
            continue
        if key == "host": cfg.server.host = str(val)
        elif key == "port": cfg.server.port = int(val)
        elif key == "cors": cfg.server.cors = bool(val)
        elif key == "model": cfg.defaults.model = str(val)
        elif key == "temperature": cfg.defaults.temperature = float(val)
        elif key == "max_tokens": cfg.defaults.max_tokens = int(val)
