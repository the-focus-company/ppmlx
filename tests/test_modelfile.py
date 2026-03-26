"""Tests for ppmlx.modelfile -- covers all parsing scenarios."""
from __future__ import annotations

import json

import pytest

from ppmlx.modelfile import (
    ModelfileConfig,
    ModelfileParseError,
    delete_modelfile,
    list_modelfiles,
    load_modelfile,
    parse_modelfile,
    save_modelfile,
)


# ---------------------------------------------------------------------------
# 1. Minimal modelfile -- only FROM line
# ---------------------------------------------------------------------------
def test_minimal_modelfile():
    cfg = parse_modelfile("FROM qwen3.5:0.8b", name="mymodel")
    assert cfg.from_model == "qwen3.5:0.8b"
    assert cfg.name == "mymodel"
    assert cfg.system is None
    assert cfg.template is None
    assert cfg.adapter is None
    assert cfg.license is None
    assert cfg.parameters == {}


# ---------------------------------------------------------------------------
# 2. Full modelfile -- all directives present (including ADAPTER)
# ---------------------------------------------------------------------------
def test_full_modelfile():
    text = '''\
FROM llama3:8b
SYSTEM """You are a helpful assistant."""
PARAMETER temperature 0.7
PARAMETER top_k 40
TEMPLATE """{{ .System }}{{ .Prompt }}"""
ADAPTER /path/to/lora-adapter
LICENSE """MIT"""
'''
    cfg = parse_modelfile(text, name="full")
    assert cfg.from_model == "llama3:8b"
    assert cfg.system == "You are a helpful assistant."
    assert cfg.parameters["temperature"] == 0.7
    assert cfg.parameters["top_k"] == 40
    assert cfg.template == "{{ .System }}{{ .Prompt }}"
    assert cfg.adapter == "/path/to/lora-adapter"
    assert cfg.license == "MIT"


# ---------------------------------------------------------------------------
# 3. Triple-quoted multiline SYSTEM
# ---------------------------------------------------------------------------
def test_triple_quoted_system():
    text = '''\
FROM base-model
SYSTEM """
Line one.
Line two.
Line three.
"""
'''
    cfg = parse_modelfile(text)
    assert "Line one." in cfg.system
    assert "Line two." in cfg.system
    assert "Line three." in cfg.system


# ---------------------------------------------------------------------------
# 4. Multiple PARAMETER lines accumulate
# ---------------------------------------------------------------------------
def test_multiple_parameters():
    text = '''\
FROM model
PARAMETER temperature 0.9
PARAMETER top_p 0.95
PARAMETER seed 42
PARAMETER num_predict 512
'''
    cfg = parse_modelfile(text)
    assert cfg.parameters["temperature"] == 0.9
    assert cfg.parameters["top_p"] == 0.95
    assert cfg.parameters["seed"] == 42
    assert cfg.parameters["num_predict"] == 512


# ---------------------------------------------------------------------------
# 5. Parameter type coercion
# ---------------------------------------------------------------------------
def test_parameter_types():
    text = '''\
FROM model
PARAMETER temperature 0.5
PARAMETER top_k 50
PARAMETER stop ["\\n", "Human:"]
PARAMETER repeat_penalty 1.1
'''
    cfg = parse_modelfile(text)
    assert isinstance(cfg.parameters["temperature"], float)
    assert cfg.parameters["temperature"] == 0.5
    assert isinstance(cfg.parameters["top_k"], int)
    assert cfg.parameters["top_k"] == 50
    assert isinstance(cfg.parameters["stop"], list)
    assert cfg.parameters["stop"] == ["\n", "Human:"]
    assert isinstance(cfg.parameters["repeat_penalty"], float)


def test_parameter_bool():
    text = "FROM model\nPARAMETER myflag true\nPARAMETER otherflag false\n"
    cfg = parse_modelfile(text)
    assert cfg.parameters["myflag"] is True
    assert cfg.parameters["otherflag"] is False


# ---------------------------------------------------------------------------
# 6. Comments are ignored
# ---------------------------------------------------------------------------
def test_comments_ignored():
    text = '''\
# This is a comment
FROM mymodel
# Another comment
PARAMETER temperature 0.3
'''
    cfg = parse_modelfile(text)
    assert cfg.from_model == "mymodel"
    assert cfg.parameters["temperature"] == 0.3


# ---------------------------------------------------------------------------
# 7. Case-insensitive directives
# ---------------------------------------------------------------------------
def test_case_insensitive_directives():
    text = '''\
from llama3
system """Hello"""
parameter temperature 0.8
'''
    cfg = parse_modelfile(text)
    assert cfg.from_model == "llama3"
    assert cfg.system == "Hello"
    assert cfg.parameters["temperature"] == 0.8


def test_mixed_case_directives():
    text = '''\
From some-model
SYSTEM """Hi"""
Parameter top_k 10
'''
    cfg = parse_modelfile(text)
    assert cfg.from_model == "some-model"
    assert cfg.system == "Hi"
    assert cfg.parameters["top_k"] == 10


# ---------------------------------------------------------------------------
# 8. Missing FROM raises ModelfileParseError
# ---------------------------------------------------------------------------
def test_missing_from_raises():
    with pytest.raises(ModelfileParseError, match="FROM"):
        parse_modelfile("PARAMETER temperature 0.5\n")


# ---------------------------------------------------------------------------
# 9. Unknown directive is ignored (no exception)
# ---------------------------------------------------------------------------
def test_unknown_directive_ignored(capsys):
    text = "FROM model\nUNKNOWNDIRECTIVE something\n"
    cfg = parse_modelfile(text)  # must not raise
    assert cfg.from_model == "model"
    captured = capsys.readouterr()
    assert "Warning" in captured.err


# ---------------------------------------------------------------------------
# 10. Save / load roundtrip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_home):
    text = '''\
FROM qwen3.5:0.8b
SYSTEM """You are concise."""
PARAMETER temperature 0.4
PARAMETER top_k 20
'''
    cfg = parse_modelfile(text, name="roundtrip")
    save_modelfile("roundtrip", cfg)

    loaded = load_modelfile("roundtrip")
    assert loaded is not None
    assert loaded.from_model == cfg.from_model
    assert loaded.system == cfg.system
    assert loaded.parameters == cfg.parameters
    assert loaded.name == "roundtrip"


# ---------------------------------------------------------------------------
# 11. list_modelfiles returns all saved names
# ---------------------------------------------------------------------------
def test_list_modelfiles(tmp_home):
    cfg1 = parse_modelfile("FROM alpha", name="alpha")
    cfg2 = parse_modelfile("FROM beta", name="beta")
    save_modelfile("alpha", cfg1)
    save_modelfile("beta", cfg2)

    names = list_modelfiles()
    assert "alpha" in names
    assert "beta" in names
    assert len(names) == 2


# ---------------------------------------------------------------------------
# 12. delete_modelfile -- save, delete, load returns None
# ---------------------------------------------------------------------------
def test_delete_modelfile(tmp_home):
    cfg = parse_modelfile("FROM deleteme", name="deleteme")
    save_modelfile("deleteme", cfg)

    result = delete_modelfile("deleteme")
    assert result is True

    assert load_modelfile("deleteme") is None


# ---------------------------------------------------------------------------
# 13. delete_nonexistent returns False
# ---------------------------------------------------------------------------
def test_delete_nonexistent(tmp_home):
    result = delete_modelfile("does-not-exist")
    assert result is False


# ---------------------------------------------------------------------------
# 14. Inline (single-line) SYSTEM without triple quotes
# ---------------------------------------------------------------------------
def test_inline_system():
    text = "FROM model\nSYSTEM You are a pirate.\n"
    cfg = parse_modelfile(text)
    assert cfg.system == "You are a pirate."


# ---------------------------------------------------------------------------
# 15. Empty / whitespace-only / comments-only -> ModelfileParseError
# ---------------------------------------------------------------------------
def test_empty_modelfile():
    with pytest.raises(ModelfileParseError):
        parse_modelfile("")


def test_whitespace_only_modelfile():
    with pytest.raises(ModelfileParseError):
        parse_modelfile("   \n\n   \n")


def test_comments_only_modelfile():
    with pytest.raises(ModelfileParseError):
        parse_modelfile("# just a comment\n# another comment\n")


# ---------------------------------------------------------------------------
# 16. TEMPLATE directive -- single-line
# ---------------------------------------------------------------------------
def test_template_single_line():
    text = "FROM model\nTEMPLATE {{ .System }}{{ .Prompt }}\n"
    cfg = parse_modelfile(text)
    assert cfg.template == "{{ .System }}{{ .Prompt }}"


# ---------------------------------------------------------------------------
# 17. TEMPLATE directive -- triple-quoted multiline
# ---------------------------------------------------------------------------
def test_template_multiline():
    text = '''\
FROM model
TEMPLATE """
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}
"""
'''
    cfg = parse_modelfile(text)
    assert "{% for message in messages %}" in cfg.template
    assert "{{ message.role }}" in cfg.template


# ---------------------------------------------------------------------------
# 18. ADAPTER directive -- basic path
# ---------------------------------------------------------------------------
def test_adapter_basic():
    text = "FROM model\nADAPTER /path/to/adapter\n"
    cfg = parse_modelfile(text)
    assert cfg.adapter == "/path/to/adapter"


def test_adapter_relative_path():
    text = "FROM model\nADAPTER ./adapters/my-lora\n"
    cfg = parse_modelfile(text)
    assert cfg.adapter == "./adapters/my-lora"


def test_adapter_huggingface_repo():
    text = "FROM model\nADAPTER org/my-lora-adapter\n"
    cfg = parse_modelfile(text)
    assert cfg.adapter == "org/my-lora-adapter"


def test_adapter_case_insensitive():
    text = "FROM model\nAdapter /some/path\n"
    cfg = parse_modelfile(text)
    assert cfg.adapter == "/some/path"


# ---------------------------------------------------------------------------
# 19. ADAPTER + TEMPLATE together
# ---------------------------------------------------------------------------
def test_adapter_and_template_together():
    text = '''\
FROM llama3:8b
SYSTEM """Custom assistant."""
TEMPLATE """{% for m in messages %}{{ m.content }}{% endfor %}"""
ADAPTER /path/to/lora
PARAMETER temperature 0.5
'''
    cfg = parse_modelfile(text, name="combo")
    assert cfg.from_model == "llama3:8b"
    assert cfg.system == "Custom assistant."
    assert cfg.template == "{% for m in messages %}{{ m.content }}{% endfor %}"
    assert cfg.adapter == "/path/to/lora"
    assert cfg.parameters["temperature"] == 0.5


# ---------------------------------------------------------------------------
# 20. Save / load roundtrip with ADAPTER and TEMPLATE
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_with_adapter_template(tmp_home):
    text = '''\
FROM llama3:8b
SYSTEM """You are concise."""
TEMPLATE """{{ .System }}{{ .Prompt }}"""
ADAPTER /models/my-lora
PARAMETER temperature 0.4
'''
    cfg = parse_modelfile(text, name="full-roundtrip")
    save_modelfile("full-roundtrip", cfg)

    loaded = load_modelfile("full-roundtrip")
    assert loaded is not None
    assert loaded.from_model == cfg.from_model
    assert loaded.system == cfg.system
    assert loaded.template == cfg.template
    assert loaded.adapter == cfg.adapter
    assert loaded.parameters == cfg.parameters
    assert loaded.name == "full-roundtrip"


# ---------------------------------------------------------------------------
# 21. ModelfileConfig to_dict / from_dict with adapter
# ---------------------------------------------------------------------------
def test_config_dict_roundtrip():
    cfg = ModelfileConfig(
        name="test",
        from_model="llama3:8b",
        system="Be helpful.",
        template="{{ .Prompt }}",
        adapter="/path/to/adapter",
        license="MIT",
        parameters={"temperature": 0.5, "top_k": 40},
    )
    d = cfg.to_dict()
    assert d["adapter"] == "/path/to/adapter"
    assert d["template"] == "{{ .Prompt }}"

    restored = ModelfileConfig.from_dict(d)
    assert restored.adapter == cfg.adapter
    assert restored.template == cfg.template
    assert restored.system == cfg.system
    assert restored.parameters == cfg.parameters


def test_config_from_dict_missing_optional_fields():
    """from_dict should handle missing optional fields gracefully."""
    d = {"name": "minimal", "from_model": "base"}
    cfg = ModelfileConfig.from_dict(d)
    assert cfg.adapter is None
    assert cfg.template is None
    assert cfg.system is None
    assert cfg.license is None
    assert cfg.parameters == {}


# ---------------------------------------------------------------------------
# 22. resolve_alias falls back to modelfile configs
# ---------------------------------------------------------------------------
def test_resolve_alias_modelfile_fallback(tmp_home):
    """resolve_alias should find a model defined via a saved modelfile."""
    from ppmlx.models import resolve_alias

    # Save a modelfile config pointing to a known alias
    cfg = parse_modelfile("FROM qwen3.5:0.8b", name="my-custom")
    save_modelfile("my-custom", cfg)

    # resolve_alias("my-custom") should resolve through the modelfile's FROM
    resolved = resolve_alias("my-custom")
    # Should resolve to the same repo as qwen3.5:0.8b
    expected = resolve_alias("qwen3.5:0.8b")
    assert resolved == expected


def test_resolve_alias_modelfile_direct_repo(tmp_home):
    """Modelfile pointing to a direct HF repo ID should resolve."""
    from ppmlx.models import resolve_alias

    cfg = parse_modelfile("FROM mlx-community/some-model", name="direct-repo")
    save_modelfile("direct-repo", cfg)

    resolved = resolve_alias("direct-repo")
    assert resolved == "mlx-community/some-model"


# ---------------------------------------------------------------------------
# 23. JSON persistence format
# ---------------------------------------------------------------------------
def test_saved_json_includes_adapter(tmp_home):
    """The persisted JSON file should contain the adapter field."""
    cfg = parse_modelfile(
        "FROM model\nADAPTER /my/adapter\n", name="json-check"
    )
    path = save_modelfile("json-check", cfg)

    data = json.loads(path.read_text())
    assert data["adapter"] == "/my/adapter"
    assert "adapter" in data


# ---------------------------------------------------------------------------
# 24. ADAPTER without a value is not an error (treated as unknown if empty)
# ---------------------------------------------------------------------------
def test_adapter_requires_value():
    """ADAPTER with no value should be treated as unknown directive."""
    text = "FROM model\nADAPTER\n"
    cfg = parse_modelfile(text)
    # "ADAPTER" alone (without trailing space) does not match "ADAPTER " prefix,
    # so it falls through to the unknown-directive warning
    assert cfg.adapter is None
