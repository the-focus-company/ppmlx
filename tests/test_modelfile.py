"""Tests for pp_llm.modelfile — covers all parsing scenarios."""
from __future__ import annotations
import pytest
from pp_llm.modelfile import (
    ModelfileConfig,
    ModelfileParseError,
    delete_modelfile,
    list_modelfiles,
    load_modelfile,
    parse_modelfile,
    save_modelfile,
)


# ---------------------------------------------------------------------------
# 1. Minimal modelfile — only FROM line
# ---------------------------------------------------------------------------
def test_minimal_modelfile():
    cfg = parse_modelfile("FROM qwen3.5:0.8b", name="mymodel")
    assert cfg.from_model == "qwen3.5:0.8b"
    assert cfg.name == "mymodel"
    assert cfg.system is None
    assert cfg.template is None
    assert cfg.license is None
    assert cfg.parameters == {}


# ---------------------------------------------------------------------------
# 2. Full modelfile — all directives present
# ---------------------------------------------------------------------------
def test_full_modelfile():
    text = '''\
FROM llama3:8b
SYSTEM """You are a helpful assistant."""
PARAMETER temperature 0.7
PARAMETER top_k 40
TEMPLATE """{{ .System }}{{ .Prompt }}"""
LICENSE """MIT"""
'''
    cfg = parse_modelfile(text, name="full")
    assert cfg.from_model == "llama3:8b"
    assert cfg.system == "You are a helpful assistant."
    assert cfg.parameters["temperature"] == 0.7
    assert cfg.parameters["top_k"] == 40
    assert cfg.template == "{{ .System }}{{ .Prompt }}"
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
# 12. delete_modelfile — save, delete, load returns None
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
# 15. Empty / whitespace-only / comments-only → ModelfileParseError
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
