# Tests for quantize.py (all mlx_lm calls mocked)
from __future__ import annotations
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
import subprocess
import pytest


def _make_mlx_lm_mock(convert_mock=None):
    """Create a fresh mlx_lm module mock with optional convert function."""
    fake = types.ModuleType("mlx_lm")
    if convert_mock is not None:
        fake.convert = convert_mock
    return fake


def test_get_output_path(tmp_home, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_home))
    monkeypatch.setitem(sys.modules, "ppmlx.config", None)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    result = qmod._get_output_path("org/repo", 4)
    expected = tmp_home / ".ppmlx" / "models" / "org--repo-4bit"
    assert result == expected



def test_quantize_calls_python_api(tmp_home, monkeypatch):
    convert_mock = MagicMock()
    fake_mlx_lm = _make_mlx_lm_mock(convert_mock)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    out_path = tmp_home / "quantized"
    qmod.quantize("org/mymodel", qmod.QuantizeConfig(bits=4, output_path=out_path))

    convert_mock.assert_called_once_with(
        hf_path="org/mymodel",
        mlx_path=str(out_path),
        quantize=True,
        q_bits=4,
        q_group_size=64,
    )



def test_quantize_fallback_to_subprocess(tmp_home, monkeypatch):
    fake_mlx_lm = _make_mlx_lm_mock()  # no convert attr → Python API returns False
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stderr = ""
    run_mock = MagicMock(return_value=fake_result)
    monkeypatch.setattr(subprocess, "run", run_mock)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    out_path = tmp_home / "quantized"
    qmod.quantize("org/mymodel", qmod.QuantizeConfig(bits=4, output_path=out_path))

    assert run_mock.called
    call_args = run_mock.call_args[0][0]
    assert "--quantize" in call_args
    assert "--hf-path" in call_args
    assert "org/mymodel" in call_args



def test_quantize_creates_alias(tmp_home, monkeypatch):
    convert_mock = MagicMock()
    fake_mlx_lm = _make_mlx_lm_mock(convert_mock)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    saved_aliases: dict[str, str] = {}

    def fake_save_user_alias(alias: str, path: str) -> None:
        saved_aliases[alias] = path

    fake_models = types.ModuleType("ppmlx.models")
    fake_models.save_user_alias = fake_save_user_alias
    fake_models.resolve_alias = lambda x: x
    monkeypatch.setitem(sys.modules, "ppmlx.models", fake_models)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    out_path = tmp_home / "quantized"
    qmod.quantize("org/mymodel", qmod.QuantizeConfig(bits=4, output_path=out_path))

    assert "mymodel-4bit" in saved_aliases



def test_quantize_with_upload(tmp_home, monkeypatch):
    convert_mock = MagicMock()
    fake_mlx_lm = _make_mlx_lm_mock(convert_mock)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    upload_mock = MagicMock()
    fake_hf_hub = types.ModuleType("huggingface_hub")
    fake_hf_hub.upload_folder = upload_mock
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    out_path = tmp_home / "quantized"
    qmod.quantize(
        "org/mymodel",
        qmod.QuantizeConfig(bits=4, output_path=out_path, upload_repo="myorg/mymodel-4bit"),
    )

    upload_mock.assert_called_once_with(
        repo_id="myorg/mymodel-4bit",
        folder_path=str(out_path),
        token=None,
    )



def test_quantize_config_defaults():
    from ppmlx.quantize import QuantizeConfig
    c = QuantizeConfig()
    assert c.bits == 4
    assert c.group_size == 64
    assert c.output_path is None
    assert c.upload_repo is None
    assert c.hf_token is None



def test_quantize_invalid_raises(tmp_home, monkeypatch):
    fake_mlx_lm = _make_mlx_lm_mock()  # no convert → Python API returns False
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stderr = "some error"
    run_mock = MagicMock(return_value=fake_result)
    monkeypatch.setattr(subprocess, "run", run_mock)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    out_path = tmp_home / "quantized"
    with pytest.raises(qmod.QuantizationError):
        qmod.quantize("org/badmodel", qmod.QuantizeConfig(output_path=out_path))



def test_resolve_source_with_alias(tmp_home, monkeypatch):
    resolve_mock = MagicMock(return_value="org/real-model")

    fake_models = types.ModuleType("ppmlx.models")
    fake_models.resolve_alias = resolve_mock
    fake_models.save_user_alias = MagicMock()
    monkeypatch.setitem(sys.modules, "ppmlx.models", fake_models)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    result = qmod._resolve_source("my-alias")
    resolve_mock.assert_called_once_with("my-alias")
    assert result == "org/real-model"



def test_quantize_returns_path(tmp_home, monkeypatch):
    convert_mock = MagicMock()
    fake_mlx_lm = _make_mlx_lm_mock(convert_mock)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    import importlib
    import ppmlx.quantize as qmod
    importlib.reload(qmod)

    out_path = tmp_home / "quantized"
    returned = qmod.quantize("org/mymodel", qmod.QuantizeConfig(bits=4, output_path=out_path))

    assert isinstance(returned, Path)
    assert returned == out_path
    assert returned.exists()
