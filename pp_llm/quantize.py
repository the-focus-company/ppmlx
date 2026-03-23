from __future__ import annotations
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


@dataclass
class QuantizeConfig:
    """Configuration for model quantization."""
    bits: Literal[2, 3, 4, 6, 8] = 4
    group_size: int = 64
    output_path: Path | None = None
    upload_repo: str | None = None
    hf_token: str | None = None


class QuantizationError(Exception):
    pass


def _get_output_path(hf_path: str, bits: int) -> Path:
    """Generate default output path for a quantized model."""
    try:
        from pp_llm.config import get_pp_llm_dir
        models_dir = get_pp_llm_dir() / "models"
    except ImportError:
        models_dir = Path.home() / ".pp-llm" / "models"
    safe_name = hf_path.replace("/", "--")
    return models_dir / f"{safe_name}-{bits}bit"


def _resolve_source(hf_path_or_alias: str) -> str:
    """Resolve alias to repo ID if needed."""
    try:
        from pp_llm.models import resolve_alias
        return resolve_alias(hf_path_or_alias)
    except Exception:
        return hf_path_or_alias


def quantize(
    hf_path_or_alias: str,
    config: QuantizeConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """
    Quantize a HuggingFace model to MLX format.

    Returns the local path where the quantized model is saved.
    """
    if config is None:
        config = QuantizeConfig()

    repo_id = _resolve_source(hf_path_or_alias)
    output_path = config.output_path or _get_output_path(repo_id, config.bits)
    output_path.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Converting {repo_id} to {config.bits}-bit MLX format...")

    success = _try_python_api(repo_id, output_path, config)
    if not success:
        success = _try_subprocess(repo_id, output_path, config, progress_callback)

    if not success:
        raise QuantizationError(
            f"Failed to quantize {repo_id}. "
            "Ensure mlx-lm is installed and the model is accessible."
        )

    if progress_callback:
        progress_callback(f"Saved to {output_path}")

    if config.upload_repo:
        _upload_to_hub(output_path, config.upload_repo, config.hf_token, progress_callback)

    _create_alias(repo_id, output_path, config.bits)

    return output_path


def _try_python_api(repo_id: str, output_path: Path, config: QuantizeConfig) -> bool:
    """Try calling mlx_lm.convert Python API."""
    try:
        import mlx_lm
        convert_fn = getattr(mlx_lm, "convert", None)
        if convert_fn is None:
            return False
        convert_fn(
            hf_path=repo_id,
            mlx_path=str(output_path),
            quantize=True,
            q_bits=config.bits,
            q_group_size=config.group_size,
        )
        return True
    except Exception as e:
        print(f"[pp-llm] Python API failed: {e}, trying subprocess...", file=sys.stderr)
        return False


def _try_subprocess(
    repo_id: str,
    output_path: Path,
    config: QuantizeConfig,
    progress_callback: Callable[[str], None] | None,
) -> bool:
    """Fall back to subprocess: python -m mlx_lm.convert ..."""
    try:
        cmd = [
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", repo_id,
            "--mlx-path", str(output_path),
            "--quantize",
            "--q-bits", str(config.bits),
            "--q-group-size", str(config.group_size),
        ]
        if config.hf_token:
            cmd += ["--hf-token", config.hf_token]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[pp-llm] mlx_lm.convert error: {result.stderr}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[pp-llm] Subprocess fallback failed: {e}", file=sys.stderr)
        return False


def _upload_to_hub(
    local_path: Path,
    repo_id: str,
    token: str | None,
    progress_callback: Callable[[str], None] | None,
) -> None:
    """Upload quantized model to HuggingFace Hub."""
    try:
        from huggingface_hub import upload_folder
        if progress_callback:
            progress_callback(f"Uploading to {repo_id}...")
        upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            token=token,
        )
    except Exception as e:
        print(f"[pp-llm] Upload failed: {e}", file=sys.stderr)


def _create_alias(repo_id: str, output_path: Path, bits: int) -> None:
    """Auto-create a user alias for the quantized model."""
    try:
        from pp_llm.models import save_user_alias
        short = repo_id.split("/")[-1].lower()
        alias = f"{short}-{bits}bit"
        save_user_alias(alias, str(output_path))
    except Exception:
        pass
