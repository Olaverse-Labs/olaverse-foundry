"""
Backend capability detection — what's available on this machine?
"""
from __future__ import annotations

import sys


def detect_backend() -> dict:
    """
    Check which optional backends are installed and available.

    Returns:
        dict with keys: torch, cuda, mps, peft, accelerate, mergekit,
        safetensors, rapidfuzz, python_version, cuda_version,
        gpu_count, gpu_vram_gb, summary.
    """
    from foundry import __version__ as _foundry_version

    result: dict = {
        "foundry_version": _foundry_version,
        "torch":          False,
        "cuda":           False,
        "mps":            False,
        "peft":           False,
        "accelerate":     False,
        "mergekit":       False,
        "safetensors":    False,
        "rapidfuzz":      False,
        "wandb":          False,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version":  None,
        "cuda_version":   None,
        "gpu_count":      0,
        "gpu_vram_gb":    [],   # list of VRAM per device in GB
    }

    try:
        import torch
        result["torch"]         = True
        result["torch_version"] = torch.__version__
        result["cuda"]          = torch.cuda.is_available()
        result["mps"]           = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if result["cuda"]:
            result["cuda_version"] = torch.version.cuda
            n = torch.cuda.device_count()
            result["gpu_count"]  = n
            result["gpu_vram_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)
                for i in range(n)
            ]
    except ImportError:
        pass

    for pkg, key in [
        ("peft",        "peft"),
        ("accelerate",  "accelerate"),
        ("mergekit",    "mergekit"),
        ("safetensors", "safetensors"),
        ("rapidfuzz",   "rapidfuzz"),
        ("wandb",       "wandb"),
    ]:
        try:
            __import__(pkg)
            result[key] = True
        except (ImportError, TypeError):
            pass

    # Build summary string
    parts = []
    if result["torch"]:
        if result["cuda"]:
            vram = result["gpu_vram_gb"]
            vram_str = "+".join(f"{v}GB" for v in vram) if vram else "?"
            parts.append(
                f"torch {result['torch_version']} "
                f"(CUDA {result['cuda_version']}, {result['gpu_count']}× GPU, {vram_str} VRAM)"
            )
        elif result["mps"]:
            parts.append(f"torch {result['torch_version']} (MPS/Apple Silicon)")
        else:
            parts.append(f"torch {result['torch_version']} (CPU only)")
    else:
        parts.append("no torch — toy/numpy backend only")

    for key in ("peft", "accelerate", "mergekit", "safetensors", "rapidfuzz", "wandb"):
        if result[key]:
            parts.append(key)

    result["summary"] = "  ".join(parts)
    return result


def require_torch(feature: str = "this feature") -> None:
    """Raise a helpful ImportError if torch is not installed."""
    info = detect_backend()
    if not info["torch"]:
        raise ImportError(
            f"{feature} requires torch. "
            "Install with: pip install olaverse-foundry[torch]"
        )
