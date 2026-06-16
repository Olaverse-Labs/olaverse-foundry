"""
Backend capability detection — what's available on this machine?
"""
from __future__ import annotations


def detect_backend() -> dict:
    """
    Check which optional backends are installed and available.

    Returns:
        dict with keys: torch, cuda, mps, peft, accelerate, mergekit, summary.
    """
    result: dict = {
        "torch":       False,
        "cuda":        False,
        "mps":         False,
        "peft":        False,
        "accelerate":  False,
        "mergekit":    False,
    }

    try:
        import torch
        result["torch"] = True
        result["cuda"]  = torch.cuda.is_available()
        result["mps"]   = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        pass

    for pkg, key in [("peft", "peft"), ("accelerate", "accelerate"), ("mergekit", "mergekit")]:
        try:
            __import__(pkg)
            result[key] = True
        except ImportError:
            pass

    # Build summary string
    parts = []
    if result["torch"]:
        if result["cuda"]:
            try:
                import torch
                parts.append(f"torch (CUDA {torch.version.cuda}, {torch.cuda.device_count()}× GPU)")
            except Exception:
                parts.append("torch (CUDA)")
        elif result["mps"]:
            parts.append("torch (MPS/Apple Silicon)")
        else:
            parts.append("torch (CPU only)")
    else:
        parts.append("no torch — toy/numpy backend only")

    for key in ("peft", "accelerate", "mergekit"):
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
