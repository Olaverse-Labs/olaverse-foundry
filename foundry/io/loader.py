from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class SeedStrategy(str, Enum):
    FROM_SCRATCH = "from_scratch"
    PRETRAINED   = "pretrained"


@dataclass
class ModelRef:
    """
    A resolved pointer to a model — HF hub ID, local path, or ID@revision.

    Examples::

        ModelRef.parse("meta-llama/Llama-3.1-8B")
        ModelRef.parse("org/model@abc1234")
        ModelRef.parse("/path/to/local/model")
    """

    repo_id:    str
    revision:   Optional[str]   = None
    local_path: Optional[Path]  = None
    dtype:      str             = "auto"
    device_map: str             = "auto"
    trust_remote_code: bool     = False
    quantize:   Optional[str]   = None      # None | "4bit" | "8bit"

    @classmethod
    def parse(cls, spec: str, **kwargs) -> "ModelRef":
        """Parse 'org/model', 'org/model@rev', or '/local/path'."""
        path = Path(spec)
        if path.exists():
            return cls(repo_id=spec, local_path=path, **kwargs)

        if "@" in spec:
            repo_id, revision = spec.rsplit("@", 1)
        else:
            repo_id, revision = spec, None

        return cls(repo_id=repo_id, revision=revision, **kwargs)

    @property
    def identifier(self) -> str:
        """Return the canonical identifier string."""
        if self.local_path:
            return str(self.local_path)
        if self.revision:
            return f"{self.repo_id}@{self.revision}"
        return self.repo_id

    def validate(self) -> None:
        """
        Check the model exists on the HF hub before any expensive run.
        Raises ValueError if the repo cannot be found.
        """
        if self.local_path:
            if not self.local_path.exists():
                raise ValueError(f"Local path does not exist: {self.local_path}")
            return
        try:
            from huggingface_hub import model_info
            model_info(self.repo_id, revision=self.revision)
        except Exception as exc:
            raise ValueError(
                f"Could not resolve model '{self.identifier}' on the HF hub: {exc}"
            ) from exc



# ── Quantization ───────────────────────────────────────────────────────────

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def resolve_dtype(dtype: str, default: str = "bfloat16"):
    """Map a dtype name to the torch dtype, falling back to ``default``."""
    import torch
    name = _DTYPES.get(dtype, default)
    return getattr(torch, name)


def build_quantization_config(quantize: Optional[str], dtype: str = "bfloat16"):
    """
    Build a ``BitsAndBytesConfig`` for 4-bit or 8-bit loading, or ``None``.

    4-bit uses NF4 with double quantization and a ``dtype`` compute type — the
    configuration that keeps quality closest to bf16 at roughly a quarter of the
    memory. This is what makes a large teacher fit next to a student on one box:
    a 27B teacher is ~54GB in bf16 and ~15GB in 4-bit.

    Quantized weights are frozen — bitsandbytes layers are not trainable in the
    ordinary sense — so use this for *teachers* and for LoRA base models, never
    for a student you intend to full-weight train.

    Raises ValueError for any value other than None / "4bit" / "8bit", and
    ImportError when transformers or bitsandbytes is missing.
    """
    if quantize is None:
        return None
    if quantize not in ("4bit", "8bit"):
        raise ValueError(f"quantize must be None, '4bit', or '8bit'; got {quantize!r}")

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "Quantized loading needs a recent transformers + bitsandbytes. "
            "Install with: pip install olaverse-foundry[torch] bitsandbytes"
        ) from None

    # Check bitsandbytes here rather than letting transformers discover it.
    # BitsAndBytesConfig.post_init() reads importlib.metadata.version(
    # "bitsandbytes") on some transformers versions and not others, so without
    # this the same call raises PackageNotFoundError on one version and
    # succeeds on another — then fails much later inside from_pretrained.
    # Neither message tells the caller what to install.
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            f"{quantize} loading requires bitsandbytes, which is not installed. "
            "Install with: pip install bitsandbytes"
        )

    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=resolve_dtype(dtype),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model(ref: ModelRef, student_class=None, model_class=None) -> Any:
    """
    Load a model from a ModelRef.

    Args:
        ref:           Parsed ModelRef.
        student_class: Unused (kept for API compatibility).
        model_class:   Transformers auto-class to use. Defaults to
                       ``AutoModelForCausalLM``. Pass ``AutoModel`` for
                       encoder-only architectures (BERT, DeBERTa, RoBERTa).

    Returns:
        A loaded model.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoModel  # noqa: F401
    except ImportError:
        raise ImportError(
            "transformers is required for model loading. "
            "Install with: pip install olaverse-foundry[torch]"
        ) from None

    cls = model_class if model_class is not None else AutoModelForCausalLM

    kwargs: dict = {
        "device_map":  ref.device_map,
        "trust_remote_code": ref.trust_remote_code,
    }

    # torch_dtype and quantization_config are mutually exclusive: bitsandbytes
    # owns the storage dtype, and passing both makes transformers warn and
    # silently ignore one of them.
    quant_config = build_quantization_config(
        ref.quantize, ref.dtype if ref.dtype != "auto" else "bfloat16"
    )
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = ref.dtype

    if ref.revision:
        kwargs["revision"] = ref.revision

    src = str(ref.local_path) if ref.local_path else ref.repo_id
    return cls.from_pretrained(src, **kwargs)


def load_tokenizer(ref: ModelRef) -> Any:
    """Load the tokenizer for a ModelRef."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required. "
            "Install with: pip install olaverse-foundry[torch]"
        ) from None
    src = str(ref.local_path) if ref.local_path else ref.repo_id
    kwargs: dict[str, Any] = {"trust_remote_code": ref.trust_remote_code}
    if ref.revision:
        kwargs["revision"] = ref.revision
    return AutoTokenizer.from_pretrained(src, **kwargs)
