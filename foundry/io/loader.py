from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
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


def load_model(ref: ModelRef, student_class=None) -> Any:
    """
    Load a causal LM model.

    Args:
        ref:           Parsed ModelRef.
        student_class: If strategy is from_scratch and ref is a custom arch,
                       pass the Student implementation class here.

    Returns:
        A loaded model (transformers AutoModelForCausalLM or custom Student).
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "transformers is required for model loading. "
            "Install with: pip install olaverse-foundry[torch]"
        )

    kwargs: dict = {
        "torch_dtype": ref.dtype,
        "device_map":  ref.device_map,
        "trust_remote_code": ref.trust_remote_code,
    }
    if ref.revision:
        kwargs["revision"] = ref.revision

    src = str(ref.local_path) if ref.local_path else ref.repo_id
    return AutoModelForCausalLM.from_pretrained(src, **kwargs)


def load_tokenizer(ref: ModelRef) -> Any:
    """Load the tokenizer for a ModelRef."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required. "
            "Install with: pip install olaverse-foundry[torch]"
        )
    src = str(ref.local_path) if ref.local_path else ref.repo_id
    kwargs = {"trust_remote_code": ref.trust_remote_code}
    if ref.revision:
        kwargs["revision"] = ref.revision
    return AutoTokenizer.from_pretrained(src, **kwargs)
