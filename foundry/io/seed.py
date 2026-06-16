"""
Seed loading strategies — the "enter a model name" feature of the factory.

Two paths:
  pretrained  — load existing weights from the HF hub or a local path.
                The cheap way to start: skip the $100k from-scratch seed.
  from_scratch — initialise from a config only (random weights).
                Two sub-paths: a standard HF architecture (load config only),
                or a user's custom Student class (load by dotted module path).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Optional

from foundry.io.loader import ModelRef, load_model, load_tokenizer


@dataclass
class SeedResult:
    """Encapsulates everything produced by the seed stage."""

    model:     Any
    tokenizer: Any
    config:    Any
    strategy:  str       # "pretrained" | "from_scratch"
    model_id:  str = ""


def load_seed(seed_cfg, **kwargs) -> SeedResult:
    """
    Main entry point for the seed stage.

    Args:
        seed_cfg: A ``SeedConfig`` from the recipe schema.
        **kwargs: Passed through to the model constructor (dtype, device_map, etc.).

    Returns:
        SeedResult with model, tokenizer, and config.

    Raises:
        ImportError: If transformers is not installed.
        ValueError:  If the model cannot be found on the HF hub.
    """
    if seed_cfg.init == "pretrained":
        return _load_pretrained(seed_cfg.model, kwargs)
    return _load_from_scratch(seed_cfg.arch, kwargs)


# ── Pretrained ──────────────────────────────────────────────────────────────

def _load_pretrained(model_spec: str, kwargs: dict) -> SeedResult:
    """Load a model with its full trained weights."""
    try:
        from transformers import AutoConfig
    except ImportError:
        raise ImportError(
            "transformers is required for pretrained loading. "
            "Install with: pip install olaverse-foundry[torch]"
        )
    ref    = ModelRef.parse(model_spec)
    model  = load_model(ref, **kwargs)
    tok    = load_tokenizer(ref)
    config = AutoConfig.from_pretrained(
        ref.repo_id,
        **({"revision": ref.revision} if ref.revision else {}),
    )
    return SeedResult(
        model=model,
        tokenizer=tok,
        config=config,
        strategy="pretrained",
        model_id=ref.identifier,
    )


# ── From scratch ────────────────────────────────────────────────────────────

def _load_from_scratch(arch_spec: str, kwargs: dict) -> SeedResult:
    """
    Dispatch to the right from-scratch path.

    - ``"org/model-name"`` → load HF config, randomly initialise.
    - ``"my_module:MyStudent"`` → import and instantiate custom Student class.
    """
    if ":" in arch_spec:
        return _load_custom_arch(arch_spec, kwargs)
    return _load_hf_random_init(arch_spec, kwargs)


def _load_hf_random_init(model_id: str, kwargs: dict) -> SeedResult:
    """
    Random-initialise a standard HF CausalLM architecture.

    Uses ``AutoConfig.from_pretrained`` to get the architecture spec,
    then ``AutoModelForCausalLM.from_config`` to build the model with
    random weights — no pretrained tensors are downloaded.

    This is the clean way to use an HF architecture as a novel seed:
    you get the same layer structure as e.g. Llama-3.1-8B but with
    random initialisation, ready for distillation from scratch.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required. "
            "Install with: pip install olaverse-foundry[torch]"
        )
    config = AutoConfig.from_pretrained(model_id)
    model  = AutoModelForCausalLM.from_config(config)
    tok    = AutoTokenizer.from_pretrained(model_id)
    return SeedResult(
        model=model,
        tokenizer=tok,
        config=config,
        strategy="from_scratch",
        model_id=f"{model_id}@random-init",
    )


def _load_custom_arch(arch_spec: str, kwargs: dict) -> SeedResult:
    """
    Import a user-defined Student class by dotted path ``"module.path:ClassName"``.

    The class must implement the ``Student`` protocol
    (``config``, ``tokenizer``, ``forward(input_ids) -> logits``).

    Example::

        # In recipe YAML:
        seed:
          arch: my_project.ola_arch:OlaModel
          init: from_scratch
    """
    if ":" not in arch_spec:
        raise ValueError(
            f"Custom arch spec must be 'module.path:ClassName', got: {arch_spec!r}"
        )
    module_path, class_name = arch_spec.rsplit(":", 1)
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Could not import arch module '{module_path}': {e}\n"
            "Make sure the module is on your PYTHONPATH."
        ) from e

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}'."
        )

    student = cls(**kwargs)
    return SeedResult(
        model=student,
        tokenizer=getattr(student, "tokenizer", None),
        config=getattr(student, "config", None),
        strategy="from_scratch",
        model_id=arch_spec,
    )
