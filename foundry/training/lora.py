"""
LoRA training for the student.

Every trainer in ``foundry.training`` takes an ``nn.Module`` student and trains
whatever in it requires grad. That is all LoRA needs: wrap the student once with
``attach_lora`` and hand the result to the trainer you were already using. There
is no ``LoRADistillTrainer`` and there should not be — LoRA is a property of the
model, not a kind of training.

::

    from foundry import TorchDistillTrainer, TeacherRegistry, HFTeacher
    from foundry.training.lora import attach_lora, LoRAConfig, merge_and_save

    teachers = TeacherRegistry()
    teachers.register(HFTeacher("Qwen/Qwen3-27B", quantize="4bit").load())

    student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-2B")
    student = attach_lora(student, LoRAConfig(rank=32))

    TorchDistillTrainer(student, teachers, config=cfg).train(pipe)
    merge_and_save(student, "./qwen3-2b-distilled")

Why it matters here: full-weight training of a 2B student needs the weights,
their gradients and two AdamW moments — roughly 30GB before activations. LoRA
trains well under 1% of that, which is the difference between an 80GB card and a
24GB one. Combined with a 4-bit teacher (``HFTeacher(..., quantize="4bit")``),
a 27B → 2B distillation fits on hardware you can actually rent.

Requires ``pip install olaverse-foundry[lego]`` for peft.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


__all__ = [
    "LoRAConfig",
    "attach_lora",
    "trainable_summary",
    "print_trainable_summary",
    "save_adapter",
    "merge_and_save",
    "to_skillpack",
]


@dataclass
class LoRAConfig:
    """
    LoRA hyper-parameters.

    Args:
        rank:            Adapter rank. 8-16 for style/format adaptation, 32-64
                         when distilling capability, which is what a
                         teacher→student run is doing.
        alpha:           Scaling numerator; the update is scaled by alpha/rank.
                         The usual choice is 2x rank.
        dropout:         Dropout on the LoRA path.
        target_modules:  Module names to adapt. ``None`` lets peft pick the
                         defaults for the architecture, which is correct for
                         mainstream decoders. Name them explicitly to include
                         the MLP projections — for distillation that is usually
                         worth it, since the MLP is where most capacity lives.
        bias:            "none" | "all" | "lora_only". "none" is standard.
        task_type:       "CAUSAL_LM" for decoders, "FEATURE_EXTRACTION" for
                         encoders being trained with ContrastiveTrainer or
                         EncoderDistillTrainer.
        modules_to_save: Extra modules trained in full and saved with the
                         adapter — e.g. a freshly initialised classification
                         head that LoRA cannot usefully adapt.
    """

    rank:            int   = 16
    alpha:           float = 32.0
    dropout:         float = 0.05
    target_modules:  Optional[list[str]] = None
    bias:            str   = "none"
    task_type:       str   = "CAUSAL_LM"
    modules_to_save: Optional[list[str]] = field(default=None)


def _require_peft():
    try:
        import peft
    except ImportError:
        raise ImportError(
            "peft is required for LoRA training. "
            "Install with: pip install olaverse-foundry[lego]"
        ) from None
    return peft


def _is_quantized(model: Any) -> bool:
    return bool(getattr(model, "is_quantized", False)
                or getattr(model, "is_loaded_in_4bit", False)
                or getattr(model, "is_loaded_in_8bit", False))


def attach_lora(model: Any, config: Optional[LoRAConfig] = None,
                gradient_checkpointing: bool = False) -> Any:
    """
    Freeze ``model`` and attach trainable LoRA adapters. Returns a ``PeftModel``.

    The result forwards like the model it wraps — a decoder still returns
    ``.logits``, an encoder still returns ``.last_hidden_state`` — so it drops
    straight into any foundry trainer.

    A quantized base is routed through peft's ``prepare_model_for_kbit_training``
    first. That step is not optional: without it the frozen 4-bit inputs never
    require grad, so nothing upstream of the adapters produces a gradient and
    the run trains silently at a flat loss.

    Args:
        model:  The student. May be quantized (4-bit/8-bit).
        config: ``LoRAConfig``; defaults are reasonable for distillation.
        gradient_checkpointing: Trade compute for activation memory. Worth it
                         for a multi-billion-parameter student on one card.
    """
    peft = _require_peft()
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = config or LoRAConfig()

    if _is_quantized(model):
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    try:
        task_type = getattr(TaskType, cfg.task_type)
    except AttributeError:
        raise ValueError(
            f"unknown task_type {cfg.task_type!r}; expected one of "
            f"{[t.name for t in TaskType]}"
        ) from None

    lora_kwargs: dict[str, Any] = {
        "r":              cfg.rank,
        "lora_alpha":     cfg.alpha,
        "lora_dropout":   cfg.dropout,
        "bias":           cfg.bias,
        "task_type":      task_type,
    }
    if cfg.target_modules is not None:
        lora_kwargs["target_modules"] = cfg.target_modules
    if cfg.modules_to_save is not None:
        lora_kwargs["modules_to_save"] = cfg.modules_to_save

    return get_peft_model(model, LoraConfig(**lora_kwargs))


# ── Reporting ──────────────────────────────────────────────────────────────

def trainable_summary(model: Any) -> dict:
    """
    Count trainable vs total parameters.

    Worth asserting on rather than eyeballing: a LoRA run where the percentage
    comes back at 100 has silently not attached, and a run where it comes back
    at 0 has frozen everything. Both train without error.
    """
    trainable = total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {
        "trainable":   trainable,
        "total":       total,
        "percent":     (100.0 * trainable / total) if total else 0.0,
    }


def print_trainable_summary(model: Any) -> None:
    """Print the trainable-parameter line."""
    s = trainable_summary(model)
    print(f"  trainable: {s['trainable']:,} / {s['total']:,}  ({s['percent']:.4f}%)")


# ── Output ─────────────────────────────────────────────────────────────────

def save_adapter(model: Any, path: str | Path, tokenizer: Any = None) -> Path:
    """
    Save just the adapter — a few MB, and the thing you share.

    Load it back onto the base with peft, or with
    ``foundry.skillpacks.load_from_peft`` to get a ``SkillPack``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(p))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(p))
    return p


def merge_and_save(model: Any, path: str | Path, tokenizer: Any = None) -> Path:
    """
    Merge the adapter into the base weights and save a standard HF directory.

    This is the deployable artifact: a plain model that ``transformers`` loads
    with no peft dependency and no adapter to remember.

    Raises RuntimeError on a quantized base. Merging a full-precision update
    into 4-bit weights cannot round-trip — peft would either refuse or silently
    degrade the result — so the honest options are to save the adapter and serve
    it on top of the quantized base, or to re-merge onto the base in bf16.
    """
    base = getattr(model, "base_model", None)
    inner = getattr(base, "model", base) if base is not None else model
    if _is_quantized(inner) or _is_quantized(model):
        raise RuntimeError(
            "cannot merge a LoRA adapter into a quantized base — the merged "
            "weights would not round-trip. Either save_adapter() and serve it "
            "on top of the quantized base, or reload the base in bfloat16 "
            "(quantize=None), attach the adapter there, and merge that."
        )

    if not hasattr(model, "merge_and_unload"):
        raise TypeError(
            "merge_and_save expects a PeftModel from attach_lora(); "
            f"got {type(model).__name__}"
        )

    merged = model.merge_and_unload()
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(p))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(p))
    return p


def to_skillpack(model: Any, name: str, base_hash: str = "") -> Any:
    """
    Convert a trained ``PeftModel``'s adapters into a foundry ``SkillPack``.

    Lets a LoRA run feed the skill-pack machinery — registry, composition,
    ``apply()`` — instead of stopping at a peft directory.
    """
    from foundry.skillpacks.pack import SkillPack

    weights: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in model.state_dict().items():
        if ".lora_A" in key or ".lora_B" in key:
            slot = "A" if ".lora_A" in key else "B"
            module = (key.split(".lora_")[0]
                         .replace("base_model.model.", ""))
            arr = tensor.detach().to("cpu").float().numpy()
            weights.setdefault(module, {})[slot] = arr

    complete = {m: mats for m, mats in weights.items() if "A" in mats and "B" in mats}
    if not complete:
        raise ValueError(
            "no LoRA weights found on this model — was it produced by attach_lora()?"
        )

    peft_cfg = getattr(model, "peft_config", {})
    active = getattr(model, "active_adapter", "default")
    cfg = peft_cfg.get(active) if isinstance(peft_cfg, dict) else None

    return SkillPack(
        name=name,
        base_hash=base_hash,
        rank=int(getattr(cfg, "r", 0)) or int(next(iter(complete.values()))["A"].shape[0]),
        alpha=float(getattr(cfg, "lora_alpha", 32.0)),
        target_modules=sorted({m.rsplit(".", 1)[-1] for m in complete}),
        weights=complete,
    )
