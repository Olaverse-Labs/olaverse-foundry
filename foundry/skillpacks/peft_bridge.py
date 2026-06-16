"""
PEFT adapter bridge — convert SkillPack ↔ HuggingFace PEFT adapter format.

Allows:
  - Publishing a foundry SkillPack as a standard HF adapter (shareable via Hub).
  - Loading any community LoRA adapter from the HF Hub into foundry.

Weight format priority (save):
  1. safetensors  — if safetensors is installed (recommended, safe)
  2. torch .bin   — if torch is installed (standard PEFT fallback)

Weight format priority (load):
  1. adapter_model.safetensors
  2. adapter_model.bin
  (No PEFT library required for either direction.)

To attach a loaded SkillPack to a live HF model via PEFT::

    from foundry.skillpacks.peft_bridge import to_peft_model
    model = to_peft_model(base_model, pack)   # requires: pip install peft
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from foundry.skillpacks.pack import SkillPack


# ── Config helpers ─────────────────────────────────────────────────────────

def peft_config_dict(pack: SkillPack) -> dict:
    """
    Return the standard PEFT adapter_config.json payload for a SkillPack.
    Pure dict — no PEFT library required.

    Extra foundry fields (``foundry_base_hash``, ``foundry_name``) are preserved
    so the round-trip through ``load_from_peft()`` reconstructs the pack exactly.
    """
    return {
        "peft_type":               "LORA",
        "task_type":               "CAUSAL_LM",
        "r":                       pack.rank,
        "lora_alpha":              pack.alpha,
        "lora_dropout":            0.0,
        "bias":                    "none",
        "target_modules":          pack.target_modules,
        "base_model_name_or_path": "",
        "foundry_base_hash":       pack.base_hash,
        "foundry_name":            pack.name,
    }


def to_peft_config(pack: SkillPack):
    """
    Return a ``peft.LoraConfig`` for this SkillPack.
    Requires: ``pip install peft``
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        raise ImportError(
            "peft is required for to_peft_config(). "
            "Install with: pip install olaverse-foundry[lego]"
        )
    return LoraConfig(
        r=pack.rank,
        lora_alpha=pack.alpha,
        target_modules=pack.target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def to_peft_model(base_model, pack: SkillPack):
    """
    Wrap a base HF model with this SkillPack's adapter using PEFT.
    Returns a PeftModel ready for further training or inference.
    Requires: ``pip install peft``
    """
    try:
        from peft import get_peft_model
    except ImportError:
        raise ImportError(
            "peft is required for to_peft_model(). "
            "Install with: pip install olaverse-foundry[lego]"
        )
    return get_peft_model(base_model, to_peft_config(pack))


# ── Weights I/O ────────────────────────────────────────────────────────────

def _state_dict_from_pack(pack: SkillPack) -> dict[str, np.ndarray]:
    """Build the PEFT-format state dict from a SkillPack's weights."""
    sd: dict[str, np.ndarray] = {}
    for module_name, mats in pack.weights.items():
        prefix = f"base_model.model.{module_name}"
        # A is (rank, d_in), B is (d_out, rank)
        sd[f"{prefix}.lora_A.weight"] = np.array(mats["A"], dtype=np.float32)
        sd[f"{prefix}.lora_B.weight"] = np.array(mats["B"], dtype=np.float32)
    return sd


def _save_weights(state_dict: dict[str, np.ndarray], path: Path) -> str:
    """
    Write adapter weights. Returns the filename used.
    Tries safetensors first, falls back to torch .bin.
    """
    try:
        from safetensors.numpy import save_file
        out = path / "adapter_model.safetensors"
        save_file(state_dict, str(out))
        return "adapter_model.safetensors"
    except ImportError:
        pass

    try:
        import torch
        out = path / "adapter_model.bin"
        torch.save({k: torch.tensor(v) for k, v in state_dict.items()}, str(out))
        return "adapter_model.bin"
    except ImportError:
        pass

    raise ImportError(
        "Either safetensors or torch is required to save adapter weights. "
        "Install with: pip install safetensors  OR  pip install torch"
    )


def _load_weights(path: Path) -> dict[str, np.ndarray]:
    """Load adapter weights from safetensors or .bin, returning numpy arrays."""
    st_path  = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if st_path.exists():
        try:
            from safetensors.numpy import load_file
            return load_file(str(st_path))
        except ImportError:
            # safetensors installed but numpy backend missing? try torch backend
            try:
                from safetensors.torch import load_file as load_torch
                return {k: v.numpy() for k, v in load_torch(str(st_path)).items()}
            except ImportError:
                pass

    if bin_path.exists():
        import torch
        sd = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        return {k: v.numpy() for k, v in sd.items()}

    raise FileNotFoundError(
        f"No adapter weights found in {path}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


# ── Public API ─────────────────────────────────────────────────────────────

def save_as_peft(pack: SkillPack, path: str | Path) -> Path:
    """
    Save a SkillPack as a standard PEFT adapter directory.

    Writes:
      adapter_config.json        — PEFT config + foundry metadata
      adapter_model.safetensors  — weights (preferred)
        OR
      adapter_model.bin          — weights (torch fallback)

    Args:
        pack: The SkillPack to export.
        path: Destination directory (created if absent).

    Returns:
        Path to the written directory.

    Example::

        save_as_peft(math_pack, "./checkpoints/ola_math_lora")
        # then push to hub:
        # from huggingface_hub import upload_folder
        # upload_folder(folder_path="./checkpoints/ola_math_lora", ...)
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    (p / "adapter_config.json").write_text(
        json.dumps(peft_config_dict(pack), indent=2)
    )
    _save_weights(_state_dict_from_pack(pack), p)
    return p


def load_from_peft(path: str | Path, name: str | None = None) -> SkillPack:
    """
    Load a PEFT adapter directory into a SkillPack.

    Works with any standard PEFT LoRA adapter — not just ones saved by foundry.
    Foundry-specific fields (base_hash, name) are read from ``foundry_*`` keys
    in adapter_config.json if present; otherwise defaults are used.

    Args:
        path: Path to the adapter directory (contains adapter_config.json).
        name: Override pack name; defaults to the directory name.

    Returns:
        SkillPack with weights loaded as numpy float32 arrays.

    Example::

        pack = load_from_peft("./checkpoints/ola_math_lora")
        registry.register(pack)
    """
    p = Path(path)
    config = json.loads((p / "adapter_config.json").read_text())
    raw    = _load_weights(p)

    # Parse PEFT state dict → {module_name: {"A": arr, "B": arr}}
    weights: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in raw.items():
        arr = np.array(tensor, dtype=np.float32)
        if ".lora_A.weight" in key:
            module = key.replace("base_model.model.", "").replace(".lora_A.weight", "")
            weights.setdefault(module, {})["A"] = arr
        elif ".lora_B.weight" in key:
            module = key.replace("base_model.model.", "").replace(".lora_B.weight", "")
            weights.setdefault(module, {})["B"] = arr

    # Drop incomplete pairs (missing A or B)
    complete = {
        mod: mats
        for mod, mats in weights.items()
        if "A" in mats and "B" in mats
    }

    return SkillPack(
        name=name or config.get("foundry_name", p.name),
        base_hash=config.get("foundry_base_hash", ""),
        rank=int(config["r"]),
        alpha=float(config["lora_alpha"]),
        target_modules=config.get("target_modules", list(complete.keys())),
        weights=complete,
    )
