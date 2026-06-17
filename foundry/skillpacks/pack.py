"""
SkillPacks — cheap, detachable LoRA adapters that snap onto a frozen base.

Each pack is bound to exactly one base model via its hash. Attaching a pack to
the wrong base is refused at runtime — no silent corruption.

M0: in-memory composition (no peft dependency). M4 wires in peft LoraConfig.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _model_hash(state_dict: dict[str, Any]) -> str:
    """
    Stable fingerprint combining weight shapes AND sampled value statistics.

    Includes the mean and std of the first 10 non-bias weight matrices so that
    two models with the same architecture but different weights produce different
    hashes (the shapes-only approach fails here).
    """
    non_bias = {k: v for k, v in state_dict.items() if "bias" not in k}
    shape_sig = {k: list(v.shape) for k, v in non_bias.items()}

    # Value stats for the first 10 weight tensors (deterministic key order)
    stat_sig: dict[str, list] = {}
    for i, (k, v) in enumerate(non_bias.items()):
        if i >= 10:
            break
        try:
            arr = v.float().cpu().numpy() if hasattr(v, "float") else np.array(v, dtype=np.float32)
            flat = arr.ravel()
            stat_sig[k] = [round(float(flat.mean()), 6), round(float(flat.std()), 6)]
        except Exception:
            stat_sig[k] = []

    raw = json.dumps({"shapes": shape_sig, "stats": stat_sig}, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class SkillPack:
    """
    A named LoRA adapter delta bound to a specific base model.

    Attributes:
        name:       Human-readable skill name (e.g. "ola_math").
        base_hash:  SHA-256[:16] fingerprint of the base model's weight shapes.
        rank:       LoRA rank.
        alpha:      LoRA scaling factor.
        target_modules: Module names the adapter applies to.
        weights:    Dict of adapter matrices (A, B pairs) stored as numpy arrays.
    """

    name:           str
    base_hash:      str
    rank:           int              = 16
    alpha:          float            = 16.0
    target_modules: list[str]        = field(default_factory=lambda: ["q_proj", "v_proj"])
    weights:        dict[str, Any]   = field(default_factory=dict)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def verify_base(self, base_hash: str) -> None:
        """Raise if this pack was trained on a different base."""
        if self.base_hash != base_hash:
            raise ValueError(
                f"SkillPack '{self.name}' was trained on base {self.base_hash!r} "
                f"but the current base is {base_hash!r}. "
                "Attaching to the wrong base would corrupt outputs."
            )

    def apply(self, weight: np.ndarray, module_name: str) -> np.ndarray:
        """
        Add LoRA delta to a weight matrix: W' = W + scaling · B · A.

        Args:
            weight:      Original weight (d_out, d_in).
            module_name: Key in self.weights for this module.

        Returns:
            Updated weight array.
        """
        if module_name not in self.weights:
            return weight
        A = self.weights[module_name]["A"]
        B = self.weights[module_name]["B"]
        return weight + self.scaling * (B @ A)

    def save(self, path: str | Path) -> None:
        """Serialize pack to a directory (numpy .npz + metadata JSON)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        meta = {
            "name":           self.name,
            "base_hash":      self.base_hash,
            "rank":           self.rank,
            "alpha":          self.alpha,
            "target_modules": self.target_modules,
        }
        (p / "meta.json").write_text(json.dumps(meta, indent=2))
        np.savez(p / "weights.npz", **{
            f"{mod}__{mat}": arr
            for mod, mats in self.weights.items()
            for mat, arr in mats.items()
        })

    @classmethod
    def load(cls, path: str | Path) -> "SkillPack":
        """Load a previously saved SkillPack."""
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        raw  = np.load(p / "weights.npz")
        weights: dict = {}
        for key, arr in raw.items():
            mod, mat = key.rsplit("__", 1)
            weights.setdefault(mod, {})[mat] = arr
        return cls(weights=weights, **meta)


class SkillRegistry:
    """
    Manages a collection of SkillPacks and composes them onto a frozen base.

    Usage::

        registry = SkillRegistry(base_state_dict=model.state_dict())
        registry.register(math_pack)
        registry.register(code_pack)
        merged = registry.snap_on("ola_math", "ola_code")
    """

    def __init__(self, base_state_dict: dict[str, Any]) -> None:
        self._base = base_state_dict
        self._base_hash = _model_hash(base_state_dict)
        self._packs: dict[str, SkillPack] = {}

    def register(self, pack: SkillPack) -> None:
        """Register a pack. Verifies base fingerprint before storing."""
        pack.verify_base(self._base_hash)
        self._packs[pack.name] = pack

    def snap_on(self, *names: str) -> dict[str, Any]:
        """
        Apply named packs in sequence onto the frozen base.

        Returns a new state dict with all requested deltas applied.
        The base is never mutated.

        Key matching: searches each state-dict key right-to-left for the first
        component that appears in ``target_modules``. This handles both bare
        module keys (``"q_proj"``) and fully-qualified keys
        (``"model.layers.0.self_attn.q_proj.weight"``).
        """
        import copy
        state = copy.deepcopy(self._base)
        for name in names:
            if name not in self._packs:
                raise KeyError(f"SkillPack '{name}' not registered.")
            pack = self._packs[name]
            for key in list(state.keys()):
                matched = None
                for part in reversed(key.split(".")):
                    if part in pack.target_modules:
                        matched = part
                        break
                if matched is not None:
                    state[key] = pack.apply(state[key], matched)
        return state

    def list_packs(self) -> list[str]:
        return list(self._packs.keys())
