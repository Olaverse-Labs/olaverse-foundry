"""
Depth up-scaling (SOLAR-style) — duplicate-and-trim layer mapping.

The SOLAR insight: copy a subset of layers from the source model into new
positions in the target model, then continue training (heal). No new random
weights — all new layers start as copies of existing ones.

The SOLAR dip: a freshly upscaled model performs worse than the seed until
healed; the distillation run is what makes the extra capacity useful.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from foundry.contracts import ArchConfig


@dataclass
class GrowthPlan:
    """The output of the planner — a layer-index map plus diagnostics."""

    src_layers:    int
    target_layers: int
    layer_map:     list[int]          # layer_map[i] = source layer to copy into position i
    scale_factor:  float
    shape_warning: str | None = None

    def summary(self) -> str:
        lines = [
            f"Growth Plan: {self.src_layers} → {self.target_layers} layers "
            f"({self.scale_factor:.2f}×)",
            f"Layer map: {self.layer_map}",
        ]
        if self.shape_warning:
            lines.append(f"⚠  {self.shape_warning}")
        return "\n".join(lines)


def upscale_layer_map(n_src: int, n_target: int) -> list[int]:
    """
    Plan which source layer feeds each target layer (SOLAR duplicate-and-trim).

    Strategy: distribute source layers evenly across target positions.
    Positions beyond n_src repeat layers from the end of the source.

    Args:
        n_src:    Number of layers in the seed model.
        n_target: Desired number of layers after upscaling.

    Returns:
        List of length n_target; each entry is a source layer index [0, n_src).

    Examples::

        upscale_layer_map(4, 8)  → [0, 1, 2, 3, 0, 1, 2, 3]   # duplicate once
        upscale_layer_map(4, 6)  → [0, 0, 1, 2, 2, 3]           # interleaved
    """
    if n_target < n_src:
        raise ValueError(
            f"target ({n_target}) must be ≥ source ({n_src}). "
            "Use pruning for shrinking, not upscale."
        )
    step = n_src / n_target
    return [min(int(i * step), n_src - 1) for i in range(n_target)]


def layers_for_param_target(cfg: ArchConfig, target_params: float) -> tuple[int, str | None]:
    """
    Compute how many layers at the seed's width approximate a parameter target.

    Args:
        cfg:           Seed ArchConfig (n_layers, d_model, vocab_size).
        target_params: Desired total parameter count (e.g. 15e9 for 15B).

    Returns:
        (n_layers, warning_or_None) — warning when shape is deep/narrow.
    """
    embed = cfg.vocab_size * cfg.d_model
    attn  = 4 * cfg.d_model ** 2
    ff    = 2 * cfg.d_model * cfg.d_ff if cfg.d_ff else 8 * cfg.d_model ** 2
    per_layer = attn + ff
    n_layers = max(1, round((target_params - embed) / per_layer))
    new_cfg = ArchConfig(
        n_layers=n_layers,
        d_model=cfg.d_model,
        vocab_size=cfg.vocab_size,
        d_ff=cfg.d_ff,
    )
    return n_layers, new_cfg.shape_warning()


def build_upscaled_state_dict(
    src_state_dict: dict[str, Any],
    layer_map:      list[int],
    layer_prefix:   str = "model.layers",
) -> dict[str, Any]:
    """
    Materialize an upscaled state dict by copying source layers per the plan.

    Args:
        src_state_dict: The source model's state dict (torch tensors or numpy arrays).
        layer_map:      Output of upscale_layer_map().
        layer_prefix:   Key prefix for layer weights (e.g. 'model.layers' for Llama).

    Returns:
        New state dict with target_layers layers.
    """
    import copy

    # Separate layer keys from non-layer keys
    layer_keys: dict[int, dict[str, Any]] = {}
    other_keys: dict[str, Any] = {}

    for key, val in src_state_dict.items():
        if key.startswith(layer_prefix + "."):
            rest = key[len(layer_prefix) + 1:]
            dot  = rest.index(".")
            idx  = int(rest[:dot])
            sub  = rest[dot + 1:]
            layer_keys.setdefault(idx, {})[sub] = val
        else:
            other_keys[key] = val

    new_state: dict[str, Any] = {}
    for k, v in other_keys.items():
        new_state[k] = v.clone() if hasattr(v, "clone") else copy.copy(v)

    for new_idx, src_idx in enumerate(layer_map):
        for sub, val in layer_keys.get(src_idx, {}).items():
            new_key = f"{layer_prefix}.{new_idx}.{sub}"
            # Use .clone() for torch tensors to avoid aliased storage across layers
            new_state[new_key] = val.clone() if hasattr(val, "clone") else copy.copy(val)

    return new_state


def plan_growth(cfg: ArchConfig, to_params: float) -> GrowthPlan:
    """
    High-level entry point: given a seed config and a target size, return a GrowthPlan.

    Args:
        cfg:       Seed ArchConfig.
        to_params: Target parameter count.

    Returns:
        GrowthPlan with layer map and any shape warnings.
    """
    n_target, warning = layers_for_param_target(cfg, to_params)
    layer_map = upscale_layer_map(cfg.n_layers, n_target)
    return GrowthPlan(
        src_layers=cfg.n_layers,
        target_layers=n_target,
        layer_map=layer_map,
        scale_factor=n_target / cfg.n_layers,
        shape_warning=warning,
    )
