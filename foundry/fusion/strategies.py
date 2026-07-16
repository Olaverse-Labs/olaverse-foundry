"""
Fusion strategies — combine multiple aligned teacher distributions into one target.

FuseLLM finding: min_ce (per-token, pick the teacher whose distribution has lowest
cross-entropy with the gold token) beats simple averaging in quality.
"""
from __future__ import annotations

from typing import Callable, Optional
import numpy as np


def min_ce(
    teacher_dists: list[np.ndarray],
    gold_ids:      np.ndarray,
    weights:       list[float] | None = None,
) -> np.ndarray:
    """
    Per-token MinCE fusion (FuseLLM best variant).

    For each position, select the teacher whose distribution assigns the
    highest probability to the gold token (lowest cross-entropy).

    Args:
        teacher_dists: List of dense arrays, each (batch, seq_len, vocab).
        gold_ids:      Ground-truth token ids, (batch, seq_len).
        weights:       Teacher weights (unused in selection, applied post-select).

    Returns:
        Fused distribution (batch, seq_len, vocab).
    """
    if len(teacher_dists) == 1:
        return teacher_dists[0]

    B, S, V = teacher_dists[0].shape
    # Gather p(gold) for each teacher: (n_teachers, B, S)
    stacked = np.stack(teacher_dists, axis=0)               # (T, B, S, V)
    gold_expanded = gold_ids[None, :, :, None]              # (1, B, S, 1)
    # Clip so indices don't go out of bounds with toy vocabs
    gold_clipped = np.clip(gold_expanded, 0, V - 1)
    gold_probs = np.take_along_axis(stacked, np.broadcast_to(gold_clipped, (len(teacher_dists), B, S, 1)), axis=3).squeeze(-1)
    best = np.argmax(gold_probs, axis=0)                    # (B, S)

    out = np.zeros((B, S, V), dtype=np.float32)
    for t_idx, dist in enumerate(teacher_dists):
        mask = (best == t_idx)                              # (B, S) bool
        if mask.any():
            out[mask] = dist[mask]

    return out


def mean_ce(
    teacher_dists: list[np.ndarray],
    gold_ids:      np.ndarray,
    weights:       list[float] | None = None,
) -> np.ndarray:
    """
    Weighted-average fusion (AvgCE baseline).

    Args:
        teacher_dists: List of dense arrays, each (batch, seq_len, vocab).
        gold_ids:      Not used for mean fusion, kept for API symmetry.
        weights:       Per-teacher weights; uniform if None.

    Returns:
        Fused distribution (batch, seq_len, vocab).
    """
    if len(teacher_dists) == 1:
        return teacher_dists[0]

    w = np.array(weights if weights else [1.0] * len(teacher_dists), dtype=np.float32)
    w /= w.sum()
    return sum(wt * d for wt, d in zip(w, teacher_dists))


# ── Registry ───────────────────────────────────────────────────────────────
_StrategyFn = Callable[[list[np.ndarray], np.ndarray, Optional[list[float]]], np.ndarray]

STRATEGY_REGISTRY: dict[str, _StrategyFn] = {
    "min_ce": min_ce,
    "mean":   mean_ce,
}


def register_strategy(name: str, fn: _StrategyFn) -> None:
    """Register a custom fusion strategy."""
    STRATEGY_REGISTRY[name] = fn
