"""
FusionKernel — orchestrates alignment + fusion + combined distillation loss.

L = α · CE(student, gold) + (1 - α) · KL(student → fused_teacher_target)
"""
from __future__ import annotations

import numpy as np

from foundry.fusion.align import IdentityAlignment
from foundry.fusion.strategies import STRATEGY_REGISTRY, min_ce


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """CE loss per token. logits: (B,S,V), targets: (B,S) int."""
    probs = _softmax(logits)
    B, S = targets.shape
    log_p = np.log(probs[np.arange(B)[:, None], np.arange(S)[None, :], targets] + 1e-9)
    return -log_p.mean()


def _kl_divergence(student_logits: np.ndarray, teacher_dist: np.ndarray) -> np.ndarray:
    """KL(student ‖ fused_teacher) per batch, averaged over tokens."""
    p = _softmax(student_logits)
    q = teacher_dist + 1e-9
    q = q / q.sum(axis=-1, keepdims=True)
    return (q * (np.log(q) - np.log(p + 1e-9))).sum(axis=-1).mean()


class FusionKernel:
    """
    Combines multiple teacher distributions into one distillation target
    and computes the combined training loss.

    Args:
        strategy:  Fusion strategy name ("min_ce" or "mean"). Default: "min_ce".
        alpha:     Weight on the gold CE term. 1-alpha goes to KL distillation.
        alignment: A TokenizerAlignment instance. Defaults to IdentityAlignment.

    Example (toy, numpy-only)::

        kernel = FusionKernel(strategy="min_ce", alpha=0.3)
        loss = kernel.loss(
            student_logits=np.random.randn(2, 8, 100),
            gold_ids=np.random.randint(0, 100, (2, 8)),
            teacher_dists=[t1_aligned, t2_aligned],
            teacher_weights=[1.0, 0.8],
        )
    """

    def __init__(
        self,
        strategy:  str   = "min_ce",
        alpha:     float = 0.3,
        alignment  = None,
    ) -> None:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(STRATEGY_REGISTRY)}"
            )
        self._fuse_fn   = STRATEGY_REGISTRY[strategy]
        self.alpha      = alpha
        self.alignment  = alignment or IdentityAlignment()

    def align_teacher(
        self,
        teacher_indices:    np.ndarray,
        teacher_probs:      np.ndarray,
        student_vocab_size: int,
    ) -> np.ndarray:
        """Map a teacher's top-k distribution into student vocab space."""
        return self.alignment.map(teacher_indices, teacher_probs, student_vocab_size)

    def fuse(
        self,
        aligned_dists:   list[np.ndarray],
        gold_ids:        np.ndarray,
        teacher_weights: list[float] | None = None,
    ) -> np.ndarray:
        """Combine aligned teacher distributions into one target."""
        return self._fuse_fn(aligned_dists, gold_ids, teacher_weights)

    def loss(
        self,
        student_logits:  np.ndarray,
        gold_ids:        np.ndarray,
        teacher_dists:   list[np.ndarray],
        teacher_weights: list[float] | None = None,
    ) -> float:
        """
        Compute: L = α·CE(student,gold) + (1-α)·KL(student ‖ fused_teacher).

        Args:
            student_logits:  (batch, seq_len, vocab_size)
            gold_ids:        (batch, seq_len) int
            teacher_dists:   List of already-aligned dense distributions,
                             each (batch, seq_len, vocab_size)
            teacher_weights: Per-teacher scalar weights.

        Returns:
            Scalar loss value.
        """
        fused = self.fuse(teacher_dists, gold_ids, teacher_weights)
        ce  = _cross_entropy(student_logits, gold_ids)
        kl  = _kl_divergence(student_logits, fused)
        return float(self.alpha * ce + (1.0 - self.alpha) * kl)
