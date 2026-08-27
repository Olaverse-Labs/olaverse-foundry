"""
Sparse top-k distillation target and KL loss.

A teacher hands back top-k distributions: ``(B, S, K)`` indices and probabilities,
about 8MB at B=8, S=2048, K=64. The dense path scatters those into a full
``(B, S, V)`` array — for a 152k-token vocabulary that is ~10GB per teacher per
step, in numpy, then copied to the device as a second ~10GB tensor. The teacher
signal did not get any richer on the way; only the zeros did.

Everything here stays in the top-k support:

* alignment maps teacher ids into student vocab space as *indices*, not a scatter
* fusion combines teachers over their supports
* the KL gathers student log-probs at exactly those indices

KL(T‖S) = Σ_v T(v)·(log T(v) − log S(v)) has no contribution from any v where
T(v)=0, so restricting the sum to the support is exact for a genuinely sparse
target. It is not identical to the dense path, which adds 1e-9 to every vocab
entry before renormalising — over 152k tokens that smears ~1.5e-4 of probability
mass across the vocabulary and pulls the target slightly off the teacher. Here
the top-k mass is renormalised over the support instead, which is the standard
top-k distillation formulation and is what the teacher actually said.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _as_tensor(array: np.ndarray, device: Any, dtype: Any):
    import torch
    return torch.as_tensor(array, dtype=dtype, device=device)


def align_sparse(alignment: Any, teacher_indices: np.ndarray,
                 teacher_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map teacher top-k ids into student vocab space, zeroing what does not map.

    Returns ``(indices, probs)`` still shaped ``(B, S, K)``. Unmapped entries get
    index 0 and probability 0 so they are safe to gather and contribute nothing.
    """
    mapped = alignment.map_indices(teacher_indices)
    unmapped = mapped < 0
    if unmapped.any():
        mapped = np.where(unmapped, 0, mapped)
        teacher_probs = np.where(unmapped, 0.0, teacher_probs)
    return mapped.astype(np.int64), teacher_probs.astype(np.float32)


def _dedup_sum(indices, probs):
    """
    Merge duplicate indices within each position, summing their probabilities.

    Needed because two teachers can nominate the same token, and a cross-tokenizer
    map can send two teacher tokens to one student token. Sorting is over the
    support axis only — a few hundred entries, not the vocabulary — so this stays
    cheap. Freed slots are left as (index 0, prob 0), which gather safely.
    """
    import torch

    order = indices.argsort(dim=-1)
    idx_sorted = indices.gather(-1, order)
    p_sorted   = probs.gather(-1, order)

    starts = torch.ones_like(idx_sorted, dtype=torch.bool)
    starts[..., 1:] = idx_sorted[..., 1:] != idx_sorted[..., :-1]

    # Every element of a run shares a segment id, so scatter_add sums the run and
    # scatter writes the run's (identical) index once.
    segment = starts.cumsum(dim=-1) - 1
    out_p   = torch.zeros_like(p_sorted).scatter_add_(-1, segment, p_sorted)
    out_idx = torch.zeros_like(idx_sorted).scatter_(-1, segment, idx_sorted)

    # Slots past the last run are untouched: force them inert.
    n_runs = starts.sum(dim=-1, keepdim=True)
    live   = torch.arange(out_p.shape[-1], device=out_p.device) < n_runs
    return torch.where(live, out_idx, 0), torch.where(live, out_p, 0.0)


def build_sparse_target(teachers_sparse: list[tuple[np.ndarray, np.ndarray]],
                        gold_ids: np.ndarray,
                        weights: Optional[list[float]],
                        strategy: str,
                        device: Any):
    """
    Fuse teachers over their top-k supports.

    ``min_ce`` *selects* one teacher per position — the one giving the gold token
    the highest probability — so the fused target is that teacher's own support,
    with no mixing and no deduplication needed. ``mean`` averages, which does
    require merging duplicate indices.

    Returns ``(indices, probs)`` torch tensors on ``device``.
    """
    import torch

    idx_list = [_as_tensor(i, device, torch.long)    for i, _ in teachers_sparse]
    p_list   = [_as_tensor(p, device, torch.float32) for _, p in teachers_sparse]

    if len(idx_list) == 1:
        # Still deduplicated: a MinED alignment maps several teacher tokens onto
        # one student token, and a top-k may legitimately repeat an id. Those are
        # mass on the same token, so they sum.
        return _dedup_sum(idx_list[0], p_list[0])

    if strategy == "min_ce":
        gold = _as_tensor(gold_ids, device, torch.long).unsqueeze(-1)   # (B, S, 1)
        # p(gold) per teacher: the top-k entry matching gold, or 0 if gold is
        # outside this teacher's support.
        gold_p = torch.stack(
            [torch.where(i == gold, p, torch.zeros_like(p)).sum(dim=-1)
             for i, p in zip(idx_list, p_list)],
            dim=0,
        )                                                               # (T, B, S)
        best = gold_p.argmax(dim=0)                                     # (B, S)
        pick = best.unsqueeze(0).unsqueeze(-1)
        idx = torch.stack(idx_list, dim=0).gather(0, pick.expand(1, *idx_list[0].shape)).squeeze(0)
        prob = torch.stack(p_list, dim=0).gather(0, pick.expand(1, *p_list[0].shape)).squeeze(0)
        return _dedup_sum(idx, prob)

    w = np.array(weights if weights else [1.0] * len(idx_list), dtype=np.float32)
    w = w / w.sum()
    idx = torch.cat(idx_list, dim=-1)
    prob = torch.cat([float(wt) * p for wt, p in zip(w, p_list)], dim=-1)
    return _dedup_sum(idx, prob)


def sparse_kl(student_logits, target_indices, target_probs, mask, eps: float = 1e-9):
    """
    Per-token KL(teacher‖student), summed over the teacher's support only.

    Args:
        student_logits: ``(B, S, V)`` student logits.
        target_indices: ``(B, S, M)`` student-space token ids.
        target_probs:   ``(B, S, M)`` probabilities; zeros are inert.
        mask:           ``(B, S)`` float mask; 0 at padding.

    Returns a scalar mean over unmasked positions, on the same per-token scale as
    the cross-entropy term so ``alpha`` balances the two honestly.
    """
    import torch
    import torch.nn.functional as F

    log_student = F.log_softmax(student_logits, dim=-1)
    log_at_support = log_student.gather(-1, target_indices)              # (B, S, M)

    # Renormalise over the support: top-k is truncated, so it does not sum to 1.
    total = target_probs.sum(dim=-1, keepdim=True)
    probs = target_probs / total.clamp(min=eps)

    # Zero-probability slots (padding, unmapped, freed dedup slots) must not
    # contribute: 0·log 0 is 0 here, not NaN.
    contrib = torch.where(
        probs > 0,
        probs * (torch.log(probs.clamp(min=eps)) - log_at_support),
        torch.zeros_like(probs),
    )
    kl_per_token = contrib.sum(dim=-1)                                   # (B, S)
    return (kl_per_token * mask).sum() / mask.sum().clamp(min=1.0)
