"""
Token alignment strategies — map a teacher's distribution into student vocab space.

IdentityAlignment — shared tokenizer fast path (O(B×S×K) scatter).
EMAlignment       — exact surface-form match; precomputed teacher→student map.
MinEDAlignment    — EM + Levenshtein residual for cross-tokenizer alignment (M2).
"""
from __future__ import annotations

import numpy as np


class IdentityAlignment:
    """
    Fast path when student and teacher share the same tokenizer.
    teacher_indices are already in student vocab space — just scatter.
    """

    def map(
        self,
        teacher_indices: np.ndarray,
        teacher_probs:   np.ndarray,
        student_vocab_size: int,
    ) -> np.ndarray:
        """
        Scatter top-k (indices, probs) into a dense vocab distribution.

        Args:
            teacher_indices: (batch, seq_len, top_k) int array.
            teacher_probs:   (batch, seq_len, top_k) float array.
            student_vocab_size: Size of student vocabulary.

        Returns:
            Dense array (batch, seq_len, student_vocab_size).
        """
        B, S, K = teacher_indices.shape
        out = np.zeros((B, S, student_vocab_size), dtype=np.float32)
        bi = np.arange(B)[:, None, None]
        si = np.arange(S)[None, :, None]
        out[bi, si, teacher_indices] = teacher_probs
        return out


class EMAlignment:
    """
    Exact-Match vocab alignment via shared surface forms.
    Precomputes a teacher→student index array once; map() is pure scatter.
    """

    def __init__(self, teacher_vocab: dict[str, int], student_vocab: dict[str, int]) -> None:
        from foundry.fusion.vocab_map import build_em_map
        self._map = build_em_map(teacher_vocab, student_vocab)

    def map(
        self,
        teacher_indices: np.ndarray,
        teacher_probs:   np.ndarray,
        student_vocab_size: int,
    ) -> np.ndarray:
        """
        Scatter teacher top-k into student vocab space.

        Args:
            teacher_indices: (batch, seq_len, top_k) int array.
            teacher_probs:   (batch, seq_len, top_k) float array.
            student_vocab_size: V_student.

        Returns:
            Dense (batch, seq_len, student_vocab_size) float32 array.
        """
        B, S, K = teacher_indices.shape
        out = np.zeros((B, S, student_vocab_size), dtype=np.float32)
        student_indices = self._map[teacher_indices]
        mask = student_indices >= 0
        if mask.any():
            bi = np.broadcast_to(np.arange(B)[:, None, None], (B, S, K))
            si = np.broadcast_to(np.arange(S)[None, :, None], (B, S, K))
            np.add.at(out, (bi[mask], si[mask], student_indices[mask]), teacher_probs[mask])
        return out


class MinEDAlignment:
    """
    Two-phase cross-tokenizer alignment (M2 full implementation).

    Phase 1 — EMAlignment: exact surface match + normalised-EM retry.
               Covers ~90% of vocab pairs at zero edit cost.
    Phase 2 — MinED residual: Levenshtein distance on normalised surface
               forms for the remaining unmatched teacher tokens.

    The full teacher→student map is built once in __init__ (offline cost).
    Subsequent .map() calls are O(B×S×K) numpy scatter — same as EMAlignment.

    Args:
        teacher_vocab: {surface_form: teacher_token_id}
        student_vocab: {surface_form: student_token_id}
        max_ed:        Maximum edit distance to accept; tokens further
                       than this remain unmatched (default 3).

    Example::

        teacher_vocab = {"▁hello": 0, "▁world": 1, "▁café": 2}
        student_vocab = {"hello": 10, "world": 11, "cafe": 12}
        align = MinEDAlignment(teacher_vocab, student_vocab)
        print(align.coverage())
        # {"total": 3, "em_matched": 0, "mined_matched": 3, ...}
    """

    def __init__(
        self,
        teacher_vocab:  dict[str, int],
        student_vocab:  dict[str, int],
        max_ed: int = 3,
    ) -> None:
        from foundry.fusion.vocab_map import build_em_map, build_mined_map
        self._em_map    = build_em_map(teacher_vocab, student_vocab)
        self._full_map  = build_mined_map(
            teacher_vocab, student_vocab, em_map=self._em_map, max_ed=max_ed
        )
        self._max_ed    = max_ed

    def coverage(self) -> dict:
        """
        Return alignment coverage statistics.

        Keys: total, em_matched, mined_matched, unmatched, em_pct, total_pct.
        """
        from foundry.fusion.vocab_map import coverage_stats
        return coverage_stats(self._em_map, self._full_map)

    def map(
        self,
        teacher_indices: np.ndarray,
        teacher_probs:   np.ndarray,
        student_vocab_size: int,
    ) -> np.ndarray:
        """
        Scatter teacher top-k into student vocab space using the full
        EM+MinED map.  Unmatched tokens (map == -1) are silently dropped.

        Args:
            teacher_indices: (batch, seq_len, top_k) int array.
            teacher_probs:   (batch, seq_len, top_k) float array.
            student_vocab_size: V_student.

        Returns:
            Dense (batch, seq_len, student_vocab_size) float32 array.
        """
        B, S, K = teacher_indices.shape
        out = np.zeros((B, S, student_vocab_size), dtype=np.float32)
        student_indices = self._full_map[teacher_indices]
        mask = student_indices >= 0
        if mask.any():
            bi = np.broadcast_to(np.arange(B)[:, None, None], (B, S, K))
            si = np.broadcast_to(np.arange(S)[None, :, None], (B, S, K))
            np.add.at(out, (bi[mask], si[mask], student_indices[mask]), teacher_probs[mask])
        return out
