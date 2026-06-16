"""
Token alignment strategies — map a teacher's distribution into student vocab space.

M0: IdentityAlignment (shared tokenizer fast path) and stub placeholders for
    EMAlignment and MinEDAlignment. Real implementations land in M2.
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
    Builds a teacher→student index map from token string overlap.

    M2 implementation note: precompute the vocab map once, cache it,
    use IdentityAlignment for the majority that match, EM only for residual.
    """

    def __init__(self, teacher_vocab: dict[str, int], student_vocab: dict[str, int]) -> None:
        self._map = self._build_map(teacher_vocab, student_vocab)

    @staticmethod
    def _build_map(
        teacher_vocab: dict[str, int],
        student_vocab: dict[str, int],
    ) -> np.ndarray:
        """Return index array: teacher_to_student[teacher_id] = student_id or -1."""
        t2s = np.full(max(teacher_vocab.values()) + 1, -1, dtype=np.int32)
        student_by_str = {v: k for k, v in student_vocab.items()}
        for token_str, t_id in teacher_vocab.items():
            s_id = student_vocab.get(token_str, -1)
            t2s[t_id] = s_id
        return t2s

    def map(
        self,
        teacher_indices: np.ndarray,
        teacher_probs:   np.ndarray,
        student_vocab_size: int,
    ) -> np.ndarray:
        B, S, K = teacher_indices.shape
        out = np.zeros((B, S, student_vocab_size), dtype=np.float32)
        student_indices = self._map[teacher_indices]   # map via lookup
        mask = student_indices >= 0
        if mask.any():
            bi = np.broadcast_to(np.arange(B)[:, None, None], (B, S, K))
            si = np.broadcast_to(np.arange(S)[None, :, None], (B, S, K))
            np.add.at(out, (bi[mask], si[mask], student_indices[mask]), teacher_probs[mask])
        return out


class MinEDAlignment:
    """
    Minimum-Edit-Distance alignment for tokens with no exact surface match.
    Covers the residual after EM — the FuseLLM improvement over plain EM.

    M0 status: interface defined; falls back to EMAlignment.
    M2 will add the MinED computation using rapidfuzz or a BPE byte walk.
    """

    def __init__(
        self,
        teacher_vocab:  dict[str, int],
        student_vocab:  dict[str, int],
        em_first: bool = True,
    ) -> None:
        self._em = EMAlignment(teacher_vocab, student_vocab)
        self._teacher_vocab = teacher_vocab
        self._student_vocab = student_vocab
        self._em_first = em_first

    def map(
        self,
        teacher_indices: np.ndarray,
        teacher_probs:   np.ndarray,
        student_vocab_size: int,
    ) -> np.ndarray:
        # M0: delegate to EM; M2 adds MinED residual layer
        return self._em.map(teacher_indices, teacher_probs, student_vocab_size)
