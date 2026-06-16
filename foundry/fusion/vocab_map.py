"""
Vocabulary mapping utilities for cross-tokenizer alignment.

Two build passes:
  build_em_map    — exact surface-form match (EM pass)
  build_mined_map — em_map + MinED residual for unmatched tokens

The maps are int32 arrays indexed by teacher token id:
  map[teacher_id] = student_id   (or -1 if no match found)

Once built, .map() is just numpy scatter — same speed regardless of
how the indices were resolved.

Optimisation notes
------------------
- MinED runs only on tokens left unmatched by EM and extended-EM (the
  normalised-form retry).  In practice this is <10% of the vocab.
- Candidates are prefix-filtered to the first normalised character, reducing
  comparisons from O(T_unmatched × S) to O(T_unmatched × S/|alpha|).
- With rapidfuzz[distance] installed, Levenshtein runs in C (~100× faster).
  Without it, a pure-Python DP handles short BPE tokens (avg 3-6 chars)
  fast enough for offline map builds.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np


# ── Token normalisation ────────────────────────────────────────────────────

# Leading markers used by SentencePiece (▁), HF fast tokenisers (Ġ),
# WordPiece (##), and SPM byte tokens (<0xNN>).
_BOUNDARY_RE = re.compile(r"^(▁+|Ġ+|#{1,2}|<0x[0-9A-Fa-f]{2}>)")


def normalise_token(token: str) -> str:
    """Strip leading BPE/SPM boundary markers and lowercase."""
    return _BOUNDARY_RE.sub("", token).lower()


# ── Edit distance backend ──────────────────────────────────────────────────

try:
    from rapidfuzz.distance.Levenshtein import distance as _lev
    _HAS_RAPIDFUZZ = True

    def _edit_distance(a: str, b: str) -> int:
        return _lev(a, b)

except ImportError:
    _HAS_RAPIDFUZZ = False

    def _edit_distance(a: str, b: str) -> int:
        """Pure-Python Levenshtein — sufficient for short BPE tokens."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        row = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            new_row = [i]
            for j, cb in enumerate(b, 1):
                new_row.append(
                    min(row[j] + 1, new_row[-1] + 1, row[j - 1] + int(ca != cb))
                )
            row = new_row
        return row[-1]


def has_rapidfuzz() -> bool:
    """True if rapidfuzz C-extension is available for fast edit distance."""
    return _HAS_RAPIDFUZZ


# ── EM map builder ─────────────────────────────────────────────────────────

def build_em_map(
    teacher_vocab: dict[str, int],
    student_vocab: dict[str, int],
) -> np.ndarray:
    """
    Build teacher→student index map via exact surface-form match.

    Args:
        teacher_vocab: {surface_form: teacher_token_id}
        student_vocab: {surface_form: student_token_id}

    Returns:
        int32 array of shape (max_teacher_id + 1,);
        unmatched entries are -1.
    """
    t2s = np.full(max(teacher_vocab.values()) + 1, -1, dtype=np.int32)
    for token_str, t_id in teacher_vocab.items():
        s_id = student_vocab.get(token_str, -1)
        t2s[t_id] = s_id
    return t2s


# ── MinED map builder ──────────────────────────────────────────────────────

def build_mined_map(
    teacher_vocab: dict[str, int],
    student_vocab: dict[str, int],
    em_map: Optional[np.ndarray] = None,
    max_ed: int = 3,
) -> np.ndarray:
    """
    Build teacher→student index map using EM + MinED residual.

    For unmatched teacher tokens (em_map[t_id] == -1), finds the closest
    student token by Levenshtein distance on normalised surface forms.

    Three-phase strategy for each unmatched teacher token:
      Phase 1 — normalised-EM: strip BPE/SPM markers and retry exact match.
      Phase 2 — prefix-filtered MinED: compare only against student tokens
                sharing the first normalised character (plus neighbours).
      Phase 3 — fallback full-vocab MinED: for tokens whose prefix bucket
                is empty (special tokens, emoji, byte sequences).

    Tokens with best edit distance > max_ed remain unmatched (-1).

    Args:
        teacher_vocab: {surface_form: teacher_token_id}
        student_vocab: {surface_form: student_token_id}
        em_map:        Pre-built EM map (built here if None).
        max_ed:        Maximum edit distance to accept a match (default 3).

    Returns:
        int32 array (max_teacher_id + 1,); unmatched = -1.
    """
    if em_map is None:
        em_map = build_em_map(teacher_vocab, student_vocab)

    mined_map = em_map.copy()

    # ── Normalised student index (extended EM) ─────────────────────────
    norm_student: dict[str, int] = {}
    for s_str, s_id in student_vocab.items():
        n = normalise_token(s_str)
        norm_student.setdefault(n, s_id)   # first match wins

    # ── Prefix-indexed student table ───────────────────────────────────
    by_prefix: dict[str, list[tuple[str, int]]] = {}
    for n_str, s_id in norm_student.items():
        prefix = n_str[:1] if n_str else "\x00"
        by_prefix.setdefault(prefix, []).append((n_str, s_id))

    # ── Residual: teacher tokens not matched by EM ─────────────────────
    unmatched = [
        (t_str, t_id)
        for t_str, t_id in teacher_vocab.items()
        if mined_map[t_id] == -1
    ]

    for t_str, t_id in unmatched:
        n_t = normalise_token(t_str)

        # Phase 1 — normalised exact match
        s_id = norm_student.get(n_t, -1)
        if s_id != -1:
            mined_map[t_id] = s_id
            continue

        # Phase 2 — MinED within prefix bucket + adjacent chars
        prefix = n_t[:1] if n_t else "\x00"
        candidates: list[tuple[str, int]] = list(by_prefix.get(prefix, []))
        for delta in (-1, 1):
            adj = chr(ord(prefix) + delta)
            candidates.extend(by_prefix.get(adj, []))

        # Phase 3 — fall back to full normalised vocab (rare)
        if not candidates:
            candidates = list(norm_student.items())

        best_id   = -1
        best_dist = max_ed + 1
        for n_s, s_id in candidates:
            d = _edit_distance(n_t, n_s)
            if d < best_dist:
                best_dist = d
                best_id   = s_id

        mined_map[t_id] = best_id   # stays -1 if nothing within max_ed

    return mined_map


# ── Coverage stats ─────────────────────────────────────────────────────────

def coverage_stats(
    em_map:    np.ndarray,
    mined_map: np.ndarray,
) -> dict:
    """
    Alignment coverage report comparing EM-only vs EM+MinED maps.

    Returns dict with:
      total         — total teacher tokens
      em_matched    — exact-match count
      mined_matched — additional matches from MinED residual
      unmatched     — still -1 after MinED
      em_pct        — EM coverage %
      total_pct     — total coverage %
    """
    total         = len(em_map)
    em_matched    = int((em_map    >= 0).sum())
    total_matched = int((mined_map >= 0).sum())
    mined_added   = total_matched - em_matched
    unmatched     = total - total_matched
    return {
        "total":         total,
        "em_matched":    em_matched,
        "mined_matched": mined_added,
        "unmatched":     unmatched,
        "em_pct":        round(100.0 * em_matched    / max(total, 1), 2),
        "total_pct":     round(100.0 * total_matched / max(total, 1), 2),
    }
