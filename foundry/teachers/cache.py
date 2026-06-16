"""
LogitCache — generate teacher signal once, reuse forever.

Caching is the single biggest cost lever: large teachers over hundreds of
billions of tokens can cost as much as training the student itself.

M0: in-memory cache (dict). M3 adds on-disk storage via safetensors/npz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class LogitCache:
    """
    Stores top-k (indices, probs) per token position for one teacher.

    Keys are (text_hash, position) tuples. The cache is populated by running
    the teacher once over the dataset; subsequent training epochs read from
    cache without touching the teacher model.

    Args:
        top_k:   Number of top logits to store per position.
        max_entries: Cap on in-memory entries (LRU eviction). 0 = unlimited.

    M3 note: replace the in-memory dict with a memory-mapped safetensors file
    sharded by chunk_id so cache fits on a fast SSD rather than RAM.
    """

    def __init__(self, top_k: int = 64, max_entries: int = 0) -> None:
        self.top_k = top_k
        self.max_entries = max_entries
        self._store: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._hits  = 0
        self._misses = 0

    def put(
        self,
        key:     tuple,
        indices: np.ndarray,
        probs:   np.ndarray,
    ) -> None:
        """Store a top-k distribution under key."""
        if self.max_entries and len(self._store) >= self.max_entries:
            # Simple FIFO eviction for M0; use LRU in M3
            evict = next(iter(self._store))
            del self._store[evict]
        self._store[key] = (indices.astype(np.int32), probs.astype(np.float32))

    def get(self, key: tuple) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached distribution. Returns None on miss."""
        result = self._store.get(key)
        if result is None:
            self._misses += 1
        else:
            self._hits += 1
        return result

    def populate(self, teacher, input_ids: np.ndarray) -> None:
        """
        Run teacher over a batch and cache all positions.

        Args:
            teacher:   Any Teacher protocol impl (ToyTeacher or HFTeacher).
            input_ids: (batch, seq_len) int array.
        """
        indices, probs = teacher.distribution(input_ids, top_k=self.top_k)
        B, S = input_ids.shape
        for b in range(B):
            for s in range(S):
                key = (int(input_ids[b, s]), b, s)
                self.put(key, indices[b, s], probs[b, s])

    @property
    def stats(self) -> dict:
        return {
            "size":       len(self._store),
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   round(self._hits / max(1, self._hits + self._misses), 3),
        }

    def save(self, path: str | Path) -> None:
        """Persist cache to a .npz file (M0 on-disk format)."""
        path = Path(path)
        arrays: dict = {}
        for i, (key, (idx, prob)) in enumerate(self._store.items()):
            arrays[f"idx_{i}"]  = idx
            arrays[f"prob_{i}"] = prob
        np.savez_compressed(path, **arrays)

    def clear(self) -> None:
        self._store.clear()
        self._hits   = 0
        self._misses = 0
