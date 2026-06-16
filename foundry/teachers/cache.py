"""
LogitCache — generate teacher signal once, reuse forever.

Caching is the single biggest cost lever: large teachers over hundreds of
billions of tokens can cost as much as training the student itself.

Two key modes:
  per-token (M0): tuple keys (token_id, batch, seq); fine-grained, big RAM
  per-batch (M3): integer keys (batch_idx); stores full (B,S,K) arrays; disk-friendly

M3 adds populate_dataset(), put_batch/get_batch, and a round-trip save/load
that persists both keys and values to a .npz file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


class LogitCache:
    """
    Stores top-k (indices, probs) distributions for one teacher.

    Supports two key types:
      - tuple  (M0): per-position key ``(token_id, batch, seq)``
      - int    (M3): per-batch key ``batch_idx``; value is ``(B, S, K)`` arrays

    Args:
        top_k:       Number of top logits to store per position.
        max_entries: Cap on in-memory entries (FIFO eviction). 0 = unlimited.
    """

    def __init__(self, top_k: int = 64, max_entries: int = 0) -> None:
        self.top_k = top_k
        self.max_entries = max_entries
        self._store: dict[Union[tuple, int], tuple[np.ndarray, np.ndarray]] = {}
        self._hits   = 0
        self._misses = 0

    # ── Per-token API (M0) ─────────────────────────────────────────────────

    def put(
        self,
        key:     tuple,
        indices: np.ndarray,
        probs:   np.ndarray,
    ) -> None:
        """Store a per-position top-k distribution under a tuple key."""
        if self.max_entries and len(self._store) >= self.max_entries:
            evict = next(iter(self._store))
            del self._store[evict]
        self._store[key] = (indices.astype(np.int32), probs.astype(np.float32))

    def get(self, key: tuple) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Retrieve per-position cached distribution. Returns None on miss."""
        result = self._store.get(key)
        if result is None:
            self._misses += 1
        else:
            self._hits += 1
        return result

    def populate(self, teacher, input_ids: np.ndarray) -> None:
        """
        Run teacher over one batch and cache per-position (M0 path).

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

    # ── Per-batch API (M3) ─────────────────────────────────────────────────

    def put_batch(
        self,
        batch_idx: int,
        indices:   np.ndarray,
        probs:     np.ndarray,
    ) -> None:
        """
        Store a full (B, S, K) distribution for an entire batch.

        Args:
            batch_idx: Integer key — index of the batch in the dataset list.
            indices:   (B, S, K) int32 top-k token indices from the teacher.
            probs:     (B, S, K) float32 top-k probabilities.
        """
        if self.max_entries and len(self._store) >= self.max_entries:
            evict = next(iter(self._store))
            del self._store[evict]
        self._store[batch_idx] = (indices.astype(np.int32), probs.astype(np.float32))

    def get_batch(
        self, batch_idx: int
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve (indices, probs) for a full batch. Returns None on miss.

        Returns:
            (indices, probs) both shaped (B, S, K), or None.
        """
        result = self._store.get(batch_idx)
        if result is None:
            self._misses += 1
        else:
            self._hits += 1
        return result

    def populate_dataset(
        self,
        teacher,
        dataset: list[np.ndarray],
    ) -> None:
        """
        Run teacher over every batch in the dataset and cache per-batch.
        After this call, get_batch(i) will hit for all i in [0, len(dataset)).

        Args:
            teacher: Any Teacher protocol impl.
            dataset: List of (B, S) int arrays — the full training dataset.
        """
        for batch_idx, batch_ids in enumerate(dataset):
            if batch_ids.ndim == 1:
                batch_ids = batch_ids[None, :]
            indices, probs = teacher.distribution(batch_ids, top_k=self.top_k)
            self.put_batch(batch_idx, indices, probs)

    # ── Stats ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "size":     len(self._store),
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
        }

    # ── Persistence (M3) ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Persist cache to a .npz file, including all keys.

        Format: arrays ``key_{i}`` (serialised key), ``idx_{i}``, ``prob_{i}``
        plus a ``n_entries`` scalar and ``is_int_key`` boolean array.
        """
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        n = len(self._store)
        arrays["n_entries"] = np.array([n], dtype=np.int64)

        is_int_flags = []
        for i, (key, (idx, prob)) in enumerate(self._store.items()):
            is_int = isinstance(key, int)
            is_int_flags.append(is_int)
            arrays[f"key_{i}"] = np.array([key] if is_int else list(key), dtype=np.int64)
            arrays[f"idx_{i}"]  = idx
            arrays[f"prob_{i}"] = prob
        arrays["is_int_key"] = np.array(is_int_flags, dtype=bool)
        np.savez_compressed(path, **arrays)

    def load(self, path: str | Path) -> None:
        """
        Load cache from a .npz file saved by ``save()``.
        Merges into the existing in-memory store.
        """
        path = Path(path)
        if not path.exists():
            # Accept path without .npz suffix (numpy appends it automatically)
            path = Path(str(path) + ".npz")
        data = np.load(path, allow_pickle=False)
        n       = int(data["n_entries"][0])
        is_ints = data["is_int_key"]
        for i in range(n):
            raw_key = data[f"key_{i}"]
            key: Union[int, tuple] = (
                int(raw_key[0]) if is_ints[i] else tuple(int(x) for x in raw_key)
            )
            idx  = data[f"idx_{i}"]
            prob = data[f"prob_{i}"]
            self._store[key] = (idx.astype(np.int32), prob.astype(np.float32))

    def clear(self) -> None:
        self._store.clear()
        self._hits   = 0
        self._misses = 0
