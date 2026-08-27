"""
LogitCache — generate teacher signal once, reuse forever.

Caching is the single biggest cost lever: large teachers over hundreds of
billions of tokens can cost as much as training the student itself.

Two key modes:
  per-token (M0): tuple keys (token_id, batch, seq); fine-grained, big RAM
  per-batch (M3): integer keys (batch_idx); stores full (B,S,K) arrays; disk-friendly

Memory
------
Cached top-k distributions are large. At ``top_k=64`` with 8x512 batches each
entry is ~2MB, so 10,000 batches — only 41M tokens — is ~21GB held in RAM. Real
distillation runs are far bigger than that, which makes an unbounded in-memory
dict the wrong default at any serious scale.

Pass ``cache_dir`` to bound it::

    cache = LogitCache(top_k=64, max_entries=512, cache_dir="./teacher_cache")

RAM then holds at most ``max_entries`` entries under **LRU** eviction, and
evicted entries spill to sharded ``.npz`` files rather than being thrown away.
A miss in RAM falls through to disk and is promoted back. Because the shards and
their index live in ``cache_dir``, reopening a cache with the same directory
picks up everything a previous *process* wrote: populate once, reuse across runs.

Eviction is LRU, not FIFO. Under FIFO a sequential epoch pass evicts precisely
the entries the next epoch reads first, so a cache smaller than the dataset
achieves a ~0% hit rate — the worst possible policy for this access pattern.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np


class LogitCache:
    """
    Stores top-k (indices, probs) distributions for one teacher.

    Supports two key types:
      - tuple  (M0): per-position key ``(token_id, batch, seq)``
      - int    (M3): per-batch key ``batch_idx``; value is ``(B, S, K)`` arrays

    Args:
        top_k:       Number of top logits to store per position.
        max_entries: Cap on in-memory entries (LRU eviction). 0 = unlimited.
        cache_dir:   Spill evicted entries here as sharded ``.npz`` files instead
                     of discarding them, and read them back on a RAM miss. The
                     directory is reused across processes, so a populated cache
                     survives the run that built it. ``None`` = memory only.
        shard_size:  Entries per shard file. Larger shards mean fewer files and
                     faster sequential reads; smaller shards mean less work to
                     serve a single random key.
    """

    def __init__(self, top_k: int = 64, max_entries: int = 0,
                 cache_dir: Optional[str | Path] = None,
                 shard_size: int = 64) -> None:
        self.top_k = top_k
        self.max_entries = max_entries
        self.shard_size = max(1, shard_size)

        # Insertion-ordered and reordered on access → popitem(last=False) is LRU.
        self._store: OrderedDict[Union[tuple, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._hits      = 0
        self._misses    = 0
        self._disk_hits = 0
        self._spills    = 0

        self._cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        self._index: dict[str, int] = {}          # serialised key → shard id
        self._pending: OrderedDict = OrderedDict()  # awaiting a shard flush
        self._shards: OrderedDict[int, dict] = OrderedDict()  # loaded shards (LRU)
        self._max_loaded_shards = 4
        self._next_shard = 0
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._read_index()

    # ── Disk tier ──────────────────────────────────────────────────────────

    @staticmethod
    def _key_str(key: Union[tuple, int]) -> str:
        """Stable text form of a key, for the on-disk index."""
        if isinstance(key, (int, np.integer)):
            return f"i:{int(key)}"
        return "t:" + ",".join(str(int(x)) for x in key)

    @staticmethod
    def _key_from_str(text: str) -> Union[tuple, int]:
        kind, _, body = text.partition(":")
        if kind == "i":
            return int(body)
        return tuple(int(x) for x in body.split(","))

    @property
    def _index_path(self) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / "index.json"

    def _shard_path(self, shard_id: int) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / f"shard_{shard_id:05d}.npz"

    def _read_index(self) -> None:
        """Adopt an index written by a previous run, if present."""
        if not self._index_path.exists():
            return
        meta = json.loads(self._index_path.read_text())
        self._index = {str(k): int(v) for k, v in meta.get("index", {}).items()}
        self._next_shard = int(meta.get("next_shard", 0))

    def _write_index(self) -> None:
        self._index_path.write_text(
            json.dumps({"index": self._index, "next_shard": self._next_shard,
                        "top_k": self.top_k})
        )

    def _spill(self, key: Union[tuple, int], value: tuple) -> None:
        """Queue an evicted entry for the next shard."""
        self._pending[key] = value
        self._spills += 1
        if len(self._pending) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self) -> None:
        """Write queued entries to one shard file and index them."""
        if not self._pending or self._cache_dir is None:
            return
        shard_id = self._next_shard
        self._next_shard += 1

        arrays: dict[str, np.ndarray] = {}
        keys = list(self._pending)
        for i, key in enumerate(keys):
            idx, prob = self._pending[key]
            arrays[f"idx_{i}"]  = idx
            arrays[f"prob_{i}"] = prob
            self._index[self._key_str(key)] = shard_id
        arrays["keys"] = np.array([self._key_str(k) for k in keys], dtype="U64")

        np.savez_compressed(self._shard_path(shard_id), **arrays)  # type: ignore[arg-type]
        self._pending.clear()
        self._write_index()

    def _load_shard(self, shard_id: int) -> dict:
        """Load a shard, keeping a small LRU of them in memory."""
        cached = self._shards.get(shard_id)
        if cached is not None:
            self._shards.move_to_end(shard_id)
            return cached

        data = np.load(self._shard_path(shard_id), allow_pickle=False)
        entries: dict = {}
        for i, key_text in enumerate(data["keys"]):
            entries[self._key_from_str(str(key_text))] = (
                data[f"idx_{i}"].astype(np.int32),
                data[f"prob_{i}"].astype(np.float32),
            )
        self._shards[shard_id] = entries
        while len(self._shards) > self._max_loaded_shards:
            self._shards.popitem(last=False)
        return entries

    def _from_disk(self, key: Union[tuple, int]) -> Optional[tuple]:
        """Look a key up in the pending buffer, then in the shards."""
        if key in self._pending:
            return self._pending[key]
        if self._cache_dir is None:
            return None
        shard_id = self._index.get(self._key_str(key))
        if shard_id is None:
            return None
        return self._load_shard(shard_id).get(key)

    def flush(self) -> None:
        """Write any queued entries to disk. Call before relying on the shards."""
        self._flush_shard()

    # ── Memory tier ────────────────────────────────────────────────────────

    def _admit(self, key: Union[tuple, int], value: tuple) -> None:
        """Insert into the memory tier, evicting the least-recently-used entry."""
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if self.max_entries and len(self._store) >= self.max_entries:
            evicted_key, evicted_value = self._store.popitem(last=False)
            if self._cache_dir is not None:
                self._spill(evicted_key, evicted_value)
        self._store[key] = value

    def _lookup(self, key: Union[tuple, int]) -> Optional[tuple]:
        """Read through memory, then disk. Counts hits and misses."""
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
            self._hits += 1
            return value

        value = self._from_disk(key)
        if value is not None:
            self._hits += 1
            self._disk_hits += 1
            self._admit(key, value)
            return value

        self._misses += 1
        return None

    # ── Per-token API (M0) ─────────────────────────────────────────────────

    def put(
        self,
        key:     tuple,
        indices: np.ndarray,
        probs:   np.ndarray,
    ) -> None:
        """Store a per-position top-k distribution under a tuple key."""
        self._admit(key, (indices.astype(np.int32), probs.astype(np.float32)))

    def get(self, key: tuple) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Retrieve per-position cached distribution. Returns None on miss."""
        return self._lookup(key)

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
        self._admit(batch_idx, (indices.astype(np.int32), probs.astype(np.float32)))

    def get_batch(
        self, batch_idx: int
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve (indices, probs) for a full batch. Returns None on miss.

        Returns:
            (indices, probs) both shaped (B, S, K), or None.
        """
        return self._lookup(batch_idx)

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
            "size":       len(self._store),
            "hits":       self._hits,
            "misses":     self._misses,
            "hit_rate":   round(self._hits / max(1, self._hits + self._misses), 3),
            # Disk-tier counters. disk_hits are included in hits: a read served
            # from a shard still saved a teacher forward pass, which is the
            # thing the cache exists to avoid.
            "disk_hits":  self._disk_hits,
            "spills":     self._spills,
            "on_disk":    len(self._index) + len(self._pending),
        }

    # ── Persistence (M3) ──────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Persist cache to a .npz file, including all keys.

        Format: arrays ``key_{i}`` (serialised key), ``idx_{i}``, ``prob_{i}``
        plus a ``n_entries`` scalar and ``is_int_key`` boolean array.

        This writes the **in-memory** tier only. When ``cache_dir`` is set the
        shards there are already the durable copy and are reloaded automatically
        by any cache opened on the same directory, so save() is for the
        memory-only mode. Pending spills are flushed first so nothing is lost
        between the two representations.
        """
        if self._cache_dir is not None:
            self._flush_shard()
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        n = len(self._store)
        arrays["n_entries"] = np.array([n], dtype=np.int64)

        is_int_flags = []
        for i, (key, (idx, prob)) in enumerate(self._store.items()):
            is_int = isinstance(key, int)
            is_int_flags.append(is_int)
            flat = [key] if is_int else list(cast("tuple[int, ...]", key))
            arrays[f"key_{i}"] = np.array(flat, dtype=np.int64)
            arrays[f"idx_{i}"]  = idx
            arrays[f"prob_{i}"] = prob
        arrays["is_int_key"] = np.array(is_int_flags, dtype=bool)
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]  # numpy stub types **kwds as bool

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
            self._admit(key, (idx.astype(np.int32), prob.astype(np.float32)))

    def clear(self) -> None:
        """Drop the in-memory tier and counters. Shards on disk are left alone."""
        self._store.clear()
        self._shards.clear()
        self._hits      = 0
        self._misses    = 0
        self._disk_hits = 0
        self._spills    = 0
