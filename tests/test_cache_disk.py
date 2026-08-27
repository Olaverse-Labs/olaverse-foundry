"""
LogitCache: LRU eviction and the disk tier.

The failure this guards is quiet and expensive. Under the old FIFO policy a
cache smaller than the dataset evicted exactly the entries the next epoch read
first, so the hit rate collapsed to ~0 and every epoch re-ran the teacher — the
one cost the cache exists to avoid. Nothing errored; the run was just slow.

numpy-only, so these run on a core install.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from foundry.teachers.cache import LogitCache


def _entry(k=4, b=2, s=3, fill=1):
    idx  = np.full((b, s, k), fill, dtype=np.int32)
    prob = np.full((b, s, k), 1.0 / k, dtype=np.float32)
    return idx, prob


class TestLRUEviction(unittest.TestCase):

    def test_evicts_least_recently_used_not_oldest(self):
        cache = LogitCache(top_k=4, max_entries=2)
        cache.put_batch(0, *_entry(fill=0))
        cache.put_batch(1, *_entry(fill=1))

        cache.get_batch(0)              # 0 is now the most recent
        cache.put_batch(2, *_entry(fill=2))   # evicts the LRU, which is 1

        self.assertIsNotNone(cache.get_batch(0), "LRU evicted the recently-used entry")
        self.assertIsNone(cache.get_batch(1))
        self.assertIsNotNone(cache.get_batch(2))

    def test_respects_max_entries(self):
        cache = LogitCache(top_k=4, max_entries=3)
        for i in range(10):
            cache.put_batch(i, *_entry(fill=i))
        self.assertLessEqual(cache.stats["size"], 3)

    def test_unlimited_by_default(self):
        cache = LogitCache(top_k=4)
        for i in range(20):
            cache.put_batch(i, *_entry(fill=i))
        self.assertEqual(cache.stats["size"], 20)

    def test_reinsert_does_not_grow_store(self):
        cache = LogitCache(top_k=4, max_entries=2)
        cache.put_batch(0, *_entry(fill=0))
        cache.put_batch(0, *_entry(fill=9))
        self.assertEqual(cache.stats["size"], 1)
        idx, _ = cache.get_batch(0)
        self.assertEqual(int(idx.flat[0]), 9, "re-insert should overwrite")

    def test_sequential_epochs_keep_hitting(self):
        """The FIFO regression test: a full second pass must hit, not miss."""
        cache = LogitCache(top_k=4, max_entries=8)
        for i in range(8):
            cache.put_batch(i, *_entry(fill=i))
        hits_before = cache.stats["hits"]
        for i in range(8):
            self.assertIsNotNone(cache.get_batch(i))
        self.assertEqual(cache.stats["hits"] - hits_before, 8)


class TestDiskTier(unittest.TestCase):

    def test_evicted_entries_survive_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=4, max_entries=2, cache_dir=tmp, shard_size=2)
            for i in range(6):
                cache.put_batch(i, *_entry(fill=i))
            cache.flush()

            for i in range(6):
                got = cache.get_batch(i)
                self.assertIsNotNone(got, f"batch {i} lost")
                self.assertEqual(int(got[0].flat[0]), i)

    def test_memory_stays_bounded_while_serving_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=4, max_entries=2, cache_dir=tmp, shard_size=2)
            for i in range(12):
                cache.put_batch(i, *_entry(fill=i))
            cache.flush()
            for i in range(12):
                cache.get_batch(i)
            self.assertLessEqual(cache.stats["size"], 2)

    def test_disk_hits_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=4, max_entries=1, cache_dir=tmp, shard_size=1)
            cache.put_batch(0, *_entry(fill=0))
            cache.put_batch(1, *_entry(fill=1))   # evicts 0 to disk
            cache.flush()
            self.assertIsNotNone(cache.get_batch(0))
            self.assertGreater(cache.stats["disk_hits"], 0)

    def test_shards_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=4, max_entries=1, cache_dir=tmp, shard_size=2)
            for i in range(5):
                cache.put_batch(i, *_entry(fill=i))
            cache.flush()
            names = {p.name for p in Path(tmp).iterdir()}
            self.assertIn("index.json", names)
            self.assertTrue(any(n.startswith("shard_") for n in names), names)

    def test_cache_survives_a_new_process(self):
        """A second cache on the same directory must reuse the first one's work."""
        with tempfile.TemporaryDirectory() as tmp:
            first = LogitCache(top_k=4, max_entries=1, cache_dir=tmp, shard_size=2)
            for i in range(6):
                first.put_batch(i, *_entry(fill=i))
            first.flush()

            second = LogitCache(top_k=4, max_entries=1, cache_dir=tmp, shard_size=2)
            self.assertEqual(second.stats["size"], 0, "fresh cache starts cold in RAM")
            got = second.get_batch(3)
            self.assertIsNotNone(got, "did not pick up shards from the previous run")
            self.assertEqual(int(got[0].flat[0]), 3)

    def test_missing_key_still_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=4, max_entries=1, cache_dir=tmp)
            cache.put_batch(0, *_entry(fill=0))
            self.assertIsNone(cache.get_batch(999))
            self.assertGreater(cache.stats["misses"], 0)

    def test_tuple_keys_round_trip_through_disk(self):
        """The per-token (M0) path uses tuple keys, not ints."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=2, max_entries=1, cache_dir=tmp, shard_size=1)
            a = (np.array([1, 2], dtype=np.int32), np.array([0.6, 0.4], dtype=np.float32))
            b = (np.array([3, 4], dtype=np.int32), np.array([0.7, 0.3], dtype=np.float32))
            cache.put((7, 0, 0), *a)
            cache.put((8, 1, 1), *b)      # evicts the first to disk
            cache.flush()

            got = cache.get((7, 0, 0))
            self.assertIsNotNone(got)
            np.testing.assert_array_equal(got[0], a[0])
            np.testing.assert_allclose(got[1], a[1])

    def test_no_cache_dir_means_eviction_discards(self):
        cache = LogitCache(top_k=4, max_entries=1)
        cache.put_batch(0, *_entry(fill=0))
        cache.put_batch(1, *_entry(fill=1))
        self.assertIsNone(cache.get_batch(0))
        self.assertEqual(cache.stats["spills"], 0)

    def test_clear_keeps_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=4, max_entries=1, cache_dir=tmp, shard_size=1)
            for i in range(4):
                cache.put_batch(i, *_entry(fill=i))
            cache.flush()
            cache.clear()
            self.assertEqual(cache.stats["size"], 0)
            self.assertIsNotNone(cache.get_batch(0), "clear() should not drop shards")


class TestSaveLoadStillWorks(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = LogitCache(top_k=4)
            cache.put_batch(0, *_entry(fill=5))
            cache.put((1, 0, 0), np.array([1, 2], dtype=np.int32),
                                 np.array([0.5, 0.5], dtype=np.float32))
            cache.save(Path(tmp) / "c.npz")

            restored = LogitCache(top_k=4)
            restored.load(Path(tmp) / "c.npz")
            self.assertIsNotNone(restored.get_batch(0))
            self.assertIsNotNone(restored.get((1, 0, 0)))


if __name__ == "__main__":
    unittest.main()
