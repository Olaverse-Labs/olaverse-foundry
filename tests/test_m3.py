"""
M3 tests — on-disk LogitCache + CachedDistillTrainer.

All tests use TinyLM / ToyTeacher — no HF downloads, no GPU.
accelerate is bypassed via use_accelerate=False.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# ── Tiny model shared across tests ─────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab: int = 100, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head  = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, **_):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


# ── LogitCache — per-batch API ──────────────────────────────────────────────

class TestLogitCacheBatchAPI(unittest.TestCase):

    def test_put_get_batch_round_trip(self):
        from foundry.teachers.cache import LogitCache
        cache = LogitCache(top_k=4)
        idx   = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=np.int32)   # (1,2,4)
        prob  = np.ones((1, 2, 4), dtype=np.float32) * 0.25
        cache.put_batch(0, idx, prob)
        result = cache.get_batch(0)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[0], idx)
        np.testing.assert_array_almost_equal(result[1], prob)

    def test_get_batch_miss_returns_none(self):
        from foundry.teachers.cache import LogitCache
        cache = LogitCache()
        self.assertIsNone(cache.get_batch(99))

    def test_get_batch_increments_miss_counter(self):
        from foundry.teachers.cache import LogitCache
        cache = LogitCache()
        cache.get_batch(0)
        self.assertEqual(cache.stats["misses"], 1)

    def test_get_batch_increments_hit_counter(self):
        from foundry.teachers.cache import LogitCache
        cache = LogitCache()
        cache.put_batch(0, np.zeros((1, 1, 2), dtype=np.int32),
                           np.ones((1, 1, 2), dtype=np.float32) * 0.5)
        cache.get_batch(0)
        self.assertEqual(cache.stats["hits"], 1)

    def test_populate_dataset_fills_all_batches(self):
        from foundry.teachers import TeacherRegistry
        from foundry.teachers.cache import LogitCache
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(5)]
        teacher = list(TeacherRegistry.from_toy(n=1, vocab_size=100))[0]
        cache   = LogitCache(top_k=4)
        cache.populate_dataset(teacher, dataset)
        for i in range(5):
            self.assertIsNotNone(cache.get_batch(i), f"batch {i} not cached")

    def test_populate_dataset_correct_shapes(self):
        from foundry.teachers import TeacherRegistry
        from foundry.teachers.cache import LogitCache
        B, S, K = 3, 6, 4
        dataset = [np.random.randint(0, 100, (B, S))]
        teacher = list(TeacherRegistry.from_toy(n=1, vocab_size=100))[0]
        cache   = LogitCache(top_k=K)
        cache.populate_dataset(teacher, dataset)
        idx, prob = cache.get_batch(0)
        self.assertEqual(idx.shape,  (B, S, K))
        self.assertEqual(prob.shape, (B, S, K))


# ── LogitCache — save / load ────────────────────────────────────────────────

class TestLogitCachePersistence(unittest.TestCase):

    def _make_cache(self) -> "LogitCache":
        from foundry.teachers.cache import LogitCache
        cache = LogitCache(top_k=4)
        # Per-token keys (M0 style)
        cache.put((0, 0, 0), np.array([1, 2], dtype=np.int32),
                             np.array([0.6, 0.4], dtype=np.float32))
        # Per-batch keys (M3 style)
        cache.put_batch(1, np.zeros((2, 4, 4), dtype=np.int32),
                           np.ones((2, 4, 4), dtype=np.float32) * 0.25)
        return cache

    def test_save_creates_npz_file(self):
        from foundry.teachers.cache import LogitCache
        cache = self._make_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            cache.save(path)
            self.assertTrue(path.exists())

    def test_load_restores_batch_entry(self):
        from foundry.teachers.cache import LogitCache
        cache = self._make_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache"
            cache.save(str(path) + ".npz")
            restored = LogitCache(top_k=4)
            restored.load(str(path) + ".npz")
            result = restored.get_batch(1)
            self.assertIsNotNone(result)
            self.assertEqual(result[0].shape, (2, 4, 4))

    def test_load_restores_tuple_entry(self):
        from foundry.teachers.cache import LogitCache
        cache = self._make_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            cache.save(path)
            restored = LogitCache(top_k=4)
            restored.load(path)
            result = restored.get((0, 0, 0))
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result[0], [1, 2])

    def test_save_load_round_trip_size(self):
        from foundry.teachers.cache import LogitCache
        cache = self._make_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.npz"
            cache.save(path)
            restored = LogitCache(top_k=4)
            restored.load(path)
            self.assertEqual(restored.stats["size"], 2)

    def test_load_merges_into_existing(self):
        from foundry.teachers.cache import LogitCache
        cache = self._make_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.npz"
            cache.save(path)
            existing = LogitCache(top_k=4)
            existing.put_batch(99, np.zeros((1, 1, 4), dtype=np.int32),
                                   np.zeros((1, 1, 4), dtype=np.float32))
            existing.load(path)
            self.assertEqual(existing.stats["size"], 3)  # 2 loaded + 1 pre-existing


# ── CachedDistillTrainer ────────────────────────────────────────────────────

class TestCachedDistillTrainer(unittest.TestCase):

    def _make_trainer(self, vocab: int = 100, n_teachers: int = 1, **cfg_kwargs):
        from foundry.training import CachedDistillTrainer, CachedDistillConfig
        from foundry.teachers import TeacherRegistry
        student  = TinyLM(vocab=vocab)
        teachers = TeacherRegistry.from_toy(n=n_teachers, vocab_size=vocab)
        cfg = CachedDistillConfig(
            epochs=1,
            device="cpu",
            use_accelerate=False,
            cache_top_k=4,
            **cfg_kwargs,
        )
        return CachedDistillTrainer(student, teachers, config=cfg)

    def test_train_returns_expected_keys(self):
        trainer = self._make_trainer()
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(3)]
        result  = trainer.train(dataset)
        self.assertIn("losses",      result)
        self.assertIn("device",      result)
        self.assertIn("cache_stats", result)

    def test_losses_list_length(self):
        trainer = self._make_trainer()
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(4)]
        result  = trainer.train(dataset)
        self.assertEqual(len(result["losses"]), 4)

    def test_losses_are_finite(self):
        trainer = self._make_trainer()
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(3)]
        result  = trainer.train(dataset)
        for l in result["losses"]:
            self.assertTrue(np.isfinite(l))

    def test_cache_stats_per_teacher(self):
        trainer = self._make_trainer(n_teachers=2)
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(3)]
        result  = trainer.train(dataset)
        self.assertEqual(len(result["cache_stats"]), 2)

    def test_cache_populated_after_first_epoch(self):
        """After train(), caches should have entries for all batches."""
        trainer = self._make_trainer(n_teachers=1)
        n_batches = 4
        dataset   = [np.random.randint(0, 100, (2, 8)) for _ in range(n_batches)]
        trainer.train(dataset)
        cache_size = trainer._caches[0].stats["size"]
        self.assertGreaterEqual(cache_size, n_batches)

    def test_second_epoch_reads_from_cache(self):
        """In a 2-epoch run, epoch 2 should have cache hits."""
        from foundry.training import CachedDistillTrainer, CachedDistillConfig
        from foundry.teachers import TeacherRegistry
        student  = TinyLM(vocab=100)
        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg = CachedDistillConfig(
            epochs=2, device="cpu", use_accelerate=False, cache_top_k=4
        )
        trainer = CachedDistillTrainer(student, teachers, config=cfg)
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(3)]
        result  = trainer.train(dataset)
        hits = trainer._caches[0].stats["hits"]
        # epoch 0: 3 cache misses + populate; epoch 1: 3 cache hits
        self.assertGreater(hits, 0)

    def test_disk_cache_saved_and_loaded(self):
        """cache_dir causes caches to be saved; second trainer loads them."""
        from foundry.training import CachedDistillTrainer, CachedDistillConfig
        from foundry.teachers import TeacherRegistry
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmp:
            # First run — populates and saves
            teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
            cfg1 = CachedDistillConfig(
                epochs=1, device="cpu", use_accelerate=False,
                cache_top_k=4, cache_dir=tmp,
            )
            t1 = CachedDistillTrainer(TinyLM(vocab=100), teachers, config=cfg1)
            t1.train(dataset)
            saved = list(Path(tmp).glob("*.npz"))
            self.assertGreater(len(saved), 0, "No .npz files saved")

            # Second run — loads from disk (no teacher inference)
            teachers2 = TeacherRegistry.from_toy(n=1, vocab_size=100)
            cfg2 = CachedDistillConfig(
                epochs=1, device="cpu", use_accelerate=False,
                cache_top_k=4, cache_dir=tmp,
            )
            t2 = CachedDistillTrainer(TinyLM(vocab=100), teachers2, config=cfg2)
            t2.train(dataset)
            # After loading from disk, all batches should hit
            hits = t2._caches[0].stats["hits"]
            self.assertGreater(hits, 0)

    def test_loss_decreases_with_caching(self):
        """Loss trend is downward over 30 steps even with caching."""
        from foundry.training import CachedDistillTrainer, CachedDistillConfig
        from foundry.teachers import TeacherRegistry
        student  = TinyLM(vocab=100, dim=64)
        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg = CachedDistillConfig(
            epochs=1, learning_rate=5e-3, device="cpu",
            use_accelerate=False, alpha=0.5, cache_top_k=4,
        )
        trainer = CachedDistillTrainer(student, teachers, config=cfg)
        fixed   = np.random.randint(0, 100, (4, 16))
        dataset = [fixed] * 30
        result  = trainer.train(dataset)
        losses  = result["losses"]
        first5  = sum(losses[:5])  / 5
        last5   = sum(losses[-5:]) / 5
        self.assertLess(last5, first5)

    def test_grad_accumulation_steps_config(self):
        """CachedDistillConfig accepts grad_accumulation_steps."""
        from foundry.training import CachedDistillConfig
        cfg = CachedDistillConfig(grad_accumulation_steps=4)
        self.assertEqual(cfg.grad_accumulation_steps, 4)

    def test_no_teachers_pure_ce(self):
        """alpha=1.0, no KL term — trainer still runs."""
        from foundry.training import CachedDistillTrainer, CachedDistillConfig
        from foundry.teachers import TeacherRegistry
        teachers = TeacherRegistry.from_toy(n=0)
        cfg = CachedDistillConfig(
            epochs=1, alpha=1.0, device="cpu", use_accelerate=False
        )
        trainer = CachedDistillTrainer(TinyLM(), teachers, config=cfg)
        result  = trainer.train([np.random.randint(0, 100, (2, 8))])
        self.assertTrue(np.isfinite(result["losses"][0]))


# ── Top-level foundry exports ───────────────────────────────────────────────

class TestFoundryM3Exports(unittest.TestCase):

    def test_m3_symbols_exported(self):
        import foundry
        for sym in ("CachedDistillTrainer", "CachedDistillConfig"):
            self.assertIn(sym, foundry.__all__, f"Missing from __all__: {sym}")
            self.assertTrue(hasattr(foundry, sym))


if __name__ == "__main__":
    unittest.main()
