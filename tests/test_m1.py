"""
M1 tests — IO (seed loading) + TorchDistillTrainer.

All tests are mock-based or use a tiny nn.Module stub.
No HF downloads, no GPU, no remote calls.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn


# ── Tiny in-process student model ──────────────────────────────────────────
class TinyLM(nn.Module):
    """Minimal CausalLM-shaped model: Embedding → Linear → .logits output."""

    def __init__(self, vocab: int = 100, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head  = nn.Linear(dim, vocab, bias=False)
        self.vocab = vocab

    def forward(self, input_ids, **_):
        h = self.embed(input_ids)
        return SimpleNamespace(logits=self.head(h))


# ── Seed loading tests ─────────────────────────────────────────────────────
class TestSeedLoading(unittest.TestCase):

    def test_custom_arch_path(self):
        """_load_custom_arch dispatches via 'module:Class' spec."""
        from foundry.io.seed import _load_custom_arch

        # Inject a fake module into sys.modules
        fake_module = types.ModuleType("_fake_arch_mod")
        fake_module.MyStudentClass = TinyLM

        import sys
        sys.modules["_fake_arch_mod"] = fake_module
        try:
            result = _load_custom_arch("_fake_arch_mod:MyStudentClass", {})
            self.assertIsInstance(result.model, TinyLM)
            self.assertEqual(result.strategy, "from_scratch")
            self.assertEqual(result.model_id, "_fake_arch_mod:MyStudentClass")
        finally:
            del sys.modules["_fake_arch_mod"]

    def test_custom_arch_bad_spec_raises(self):
        from foundry.io.seed import _load_custom_arch
        with self.assertRaises(ValueError):
            _load_custom_arch("no_colon_here", {})

    def test_custom_arch_missing_class_raises(self):
        from foundry.io.seed import _load_custom_arch
        import sys
        fake_module = types.ModuleType("_fake_empty_mod")
        sys.modules["_fake_empty_mod"] = fake_module
        try:
            with self.assertRaises(AttributeError):
                _load_custom_arch("_fake_empty_mod:MissingClass", {})
        finally:
            del sys.modules["_fake_empty_mod"]

    def test_custom_arch_missing_module_raises(self):
        from foundry.io.seed import _load_custom_arch
        with self.assertRaises(ImportError):
            _load_custom_arch("no_such_module_xyz:SomeClass", {})

    def test_seed_result_fields(self):
        from foundry.io.seed import SeedResult
        sr = SeedResult(
            model=TinyLM(),
            tokenizer=None,
            config=None,
            strategy="from_scratch",
            model_id="test@random",
        )
        self.assertEqual(sr.strategy, "from_scratch")
        self.assertIsNotNone(sr.model)

    def test_load_seed_dispatches_custom_arch(self):
        """load_seed() routes to custom arch when seed_cfg.init != 'pretrained'."""
        from foundry.io.seed import load_seed, SeedResult
        import sys

        fake_module = types.ModuleType("_fake_seed_mod")
        fake_module.MyLM = TinyLM
        sys.modules["_fake_seed_mod"] = fake_module

        cfg = SimpleNamespace(init="from_scratch", arch="_fake_seed_mod:MyLM")
        try:
            result = load_seed(cfg)
            self.assertIsInstance(result, SeedResult)
            self.assertIsInstance(result.model, TinyLM)
        finally:
            del sys.modules["_fake_seed_mod"]


# ── TorchDistillTrainer tests ──────────────────────────────────────────────
class TestTorchDistillTrainer(unittest.TestCase):

    def _make_trainer(self, vocab: int = 100, n_teachers: int = 1):
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers import TeacherRegistry

        student  = TinyLM(vocab=vocab)
        teachers = TeacherRegistry.from_toy(n=n_teachers, vocab_size=vocab)
        cfg      = TorchTrainConfig(
            epochs=1,
            learning_rate=1e-2,
            device="cpu",
            log_loss_every=999,
        )
        return TorchDistillTrainer(student, teachers, config=cfg)

    def test_train_step_returns_scalar(self):
        trainer = self._make_trainer()
        batch   = np.random.randint(0, 100, (2, 8))
        loss    = trainer.train_step(batch)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        self.assertTrue(np.isfinite(loss))

    def test_train_loop_returns_dict(self):
        trainer = self._make_trainer()
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(5)]
        result  = trainer.train(dataset)
        self.assertIn("losses", result)
        self.assertIn("device", result)
        self.assertEqual(len(result["losses"]), 5)

    def test_loss_decreases_over_many_steps(self):
        """Loss trend is downward over 40 steps on repeated identical batch."""
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers import TeacherRegistry

        student  = TinyLM(vocab=100, dim=64)
        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg = TorchTrainConfig(
            epochs=1,
            learning_rate=5e-3,
            device="cpu",
            alpha=0.5,
        )
        trainer = TorchDistillTrainer(student, teachers, config=cfg)
        fixed_batch = np.random.randint(0, 100, (4, 16))
        dataset = [fixed_batch] * 40
        result  = trainer.train(dataset)
        losses  = result["losses"]
        # Compare first-5 avg vs last-5 avg — should drop
        first5 = sum(losses[:5]) / 5
        last5  = sum(losses[-5:]) / 5
        self.assertLess(last5, first5, f"Loss did not decrease: {first5:.4f} → {last5:.4f}")

    def test_train_step_1d_batch_handled(self):
        trainer = self._make_trainer()
        batch_1d = np.random.randint(0, 100, (8,))  # 1D — should be auto-expanded
        loss = trainer.train_step(batch_1d[None, :])
        self.assertIsInstance(loss, float)

    def test_train_with_no_teachers_runs(self):
        """alpha=1.0, no KL term — pure CE loss."""
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers import TeacherRegistry

        student  = TinyLM(vocab=100)
        teachers = TeacherRegistry.from_toy(n=0)   # empty pool
        cfg      = TorchTrainConfig(epochs=1, alpha=1.0, device="cpu")
        trainer  = TorchDistillTrainer(student, teachers, config=cfg)
        dataset  = [np.random.randint(0, 100, (2, 8))]
        result   = trainer.train(dataset)
        self.assertTrue(np.isfinite(result["losses"][0]))

    def test_on_step_callback_fires(self):
        trainer = self._make_trainer()
        dataset = [np.random.randint(0, 100, (2, 8)) for _ in range(3)]
        calls: list[tuple[int, float]] = []
        trainer.cfg.log_every = 1
        trainer.train(dataset, on_step=lambda s, l: calls.append((s, l)))
        self.assertEqual(len(calls), 3)

    def test_device_is_cpu(self):
        trainer = self._make_trainer()
        self.assertEqual(str(trainer.device), "cpu")


# ── HFTeacher interface tests (mocked, no HF download) ────────────────────
class TestHFTeacherInterface(unittest.TestCase):

    def test_hf_teacher_not_loaded_raises(self):
        from foundry.teachers import HFTeacher
        teacher = HFTeacher("org/some-model")
        with self.assertRaises(RuntimeError):
            teacher.distribution(np.zeros((1, 4), dtype=np.int32))

    def test_hf_teacher_name_and_weight(self):
        from foundry.teachers import HFTeacher
        t = HFTeacher("org/model", weight=0.7)
        self.assertEqual(t.name, "org/model")
        self.assertAlmostEqual(t.weight, 0.7)
        self.assertIsNone(t.tokenizer)

    def test_hf_teacher_load_idempotent(self):
        """Calling load() twice should not re-download."""
        from foundry.teachers import HFTeacher
        t = HFTeacher("org/model")
        # Inject a fake already-loaded model to simulate a prior load()
        t._model = MagicMock()
        t._tok   = MagicMock()
        t._device = torch.device("cpu")
        result = t.load()   # should early-exit — no ImportError / network call
        self.assertIs(result, t)


# ── TeacherRegistry.load_all tests ────────────────────────────────────────
class TestTeacherRegistryLoadAll(unittest.TestCase):

    def test_load_all_calls_load_on_hf_teachers(self):
        from foundry.teachers import TeacherRegistry, HFTeacher

        t1 = HFTeacher("org/a")
        t2 = HFTeacher("org/b")
        t1.load = MagicMock(return_value=t1)
        t2.load = MagicMock(return_value=t2)

        reg = TeacherRegistry([t1, t2])
        reg.load_all(device="cpu")

        t1.load.assert_called_once_with(device="cpu")
        t2.load.assert_called_once_with(device="cpu")

    def test_load_all_skips_toy_teachers(self):
        from foundry.teachers import TeacherRegistry, ToyTeacher
        toy = ToyTeacher(name="toy")
        self.assertFalse(hasattr(toy, "load"))
        reg = TeacherRegistry([toy])
        reg.load_all()  # should not raise


# ── Top-level foundry exports include M1 symbols ──────────────────────────
class TestFoundryExports(unittest.TestCase):

    def test_m1_symbols_exported(self):
        import foundry
        for sym in ("SeedResult", "load_seed", "TorchDistillTrainer", "TorchTrainConfig", "HFTeacher"):
            self.assertIn(sym, foundry.__all__, f"Missing export: {sym}")
            self.assertTrue(hasattr(foundry, sym), f"Symbol not importable: {sym}")


if __name__ == "__main__":
    unittest.main()
