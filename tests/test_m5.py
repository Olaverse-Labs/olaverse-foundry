"""
M5 tests — EmbeddingDistillTrainer, EmbedRecipe schema, doctor, backends.

No HF downloads. ToyEmbeddingTeacher + TinyEncoder for all trainer tests.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# ── Fixtures ───────────────────────────────────────────────────────────────

class TinyEncoder(nn.Module):
    """Minimal BERT-shaped encoder: Embedding → Linear → last_hidden_state."""

    def __init__(self, vocab: int = 100, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj  = nn.Linear(dim, dim)

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.proj(self.embed(input_ids))   # (B, S, D)
        return SimpleNamespace(last_hidden_state=h)


def _batch(B=2, S=8, vocab=100):
    return {
        "input_ids":      np.random.randint(0, vocab, (B, S)).astype(np.int32),
        "attention_mask": np.ones((B, S), dtype=np.int32),
    }


# ── ToyEmbeddingTeacher ────────────────────────────────────────────────────

class TestToyEmbeddingTeacher(unittest.TestCase):

    def test_encode_shape(self):
        from foundry.training import ToyEmbeddingTeacher
        t   = ToyEmbeddingTeacher(dim=32)
        ids = np.random.randint(0, 100, (4, 8))
        out = t.encode(ids)
        self.assertEqual(out.shape, (4, 32))

    def test_encode_unit_vectors(self):
        from foundry.training import ToyEmbeddingTeacher
        t   = ToyEmbeddingTeacher(dim=16)
        ids = np.random.randint(0, 100, (3, 6))
        out = t.encode(ids)
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

    def test_encode_reproducible(self):
        from foundry.training import ToyEmbeddingTeacher
        t   = ToyEmbeddingTeacher(dim=16, seed=7)
        ids = np.array([[1, 2, 3]])
        out1 = t.encode(ids)
        out2 = t.encode(ids)
        np.testing.assert_array_equal(out1, out2)

    def test_encode_with_attention_mask(self):
        from foundry.training import ToyEmbeddingTeacher
        t    = ToyEmbeddingTeacher(dim=16)
        ids  = np.random.randint(0, 100, (2, 6))
        mask = np.ones((2, 6), dtype=np.int32)
        out  = t.encode(ids, attention_mask=mask)
        self.assertEqual(out.shape, (2, 16))


# ── EmbeddingDistillTrainer ────────────────────────────────────────────────

class TestEmbeddingDistillTrainer(unittest.TestCase):

    def _make_trainer(self, loss="mse", pool="mean", normalize=True, epochs=1):
        from foundry.training import (
            EmbeddingDistillTrainer, EmbeddingDistillConfig, ToyEmbeddingTeacher,
        )
        student = TinyEncoder(vocab=100, dim=32)
        teacher = ToyEmbeddingTeacher(dim=32)
        cfg = EmbeddingDistillConfig(
            loss=loss, pool=pool, normalize=normalize,
            epochs=epochs, device="cpu", learning_rate=1e-3,
        )
        return EmbeddingDistillTrainer(student, teacher, config=cfg)

    def test_train_step_returns_scalar(self):
        trainer = self._make_trainer()
        loss = trainer.train_step(_batch())
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))

    def test_mse_loss_positive(self):
        trainer = self._make_trainer(loss="mse")
        loss = trainer.train_step(_batch())
        self.assertGreater(loss, 0.0)

    def test_cosine_loss_positive(self):
        trainer = self._make_trainer(loss="cosine")
        loss = trainer.train_step(_batch())
        self.assertGreater(loss, 0.0)

    def test_train_returns_dict(self):
        trainer = self._make_trainer()
        result  = trainer.train([_batch() for _ in range(5)])
        self.assertIn("losses", result)
        self.assertIn("device", result)
        self.assertEqual(len(result["losses"]), 5)

    def test_loss_decreases_over_steps(self):
        from foundry.training import (
            EmbeddingDistillTrainer, EmbeddingDistillConfig, ToyEmbeddingTeacher,
        )
        student = TinyEncoder(vocab=100, dim=32)
        teacher = ToyEmbeddingTeacher(dim=32)
        cfg = EmbeddingDistillConfig(
            loss="mse", normalize=True, device="cpu",
            learning_rate=5e-3, epochs=1,
        )
        trainer = EmbeddingDistillTrainer(student, teacher, config=cfg)
        fixed   = _batch(B=4, S=12)
        result  = trainer.train([fixed] * 30)
        losses  = result["losses"]
        first5  = sum(losses[:5]) / 5
        last5   = sum(losses[-5:]) / 5
        self.assertLess(last5, first5)

    def test_cls_pool(self):
        trainer = self._make_trainer(pool="cls")
        loss = trainer.train_step(_batch())
        self.assertTrue(np.isfinite(loss))

    def test_no_attention_mask_in_batch(self):
        """Batch without attention_mask should default to all-ones."""
        from foundry.training import (
            EmbeddingDistillTrainer, EmbeddingDistillConfig, ToyEmbeddingTeacher,
        )
        student = TinyEncoder()
        teacher = ToyEmbeddingTeacher(dim=32)
        trainer = EmbeddingDistillTrainer(
            student, teacher, EmbeddingDistillConfig(device="cpu")
        )
        batch = {"input_ids": np.random.randint(0, 100, (2, 8))}
        loss  = trainer.train_step(batch)
        self.assertTrue(np.isfinite(loss))

    def test_on_step_callback(self):
        trainer = self._make_trainer()
        calls   = []
        trainer.cfg.log_every = 1
        trainer.train([_batch()], on_step=lambda s, l: calls.append((s, l)))
        self.assertEqual(len(calls), 1)

    def test_device_is_cpu(self):
        trainer = self._make_trainer()
        self.assertEqual(str(trainer.device), "cpu")

    def test_multi_epoch(self):
        trainer = self._make_trainer(epochs=3)
        n       = 4
        result  = trainer.train([_batch() for _ in range(n)])
        # 3 epochs × n batches
        self.assertEqual(len(result["losses"]), 3 * n)


# ── EmbedRecipe schema ─────────────────────────────────────────────────────

class TestEmbedRecipeSchema(unittest.TestCase):

    def _valid_yaml_dict(self):
        return {
            "seed":     {"init": "pretrained", "model": "org/student-200m"},
            "teachers": [{"role": "embed_teacher", "model": "org/bge-large", "weight": 1.0}],
            "fusion":   {"embed_loss": "cosine", "embed_pool": "mean"},
            "heal":     {"tokens": "1B", "alpha": 0.0},
        }

    def test_valid_embed_recipe(self):
        from foundry.recipes import EmbedRecipe
        r = EmbedRecipe.model_validate(self._valid_yaml_dict())
        self.assertEqual(r.fusion.embed_loss, "cosine")
        self.assertEqual(r.teachers[0].model, "org/bge-large")

    def test_requires_at_least_one_teacher(self):
        from foundry.recipes import EmbedRecipe
        from pydantic import ValidationError
        data = self._valid_yaml_dict()
        data["teachers"] = []
        with self.assertRaises(ValidationError):
            EmbedRecipe.model_validate(data)

    def test_warns_on_multiple_teachers(self):
        from foundry.recipes import EmbedRecipe
        data = self._valid_yaml_dict()
        data["teachers"] = [
            {"role": "a", "model": "org/a", "weight": 1.0},
            {"role": "b", "model": "org/b", "weight": 0.5},
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EmbedRecipe.model_validate(data)
            self.assertTrue(any("multiple" in str(x.message).lower() or
                                "teachers" in str(x.message).lower()
                                for x in w))

    def test_cosine_and_mse_loss_valid(self):
        from foundry.recipes import EmbedRecipe
        for loss in ("cosine", "mse"):
            data = self._valid_yaml_dict()
            data["fusion"]["embed_loss"] = loss
            r = EmbedRecipe.model_validate(data)
            self.assertEqual(r.fusion.embed_loss, loss)

    def test_invalid_loss_rejected(self):
        from foundry.recipes import EmbedRecipe
        from pydantic import ValidationError
        data = self._valid_yaml_dict()
        data["fusion"]["embed_loss"] = "kl_div"
        with self.assertRaises(ValidationError):
            EmbedRecipe.model_validate(data)


# ── FoundryRecipe schema hardening ────────────────────────────────────────

class TestFoundryRecipeHardening(unittest.TestCase):

    def _base(self):
        return {
            "seed":     {"init": "pretrained", "model": "org/llm"},
            "teachers": [{"role": "r", "model": "org/t", "weight": 1.0}],
            "heal":     {"tokens": "10B", "alpha": 0.3},
        }

    def test_heal_without_teachers_raises(self):
        from foundry.recipes import FoundryRecipe
        from pydantic import ValidationError
        data = self._base()
        data["teachers"] = []
        with self.assertRaises(ValidationError):
            FoundryRecipe.model_validate(data)

    def test_grow_with_from_scratch_raises(self):
        from foundry.recipes import FoundryRecipe
        from pydantic import ValidationError
        data = {
            "seed": {"init": "from_scratch", "arch": "my_mod:MyLM"},
            "grow": {"method": "depth_upscale", "to_params": "15B"},
            "teachers": [{"role": "r", "model": "org/t", "weight": 1.0}],
        }
        with self.assertRaises(ValidationError):
            FoundryRecipe.model_validate(data)

    def test_valid_recipe_passes(self):
        from foundry.recipes import FoundryRecipe
        r = FoundryRecipe.model_validate(self._base())
        self.assertEqual(r.seed.model, "org/llm")


# ── detect_backend ─────────────────────────────────────────────────────────

class TestDetectBackendM5(unittest.TestCase):

    def test_new_keys_present(self):
        from foundry.backends import detect_backend
        info = detect_backend()
        for key in ("safetensors", "rapidfuzz", "python_version",
                    "torch_version", "gpu_count", "gpu_vram_gb"):
            self.assertIn(key, info, f"Missing key: {key}")

    def test_python_version_format(self):
        from foundry.backends import detect_backend
        info = detect_backend()
        parts = info["python_version"].split(".")
        self.assertEqual(len(parts), 3)

    def test_safetensors_true(self):
        from foundry.backends import detect_backend
        info = detect_backend()
        self.assertTrue(info["safetensors"], "safetensors should be installed")

    def test_gpu_vram_is_list(self):
        from foundry.backends import detect_backend
        info = detect_backend()
        self.assertIsInstance(info["gpu_vram_gb"], list)


# ── Top-level foundry exports ──────────────────────────────────────────────

class TestFoundryM5Exports(unittest.TestCase):

    def test_m5_symbols_exported(self):
        import foundry
        for sym in (
            "EmbeddingDistillTrainer", "EmbeddingDistillConfig",
            "ToyEmbeddingTeacher", "EmbedRecipe", "EmbedFusionConfig",
        ):
            self.assertIn(sym, foundry.__all__, f"Missing from __all__: {sym}")
            self.assertTrue(hasattr(foundry, sym), f"Not importable: {sym}")


if __name__ == "__main__":
    unittest.main()
