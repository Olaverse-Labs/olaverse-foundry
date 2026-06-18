"""
Tests for the base-encoder training paths:
  • MLMTrainer            — masked-LM pretraining from scratch (teacherless)
  • EncoderDistillTrainer — token-level hidden-state distillation from a teacher
  • WithMLMHead           — MLM-head wrapper for custom encoders

All tests use tiny in-process nn.Modules — no HF downloads, no GPU.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# ── Tiny encoders ───────────────────────────────────────────────────────────

class TinyEncoder(nn.Module):
    """Returns .last_hidden_state of shape (B, S, dim)."""
    def __init__(self, vocab=64, dim=24):
        super().__init__()
        self.embeddings = nn.Embedding(vocab, dim)
        self.proj       = nn.Linear(dim, dim)
    def get_input_embeddings(self):
        return self.embeddings
    def forward(self, input_ids, attention_mask=None, **_):
        return SimpleNamespace(last_hidden_state=self.proj(self.embeddings(input_ids)))


class TinyMaskedLM(nn.Module):
    """Encoder + MLM head — returns .logits of shape (B, S, vocab)."""
    def __init__(self, vocab=64, dim=24):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head  = nn.Linear(dim, vocab)
    def forward(self, input_ids, attention_mask=None, **_):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


class FakeTokenizer:
    """Minimal tokenizer exposing what MLMTrainer needs."""
    def __init__(self, vocab=64):
        self._vocab        = vocab
        self.mask_token_id = 3
        self.pad_token_id  = 0
        self.all_special_ids = [0, 1, 2, 3]
    def __len__(self):
        return self._vocab


def _embed_batches(n=6, B=2, S=10, vocab=64):
    return [
        {
            "input_ids":      np.random.randint(4, vocab, (B, S)).astype(np.int32),
            "attention_mask": np.ones((B, S), dtype=np.int32),
        }
        for _ in range(n)
    ]


# ── MLM pretraining ──────────────────────────────────────────────────────────

class TestMLMTrainer(unittest.TestCase):

    def _trainer(self, **cfg_kw):
        from foundry.training import MLMTrainer, MLMConfig
        vocab = 64
        cfg = MLMConfig(device="cpu", epochs=1, log_every=1, **cfg_kw)
        return MLMTrainer(TinyMaskedLM(vocab), tokenizer=FakeTokenizer(vocab), config=cfg)

    def test_losses_finite(self):
        trainer = self._trainer()
        result  = trainer.train(_embed_batches())
        self.assertEqual(len(result["losses"]), 6)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_loss_decreases_on_repeated_batch(self):
        fixed = {
            "input_ids":      np.random.randint(4, 64, (2, 10)).astype(np.int32),
            "attention_mask": np.ones((2, 10), dtype=np.int32),
        }
        trainer = self._trainer(learning_rate=5e-3)
        result  = trainer.train([fixed] * 40)
        losses  = result["losses"]
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)

    def test_requires_mask_token(self):
        from foundry.training import MLMTrainer, MLMConfig
        with self.assertRaises(ValueError):
            MLMTrainer(TinyMaskedLM(), tokenizer=None, config=MLMConfig(vocab_size=64))

    def test_config_only_no_tokenizer(self):
        from foundry.training import MLMTrainer, MLMConfig
        cfg = MLMConfig(device="cpu", epochs=1, mask_token_id=3, vocab_size=64, pad_token_id=0)
        trainer = MLMTrainer(TinyMaskedLM(64), tokenizer=None, config=cfg)
        result  = trainer.train(_embed_batches())
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_eval_loop(self):
        trainer = self._trainer(eval_every=1)
        result  = trainer.train(_embed_batches(n=4), eval_dataset=_embed_batches(n=2))
        self.assertGreater(len(result["eval_losses"]), 0)
        for ev in result["eval_losses"].values():
            self.assertTrue(np.isfinite(ev))

    def test_save_and_resume(self):
        trainer = self._trainer()
        trainer.train(_embed_batches(n=4))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = trainer.save_checkpoint(tmp)
            self.assertTrue(ckpt.exists())
            trainer2 = self._trainer()
            before   = next(trainer2.student.parameters()).clone()
            trainer2.resume_from_checkpoint(tmp)
            after    = next(trainer2.student.parameters()).clone()
            self.assertFalse(torch.allclose(before, after))

    def test_with_mlm_head_wrapper(self):
        """A custom encoder (last_hidden_state only) becomes MLM-trainable via WithMLMHead."""
        from foundry.training import MLMTrainer, MLMConfig, WithMLMHead
        vocab, dim = 64, 24
        student = WithMLMHead(TinyEncoder(vocab, dim), hidden_size=dim, vocab_size=vocab)
        cfg     = MLMConfig(device="cpu", epochs=1, mask_token_id=3, vocab_size=vocab)
        trainer = MLMTrainer(student, tokenizer=FakeTokenizer(vocab), config=cfg)
        result  = trainer.train(_embed_batches())
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))


# ── MLM numerical stability (regression: empty-mask → NaN) ─────────────────────

class TestMLMNoNaN(unittest.TestCase):
    """Tiny batch×seq makes zero-mask batches likely; previously a zero-mask
    batch produced a NaN CrossEntropy that poisoned the whole run."""

    def _trainer(self, **kw):
        from foundry.training import MLMTrainer, MLMConfig
        vocab = 64
        cfg = MLMConfig(device="cpu", epochs=1, log_every=1, **kw)
        return MLMTrainer(TinyMaskedLM(vocab), tokenizer=FakeTokenizer(vocab), config=cfg)

    def test_mask_always_marks_at_least_one(self):
        import torch
        trainer = self._trainer()
        ids  = torch.randint(4, 64, (2, 6))
        attn = torch.ones(2, 6, dtype=torch.long)
        for _ in range(200):                       # would hit a zero-mask draw without the guard
            _inp, labels = trainer._mask_tokens(ids, attn)
            self.assertGreater(int((labels != -100).sum()), 0)

    def test_tiny_short_batches_never_nan(self):
        # Short sequences across many steps: the exact condition that used to NaN.
        trainer = self._trainer(learning_rate=3e-4)
        data    = _embed_batches(n=60, B=2, S=6)
        result  = trainer.train(data)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]),
                        "MLM produced a non-finite loss on tiny batches")

    def test_repeated_short_batch_stays_finite_all_lrs(self):
        for lr in (3e-4, 1e-3, 5e-3):
            trainer = self._trainer(learning_rate=lr)
            fixed   = _embed_batches(n=1, B=2, S=6)[0]
            result  = trainer.train([fixed] * 40)
            self.assertTrue(all(np.isfinite(l) for l in result["losses"]),
                            f"non-finite MLM loss at lr={lr}")


# ── Token-level encoder distillation ──────────────────────────────────────────

class TestEncoderDistillTrainer(unittest.TestCase):

    def test_same_dim_losses_finite(self):
        from foundry.training import EncoderDistillTrainer, EncoderDistillConfig
        student = TinyEncoder(64, 24)
        teacher = TinyEncoder(64, 24)
        cfg     = EncoderDistillConfig(device="cpu", epochs=1, log_every=1)
        trainer = EncoderDistillTrainer(student, teacher, config=cfg)
        result  = trainer.train(_embed_batches())
        self.assertEqual(len(result["losses"]), 6)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_projection_when_dims_differ(self):
        from foundry.training import EncoderDistillTrainer, EncoderDistillConfig
        student = TinyEncoder(64, 16)     # student dim 16
        teacher = TinyEncoder(64, 32)     # teacher dim 32 -> projector added
        cfg     = EncoderDistillConfig(device="cpu", epochs=1)
        trainer = EncoderDistillTrainer(student, teacher, config=cfg)
        result  = trainer.train(_embed_batches())
        self.assertIsNotNone(trainer._projector)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_loss_decreases_cosine(self):
        from foundry.training import EncoderDistillTrainer, EncoderDistillConfig
        # Teacher is fixed (eval); student should move toward it on a repeated batch.
        student = TinyEncoder(64, 24)
        teacher = TinyEncoder(64, 24)
        cfg     = EncoderDistillConfig(device="cpu", epochs=1, loss="cosine", learning_rate=5e-3)
        trainer = EncoderDistillTrainer(student, teacher, config=cfg)
        fixed   = _embed_batches(n=1)[0]
        result  = trainer.train([fixed] * 40)
        losses  = result["losses"]
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)

    def test_save_and_resume(self):
        from foundry.training import EncoderDistillTrainer, EncoderDistillConfig
        cfg     = EncoderDistillConfig(device="cpu", epochs=1)
        trainer = EncoderDistillTrainer(TinyEncoder(64, 24), TinyEncoder(64, 24), config=cfg)
        trainer.train(_embed_batches(n=4))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = trainer.save_checkpoint(tmp)
            self.assertTrue(ckpt.exists())

    def test_plugs_into_datapipeline(self):
        """End-to-end: DataPipeline(embed) -> EncoderDistillTrainer."""
        from foundry.data import DataPipeline
        from foundry.training import EncoderDistillTrainer, EncoderDistillConfig
        rows = [{"input_ids": np.random.randint(4, 64, 10).tolist(),
                 "attention_mask": [1] * 10} for _ in range(8)]
        pipe = DataPipeline(rows, batch_size=4, max_length=10, mode="embed")
        cfg  = EncoderDistillConfig(device="cpu", epochs=1)
        trainer = EncoderDistillTrainer(TinyEncoder(64, 24), TinyEncoder(64, 24), config=cfg)
        result  = trainer.train(pipe)
        self.assertGreater(len(result["losses"]), 0)


# ── Exports ────────────────────────────────────────────────────────────────

class TestExports(unittest.TestCase):
    def test_exported_from_foundry(self):
        import foundry
        for name in ("MLMTrainer", "MLMConfig", "WithMLMHead",
                     "EncoderDistillTrainer", "EncoderDistillConfig"):
            self.assertIn(name, foundry.__all__)


if __name__ == "__main__":
    unittest.main()
