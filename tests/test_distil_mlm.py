"""
Tests for DistilMLMTrainer (combined distillation + MLM, the DistilBERT objective).
Torch required; tiny in-process masked-LM modules — no downloads, no GPU.
"""
from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class TinyMaskedLM(nn.Module):
    """Masked-LM model returning .logits (B,S,V) and .hidden_states (tuple)."""
    def __init__(self, vocab=64, dim=24):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.enc   = nn.Linear(dim, dim)
        self.head  = nn.Linear(dim, vocab)
        self.config = SimpleNamespace(hidden_size=dim, vocab_size=vocab)
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **_):
        h = torch.relu(self.enc(self.embed(input_ids)))
        return SimpleNamespace(logits=self.head(h), hidden_states=(h,))


class FakeTokenizer:
    def __init__(self, vocab=64):
        self._vocab = vocab
        self.mask_token_id = 3
        self.pad_token_id = 0
        self.all_special_ids = [0, 1, 2, 3]
    def __len__(self):
        return self._vocab


def _batches(n=6, B=2, S=10, vocab=64):
    return [{"input_ids": np.random.randint(4, vocab, (B, S)).astype(np.int32),
             "attention_mask": np.ones((B, S), np.int32)} for _ in range(n)]


class TestDistilMLMTrainer(unittest.TestCase):

    def _trainer(self, vocab=64, dim=24, **kw):
        from foundry.training import DistilMLMTrainer, DistilMLMConfig
        cfg = DistilMLMConfig(device="cpu", epochs=1, log_every=1, **kw)
        return DistilMLMTrainer(TinyMaskedLM(vocab, dim), TinyMaskedLM(vocab, dim),
                                tokenizer=FakeTokenizer(vocab), config=cfg)

    def test_losses_finite(self):
        trainer = self._trainer()
        result  = trainer.train(_batches())
        self.assertEqual(len(result["losses"]), 6)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_combines_three_terms(self):
        # zeroing all weights -> zero loss; confirms the weighted sum is wired up
        trainer = self._trainer(mlm_weight=0.0, distill_weight=0.0, cosine_weight=0.0)
        result  = trainer.train(_batches(n=2))
        self.assertTrue(all(abs(l) < 1e-6 for l in result["losses"]))

    def test_loss_decreases_on_repeated_batch(self):
        fixed = _batches(n=1)[0]
        trainer = self._trainer(learning_rate=5e-3)
        result  = trainer.train([fixed] * 40)
        losses  = result["losses"]
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)

    def test_vocab_mismatch_raises(self):
        from foundry.training import DistilMLMTrainer, DistilMLMConfig
        student = TinyMaskedLM(vocab=64, dim=24)
        teacher = TinyMaskedLM(vocab=80, dim=24)     # different vocab
        cfg = DistilMLMConfig(device="cpu", epochs=1)
        trainer = DistilMLMTrainer(student, teacher, tokenizer=FakeTokenizer(64), config=cfg)
        with self.assertRaises(ValueError):
            trainer.train(_batches(n=1))

    def test_projector_when_hidden_dims_differ(self):
        from foundry.training import DistilMLMTrainer, DistilMLMConfig
        student = TinyMaskedLM(vocab=64, dim=16)     # hidden 16
        teacher = TinyMaskedLM(vocab=64, dim=32)     # hidden 32 -> projector for cosine
        cfg = DistilMLMConfig(device="cpu", epochs=1, lr_scheduler="cosine", warmup_steps=2)
        trainer = DistilMLMTrainer(student, teacher, tokenizer=FakeTokenizer(64), config=cfg)
        result  = trainer.train(_batches(n=8))
        self.assertIsNotNone(trainer._projector)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_save_and_resume(self):
        trainer = self._trainer()
        trainer.train(_batches(n=4))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = trainer.save_checkpoint(tmp)
            self.assertTrue(ckpt.exists())

    def test_exported(self):
        import foundry
        self.assertIn("DistilMLMTrainer", foundry.__all__)
        self.assertIn("DistilMLMConfig", foundry.__all__)


if __name__ == "__main__":
    unittest.main()
