"""
ContrastiveTrainer tests (torch). Tiny in-process encoder + fake tokenizer.
"""
from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class TinyEncoder(nn.Module):
    """Returns .last_hidden_state of shape (B, S, dim)."""
    def __init__(self, vocab=64, dim=24):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj  = nn.Linear(dim, dim)
        self.config = SimpleNamespace(hidden_size=dim, vocab_size=vocab)
    def forward(self, input_ids, attention_mask=None, **_):
        return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))


class FakeTokenizer:
    """Hashes each string into a short fixed token sequence (no real vocab needed)."""
    def __init__(self, vocab=64, seq=8):
        self.vocab = vocab
        self.seq = seq
    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        ids = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            ids.append(rng.integers(1, self.vocab, self.seq).tolist())
        ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _pairs(n=20):
    # anchor i and positive i share a token signature (same text stem) → learnable
    return [{"anchor": f"anchor sentence {i}", "positive": f"positive sentence {i}"}
            for i in range(n)]


class TestContrastiveTrainer(unittest.TestCase):

    def _trainer(self, **kw):
        from foundry.training import ContrastiveTrainer, ContrastiveConfig
        cfg = ContrastiveConfig(device="cpu", epochs=1, batch_size=8, log_every=1, **kw)
        return ContrastiveTrainer(TinyEncoder(), FakeTokenizer(), config=cfg)

    def test_losses_finite(self):
        trainer = self._trainer()
        result  = trainer.train(_pairs(24), shuffle=False)
        self.assertGreater(len(result["losses"]), 0)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_loss_decreases(self):
        from foundry.training import ContrastiveTrainer, ContrastiveConfig
        # repeat a fixed set of distinguishable pairs so the model can separate them
        torch.manual_seed(0)
        pairs = [{"anchor": f"q{i}", "positive": f"p{i}"} for i in range(8)]
        cfg = ContrastiveConfig(device="cpu", epochs=1, batch_size=8, learning_rate=1e-2)
        trainer = ContrastiveTrainer(TinyEncoder(64, 24), FakeTokenizer(), config=cfg)
        result = trainer.train(pairs * 30, shuffle=False)
        losses = result["losses"]
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)

    def test_with_hard_negatives(self):
        trainer = self._trainer()
        pairs = [{"anchor": f"a{i}", "positive": f"p{i}", "negative": f"n{i}"} for i in range(16)]
        result = trainer.train(pairs, shuffle=False)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_requires_tokenizer(self):
        from foundry.training import ContrastiveTrainer, ContrastiveConfig
        with self.assertRaises(ValueError):
            ContrastiveTrainer(TinyEncoder(), None, config=ContrastiveConfig(device="cpu"))

    def test_save_checkpoint(self):
        trainer = self._trainer()
        trainer.train(_pairs(16), shuffle=False)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(trainer.save_checkpoint(tmp).exists())

    def test_encode_shape(self):
        trainer = self._trainer()
        emb = trainer.encode(["hello", "world", "foo"])
        self.assertEqual(emb.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
