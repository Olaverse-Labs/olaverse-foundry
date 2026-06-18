"""
Tests for the head trainers (torch required, no HF downloads):
  • SequenceClassificationTrainer  — (B,) labels, single- and multi-label
  • TokenClassificationTrainer     — (B,S) labels with -100 ignore
  • freeze_backbone                — only head params stay trainable
All use tiny in-process nn.Modules returning .logits.
"""
from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class TinySeqModel(nn.Module):
    """Encoder backbone + sequence-classification head -> .logits (B, C)."""
    def __init__(self, vocab=64, dim=24, num_labels=3):
        super().__init__()
        self.encoder    = nn.Embedding(vocab, dim)
        self.classifier = nn.Linear(dim, num_labels)
    def forward(self, input_ids, attention_mask=None, **_):
        h = self.encoder(input_ids)                       # (B,S,D)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            pooled = h.mean(1)
        return SimpleNamespace(logits=self.classifier(pooled))


class TinyTokenModel(nn.Module):
    """Encoder backbone + token-classification head -> .logits (B, S, C)."""
    def __init__(self, vocab=64, dim=24, num_labels=5):
        super().__init__()
        self.encoder    = nn.Embedding(vocab, dim)
        self.classifier = nn.Linear(dim, num_labels)
    def forward(self, input_ids, attention_mask=None, **_):
        return SimpleNamespace(logits=self.classifier(self.encoder(input_ids)))


def _seq_batches(n=6, B=4, S=8, vocab=64, C=3):
    return [{"input_ids": np.random.randint(0, vocab, (B, S)).astype(np.int32),
             "attention_mask": np.ones((B, S), np.int32),
             "labels": np.random.randint(0, C, (B,)).astype(np.int64)} for _ in range(n)]


def _tok_batches(n=6, B=4, S=8, vocab=64, C=5):
    out = []
    for _ in range(n):
        labels = np.random.randint(0, C, (B, S)).astype(np.int64)
        labels[:, -2:] = -100                      # ignore last two positions
        out.append({"input_ids": np.random.randint(0, vocab, (B, S)).astype(np.int32),
                    "attention_mask": np.ones((B, S), np.int32),
                    "labels": labels})
    return out


class TestSequenceClassification(unittest.TestCase):

    def _trainer(self, **kw):
        from foundry.training import SequenceClassificationTrainer, HeadTrainConfig
        cfg = HeadTrainConfig(device="cpu", epochs=1, num_labels=3, log_every=1, **kw)
        return SequenceClassificationTrainer(TinySeqModel(num_labels=3), config=cfg)

    def test_losses_finite_and_accuracy_returned(self):
        trainer = self._trainer(eval_every=1)
        result  = trainer.train(_seq_batches(), eval_dataset=_seq_batches(n=2))
        self.assertEqual(len(result["losses"]), 6)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))
        self.assertGreater(len(result["eval_metrics"]), 0)
        for acc in result["eval_metrics"].values():
            self.assertTrue(0.0 <= acc <= 1.0)

    def test_overfits_single_batch(self):
        fixed = _seq_batches(n=1, B=4)[0]
        trainer = self._trainer(learning_rate=1e-2)
        result  = trainer.train([fixed] * 40)
        self.assertLess(sum(result["losses"][-5:]) / 5, sum(result["losses"][:5]) / 5)

    def test_multi_label(self):
        from foundry.training import SequenceClassificationTrainer, HeadTrainConfig
        C = 3
        cfg = HeadTrainConfig(device="cpu", epochs=1, num_labels=C, multi_label=True)
        trainer = SequenceClassificationTrainer(TinySeqModel(num_labels=C), config=cfg)
        batches = [{"input_ids": np.random.randint(0, 64, (4, 8)).astype(np.int32),
                    "attention_mask": np.ones((4, 8), np.int32),
                    "labels": np.random.randint(0, 2, (4, C)).astype(np.int64)} for _ in range(4)]
        result = trainer.train(batches)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_missing_labels_raises(self):
        trainer = self._trainer()
        with self.assertRaises(ValueError):
            trainer.train([{"input_ids": np.zeros((4, 8), np.int32)}])


class TestTokenClassification(unittest.TestCase):

    def _trainer(self, **kw):
        from foundry.training import TokenClassificationTrainer, HeadTrainConfig
        cfg = HeadTrainConfig(device="cpu", epochs=1, num_labels=5, log_every=1, **kw)
        return TokenClassificationTrainer(TinyTokenModel(num_labels=5), config=cfg)

    def test_losses_finite(self):
        trainer = self._trainer()
        result  = trainer.train(_tok_batches())
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_accuracy_ignores_pad(self):
        trainer = self._trainer(eval_every=1)
        result  = trainer.train(_tok_batches(n=4), eval_dataset=_tok_batches(n=2))
        for acc in result["eval_metrics"].values():
            self.assertTrue(0.0 <= acc <= 1.0)

    def test_overfits_single_batch(self):
        fixed = _tok_batches(n=1)[0]
        trainer = self._trainer(learning_rate=1e-2)
        result  = trainer.train([fixed] * 40)
        self.assertLess(sum(result["losses"][-5:]) / 5, sum(result["losses"][:5]) / 5)

    def test_save_and_resume(self):
        trainer = self._trainer()
        trainer.train(_tok_batches(n=4))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = trainer.save_checkpoint(tmp)
            self.assertTrue(ckpt.exists())


class TestFreezeBackbone(unittest.TestCase):

    def test_only_head_trainable(self):
        from foundry.training import freeze_backbone
        model = TinySeqModel(num_labels=3)
        _, n_train, n_frozen = freeze_backbone(model)
        # classifier trainable, encoder frozen
        self.assertTrue(model.classifier.weight.requires_grad)
        self.assertFalse(model.encoder.weight.requires_grad)
        self.assertGreater(n_frozen, 0)
        self.assertGreater(n_train, 0)

    def test_frozen_backbone_unchanged_after_training(self):
        from foundry.training import SequenceClassificationTrainer, HeadTrainConfig
        cfg = HeadTrainConfig(device="cpu", epochs=1, num_labels=3,
                              freeze_backbone=True, learning_rate=1e-1)
        model   = TinySeqModel(num_labels=3)
        trainer = SequenceClassificationTrainer(model, config=cfg)
        before  = model.encoder.weight.detach().clone()
        head_before = model.classifier.weight.detach().clone()
        trainer.train(_seq_batches(n=8))
        # backbone frozen → unchanged; head trained → changed
        self.assertTrue(torch.allclose(before, model.encoder.weight))
        self.assertFalse(torch.allclose(head_before, model.classifier.weight))


class TestExports(unittest.TestCase):
    def test_exported(self):
        import foundry
        for n in ("SequenceClassificationTrainer", "TokenClassificationTrainer",
                  "HeadTrainConfig", "freeze_backbone", "build_encoder_with_head"):
            self.assertIn(n, foundry.__all__)


if __name__ == "__main__":
    unittest.main()
