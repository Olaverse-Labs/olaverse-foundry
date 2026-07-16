"""
QAT tests (torch required). Tiny in-process modules — no downloads, no GPU.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class TinySeqModel(nn.Module):
    def __init__(self, vocab=64, dim=24, num_labels=3):
        super().__init__()
        self.encoder    = nn.Embedding(vocab, dim)
        self.dense      = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, num_labels)
    def forward(self, input_ids, attention_mask=None, **_):
        h = torch.relu(self.dense(self.encoder(input_ids).mean(1)))
        return SimpleNamespace(logits=self.classifier(h))


class TestPrepareQAT(unittest.TestCase):

    def test_swaps_linear_for_qat(self):
        from foundry import prepare_qat
        model = prepare_qat(TinySeqModel())
        self.assertEqual(type(model.dense).__name__, "_QATLinear")
        self.assertEqual(type(model.classifier).__name__, "_QATLinear")
        # embeddings untouched
        self.assertIsInstance(model.encoder, nn.Embedding)

    def test_skip_keeps_layer_float(self):
        from foundry import prepare_qat
        model = prepare_qat(TinySeqModel(), skip=("classifier",))
        self.assertEqual(type(model.dense).__name__, "_QATLinear")
        self.assertEqual(type(model.classifier).__name__, "Linear")

    def test_forward_and_gradients_flow(self):
        from foundry import prepare_qat
        model = prepare_qat(TinySeqModel())
        ids   = torch.randint(0, 64, (4, 8))
        out   = model(input_ids=ids).logits
        self.assertEqual(out.shape, (4, 3))
        out.sum().backward()
        self.assertIsNotNone(model.dense.weight.grad)   # STE passes gradient through

    def test_int4_fewer_levels_than_int8(self):
        from foundry import quantize_tensor
        w = torch.randn(8, 16)
        q8, _ = quantize_tensor(w, bits=8)
        q4, _ = quantize_tensor(w, bits=4)
        self.assertLessEqual(q4.unique().numel(), 15)   # int4 symmetric: [-7,7]
        self.assertGreater(q8.unique().numel(), q4.unique().numel())

    def test_qat_trains_with_head_trainer(self):
        from foundry import prepare_qat
        from foundry.training import SequenceClassificationTrainer, HeadTrainConfig
        model = prepare_qat(TinySeqModel(num_labels=3))
        cfg   = HeadTrainConfig(device="cpu", epochs=1, num_labels=3, learning_rate=1e-2)
        trainer = SequenceClassificationTrainer(model, config=cfg)
        batches = [{"input_ids": np.random.randint(0, 64, (4, 8)).astype(np.int32),
                    "attention_mask": np.ones((4, 8), np.int32),
                    "labels": np.random.randint(0, 3, (4,)).astype(np.int64)} for _ in range(10)]
        result = trainer.train(batches)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))


class TestExport(unittest.TestCase):

    def test_int8_state_dict(self):
        from foundry import prepare_qat, int8_state_dict
        model = prepare_qat(TinySeqModel())
        sd = int8_state_dict(model)
        self.assertIn("dense.weight_int", sd)
        self.assertIn("dense.weight_scale", sd)
        self.assertEqual(sd["dense.weight_int"].dtype, torch.int8)

    def test_export_reports_compression(self):
        from foundry import prepare_qat, export_quantized
        model = prepare_qat(TinySeqModel())
        with tempfile.TemporaryDirectory() as tmp:
            rep8 = export_quantized(model, tmp, weight_bits=8, save_model=False)
            self.assertGreater(rep8["compression"], 1.5)   # int8 vs bf16 ~2x
            rep4 = export_quantized(model, tmp, weight_bits=4, save_model=False)
            self.assertGreater(rep4["compression"], rep8["compression"])

    def test_export_writes_metadata(self):
        from foundry import prepare_qat, export_quantized
        from pathlib import Path
        import json
        model = prepare_qat(TinySeqModel())
        with tempfile.TemporaryDirectory() as tmp:
            export_quantized(model, tmp, weight_bits=8, save_model=False)
            meta = json.loads((Path(tmp) / "quantization.json").read_text())
            self.assertEqual(meta["weight_bits"], 8)


class TestExports(unittest.TestCase):
    def test_exported(self):
        import foundry
        for n in ("prepare_qat", "QATConfig", "export_quantized",
                  "int8_state_dict", "quantize_tensor"):
            self.assertIn(n, foundry.__all__)


if __name__ == "__main__":
    unittest.main()
