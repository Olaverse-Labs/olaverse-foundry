"""
LoRA training path — foundry.training.lora.

Uses a real GPT2 built from a local config (no download, ~32k params). The point
of these tests is the integration: that a LoRA-wrapped student still forwards
like the model it wraps, that the optimizer owns only the adapter, and that the
run produces a plain HuggingFace directory at the end.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import numpy as np
import torch

from foundry.teachers import TeacherRegistry, ToyTeacher
from foundry.training.lora import (
    LoRAConfig,
    attach_lora,
    merge_and_save,
    save_adapter,
    to_skillpack,
    trainable_summary,
)

VOCAB = 128


def tiny_causal_lm(seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=VOCAB, n_positions=64, n_embd=32, n_layer=2, n_head=2,
        bos_token_id=0, eos_token_id=1,
    ))


def lora_lm(rank: int = 4):
    return attach_lora(tiny_causal_lm(),
                       LoRAConfig(rank=rank, alpha=rank * 2, target_modules=["c_attn"]))


class TestAttachLora(unittest.TestCase):

    def test_only_a_small_fraction_trains(self):
        summary = trainable_summary(lora_lm())
        self.assertGreater(summary["trainable"], 0)
        self.assertLess(summary["percent"], 25.0)

    def test_base_weights_are_frozen(self):
        model = lora_lm()
        frozen = [n for n, p in model.named_parameters()
                  if "lora_" not in n and p.requires_grad]
        self.assertEqual(frozen, [], f"base weights left trainable: {frozen[:3]}")

    def test_still_forwards_like_a_causal_lm(self):
        out = lora_lm()(input_ids=torch.randint(0, VOCAB, (2, 8)))
        self.assertTrue(hasattr(out, "logits"))
        self.assertEqual(out.logits.shape, (2, 8, VOCAB))

    def test_rejects_unknown_task_type(self):
        with self.assertRaises(ValueError):
            attach_lora(tiny_causal_lm(), LoRAConfig(task_type="NOT_A_TASK"))

    def test_rank_is_honoured(self):
        model = lora_lm(rank=8)
        ranks = {p.shape[0] for n, p in model.named_parameters() if "lora_A" in n}
        self.assertEqual(ranks, {8})


class TestTrainableSummary(unittest.TestCase):

    def test_plain_model_is_fully_trainable(self):
        self.assertAlmostEqual(trainable_summary(tiny_causal_lm())["percent"], 100.0, places=3)

    def test_totals_are_consistent(self):
        s = trainable_summary(lora_lm())
        self.assertLessEqual(s["trainable"], s["total"])
        self.assertAlmostEqual(s["percent"], 100.0 * s["trainable"] / s["total"], places=6)


class TestTrainsThroughFoundryTrainer(unittest.TestCase):
    """A LoRA student must drop into the existing trainers unchanged."""

    def _train(self, model, epochs=2):
        from foundry.training.torch_distill import TorchDistillTrainer, TorchTrainConfig
        registry = TeacherRegistry([ToyTeacher(name="toy", vocab_size=VOCAB)])
        dataset = [np.random.randint(0, VOCAB, (2, 8)) for _ in range(4)]
        cfg = TorchTrainConfig(device="cpu", epochs=epochs, seed=0)
        trainer = TorchDistillTrainer(model, registry, config=cfg)
        return trainer, trainer.train(dataset)

    def test_optimizer_owns_only_the_adapter(self):
        """The whole point: without this LoRA saves nothing on optimizer state."""
        model = lora_lm()
        trainer, _ = self._train(model, epochs=1)
        owned = sum(len(g["params"]) for g in trainer._optimizer.param_groups)
        lora_tensors = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
        self.assertEqual(owned, lora_tensors)
        self.assertLess(owned, len(list(model.parameters())))

    def test_loss_is_finite(self):
        _, out = self._train(lora_lm())
        self.assertTrue(all(np.isfinite(x) for x in out["losses"]))

    def test_base_weights_do_not_move(self):
        model = lora_lm()
        before = {n: p.detach().clone()
                  for n, p in model.named_parameters() if "lora_" not in n}
        self._train(model)
        for name, original in before.items():
            current = dict(model.named_parameters())[name]
            torch.testing.assert_close(current.detach(), original,
                                       msg=f"frozen weight moved: {name}")

    def test_adapter_weights_do_move(self):
        model = lora_lm()
        before = {n: p.detach().clone()
                  for n, p in model.named_parameters() if "lora_B" in n}
        self._train(model, epochs=3)
        moved = any(
            not torch.allclose(dict(model.named_parameters())[n].detach(), original)
            for n, original in before.items()
        )
        self.assertTrue(moved, "adapter did not train")


class TestOutputArtifacts(unittest.TestCase):

    def test_save_adapter_writes_peft_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_adapter(lora_lm(), Path(tmp) / "adapter")
            names = {p.name for p in out.iterdir()}
            self.assertIn("adapter_config.json", names)
            self.assertTrue({"adapter_model.safetensors", "adapter_model.bin"} & names)

    def test_merge_produces_a_plain_hf_directory(self):
        from transformers import AutoModelForCausalLM
        with tempfile.TemporaryDirectory() as tmp:
            model = lora_lm()
            base_params = sum(p.numel() for p in tiny_causal_lm().parameters())
            out = merge_and_save(model, Path(tmp) / "merged")

            reloaded = AutoModelForCausalLM.from_pretrained(str(out))
            self.assertEqual(sum(p.numel() for p in reloaded.parameters()), base_params,
                             "merged model should have no adapter left in it")

    def test_merge_rejects_a_non_peft_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                merge_and_save(tiny_causal_lm(), Path(tmp) / "x")

    def test_to_skillpack_round_trips_shapes(self):
        model = lora_lm(rank=4)
        pack = to_skillpack(model, name="probe", base_hash="deadbeef")
        self.assertEqual(pack.name, "probe")
        self.assertEqual(pack.rank, 4)
        self.assertGreater(len(pack.weights), 0)
        for mats in pack.weights.values():
            self.assertEqual(mats["A"].shape[0], 4)
            self.assertEqual(mats["B"].shape[1], 4)

    def test_to_skillpack_rejects_a_plain_model(self):
        with self.assertRaises(ValueError):
            to_skillpack(tiny_causal_lm(), name="nope")


if __name__ == "__main__":
    unittest.main()
