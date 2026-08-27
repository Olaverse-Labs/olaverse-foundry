"""
Quantized model loading — foundry.io.loader.

A 27B teacher is ~54GB in bf16 and ~15GB in 4-bit. Since a teacher is only ever
run forward, quantizing it is nearly free in quality terms and is what decides
whether a large-teacher distillation fits on one machine.

These tests cover config construction and the kwargs the loader builds. They do
not load a quantized model — that needs bitsandbytes and a CUDA device.
"""
from __future__ import annotations

import unittest

import pytest

pytest.importorskip("transformers")

from foundry.io.loader import ModelRef, build_quantization_config, resolve_dtype


class TestBuildQuantizationConfig(unittest.TestCase):

    def test_none_returns_none(self):
        self.assertIsNone(build_quantization_config(None))

    def test_4bit_uses_nf4_and_double_quant(self):
        cfg = build_quantization_config("4bit", "bfloat16")
        self.assertTrue(cfg.load_in_4bit)
        self.assertEqual(cfg.bnb_4bit_quant_type, "nf4")
        self.assertTrue(cfg.bnb_4bit_use_double_quant)

    def test_4bit_compute_dtype_follows_argument(self):
        import torch
        self.assertEqual(
            build_quantization_config("4bit", "float16").bnb_4bit_compute_dtype,
            torch.float16,
        )

    def test_8bit(self):
        cfg = build_quantization_config("8bit")
        self.assertTrue(cfg.load_in_8bit)
        self.assertFalse(getattr(cfg, "load_in_4bit", False))

    def test_rejects_anything_else(self):
        for bad in ("3bit", "int8", "yes", ""):
            with self.assertRaises(ValueError, msg=bad):
                build_quantization_config(bad)


class TestResolveDtype(unittest.TestCase):

    def test_known_names(self):
        import torch
        self.assertEqual(resolve_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(resolve_dtype("float16"),  torch.float16)
        self.assertEqual(resolve_dtype("float32"),  torch.float32)

    def test_unknown_falls_back_to_default(self):
        import torch
        self.assertEqual(resolve_dtype("auto"), torch.bfloat16)
        self.assertEqual(resolve_dtype("auto", default="float32"), torch.float32)


class TestModelRefCarriesQuantize(unittest.TestCase):

    def test_defaults_to_none(self):
        self.assertIsNone(ModelRef.parse("org/model").quantize)

    def test_parse_accepts_quantize(self):
        self.assertEqual(ModelRef.parse("org/model", quantize="4bit").quantize, "4bit")

    def test_quantize_survives_revision_parsing(self):
        ref = ModelRef.parse("org/model@abc123", quantize="8bit")
        self.assertEqual(ref.revision, "abc123")
        self.assertEqual(ref.quantize, "8bit")


class TestLoaderKwargs(unittest.TestCase):
    """torch_dtype and quantization_config must never both be sent."""

    def _kwargs_for(self, **ref_kwargs):
        captured = {}

        class FakeAuto:
            @staticmethod
            def from_pretrained(src, **kwargs):
                captured.update(kwargs)
                return object()

        from foundry.io.loader import load_model
        load_model(ModelRef.parse("org/model", **ref_kwargs), model_class=FakeAuto)
        return captured

    def test_unquantized_passes_torch_dtype(self):
        kwargs = self._kwargs_for()
        self.assertIn("torch_dtype", kwargs)
        self.assertNotIn("quantization_config", kwargs)

    def test_quantized_passes_config_and_not_dtype(self):
        kwargs = self._kwargs_for(quantize="4bit")
        self.assertIn("quantization_config", kwargs)
        self.assertNotIn("torch_dtype", kwargs)

    def test_device_map_always_forwarded(self):
        self.assertEqual(self._kwargs_for(quantize="4bit")["device_map"], "auto")


class TestTeacherPlacement(unittest.TestCase):
    """A placed model must not be moved with .to() — bitsandbytes refuses, and
    accelerate's device_map hooks are invalidated by it."""

    def setUp(self):
        from foundry.teachers.registry import HFTeacher
        self.teacher = HFTeacher("org/model")

    def _placed(self, **attrs):
        class FakeModel:
            pass
        model = FakeModel()
        for k, v in attrs.items():
            setattr(model, k, v)
        self.teacher._model = model
        return self.teacher._is_placed()

    def test_plain_model_is_not_placed(self):
        self.assertFalse(self._placed())

    def test_device_map_counts_as_placed(self):
        self.assertTrue(self._placed(hf_device_map={"": 0}))

    def test_empty_device_map_is_not_placed(self):
        self.assertFalse(self._placed(hf_device_map={}))

    def test_quantized_counts_as_placed(self):
        self.assertTrue(self._placed(is_quantized=True))
        self.assertTrue(self._placed(is_loaded_in_4bit=True))
        self.assertTrue(self._placed(is_loaded_in_8bit=True))

    def test_teacher_carries_quantize_into_its_ref(self):
        from foundry.teachers.registry import HFTeacher
        self.assertEqual(HFTeacher("org/m", quantize="4bit").quantize, "4bit")


if __name__ == "__main__":
    unittest.main()
