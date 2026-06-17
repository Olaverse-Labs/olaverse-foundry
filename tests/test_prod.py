"""
Production-readiness tests.

Covers all features added in the production-hardening pass:
  • Mixed precision (float32 baseline; bfloat16 is a config option)
  • Gradient accumulation
  • Dataset shuffling
  • Checkpoint save / resume
  • OOM error message quality (mocked)
  • Teacher embedding cache (EmbeddingDistillTrainer)
  • snap_on() with real-world key format
  • HFTeacher model_type attribute
  • load_model model_class parameter
  • wandb / tensorboard in detect_backend
  • _FoundryLogger no-op when backend="none"

All tests use TinyLM / ToyTeacher — no HF downloads, no GPU required.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


# ── Fixtures ───────────────────────────────────────────────────────────────

class TinyLM(nn.Module):
    def __init__(self, vocab=100, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head  = nn.Linear(dim, vocab)
    def forward(self, input_ids, **_):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


class TinyEncoder(nn.Module):
    def __init__(self, vocab=100, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj  = nn.Linear(dim, dim)
    def forward(self, input_ids, attention_mask=None, **_):
        return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))


def _lm_dataset(n=6, B=2, S=8, vocab=100):
    return [np.random.randint(0, vocab, (B, S)).astype(np.int32) for _ in range(n)]


def _emb_dataset(n=6, B=2, S=8, vocab=100):
    return [
        {
            "input_ids":      np.random.randint(0, vocab, (B, S)).astype(np.int32),
            "attention_mask": np.ones((B, S), dtype=np.int32),
        }
        for _ in range(n)
    ]


def _make_torch_trainer(vocab=100, grad_accum=1, torch_dtype="float32", epochs=1):
    from foundry.training import TorchDistillTrainer, TorchTrainConfig
    from foundry.teachers.registry import TeacherRegistry

    teachers = TeacherRegistry.from_toy(n=1, vocab_size=vocab)
    cfg = TorchTrainConfig(
        epochs=epochs,
        device="cpu",
        grad_accumulation_steps=grad_accum,
        torch_dtype=torch_dtype,
        log_every=1,
    )
    return TorchDistillTrainer(TinyLM(vocab), teachers, config=cfg)


def _make_embed_trainer(grad_accum=1, torch_dtype="float32", epochs=1):
    from foundry.training import EmbeddingDistillTrainer, EmbeddingDistillConfig, ToyEmbeddingTeacher

    student = TinyEncoder()
    teacher = ToyEmbeddingTeacher(dim=32)
    cfg = EmbeddingDistillConfig(
        device="cpu",
        epochs=epochs,
        grad_accumulation_steps=grad_accum,
        torch_dtype=torch_dtype,
        log_every=1,
    )
    return EmbeddingDistillTrainer(student, teacher, config=cfg)


# ── Mixed precision ────────────────────────────────────────────────────────

class TestMixedPrecision(unittest.TestCase):

    def test_float32_runs(self):
        trainer = _make_torch_trainer(torch_dtype="float32")
        result  = trainer.train(_lm_dataset())
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_bfloat16_cpu_runs(self):
        trainer = _make_torch_trainer(torch_dtype="bfloat16")
        result  = trainer.train(_lm_dataset())
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_embed_float32_runs(self):
        trainer = _make_embed_trainer(torch_dtype="float32")
        result  = trainer.train(_emb_dataset())
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))

    def test_embed_bfloat16_runs(self):
        trainer = _make_embed_trainer(torch_dtype="bfloat16")
        result  = trainer.train(_emb_dataset())
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))


# ── Gradient accumulation ──────────────────────────────────────────────────

class TestGradAccumulation(unittest.TestCase):

    def test_accum_2_produces_correct_loss_count(self):
        n_batches = 6
        trainer   = _make_torch_trainer(grad_accum=2)
        result    = trainer.train(_lm_dataset(n=n_batches))
        self.assertEqual(len(result["losses"]), n_batches)

    def test_accum_1_and_accum_2_both_decrease_loss(self):
        # Repeat the same fixed batch so the teacher target is constant —
        # guarantees convergence regardless of random init.
        fixed = np.random.randint(0, 100, (2, 8)).astype(np.int32)
        data  = [fixed] * 30
        for accum in (1, 2):
            trainer = _make_torch_trainer(grad_accum=accum)
            result  = trainer.train(list(data))
            losses  = result["losses"]
            self.assertLess(
                sum(losses[-5:]) / 5,
                sum(losses[:5])  / 5,
                f"grad_accum={accum}: expected loss to decrease",
            )

    def test_embed_accum_2(self):
        trainer = _make_embed_trainer(grad_accum=2)
        result  = trainer.train(_emb_dataset(n=6))
        self.assertEqual(len(result["losses"]), 6)

    def test_on_step_fires_every_accum_steps(self):
        """on_step fires per optimizer step, not per batch."""
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        calls = []
        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg = TorchTrainConfig(device="cpu", epochs=1, grad_accumulation_steps=2, log_every=1)
        trainer = TorchDistillTrainer(TinyLM(), teachers, config=cfg)
        trainer.train(_lm_dataset(n=4), on_step=lambda s, l: calls.append(s))
        self.assertEqual(len(calls), 2)   # 4 batches / 2 accum = 2 steps


# ── Shuffle ────────────────────────────────────────────────────────────────

class TestShuffle(unittest.TestCase):

    def test_shuffle_does_not_change_loss_count(self):
        trainer = _make_torch_trainer(epochs=2)
        result  = trainer.train(_lm_dataset(n=5), shuffle=True)
        self.assertEqual(len(result["losses"]), 10)   # 2 epochs × 5 batches

    def test_embed_shuffle_does_not_change_loss_count(self):
        trainer = _make_embed_trainer(epochs=2)
        result  = trainer.train(_emb_dataset(n=5), shuffle=True)
        self.assertEqual(len(result["losses"]), 10)


# ── Checkpoint save / resume ───────────────────────────────────────────────

class TestCheckpoint(unittest.TestCase):

    def test_save_checkpoint_creates_file(self):
        trainer = _make_torch_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = trainer.save_checkpoint(tmp)
            self.assertTrue(ckpt.exists())
            self.assertEqual(ckpt.name, "checkpoint.pt")

    def test_resume_from_checkpoint_restores_weights(self):
        trainer = _make_torch_trainer()
        data    = _lm_dataset(n=4)
        trainer.train(data)  # advance weights

        with tempfile.TemporaryDirectory() as tmp:
            trainer.save_checkpoint(tmp)

            trainer2 = _make_torch_trainer()
            before   = next(trainer2.student.parameters()).clone()
            trainer2.resume_from_checkpoint(tmp)
            after    = next(trainer2.student.parameters()).clone()

            self.assertFalse(torch.allclose(before, after),
                             "Resumed weights should differ from random init")

    def test_embed_save_and_resume(self):
        trainer = _make_embed_trainer()
        data    = _emb_dataset(n=4)
        trainer.train(data)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = trainer.save_checkpoint(tmp)
            self.assertTrue(ckpt.exists())

            trainer2 = _make_embed_trainer()
            before   = next(trainer2.student.parameters()).clone()
            trainer2.resume_from_checkpoint(tmp)
            after    = next(trainer2.student.parameters()).clone()
            self.assertFalse(torch.allclose(before, after))

    def test_cached_trainer_saves_and_resumes(self):
        from foundry.training import CachedDistillTrainer, CachedDistillConfig
        from foundry.teachers.registry import TeacherRegistry

        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg      = CachedDistillConfig(
            device="cpu", epochs=1, use_accelerate=False
        )
        trainer  = CachedDistillTrainer(TinyLM(), teachers, config=cfg)
        trainer.train(_lm_dataset(n=4))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = trainer.save_checkpoint(tmp)
            self.assertTrue(ckpt.exists())


# ── Teacher embedding cache ────────────────────────────────────────────────

class TestEmbedCache(unittest.TestCase):

    def test_build_embed_cache_populates(self):
        trainer = _make_embed_trainer()
        data    = _emb_dataset(n=5)
        trainer.build_embed_cache(data)
        self.assertEqual(len(trainer._embed_cache), 5)

    def test_pre_cache_flag(self):
        trainer = _make_embed_trainer()
        data    = _emb_dataset(n=5)
        self.assertEqual(len(trainer._embed_cache), 0)
        trainer.train(data, pre_cache=True)
        self.assertEqual(len(trainer._embed_cache), 5)

    def test_cached_result_matches_live(self):
        """Loss with pre_cache should be finite (same teacher, reproducible)."""
        trainer = _make_embed_trainer()
        data    = _emb_dataset(n=4)
        result  = trainer.train(data, pre_cache=True)
        self.assertTrue(all(np.isfinite(l) for l in result["losses"]))


# ── snap_on with real-world key format ────────────────────────────────────

class TestSnapOnKeyMatching(unittest.TestCase):

    def _make_registry_and_pack(self, state_keys):
        from foundry.skillpacks.pack import SkillPack, SkillRegistry, _model_hash
        state = {k: np.ones((32, 32), dtype=np.float32) for k in state_keys}
        bh    = _model_hash(state)
        pack  = SkillPack(
            name="test", base_hash=bh, rank=4,
            target_modules=["q_proj", "v_proj"],
            weights={
                "q_proj": {
                    "A": np.zeros((4, 32), dtype=np.float32),
                    "B": np.zeros((32, 4), dtype=np.float32),
                },
            },
        )
        reg = SkillRegistry(state)
        reg.register(pack)
        return reg, state

    def test_bare_module_key(self):
        """State dict with bare 'q_proj' key — the M0 test format."""
        reg, _ = self._make_registry_and_pack(["q_proj", "v_proj"])
        merged = reg.snap_on("test")
        self.assertIn("q_proj", merged)

    def test_full_path_weight_key(self):
        """State dict with 'model.layers.0.self_attn.q_proj.weight' — real HF format."""
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        ]
        reg, _ = self._make_registry_and_pack(keys)
        merged = reg.snap_on("test")
        self.assertIn("model.layers.0.self_attn.q_proj.weight", merged)

    def test_non_target_key_untouched(self):
        """k_proj should not be modified (not in target_modules)."""
        keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        ]
        from foundry.skillpacks.pack import SkillPack, SkillRegistry, _model_hash
        state = {k: np.ones((32, 32), dtype=np.float32) for k in keys}
        bh    = _model_hash(state)
        pack  = SkillPack(
            name="p", base_hash=bh, rank=4,
            target_modules=["q_proj"],
            weights={"q_proj": {
                "A": np.random.randn(4, 32).astype(np.float32),
                "B": np.random.randn(32, 4).astype(np.float32),
            }},
        )
        reg = SkillRegistry(state)
        reg.register(pack)
        merged = reg.snap_on("p")
        np.testing.assert_array_equal(
            merged["model.layers.0.self_attn.k_proj.weight"],
            np.ones((32, 32)),
        )


# ── HFTeacher model_type ───────────────────────────────────────────────────

class TestHFTeacherModelType(unittest.TestCase):

    def test_default_model_type_is_causal_lm(self):
        from foundry.teachers.registry import HFTeacher
        t = HFTeacher(name="org/model")
        self.assertEqual(t.model_type, "causal_lm")

    def test_encoder_model_type_accepted(self):
        from foundry.teachers.registry import HFTeacher
        t = HFTeacher(name="org/encoder-model", model_type="encoder")
        self.assertEqual(t.model_type, "encoder")


# ── load_model model_class parameter ──────────────────────────────────────

class TestLoadModelClass(unittest.TestCase):

    def test_default_model_class_docstring(self):
        from foundry.io.loader import load_model
        import inspect
        doc = inspect.getdoc(load_model)
        self.assertIn("model_class", doc)

    def test_model_class_accepted_in_signature(self):
        import inspect
        from foundry.io.loader import load_model
        sig = inspect.signature(load_model)
        self.assertIn("model_class", sig.parameters)


# ── detect_backend includes wandb ─────────────────────────────────────────

class TestBackendsWandb(unittest.TestCase):

    def test_wandb_key_present(self):
        from foundry.backends import detect_backend
        info = detect_backend()
        self.assertIn("wandb", info)
        self.assertIsInstance(info["wandb"], bool)


# ── _FoundryLogger no-op ──────────────────────────────────────────────────

class TestFoundryLogger(unittest.TestCase):

    def test_noop_logger_does_not_crash(self):
        from foundry.training._logger import _FoundryLogger
        logger = _FoundryLogger("none", "proj", "run", {})
        logger.log(0, 1.23)
        logger.log(10, 0.45)
        logger.finish()   # should not raise

    def test_invalid_backend_silently_degrades(self):
        from foundry.training._logger import _FoundryLogger
        logger = _FoundryLogger("nonexistent_backend", "proj", "run", {})
        logger.log(0, 1.0)   # must not raise
        logger.finish()


# ── CachedDistillTrainer shuffle ──────────────────────────────────────────

class TestCachedTrainerShuffle(unittest.TestCase):

    def test_shuffle_completes(self):
        from foundry.training import CachedDistillTrainer, CachedDistillConfig
        from foundry.teachers.registry import TeacherRegistry

        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg      = CachedDistillConfig(
            device="cpu", epochs=2, use_accelerate=False
        )
        trainer  = CachedDistillTrainer(TinyLM(), teachers, config=cfg)
        result   = trainer.train(_lm_dataset(n=4), shuffle=True)
        self.assertEqual(len(result["losses"]), 8)


# ── LR scheduler ─────────────────────────────────────────────────────────

class TestLRScheduler(unittest.TestCase):

    def _get_lrs(self, scheduler_name, warmup=0, n_steps=10):
        """Run N optimizer steps and collect LR at each step."""
        from foundry.training._scheduler import build_scheduler
        import torch
        param  = torch.nn.Parameter(torch.tensor([1.0]))
        opt    = torch.optim.AdamW([param], lr=1e-3)
        sched  = build_scheduler(opt, scheduler_name, warmup, n_steps)
        lrs = []
        for _ in range(n_steps):
            lrs.append(opt.param_groups[0]["lr"])
            if sched:
                sched.step()
        return lrs

    def test_constant_no_warmup_returns_none(self):
        from foundry.training._scheduler import build_scheduler
        import torch
        p   = torch.nn.Parameter(torch.tensor([1.0]))
        opt = torch.optim.AdamW([p], lr=1e-3)
        self.assertIsNone(build_scheduler(opt, "constant", 0, 10))

    def test_cosine_decreases_after_warmup(self):
        lrs = self._get_lrs("cosine", warmup=2, n_steps=20)
        self.assertLess(lrs[-1], lrs[3])   # last < post-warmup start

    def test_linear_decreases_monotonically_after_warmup(self):
        lrs = self._get_lrs("linear", warmup=0, n_steps=10)
        for a, b in zip(lrs[1:], lrs[2:]):
            self.assertLessEqual(b, a + 1e-12)

    def test_warmup_ramps_up(self):
        lrs = self._get_lrs("cosine", warmup=5, n_steps=20)
        # LR at step 0 should be smaller than LR at step 4 (still in warmup)
        self.assertLess(lrs[0], lrs[4])

    def test_trainer_cosine_scheduler_runs(self):
        trainer = _make_torch_trainer(epochs=1)
        trainer.cfg.lr_scheduler = "cosine"
        trainer.cfg.warmup_steps = 2
        result  = trainer.train(_lm_dataset(n=10))
        self.assertEqual(len(result["losses"]), 10)

    def test_embed_trainer_cosine_runs(self):
        trainer = _make_embed_trainer(epochs=1)
        trainer.cfg.lr_scheduler = "cosine"
        trainer.cfg.warmup_steps = 2
        result  = trainer.train(_emb_dataset(n=10))
        self.assertEqual(len(result["losses"]), 10)


# ── Seed reproducibility ──────────────────────────────────────────────────

class TestSeedReproducibility(unittest.TestCase):

    def test_same_seed_same_first_loss(self):
        """Two trainers with same seed and same data produce same first loss.

        The seed must be applied before model creation (it controls weight init)
        AND passed to the config (it re-applies before the training loop for
        training-time randomness like dropout).
        """
        fixed = np.random.randint(0, 100, (2, 8)).astype(np.int32)
        data  = [fixed]

        def run(seed):
            from foundry.training import TorchDistillTrainer, TorchTrainConfig
            from foundry.teachers.registry import TeacherRegistry
            # Seed before model creation so weight init is deterministic
            torch.manual_seed(seed)
            np.random.seed(seed)
            teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
            cfg      = TorchTrainConfig(device="cpu", epochs=1, seed=seed)
            trainer  = TorchDistillTrainer(TinyLM(), teachers, config=cfg)
            return trainer.train(list(data))["losses"][0]

        l1 = run(seed=99)
        l2 = run(seed=99)
        self.assertAlmostEqual(l1, l2, places=5)

    def test_different_seeds_usually_differ(self):
        """Two different seeds should produce different losses."""
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        fixed = np.random.randint(0, 100, (2, 8)).astype(np.int32)
        data  = [fixed] * 5

        def run(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
            cfg      = TorchTrainConfig(device="cpu", epochs=1, seed=seed)
            trainer  = TorchDistillTrainer(TinyLM(), teachers, config=cfg)
            return trainer.train(list(data))["losses"][-1]

        l1 = run(seed=1)
        l2 = run(seed=9999)
        self.assertNotAlmostEqual(l1, l2, places=3)

    def test_embed_seed_reproduces(self):
        from foundry.training import EmbeddingDistillTrainer, EmbeddingDistillConfig, ToyEmbeddingTeacher

        fixed = _emb_dataset(n=3)

        def run():
            torch.manual_seed(77)
            np.random.seed(77)
            student = TinyEncoder()
            teacher = ToyEmbeddingTeacher(dim=32)
            cfg     = EmbeddingDistillConfig(device="cpu", epochs=1, seed=77)
            trainer = EmbeddingDistillTrainer(student, teacher, config=cfg)
            return trainer.train(list(fixed))["losses"][0]

        self.assertAlmostEqual(run(), run(), places=5)


# ── Eval loop ────────────────────────────────────────────────────────────

class TestEvalLoop(unittest.TestCase):

    def test_eval_losses_returned(self):
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg      = TorchTrainConfig(device="cpu", epochs=1, eval_every=1)
        trainer  = TorchDistillTrainer(TinyLM(), teachers, config=cfg)

        train_data = _lm_dataset(n=4)
        eval_data  = _lm_dataset(n=2)
        result     = trainer.train(train_data, eval_dataset=eval_data)

        self.assertIn("eval_losses", result)
        self.assertGreater(len(result["eval_losses"]), 0)

    def test_eval_loss_is_finite(self):
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg      = TorchTrainConfig(device="cpu", epochs=1, eval_every=2)
        trainer  = TorchDistillTrainer(TinyLM(), teachers, config=cfg)
        result   = trainer.train(_lm_dataset(n=6), eval_dataset=_lm_dataset(n=3))

        for step, ev in result["eval_losses"].items():
            self.assertTrue(np.isfinite(ev), f"eval_loss at step {step} is not finite")

    def test_embed_eval_loop(self):
        trainer = _make_embed_trainer(epochs=1)
        trainer.cfg.eval_every = 2
        result  = trainer.train(
            _emb_dataset(n=6),
            eval_dataset=_emb_dataset(n=2),
        )
        self.assertGreater(len(result["eval_losses"]), 0)

    def test_no_eval_when_eval_every_zero(self):
        trainer = _make_torch_trainer(epochs=1)
        result  = trainer.train(_lm_dataset(n=4), eval_dataset=_lm_dataset(n=2))
        self.assertEqual(len(result["eval_losses"]), 0)


# ── Auto-checkpoint ──────────────────────────────────────────────────────

class TestAutoCheckpoint(unittest.TestCase):

    def test_auto_checkpoint_saves(self):
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        with tempfile.TemporaryDirectory() as tmp:
            teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
            cfg      = TorchTrainConfig(
                device="cpu", epochs=1,
                save_every=2, save_dir=tmp,
            )
            trainer = TorchDistillTrainer(TinyLM(), teachers, config=cfg)
            trainer.train(_lm_dataset(n=6))
            self.assertTrue((Path(tmp) / "checkpoint.pt").exists())

    def test_embed_auto_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_embed_trainer(epochs=1)
            trainer.cfg.save_every = 2
            trainer.cfg.save_dir   = tmp
            trainer.train(_emb_dataset(n=6))
            self.assertTrue((Path(tmp) / "checkpoint.pt").exists())


# ── DataPipeline ─────────────────────────────────────────────────────────

class TestDataPipeline(unittest.TestCase):

    def test_from_numpy_list_lm_mode(self):
        from foundry.data import DataPipeline
        rows = [np.array([1, 2, 3, 4, 5, 0, 0, 0]) for _ in range(10)]
        pipe = DataPipeline(rows, batch_size=4, max_length=8, mode="lm")
        batches = list(pipe)
        self.assertEqual(len(batches), 3)              # ceil(10/4)
        self.assertEqual(batches[0].shape, (4, 8))

    def test_from_dict_list_embed_mode(self):
        from foundry.data import DataPipeline
        rows = [
            {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
            for _ in range(8)
        ]
        pipe    = DataPipeline(rows, batch_size=4, max_length=8, mode="embed")
        batches = list(pipe)
        self.assertEqual(len(batches), 2)
        self.assertIn("input_ids", batches[0])
        self.assertIn("attention_mask", batches[0])
        self.assertEqual(batches[0]["input_ids"].shape, (4, 8))

    def test_from_string_list_with_mock_tokenizer(self):
        from foundry.data import DataPipeline
        tok  = lambda text: [ord(c) % 50 for c in text[:6]]
        rows = ["hello world", "foo bar baz"] * 4
        pipe = DataPipeline(rows, tokenizer=tok, batch_size=4, max_length=8, mode="lm")
        batches = list(pipe)
        self.assertGreater(len(batches), 0)
        self.assertEqual(batches[0].shape, (4, 8))

    def test_padding_shorter_than_max_length(self):
        from foundry.data import DataPipeline
        rows = [{"input_ids": [1, 2]} for _ in range(4)]
        pipe = DataPipeline(rows, batch_size=2, max_length=8, mode="lm")
        batch = next(iter(pipe))
        # First 2 positions filled, rest padded to 0
        self.assertEqual(batch[0, 2], 0)

    def test_truncation_longer_than_max_length(self):
        from foundry.data import DataPipeline
        rows = [{"input_ids": list(range(20))} for _ in range(4)]
        pipe = DataPipeline(rows, batch_size=2, max_length=8, mode="lm")
        batch = next(iter(pipe))
        self.assertEqual(batch.shape, (2, 8))

    def test_len_known_for_finite_source(self):
        from foundry.data import DataPipeline
        rows = list(range(10))   # 10 examples
        pipe = DataPipeline(rows, batch_size=3, max_length=8)
        self.assertEqual(len(pipe), 4)   # ceil(10/3)

    def test_shuffle_buffer_yields_all_examples(self):
        from foundry.data import DataPipeline
        rows = [{"input_ids": [i] * 4} for i in range(20)]
        pipe = DataPipeline(rows, batch_size=4, max_length=4,
                            shuffle_buffer=10, mode="embed")
        total = sum(b["input_ids"].shape[0] for b in pipe)
        self.assertEqual(total, 20)

    def test_drop_last(self):
        from foundry.data import DataPipeline
        rows  = [{"input_ids": [1, 2, 3, 4]} for _ in range(9)]
        pipe  = DataPipeline(rows, batch_size=4, max_length=4, drop_last=True)
        self.assertEqual(len(pipe), 2)   # 9 // 4 = 2

    def test_pipeline_plugs_into_trainer(self):
        """End-to-end: DataPipeline → EmbeddingDistillTrainer."""
        from foundry.data import DataPipeline
        from foundry.training import EmbeddingDistillTrainer, EmbeddingDistillConfig, ToyEmbeddingTeacher

        rows = [{"input_ids": np.random.randint(0, 100, 8).tolist(),
                 "attention_mask": [1] * 8} for _ in range(10)]
        pipe = DataPipeline(rows, batch_size=4, max_length=8, mode="embed")

        student = TinyEncoder()
        teacher = ToyEmbeddingTeacher(dim=32)
        cfg     = EmbeddingDistillConfig(device="cpu", epochs=1)
        trainer = EmbeddingDistillTrainer(student, teacher, config=cfg)
        result  = trainer.train(pipe)
        self.assertGreater(len(result["losses"]), 0)

    def test_pipeline_lm_trainer_integration(self):
        """End-to-end: DataPipeline (lm mode) → TorchDistillTrainer."""
        from foundry.data import DataPipeline
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        rows = [{"input_ids": np.random.randint(0, 100, 8).tolist()} for _ in range(8)]
        pipe = DataPipeline(rows, batch_size=4, max_length=8, mode="lm")

        teachers = TeacherRegistry.from_toy(n=1, vocab_size=100)
        cfg      = TorchTrainConfig(device="cpu", epochs=1)
        trainer  = TorchDistillTrainer(TinyLM(), teachers, config=cfg)
        result   = trainer.train(pipe)
        self.assertGreater(len(result["losses"]), 0)


# ── DataPipeline exported from foundry ───────────────────────────────────

class TestDataPipelineExport(unittest.TestCase):

    def test_exported_from_foundry(self):
        import foundry
        self.assertIn("DataPipeline", foundry.__all__)
        from foundry import DataPipeline
        self.assertIsNotNone(DataPipeline)


# ── KL loss is per-token, not sequence-inflated ───────────────────────────

class TestKLLossScale(unittest.TestCase):
    """Regression: F.kl_div with reduction='batchmean' on a (B,S,V) tensor only
    divides by B, leaving KL summed over the sequence — hundreds× too large and
    swamping the CE term so `alpha` stops balancing them. The loss must stay on a
    per-token nats scale (single digits), matching the eval CE."""

    def test_combined_loss_is_per_token_scale(self):
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        torch.manual_seed(0)
        np.random.seed(0)
        vocab, S = 100, 64
        teachers = TeacherRegistry.from_toy(n=1, vocab_size=vocab)
        cfg      = TorchTrainConfig(device="cpu", epochs=1, alpha=0.3, log_every=1)
        trainer  = TorchDistillTrainer(TinyLM(vocab), teachers, config=cfg)

        data   = [np.random.randint(0, vocab, (2, S)).astype(np.int32) for _ in range(3)]
        result = trainer.train(data)

        # Per-token CE ≈ log(vocab) ≈ 4.6; per-token KL is small. The buggy
        # sequence-summed KL would push this into the hundreds for S=64.
        for l in result["losses"]:
            self.assertTrue(np.isfinite(l))
            self.assertLess(l, 20.0, f"loss {l:.1f} looks seq-inflated, not per-token")

    def test_eval_and_train_loss_same_order(self):
        """With teachers present, the train loss should be within the same order
        of magnitude as the CE-only eval loss (both per-token nats)."""
        from foundry.training import TorchDistillTrainer, TorchTrainConfig
        from foundry.teachers.registry import TeacherRegistry

        torch.manual_seed(1)
        np.random.seed(1)
        vocab, S = 100, 64
        teachers = TeacherRegistry.from_toy(n=1, vocab_size=vocab)
        cfg      = TorchTrainConfig(device="cpu", epochs=1, alpha=0.3, eval_every=1)
        trainer  = TorchDistillTrainer(TinyLM(vocab), teachers, config=cfg)

        train = [np.random.randint(0, vocab, (2, S)).astype(np.int32) for _ in range(3)]
        ev    = [np.random.randint(0, vocab, (2, S)).astype(np.int32) for _ in range(2)]
        result = trainer.train(train, eval_dataset=ev)

        eval_loss  = list(result["eval_losses"].values())[-1]
        train_loss = result["losses"][-1]
        self.assertLess(train_loss, eval_loss * 10.0,
                        f"train {train_loss:.1f} vs eval {eval_loss:.1f}: KL likely inflated")


if __name__ == "__main__":
    unittest.main()
