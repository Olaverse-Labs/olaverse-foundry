"""
Sparse top-k distillation target and KL — foundry.training._sparse_kl.

The contract these tests defend is equivalence: computing the KL over the
teacher's top-k support must give the same number as scattering that teacher
into a dense (B, S, vocab) target and summing over the whole vocabulary. KL has
no contribution from any token the teacher gave zero probability, so the dense
sum is doing arithmetic on zeros — ~10GB of them per teacher per step at a 152k
vocabulary.

If these drift, the sparse path is silently training against a different target
than the one the dense path defines, which no loss curve would reveal.
"""
from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")

import numpy as np
import torch
import torch.nn.functional as F

from foundry.fusion.align import EMAlignment, IdentityAlignment, MinEDAlignment
from foundry.fusion.strategies import STRATEGY_REGISTRY
from foundry.training._sparse_kl import (
    align_sparse,
    build_sparse_target,
    sparse_kl,
)

B, S, V, K = 2, 5, 50, 8


def _teacher(seed: int, vocab: int = V):
    rng = np.random.RandomState(seed)
    idx = np.stack([rng.choice(vocab, K, replace=False)
                    for _ in range(B * S)]).reshape(B, S, K).astype(np.int64)
    probs = rng.dirichlet(np.ones(K), size=B * S).reshape(B, S, K).astype(np.float32)
    return idx, probs


def _dense_kl(dists, gold, weights, strategy, logits, mask):
    """The dense reference, without the 1e-9 vocabulary-wide smoothing."""
    fused = STRATEGY_REGISTRY[strategy](dists, gold, weights)
    target = torch.tensor(fused, dtype=torch.float32)
    target = target / target.sum(-1, keepdim=True).clamp(min=1e-9)
    kl = F.kl_div(F.log_softmax(logits, dim=-1), target,
                  reduction="none", log_target=False).sum(-1)
    return float((kl * mask).sum() / mask.sum())


class SparseKLTestCase(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.logits = torch.randn(B, S, V)
        self.mask   = torch.ones(B, S)
        self.gold   = np.random.randint(0, V, (B, S))

    def _compare(self, teachers, weights, strategy, alignment, places=6):
        dense = [alignment.map(i, p, V) for i, p in teachers]
        expected = _dense_kl(dense, self.gold, weights, strategy, self.logits, self.mask)

        sparse = [align_sparse(alignment, i, p) for i, p in teachers]
        idx, prob = build_sparse_target(sparse, self.gold, weights, strategy, "cpu")
        actual = float(sparse_kl(self.logits, idx, prob, self.mask))

        self.assertAlmostEqual(expected, actual, places=places,
                               msg=f"{strategy} diverged: dense {expected} vs sparse {actual}")
        return actual


class TestMatchesDenseSameTokenizer(SparseKLTestCase):

    def test_min_ce_one_teacher(self):
        self._compare([_teacher(0)], [1.0], "min_ce", IdentityAlignment())

    def test_min_ce_two_teachers(self):
        self._compare([_teacher(0), _teacher(1)], [1.0, 0.7], "min_ce", IdentityAlignment())

    def test_min_ce_three_teachers(self):
        self._compare([_teacher(i) for i in range(3)], [1.0, 0.7, 0.4],
                      "min_ce", IdentityAlignment())

    def test_mean_two_teachers(self):
        self._compare([_teacher(0), _teacher(1)], [1.0, 0.7], "mean", IdentityAlignment())

    def test_mean_three_teachers(self):
        self._compare([_teacher(i) for i in range(3)], [1.0, 0.7, 0.4],
                      "mean", IdentityAlignment())

    def test_mean_uniform_weights(self):
        self._compare([_teacher(0), _teacher(1)], None, "mean", IdentityAlignment())


class TestMatchesDenseCrossTokenizer(SparseKLTestCase):
    """Unmapped teacher tokens must be dropped, colliding ones summed."""

    def _vocabs(self, step: int):
        teacher_vocab = {f"t{i}": i for i in range(V)}
        student_vocab = {f"t{i}": i for i in range(0, V, step)}
        return teacher_vocab, student_vocab

    def test_exact_match_alignment_with_gaps(self):
        tv, sv = self._vocabs(2)          # half the teacher vocab has no student token
        self._compare([_teacher(9)], [1.0], "min_ce", EMAlignment(tv, sv))

    def test_mined_alignment_collides_tokens(self):
        """MinED maps several teacher tokens onto one student token."""
        tv, sv = self._vocabs(4)
        self._compare([_teacher(3)], [1.0], "min_ce", MinEDAlignment(tv, sv))

    def test_mined_alignment_multi_teacher_mean(self):
        tv, sv = self._vocabs(4)
        self._compare([_teacher(3), _teacher(5)], [1.0, 0.5], "mean",
                      MinEDAlignment(tv, sv))


class TestAlignSparse(unittest.TestCase):

    def test_identity_passes_indices_through(self):
        idx, probs = _teacher(0)
        out_idx, out_probs = align_sparse(IdentityAlignment(), idx, probs)
        np.testing.assert_array_equal(out_idx, idx)
        np.testing.assert_allclose(out_probs, probs)

    def test_unmapped_tokens_get_zero_probability(self):
        teacher_vocab = {"a": 0, "b": 1, "c": 2}
        student_vocab = {"a": 5}
        idx = np.array([[[0, 1, 2]]], dtype=np.int64)
        probs = np.array([[[0.5, 0.3, 0.2]]], dtype=np.float32)
        out_idx, out_probs = align_sparse(EMAlignment(teacher_vocab, student_vocab), idx, probs)
        self.assertTrue((out_idx >= 0).all(), "negative index would crash gather")
        np.testing.assert_allclose(out_probs[0, 0], [0.5, 0.0, 0.0])


class TestSparseKLProperties(SparseKLTestCase):

    def test_zero_when_student_matches_teacher(self):
        """KL(T||T) = 0 — the tightest available check on the maths."""
        idx, probs = _teacher(0)
        probs = probs / probs.sum(-1, keepdims=True)
        logits = torch.full((B, S, V), -30.0)
        bi = torch.arange(B)[:, None, None]
        si = torch.arange(S)[None, :, None]
        logits[bi, si, torch.tensor(idx)] = torch.log(torch.tensor(probs))

        value = float(sparse_kl(logits, torch.tensor(idx),
                                torch.tensor(probs), self.mask))
        self.assertAlmostEqual(value, 0.0, places=4)

    def test_non_negative(self):
        idx, probs = _teacher(4)
        value = float(sparse_kl(self.logits, torch.tensor(idx),
                                torch.tensor(probs), self.mask))
        self.assertGreaterEqual(value, -1e-6)

    def test_masked_positions_are_excluded(self):
        idx, probs = _teacher(2)
        mask = torch.ones(B, S)
        mask[:, -2:] = 0.0
        full = float(sparse_kl(self.logits, torch.tensor(idx), torch.tensor(probs),
                               torch.ones(B, S)))
        masked = float(sparse_kl(self.logits, torch.tensor(idx), torch.tensor(probs), mask))
        self.assertNotAlmostEqual(full, masked, places=5)

    def test_all_zero_probabilities_do_not_produce_nan(self):
        idx = np.zeros((B, S, K), dtype=np.int64)
        probs = np.zeros((B, S, K), dtype=np.float32)
        value = float(sparse_kl(self.logits, torch.tensor(idx),
                                torch.tensor(probs), self.mask))
        self.assertTrue(np.isfinite(value), "0 * log 0 leaked a NaN")

    def test_gradient_flows_to_the_student(self):
        idx, probs = _teacher(7)
        logits = self.logits.clone().requires_grad_(True)
        sparse_kl(logits, torch.tensor(idx), torch.tensor(probs), self.mask).backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


class TestTrainerAgreement(unittest.TestCase):
    """The two config paths must train against the same target."""

    def _losses(self, sparse: bool):
        from types import SimpleNamespace
        import torch.nn as nn
        from foundry.teachers import TeacherRegistry, ToyTeacher
        from foundry.training.torch_distill import TorchDistillTrainer, TorchTrainConfig

        class TinyLM(nn.Module):
            def __init__(self, vocab=64, dim=16):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.head  = nn.Linear(dim, vocab)
            def forward(self, input_ids, **_):
                return SimpleNamespace(logits=self.head(self.embed(input_ids)))

        torch.manual_seed(0)
        np.random.seed(0)
        registry = TeacherRegistry([ToyTeacher(name="toy", vocab_size=64)])
        dataset = [np.random.randint(0, 64, (2, 8)) for _ in range(3)]
        cfg = TorchTrainConfig(device="cpu", epochs=1, seed=0, top_k=8, sparse_kl=sparse)
        return TorchDistillTrainer(TinyLM(), registry, config=cfg).train(dataset)["losses"]

    def test_sparse_is_the_default(self):
        from foundry.training.torch_distill import TorchTrainConfig
        self.assertTrue(TorchTrainConfig().sparse_kl)

    def test_both_paths_agree(self):
        sparse = self._losses(True)
        dense  = self._losses(False)
        self.assertEqual(len(sparse), len(dense))
        for i, (a, b) in enumerate(zip(sparse, dense)):
            self.assertAlmostEqual(a, b, places=4, msg=f"step {i}: {a} vs {b}")


if __name__ == "__main__":
    unittest.main()
