"""
Fusion-strategy name resolution.

The registry key is "mean", but the docs, the CLI help and the recipe examples
have always said "mean_ce". The trainers looked the name up with
``STRATEGY_REGISTRY.get(name, STRATEGY_REGISTRY["min_ce"])``, so the documented
spelling — and any typo — silently selected MinCE. A run configured to average
its teachers picked one per token instead, and nothing in the output said so.

numpy-only: runs on a core install.
"""
from __future__ import annotations

import importlib.util
import unittest

from foundry.fusion.strategies import (
    STRATEGY_REGISTRY,
    canonical_strategy,
    mean_ce,
    min_ce,
    resolve_strategy,
)


class TestCanonicalStrategy(unittest.TestCase):

    def test_registry_keys_map_to_themselves(self):
        for key in STRATEGY_REGISTRY:
            self.assertEqual(canonical_strategy(key), key)

    def test_documented_alias_resolves(self):
        """docs/training/config.md, cli.md and recipes.md all say "mean_ce"."""
        self.assertEqual(canonical_strategy("mean_ce"), "mean")

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            canonical_strategy("nonsense")

    def test_error_names_the_valid_options(self):
        with self.assertRaises(ValueError) as ctx:
            canonical_strategy("mincе")           # note: Cyrillic е, a real typo
        message = str(ctx.exception)
        self.assertIn("min_ce", message)
        self.assertIn("mean", message)

    def test_empty_and_none_are_rejected(self):
        for bad in ("", "MIN_CE", "Mean"):
            with self.assertRaises(ValueError, msg=bad):
                canonical_strategy(bad)


class TestResolveStrategy(unittest.TestCase):

    def test_min_ce(self):
        self.assertIs(resolve_strategy("min_ce"), min_ce)

    def test_mean_and_alias_are_the_same_function(self):
        self.assertIs(resolve_strategy("mean"), mean_ce)
        self.assertIs(resolve_strategy("mean_ce"), mean_ce)

    def test_alias_does_not_silently_become_min_ce(self):
        """The regression: "mean_ce" must not resolve to MinCE."""
        self.assertIsNot(resolve_strategy("mean_ce"), min_ce)

    def test_unknown_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            resolve_strategy("typo")


class TestRecipeSchemaAcceptsTheDocumentedSpelling(unittest.TestCase):

    def test_all_three_spellings_validate(self):
        from foundry.recipes.schema import FusionConfig
        for name in ("min_ce", "mean", "mean_ce"):
            self.assertEqual(FusionConfig(strategy=name).strategy, name)

    def test_nonsense_still_rejected(self):
        import pydantic
        from foundry.recipes.schema import FusionConfig
        with self.assertRaises(pydantic.ValidationError):
            FusionConfig(strategy="nonsense")


# The name-resolution tests above are numpy-only and run on a core install.
# The trainer test below drives a real training step, so it needs torch.
needs_torch = unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "trainer integration needs torch",
)


@needs_torch
class TestTrainerHonoursTheStrategy(unittest.TestCase):
    """
    The end-to-end regression. If "mean_ce" silently resolved to MinCE — as it
    did — these two runs would produce identical losses, because they would be
    the same computation. They must differ.
    """

    def _losses(self, strategy: str):
        import numpy as np
        import torch
        import torch.nn as nn
        from types import SimpleNamespace
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
        # Two teachers that disagree. ToyTeacher derives its RNG from
        # self.seed + input_ids.sum(), so two teachers left on the default seed
        # return *identical* distributions — every strategy then collapses to
        # the same answer and the test proves nothing. Different seeds is what
        # makes min_ce and mean distinguishable at all.
        registry = TeacherRegistry([
            ToyTeacher(name="a", vocab_size=64, weight=1.0, seed=1),
            ToyTeacher(name="b", vocab_size=64, weight=0.3, seed=999),
        ])
        dataset = [np.random.randint(0, 64, (2, 8)) for _ in range(3)]
        cfg = TorchTrainConfig(device="cpu", epochs=1, seed=0, top_k=8,
                               fusion_strategy=strategy)
        return TorchDistillTrainer(TinyLM(), registry, config=cfg).train(dataset)["losses"]

    def test_min_ce_and_mean_ce_differ(self):
        min_losses  = self._losses("min_ce")
        mean_losses = self._losses("mean_ce")
        self.assertEqual(len(min_losses), len(mean_losses))
        self.assertNotEqual(
            [round(x, 6) for x in min_losses],
            [round(x, 6) for x in mean_losses],
            "mean_ce produced MinCE's losses — the alias is resolving wrongly",
        )

    def test_mean_and_mean_ce_are_identical(self):
        self.assertEqual(
            [round(x, 6) for x in self._losses("mean")],
            [round(x, 6) for x in self._losses("mean_ce")],
        )

    def test_unknown_strategy_raises_instead_of_training(self):
        with self.assertRaises(ValueError):
            self._losses("not_a_strategy")


if __name__ == "__main__":
    unittest.main()
