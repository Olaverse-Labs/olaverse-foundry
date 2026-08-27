"""
Tests for foundry.quality — the pre-training gate on synthesised data.

numpy-only: these run on a core install, which is the point. The failure modes
being caught here (duplicate pairs, identity pairs, false negatives, collapsed
encoders) all produce a *plausible* training run, so they need to be caught
before training, not after.
"""
from __future__ import annotations

import unittest

import numpy as np

from foundry.quality import (
    clean_pairs,
    dedup_pairs,
    drop_degenerate_pairs,
    duplicate_rate,
    embedding_health,
    normalise_text,
    quality_report,
)


class TestNormaliseText(unittest.TestCase):

    def test_collapses_whitespace_and_case(self):
        self.assertEqual(normalise_text("  Water   RIVER \n"), "water river")

    def test_keeps_punctuation(self):
        """Punctuation can be meaningful; only spacing and case are noise."""
        self.assertNotEqual(normalise_text("omi, odo"), normalise_text("omi odo"))

    def test_handles_non_strings(self):
        self.assertEqual(normalise_text(42), "42")


class TestDuplicateRate(unittest.TestCase):

    def test_empty_is_zero(self):
        self.assertEqual(duplicate_rate([]), 0.0)

    def test_all_unique_is_zero(self):
        self.assertEqual(duplicate_rate(["a", "b", "c"]), 0.0)

    def test_all_same_approaches_one(self):
        self.assertAlmostEqual(duplicate_rate(["a"] * 4), 0.75)

    def test_normalisation_counts_case_variants(self):
        self.assertAlmostEqual(duplicate_rate(["Water", "water", "WATER"]), 2 / 3)


class TestDedupPairs(unittest.TestCase):

    def test_keeps_first_occurrence(self):
        pairs = [{"anchor": "a", "positive": "b", "tag": 1},
                 {"anchor": "a", "positive": "b", "tag": 2}]
        out = dedup_pairs(pairs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tag"], 1)

    def test_differing_positive_is_not_a_duplicate(self):
        pairs = [{"anchor": "a", "positive": "b"}, {"anchor": "a", "positive": "c"}]
        self.assertEqual(len(dedup_pairs(pairs)), 2)

    def test_case_and_spacing_are_duplicates(self):
        pairs = [{"anchor": "Water", "positive": "River"},
                 {"anchor": "water", "positive": "  river "}]
        self.assertEqual(len(dedup_pairs(pairs)), 1)


class TestDropDegeneratePairs(unittest.TestCase):

    def test_drops_identity_pairs(self):
        """Untranslated segments come back as anchor == positive."""
        pairs = [{"anchor": "omi", "positive": "omi"}, {"anchor": "omi", "positive": "odo"}]
        out = drop_degenerate_pairs(pairs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["positive"], "odo")

    def test_drops_empty_sides(self):
        pairs = [{"anchor": "", "positive": "odo"},
                 {"anchor": "omi", "positive": "   "},
                 {"anchor": "omi", "positive": "odo"}]
        self.assertEqual(len(drop_degenerate_pairs(pairs)), 1)

    def test_strips_false_negative_but_keeps_pair(self):
        pairs = [{"anchor": "omi", "positive": "odo", "negative": "odo"}]
        out = drop_degenerate_pairs(pairs)
        self.assertEqual(len(out), 1)
        self.assertNotIn("negative", out[0])

    def test_keeps_a_genuine_negative(self):
        pairs = [{"anchor": "omi", "positive": "odo", "negative": "ounje"}]
        self.assertEqual(drop_degenerate_pairs(pairs)[0]["negative"], "ounje")

    def test_does_not_mutate_input(self):
        pairs = [{"anchor": "omi", "positive": "odo", "negative": "odo"}]
        drop_degenerate_pairs(pairs)
        self.assertIn("negative", pairs[0])

    def test_min_chars(self):
        pairs = [{"anchor": "a", "positive": "bb"}, {"anchor": "aaa", "positive": "bbb"}]
        self.assertEqual(len(drop_degenerate_pairs(pairs, min_chars=3)), 1)


class TestCleanPairs(unittest.TestCase):

    def test_dedups_and_drops_together(self):
        pairs = [
            {"anchor": "omi", "positive": "odo"},      # keep
            {"anchor": "omi", "positive": "odo"},      # duplicate
            {"anchor": "ile", "positive": "ile"},      # identity
            {"anchor": "",    "positive": "odo"},      # empty
        ]
        self.assertEqual(len(clean_pairs(pairs)), 1)

    def test_empty_input(self):
        self.assertEqual(clean_pairs([]), [])


class TestEmbeddingHealth(unittest.TestCase):

    def test_detects_collapse(self):
        collapsed = np.ones((8, 16), dtype=np.float32)
        health = embedding_health(collapsed)
        self.assertTrue(health["collapsed"])
        self.assertAlmostEqual(health["mean_similarity"], 1.0, places=5)

    def test_orthogonal_embeddings_are_healthy(self):
        health = embedding_health(np.eye(8, dtype=np.float32))
        self.assertFalse(health["collapsed"])
        self.assertLess(health["mean_similarity"], 0.5)

    def test_effective_dimensions_tracks_spread(self):
        spread = embedding_health(np.eye(8, dtype=np.float32))["effective_dimensions"]
        collapsed = embedding_health(
            np.ones((8, 8), dtype=np.float32) + np.eye(8) * 1e-6
        )["effective_dimensions"]
        self.assertGreater(spread, collapsed)

    def test_rejects_1d_input(self):
        with self.assertRaises(ValueError):
            embedding_health(np.zeros(8))

    def test_rejects_single_embedding(self):
        with self.assertRaises(ValueError):
            embedding_health(np.zeros((1, 4)))

    def test_unnormalised_input_is_handled(self):
        """Scaling a row must not change its cosine similarity to the others."""
        base = np.random.RandomState(0).randn(6, 5)
        scaled = base * np.array([[1.0], [10.0], [0.1], [5.0], [1.0], [2.0]])
        np.testing.assert_allclose(
            embedding_health(base)["mean_similarity"],
            embedding_health(scaled)["mean_similarity"],
            atol=1e-9,
        )


class TestQualityReport(unittest.TestCase):

    def _pairs(self):
        return [
            {"anchor": "omi",  "positive": "odo",  "negative": "ounje"},
            {"anchor": "omi",  "positive": "odo",  "negative": "ounje"},   # duplicate
            {"anchor": "ile",  "positive": "ile"},                          # identity
            {"anchor": "oja",  "positive": "eja",  "negative": "eja"},      # false negative
            {"anchor": "",     "positive": "iwe"},                          # empty
        ]

    def test_counts(self):
        r = quality_report(self._pairs())
        self.assertEqual(r["total_pairs"], 5)
        self.assertEqual(r["identity_pairs"], 1)
        self.assertEqual(r["empty_sides"], 1)
        self.assertEqual(r["false_negatives"], 1)

    def test_usable_is_never_more_than_total(self):
        r = quality_report(self._pairs())
        self.assertLessEqual(r["usable_pairs"], r["total_pairs"])
        self.assertLessEqual(r["unique_pairs"], r["total_pairs"])

    def test_usable_matches_clean_pairs(self):
        pairs = self._pairs()
        self.assertEqual(quality_report(pairs)["usable_pairs"], len(clean_pairs(pairs)))

    def test_encoder_section_only_with_embeddings(self):
        self.assertNotIn("encoder", quality_report(self._pairs()))
        r = quality_report(self._pairs(), embeddings=np.eye(4, dtype=np.float32))
        self.assertIn("encoder", r)

    def test_empty_input(self):
        r = quality_report([])
        self.assertEqual(r["total_pairs"], 0)
        self.assertEqual(r["usable_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
