"""
End-to-end integration on CPU, with real ``transformers`` models.

Every other test in this suite runs against hand-rolled ``nn.Module`` stubs
(``TinyLM``, ``TinyEncoder``, ``ToyTeacher``). Those prove the training maths,
but they never touch ``AutoModel``, ``AutoTokenizer``, ``config.json``,
``save_pretrained`` or the safetensors round-trip — so nothing here verified the
claim the README leads with: that foundry output is *"a standard HuggingFace
directory that production code loads with transformers alone"*.

These tests build genuine ``BertModel`` / ``BertTokenizerFast`` instances from
local configs and a local vocab, save them as real HF directories, and run the
flagship pipeline over them:

    synthesise pairs → mine hard negatives → contrastive train
                     → encode → evaluate retrieval → save → reload

No network, no Hub download, no GPU. The models are ~25k params, so the whole
file runs in seconds.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


# ── A real, tiny, locally-built HF encoder ─────────────────────────────────

# Enough real words that the synthetic retrieval task below is actually
# learnable; the rest is filler to give the tokenizer a vocabulary.
TOPIC_WORDS = [
    "water", "river", "rain", "well",
    "food", "market", "maize", "yam",
    "school", "teacher", "book", "class",
    "road", "bus", "bridge", "town",
]
VOCAB = (
    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    + TOPIC_WORDS
    + [f"tok{i}" for i in range(96)]
)


def build_tokenizer(vocab: list[str]):
    """
    A real ``PreTrainedTokenizerFast`` over an explicit word-level vocabulary.

    Built with the ``tokenizers`` library rather than ``BertTokenizer(vocab_file=…)``:
    transformers 5.x no longer populates a Bert vocabulary from a plain vocab.txt,
    so that path silently yields an all-[UNK] tokenizer — which looks like model
    collapse downstream and is very hard to diagnose from a failing assertion.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    ids = {word: i for i, word in enumerate(vocab)}
    backend = Tokenizer(WordLevel(vocab=ids, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )


def build_encoder_dir(path: Path, hidden: int = 32, layers: int = 2,
                      seed: int = 0) -> Path:
    """Write a genuine HF encoder directory (config + safetensors + tokenizer)."""
    from transformers import BertConfig, BertModel

    path.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(VOCAB)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=hidden * 2,
        max_position_embeddings=64,
    )
    torch.manual_seed(seed)
    model = BertModel(config)
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


def topic_pairs(n_per_topic: int = 6) -> list[dict]:
    """Pairs whose anchor and positive share a topic word — trivially learnable."""
    topics = [TOPIC_WORDS[i:i + 4] for i in range(0, len(TOPIC_WORDS), 4)]
    pairs = []
    for topic in topics:
        head = topic[0]
        for i in range(n_per_topic):
            rest = topic[1:]
            pairs.append({
                "anchor":   f"{head} {rest[i % len(rest)]}",
                "positive": f"{rest[(i + 1) % len(rest)]} {head}",
            })
    return pairs


class RealModelTestCase(unittest.TestCase):
    """Builds one real encoder directory per test class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.enc_dir = build_encoder_dir(cls.root / "encoder")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def load(self):
        from transformers import AutoModel, AutoTokenizer
        return (AutoModel.from_pretrained(str(self.enc_dir)),
                AutoTokenizer.from_pretrained(str(self.enc_dir)))


# ── foundry.io.loader against a real on-disk model ─────────────────────────

class TestLoaderOnRealDirectory(RealModelTestCase):

    def test_load_model_from_local_path(self):
        from foundry.io.loader import ModelRef, load_model
        model = load_model(ModelRef.parse(str(self.enc_dir)))
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_load_tokenizer_from_local_path(self):
        from foundry.io.loader import ModelRef, load_tokenizer
        tok = load_tokenizer(ModelRef.parse(str(self.enc_dir)))
        self.assertIn("input_ids", tok("water river"))

    def test_written_directory_is_hf_standard(self):
        names = {p.name for p in self.enc_dir.iterdir()}
        self.assertIn("config.json", names)
        self.assertIn("tokenizer_config.json", names)
        self.assertTrue({"model.safetensors", "pytorch_model.bin"} & names)
        cfg = json.loads((self.enc_dir / "config.json").read_text())
        self.assertEqual(cfg["model_type"], "bert")


# ── encode_texts on a real model ───────────────────────────────────────────

class TestEncodeRealModel(RealModelTestCase):

    def test_shape_and_normalisation(self):
        from foundry.retrieval import encode_texts
        model, tok = self.load()
        emb = encode_texts(model, tok, ["water river", "school book", "road bus"],
                           pool="mean", normalize=True, device="cpu")
        self.assertEqual(emb.shape[0], 3)
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)

    def test_cls_pooling_differs_from_mean(self):
        from foundry.retrieval import encode_texts
        model, tok = self.load()
        texts = ["water river well", "market maize yam"]
        mean = encode_texts(model, tok, texts, pool="mean", device="cpu")
        cls  = encode_texts(model, tok, texts, pool="cls",  device="cpu")
        self.assertFalse(np.allclose(mean, cls))

    def test_encode_is_deterministic(self):
        from foundry.retrieval import encode_texts
        model, tok = self.load()
        a = encode_texts(model, tok, ["water river"], device="cpu")
        b = encode_texts(model, tok, ["water river"], device="cpu")
        np.testing.assert_allclose(a, b, atol=1e-6)


# ── hard-negative mining with a real encoder ───────────────────────────────

class TestMineHardNegativesReal(RealModelTestCase):

    def test_every_pair_gets_a_negative(self):
        from foundry.synthetic import mine_hard_negatives
        model, tok = self.load()
        pairs = topic_pairs(4)
        mined = mine_hard_negatives(pairs, model, tok, device="cpu")
        self.assertEqual(len(mined), len(pairs))
        self.assertTrue(all("negative" in p for p in mined))

    def test_negative_is_never_the_positive(self):
        from foundry.synthetic import mine_hard_negatives
        model, tok = self.load()
        mined = mine_hard_negatives(topic_pairs(4), model, tok, device="cpu")
        for p in mined:
            self.assertNotEqual(p["negative"], p["positive"])

    def test_originals_are_not_mutated(self):
        from foundry.synthetic import mine_hard_negatives
        model, tok = self.load()
        pairs = topic_pairs(4)
        mine_hard_negatives(pairs, model, tok, device="cpu")
        self.assertTrue(all("negative" not in p for p in pairs))


# ── retrieval evaluation ───────────────────────────────────────────────────

class TestEvaluateRetrievalReal(RealModelTestCase):

    def test_metrics_are_in_range(self):
        from foundry.retrieval import encode_texts, evaluate_retrieval
        model, tok = self.load()
        pairs   = topic_pairs(3)
        queries = [p["anchor"] for p in pairs]
        corpus  = [p["positive"] for p in pairs]
        q = encode_texts(model, tok, queries, device="cpu")
        c = encode_texts(model, tok, corpus,  device="cpu")
        res = evaluate_retrieval(q, c, [[i] for i in range(len(pairs))], k=5)
        for key, val in res.items():
            self.assertGreaterEqual(val, 0.0, key)
            self.assertLessEqual(val, 1.0, key)

    def test_perfect_retrieval_scores_one(self):
        """A corpus identical to the queries must rank each query first."""
        from foundry.retrieval import encode_texts, evaluate_retrieval
        model, tok = self.load()
        texts = [p["anchor"] for p in topic_pairs(3)]
        e = encode_texts(model, tok, texts, device="cpu")
        res = evaluate_retrieval(e, e, [[i] for i in range(len(texts))], k=5)
        self.assertAlmostEqual(res["ndcg@5"], 1.0, places=5)


# ── Contrastive training on a real HF encoder ──────────────────────────────

class TestContrastiveTrainingReal(RealModelTestCase):
    """Trains once for the whole class — a real BertModel on CPU is not free."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from foundry.retrieval import encode_texts
        from foundry.training.contrastive import ContrastiveConfig, ContrastiveTrainer

        cls.texts = [p["anchor"] for p in topic_pairs(6)]
        fresh, tok = cls.load(cls)
        cls.before = encode_texts(fresh, tok, cls.texts, device="cpu")

        cfg = ContrastiveConfig(device="cpu", epochs=15, batch_size=8,
                                learning_rate=5e-4, max_length=16, seed=0)
        cls.model, cls.tok = fresh, tok
        cls.out = ContrastiveTrainer(fresh, tok, config=cfg).train(topic_pairs(6))

    def test_loss_decreases(self):
        losses = self.out["losses"]
        first = sum(losses[:5]) / 5
        last  = sum(losses[-5:]) / 5
        self.assertLess(last, first, "loss did not fall: %.4f -> %.4f" % (first, last))

    def test_losses_are_finite(self):
        self.assertTrue(all(np.isfinite(x) for x in self.out["losses"]))

    def test_training_changes_the_embeddings(self):
        """Weights must actually move: a loss curve alone does not prove that.

        Deliberately not asserting a *direction* here. On this 4-topic task
        InfoNCE clusters the embeddings tighter rather than spreading them, so
        an assertion like "similarity goes down" holds only for some RNG states.
        Genuine collapse is covered by TestCollapseDetectionReal.
        """
        from foundry.retrieval import encode_texts
        after = encode_texts(self.model, self.tok, self.texts, device="cpu")
        self.assertFalse(np.allclose(after, self.before, atol=1e-4))


# ── The README's central claim: output is a standard HF directory ──────────

class TestHuggingFaceRoundTrip(RealModelTestCase):
    """
    "everything exits as a standard HuggingFace directory that production code
    loads with transformers alone" — README.

    Nothing else in the suite checks this, so it is checked here: train, save,
    then reload with nothing but transformers and confirm the weights survived.
    """

    def test_trained_model_reloads_with_transformers_alone(self):
        from transformers import AutoModel, AutoTokenizer
        from foundry.training.contrastive import ContrastiveConfig, ContrastiveTrainer

        model, tok = self.load()
        cfg = ContrastiveConfig(device="cpu", epochs=3, batch_size=8,
                                learning_rate=5e-4, max_length=16, seed=0)
        ContrastiveTrainer(model, tok, config=cfg).train(topic_pairs(4))

        out_dir = self.root / "trained"
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))

        reloaded = AutoModel.from_pretrained(str(out_dir))
        retok    = AutoTokenizer.from_pretrained(str(out_dir))

        for (name, a), (_, b) in zip(model.state_dict().items(),
                                     reloaded.state_dict().items()):
            torch.testing.assert_close(a, b, msg=f"weight drift in {name}")

        self.assertEqual(retok("water river")["input_ids"],
                         tok("water river")["input_ids"])

    def test_reloaded_model_encodes_identically(self):
        from transformers import AutoModel, AutoTokenizer
        from foundry.retrieval import encode_texts

        model, tok = self.load()
        out_dir = self.root / "roundtrip"
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))

        reloaded = AutoModel.from_pretrained(str(out_dir))
        retok    = AutoTokenizer.from_pretrained(str(out_dir))

        texts = ["water river", "school book"]
        np.testing.assert_allclose(
            encode_texts(model, tok, texts, device="cpu"),
            encode_texts(reloaded, retok, texts, device="cpu"),
            atol=1e-5,
        )


# ── compare_retrievers over real local model directories ──────────────────

class TestCompareRetrieversReal(RealModelTestCase):

    def test_benchmarks_two_real_models_on_cpu(self):
        """The head-to-head table the README sells, run for real on CPU."""
        from foundry.retrieval import compare_retrievers

        second = build_encoder_dir(self.root / "encoder_b", hidden=64, seed=7)
        pairs   = topic_pairs(3)
        queries = [p["anchor"] for p in pairs]
        corpus  = [p["positive"] for p in pairs]
        qrels   = [[i] for i in range(len(pairs))]

        results = compare_retrievers(
            {"small": str(self.enc_dir), "wider": str(second)},
            queries, corpus, qrels, k=5, device="cpu", max_length=16,
        )

        self.assertEqual(set(results), {"small", "wider"})
        for name, row in results.items():
            self.assertIn("params_m", row, name)
            self.assertFalse(np.isnan(row["params_m"]), f"{name} failed to load")
            self.assertGreater(row["params_m"], 0.0, name)

    def test_wider_model_reports_more_params(self):
        from foundry.retrieval import compare_retrievers

        second = build_encoder_dir(self.root / "encoder_c", hidden=64, seed=7)
        pairs   = topic_pairs(2)
        results = compare_retrievers(
            {"small": str(self.enc_dir), "wider": str(second)},
            [p["anchor"] for p in pairs], [p["positive"] for p in pairs],
            [[i] for i in range(len(pairs))], k=3, device="cpu", max_length=16,
        )
        self.assertGreater(results["wider"]["params_m"], results["small"]["params_m"])


# ── Collapse detection against a genuinely collapsed encoder ───────────────

class TestCollapseDetectionReal(RealModelTestCase):
    """
    An untrained encoder over an all-[UNK] tokenizer maps every input to the
    same vector. evaluate_retrieval still returns a plausible mid-range nDCG for
    it, because ranking identical vectors is decided by tie-breaking. That is
    the number a user would publish. embedding_health is what catches it.
    """

    def _collapsed_embeddings(self):
        from foundry.retrieval import encode_texts
        model, _ = self.load()
        # A tokenizer whose vocabulary shares nothing with the texts: every
        # input becomes the same [CLS] [UNK] [SEP] sequence.
        blind = build_tokenizer(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        texts = [p["anchor"] for p in topic_pairs(3)]
        return encode_texts(model, blind, texts, device="cpu"), texts

    def test_collapsed_encoder_is_flagged(self):
        from foundry.quality import embedding_health
        emb, _ = self._collapsed_embeddings()
        health = embedding_health(emb)
        self.assertTrue(health["collapsed"], health)
        self.assertLess(health["effective_dimensions"], 2.0)

    def test_collapse_scores_a_plausible_ndcg_anyway(self):
        """The bug this guards: a broken encoder does not score zero."""
        from foundry.retrieval import evaluate_retrieval
        emb, texts = self._collapsed_embeddings()
        res = evaluate_retrieval(emb, emb, [[i] for i in range(len(texts))], k=5)
        self.assertGreater(res["ndcg@5"], 0.0)
        self.assertLess(res["ndcg@5"], 1.0)

    def test_trained_encoder_is_not_flagged(self):
        from foundry.quality import embedding_health
        from foundry.retrieval import encode_texts
        from foundry.training.contrastive import ContrastiveConfig, ContrastiveTrainer

        model, tok = self.load()
        cfg = ContrastiveConfig(device="cpu", epochs=15, batch_size=8,
                                learning_rate=5e-4, max_length=16, seed=0)
        ContrastiveTrainer(model, tok, config=cfg).train(topic_pairs(6))

        texts = [p["anchor"] for p in topic_pairs(6)]
        health = embedding_health(encode_texts(model, tok, texts, device="cpu"))
        self.assertFalse(health["collapsed"], health)


if __name__ == "__main__":
    unittest.main()
