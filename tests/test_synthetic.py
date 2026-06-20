"""
Synthetic-data tests. LLM-based generation is tested with a fake generator (no model
download); mining is exercised with a tiny torch encoder.
"""
import numpy as np
import pytest


def test_generate_hard_negatives_with_callable():
    from foundry import generate_hard_negatives
    pairs = [{"anchor": f"q{i}", "positive": f"p{i}"} for i in range(3)]
    gen = lambda prompts: [f"neg-{i}" for i in range(len(prompts))]   # fake LLM
    out = generate_hard_negatives(pairs, gen)
    assert all("negative" in p for p in out)
    assert out[0]["negative"] == "neg-0"
    assert out[0]["anchor"] == "q0" and out[0]["positive"] == "p0"   # originals preserved


def test_synthesize_pairs_with_callable():
    from foundry import synthesize_pairs
    passages = ["the sky is blue", "water is wet"]
    gen = lambda prompts: ["q1", "q2"]
    out = synthesize_pairs(passages, gen)
    assert out[0] == {"anchor": "q1", "positive": "the sky is blue"}


def test_generator_type_error():
    from foundry.synthetic import _as_callable
    with pytest.raises(TypeError):
        _as_callable(42)


def test_synthesize_parallel_with_callable():
    from foundry import synthesize_parallel
    src = ["hello", "world"]
    # fake translator: prefix the lang code
    tr = lambda texts, lang: [f"[{lang}] {t}" for t in texts]
    pairs = synthesize_parallel(src, tr, ["sw", "yo"])
    assert len(pairs) == 4                                  # 2 sentences × 2 languages
    assert {"anchor": "hello", "positive": "[sw] hello"} in pairs
    assert {"anchor": "world", "positive": "[yo] world"} in pairs


def test_exports():
    import foundry
    for n in ("load_generator", "generate_hard_negatives", "synthesize_pairs",
              "mine_hard_negatives", "llm_generate", "load_translator",
              "translate_texts", "synthesize_parallel"):
        assert n in foundry.__all__


# ── mining (torch) ──────────────────────────────────────────────────────────

class _MiningTorch:
    """Wrapped so the module imports without torch present."""
    @staticmethod
    def run():
        from types import SimpleNamespace
        import torch
        import torch.nn as nn
        from foundry import mine_hard_negatives

        class TinyEnc(nn.Module):
            def __init__(self, vocab=64, dim=16):
                super().__init__(); self.emb = nn.Embedding(vocab, dim)
            def forward(self, input_ids, attention_mask=None, **_):
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))

        class FakeTok:
            def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
                ids = []
                for t in texts:
                    rng = np.random.default_rng(abs(hash(t)) % (2**32))
                    ids.append(rng.integers(1, 64, 6).tolist())
                ids = torch.tensor(ids, dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        pairs = [{"anchor": f"a{i}", "positive": f"p{i}"} for i in range(6)]
        out = mine_hard_negatives(pairs, TinyEnc(), FakeTok(), batch_size=8)
        assert any("negative" in p for p in out)


def test_mine_hard_negatives():
    pytest.importorskip("torch")
    _MiningTorch.run()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
