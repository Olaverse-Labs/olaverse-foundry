"""
Retrieval-eval metric tests — pure numpy, run anywhere.
"""
import numpy as np
import pytest


def test_ndcg_perfect():
    from foundry import ndcg_at_k
    # all relevant at the top
    assert ndcg_at_k([1, 1, 0, 0], num_relevant=2, k=10) == 1.0


def test_ndcg_worse_when_relevant_lower():
    from foundry import ndcg_at_k
    top    = ndcg_at_k([1, 0, 0, 0], 1, 10)
    bottom = ndcg_at_k([0, 0, 0, 1], 1, 10)
    assert top > bottom > 0


def test_recall_at_k():
    from foundry import recall_at_k
    assert recall_at_k([1, 0, 1, 0], num_relevant=3, k=4) == pytest.approx(2 / 3)
    assert recall_at_k([0, 0], num_relevant=0, k=2) == 0.0


def test_evaluate_retrieval_perfect():
    from foundry import evaluate_retrieval
    # query i is identical to corpus i → ranks itself first
    emb = np.eye(4, dtype=np.float32)
    qrels = [{0}, {1}, {2}, {3}]
    m = evaluate_retrieval(emb, emb, qrels, k=1)
    assert m["ndcg@1"] == 1.0
    assert m["recall@1"] == 1.0


def test_evaluate_retrieval_partial():
    from foundry import evaluate_retrieval
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    qrels = [{0}, {2}]
    m = evaluate_retrieval(q, c, qrels, k=2)
    assert 0.0 < m["ndcg@2"] <= 1.0


def test_exports():
    import foundry
    for n in ("ContrastiveTrainer", "ContrastiveConfig", "evaluate_retrieval",
              "compare_retrievers", "print_retrieval_comparison", "encode_texts"):
        assert n in foundry.__all__


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
