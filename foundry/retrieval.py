"""
Retrieval evaluation — nDCG / Recall for (cross-lingual) retrieval, and a head-to-head
comparison against other models.

Metrics are pure numpy. ``compare_retrievers`` encodes queries + corpus with each
model (its own tokenizer), ranks by cosine similarity, and scores against relevance
judgements (qrels) — so you can measure your retriever against bge-m3 / e5 / LaBSE on
the same data.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ── Metrics (numpy) ──────────────────────────────────────────────────────────

def ndcg_at_k(ranked_rel, num_relevant: int, k: int = 10) -> float:
    """nDCG@k from a 0/1 relevance list in *ranked* order (top result first)."""
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ranked_rel[:k]))
    ideal = min(num_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_rel, num_relevant: int, k: int = 10) -> float:
    """Recall@k — fraction of relevant docs found in the top k."""
    return (sum(ranked_rel[:k]) / num_relevant) if num_relevant > 0 else 0.0


def evaluate_retrieval(query_emb: np.ndarray, corpus_emb: np.ndarray,
                       qrels, k: int = 10) -> dict:
    """
    Score embeddings against relevance judgements.

    Args:
        query_emb:  (Nq, D) query embeddings.
        corpus_emb: (Nc, D) corpus embeddings.
        qrels:      list of length Nq; ``qrels[i]`` is the set of *corpus indices*
                    relevant to query i.
        k:          cutoff.

    Returns ``{"ndcg@k": ..., "recall@k": ...}``.
    """
    q = np.asarray(query_emb, dtype=np.float32)
    c = np.asarray(corpus_emb, dtype=np.float32)
    sims = q @ c.T                                  # (Nq, Nc) — embeddings are normalised
    ndcgs, recalls = [], []
    for qi in range(q.shape[0]):
        rel_set = set(int(x) for x in qrels[qi])
        if not rel_set:
            continue
        order  = np.argsort(-sims[qi])
        ranked = [1 if int(ci) in rel_set else 0 for ci in order]
        ndcgs.append(ndcg_at_k(ranked, len(rel_set), k))
        recalls.append(recall_at_k(ranked, len(rel_set), k))
    return {f"ndcg@{k}": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
            f"recall@{k}": round(float(np.mean(recalls)) if recalls else 0.0, 4)}


# ── Encoding ──────────────────────────────────────────────────────────────────

def _pool(hidden, mask, mode):
    if mode == "cls":
        return hidden[:, 0]
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def encode_texts(model, tokenizer, texts, pool: str = "mean", normalize: bool = True,
                 max_length: int = 128, batch_size: int = 64, device=None) -> np.ndarray:
    """Encode a list of strings → (N, D) numpy embeddings (no grad)."""
    import torch
    import torch.nn.functional as F
    dev = device if device is not None else next(model.parameters()).device
    model = model.to(dev)          # keep model + inputs on the same device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            emb = _pool(h, enc["attention_mask"], pool).float()
            if normalize:
                emb = F.normalize(emb, dim=-1)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0)


# ── Compare ────────────────────────────────────────────────────────────────────

def compare_retrievers(models, queries, corpus, qrels, k: int = 10,
                       pool: str = "mean", max_length: int = 128,
                       batch_size: int = 64, device: str = "cuda") -> dict:
    """
    Encode + score several models on the same retrieval set.

    ``models`` is a ``{name: model_id_or_path}`` dict. Returns
    ``{name: {ndcg@k, recall@k, params_m}}``. One model failing is recorded as NaN.
    """
    from transformers import AutoModel, AutoTokenizer
    results: dict[str, Any] = {}
    for name, base in (models.items() if isinstance(models, dict) else [(m, m) for m in models]):
        print(f"[retrieval] encoding + scoring: {name} …")
        try:
            tok = AutoTokenizer.from_pretrained(base)
            mdl = AutoModel.from_pretrained(base).to(device)
            q_emb = encode_texts(mdl, tok, queries, pool, True, max_length, batch_size, device)
            c_emb = encode_texts(mdl, tok, corpus,  pool, True, max_length, batch_size, device)
            m = evaluate_retrieval(q_emb, c_emb, qrels, k)
            m["params_m"] = round(sum(p.numel() for p in mdl.parameters()) / 1e6, 1)
            results[name] = m
            del mdl
        except Exception as exc:
            print(f"[retrieval]   {name} failed: {exc}")
            results[name] = {f"ndcg@{k}": float("nan"), f"recall@{k}": float("nan"),
                             "params_m": float("nan"), "error": str(exc)}
    return results


def print_retrieval_comparison(results: dict, k: int = 10) -> None:
    """Pretty-print the retrieval comparison, best nDCG first."""
    key = f"ndcg@{k}"
    def sort_key(kv):
        v = kv[1].get(key)
        return v if isinstance(v, (int, float)) and v == v else -1.0
    rows = sorted(results.items(), key=sort_key, reverse=True)
    w = max((len(n) for n in results), default=5) + 2
    print()
    print(f"  {'model':{w}}  {key:>9}  {'recall@'+str(k):>10}  {'params(M)':>10}")
    print("  " + "─" * (w + 35))
    for name, m in rows:
        print(f"  {name:{w}}  {str(m.get(key)):>9}  {str(m.get('recall@'+str(k))):>10}  "
              f"{str(m.get('params_m')):>10}")
    print()
