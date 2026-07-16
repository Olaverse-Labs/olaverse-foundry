# Retrieval evaluation

The retrieval harness scores an embedding model on (cross-lingual) retrieval — **nDCG@k** and **Recall@k** — and compares it head-to-head with other retrievers on the same data, each model encoded with **its own** tokenizer, pooling, and prefixes for a fair comparison. Metrics are pure numpy.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Compare several retrievers

```python
from foundry import compare_retrievers, print_retrieval_comparison

queries = ["how tall is kilimanjaro", ...]        # Nq strings
corpus  = ["Kilimanjaro rises 5,895 m ...", ...]  # Nc strings
qrels   = [{0}, {3, 7}, ...]                      # per query: set of relevant corpus indices

results = compare_retrievers(
    {"My Retriever": "./my-retriever",
     "e5-base":      "intfloat/multilingual-e5-base",
     "LaBSE":        "sentence-transformers/LaBSE"},
    queries, corpus, qrels, k=10, device="cuda",
)
print_retrieval_comparison(results, k=10)
```

```
  model            ndcg@10   recall@10   params(M)
  ─────────────────────────────────────────────────
  e5-base           0.8412      0.9120       278.0
  My Retriever      0.8305      0.9040        30.0
  LaBSE             0.7980      0.8760       470.8
```

Known model families are auto-configured (explicit `specs` always wins):

| Family | Pooling | Prefixes |
|---|---|---|
| e5 | mean | `"query: "` / `"passage: "` |
| bge | cls | — |
| LaBSE | cls | — |
| everything else | the `pool` argument (default `"mean"`) | — |

To override or add a model, pass `specs={"My Retriever": {"pool": "cls", "query_prefix": "q: "}}`. A failing model is recorded as `NaN` rather than sinking the whole table.

---

## `evaluate_retrieval`

Score embeddings you already have:

```python
from foundry import evaluate_retrieval

evaluate_retrieval(query_emb, corpus_emb, qrels, k=10)
# → {"ndcg@10": 0.83, "recall@10": 0.9}
```

`query_emb` is `(Nq, D)`, `corpus_emb` is `(Nc, D)` (both L2-normalised — ranking uses the dot product), and `qrels[i]` is the set of corpus indices relevant to query `i`. Queries with empty qrels are skipped.

`ndcg_at_k(ranked_rel, num_relevant, k)` and `recall_at_k(ranked_rel, num_relevant, k)` are also exported if you want the raw per-query metrics.

---

## `encode_texts`

No-grad batch encoding to numpy — the building block used everywhere:

```python
from foundry import encode_texts

emb = encode_texts(model, tokenizer, texts,
                   pool="mean",        # "mean" | "cls"
                   normalize=True,
                   max_length=128, batch_size=64,
                   device=None,        # default: the model's current device
                   prefix="")          # e.g. "query: " for e5
# → (N, D) numpy array
```

The model is moved to the target device, so inputs and weights never end up on different devices.

---

## Getting training pairs

To *train* the retriever see [ContrastiveTrainer](training/contrastive.md); to manufacture pairs and hard negatives when you have little or no data, see [Synthetic data](synthetic.md).
