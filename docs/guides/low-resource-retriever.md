# Guide: a retriever for a low-resource language

**Goal:** semantic search in a language with little or no training data — no query–passage pairs, maybe no parallel text at all.

**The route:** manufacture the training pairs (translate → mine hard negatives) → contrastive training → benchmark against e5/LaBSE. This is the exact pipeline the [synthetic data](../synthetic.md) module was built for.

**You need:** a GPU, a multilingual base encoder (yours, or any HF encoder), an English (or other high-resource) corpus in your domain, and a small hand-labeled retrieval set for evaluation — even 50–100 queries with judged passages is enough to rank models.

---

## 1. Manufacture parallel pairs

No data in the target language? Translate your high-resource corpus with an open MT model. MADLAD-400 covers 400+ languages — far better at genuinely low-resource ones than any general LLM — and is Apache-licensed, so the output is commercially clean.

```python
from foundry import load_translator, synthesize_parallel

tr = load_translator("google/madlad400-3b-mt", device="auto", dtype="bfloat16")

pairs = synthesize_parallel(
    english_sentences, tr,
    target_langs=["yo"],                  # ISO codes: "yo", "sw", "ha", ...
    batch_size=32, max_new_tokens=128,
)
# → [{"anchor": english, "positive": yoruba_translation}, ...]
```

Cross-lingual anchor/positive pairs are exactly what aligns a multilingual embedding space: the model learns to place a sentence and its translation at the same point.

!!! note "Already have pairs?"
    Skip to step 2. And if your language is high-resource enough that an LLM writes it well, [`synthesize_pairs` / `generate_hard_negatives`](../synthetic.md#llm-generation-high-resource-languages) can generate query↔passage pairs directly.

## 2. Mine hard negatives

In-batch negatives (step 3) are random — easy to tell apart. Retrieval quality comes from **hard** negatives: passages that look right but aren't. For low-resource languages, don't ask an LLM — mine them with the encoder itself:

```python
from transformers import AutoModel, AutoTokenizer
from foundry import mine_hard_negatives

model = AutoModel.from_pretrained("./my-base")     # or any multilingual encoder
tok   = AutoTokenizer.from_pretrained("./my-base")

pairs = mine_hard_negatives(pairs, model, tok, device="cuda",
                            batch_size=64, skip_top=1)
# each pair gains a "negative": the most confusable *other* positive
```

`skip_top=1` skips the single highest-scoring candidate — it's often a near-duplicate of the true answer, and training against near-duplicates teaches the model the wrong lesson.

## 3. Contrastive training

InfoNCE with in-batch negatives — the e5/bge/LaBSE recipe. Every other positive in the batch is a free negative, and the mined `"negative"` fields join the candidate pool automatically:

```python
from foundry import ContrastiveTrainer, ContrastiveConfig

trainer = ContrastiveTrainer(model, tok, ContrastiveConfig(
    batch_size=64,            # in-batch negatives = batch_size − 1: bigger is better
    temperature=0.05,
    epochs=1, lr_scheduler="cosine", warmup_steps=100,
    device="cuda", torch_dtype="bfloat16",
))
result = trainer.train(pairs)
print(f"loss: {result['losses'][0]:.3f} -> {result['losses'][-1]:.3f}")

model.save_pretrained("./my-retriever")
tok.save_pretrained("./my-retriever")
```

If you hit CUDA OOM, cut `max_length` before you cut `batch_size` — batch size is where the training signal comes from.

## 4. Benchmark it

Score against the strong multilingual baselines on *your* evaluation set. Each model is encoded with its own tokenizer, pooling, and prefixes (e5's `query:`/`passage:` prefixes are applied automatically):

```python
from foundry import compare_retrievers, print_retrieval_comparison

# queries: list[str] · corpus: list[str] · qrels[i]: set of corpus indices relevant to query i
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
  My Retriever      0.8305      0.9040        30.0
  e5-base           0.8412      0.9120       278.0
  LaBSE             0.7980      0.8760       470.8
```

The baselines saw little or none of your language during their training — a small model trained on synthesized in-language pairs closing the gap to e5 (at a tenth of the size) is a typical outcome, and beating the baselines outright is realistic when the domain is narrow.

## 5. Serve it

`./my-retriever` is a standard HF directory. At query time, embed with the same pooling you trained with:

```python
from foundry import encode_texts

corpus_emb = encode_texts(model, tok, corpus, pool="mean", device="cuda")   # index once
q_emb      = encode_texts(model, tok, ["ibeere kan"], pool="mean", device="cuda")
scores     = q_emb @ corpus_emb.T                                            # cosine (normalised)
```

---

## Where to go from here

- Iterate on data, not knobs: more/cleaner synthetic pairs move nDCG more than any hyperparameter. [Synthetic data →](../synthetic.md)
- Distil a bigger retriever into yours first, then contrastive-tune: [`EmbeddingDistillTrainer`](../training/embed.md)
- Every metric and helper used above: [Retrieval evaluation →](../retrieval.md)
