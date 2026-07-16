# ContrastiveTrainer — InfoNCE retrieval training

`ContrastiveTrainer` turns an encoder into an **embedding model for retrieval** using the InfoNCE / MultipleNegativesRanking loss — the recipe behind e5, bge-m3, and LaBSE. It trains on **pairs**: for each anchor, its positive is the correct match and every *other* positive in the batch acts as an in-batch negative.

For **cross-lingual** retrieval, make anchor and positive different languages (parallel sentences) to align the multilingual space, and/or use query↔passage pairs for retrieval quality. Bigger `batch_size` = more in-batch negatives = better.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Contracts

- **model** — `model(input_ids=..., attention_mask=...)` returns `.last_hidden_state` (any HF `AutoModel` encoder).
- **tokenizer** — required; pairs are raw text and the trainer tokenises them.
- **dataset** — an iterable of dicts: `{"anchor": text, "positive": text}`, optionally with a `"negative"` key holding a hard negative (see [Synthetic data](../synthetic.md) for ways to get one).

---

## Quick start

```python
from transformers import AutoModel, AutoTokenizer
from foundry import ContrastiveTrainer, ContrastiveConfig

model = AutoModel.from_pretrained("./my-base")
tok   = AutoTokenizer.from_pretrained("./my-base")

pairs = [
    {"anchor": "How tall is Kilimanjaro?", "positive": "Kilimanjaro rises 5,895 m above sea level."},
    {"anchor": "Mlima Kilimanjaro una urefu gani?", "positive": "Kilimanjaro rises 5,895 m above sea level."},
    # ... {"anchor", "positive"[, "negative"]} dicts
]

trainer = ContrastiveTrainer(model, tok, ContrastiveConfig(
    batch_size=64,               # in-batch negatives = batch_size − 1
    temperature=0.05,
    epochs=1, lr_scheduler="cosine", warmup_steps=100,
    device="cuda", torch_dtype="bfloat16",
))
result = trainer.train(pairs)
print(result["losses"][-1])

model.save_pretrained("./my-retriever")
tok.save_pretrained("./my-retriever")
```

Evaluate the result with [`evaluate_retrieval` / `compare_retrievers`](../retrieval.md).

---

## Loss

For a batch of B pairs, anchors and candidates are encoded, pooled (`pool="mean"` or `"cls"`), optionally L2-normalised, and scored:

```
scores = (A @ candidates.T) / temperature      # (B, B) — or (B, 2B) with hard negatives
loss   = cross_entropy(scores, diagonal labels)
```

If **every** pair in the batch has a non-empty `"negative"`, the negatives are encoded too and appended to the candidate pool (`(B, 2B)` scores) — each hard negative competes with the true positive for every anchor.

Batches smaller than 2 pairs are skipped (in-batch negatives need at least one other pair).

---

## `encode`

```python
emb = trainer.encode(["some text", "more text"])   # (N, D) tensor, with grad
```

Tokenises + encodes with the configured pooling and normalisation. For inference-time (no-grad, numpy) encoding use [`foundry.encode_texts`](../retrieval.md#encode_texts).

---

## Config — `ContrastiveConfig`

| Field | Default | Description |
|---|---|---|
| `pool` | `"mean"` | `"mean"` or `"cls"` pooling |
| `temperature` | `0.05` | Scales the similarity logits |
| `normalize` | `True` | L2-normalise embeddings (cosine similarity) |
| `batch_size` | `32` | Pairs per batch; in-batch negatives = `batch_size − 1` |
| `max_length` | `128` | Tokenizer truncation length |
| `anchor_key` / `positive_key` / `negative_key` | `"anchor"` / `"positive"` / `"negative"` | Dict keys in your pairs |
| `learning_rate` | `2e-5` | AdamW LR |
| `epochs` | `1` | Passes over the dataset |
| `weight_decay` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `device` | `"auto"` | Device selection |
| `grad_accumulation_steps` | `1` | Accumulate over N batches |
| `torch_dtype` | `"float32"` | Mixed precision |
| `lr_scheduler` / `warmup_steps` | `"cosine"` / `0` | Schedule |
| `save_every` / `save_dir` | `0` / `""` | Auto-checkpoint |
| `log_every` / `log_backend` | `50` / `"none"` | Logging |
| `seed` | `42` | Reproducibility |

`train(dataset, on_step=None, shuffle=True, total_steps=None)` returns `{"losses", "device"}`. `save_checkpoint` / `resume_from_checkpoint` persist model + optimizer state, same as the other trainers.
