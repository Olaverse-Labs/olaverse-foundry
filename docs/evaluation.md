# Evaluation harness

The evaluation harness measures how good a base encoder is, **head-to-head** with other models. It fine-tunes the same task head on each model with identical data and config, then prints an accuracy / macro-F1 / params table — so "better" is a table, not a vibe.

Each model is tokenised with **its own** tokenizer (the fair thing to do), so you pass **raw rows**, not pre-tokenised ids. Metrics are pure numpy (no sklearn).

This page covers **classification / NER** evaluation. For embedding models, see [Retrieval evaluation](retrieval.md) (nDCG / Recall, `compare_retrievers`).

```bash
pip install "olaverse-foundry[torch]"
```

---

## Compare several encoders

```python
from foundry import compare_encoders, print_comparison

train_rows = [{"text": r["text"], "label": r["label"]} for r in ds["train"]]
eval_rows  = [{"text": r["text"], "label": r["label"]} for r in ds["validation"]]

results = compare_encoders(
    {"My Base": "./my-base",
     "mBERT":   "google-bert/bert-base-multilingual-cased",
     "e5-base": "intfloat/multilingual-e5-base"},
    train_rows, eval_rows, num_labels=NUM, task="sequence",
)
print_comparison(results)
```

```
  model         accuracy   macro_f1   params(M)
  ──────────────────────────────────────────────
  My Base           0.82       0.79        30.0
  mBERT             0.80       0.77       178.0
  e5-base           0.83       0.80       278.0
```

For NER, pass rows of `{"tokens": [...], "ner_tags": [...]}` and `task="token"`:

```python
ner_results = compare_encoders(
    models, ner_train, ner_eval, num_labels=TAGS, task="token",
    tokens_key="tokens", tags_key="ner_tags",
)
```

---

## `evaluate_encoder`

Score a single model:

```python
evaluate_encoder(
    base, train_rows, eval_rows, num_labels, task="sequence",
    *, tokenizer=None, text_key="text", label_key="label",
    tokens_key="tokens", tags_key="ner_tags",
    max_length=128, batch_size=16, config=None,
)
# → {"accuracy": ..., "macro_f1": ..., "params_m": ...}
```

`base` may be a model id, a local path, or a ready model object (pass `tokenizer=` for the latter). `config` is an optional [`HeadTrainConfig`](training/heads.md) (its `num_labels` is overridden).

---

## `compare_encoders`

```python
compare_encoders(models, train_rows, eval_rows, num_labels, task="sequence", **kw)
# → {name: {"accuracy", "macro_f1", "params_m"}}
```

`models` is a `{name: base}` dict or a list of base ids/paths. Extra kwargs pass through to `evaluate_encoder`. If one model fails (e.g. can't be loaded), it's recorded as `NaN` rather than sinking the whole table.

---

## `macro_f1`

```python
macro_f1(preds, labels, num_labels) -> float
```

Unweighted mean F1 over the classes that appear in `labels` (numpy arrays). `print_comparison(results, sort_by="macro_f1")` renders the table, best first.
