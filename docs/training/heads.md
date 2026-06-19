# Task heads — classification & NER

Once you have a base encoder (from [`MLMTrainer`](mlm.md), [`EncoderDistillTrainer`](encoder-distill.md), or any HF encoder), you fine-tune **task heads** on top of it:

- **`SequenceClassificationTrainer`** — classification, language ID, moderation, sentiment, reranking-as-classification. Labels are `(B,)` class ids (or `(B, num_labels)` floats when `multi_label=True`).
- **`TokenClassificationTrainer`** — NER and other token-level tasks. Labels are `(B, S)` tag ids with `-100` at pad / sub-word positions (ignored in the loss).

Both are **model-agnostic**: they accept any model whose `forward(input_ids=..., attention_mask=...)` returns `.logits` — any HF `AutoModelForSequenceClassification` / `AutoModelForTokenClassification`, or your own head module.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Attach a head in one line

`build_encoder_with_head` bolts a fresh head onto a saved base (or model id):

```python
from foundry import build_encoder_with_head

clf = build_encoder_with_head("./my-base", num_labels=4, task="sequence")
ner = build_encoder_with_head("./my-base", num_labels=7, task="token")
```

---

## Sequence classification

```python
from foundry import (DataPipeline, SequenceClassificationTrainer,
                     HeadTrainConfig, build_encoder_with_head)

# rows carry a label; DataPipeline emits {"input_ids","attention_mask","labels"}
train = DataPipeline(train_rows, batch_size=16, max_length=128,
                     mode="embed", label_column="label")
ev    = DataPipeline(eval_rows,  batch_size=16, max_length=128,
                     mode="embed", label_column="label")

model   = build_encoder_with_head("./my-base", num_labels=NUM, task="sequence")
trainer = SequenceClassificationTrainer(model, HeadTrainConfig(
    num_labels=NUM, device="cuda", torch_dtype="bfloat16",
    epochs=3, lr_scheduler="cosine", warmup_steps=20, eval_every=50,
))
result = trainer.train(train, eval_dataset=ev)
print(result["eval_metrics"])    # {step: accuracy}
```

---

## Token classification (NER)

Word-level tags are aligned to sub-word tokens with `-100` on continuation / special positions so they are ignored in the loss. With `DataPipeline(label_column=...)`, list-valued labels are padded to `max_length` with `-100` automatically.

```python
from foundry import TokenClassificationTrainer

ner_train = DataPipeline(ner_rows, batch_size=16, max_length=128,
                         mode="embed", label_column="ner")   # each row: {"input_ids", "ner": [...]}
model   = build_encoder_with_head("./my-base", num_labels=TAGS, task="token")
trainer = TokenClassificationTrainer(model, HeadTrainConfig(num_labels=TAGS, device="cuda"))
trainer.train(ner_train, eval_dataset=ner_eval)
```

---

## Frozen backbone (shared encoder)

Set `freeze_backbone=True` to train **only the head** while the encoder stays frozen. Many heads can then share one base encoder / one forward pass — the small-footprint, on-device path.

```python
from foundry import freeze_backbone

# inside the config
HeadTrainConfig(num_labels=NUM, freeze_backbone=True, learning_rate=1e-3)

# or inspect/apply manually
model, n_trainable, n_frozen = freeze_backbone(model)
print(f"head: {n_trainable/1e3:.1f}K trainable | backbone: {n_frozen/1e6:.1f}M frozen")
```

`freeze_backbone` keeps trainable only the parameters whose names contain a head keyword (`classifier`, `score`, `head`).

---

## Config — `HeadTrainConfig`

| Field | Default | Description |
|---|---|---|
| `num_labels` | `2` | Number of classes / tags |
| `multi_label` | `False` | Sequence task → `BCEWithLogits` over `(B, num_labels)` labels |
| `freeze_backbone` | `False` | Train only the head |
| `pad_token_id` | `0` | Mask derivation when absent |
| `learning_rate` | `2e-5` | AdamW LR (use a higher LR when frozen) |
| `epochs` | `3` | Passes over the dataset |
| `device` / `torch_dtype` | `"auto"` / `"float32"` | Device & mixed precision |
| `grad_accumulation_steps` | `1` | Accumulate over N batches |
| `lr_scheduler` / `warmup_steps` | `"cosine"` / `0` | Schedule |
| `eval_every` | `0` | Eval cadence |
| `save_every` / `save_dir` | `0` / `""` | Auto-checkpoint |
| `seed` | `42` | Reproducibility |

---

## Results & prediction

`train()` returns `{"losses", "eval_losses", "eval_metrics", "device"}`, where `eval_metrics` maps optimizer step → accuracy. For metric collection (used by the [evaluation harness](../evaluation.md)), `trainer.predict(dataset)` returns flat `(preds, labels)` numpy arrays (token tasks drop `-100` positions).
