# Guide: build a small classifier

**Goal:** a ~11M-parameter text classifier that holds its own against mBERT (178M) — small enough for cheap CPU serving, built from a strong public teacher.

**The route:** distil a base encoder (int8-aware from the start) → attach a classification head → benchmark it head-to-head. Four steps, each a few lines.

**You need:** a GPU for step 1 (a single consumer card is fine), raw text in your domain/language, and a labeled classification dataset for steps 2–3.

---

## 1. Build the base — QAT distillation

We don't pretrain from scratch — that needs billions of tokens. Instead we pick a strong multilingual encoder as **teacher** and distil its per-token representations into a small architecture we design. Wrapping the student with `prepare_qat` *before* training makes it quantization-aware, so the int8 export in this step costs almost no quality.

```python
import torch
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
from foundry import (
    DataPipeline, EncoderDistillTrainer, EncoderDistillConfig,
    prepare_qat, QATConfig, export_quantized,
)

# Teacher (any HF encoder) + its tokenizer; the student shares the vocab
teacher = AutoModel.from_pretrained("intfloat/multilingual-e5-base")
tok     = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

student = BertModel(BertConfig(
    vocab_size=tok.vocab_size, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1024, max_position_embeddings=512,
))
student = prepare_qat(student, QATConfig(weight_bits=8))   # int8-aware

pipe = DataPipeline(text_rows, tokenizer=tok, text_column="text",
                    batch_size=16, max_length=128, mode="embed")

EncoderDistillTrainer(student, teacher, EncoderDistillConfig(
    device="cuda", torch_dtype="bfloat16", epochs=1, loss="mse",
    lr_scheduler="cosine", warmup_steps=20,
)).train(pipe)

report = export_quantized(student, "./my-base", weight_bits=8)
tok.save_pretrained("./my-base")
print(report)            # {'orig_mb', 'quant_mb', 'compression', ...}
```

`text_rows` is your raw in-domain text — the more it looks like what the classifier will see in production, the better. Distillation is data-efficient; a few million tokens already gives a usable base.

!!! tip "Pretrain from scratch instead"
    No suitable teacher for your language? Swap this step for [`MLMTrainer`](../training/mlm.md): train an `AutoModelForMaskedLM` of your own architecture on raw text, then save its encoder body as `./my-base`. Expect to need much more text. A middle path — teacher **plus** raw text — is [`DistilMLMTrainer`](../training/distil-mlm.md).

## 2. Add a classification head

`build_encoder_with_head` bolts a fresh head onto the base; the trainer fine-tunes both (or just the head, if you freeze the backbone).

```python
from foundry import (DataPipeline, SequenceClassificationTrainer,
                     HeadTrainConfig, build_encoder_with_head)

train = DataPipeline(train_rows, batch_size=16, max_length=128,
                     mode="embed", label_column="label")   # rows: {"input_ids"/"text", "label"}
ev    = DataPipeline(eval_rows,  batch_size=16, max_length=128,
                     mode="embed", label_column="label")

model = build_encoder_with_head("./my-base", num_labels=NUM, task="sequence")
res = SequenceClassificationTrainer(model, HeadTrainConfig(
    num_labels=NUM, device="cuda", torch_dtype="bfloat16",
    epochs=3, lr_scheduler="cosine", warmup_steps=20, eval_every=50,
    # freeze_backbone=True,   # train only the head to share one frozen encoder
)).train(train, eval_dataset=ev)
print("accuracy:", res["eval_metrics"])
```

For NER, use `task="token"` + [`TokenClassificationTrainer`](../training/heads.md) with `label_column="ner"` (token labels are padded with `-100`).

## 3. Evaluate head-to-head

"Better" should be a table, not a vibe. `compare_encoders` fine-tunes the **same** head on each model with identical data and config — each model tokenised with its own tokenizer — and reports accuracy / macro-F1 / size:

```python
from foundry import compare_encoders, print_comparison

# raw rows: {"text": ..., "label": ...}
results = compare_encoders(
    {"My Base": "./my-base",
     "mBERT":   "google-bert/bert-base-multilingual-cased",
     "e5-base": "intfloat/multilingual-e5-base"},
    train_rows, eval_rows, num_labels=NUM, task="sequence",
)
print_comparison(results)
```

```
  model      accuracy   macro_f1   params(M)
  ───────────────────────────────────────────
  My Base        0.82       0.79        11.0
  mBERT          0.80       0.77       178.0
  e5-base        0.83       0.80       278.0
```

A result like this — within a point of models 16–25× larger — is the realistic target for a well-distilled base evaluated in its own domain.

## 4. Ship it

The base and the head model are standard HuggingFace directories — production code needs `transformers` only, not foundry:

```python
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# the base encoder → token / pooled representations
base = AutoModel.from_pretrained("./my-base")

# the fine-tuned classifier from step 2 (after model.save_pretrained("./my-clf"))
clf  = AutoModelForSequenceClassification.from_pretrained("./my-clf")
tok  = AutoTokenizer.from_pretrained("./my-base")
```

---

## Where to go from here

- More heads on the same base — set `freeze_backbone=True` and train each head separately; they all share one frozen encoder in memory. [Task heads →](../training/heads.md)
- Need retrieval instead of classification? → [Guide: low-resource retriever](low-resource-retriever.md)
- Every knob used above → [Config reference](../training/config.md)
