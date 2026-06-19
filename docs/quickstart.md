# Quick Start

End-to-end examples for the main workflows — causal-LM distillation, embedding distillation, and a full **encoder build** (base → heads → quantize → evaluate). They run on CPU for prototyping and scale to GPU with a single config change.

---

## Example 1 — Causal LM distillation

Distill a large instruction-tuned model into a smaller one using `TorchDistillTrainer`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from foundry import (
    DataPipeline,
    TorchDistillTrainer,
    TorchTrainConfig,
    TeacherRegistry,
    HFTeacher,
)

# ── 1. Load student ───────────────────────────────────────────────────
torch.manual_seed(42)   # set BEFORE model creation for full reproducibility
student = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    torch_dtype=torch.bfloat16,
)

# ── 2. Define teachers ────────────────────────────────────────────────
teachers = TeacherRegistry([
    HFTeacher("Qwen/Qwen2.5-7B-Instruct", weight=1.0),
    HFTeacher("Qwen/Qwen2.5-14B-Instruct", weight=0.7),
])
teachers.load_all()

# ── 3. Stream training data ───────────────────────────────────────────
from datasets import load_dataset

tok     = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
raw_ds  = load_dataset("allenai/c4", "en", split="train", streaming=True)

pipe = DataPipeline(
    source         = raw_ds,
    tokenizer      = tok,
    batch_size     = 8,
    max_length     = 512,
    mode           = "lm",
    shuffle_buffer = 20_000,
    text_column    = "text",
)

eval_ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
eval_pipe = DataPipeline(
    source      = eval_ds,
    tokenizer   = tok,
    batch_size  = 8,
    max_length  = 512,
    mode        = "lm",
)

# ── 4. Configure and train ────────────────────────────────────────────
trainer = TorchDistillTrainer(
    student  = student,
    teachers = teachers,
    config   = TorchTrainConfig(
        epochs                  = 1,
        learning_rate           = 2e-5,
        alpha                   = 0.3,           # CE weight; (1-alpha) = KL weight
        lr_scheduler            = "cosine",
        warmup_steps            = 500,
        torch_dtype             = "bfloat16",
        grad_accumulation_steps = 8,             # effective batch = 8 × 8 = 64
        max_grad_norm           = 1.0,
        eval_every              = 200,
        save_every              = 500,
        save_dir                = "/checkpoints/qwen-1.5b-distil",
        log_backend             = "wandb",
        project                 = "olaverse-foundry",
        run_name                = "qwen-1.5b-c4-run1",
        seed                    = 42,
    ),
)

result = trainer.train(
    pipe,
    eval_dataset = eval_pipe,
    total_steps  = 10_000,   # override for LR scheduler when source is streaming
)

print("Final loss:  ", result["losses"][-1])
print("Eval losses: ", result["eval_losses"])
print("Device:      ", result["device"])

# ── 5. Save ───────────────────────────────────────────────────────────
student.save_pretrained("/checkpoints/qwen-1.5b-distil/final")
tok.save_pretrained("/checkpoints/qwen-1.5b-distil/final")
```

---

## Example 2 — Embedding distillation

Distill a large bi-encoder teacher into a smaller student using `EmbeddingDistillTrainer`.

```python
import torch
from transformers import AutoModel, AutoTokenizer
from foundry import (
    DataPipeline,
    EmbeddingDistillTrainer,
    EmbeddingDistillConfig,
)

# ── 1. Models ─────────────────────────────────────────────────────────
torch.manual_seed(42)
student  = AutoModel.from_pretrained("microsoft/deberta-v3-base")
teacher  = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
tok      = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# ── 2. Data ───────────────────────────────────────────────────────────
from datasets import load_dataset

ds = load_dataset("sentence-transformers/natural-questions", split="train")

pipe = DataPipeline(
    source         = ds,
    tokenizer      = tok,
    batch_size     = 32,
    max_length     = 128,
    mode           = "embed",
    shuffle_buffer = 10_000,
    text_column    = "query",       # column in your dataset that has the text
)

eval_pipe = DataPipeline(
    source      = load_dataset("sentence-transformers/natural-questions", split="test"),
    tokenizer   = tok,
    batch_size  = 32,
    max_length  = 128,
    mode        = "embed",
    text_column = "query",
)

# ── 3. Train ──────────────────────────────────────────────────────────
trainer = EmbeddingDistillTrainer(
    student = student,
    teacher = teacher,
    config  = EmbeddingDistillConfig(
        loss                    = "cosine",     # "cosine" | "mse"
        pool                    = "mean",        # "mean" | "cls"
        normalize               = True,
        temperature             = 0.05,
        epochs                  = 3,
        learning_rate           = 2e-5,
        lr_scheduler            = "cosine",
        warmup_steps            = 200,
        torch_dtype             = "bfloat16",
        grad_accumulation_steps = 4,
        eval_every              = 500,
        save_every              = 1000,
        save_dir                = "/checkpoints/deberta-embed",
        log_backend             = "wandb",
        project                 = "olaverse-foundry",
        run_name                = "deberta-nq-embed",
        seed                    = 42,
    ),
)

result = trainer.train(pipe, eval_dataset=eval_pipe)
print("Eval losses:", result["eval_losses"])
```

---

## Example 3 — Resume from checkpoint

```python
trainer = TorchDistillTrainer(student=student, teachers=teachers, config=cfg)
trainer.resume_from_checkpoint("/checkpoints/qwen-1.5b-distil")

result = trainer.train(pipe)
```

---

## Example 4 — Multi-epoch with logit cache (CachedDistillTrainer)

When you have multiple epochs, use `CachedDistillTrainer` — teachers run once, then all subsequent epochs read from cache at zero cost.

```python
from foundry import CachedDistillTrainer, CachedDistillConfig

trainer = CachedDistillTrainer(
    student  = student,
    teachers = teachers,
    config   = CachedDistillConfig(
        epochs       = 5,
        cache_dir    = "/tmp/logit_cache",
        cache_top_k  = 64,
        torch_dtype  = "bfloat16",
        lr_scheduler = "cosine",
        warmup_steps = 500,
    ),
)

result = trainer.train(dataset)
print(result["cache_stats"])   # hits / misses per teacher
```

---

## Example 5 — Build an encoder: base → heads → quantize → evaluate

A full encoder workflow. Distil a small base from a strong teacher (with QAT so it's int8-ready), attach a classification head, then compare it head-to-head with other models.

### 5a. Build the base (QAT distillation)

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

!!! tip "Pretrain from scratch instead"
    To build the base without a teacher, swap 5a for [`MLMTrainer`](training/mlm.md): train an `AutoModelForMaskedLM` of your own architecture on raw text, then save its encoder body as `./my-base`.

### 5b. Add a classification head

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

For NER, use `task="token"` + `TokenClassificationTrainer` with `label_column="ner"` (token labels are padded with `-100`).

### 5c. Evaluate head-to-head

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

### 5d. Use it

The base and the head model are standard HuggingFace directories — load them with `transformers`:

```python
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# the base encoder → token / pooled representations
base = AutoModel.from_pretrained("./my-base")

# the fine-tuned classifier from 5b (after model.save_pretrained("./my-clf"))
clf  = AutoModelForSequenceClassification.from_pretrained("./my-clf")
tok  = AutoTokenizer.from_pretrained("./my-base")
```

For **causal-LM** generation use [`load_for_inference` / `generate`](inference.md) instead — those are decoder helpers.

---

## Next steps

- [Training reference →](training/index.md) — every trainer and config field
- [MLM pretraining →](training/mlm.md) · [Encoder distillation →](training/encoder-distill.md) · [Task heads →](training/heads.md)
- [Quantization (QAT) →](quantization.md) — int8/int4-aware training and export
- [Evaluation →](evaluation.md) — head-to-head model comparison tables
- [DataPipeline →](data.md) — streaming, shuffle, labels, custom columns
- [Skill Packs →](skillpacks.md) — attach LoRA adapters to a base
- [YAML Recipes →](recipes.md) — define the full pipeline in a single file
