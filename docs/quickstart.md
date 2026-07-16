# Quick Start

## Your first distillation — 60 seconds, no GPU

Distil a tiny public teacher into an even smaller student of your own design. ~17 MB download, runs on a laptop:

```bash
pip install "olaverse-foundry[torch]"
```

```python
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
from foundry import DataPipeline, EncoderDistillTrainer, EncoderDistillConfig

teacher = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")   # BERT-tiny, 4M params
tok     = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

student = BertModel(BertConfig(                                 # your own, even smaller
    vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=2, intermediate_size=128,
))

texts = ["distillation copies a big model into a small one",
         "the student learns the teacher's representations",
         "this runs on a laptop in under a minute"] * 8

pipe    = DataPipeline(texts, tokenizer=tok, batch_size=4, max_length=32, mode="embed")
trainer = EncoderDistillTrainer(student, teacher, EncoderDistillConfig(epochs=30))
result  = trainer.train(pipe)

print(f"loss: {result['losses'][0]:.3f} -> {result['losses'][-1]:.3f} on {result['device']}")
```

```
loss: 1.443 -> 0.927 on mps
```

That falling loss is the student learning to reproduce the teacher's per-token representations. Every real workflow below is this same shape — pick a **trainer**, feed it a **`DataPipeline`**, read the **result dict** — just with bigger models, real data, and a GPU.

Not sure which trainer your goal needs? **[Which trainer do I need? →](choosing.md)** New to the terminology? **[Concepts & glossary →](concepts.md)**

---

The rest of this page is production-shaped examples: causal-LM distillation, embedding distillation, checkpoint resume, and multi-epoch caching. For complete narrated builds, see the guides: [small classifier](guides/small-classifier.md) · [low-resource retriever](guides/low-resource-retriever.md).

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

## Example 5 — Build an encoder from scratch

The full encoder workflow — distil a base with QAT, attach a classification head, evaluate it head-to-head against mBERT and e5 — has its own narrated guide:

**[Guide: build a small classifier →](guides/small-classifier.md)**

---

## Next steps

- [Which trainer do I need? →](choosing.md) — map your goal to the right tool
- [Concepts & glossary →](concepts.md) — every term the docs assume, in plain language
- Guides: [build a small classifier →](guides/small-classifier.md) · [low-resource retriever →](guides/low-resource-retriever.md)
- [Training reference →](training/index.md) — every trainer and config field
- [DataPipeline →](data.md) — streaming, shuffle, labels, custom columns
- [Quantization (QAT) →](quantization.md) · [Skill Packs →](skillpacks.md) · [YAML Recipes →](recipes.md)
