<div class="ov-hero">
  <div class="ov-hero-badge">v0.2.0</div>
  <h1 class="ov-hero-title">olaverse-foundry</h1>
  <p class="ov-hero-sub">A toolkit for building model families — pretrain, distil, grow, adapt, quantize, evaluate</p>
  <div class="ov-hero-install">
    <span class="ov-hero-install-label">pip install olaverse-foundry</span>
  </div>
  <div class="ov-hero-links">
    <a href="quickstart/" class="md-button md-button--primary">Quick Start</a>
    <a href="https://github.com/Olaverse-Labs/olaverse-foundry" class="md-button" target="_blank">GitHub</a>
    <a href="https://pypi.org/project/olaverse-foundry/" class="md-button" target="_blank">PyPI</a>
  </div>
</div>

---

## What is olaverse-foundry?

`olaverse-foundry` is a general-purpose toolkit for **building** transformer models — decoder (causal-LM) or encoder — from existing ones or from scratch. It works with any HuggingFace model and any of your own `nn.Module` architectures.

A single pipeline takes you end to end:

```
pretrain / distil  →  grow  →  add heads  →  quantize  →  evaluate  →  serve
```

Everything is **model-agnostic**: pass an HF `AutoModel*`, or your own module that returns `.logits` / `.last_hidden_state`, and the same trainers, growth, QAT, and eval tooling apply.

---

## What you can do

<div class="ov-grid">

<div class="ov-card">
  <div class="ov-card-icon">🔥</div>
  <div class="ov-card-title">Distillation</div>
  <div class="ov-card-body">Transfer knowledge from one or many teachers into a smaller student. CE+KL for causal LMs, pooled MSE/cosine for embeddings, and token-level hidden-state distillation for encoders. Logit caching and cross-tokenizer alignment included.</div>
  <a href="training/" class="ov-card-link">Explore Training →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📚</div>
  <div class="ov-card-title">Pretraining (MLM)</div>
  <div class="ov-card-body">Train an encoder backbone from scratch with masked-language-modeling — your own architecture, your own tokenizer, no teacher required.</div>
  <a href="training/mlm/" class="ov-card-link">Explore MLM →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🎯</div>
  <div class="ov-card-title">Task heads</div>
  <div class="ov-card-body">Fine-tune sequence- and token-classification heads on any base encoder. Full fine-tune or frozen-backbone (train only the head) so many heads share one encoder.</div>
  <a href="training/heads/" class="ov-card-link">Explore Heads →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📐</div>
  <div class="ov-card-title">Growth & scaling</div>
  <div class="ov-card-body">SOLAR-style depth up-scaling by duplicating layers — native (no external merge tool). The layer prefix is auto-detected, so it works on Llama, BERT, GPT-2, and more.</div>
  <a href="growth/" class="ov-card-link">Explore Growth →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🪶</div>
  <div class="ov-card-title">Quantization (QAT)</div>
  <div class="ov-card-body">Quantization-aware training with int8/int4 fake-quant, plus int8 weight export and a footprint report — keep accuracy on-device.</div>
  <a href="quantization/" class="ov-card-link">Explore QAT →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🧩</div>
  <div class="ov-card-title">Skill packs (LoRA)</div>
  <div class="ov-card-body">Detachable LoRA adapters bound to a base-model hash. Snap them onto a frozen base; PEFT-format round-trip included.</div>
  <a href="skillpacks/" class="ov-card-link">Explore Skill Packs →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📦</div>
  <div class="ov-card-title">DataPipeline</div>
  <div class="ov-card-body">One adapter for HF datasets (incl. streaming), text lists, dicts, and numpy. <code>lm</code> / <code>embed</code> modes, reservoir shuffle, and labels for head training.</div>
  <a href="data/" class="ov-card-link">Explore DataPipeline →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📊</div>
  <div class="ov-card-title">Evaluation</div>
  <div class="ov-card-body">Head-to-head model comparison: fine-tune the same head on each model and print an accuracy / macro-F1 / params table — "better" as a table, not a vibe.</div>
  <a href="evaluation/" class="ov-card-link">Explore Evaluation →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">⚡</div>
  <div class="ov-card-title">Inference</div>
  <div class="ov-card-body">Load any trained model for generation, optionally 4-bit/8-bit quantized, with an optional skill pack merged in.</div>
  <a href="inference/" class="ov-card-link">Explore Inference →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📋</div>
  <div class="ov-card-title">YAML recipes</div>
  <div class="ov-card-body">Pydantic-validated recipe files describing a whole build. Preview the plan before spending a GPU hour.</div>
  <a href="recipes/" class="ov-card-link">Explore Recipes →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🖥️</div>
  <div class="ov-card-title">CLI</div>
  <div class="ov-card-body"><code>foundry doctor</code> checks your environment, <code>foundry plan</code> previews a recipe, <code>foundry run</code> / <code>foundry embed</code> execute it.</div>
  <a href="cli/" class="ov-card-link">Explore CLI →</a>
</div>

</div>

---

## Install

```bash
# Core — schema validation, growth planning, recipe parsing (no GPU required)
pip install olaverse-foundry

# Real training + inference (torch, transformers, safetensors, accelerate)
pip install "olaverse-foundry[torch]"

# LoRA skill packs
pip install "olaverse-foundry[torch,lego]"

# HuggingFace dataset streaming
pip install "olaverse-foundry[torch,data]"

# Fast cross-tokenizer alignment (rapidfuzz)
pip install "olaverse-foundry[torch,align]"

# W&B experiment tracking
pip install "olaverse-foundry[torch,logging]"

# Everything
pip install "olaverse-foundry[all]"
```

Quantized inference additionally needs `bitsandbytes`; QAT and growth need only `[torch]`.

---

## Trainers at a glance

| Trainer | Builds | Teacher? | Notes |
|---|---|---|---|
| [`TorchDistillTrainer`](training/torch.md) | causal LM | yes (1+) | CE + KL, teachers run every step |
| [`CachedDistillTrainer`](training/cached.md) | causal LM | yes (1+) | caches teacher logits; multi-GPU via accelerate |
| [`EmbeddingDistillTrainer`](training/embed.md) | embedding model | yes | pooled MSE / cosine for bi-encoders / rerankers |
| [`MLMTrainer`](training/mlm.md) | encoder base | **no** | masked-LM pretraining from scratch |
| [`EncoderDistillTrainer`](training/encoder-distill.md) | encoder base | yes | token-level hidden-state distillation |
| [`SequenceClassificationTrainer`](training/heads.md) | classifier head | — | sequence labels; full or frozen backbone |
| [`TokenClassificationTrainer`](training/heads.md) | token head (NER) | — | token labels; full or frozen backbone |

Every trainer shares the same production feature set: mixed precision, gradient accumulation, LR scheduler with warmup, reproducible seed, checkpoint save/resume, auto-checkpoint, eval loop, OOM handling, and W&B / TensorBoard logging.

---

## Quick example — distil a causal LM

```python
import torch, numpy as np
from foundry import TorchDistillTrainer, TorchTrainConfig, TeacherRegistry
from foundry.teachers import ToyTeacher

student  = torch.nn.Linear(16, 32)                       # any model with .logits
teachers = TeacherRegistry([ToyTeacher(vocab=32)])       # or HFTeacher(...)
data     = [np.random.randint(0, 32, (4, 16)) for _ in range(50)]

trainer = TorchDistillTrainer(student, teachers, TorchTrainConfig(
    epochs=2, lr_scheduler="cosine", warmup_steps=5,
    save_every=25, save_dir="/tmp/run",
))
result = trainer.train(data)
print(result["losses"][-1])
```

---

## Links

- **GitHub** — [Olaverse-Labs/olaverse-foundry](https://github.com/Olaverse-Labs/olaverse-foundry)
- **PyPI** — [pypi.org/project/olaverse-foundry](https://pypi.org/project/olaverse-foundry/)
- **olaverse SDK** — [ready-to-use models](https://Olaverse-Labs.github.io/olaverse/)
