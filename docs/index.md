<div class="ov-hero">
  <div class="ov-hero-badge">v0.1.0 — First Release</div>
  <h1 class="ov-hero-title">olaverse-foundry</h1>
  <p class="ov-hero-sub">Build model families from a single pretrained seed</p>
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

`olaverse-foundry` is the model-building layer of the Olaverse ecosystem. Where [`olaverse`](https://Olaverse-Labs.github.io/olaverse/) gives you **ready-to-use** African NLP models, **foundry** lets you **build** new ones — distilling large teachers into small students, growing models to new sizes, fusing capabilities, and packaging skills as detachable adapters.

```
seed → grow → distil / fuse → freeze → skill packs
```

> **Looking for ready-to-use models?** Head to the [Olaverse SDK docs](https://Olaverse-Labs.github.io/olaverse/) for LIDLite5, DiacNet, MIST, LegalPeace, and more.

---

## Core concepts

<div class="ov-grid">

<div class="ov-card">
  <div class="ov-card-icon">🔥</div>
  <div class="ov-card-title">Distillation</div>
  <div class="ov-card-body">Transfer knowledge from one or many teacher models into a smaller student. Supports CE+KL loss, logit caching, embedding alignment, and mixed precision — all from a single config.</div>
  <a href="training/" class="ov-card-link">Explore Training →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📦</div>
  <div class="ov-card-title">DataPipeline</div>
  <div class="ov-card-body">Unified dataset adapter that accepts HuggingFace datasets (including streaming), plain text lists, and numpy arrays. One API for <code>lm</code> and <code>embed</code> modes with reservoir shuffle.</div>
  <a href="data/" class="ov-card-link">Explore DataPipeline →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🧩</div>
  <div class="ov-card-title">Skill Packs</div>
  <div class="ov-card-body">Detachable LoRA adapters bound to a base model hash. Apply math, code, or reasoning skills to any compatible base. PEFT-format round-trip included.</div>
  <a href="skillpacks/" class="ov-card-link">Explore Skill Packs →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📐</div>
  <div class="ov-card-title">Growth & Fusion</div>
  <div class="ov-card-body">SOLAR-style depth up-scaling via layer duplication. Generates mergekit-compatible YAML. Vocabulary alignment for cross-tokenizer knowledge transfer.</div>
  <a href="growth/" class="ov-card-link">Explore Growth →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">📋</div>
  <div class="ov-card-title">YAML Recipes</div>
  <div class="ov-card-body">Pydantic-validated recipe files that describe an entire model factory pipeline. Validate your plan before spending a single GPU hour.</div>
  <a href="recipes/" class="ov-card-link">Explore Recipes →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🖥️</div>
  <div class="ov-card-title">CLI</div>
  <div class="ov-card-body"><code>foundry doctor</code> checks your environment. <code>foundry plan</code> previews a recipe. <code>foundry run</code> executes it. No Python boilerplate needed.</div>
  <a href="cli/" class="ov-card-link">Explore CLI →</a>
</div>

</div>

---

## Install

```bash
# Core — schema validation, planning (no GPU required)
pip install olaverse-foundry

# GPU training
pip install "olaverse-foundry[torch]"

# LoRA skill packs
pip install "olaverse-foundry[torch,lego]"

# HuggingFace datasets support
pip install "olaverse-foundry[torch,data]"

# W&B experiment tracking
pip install "olaverse-foundry[torch,logging]"

# Everything
pip install "olaverse-foundry[all]"
```

---

## Trainer comparison

| Trainer | When to use | Multi-GPU | Teacher cost |
|---|---|---|---|
| `TorchDistillTrainer` | Single GPU, small-to-mid datasets | No | Every step |
| `CachedDistillTrainer` | Multi-epoch, large datasets | Yes (accelerate) | First pass only |
| `EmbeddingDistillTrainer` | Bi-encoder / reranker distillation | No | Every step |

All three trainers share the same production feature set: mixed precision, gradient accumulation, LR scheduler, reproducible seed, checkpoint save/resume, eval loop, and W&B / TensorBoard logging.

---

## Quick example

```python
import torch
import numpy as np
from foundry import TorchDistillTrainer, TorchTrainConfig
from foundry.teachers import ToyTeacher, TeacherRegistry

# A tiny student (replace with your real model)
student = torch.nn.Linear(16, 32)

# Teachers
teachers = TeacherRegistry([ToyTeacher(vocab=32)])

# Data — any iterable of (B, S) int arrays
data = [np.random.randint(0, 32, (4, 16)) for _ in range(50)]

trainer = TorchDistillTrainer(
    student  = student,
    teachers = teachers,
    config   = TorchTrainConfig(
        epochs       = 2,
        lr_scheduler = "cosine",
        warmup_steps = 5,
        torch_dtype  = "float32",
        save_every   = 25,
        save_dir     = "/tmp/my_run",
    ),
)

result = trainer.train(data)
print(result["losses"][-1])   # final loss
```

---

## Links

- **Main SDK** — [olaverse](https://Olaverse-Labs.github.io/olaverse/) — ready-to-use African NLP models
- **GitHub** — [Olaverse-Labs/olaverse-foundry](https://github.com/Olaverse-Labs/olaverse-foundry)
- **PyPI** — [pypi.org/project/olaverse-foundry](https://pypi.org/project/olaverse-foundry/)
- **Hugging Face** — [huggingface.co/olaverse](https://huggingface.co/olaverse)
