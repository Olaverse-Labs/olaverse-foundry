# Training

`olaverse-foundry` ships a family of production-ready trainers that share the same config base and feature set. They fall into two groups: **distillation** trainers (learn from teachers) and **encoder / head** trainers (pretrain a base and add task heads). Choose based on what you're building.

| Trainer | Builds | Page |
|---|---|---|
| `TorchDistillTrainer` | causal LM (CE+KL, teachers live) | [reference](torch.md) |
| `CachedDistillTrainer` | causal LM (cached logits, multi-GPU) | [reference](cached.md) |
| `EmbeddingDistillTrainer` | embedding model (pooled MSE/cosine) | [reference](embed.md) |
| `MLMTrainer` | encoder base from scratch (masked LM) | [reference](mlm.md) |
| `EncoderDistillTrainer` | encoder base (token-level distillation) | [reference](encoder-distill.md) |
| `DistilMLMTrainer` | encoder base (distillation + MLM combined, DistilBERT-style) | [reference](distil-mlm.md) |
| `ContrastiveTrainer` | retrieval embedding model (InfoNCE) | [reference](contrastive.md) |
| `SequenceClassificationTrainer` | classification / langID / moderation head | [reference](heads.md) |
| `TokenClassificationTrainer` | NER / token-classification head | [reference](heads.md) |

---

## Distillation trainers

<div class="ov-compare-grid">

<div class="ov-compare-card">
  <div class="ov-compare-title">TorchDistillTrainer</div>
  <div class="ov-compare-badge ov-badge-green">Single GPU</div>
  <ul>
    <li>CE + KL loss against one or many teachers</li>
    <li>Teachers run live every step</li>
    <li>Best for: short runs, small datasets, prototyping</li>
    <li><a href="torch/">Full reference →</a></li>
  </ul>
</div>

<div class="ov-compare-card">
  <div class="ov-compare-title">CachedDistillTrainer</div>
  <div class="ov-compare-badge ov-badge-purple">Multi-GPU via accelerate</div>
  <ul>
    <li>Teachers run once; cached to disk</li>
    <li>Subsequent epochs are free</li>
    <li>Best for: multi-epoch training on large datasets</li>
    <li><a href="cached/">Full reference →</a></li>
  </ul>
</div>

</div>

<div class="ov-compare-grid" style="margin-top:1rem">

<div class="ov-compare-card">
  <div class="ov-compare-title">EmbeddingDistillTrainer</div>
  <div class="ov-compare-badge ov-badge-green">Single GPU</div>
  <ul>
    <li>MSE / cosine loss on pooled sentence vectors</li>
    <li>For bi-encoder and reranker distillation</li>
    <li>Best for: embedding model compression</li>
    <li><a href="embed/">Full reference →</a></li>
  </ul>
</div>

<div class="ov-compare-card">
  <div class="ov-compare-title">Shared feature set</div>
  <div class="ov-compare-badge ov-badge-purple">All trainers</div>
  <ul>
    <li>Mixed precision (bfloat16 / float16)</li>
    <li>Gradient accumulation</li>
    <li>LR scheduler with linear warmup</li>
    <li>Checkpoint save / resume</li>
    <li>Eval loop &amp; W&B / TensorBoard logging</li>
  </ul>
</div>

</div>

---

## Shared production features

### Mixed precision

Set `torch_dtype` in any config:

```python
TorchTrainConfig(torch_dtype="bfloat16")   # recommended for A100/H100
TorchTrainConfig(torch_dtype="float16")    # for older GPUs
TorchTrainConfig(torch_dtype="float32")    # default — CPU safe
```

### Gradient accumulation

```python
TorchTrainConfig(
    batch_size              = 4,
    grad_accumulation_steps = 8,   # effective batch = 32
)
```

### LR scheduler

```python
TorchTrainConfig(
    lr_scheduler = "cosine",   # "constant" | "linear" | "cosine"
    warmup_steps = 500,
)
```

The scheduler wraps a `LambdaLR` over the optimizer. Linear warmup runs for `warmup_steps`, then the chosen schedule decays over the remaining steps. `"constant"` with no warmup returns `None` (no scheduler overhead).

### Reproducible seed

```python
# Set BEFORE creating your model for full reproducibility
import torch, numpy as np
torch.manual_seed(42)
np.random.seed(42)

student = MyModel()
trainer = TorchDistillTrainer(student, teachers, TorchTrainConfig(seed=42))
```

The trainer calls `_seed_everything()` at the start of `train()`, which re-seeds torch, numpy, and random. For the model weights to also be reproducible, seed before construction.

### Checkpoint save / resume

```python
# Manual save
trainer.save_checkpoint("/checkpoints/step_500")

# Auto-checkpoint every N optimizer steps
TorchTrainConfig(save_every=500, save_dir="/checkpoints/run1")

# Resume
trainer.resume_from_checkpoint("/checkpoints/run1")
result = trainer.train(dataset)
```

`checkpoint.pt` contains model weights, optimizer state, and the config dict. Checkpoints are loaded with `torch.load(..., weights_only=True)`, so resuming never executes arbitrary pickled code.

### Eval loop

```python
TorchTrainConfig(eval_every=100)   # evaluate every 100 optimizer steps

result = trainer.train(train_pipe, eval_dataset=eval_pipe)
print(result["eval_losses"])   # {100: 1.23, 200: 1.18, ...}
```

### W&B / TensorBoard logging

```bash
pip install "olaverse-foundry[logging]"   # W&B
pip install tensorboard                   # TensorBoard
```

```python
TorchTrainConfig(
    log_backend = "wandb",        # "wandb" | "tensorboard" | "none"
    project     = "my-project",
    run_name    = "exp-001",
)
```

The logger silently degrades to no-op if the backend is not installed.

---

## Config reference

See [Config Reference →](config.md) for a full table of every field across all three trainers.
