# Config Reference

All trainer configs share a common base (`TrainConfig`) extended by `TorchTrainConfig` which is further extended by `CachedDistillConfig`. `EmbeddingDistillConfig` is standalone with a similar set of fields.

---

## TrainConfig (base)

Shared by `TorchDistillTrainer` and `CachedDistillTrainer`.

| Field | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `1e-4` | AdamW learning rate |
| `epochs` | `int` | `3` | Number of training epochs |
| `batch_size` | `int` | `4` | Input batch size (informational; actual size comes from your dataset) |
| `alpha` | `float` | `0.3` | CE loss weight. KL weight = `1 - alpha`. |
| `fusion_strategy` | `str` | `"min_ce"` | How to fuse multiple teacher distributions: `"min_ce"` or `"mean_ce"` |
| `top_k` | `int` | `64` | Top-k teacher logits to consider during fusion |
| `log_every` | `int` | `10` | Print / callback loss every N optimizer steps |
| `seed` | `int` | `42` | Random seed for torch, numpy, and random |
| `lr_scheduler` | `str` | `"constant"` | LR schedule: `"constant"` / `"cosine"` / `"linear"` |
| `warmup_steps` | `int` | `0` | Linear warmup steps before the main schedule |
| `eval_every` | `int` | `0` | Run eval loop every N steps. `0` = disabled. |
| `save_every` | `int` | `0` | Auto-checkpoint every N steps. `0` = disabled. |
| `save_dir` | `str` | `""` | Directory for auto-checkpoints |

---

## TorchTrainConfig

Extends `TrainConfig` with torch-specific fields.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_grad_norm` | `float` | `1.0` | Gradient clipping norm |
| `weight_decay` | `float` | `0.01` | AdamW weight decay |
| `device` | `str` | `"auto"` | `"auto"` (CUDA→MPS→CPU), `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"` |
| `grad_accumulation_steps` | `int` | `1` | Accumulate gradients over N batches before stepping |
| `torch_dtype` | `str` | `"float32"` | Mixed precision: `"float32"` / `"bfloat16"` / `"float16"` |
| `log_backend` | `str` | `"none"` | Logging backend: `"wandb"` / `"tensorboard"` / `"none"` |
| `run_name` | `str` | `""` | W&B or TensorBoard run name |
| `project` | `str` | `"olaverse-foundry"` | W&B project name |

---

## CachedDistillConfig

Extends `TorchTrainConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `cache_dir` | `str \| None` | `None` | Directory to save/load `.npz` logit cache files. `None` = in-memory only. |
| `cache_top_k` | `int` | `64` | Top-k logits to cache per token position |
| `use_accelerate` | `bool` | `True` | Use `accelerate.Accelerator` for DDP/FSDP. Falls back to plain torch if not installed. |

---

## EmbeddingDistillConfig

Standalone config for `EmbeddingDistillTrainer`.

| Field | Type | Default | Description |
|---|---|---|---|
| `loss` | `str` | `"cosine"` | Loss function: `"cosine"` or `"mse"` |
| `pool` | `str` | `"mean"` | Pooling: `"mean"` or `"cls"` |
| `normalize` | `bool` | `True` | L2-normalise embeddings before loss computation |
| `temperature` | `float` | `0.05` | Scaling factor applied before cosine similarity |
| `learning_rate` | `float` | `2e-5` | AdamW learning rate |
| `epochs` | `int` | `3` | Training epochs |
| `weight_decay` | `float` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping norm |
| `device` | `str` | `"auto"` | Device selection |
| `log_every` | `int` | `10` | Log every N steps |
| `lr_scheduler` | `str` | `"constant"` | `"constant"` / `"cosine"` / `"linear"` |
| `warmup_steps` | `int` | `0` | Linear warmup steps |
| `grad_accumulation_steps` | `int` | `1` | Gradient accumulation steps |
| `torch_dtype` | `str` | `"float32"` | Mixed precision dtype |
| `eval_every` | `int` | `0` | Eval every N steps |
| `save_every` | `int` | `0` | Auto-checkpoint every N steps |
| `save_dir` | `str` | `""` | Checkpoint directory |
| `log_backend` | `str` | `"none"` | Logging backend |
| `run_name` | `str` | `""` | Run name |
| `project` | `str` | `"olaverse-foundry"` | W&B project name |
| `seed` | `int` | `42` | Random seed |

---

## Inheritance diagram

```
TrainConfig
└── TorchTrainConfig
    └── CachedDistillConfig

EmbeddingDistillConfig  (standalone, same fields)
```

---

## LR scheduler details

| `lr_scheduler` | `warmup_steps` | Behaviour |
|---|---|---|
| `"constant"` | `0` | Fixed LR throughout. No scheduler object created. |
| `"constant"` | `> 0` | Linear warmup, then constant LR. |
| `"cosine"` | any | Linear warmup, then cosine decay to 0. |
| `"linear"` | any | Linear warmup, then linear decay to 0. |

`total_steps` for the decay phase is computed as:

```
total_steps = ceil(len(dataset) × epochs / grad_accumulation_steps)
```

For streaming datasets (`len()` raises `TypeError`), `total_steps` defaults to `0` which keeps LR constant after warmup. Override with `trainer.train(..., total_steps=N)`.

---

## Fusion strategies

| `fusion_strategy` | Description |
|---|---|
| `"min_ce"` | Per-position, pick the teacher distribution with the lowest CE against ground truth. Encourages the student to learn from the most confident teacher. |
| `"mean_ce"` | Weighted average of all teacher distributions (by `teacher.weight`). Smooths over teacher disagreements. |

---

## Other configs

The encoder, head, and quantization trainers have their own configs (same shared production fields — mixed precision, grad accumulation, LR schedule, eval, checkpointing). Each is documented on its page:

| Config | Trainer | Page |
|---|---|---|
| `MLMConfig` | `MLMTrainer` | [MLM pretraining](mlm.md) |
| `EncoderDistillConfig` | `EncoderDistillTrainer` | [Encoder distillation](encoder-distill.md) |
| `HeadTrainConfig` | `SequenceClassificationTrainer` / `TokenClassificationTrainer` | [Task heads](heads.md) |
| `QATConfig` | `prepare_qat` | [Quantization](../quantization.md) |
