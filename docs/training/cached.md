# CachedDistillTrainer

Distillation trainer with on-disk `LogitCache` and `accelerate` support. Teachers run a single pass over the dataset; every subsequent epoch reads directly from the cache — zero teacher inference cost.

```bash
pip install "olaverse-foundry[torch]"
```

---

## When to use

Use `CachedDistillTrainer` instead of `TorchDistillTrainer` when:

- You plan **more than 1 epoch** — teacher cost is paid once regardless
- Your dataset is **large** — cache files stay on disk between runs
- You want **multi-GPU** training via `accelerate`

---

## Usage

```python
from foundry import CachedDistillTrainer, CachedDistillConfig, TeacherRegistry, HFTeacher

teachers = TeacherRegistry([HFTeacher("meta-llama/Llama-3.1-70B", weight=1.0)])
teachers.load_all()

trainer = CachedDistillTrainer(
    student  = my_model,
    teachers = teachers,
    config   = CachedDistillConfig(
        epochs          = 5,
        cache_dir       = "/tmp/logit_cache",   # save caches to disk
        cache_top_k     = 64,
        use_accelerate  = True,
        torch_dtype     = "bfloat16",
        lr_scheduler    = "cosine",
        warmup_steps    = 500,
        save_every      = 1000,
        save_dir        = "/checkpoints/run1",
        log_backend     = "wandb",
    ),
)

result = trainer.train(dataset)
print(result["cache_stats"])   # [{"hits": 4000, "misses": 1000}, ...]
```

---

## Caching behaviour

On first call to `train()`:

1. `load_caches()` tries to load `.npz` files from `cache_dir`. Returns `False` if any are missing.
2. `build_caches()` materialises streaming datasets to a list, runs each teacher once, saves to disk.
3. All subsequent epochs read from `LogitCache` — teachers are never called again.

On subsequent runs (same `cache_dir`):

1. `load_caches()` finds the `.npz` files and loads them — `build_caches()` is skipped entirely.

```python
# Pre-build caches separately (optional, e.g. on a different machine)
trainer.build_caches(dataset)

# Later, run training without touching teachers
result = trainer.train(dataset)
```

---

## Constructor

```python
CachedDistillTrainer(
    student,
    teachers,
    config    = None,    # defaults to CachedDistillConfig()
    alignment = None,
)
```

---

## `train()`

```python
result = trainer.train(
    dataset,
    eval_dataset = None,
    on_step      = None,
    shuffle      = False,
    total_steps  = None,
)
```

Returns `dict` with:

| Key | Type | Description |
|---|---|---|
| `losses` | `list[float]` | Loss after each optimizer step |
| `eval_losses` | `dict[int, float]` | `{step: eval_loss}` |
| `device` | `str` | Device used |
| `cache_stats` | `list[dict]` | Per-teacher `{"hits": N, "misses": N}` |

---

## Checkpoint methods

`save_checkpoint` additionally saves each teacher's in-memory cache as `cache_teacher_N.npz` alongside `checkpoint.pt`. `resume_from_checkpoint` reloads both.

```python
trainer.save_checkpoint("/checkpoints/step_1000")
trainer.resume_from_checkpoint("/checkpoints/step_1000")
```

---

## CachedDistillConfig

Inherits all fields from `TorchTrainConfig` plus:

| Field | Type | Default | Description |
|---|---|---|---|
| `cache_dir` | `str \| None` | `None` | Directory to save/load `.npz` cache files. `None` = memory only. |
| `cache_top_k` | `int` | `64` | Number of top-k logits to cache per token position. |
| `use_accelerate` | `bool` | `True` | Try to init `accelerate.Accelerator` for DDP/FSDP. Falls back to plain torch if not installed. |

See [Config Reference →](config.md) for inherited fields.

---

## Multi-GPU with accelerate

```bash
pip install accelerate
accelerate config   # set up your hardware profile once
```

```bash
accelerate launch my_training_script.py
```

```python
CachedDistillConfig(use_accelerate=True)   # default
```

The accelerator handles gradient accumulation and grad clipping automatically via `accelerator.accumulate()`.
