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

### Memory, and bounding it

Cached top-k distributions are large. At `top_k=64` with 8x512 batches each
entry is ~2MB:

| batches | tokens | RAM at `top_k=64` |
| --- | --- | --- |
| 10,000 | 41M | ~21 GB |
| 10,000 (seq 2048) | 164M | ~84 GB |

Real distillation runs are far bigger than that, so an unbounded in-memory cache
is the wrong default at scale. Give `LogitCache` a `cache_dir` and a
`max_entries` cap:

```python
from foundry.teachers import LogitCache

cache = LogitCache(top_k=64, max_entries=512, cache_dir="./teacher_cache")
```

RAM then holds at most `max_entries` entries under **LRU** eviction, and evicted
entries spill to sharded `.npz` files rather than being discarded. A miss in RAM
falls through to disk and is promoted back. `stats` reports `disk_hits` and
`spills` alongside the usual counters.

Because the shards and their index live in `cache_dir`, opening a cache on the
same directory picks up whatever a previous *process* wrote — populate once,
reuse across runs and machines. Call `flush()` before relying on the shards.

!!! note "Why LRU and not FIFO"
    Training reads batches in order. Under FIFO, a cache smaller than the
    dataset evicts exactly the entries the next epoch reads first, so the hit
    rate collapses to ~0 and every epoch re-runs the teachers — the one cost the
    cache exists to avoid. Nothing errors; the run is simply slow.

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
