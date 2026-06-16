# TorchDistillTrainer

Single-GPU distillation trainer with CE + KL loss. Teachers run live on every step, making this the simplest trainer to use when you have a small dataset or are prototyping.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Usage

```python
from foundry import TorchDistillTrainer, TorchTrainConfig, TeacherRegistry, HFTeacher

teachers = TeacherRegistry([HFTeacher("meta-llama/Llama-3.1-8B", weight=1.0)])
teachers.load_all()

trainer = TorchDistillTrainer(
    student  = my_model,
    teachers = teachers,
    config   = TorchTrainConfig(
        epochs                  = 1,
        learning_rate           = 2e-5,
        alpha                   = 0.3,
        lr_scheduler            = "cosine",
        warmup_steps            = 200,
        torch_dtype             = "bfloat16",
        grad_accumulation_steps = 4,
        eval_every              = 100,
        save_every              = 500,
        save_dir                = "/ckpt/run1",
        log_backend             = "wandb",
    ),
)

result = trainer.train(
    dataset,
    eval_dataset = eval_dataset,
    shuffle      = True,
    total_steps  = 5000,   # for LR scheduler when source is streaming
)
```

---

## Constructor

```python
TorchDistillTrainer(
    student,
    teachers,
    config    = None,   # defaults to TorchTrainConfig()
    alignment = None,   # tokenizer alignment, defaults to IdentityAlignment
)
```

| Param | Type | Description |
|---|---|---|
| `student` | `nn.Module` | Model whose `.forward()` returns `.logits` |
| `teachers` | `TeacherRegistry` | One or more teachers with relative weights |
| `config` | `TorchTrainConfig` | Training configuration |
| `alignment` | `FusionKernel` subclass | Cross-tokenizer vocab alignment (optional) |

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

| Param | Type | Description |
|---|---|---|
| `dataset` | iterable | Batches of `(B, S)` int arrays, or a `DataPipeline` |
| `eval_dataset` | iterable | Optional held-out dataset for eval loop |
| `on_step` | `Callable[[int, float], None]` | Callback fired every `log_every` steps |
| `shuffle` | `bool` | Shuffle dataset each epoch (finite datasets only) |
| `total_steps` | `int` | Override total optimizer steps for LR scheduler. Use when dataset length is unknown (streaming). |

**Returns** `dict` with:

| Key | Type | Description |
|---|---|---|
| `losses` | `list[float]` | Loss after each optimizer step |
| `eval_losses` | `dict[int, float]` | `{step: eval_loss}` |
| `device` | `str` | Device used (e.g. `"cuda:0"`) |

---

## `train_step()`

```python
loss = trainer.train_step(
    input_ids,
    is_first_accum = True,
    is_last_accum  = True,
)
```

Call directly for custom training loops. `input_ids` is a `(B, S)` numpy array or torch tensor. Set `is_first_accum=False` to skip zeroing gradients and `is_last_accum=False` to skip the optimizer step (for gradient accumulation).

---

## Checkpoint methods

```python
trainer.save_checkpoint("/checkpoints/step_1000")
trainer.resume_from_checkpoint("/checkpoints/step_1000")
```

`save_checkpoint` writes `checkpoint.pt` containing model weights, optimizer state, and config. `resume_from_checkpoint` accepts either a directory or a `.pt` file path.

---

## TorchTrainConfig

```python
from foundry import TorchTrainConfig

config = TorchTrainConfig(
    # Base
    learning_rate           = 1e-4,
    epochs                  = 3,
    batch_size              = 4,
    alpha                   = 0.3,
    fusion_strategy         = "min_ce",
    top_k                   = 64,
    log_every               = 10,
    seed                    = 42,
    lr_scheduler            = "constant",
    warmup_steps            = 0,
    eval_every              = 0,
    save_every              = 0,
    save_dir                = "",
    # Torch-specific
    max_grad_norm           = 1.0,
    weight_decay            = 0.01,
    device                  = "auto",
    grad_accumulation_steps = 1,
    torch_dtype             = "float32",
    log_backend             = "none",
    run_name                = "",
    project                 = "olaverse-foundry",
)
```

See [Config Reference →](config.md) for full field descriptions.

---

## Loss function

The trainer minimises:

```
loss = alpha × CE(student_logits, gold_tokens)
     + (1 - alpha) × KL(student_logits, fused_teacher_distribution)
```

`alpha=0.3` puts 30% weight on CE (ground truth) and 70% on KL (teacher).

When multiple teachers are present, their distributions are fused according to `fusion_strategy`:

- `"min_ce"` (default) — picks the teacher distribution with lowest CE against ground truth per position
- `"mean_ce"` — averages distributions weighted by teacher `weight`

---

## OOM handling

CUDA out-of-memory errors are caught and re-raised with actionable suggestions:

```
RuntimeError: CUDA out of memory. Suggestions:
  • Reduce batch size
  • Increase grad_accumulation_steps
  • Set torch_dtype='bfloat16'
  Original: CUDA out of memory. ...
```
