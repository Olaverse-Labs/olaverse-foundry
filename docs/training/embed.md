# EmbeddingDistillTrainer

Distillation trainer for sentence embedding models. Uses MSE or cosine loss on mean-pooled (or CLS-pooled) vectors — the right tool for compressing bi-encoders and rerankers.

```bash
pip install "olaverse-foundry[torch]"
```

---

## When to use

Use `EmbeddingDistillTrainer` when:

- Your **teacher and student are encoders** (`AutoModel`, not `AutoModelForCausalLM`)
- You want to compress a **large bi-encoder** (e.g. `bge-large-en`) into a smaller one
- You are training a **reranker or semantic similarity** model

---

## Usage

```python
from transformers import AutoModel, AutoTokenizer
from foundry import EmbeddingDistillTrainer, EmbeddingDistillConfig, DataPipeline

student = AutoModel.from_pretrained("microsoft/deberta-v3-base")
teacher = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
tok     = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

pipe = DataPipeline(
    source      = my_dataset,
    tokenizer   = tok,
    batch_size  = 32,
    max_length  = 128,
    mode        = "embed",    # yields {"input_ids": ..., "attention_mask": ...}
)

trainer = EmbeddingDistillTrainer(
    student = student,
    teacher = teacher,
    config  = EmbeddingDistillConfig(
        loss         = "cosine",
        pool         = "mean",
        normalize    = True,
        temperature  = 0.05,
        epochs       = 3,
        lr_scheduler = "cosine",
        warmup_steps = 200,
        torch_dtype  = "bfloat16",
        eval_every   = 500,
        save_every   = 1000,
        save_dir     = "/checkpoints/embed",
        log_backend  = "wandb",
    ),
)

result = trainer.train(pipe, eval_dataset=eval_pipe)
print(result["eval_losses"])
```

---

## Constructor

```python
EmbeddingDistillTrainer(
    student,
    teacher,
    config = None,   # defaults to EmbeddingDistillConfig()
)
```

| Param | Type | Description |
|---|---|---|
| `student` | `nn.Module` | Encoder whose `forward()` returns `.last_hidden_state` |
| `teacher` | `nn.Module` | Larger encoder — same interface |
| `config` | `EmbeddingDistillConfig` | Training configuration |

---

## `train()`

```python
result = trainer.train(
    dataset,
    eval_dataset = None,
    on_step      = None,
    shuffle      = False,
    pre_cache    = False,
    total_steps  = None,
)
```

| Param | Type | Description |
|---|---|---|
| `dataset` | iterable | Batches of `{"input_ids": ..., "attention_mask": ...}` dicts, or a `DataPipeline(mode="embed")` |
| `eval_dataset` | iterable | Optional eval set |
| `on_step` | callable | Callback(step, loss) |
| `shuffle` | `bool` | Shuffle dataset per epoch |
| `pre_cache` | `bool` | Pre-compute teacher embeddings for all batches before training starts |
| `total_steps` | `int` | Override total steps for LR scheduler |

**Returns** `dict` with `losses`, `eval_losses`, `device`.

---

## Loss functions

### Cosine (`loss="cosine"`)

```
loss = 1 - cosine_similarity(student_emb, teacher_emb).mean()
```

With `normalize=True`, both embeddings are L2-normalised before the similarity is computed (equivalent to dot product after normalisation). This is the standard loss for bi-encoder distillation.

### MSE (`loss="mse"`)

```
loss = mean_squared_error(student_emb, teacher_emb)
```

Useful when preserving the absolute scale of embeddings matters, e.g. for regression tasks.

---

## Pooling

| `pool` | Description |
|---|---|
| `"mean"` | Average over all non-padding token positions (recommended) |
| `"cls"` | Use the first token's hidden state |

---

## Pre-caching teacher embeddings

```python
result = trainer.train(pipe, pre_cache=True)
```

With `pre_cache=True`, teacher embeddings are computed and stored in memory before the training loop starts. Subsequent epochs reuse the cache — teacher inference cost is paid once. Useful for multi-epoch runs on finite datasets when the dataset fits in memory.

---

## EmbeddingDistillConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `loss` | `str` | `"cosine"` | Loss function: `"cosine"` or `"mse"` |
| `pool` | `str` | `"mean"` | Pooling strategy: `"mean"` or `"cls"` |
| `normalize` | `bool` | `True` | L2-normalise embeddings before loss |
| `temperature` | `float` | `0.05` | Scaling factor applied before cosine similarity |
| `learning_rate` | `float` | `2e-5` | AdamW learning rate |
| `epochs` | `int` | `3` | Training epochs |
| `weight_decay` | `float` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping norm |
| `device` | `str` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"` / `"mps"` |
| `log_every` | `int` | `10` | Log loss every N steps |
| `lr_scheduler` | `str` | `"constant"` | `"constant"` / `"cosine"` / `"linear"` |
| `warmup_steps` | `int` | `0` | Linear warmup steps |
| `grad_accumulation_steps` | `int` | `1` | Gradient accumulation |
| `torch_dtype` | `str` | `"float32"` | Mixed precision dtype |
| `eval_every` | `int` | `0` | Eval every N steps (0 = off) |
| `save_every` | `int` | `0` | Auto-checkpoint every N steps (0 = off) |
| `save_dir` | `str` | `""` | Checkpoint directory |
| `log_backend` | `str` | `"none"` | `"wandb"` / `"tensorboard"` / `"none"` |
| `run_name` | `str` | `""` | W&B / TB run name |
| `project` | `str` | `"olaverse-foundry"` | W&B project name |
| `seed` | `int` | `42` | Random seed |
