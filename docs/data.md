# DataPipeline

`DataPipeline` is a unified dataset adapter for all three trainers. It accepts HuggingFace datasets (including streaming), Python lists of strings, dicts, or numpy arrays — and converts them into trainer-ready batches.

```bash
pip install "olaverse-foundry[data]"   # for HuggingFace datasets support
```

---

## Modes

| Mode | Output per batch | Use with |
|---|---|---|
| `"lm"` | `(B, S)` int array | `TorchDistillTrainer`, `CachedDistillTrainer` |
| `"embed"` | `{"input_ids": (B, S), "attention_mask": (B, S)}` | `EmbeddingDistillTrainer` |

---

## Constructor

```python
DataPipeline(
    source,
    tokenizer      = None,
    batch_size     = 8,
    max_length     = 512,
    mode           = "lm",
    shuffle_buffer = 0,
    text_column    = "text",
    ids_column     = "input_ids",
    mask_column    = "attention_mask",
    pad_id         = 0,
    drop_last      = False,
)
```

| Param | Type | Default | Description |
|---|---|---|---|
| `source` | `Dataset \| IterableDataset \| list[str] \| list[dict] \| list[np.ndarray]` | — | Input data source |
| `tokenizer` | `PreTrainedTokenizer` | `None` | Required when source contains raw text strings |
| `batch_size` | `int` | `8` | Examples per batch |
| `max_length` | `int` | `512` | Truncate / pad sequences to this length |
| `mode` | `str` | `"lm"` | `"lm"` or `"embed"` |
| `shuffle_buffer` | `int` | `0` | Reservoir buffer size for streaming shuffle. `0` = no shuffle. |
| `text_column` | `str` | `"text"` | Column name for text when source is a HF dataset of dicts |
| `ids_column` | `str` | `"input_ids"` | Column name for pre-tokenized IDs |
| `mask_column` | `str` | `"attention_mask"` | Column name for attention mask |
| `pad_id` | `int` | `0` | Padding token ID |
| `drop_last` | `bool` | `False` | Drop the last incomplete batch |

---

## Source types

### HuggingFace Dataset (finite)

```python
from datasets import load_dataset
from foundry import DataPipeline

ds   = load_dataset("allenai/c4", "en", split="train[:10000]")
tok  = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

pipe = DataPipeline(source=ds, tokenizer=tok, batch_size=8, max_length=512)
```

`len(pipe)` returns the number of batches.

### HuggingFace IterableDataset (streaming)

```python
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

pipe = DataPipeline(
    source         = ds,
    tokenizer      = tok,
    batch_size     = 8,
    max_length     = 512,
    shuffle_buffer = 10_000,   # reservoir buffer
)
```

`len(pipe)` raises `TypeError` for streaming sources — pass `total_steps` to `trainer.train()` to keep the LR scheduler working.

### List of strings

```python
texts = ["Bawo ni, se dara ni?", "Kedu ka ị mere?", "How far, you dey?"]

pipe = DataPipeline(source=texts, tokenizer=tok, batch_size=4)
```

### List of pre-tokenized dicts

```python
examples = [
    {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
    ...
]

pipe = DataPipeline(source=examples, mode="embed")
```

### List of numpy arrays

```python
import numpy as np
arrays = [np.random.randint(0, 32000, (512,)) for _ in range(1000)]

pipe = DataPipeline(source=arrays, batch_size=8)
```

---

## Streaming shuffle

For streaming sources, exact shuffle is impossible. `DataPipeline` implements a **reservoir buffer**: it reads `shuffle_buffer` examples into memory, then yields a random sample, replacing it with the next incoming example.

```python
pipe = DataPipeline(
    source         = streaming_ds,
    tokenizer      = tok,
    batch_size     = 8,
    shuffle_buffer = 50_000,   # larger = better shuffle, more memory
)
```

For finite sources, pass `shuffle=True` to `trainer.train()` instead — that shuffles the full dataset each epoch.

---

## Embed mode

In `"embed"` mode, each batch is a dict ready to pass to an encoder's `forward()`:

```python
pipe = DataPipeline(source=ds, tokenizer=tok, mode="embed", batch_size=32)

for batch in pipe:
    # batch = {"input_ids": np.array (B, S), "attention_mask": np.array (B, S)}
    out = student(**{k: torch.tensor(v) for k, v in batch.items()})
```

---

## Length

```python
len(pipe)   # number of batches for finite sources
            # raises TypeError for streaming sources
```

---

## Example: train with DataPipeline

```python
from datasets import load_dataset
from foundry import DataPipeline, TorchDistillTrainer, TorchTrainConfig

ds   = load_dataset("allenai/c4", "en", split="train", streaming=True)
tok  = AutoTokenizer.from_pretrained("my_model")

pipe = DataPipeline(
    source         = ds,
    tokenizer      = tok,
    batch_size     = 8,
    max_length     = 1024,
    mode           = "lm",
    shuffle_buffer = 20_000,
)

trainer = TorchDistillTrainer(student, teachers, TorchTrainConfig(epochs=1))
result  = trainer.train(pipe, total_steps=50_000)
```
