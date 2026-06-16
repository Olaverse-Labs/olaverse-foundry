# olaverse-foundry

**Build model families from a single pretrained seed.**

`olaverse-foundry` is the training and model-factory layer of the Olaverse ecosystem. Where `olaverse` gives you ready-to-use NLP models, `foundry` lets you build new ones — distilling, growing, fusing, and adapting them for production.

```
seed → grow → distil / fuse → freeze → skill packs
```

---

## Install

```bash
# Core (schema validation, growth planning — no GPU required)
pip install olaverse-foundry

# GPU training
pip install olaverse-foundry[torch]

# LoRA skill packs
pip install olaverse-foundry[torch,lego]

# Everything
pip install olaverse-foundry[all]
```

---

## Quick start — embedding distillation (200M student)

```python
from foundry import DataPipeline, EmbeddingDistillTrainer, EmbeddingDistillConfig
from transformers import AutoModel, AutoTokenizer

# Load student and teacher
student = AutoModel.from_pretrained("microsoft/deberta-v3-base")
teacher = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
tok     = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Stream data
pipe = DataPipeline(
    source       = my_hf_dataset,
    tokenizer    = tok,
    batch_size   = 32,
    max_length   = 128,
    mode         = "embed",
    shuffle_buffer = 10_000,
)

# Train
trainer = EmbeddingDistillTrainer(
    student = student,
    teacher = teacher,
    config  = EmbeddingDistillConfig(
        loss         = "cosine",
        pool         = "mean",
        epochs       = 3,
        lr_scheduler = "cosine",
        warmup_steps = 200,
        torch_dtype  = "bfloat16",
        save_every   = 1000,
        save_dir     = "/checkpoints/embed-200m",
        log_backend  = "wandb",
    ),
)

result = trainer.train(pipe, eval_dataset=eval_pipe)
print(result["eval_losses"])
```

---

## Quick start — causal LM distillation with multiple teachers

```python
from foundry import (
    DataPipeline, TorchDistillTrainer, TorchTrainConfig,
    TeacherRegistry, FoundryRecipe,
)

# Build a registry of teachers
teachers = TeacherRegistry.from_names(
    ["meta-llama/Llama-3.1-70B", "Qwen/Qwen2-72B-Instruct"],
    weights=[1.0, 0.8],
)
teachers.load_all()

# Stream training data
pipe = DataPipeline(
    source     = my_dataset,
    tokenizer  = tok,
    batch_size = 8,
    max_length = 2048,
    mode       = "lm",
)

trainer = TorchDistillTrainer(
    student  = my_3b_model,
    teachers = teachers,
    config   = TorchTrainConfig(
        epochs                = 1,
        lr_scheduler          = "cosine",
        warmup_steps          = 500,
        torch_dtype           = "bfloat16",
        grad_accumulation_steps = 8,
        save_every            = 500,
        save_dir              = "/checkpoints/run1",
        eval_every            = 100,
        log_backend           = "wandb",
    ),
)

result = trainer.train(pipe, eval_dataset=eval_pipe)
```

---

## Key components

| Module | What it does |
|---|---|
| `DataPipeline` | Converts HF datasets, string lists, or numpy arrays into trainer-ready batches. Supports streaming and reservoir shuffle. |
| `TorchDistillTrainer` | Single-GPU distillation: CE + KL loss against one or more teachers. |
| `CachedDistillTrainer` | Like `TorchDistillTrainer` but caches teacher logits on disk after the first pass. Subsequent epochs are free. Supports `accelerate` for multi-GPU. |
| `EmbeddingDistillTrainer` | MSE / cosine loss on pooled sentence vectors. Use for bi-encoder / reranker distillation. |
| `TeacherRegistry` | Pool of HF teacher models with relative weights. Handles `AutoModelForCausalLM` and `AutoModel` (encoders). |
| `LogitCache` | In-memory + on-disk cache for top-k teacher logit distributions. |
| `GrowthPlan` / `plan_growth` | Depth up-scaling via SOLAR-style layer duplication. Generates mergekit-compatible YAML. |
| `SkillPack` / `SkillRegistry` | Detachable LoRA adapters bound to a specific base model hash. |
| `save_as_peft` / `load_from_peft` | PEFT-format adapter round-trip (no peft library required). |
| `MinEDAlignment` | Cross-tokenizer vocabulary alignment via edit distance. |
| `DataPipeline` | Unified dataset adapter — HF datasets, streaming, raw text, numpy. |
| `FoundryRecipe` / `EmbedRecipe` | Pydantic-validated YAML recipes — fail fast before GPU spend. |

---

## Training features

All trainers share the same production-ready feature set:

- **Mixed precision** — `torch_dtype="bfloat16"` or `"float16"`
- **Gradient accumulation** — `grad_accumulation_steps=N`
- **LR scheduler** — `"cosine"` / `"linear"` / `"constant"` with linear warmup
- **Reproducibility** — `seed=42` sets torch + numpy + random before training
- **Checkpointing** — `save_checkpoint(path)` / `resume_from_checkpoint(path)`
- **Auto-checkpoint** — `save_every=N, save_dir="/path"` saves every N steps
- **Eval loop** — `eval_every=N` evaluates on a held-out set every N steps
- **W&B / TensorBoard** — `log_backend="wandb"` or `"tensorboard"`
- **OOM handling** — CUDA OOM raises with actionable suggestions
- **Streaming datasets** — `DataPipeline` wraps any HF `IterableDataset`
- **Dataset shuffling** — `shuffle=True` or `shuffle_buffer=N` for streaming

---

## CLI

```bash
# Check your environment
foundry doctor

# Preview a recipe plan (no GPU spend)
foundry plan recipe.yaml

# Run a recipe
foundry run recipe.yaml

# Run an embedding distillation recipe
foundry embed recipe.yaml

# List fusion strategies
foundry strategies
```

---

## Recipe YAML

```yaml
# recipe.yaml — full causal-LM factory
seed:
  model: meta-llama/Llama-3.1-8B
  init: pretrained

grow:
  method: depth_upscale
  to_params: 15B

teachers:
  - role: reasoning
    model: meta-llama/Llama-3.1-70B
    weight: 1.0

fusion:
  strategy: min_ce
  align: min_ed
  cache: topk_64

heal:
  tokens: 100B
  alpha: 0.3

output:
  freeze_base: true
  skillpacks: [ola_math, ola_code]
```

---

## Optional extras

| Extra | Installs | When to use |
|---|---|---|
| `[torch]` | torch, transformers, safetensors, accelerate | Real training |
| `[lego]` | peft | LoRA skill packs |
| `[merge]` | mergekit | SOLAR depth up-scaling |
| `[data]` | datasets | HuggingFace dataset streaming |
| `[align]` | rapidfuzz | Fast cross-tokenizer alignment (100× speedup) |
| `[logging]` | wandb | Experiment tracking |
| `[all]` | everything | Full setup |

---

## Links

- **Main SDK** — [olaverse](https://pypi.org/project/olaverse/) — ready-to-use African NLP models
- **Homepage** — [olaverse.co.uk](https://olaverse.co.uk)
- **GitHub** — [Olaverse-Labs/olaverse-foundry](https://github.com/Olaverse-Labs/olaverse-foundry)
- **Issues** — [GitHub Issues](https://github.com/Olaverse-Labs/olaverse-foundry/issues)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
