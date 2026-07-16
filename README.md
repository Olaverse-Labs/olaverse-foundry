# olaverse-foundry

**Small, specialised models from big general ones — even when your language or domain has no training data.**

The normal way assumes you have data. `foundry` is the pipeline for when you don't: **synthesize** the training data (translation into 400+ languages, LLM query generation, encoder-mined hard negatives), **distil or contrastively train** a small model on it, and **prove it** head-to-head against mBERT / e5 / LaBSE — one library, one afternoon, and everything exits as a standard HuggingFace directory that production code loads with `transformers` alone.

`foundry` is the model-building layer of the Olaverse ecosystem (where [`olaverse`](https://pypi.org/project/olaverse/) gives you ready-to-use models). It is model-agnostic: any HuggingFace model or your own `nn.Module` works.

---

## Why foundry, honestly

If you have plenty of data and a standard task, use the standard tool — HF `Trainer` for classifiers, [sentence-transformers](https://sbert.net) for embeddings, TRL for LLM distillation. Foundry earns its place where those assume things you don't have:

- **No data in your language.** `synthesize_parallel` (MADLAD-400 into Yoruba, Swahili, Hausa, …) → `mine_hard_negatives` → `ContrastiveTrainer` → `compare_retrievers` is a complete zero-to-benchmarked-retriever pipeline. The pieces exist elsewhere; the pipeline doesn't.
- **The DistilBERT objective, maintained.** Combined distillation + MLM (`DistilMLMTrainer`) still isn't in HF `Trainer` — people copy 2019-era scripts.
- **Multi-teacher distillation that doesn't melt your budget.** Weighted teacher pools with per-token fusion (`min_ce` / `mean_ce`), and disk-cached top-k logits so every epoch after the first runs without the teachers.
- **"Better" as a table, not a vibe.** The eval harness fine-tunes the *same* head on every model (or encodes with each model's *own* pooling and prefixes, for retrieval) and prints accuracy / nDCG / params side by side.
- **One workflow.** Every trainer takes the same `DataPipeline`, the same config shape, and the same checkpoint/eval/logging features — and every artifact is a plain HF directory.

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

## The flagship: no data → benchmarked retriever

```python
from foundry import (load_translator, synthesize_parallel, mine_hard_negatives,
                     ContrastiveTrainer, ContrastiveConfig,
                     compare_retrievers, print_retrieval_comparison)
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("my/multilingual-base")
tok   = AutoTokenizer.from_pretrained("my/multilingual-base")

# 1. Manufacture pairs for a language with no data (open MT model, 400+ languages)
tr    = load_translator("google/madlad400-3b-mt")
pairs = synthesize_parallel(english_corpus, tr, target_langs=["yo"])

# 2. Mine hard negatives with the encoder itself (no LLM)
pairs = mine_hard_negatives(pairs, model, tok, device="cuda")

# 3. Contrastive training — the e5 / bge recipe
ContrastiveTrainer(model, tok, ContrastiveConfig(batch_size=64, device="cuda")).train(pairs)
model.save_pretrained("./my-retriever"); tok.save_pretrained("./my-retriever")

# 4. Prove it — each baseline encoded with its own pooling & prefixes
results = compare_retrievers({"mine": "./my-retriever",
                              "e5":   "intfloat/multilingual-e5-base",
                              "LaBSE": "sentence-transformers/LaBSE"},
                             queries, corpus, qrels)
print_retrieval_comparison(results)
```

The narrated version, with what to expect at each step: [Guide — a retriever for a low-resource language](https://olaverse-labs.github.io/olaverse-foundry/guides/low-resource-retriever/).

---

## More examples

### Embedding distillation (200M student)

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

### Causal LM distillation with multiple teachers

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
| `MLMTrainer` | Masked-language-modeling pretraining of an encoder backbone from scratch (no teacher). `WithMLMHead` adds an MLM head to a custom encoder. |
| `EncoderDistillTrainer` | Token-level hidden-state distillation from a teacher encoder into a smaller arch (auto projection). |
| `DistilMLMTrainer` | Combined distillation + MLM in one loss — the DistilBERT objective. |
| `ContrastiveTrainer` | InfoNCE / MultipleNegativesRanking on `{anchor, positive[, negative]}` pairs — the e5/bge retrieval recipe. |
| `synthesize_parallel` / `synthesize_pairs` / `mine_hard_negatives` | Synthetic training data: MT translation for no-data languages, LLM query generation, encoder-mined hard negatives. |
| `compare_retrievers` / `evaluate_retrieval` | nDCG@k / Recall@k, and a head-to-head retriever table with per-model pooling & prefixes. |
| `SequenceClassificationTrainer` / `TokenClassificationTrainer` | Fine-tune classification / NER heads on any base. Full fine-tune or `freeze_backbone`. `build_encoder_with_head` attaches a head in one line. |
| `prepare_qat` / `export_quantized` | Quantization-aware training (int8/int4 fake-quant) + int8 weight export and footprint report. |
| `compare_encoders` / `evaluate_encoder` | Head-to-head accuracy / macro-F1 table across models. |
| `load_for_inference` / `generate` | Load a built model (optional 4-bit/8-bit, optional skill pack) and generate. |
| `TeacherRegistry` | Pool of HF teacher models with relative weights. Handles `AutoModelForCausalLM` and `AutoModel` (encoders). |
| `LogitCache` | In-memory + on-disk cache for top-k teacher logit distributions. |
| `GrowthPlan` / `plan_growth` / `detect_layer_prefix` | Depth up-scaling via SOLAR-style layer duplication. Native merge (no external deps); layer prefix auto-detected for any arch. |
| `SkillPack` / `SkillRegistry` | Detachable LoRA adapters bound to a specific base model hash. |
| `save_as_peft` / `load_from_peft` | PEFT-format adapter round-trip (no peft library required). |
| `MinEDAlignment` | Cross-tokenizer vocabulary alignment via edit distance. |
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
| `[torch]` | torch, transformers, safetensors, accelerate | Real training (incl. native SOLAR depth up-scaling) |
| `[lego]` | peft | LoRA skill packs |
| `[data]` | datasets | HuggingFace dataset streaming |
| `[align]` | rapidfuzz | Fast cross-tokenizer alignment (100× speedup) |
| `[logging]` | wandb | Experiment tracking |
| `[docs]` | mkdocs-material | Build the documentation site locally |
| `[all]` | everything (runtime extras) | Full setup |

---

## Documentation

Full docs: [olaverse-labs.github.io/olaverse-foundry](https://olaverse-labs.github.io/olaverse-foundry/) (auto-deployed from `main`).

Build or preview the site locally:

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # validate (no broken links / nav)
```

---

## Links

- **Main SDK** — [olaverse](https://pypi.org/project/olaverse/) — ready-to-use models
- **Homepage** — [olaverse.co.uk](https://olaverse.co.uk)
- **GitHub** — [Olaverse-Labs/olaverse-foundry](https://github.com/Olaverse-Labs/olaverse-foundry)
- **Issues** — [GitHub Issues](https://github.com/Olaverse-Labs/olaverse-foundry/issues)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
