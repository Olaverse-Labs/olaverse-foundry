# Changelog

---

## Unreleased

### Fixes

- Python 3.9 support actually works: a runtime `X | None` type alias in the fusion strategy registry broke every import of `foundry.fusion` on 3.9 (CI had never been green).

---

## v0.2.0 — 2026-07-16

### Encoder base models

- **`MLMTrainer`** — masked-language-modeling pretraining of an encoder backbone from scratch (teacherless). `WithMLMHead` adds an MLM head to a custom encoder.
- **`EncoderDistillTrainer`** — token-level hidden-state distillation from a teacher encoder into a smaller architecture, with automatic student→teacher projection.
- **`DistilMLMTrainer`** — combined distillation + MLM in a single multi-part loss (the DistilBERT objective: MLM CE + temperature-scaled KL + hidden-state cosine).

### Retrieval

- **`ContrastiveTrainer`** — InfoNCE / MultipleNegativesRanking training on `{anchor, positive[, negative]}` pairs, with in-batch negatives and optional hard negatives, for (cross-lingual) retrieval.
- `evaluate_retrieval()` / `compare_retrievers()` / `print_retrieval_comparison()` — nDCG@k / Recall@k scoring and a head-to-head model table; each model encoded with its own tokenizer, pooling, and prefixes (e5 / bge / LaBSE auto-configured).
- `encode_texts()` — batched no-grad encoding to numpy, with pooling, normalisation, and prefix support.

### Synthetic data

- `synthesize_pairs()` / `generate_hard_negatives()` — query + hard-negative generation with an open, Apache-licensed instruct LLM (`load_generator`, Qwen/Mistral).
- `mine_hard_negatives()` — encoder-based hard-negative mining (LLM-free; the right choice for low-resource languages).
- `synthesize_parallel()` / `translate_texts()` — synthetic parallel pairs for no-data languages via an open MT model (`load_translator`, MADLAD-400).

### Task heads

- **`SequenceClassificationTrainer`** / **`TokenClassificationTrainer`** — fine-tune classification / NER heads on any base encoder (model-agnostic; any model returning `.logits`).
- `freeze_backbone()` + `HeadTrainConfig(freeze_backbone=True)` — train only the head so many heads share one frozen encoder.
- `build_encoder_with_head(base, num_labels, task)` — attach a fresh head in one line.
- `DataPipeline(label_column=...)` — emit `{input_ids, attention_mask, labels}` (scalar or `-100`-padded token labels).

### Quantization-aware training

- `prepare_qat(model, QATConfig)` — int8/int4 fake-quant (straight-through) on any model's linears; train with any trainer.
- `export_quantized()` (footprint report), `int8_state_dict()` (packed int8 + scales), `quantize_tensor()`.

### Evaluation & inference

- `compare_encoders()` / `evaluate_encoder()` / `print_comparison()` / `macro_f1()` — head-to-head accuracy / macro-F1 table (each model tokenised with its own tokenizer).
- `load_for_inference()` (optional 4-bit/8-bit, optional skill-pack merge) and `generate()`.

### Growth

- **Native merge** — `run_merge()` materialises the grown model with transformers + safetensors; no external merge tool required.
- `detect_layer_prefix()` — auto-detects the transformer block prefix, so growth works on Llama, BERT, GPT-2, and more.

### Fixes

- **Security** — all trainers now load checkpoints with `torch.load(..., weights_only=True)`, so resuming from a checkpoint can never execute arbitrary pickled code.
- The test suite now skips torch-dependent tests cleanly when torch is not installed, instead of failing at collection.
- `MLMTrainer` no longer produces a NaN loss when a batch masks zero tokens.
- `recipe.run()` raises instead of silently falling back to a numpy stub when torch is absent, and refuses to train on synthetic random tokens.
- Removed the `mergekit` dependency (the native merge backend replaces it).

---

## v0.1.0 — 2026-06-16

First public release of `olaverse-foundry`.

### Trainers

- **`TorchDistillTrainer`** — single-GPU CE+KL distillation against one or many teachers
- **`CachedDistillTrainer`** — multi-epoch distillation with on-disk `LogitCache` + `accelerate` DDP/FSDP support
- **`EmbeddingDistillTrainer`** — MSE/cosine loss on pooled sentence vectors for bi-encoder distillation

### Production training features (all trainers)

- Mixed precision — `torch_dtype="bfloat16"` / `"float16"` / `"float32"`
- Gradient accumulation — `grad_accumulation_steps=N`
- LR scheduler — `"cosine"` / `"linear"` / `"constant"` with linear warmup
- Reproducible seed — `seed=42` wires torch + numpy + random
- Checkpoint save/resume — `save_checkpoint()` / `resume_from_checkpoint()`
- Auto-checkpoint — `save_every=N, save_dir=...`
- Eval loop — `eval_every=N` with held-out dataset
- W&B / TensorBoard logging — `log_backend="wandb"` or `"tensorboard"`
- OOM handling — CUDA OOM caught and re-raised with actionable suggestions
- `on_step` callback for custom progress tracking

### DataPipeline

- Unified dataset adapter for HF `Dataset` / `IterableDataset`, `list[str]`, `list[dict]`, `list[np.ndarray]`
- Modes: `"lm"` (int arrays) and `"embed"` (input_ids + attention_mask dicts)
- Reservoir shuffle buffer for streaming sources
- `len()` for finite sources; `TypeError` for streaming (pass `total_steps=` to trainer)

### Teachers

- `TeacherRegistry` — pool of teachers with relative weights
- `HFTeacher` — supports `model_type="causal_lm"` and `model_type="encoder"` (for embedding teachers)
- `ToyTeacher` / `ToyEmbeddingTeacher` — lightweight test stubs
- `LogitCache` — top-k logit storage with `.npz` serialisation

### Model loading

- `load_model(ref, model_class=None)` — `model_class` parameter for encoder vs causal LM

### Skill packs

- `SkillPack` / `SkillRegistry` — detachable LoRA adapters
- `snap_on()` — right-to-left key matching handles HF's deeply-nested state dict keys
- PEFT format round-trip: `save_as_peft()` / `load_from_peft()` / `peft_config_dict()`

### Growth & fusion

- `plan_growth()` / `GrowthPlan` — SOLAR depth up-scaling
- `upscale_layer_map()` / `layers_for_param_target()`
- `growth_plan_to_mergekit_yaml()` / `save_mergekit_config()` / `run_merge()`
- `MinEDAlignment` — cross-tokenizer vocab alignment via edit distance
- Fusion strategies: `min_ce`, `mean_ce`

### Recipes

- `FoundryRecipe` / `EmbedRecipe` — Pydantic-validated YAML recipe files
- `Recipe.load()` — auto-detect recipe type

### CLI

- `foundry doctor` — environment check
- `foundry plan` / `foundry run` — causal LM recipes
- `foundry embed` — embedding recipes
- `foundry strategies` — list fusion strategies
- `foundry backends` — backend summary

### Backends

- `detect_backend()` — torch, cuda, mps, accelerate, peft, safetensors, wandb, rapidfuzz

### Optional extras

| Extra | What it installs |
|---|---|
| `[torch]` | torch, transformers, safetensors, accelerate |
| `[lego]` | peft |
| `[data]` | datasets |
| `[align]` | rapidfuzz |
| `[logging]` | wandb |
| `[all]` | all of the above |
