# Changelog

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

- `detect_backend()` — torch, cuda, mps, accelerate, peft, datasets, wandb, rapidfuzz, mergekit

### Optional extras

| Extra | What it installs |
|---|---|
| `[torch]` | torch, transformers, safetensors, accelerate |
| `[lego]` | peft |
| `[merge]` | mergekit |
| `[data]` | datasets |
| `[align]` | rapidfuzz |
| `[logging]` | wandb |
| `[all]` | all of the above |
