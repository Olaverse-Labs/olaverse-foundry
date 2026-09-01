# Changelog

---

## v0.3.0 — 2026-09-01

Minor, not patch: several changes alter results rather than only adding surface.
`sparse_kl` is on by default and shifts loss values slightly; `IdentityAlignment`
now sums colliding probability mass instead of dropping it, which changes the
distillation target; `compare_retrievers` defaults to `device="auto"` rather
than `"cuda"`; `params_m` is no longer rounded inside the result; 8-bit config
construction now requires bitsandbytes; and `mine_hard_negatives` omits the
negative key rather than emitting a false one. Pin `olaverse-foundry==0.2.1` if
you need the old behaviour.

### Fusion

- **Fix: `fusion_strategy="mean_ce"` silently ran MinCE.** The registry key is
  `"mean"`, but the docs, CLI help and recipe examples have always said
  `"mean_ce"` — and the trainers looked it up with
  `STRATEGY_REGISTRY.get(name, STRATEGY_REGISTRY["min_ce"])`, so the documented
  spelling fell through to the default. A run configured to average its teachers
  selected one per token instead, with nothing in the output to indicate it. The
  same silent fallback swallowed any typo. `"mean_ce"` is now an accepted alias,
  an unrecognised name raises `ValueError`, and `FusionConfig.strategy` accepts
  the documented spelling.

### Distillation loss

- **The KL is now computed over the teacher's top-k support** (`sparse_kl=True`,
  the default) instead of scattering it into a dense `(B, S, vocab)` target.
  A teacher returns ~8MB of top-k data at `B=8, S=2048, K=64`; the dense path
  turned that into ~10GB per teacher per step at a 152k vocabulary, in numpy,
  then copied it to the device again. `KL(T‖S)` has no contribution where
  `T(v) = 0`, so that was arithmetic on zeros. Measured at `V=152k`: **730×
  less memory and ~56× faster**. This affected the same-tokenizer path
  (`IdentityAlignment`) exactly as much as the cross-tokenizer one, so it was
  the binding constraint on any large-vocabulary run.
- The two paths differ very slightly: the dense one adds `1e-9` to *every* vocab
  entry before renormalising, smearing ~1.5e-4 of mass across the vocabulary.
  The sparse path renormalises the top-k mass over its own support. Against a
  dense reference without that smoothing the two agree to float32 precision.
  Pass `sparse_kl=False` to reproduce an older run.
- **Fix: `IdentityAlignment` and `EMAlignment` disagreed on collisions.**
  Identity used plain assignment, so when a token appeared twice in one top-k
  the earlier probability was silently dropped; EM used `np.add.at` and summed.
  Both now sum, which is correct — those are mass on the same token — and the
  sparse path deduplicates to match, including for a single teacher, since a
  `MinEDAlignment` maps several teacher tokens onto one student token.

### Large-model distillation

Three changes that together make a large-teacher → small-student run fit on one
machine. A 27B teacher is ~54GB in bf16; a 2B student full-weight trained needs
~30GB of weights, gradients and optimizer moments before activations.

- **New: `foundry.training.lora`** — LoRA training for the student.
  `attach_lora` freezes the base and returns a `PeftModel` that drops straight
  into any existing trainer, because every foundry trainer already trains
  whatever requires grad. There is deliberately no `LoRADistillTrainer`: LoRA is
  a property of the model, not a kind of training. `merge_and_save` produces a
  plain HuggingFace directory; `save_adapter` produces a PEFT directory;
  `to_skillpack` feeds the existing skill-pack machinery. Quantized bases are
  routed through `prepare_model_for_kbit_training`, without which the run trains
  at a flat loss and never errors.
- **`quantize` fails fast with an actionable error when bitsandbytes is absent.**
  `BitsAndBytesConfig.post_init()` reads
  `importlib.metadata.version("bitsandbytes")` on some transformers releases and
  not others, so the same call raised `PackageNotFoundError` on one version and
  succeeded on another — then failed much later inside `from_pretrained`.
  Neither message named what to install. foundry now checks up front, so the
  behaviour is identical on every transformers version.
- **`quantize="4bit"` / `"8bit"` now works for training, not just inference.**
  `io.loader` built `torch_dtype` and `device_map` but never a
  `quantization_config`, even though `inference.py` already constructed one — so
  a teacher could not be quantized through the library that loads it. `ModelRef`
  and `HFTeacher` both take `quantize` now, and the BitsAndBytes config is built
  in one shared place instead of two. A 27B teacher drops from ~54GB to ~15GB.
- **Fix: `HFTeacher.load()` called `.to(device)` on an already-placed model.**
  A bitsandbytes-quantized model refuses `.to()` outright, and a model dispatched
  by accelerate (`device_map="auto"`, the default) carries hooks that `.to()`
  invalidates — which is how a sharded teacher ends up half on the wrong device.
  It now detects placement and leaves those models where transformers put them.
- **Fix: 7 of 8 trainers built their optimizer over `model.parameters()`**,
  ignoring `requires_grad` (only `heads.py` filtered). AdamW skips parameters
  whose grad is `None`, so this was not a memory leak, but it made a LoRA or
  partially-frozen run unpredictable: clipping walked tensors that never had
  gradients, and anything later flipping `requires_grad` would silently start
  updating weights the caller froze. All trainers now share
  `training._params.trainable_parameters`.

### Teacher logit cache

- **`LogitCache` can now be disk-backed and bounded.** It was a plain in-memory
  dict, and `populate_dataset()` filled it for the whole dataset up front — at
  `top_k=64` with 8x512 batches that is ~21GB of RAM for only 41M tokens, which
  does not survive contact with a real run. Passing `cache_dir` caps RAM at
  `max_entries` and spills evictions to sharded `.npz` files, reading them back
  on a miss. The shard index lives in `cache_dir`, so a populated cache is
  reused by later processes rather than rebuilt.
- **Eviction is LRU, not FIFO.** Training reads batches in order, so FIFO evicted
  precisely the entries the next epoch read first: a cache smaller than the
  dataset achieved a ~0% hit rate and re-ran the teachers every epoch — the one
  cost the cache exists to avoid, with nothing in the output to indicate it.
- `stats` gains `disk_hits`, `spills` and `on_disk`.

### Data quality

- **New: `foundry.quality`** — the gate between synthesis and training.
  `clean_pairs` / `dedup_pairs` / `drop_degenerate_pairs` remove duplicate pairs,
  untranslated `anchor == positive` pairs and false negatives; `embedding_health`
  detects a collapsed encoder; `quality_report` / `print_quality_report`
  summarise a pair set before you train on it. numpy-only, so it runs on a core
  install. Language ID, translation adequacy and toxicity are deliberately out of
  scope — those need real models.
- **Fix: `mine_hard_negatives` could return a false negative.** It skipped
  duplicate candidates by index, but duplicate passages are common in translated
  corpora, so a "negative" textually identical to the pair's own positive could
  be selected — asking InfoNCE to push two identical strings apart. It now skips
  by normalised text, and omits the key entirely when no distinct candidate
  exists, so a missing negative is distinguishable from a bad one.

### Retrieval

- **Fix: `compare_retrievers` no longer defaults to `device="cuda"`.** The
  benchmark helper called `.to("cuda")` unconditionally, so it raised on any
  CPU-only machine. It now resolves `"auto"`.
- **Fix: `params_m` is no longer rounded inside the result.** Rounding to 1dp
  made every model under 50k params report `0.0` with the true count
  unrecoverable from the dict callers publish as a benchmark table. Rounding
  moved to `print_retrieval_comparison`.

### Recipes

- `EmbedRecipe.run()` validates `seed.model` up front. It was passed straight to
  `AutoModel.from_pretrained`, so an embed recipe missing that field failed deep
  inside transformers instead of naming the missing field. Unlike `FoundryRecipe`
  there is no random-init path for embeddings.

### Testing & CI

- **New: real-model CPU integration suite** (`tests/test_integration_cpu.py`).
  The rest of the suite runs on hand-rolled `nn.Module` stubs, which prove the
  training maths but never touch `AutoModel`, `config.json`, `save_pretrained` or
  the safetensors round-trip. These tests build genuine `BertModel` and tokenizer
  instances, save them as real HuggingFace directories, and run the pipeline
  end-to-end — including the README's claim that output reloads with
  `transformers` alone. No network, no GPU, ~9s.
- **New: `core-install` CI job.** Every existing job installed `[torch,lego,data,dev]`,
  so nothing verified the advertised `pip install olaverse-foundry`. On a core
  install the suite failed — `tests/test_m4.py` imported the PEFT weight bridge
  unguarded, erroring 14 tests instead of skipping them. Fixed, and now covered.
- **The import check moved to Python 3.9** (the version floor, not 3.11) and now
  walks every submodule rather than the top-level re-exports. The 0.2.1 bug — a
  runtime `X | None` alias breaking every `foundry.fusion` import on 3.9 — could
  not have been caught by a check running on 3.11.
- **New: `quality` CI job** running `ruff` and `mypy`. The library ships
  `py.typed` but nothing verified those annotations. Ruff is scoped to
  correctness rules only; pyupgrade is deliberately disabled, since it rewrites
  `Optional[X]` to `X | None` and would reintroduce the 0.2.0 break.
- 25 `raise ... from None` on errors that translate an opaque failure into an
  actionable one, so a chained `No module named 'torch'` no longer buries the
  message explaining how to fix it. 40 dead imports removed.

### Packaging

- **PEP 561 type marker** — `foundry/py.typed` ships in the wheel, so mypy and
  pyright now read the library's annotations instead of treating it as untyped.
- PyPI metadata gained `Documentation` and `Changelog` links, so the docs site is
  reachable from the project sidebar rather than only from inside the README.
- Releases are now tag-driven: pushing `vX.Y.Z` builds and publishes to PyPI via
  trusted publishing, and refuses to run if the tag and `pyproject.toml` disagree.

### Docs

- The docs site adopts the "Paper / Ink" theme shared with the Olaverse SDK docs
  and the marketing site.
- The landing-page version badge is generated from `foundry/__init__.py` at build
  time, so it can't drift behind a release.

---

## v0.2.1 — 2026-07-16

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
