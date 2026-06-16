# CLI

`olaverse-foundry` ships a `foundry` CLI for environment checks, recipe previews, and runs — no Python boilerplate required.

```bash
pip install olaverse-foundry
foundry --help
```

---

## `foundry doctor`

Check your environment and print a summary of available backends.

```bash
foundry doctor
```

```
olaverse-foundry v0.1.0 — environment check
────────────────────────────────────────────
torch          ✓  2.3.1+cu121
transformers   ✓  4.41.2
accelerate     ✓  0.30.1
safetensors    ✓  0.4.3
peft           ✓  0.11.0
datasets       ✓  2.19.1
wandb          ✗  not installed  (pip install wandb)
rapidfuzz      ✗  not installed  (pip install rapidfuzz)
mergekit       ✗  not installed  (pip install mergekit)

Device:        cuda  (NVIDIA A100 80GB, 4 GPUs)
Mixed prec:    bfloat16 supported
```

---

## `foundry plan`

Preview a recipe without executing it. No GPU or model downloads required.

```bash
foundry plan recipe.yaml
```

```
[foundry] Plan: meta-llama/Llama-3.1-8B → 15B (48 layers)
[foundry]   Teachers: meta-llama/Llama-3.1-70B (w=1.0)
[foundry]   Fusion: min_ce, cache: topk_64, align: min_ed
[foundry]   Heal: 100B tokens, alpha=0.3
[foundry]   Output: freeze base + ola_math, ola_code
[foundry] Estimated GPU hours: ~84h on 8×H100
```

```bash
# Validate schema only (exit 0 = valid, exit 1 = invalid)
foundry plan recipe.yaml --validate-only
```

---

## `foundry run`

Execute a causal LM factory recipe end-to-end.

```bash
foundry run recipe.yaml
foundry run recipe.yaml --output /custom/output/path
```

Progress is logged to stdout. Pass `--log-backend wandb` to also log to W&B.

---

## `foundry embed`

Execute an embedding distillation recipe.

```bash
foundry embed embed_recipe.yaml
foundry embed embed_recipe.yaml --output /checkpoints/embed-run
```

---

## `foundry strategies`

List all available fusion strategies with descriptions.

```bash
foundry strategies
```

```
Available fusion strategies
────────────────────────────────────────────
min_ce     Per-position: pick the teacher with lowest CE against ground truth.
           Best when teachers specialise in different domains.

mean_ce    Weighted average of all teacher distributions.
           Best for ensemble distillation when teachers are similar.
```

---

## `foundry backends`

Same as `foundry doctor` — alias for quick environment inspection.

```bash
foundry backends
```

---

## Global flags

| Flag | Description |
|---|---|
| `--help` | Show help |
| `--version` | Print foundry version |

All subcommands accept `--help`:

```bash
foundry run --help
foundry embed --help
```
