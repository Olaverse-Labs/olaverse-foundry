# CLI

`olaverse-foundry` ships a `foundry` CLI for environment checks, recipe previews, and runs. If the `foundry` script isn't on your `PATH` (e.g. inside some notebook kernels), use `python -m foundry` — it's equivalent.

```bash
pip install olaverse-foundry
foundry --help          # or: python -m foundry --help
```

Commands: `doctor`, `plan`, `run`, `embed`, `strategies`.

---

## `foundry doctor`

Check your environment and print a summary of available backends — foundry version, Python, torch/CUDA/MPS, and which optional packages are installed (accelerate, safetensors, peft, rapidfuzz, wandb). It also tells you whether real (GPU) training is enabled.

```bash
foundry doctor
```

Use it first on any new machine to confirm torch + CUDA are present.

---

## `foundry plan`

Preview a recipe without executing it — no GPU or model downloads. Prints the staged plan (seed, data, grow, teachers, heal, output) and any shape warnings.

```bash
foundry plan recipe.yaml
```

See [Recipes](recipes.md) for the YAML schema.

---

## `foundry run`

Execute a causal-LM recipe end-to-end (seed → optional grow → distil/heal → save). Requires `[torch]` and a GPU; with no torch it raises rather than silently degrading.

```bash
foundry run recipe.yaml
```

The recipe must provide training data (a `data:` block) — see [Recipes](recipes.md).

---

## `foundry embed`

Execute an embedding-distillation recipe (student encoder ← teacher embeddings). The recipe needs a `data:` block so the CLI can build the training pipeline.

```bash
foundry embed embed_recipe.yaml
```

---

## `foundry strategies`

List the available fusion strategies with descriptions.

```bash
foundry strategies
```

```
min_ce    MinCE  — per token, pick the teacher with highest p(gold).
mean_ce   MeanCE — weighted average over all teacher distributions.
```

---

## Help

Every command supports `--help`:

```bash
foundry --help
foundry run --help
python -m foundry doctor
```
