# YAML Recipes

Recipes let you describe an entire model factory pipeline in a single validated YAML file. The recipe is parsed and validated by Pydantic before any GPU is touched — fail fast on config errors, not after hours of compute.

---

## Causal LM recipe

```yaml
# recipe.yaml
seed:
  model: meta-llama/Llama-3.1-8B
  init: pretrained                # "pretrained" | "random"

grow:
  method: depth_upscale
  to_params: 15B                  # target parameter count

teachers:
  - role: reasoning
    model: meta-llama/Llama-3.1-70B
    weight: 1.0
  - role: coding
    model: Qwen/Qwen2.5-72B-Instruct
    weight: 0.8

fusion:
  strategy: min_ce                # "min_ce" | "mean_ce"
  align: min_ed                   # "min_ed" | "identity"
  cache: topk_64                  # top-k logit cache

heal:
  tokens: 100B                    # healing corpus size
  alpha: 0.3                      # CE weight (1-alpha = KL weight)

output:
  freeze_base: true
  skillpacks:
    - ola_math
    - ola_code
    - ola_reason
```

---

## Embedding recipe

```yaml
# embed_recipe.yaml
student:
  model: microsoft/deberta-v3-base
  pool: mean                      # "mean" | "cls"
  normalize: true

teacher:
  model: BAAI/bge-large-en-v1.5
  pool: mean
  normalize: true

training:
  loss: cosine                    # "cosine" | "mse"
  temperature: 0.05
  epochs: 3
  learning_rate: 2e-5
  lr_scheduler: cosine
  warmup_steps: 200
  torch_dtype: bfloat16
  save_every: 1000
  save_dir: /checkpoints/embed

data:
  source: sentence-transformers/natural-questions
  split: train
  text_column: query
  batch_size: 32
  max_length: 128
  shuffle_buffer: 10000
```

---

## Loading a recipe

```python
from foundry import Recipe, FoundryRecipe, EmbedRecipe

# Auto-detect type
recipe = Recipe.load("recipe.yaml")
print(recipe.plan())   # preview without executing

# Explicit types
causal = FoundryRecipe.load("recipe.yaml")
embed  = EmbedRecipe.load("embed_recipe.yaml")
```

---

## Previewing a recipe plan

```python
for line in recipe.plan():
    print(line)
```

```
[foundry] Plan: meta-llama/Llama-3.1-8B → 15B (48 layers)
[foundry]   Teachers: meta-llama/Llama-3.1-70B (w=1.0), Qwen/Qwen2.5-72B (w=0.8)
[foundry]   Fusion: min_ce, cache: topk_64
[foundry]   Heal: 100B tokens, alpha=0.3
[foundry]   Output: freeze base + 3 skill packs
[foundry] Estimated GPU hours: ~84h on 8×H100
```

---

## Running a recipe

```python
result = recipe.run()
```

Or via CLI:

```bash
foundry plan recipe.yaml    # preview only
foundry run recipe.yaml     # execute
foundry embed recipe.yaml   # embedding recipe
```

---

## Validation

Recipes are validated with Pydantic v2 on load. Invalid configs raise immediately:

```python
from foundry import FoundryRecipe

recipe = FoundryRecipe.load("recipe.yaml")
# Raises ValidationError if:
#   - unknown fusion strategy
#   - to_params is not parseable
#   - teacher weight is negative
#   - alpha is outside [0, 1]
```

This means you can CI-validate your recipe files without any GPU or model downloads:

```bash
foundry plan recipe.yaml --validate-only
```
