# Growth & Scaling

`olaverse-foundry` implements SOLAR-style depth up-scaling: duplicate existing layers from a pretrained model to grow it to a target parameter count, then heal with distillation.

```bash
pip install "olaverse-foundry[merge]"   # for mergekit passthrough
```

---

## Concepts

The depth up-scaling approach (from the [SOLAR paper](https://arxiv.org/abs/2312.15166)) duplicates transformer layers rather than reinitialising them. The duplicate layers start with the same weights as the originals — so the model still produces reasonable output immediately. A short healing run (distillation or SFT) then merges the representations.

```
7B (32 layers) → duplicate 16 layers → 15B (48 layers) → heal → production model
```

---

## Planning a growth

```python
from foundry import plan_growth, GrowthPlan

plan = plan_growth(
    source_model  = "meta-llama/Llama-3.1-8B",
    target_params = 15e9,
    method        = "depth_upscale",
)

print(plan)
# GrowthPlan(
#   source_layers=32, target_layers=48,
#   duplicate_map={32: 16, 33: 17, ...},
#   estimated_params=14.8B
# )
```

---

## Layer map

```python
from foundry import upscale_layer_map

layer_map = upscale_layer_map(source_layers=32, target_layers=48)
# {0: 0, 1: 1, ..., 31: 31, 32: 16, 33: 17, ..., 47: 31}
```

The first 32 layers are identity-mapped; layers 32–47 duplicate layers 16–31.

---

## Estimating target layer count

```python
from foundry import layers_for_param_target

n_layers = layers_for_param_target(
    source_model  = "meta-llama/Llama-3.1-8B",
    target_params = 15e9,
)
# → 48
```

---

## Generating mergekit YAML

```python
from foundry import growth_plan_to_mergekit_yaml, save_mergekit_config

yaml_str = growth_plan_to_mergekit_yaml(plan, base_model="meta-llama/Llama-3.1-8B")
save_mergekit_config(yaml_str, "/tmp/merge_config.yml")
```

The generated YAML is directly usable with `mergekit-yaml`:

```bash
mergekit-yaml /tmp/merge_config.yml /output/llama-15b
```

---

## Running the merge

```python
from foundry import run_merge

run_merge(
    config_path  = "/tmp/merge_config.yml",
    output_path  = "/output/llama-15b",
)
```

Requires `mergekit` installed (`pip install mergekit`).

---

## Full pipeline example

```python
from foundry import plan_growth, growth_plan_to_mergekit_yaml, save_mergekit_config, run_merge

# 1. Plan
plan = plan_growth("meta-llama/Llama-3.1-8B", target_params=15e9)

# 2. Generate mergekit config
yaml_str = growth_plan_to_mergekit_yaml(plan, "meta-llama/Llama-3.1-8B")
save_mergekit_config(yaml_str, "/tmp/merge.yml")

# 3. Run merge (creates the grown model on disk)
run_merge("/tmp/merge.yml", "/output/llama-15b-raw")

# 4. Heal with distillation
from transformers import AutoModelForCausalLM
from foundry import TorchDistillTrainer, TorchTrainConfig, TeacherRegistry, HFTeacher

grown  = AutoModelForCausalLM.from_pretrained("/output/llama-15b-raw")
teachers = TeacherRegistry([HFTeacher("meta-llama/Llama-3.1-8B", weight=1.0)])
teachers.load_all()

trainer = TorchDistillTrainer(
    student  = grown,
    teachers = teachers,
    config   = TorchTrainConfig(
        epochs       = 1,
        alpha        = 0.3,
        lr_scheduler = "cosine",
        torch_dtype  = "bfloat16",
    ),
)
trainer.train(healing_dataset)
grown.save_pretrained("/output/llama-15b-healed")
```

---

## Vocabulary alignment

When distilling across tokenizers (e.g. teacher uses BPE, student uses WordPiece), use `MinEDAlignment` to map teacher token IDs to the nearest student tokens:

```python
from foundry import MinEDAlignment

alignment = MinEDAlignment(teacher_tokenizer, student_tokenizer)

trainer = TorchDistillTrainer(
    student   = student,
    teachers  = teachers,
    alignment = alignment,
    config    = TorchTrainConfig(...),
)
```

`MinEDAlignment` uses edit distance to find the closest-surface-form mapping between vocabularies. Install `rapidfuzz` for ~100× speedup:

```bash
pip install "olaverse-foundry[align]"
```
