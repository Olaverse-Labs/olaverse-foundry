# Growth & scaling

`olaverse-foundry` implements SOLAR-style depth up-scaling: duplicate existing layers from a pretrained model to grow it to a target parameter count, then heal with distillation. The merge runs **natively** (transformers + safetensors) — no external merge tool required — and the layer prefix is **auto-detected**, so it works on Llama, BERT, GPT-2, and other architectures.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Concepts

The depth up-scaling approach (from the [SOLAR paper](https://arxiv.org/abs/2312.15166)) duplicates transformer layers rather than reinitialising them. The duplicated layers start with the same weights as the originals, so the grown model still produces reasonable output immediately. A short healing run (distillation or SFT) then adapts the extra capacity.

```
8B (32 layers) → duplicate layers → 15B (48 layers) → heal → production model
```

---

## Plan a growth

`plan_growth` takes an `ArchConfig` describing the seed and a target parameter count, and returns a `GrowthPlan`:

```python
from foundry import plan_growth
from foundry.contracts import ArchConfig
from transformers import AutoConfig

hf   = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
arch = ArchConfig(
    n_layers   = hf.num_hidden_layers,
    d_model    = hf.hidden_size,
    vocab_size = hf.vocab_size,
    d_ff       = getattr(hf, "intermediate_size", 0) or 0,
)

plan = plan_growth(arch, to_params=15e9)
print(plan.summary())
# Growth Plan: 32 → 48 layers (1.50×)
# Layer map: [0, 1, ..., 31, 16, 17, ..., 31]
```

`GrowthPlan` fields: `src_layers`, `target_layers`, `layer_map` (source layer index per output position), `scale_factor`, and `shape_warning` (set when the resulting shape is unusually deep/narrow).

### Helpers

```python
from foundry import upscale_layer_map, layers_for_param_target

upscale_layer_map(n_src=32, n_target=48)          # → [0, 1, ..., 31, 16, ..., 31]
layers_for_param_target(arch, target_params=15e9) # → (48, warning_or_None)
```

---

## Run the merge (native)

`run_merge` loads the seed, rebuilds it at the grown depth, copies the duplicated layers, and saves a standard HF model directory — loadable with `AutoModelForCausalLM.from_pretrained`.

```python
from foundry import run_merge

run_merge(plan, "meta-llama/Llama-3.1-8B", "./grown-15b", dtype="bfloat16")
# → Path("./grown-15b")   (no external merge tool needed)
```

A passthrough YAML is also written alongside the output for reference/interop.

---

## Any architecture — layer-prefix detection

The transformer block list lives under a different prefix per architecture. `detect_layer_prefix` finds it automatically, and `build_upscaled_state_dict` uses it under the hood:

```python
from foundry import detect_layer_prefix, build_upscaled_state_dict, upscale_layer_map

detect_layer_prefix(state_dict)   # "model.layers" (Llama) | "encoder.layer" (BERT) | "transformer.h" (GPT-2)

new_state = build_upscaled_state_dict(
    src_state_dict = model.state_dict(),
    layer_map      = upscale_layer_map(n_src, n_target),
    layer_prefix   = None,            # None = auto-detect
)
```

This is what makes growth work on encoders (BERT-style backbones) as well as decoders.

---

## Passthrough YAML (interop)

For interoperability you can emit the passthrough slice config as a dict / file (pure Python, no dependency):

```python
from foundry import growth_plan_to_mergekit_yaml, save_mergekit_config

cfg  = growth_plan_to_mergekit_yaml(plan, seed_path="meta-llama/Llama-3.1-8B", dtype="bfloat16")
path = save_mergekit_config(plan, "meta-llama/Llama-3.1-8B", "./passthrough.yaml")
```

---

## Heal the grown model

A freshly grown model under-performs the seed until healed. Feed it back into a distiller as the student:

```python
from transformers import AutoModelForCausalLM
from foundry import TorchDistillTrainer, TorchTrainConfig, TeacherRegistry, HFTeacher

grown    = AutoModelForCausalLM.from_pretrained("./grown-15b")
teachers = TeacherRegistry([HFTeacher("meta-llama/Llama-3.1-8B", weight=1.0)])
teachers.load_all()

trainer = TorchDistillTrainer(grown, teachers, TorchTrainConfig(
    epochs=1, alpha=0.3, lr_scheduler="cosine", torch_dtype="bfloat16",
))
trainer.train(healing_dataset)
grown.save_pretrained("./grown-15b-healed")
```

---

## Cross-tokenizer alignment

When distilling across tokenizers (e.g. teacher uses BPE, student uses WordPiece), use `MinEDAlignment` to map teacher token IDs to the nearest student tokens via edit distance:

```python
from foundry import MinEDAlignment

alignment = MinEDAlignment(teacher_tokenizer, student_tokenizer)
trainer   = TorchDistillTrainer(student, teachers, alignment=alignment, config=TorchTrainConfig(...))
```

Install `rapidfuzz` for a ~100× speedup:

```bash
pip install "olaverse-foundry[align]"
```

Also available: `IdentityAlignment` (shared tokenizer, fast path) and `EMAlignment` (exact surface-form match).
