# Skill Packs

Skill packs are detachable LoRA adapters that are cryptographically bound to a specific base model. Apply a math, code, or reasoning adapter to any compatible base — then detach it cleanly when you no longer need it.

```bash
pip install "olaverse-foundry[torch,lego]"
```

---

## Concepts

| Term | Description |
|---|---|
| `SkillPack` | A LoRA adapter (rank, alpha, target modules, weights) bound to a base model hash |
| `SkillRegistry` | A named collection of `SkillPack`s that can be applied or snapped onto a model state dict |
| `snap_on()` | Apply one or more named packs to a state dict, returning the merged weights |
| `snap_off()` | Remove applied packs, returning the base state dict |

---

## Creating a SkillPack

```python
from foundry import SkillPack, SkillRegistry

# Define a LoRA adapter for query and value projections
math_pack = SkillPack(
    name           = "ola_math",
    base_model_ref = "meta-llama/Llama-3.1-8B",
    rank           = 16,
    alpha          = 32.0,
    target_modules = ["q_proj", "v_proj"],
    lora_weights   = {
        "q_proj": (lora_A_q, lora_B_q),   # (rank, hidden), (hidden, rank) tensors
        "v_proj": (lora_A_v, lora_B_v),
    },
)

registry = SkillRegistry(base_model_state_dict, packs={"ola_math": math_pack})
```

---

## Applying packs

```python
# Apply one or more named packs to the base state dict
merged_state = registry.snap_on("ola_math")

# Apply multiple packs at once (merged sequentially)
merged_state = registry.snap_on("ola_math", "ola_code")

# Load the merged weights into your model
model.load_state_dict(merged_state)
```

`snap_on()` correctly handles both bare module names (`"q_proj"`) and fully qualified HuggingFace keys (`"model.layers.0.self_attn.q_proj.weight"`). It searches right-to-left through each key's parts to find the matching module name.

---

## Detaching packs

```python
# Returns the original base state dict (no LoRA deltas)
base_state = registry.snap_off()
model.load_state_dict(base_state)
```

---

## PEFT format round-trip

Save and load adapters in PEFT format (no `peft` library required at save time):

```python
from foundry import save_as_peft, load_from_peft, peft_config_dict

# Save
save_as_peft(math_pack, "/adapters/ola_math")

# Load
math_pack = load_from_peft("/adapters/ola_math")

# Get the PEFT config as a dict (for peft library compatibility)
cfg = peft_config_dict(math_pack)
```

---

## Target module matching

`snap_on()` uses right-to-left key matching to handle HuggingFace's deeply nested state dict keys:

```
# Bare name in target_modules:  "q_proj"
# HF key in state dict:  "model.layers.7.self_attn.q_proj.weight"

# Split by ".":  ["model", "layers", "7", "self_attn", "q_proj", "weight"]
# Search reversed:  "weight" → no, "q_proj" → MATCH ✓
```

This means you can define `target_modules = ["q_proj", "v_proj"]` and it will correctly match all layers, regardless of depth.

---

## Example: train LoRA then package as a skill

```python
import torch
from peft import get_peft_model, LoraConfig
from foundry import SkillPack, SkillRegistry, save_as_peft

# 1. Train with PEFT
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
peft_cfg   = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(base_model, peft_cfg)

# ... training ...

# 2. Extract LoRA weights
lora_weights = {
    name: (param.lora_A["default"].weight.data,
           param.lora_B["default"].weight.data)
    for name, param in peft_model.named_modules()
    if hasattr(param, "lora_A")
}

# 3. Package as SkillPack
pack = SkillPack(
    name           = "ola_math_v1",
    base_model_ref = "meta-llama/Llama-3.1-8B",
    rank           = 16,
    alpha          = 32.0,
    target_modules = ["q_proj", "v_proj"],
    lora_weights   = lora_weights,
)

save_as_peft(pack, "/adapters/ola_math_v1")
```
