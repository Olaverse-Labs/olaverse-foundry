# Inference

`foundry.inference` is a small helper for generating text from a model you've built — optionally 4-bit/8-bit quantized, and optionally with a [skill pack](skillpacks.md) merged in.

```bash
pip install "olaverse-foundry[torch]"
# 4-bit / 8-bit loading also needs bitsandbytes:
pip install bitsandbytes
```

---

## `load_for_inference`

```python
from foundry import load_for_inference, generate

model, tok = load_for_inference("./my-model", dtype="bfloat16")
print(generate(model, tok, "The history of the printing press began", max_new_tokens=60))
```

```python
load_for_inference(
    path,
    quantize          = None,        # None | "4bit" | "8bit"
    device_map        = "auto",
    dtype             = "bfloat16",  # used when not quantizing
    skillpack         = None,        # name of a skill pack to merge before serving
    skill_dir         = None,        # PEFT-format dir the pack was saved to
    trust_remote_code = False,
)
# → (model, tokenizer)
```

| Param | Description |
|---|---|
| `path` | Local directory or HF id of the model |
| `quantize` | `"4bit"` / `"8bit"` quantized load via bitsandbytes (NF4 + double-quant for 4-bit), or `None` |
| `device_map` | Forwarded to `from_pretrained` |
| `dtype` | Compute dtype when not quantizing |
| `skillpack` / `skill_dir` | Merge a saved LoRA skill pack into the weights before serving |

### Quantized load

```python
model, tok = load_for_inference("./my-model", quantize="4bit")
```

### With a skill pack merged in

```python
model, tok = load_for_inference("./my-model", skillpack="my_pack", skill_dir="./packs/my_pack")
```

---

## `generate`

```python
generate(
    model, tokenizer, prompt,
    max_new_tokens = 256,
    temperature    = 0.7,
    top_p          = 0.9,
    do_sample      = True,
)
# → decoded string of the newly generated tokens
```

```python
text = generate(model, tok, "Explain depth up-scaling in one sentence.",
                max_new_tokens=80, do_sample=False)
```
