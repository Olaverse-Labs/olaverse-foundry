# Quantization-aware training (QAT)

QAT inserts **fake quantization** into the forward pass during training: weights (and optionally activations) are rounded to int8/int4 and back, with a straight-through gradient. The model learns to be robust to quantization, so the final int8/int4 deployment keeps far more accuracy than post-training quantization.

It is **model-agnostic**: `prepare_qat` swaps the `nn.Linear` layers of *any* model (HF or custom), then you train it with **any** foundry trainer (MLM, encoder distillation, the head trainers, the distillers, …).

```bash
pip install "olaverse-foundry[torch]"
```

---

## Workflow

```python
from foundry import prepare_qat, QATConfig, export_quantized
from foundry import EncoderDistillTrainer, EncoderDistillConfig

# 1. Wrap any model in fake-quant linears
student = prepare_qat(student, QATConfig(weight_bits=8, per_channel=True))

# 2. Train it as usual with any trainer
EncoderDistillTrainer(student, teacher, EncoderDistillConfig(...)).train(pipe)

# 3. Export the quantization-robust model + a footprint report
report = export_quantized(student, "./model-int8", weight_bits=8)
print(report)
# {'orig_mb': ..., 'quant_mb': ..., 'compression': ..., 'weight_bits': 8, 'path': ...}
```

---

## `prepare_qat`

```python
prepare_qat(model, config=QATConfig(), skip=())
```

Replaces every `nn.Linear` in `model` (in place) with a fake-quantized QAT version, copying the original weights. `skip` is a tuple of name substrings to leave un-quantized (e.g. `skip=("lm_head",)`).

### `QATConfig`

| Field | Default | Description |
|---|---|---|
| `weight_bits` | `8` | `8` (int8) or `4` (int4) for weight fake-quant |
| `act_bits` | `0` | `0` disables activation quant; `8` enables dynamic int8 activations |
| `per_channel` | `True` | Per-output-channel weight scales (recommended) vs per-tensor |
| `symmetric` | `True` | Symmetric quantization (zero-point fixed at 0) |

---

## Export & inspection

### `export_quantized`

```python
export_quantized(model, path, weight_bits=8, per_channel=True, save_model=True)
```

Saves the (quantization-robust) model to `path`, writes `quantization.json` describing the scheme, and returns a footprint report comparing the original (bf16) vs quantized weight size and the compression ratio.

### `int8_state_dict`

```python
sd = int8_state_dict(model)
# {"<layer>.weight_int": int8 tensor, "<layer>.weight_scale": fp tensor, ...}
```

Real packed integer weights + per-channel scales, for hand-off to an int runtime (ONNX Runtime, ExecuTorch, TFLite). Non-quantized parameters are left as-is.

### `quantize_tensor`

```python
q, scale = quantize_tensor(weight, bits=8, per_channel=True)
```

Returns the packed int8 weights and the scale for a single tensor — useful for inspection.

!!! note "Where the int packing happens"
    QAT makes the model **robust** to quantization. The final on-device int8/int4 artifact is produced by your target runtime, which consumes `int8_state_dict(model)`. `export_quantized` reports the theoretical packed size and saves the trained weights.
