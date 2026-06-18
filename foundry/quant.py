"""
Quantization-aware training (QAT) for on-device base encoders.

QAT inserts *fake quantization* into the forward pass during training: weights
(and optionally activations) are rounded to int8/int4 and back, with a
straight-through gradient. The model learns to be robust to quantization, so the
final int8/int4 deployment keeps far more accuracy than post-training
quantization — the "quality-per-MB" story for on-device models.

It is **model-agnostic**: ``prepare_qat`` swaps the ``nn.Linear`` layers of *any*
model (HF or custom) for QAT versions, then you train it with any foundry trainer
(MLMTrainer, EncoderDistillTrainer, the head trainers, …). After training,
``int8_state_dict`` gives real packed int weights + scales for your runtime, and
``export_quantized`` saves the model plus a footprint report.

Example::

    from foundry import prepare_qat, QATConfig, export_quantized
    from foundry import EncoderDistillTrainer, EncoderDistillConfig

    student = prepare_qat(student, QATConfig(weight_bits=8))   # any model
    EncoderDistillTrainer(student, teacher, EncoderDistillConfig(...)).train(pipe)
    report = export_quantized(student, "./base-int8", weight_bits=8)
    print(report)   # {'orig_mb':..., 'quant_mb':..., 'compression':...}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QATConfig:
    """QAT settings.

    weight_bits: 8 (int8) or 4 (int4) for weight fake-quant.
    act_bits:    0 disables activation quant; 8 enables dynamic int8 activations.
    per_channel: per-output-channel weight scales (recommended) vs per-tensor.
    symmetric:   symmetric quant (zero-point fixed at 0).
    """
    weight_bits: int  = 8
    act_bits:    int  = 0
    per_channel: bool = True
    symmetric:   bool = True


def _qrange(bits: int):
    # symmetric signed range, reserving -2**(b-1) (e.g. int8 -> [-127, 127])
    qmax = (1 << (bits - 1)) - 1
    return -qmax, qmax


def _make_ste():
    """Build the straight-through fake-quant autograd Function (lazy; needs torch)."""
    import torch

    class _FakeQuantSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale, qmin, qmax):
            q = torch.clamp(torch.round(x / scale), qmin, qmax)
            return q * scale

        @staticmethod
        def backward(ctx, grad_out):
            # straight-through: gradient passes unchanged to x
            return grad_out, None, None, None

    return _FakeQuantSTE


def _weight_scale(weight, per_channel: bool, qmax: int):
    import torch
    if per_channel:
        amax = weight.detach().abs().amax(dim=1, keepdim=True)
    else:
        amax = weight.detach().abs().max()
    return (amax / qmax).clamp(min=1e-8)


def quantize_tensor(weight, bits: int, per_channel: bool = True):
    """Return (int_weights, scale) — real packed integer weights for export."""
    import torch
    qmin, qmax = _qrange(bits)
    scale = _weight_scale(weight, per_channel, qmax)
    q = torch.clamp(torch.round(weight / scale), qmin, qmax).to(torch.int8)
    return q, scale


class QATLinear:
    """nn.Linear with fake-quantized weights (and optional activations).

    Constructed via :func:`prepare_qat`; not usually instantiated directly.
    """

    def __new__(cls, in_features, out_features, bias=True, config: QATConfig | None = None):
        import torch.nn as nn

        cfg = config or QATConfig()
        ste = _make_ste()
        w_qmin, w_qmax = _qrange(cfg.weight_bits)
        a_qmin, a_qmax = _qrange(cfg.act_bits) if cfg.act_bits > 0 else (0, 0)

        class _QATLinear(nn.Linear):
            def __init__(self, i, o, b):
                super().__init__(i, o, bias=b)
                self.qat_config = cfg

            def forward(self, x):
                import torch
                import torch.nn.functional as F
                if cfg.act_bits > 0:
                    a_scale = (x.detach().abs().max() / a_qmax).clamp(min=1e-8)
                    x = ste.apply(x, a_scale, a_qmin, a_qmax)
                w_scale = _weight_scale(self.weight, cfg.per_channel, w_qmax)
                wq = ste.apply(self.weight, w_scale, w_qmin, w_qmax)
                return F.linear(x, wq, self.bias)

        return _QATLinear(in_features, out_features, bias)


def prepare_qat(model, config: QATConfig | None = None, skip: tuple = ()):
    """
    Replace every ``nn.Linear`` in ``model`` with a fake-quantized QAT version
    (in place), copying the original weights. Train the returned model with any
    foundry trainer, then export with :func:`export_quantized`.

    Args:
        model:  Any ``nn.Module`` (HF or custom).
        config: QATConfig (defaults to int8, per-channel, weight-only).
        skip:   Substrings of module names to leave un-quantized (e.g. ("lm_head",)).

    Returns:
        The same model with QAT linears swapped in.
    """
    try:
        import torch  # noqa: F401
        import torch.nn as nn
    except ImportError:
        raise ImportError("torch is required for QAT. Install with: pip install olaverse-foundry[torch]")

    cfg = config or QATConfig()

    def _swap(module, prefix=""):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                qat = QATLinear(child.in_features, child.out_features,
                                bias=child.bias is not None, config=cfg)
                qat.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    qat.bias.data.copy_(child.bias.data)
                qat.to(child.weight.device)
                setattr(module, name, qat)
            else:
                _swap(child, full)

    _swap(model)
    return model


def int8_state_dict(model, config: QATConfig | None = None) -> dict:
    """
    Return real packed integer weights + scales for QAT linears, e.g.

        {"<layer>.weight_int": int8 tensor, "<layer>.weight_scale": float tensor, ...}

    for hand-off to an int runtime (ONNX/ExecuTorch/TFLite). Non-QAT params are
    left as-is.
    """
    import torch.nn as nn
    cfg = config or QATConfig()
    out: dict[str, Any] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and type(mod).__name__ == "_QATLinear":
            q, scale = quantize_tensor(mod.weight, cfg.weight_bits, cfg.per_channel)
            out[f"{name}.weight_int"]   = q
            out[f"{name}.weight_scale"] = scale
            if mod.bias is not None:
                out[f"{name}.bias"] = mod.bias.detach()
    return out


def export_quantized(model, path: str | Path, weight_bits: int = 8,
                     per_channel: bool = True, save_model: bool = True) -> dict:
    """
    Save the QAT model and report the quantized footprint.

    Saves the (quantization-robust) model to ``path`` and writes
    ``quantization.json`` describing the scheme. Returns a footprint report with
    the original vs quantized weight size and the compression ratio.

    Note: this packs the *theoretical* int size; the final on-device artifact is
    produced by your target runtime (ONNX/ExecuTorch/TFLite), which consumes
    ``int8_state_dict(model)``.
    """
    import json
    import torch.nn as nn

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    orig_bytes = quant_bytes = 0
    for _name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            n = mod.weight.numel()
            orig_bytes  += n * 2                         # bf16/fp16 baseline
            quant_bytes += n * weight_bits / 8           # int8/int4 weights
            if per_channel:
                quant_bytes += mod.weight.shape[0] * 4   # per-channel scales (fp32)

    if save_model:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(p))
        else:
            import torch
            torch.save(model.state_dict(), p / "model.pt")

    meta = {"weight_bits": weight_bits, "per_channel": per_channel,
            "scheme": "symmetric-signed", "note": "QAT fake-quant; pack via runtime"}
    (p / "quantization.json").write_text(json.dumps(meta, indent=2))

    report = {
        "orig_mb":     round(orig_bytes / 1e6, 2),
        "quant_mb":    round(quant_bytes / 1e6, 2),
        "compression": round(orig_bytes / max(1, quant_bytes), 2),
        "weight_bits": weight_bits,
        "path":        str(p),
    }
    return report
