"""
Lightweight inference helpers for foundry-built models.

foundry's job is to *build* models (distill / grow / fuse / adapt); the
`olaverse` SDK is the high-level serving layer. These helpers exist so you can
smoke-test a freshly trained model in the same session — load it (optionally
4-bit / 8-bit quantized), optionally snap a skill pack on, and generate text —
without pulling in a separate stack.

Example::

    from foundry.inference import load_for_inference, generate

    model, tok = load_for_inference("./grown_15b", quantize="4bit")
    print(generate(model, tok, "Explain SOLAR depth upscaling in one sentence."))
"""
from __future__ import annotations

from typing import Any, Optional


def load_for_inference(
    path:        str,
    quantize:    Optional[str] = None,
    device_map:  str = "auto",
    dtype:       str = "bfloat16",
    skillpack:   Optional[str] = None,
    skill_dir:   Optional[str] = None,
    trust_remote_code: bool = False,
):
    """
    Load a causal-LM for generation.

    Args:
        path:        Local dir or HF id of the model.
        quantize:    ``None`` | ``"4bit"`` | ``"8bit"`` (needs bitsandbytes).
        device_map:  Passed to ``from_pretrained`` (default ``"auto"``).
        dtype:       Compute dtype when not quantizing.
        skillpack:   Optional skill-pack name to merge before serving.
        skill_dir:   Directory the skill pack was saved to (PEFT format).
        trust_remote_code: Forwarded to transformers.

    Returns:
        ``(model, tokenizer)``.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers + torch are required for inference. "
            "Install with: pip install olaverse-foundry[torch]"
        )

    td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(dtype, torch.bfloat16)

    kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": trust_remote_code}

    if quantize in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "Quantized loading needs a recent transformers + bitsandbytes. "
                "Install with: pip install bitsandbytes"
            )
        if quantize == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=td,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize is not None:
        raise ValueError(f"quantize must be None, '4bit', or '8bit'; got {quantize!r}")
    else:
        kwargs["torch_dtype"] = td

    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    tok   = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)

    if skillpack is not None:
        if skill_dir is None:
            raise ValueError("skill_dir is required when skillpack is given.")
        _merge_skillpack(model, skillpack, skill_dir)

    model.eval()
    return model, tok


def _merge_skillpack(model, name: str, skill_dir: str) -> None:
    """Merge a PEFT-format skill pack into the live model weights."""
    from foundry.skillpacks import load_from_peft, SkillRegistry

    pack  = load_from_peft(skill_dir, name=name)
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    reg   = SkillRegistry(state)
    reg.register(pack)
    merged = reg.snap_on(name)

    import torch
    sd = model.state_dict()
    for k, v in merged.items():
        if k in sd:
            sd[k] = torch.as_tensor(v, dtype=sd[k].dtype, device=sd[k].device)
    model.load_state_dict(sd, strict=False)


def generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int   = 256,
    temperature:    float = 0.7,
    top_p:          float = 0.9,
    do_sample:      bool  = True,
) -> str:
    """Generate a completion for ``prompt`` and return the decoded new tokens."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
