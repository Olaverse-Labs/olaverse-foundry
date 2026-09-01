# LoRA Training

Full-weight training of a 2B student needs the weights, their gradients and two
AdamW moments — roughly 30GB before activations. LoRA trains well under 1% of
that. Combined with a 4-bit teacher, a 27B → 2B distillation fits on hardware you
can actually rent.

There is no `LoRADistillTrainer`, on purpose. Every foundry trainer takes an
`nn.Module` student and trains whatever in it requires grad, so LoRA is a
property of the *model*, not a kind of training. Wrap the student once and hand
it to the trainer you were already using.

Requires `pip install olaverse-foundry[lego]`.

## The 27B → 2B run

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from foundry import TeacherRegistry, HFTeacher, TorchDistillTrainer, TorchTrainConfig
from foundry.data import DataPipeline
from foundry.training.lora import attach_lora, LoRAConfig, merge_and_save, print_trainable_summary

# Teacher: forward-only, so quantize it. ~54GB in bf16, ~15GB in 4-bit.
teachers = TeacherRegistry([
    HFTeacher("Qwen/Qwen3-27B", quantize="4bit").load()
])

student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-2B", torch_dtype="bfloat16")
student = attach_lora(student, LoRAConfig(
    rank=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
))
print_trainable_summary(student)
#   trainable: 33,554,432 / 2,027,175,936  (1.6552%)

tok  = AutoTokenizer.from_pretrained("Qwen/Qwen3-2B")
pipe = DataPipeline(my_dataset, tokenizer=tok, batch_size=4, max_length=1024, mode="lm")

cfg = TorchTrainConfig(device="cuda", epochs=1, torch_dtype="bfloat16",
                       learning_rate=1e-4, lr_scheduler="cosine", warmup_steps=100)
TorchDistillTrainer(student, teachers, config=cfg).train(pipe)

merge_and_save(student, "./qwen3-2b-distilled", tokenizer=tok)
```

Both models are the same family, so their tokenizers match and the default
`IdentityAlignment` is correct. Distilling across families needs the vocabulary
mapping in `foundry.fusion`.

## Checking it attached

A LoRA run that silently failed to attach still trains — at full cost, updating
everything. A run that froze too much trains at a flat loss. Neither errors, so
assert on it:

```python
from foundry.training.lora import trainable_summary

s = trainable_summary(student)
assert 0.1 < s["percent"] < 10, s     # not 100 (didn't attach), not 0 (froze all)
```

## Choosing `target_modules`

`None` lets peft pick the architecture defaults — attention projections only,
which is right for style and format adaptation. **For distillation, name the MLP
projections too.** Most of a transformer's capacity lives in the MLP, and
capability transfer that cannot touch it is working with one hand tied.

| Goal | rank | target_modules |
| --- | --- | --- |
| Format / tone | 8–16 | `None` (attention defaults) |
| Domain adaptation | 16–32 | attention + `down_proj` |
| Capability distillation | 32–64 | attention + all MLP projections |

## Encoders

For `ContrastiveTrainer` or `EncoderDistillTrainer`, set the task type:

```python
student = attach_lora(encoder, LoRAConfig(rank=16, task_type="FEATURE_EXTRACTION"))
```

## Output

| Function | Produces | Use when |
| --- | --- | --- |
| `save_adapter(model, path)` | PEFT adapter directory, a few MB | Sharing, or serving on top of a quantized base |
| `merge_and_save(model, path)` | Plain HF directory, no peft needed at load | Deployment |
| `to_skillpack(model, name)` | A foundry `SkillPack` | Feeding the skill-pack registry and composition |

`merge_and_save` **raises on a quantized base**. Merging a full-precision update
into 4-bit weights cannot round-trip, so the honest options are to serve the
adapter on top of the quantized base, or to reload the base in bf16, attach the
adapter there, and merge that.

## Gradient checkpointing

```python
student = attach_lora(student, cfg, gradient_checkpointing=True)
```

Trades compute for activation memory. For a multi-billion-parameter student on
one card it is usually the difference between fitting and not. On a quantized
base this is routed through peft's `prepare_model_for_kbit_training`, which is
not optional — without it the frozen 4-bit inputs never require grad, nothing
upstream of the adapters produces a gradient, and the run trains at a flat loss
with no error.
