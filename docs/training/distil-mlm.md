# DistilMLMTrainer — combined distillation + MLM

`DistilMLMTrainer` trains a student encoder against a teacher with a *single* multi-part loss — **the DistilBERT objective** — so the distillation and the data-learning signals never fight each other (the failure mode of running distillation then MLM sequentially):

```
L = w_mlm · CE(student_mlm, masked_labels)               # learn from data
  + w_ce  · T² · KL(student_logits/T ‖ teacher_logits/T) # copy the teacher's soft predictions
  + w_cos · (1 − cos(student_hidden, teacher_hidden))    # align representations
```

Use it when you have a strong teacher **and** raw text in your target domain, and want the student to both mimic the teacher and keep learning from the data.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Contracts

- **student / teacher** — masked-LM models: `forward(input_ids=..., attention_mask=..., output_hidden_states=True)` returns `.logits` of shape `(B, S, V)` and `.hidden_states` (the last entry is the final encoder state). Any HF `AutoModelForMaskedLM` satisfies this.
- **Shared vocabulary** — student and teacher must share a vocab so the logit-level KL aligns. This holds when the student is warm-started from the teacher (fewer layers, same embeddings). Hidden sizes may differ — a trainable projection is added automatically for the cosine loss.
- **tokenizer** — pass one so the mask / pad / special token ids and vocab size are inferred; otherwise set `mask_token_id` (required), `pad_token_id`, and `vocab_size` in the config.

---

## Quick start

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForMaskedLM
from foundry import DataPipeline, DistilMLMTrainer, DistilMLMConfig

teacher = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased")
tok     = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

# 4-layer student sharing the teacher's vocab
student = BertForMaskedLM(BertConfig(
    vocab_size=tok.vocab_size, hidden_size=312, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1200, max_position_embeddings=512,
))

pipe = DataPipeline(text_rows, tokenizer=tok, text_column="text",
                    batch_size=16, max_length=128, mode="embed")

trainer = DistilMLMTrainer(student, teacher, tokenizer=tok, config=DistilMLMConfig(
    device="cuda", torch_dtype="bfloat16",
    epochs=1, lr_scheduler="cosine", warmup_steps=100,
))
result = trainer.train(pipe)
print(result["losses"][-1])

student.save_pretrained("./my-distil-base")
tok.save_pretrained("./my-distil-base")
```

---

## Config — `DistilMLMConfig`

| Field | Default | Description |
|---|---|---|
| `mlm_weight` | `2.0` | Weight of the MLM cross-entropy term |
| `distill_weight` | `5.0` | Weight of the KL distillation term |
| `cosine_weight` | `1.0` | Weight of the hidden-state cosine term |
| `temperature` | `2.0` | Softmax temperature for the KL term |
| `mask_prob` | `0.15` | Fraction of tokens masked |
| `mask_token_id` | `None` | Required if no tokenizer is given |
| `pad_token_id` | `0` | Used to derive the attention mask if absent |
| `vocab_size` | `None` | Inferred from the tokenizer if `None` |
| `learning_rate` | `5e-5` | AdamW LR |
| `epochs` | `1` | Passes over the dataset |
| `weight_decay` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `device` | `"auto"` | Device selection |
| `grad_accumulation_steps` | `1` | Accumulate over N batches |
| `torch_dtype` | `"float32"` | Mixed precision |
| `lr_scheduler` / `warmup_steps` | `"cosine"` / `0` | Schedule |
| `eval_every` | `0` | Eval cadence (0 = off) |
| `save_every` / `save_dir` | `0` / `""` | Auto-checkpoint |
| `log_every` / `log_backend` | `50` / `"none"` | Logging |
| `seed` | `42` | Reproducibility |

The DistilBERT default loss weights (`2 / 5 / 1`) are a good starting point.

`train()` has the same signature and return value as the other trainers; `save_checkpoint` / `resume_from_checkpoint` persist the student, the projector (if created), and optimizer state.

---

## vs the other encoder trainers

| You have | Use |
|---|---|
| Only raw text, no teacher | [`MLMTrainer`](mlm.md) |
| A teacher, little raw text | [`EncoderDistillTrainer`](encoder-distill.md) |
| A teacher **and** in-domain raw text (shared vocab) | `DistilMLMTrainer` |
