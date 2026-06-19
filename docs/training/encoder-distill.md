# EncoderDistillTrainer — token-level encoder distillation

`EncoderDistillTrainer` builds a base encoder by **distilling a strong teacher encoder's per-token hidden states** into your smaller architecture (the DistilBERT / MiniLM-style recipe). Because every token position is supervised — not just the pooled sentence vector — the student keeps token-level representations, so token-classification heads (NER) work on the result.

Use it to compress a strong existing encoder into your own architecture, especially when you have limited raw text: distillation is far more data-efficient than MLM-from-scratch.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Contracts

- **student** — `student(input_ids=..., attention_mask=...)` returns `.last_hidden_state` of shape `(B, S, D_student)`.
- **teacher** — same call shape, returns `.last_hidden_state` of shape `(B, S, D_teacher)` (any HF `AutoModel` encoder). If the hidden sizes differ, a trainable linear projection `D_student → D_teacher` is added automatically.

---

## Quick start

```python
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
from foundry import DataPipeline, EncoderDistillTrainer, EncoderDistillConfig

teacher = AutoModel.from_pretrained("intfloat/multilingual-e5-base")
t_tok   = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

# small student; share the teacher's vocab so its token ids are valid
student = BertModel(BertConfig(
    vocab_size=t_tok.vocab_size, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1024, max_position_embeddings=512,
))

pipe = DataPipeline(text_rows, tokenizer=t_tok, text_column="text",
                    batch_size=16, max_length=128, mode="embed")

trainer = EncoderDistillTrainer(student, teacher, EncoderDistillConfig(
    device="cuda", torch_dtype="bfloat16",
    epochs=1, loss="mse",                 # "mse" | "cosine"
    lr_scheduler="cosine", warmup_steps=20,
))
result = trainer.train(pipe)
print(result["losses"][-1])
print("projection added:", trainer._projector is not None)

student.save_pretrained("./my-base")
t_tok.save_pretrained("./my-base")
```

---

## Loss

The loss is the per-token distance between the student's (optionally projected) hidden states and the teacher's, masked by `attention_mask` so padding is ignored:

- `loss="mse"` — mean squared error over real-token elements.
- `loss="cosine"` — `1 − cosine_similarity` averaged over real tokens.

---

## Config — `EncoderDistillConfig`

| Field | Default | Description |
|---|---|---|
| `loss` | `"mse"` | `"mse"` or `"cosine"` on per-token hidden states |
| `pad_token_id` | `0` | Used to derive the attention mask if absent |
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

`train()` has the same signature and return value as the other trainers; `save_checkpoint` / `resume_from_checkpoint` persist the student, the projector (if created), and optimizer state.
