# MLMTrainer — masked-language-modeling pretraining

`MLMTrainer` pretrains an **encoder backbone from scratch** — your own architecture, your own tokenizer, **no teacher required**. It is the teacherless path to a base encoder: the model learns general representations directly from raw text by predicting masked tokens.

Because every token position is supervised (not just a pooled vector), the resulting base keeps the token-level representations that token-classification heads such as NER need.

```bash
pip install "olaverse-foundry[torch]"
```

---

## Student contract

`student(input_ids=..., attention_mask=...)` must return an object with a `.logits` tensor of shape `(B, S, vocab_size)` — i.e. an encoder **with an MLM head**. Any HuggingFace `AutoModelForMaskedLM` satisfies this.

For a custom encoder that only returns `.last_hidden_state`, wrap it with `WithMLMHead`:

```python
from foundry import WithMLMHead

student = WithMLMHead(MyEncoder(...), hidden_size=384, vocab_size=32000)
```

---

## Quick start

```python
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast
from foundry import DataPipeline, MLMTrainer, MLMConfig

# Your architecture, random init (= from scratch)
student = BertForMaskedLM(BertConfig(
    vocab_size=len(tok), hidden_size=384, num_hidden_layers=6,
    num_attention_heads=6, intermediate_size=1152, max_position_embeddings=512,
))

# embed-mode pipeline → {"input_ids", "attention_mask"} batches
pipe = DataPipeline(text_rows, tokenizer=tok, text_column="text",
                    batch_size=16, max_length=128, mode="embed")

trainer = MLMTrainer(student, tokenizer=tok, config=MLMConfig(
    device="cuda", torch_dtype="bfloat16",
    epochs=1, mask_prob=0.15,
    lr_scheduler="cosine", warmup_steps=100,
    eval_every=200, save_every=500, save_dir="/ckpts/base",
))
result = trainer.train(pipe, eval_dataset=eval_pipe)
print(result["losses"][-1], result["eval_losses"])

# the base encoder is the body inside the MLM model
student.bert.save_pretrained("./my-base")
tok.save_pretrained("./my-base")
```

---

## Masking

`MLMTrainer` applies BERT-style dynamic masking each step: `mask_prob` of non-special, non-padding tokens are selected; of those, 80 % become `[MASK]`, 10 % a random token, 10 % are left unchanged. The loss is computed only on selected positions. Special tokens and padding are never masked, and at least one position is always masked so the loss is well-defined.

It needs the mask token id — pass a tokenizer with a `mask_token`, or set `MLMConfig.mask_token_id` (and `vocab_size`) explicitly:

```python
MLMConfig(mask_token_id=4, vocab_size=8000, pad_token_id=0)
```

---

## Config — `MLMConfig`

| Field | Default | Description |
|---|---|---|
| `mask_prob` | `0.15` | Fraction of eligible tokens to mask |
| `mask_token_id` | `None` | Required if no tokenizer is passed |
| `pad_token_id` | `0` | Padding id (never masked) |
| `vocab_size` | `None` | Inferred from the tokenizer if omitted |
| `learning_rate` | `1e-4` | AdamW LR |
| `epochs` | `1` | Passes over the dataset |
| `weight_decay` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `1.0` | Gradient clipping |
| `device` | `"auto"` | `"auto"` / `"cuda"` / `"mps"` / `"cpu"` |
| `grad_accumulation_steps` | `1` | Accumulate over N batches |
| `torch_dtype` | `"float32"` | `"bfloat16"` / `"float16"` / `"float32"` |
| `lr_scheduler` | `"cosine"` | `"constant"` / `"cosine"` / `"linear"` |
| `warmup_steps` | `0` | Linear warmup steps |
| `eval_every` | `0` | Eval every N optimizer steps (0 = off) |
| `save_every` / `save_dir` | `0` / `""` | Auto-checkpoint |
| `log_every` | `50` | `on_step` callback cadence |
| `log_backend` | `"none"` | `"wandb"` / `"tensorboard"` / `"none"` |
| `seed` | `42` | Seeds torch + numpy + random |

---

## `train()`

```python
trainer.train(
    dataset,                 # embed-mode DataPipeline, or list of dicts/arrays
    eval_dataset = None,
    on_step      = None,     # callback(global_step, loss)
    shuffle      = False,
    total_steps  = None,     # override for streaming (LR scheduler)
)
# → {"losses": [...], "eval_losses": {step: loss}, "device": "cuda"}
```

Also available: `save_checkpoint(path)` and `resume_from_checkpoint(path)` (model + optimizer state).
