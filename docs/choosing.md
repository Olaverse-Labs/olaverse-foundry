# Which trainer do I need?

Foundry ships nine trainers. You need **one or two** of them — which ones depends entirely on what you're building and what you already have. Start from your goal:

---

## I want a model that *generates text* (causal LM)

You need a **teacher** — a bigger causal LM the student learns from.

| Situation | Use |
|---|---|
| Short run, one GPU, prototyping | [`TorchDistillTrainer`](training/torch.md) — teachers run live every step |
| Multiple epochs and/or multiple GPUs | [`CachedDistillTrainer`](training/cached.md) — teachers run once, logits cached to disk, later epochs are free |

Both accept several teachers at once with relative weights (`TeacherRegistry`).

---

## I want an *embedding model* (search / retrieval / similarity)

| What you have | Use |
|---|---|
| A strong embedding teacher (bge, e5, …) and raw text | [`EmbeddingDistillTrainer`](training/embed.md) — copy the teacher's sentence vectors into a smaller model |
| `{anchor, positive}` pairs — or the ability to [synthesize them](synthetic.md) | [`ContrastiveTrainer`](training/contrastive.md) — InfoNCE, the e5/bge training recipe |
| Both | Distil first, then contrastive fine-tune the result |

No pairs and no teacher? [Synthetic data](synthetic.md) manufactures pairs from raw passages — or from *nothing* in the target language, via translation.

Measure the result with [retrieval evaluation](retrieval.md) (nDCG / Recall vs e5, LaBSE, …).

---

## I want an *encoder base* for classification / NER

This is a two-step build: make a **base**, then add a **head** (next section).

| What you have | Use |
|---|---|
| Only raw text — no teacher | [`MLMTrainer`](training/mlm.md) — masked-LM pretraining from scratch |
| A teacher encoder, limited raw text | [`EncoderDistillTrainer`](training/encoder-distill.md) — copy the teacher's per-token states; much more data-efficient than MLM from scratch |
| A teacher **and** in-domain text, student shares the teacher's vocab | [`DistilMLMTrainer`](training/distil-mlm.md) — distillation + MLM in one loss (the DistilBERT recipe) |

---

## I have a base — I want a *task model*

| Task | Use |
|---|---|
| Classification (topic, sentiment, langID, moderation) | [`SequenceClassificationTrainer`](training/heads.md) |
| NER / token tagging | [`TokenClassificationTrainer`](training/heads.md) |

`build_encoder_with_head("./my-base", num_labels, task)` attaches a fresh head in one line. Set `freeze_backbone=True` to train only the head — many heads can then share one frozen encoder. Compare bases head-to-head with the [evaluation harness](evaluation.md).

---

## I want the model *smaller / cheaper to serve*

| Goal | Use |
|---|---|
| int8/int4 weights | [`prepare_qat` + `export_quantized`](quantization.md) — wrap the model before training with **any** trainer above, export packed weights after |
| Swappable task adapters instead of full fine-tunes | [Skill packs](skillpacks.md) — detachable LoRA adapters with PEFT round-trip |
| A *bigger* model from the one you have | [Growth](growth.md) — SOLAR-style depth up-scaling |

---

## Python, YAML recipes, or the CLI?

Three layers drive the same machinery — pick by how repeatable the run needs to be:

| Layer | When |
|---|---|
| **Python API** (everything above) | Exploring, notebooks, custom loops — the full feature set |
| **[YAML recipes](recipes.md)** + `foundry run` | A run you'll repeat or hand to someone else: the whole pipeline in one reviewable file |
| **[CLI](cli.md)** | `foundry doctor` (check your install), `foundry plan` (preview a recipe's stages and cost) |

If in doubt, start with the Python API — recipes cover the common paths, not every option.

---

## Still unsure?

- New to the terminology? → [Concepts & glossary](concepts.md)
- Want to see a full build? → [Small classifier](guides/small-classifier.md) · [Low-resource retriever](guides/low-resource-retriever.md)
