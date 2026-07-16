# Concepts & glossary

Plain-language explanations of every term the docs assume. Two dictionaries: the standard ML vocabulary, and the handful of words Foundry adds on top.

---

## The core idea in one paragraph

Foundry builds **small, specialised models out of big, general ones**. A big model (the **teacher**) already knows a lot; training a small model (the **student**) to imitate it — called **distillation** — is far cheaper and needs far less data than training the small model from scratch. Foundry covers that, plus the steps around it: preparing data, adding task heads, shrinking weights for deployment, and measuring whether the result is actually good.

---

## ML vocabulary

**Teacher / student** — the big model being imitated and the small model learning from it. In Foundry, teachers are wrapped in a `TeacherRegistry`; a student is any `nn.Module`.

**Distillation** — training the student to match the teacher's *outputs* (its probability distributions, its hidden representations, or its sentence vectors) instead of — or in addition to — the raw labels. The student learns *how the teacher thinks*, which carries far more signal per example than a hard label.

**Logits** — the raw scores a model produces before they're turned into probabilities. Distillation compares student and teacher logits.

**CE + KL, and `alpha`** — the causal-LM distillation loss has two parts: **cross-entropy (CE)** against the real next token ("learn from data") and **KL divergence** against the teacher's distribution ("copy the teacher"). `alpha` sets the balance: `alpha=0.3` means 30% data, 70% teacher.

**Temperature** — softens both distributions before the KL comparison so the student also learns which *wrong* answers the teacher considers nearly right. Higher = softer.

**Logit cache** — teacher outputs don't change between epochs, so `CachedDistillTrainer` computes them once and stores the top-k logits on disk. Every epoch after the first reads from the cache and never runs the teacher.

**MLM (masked language modeling)** — hide ~15% of the tokens and train the model to fill them in. This is how encoders like BERT are pretrained from raw, unlabeled text.

**Encoder vs. causal LM (decoder)** — an *encoder* reads a whole sentence and produces representations (for classification, NER, embeddings). A *causal LM* generates text left to right. Foundry builds both; the trainers differ.

**Embedding / bi-encoder** — a model that turns a text into a single vector such that similar texts land close together. Search works by embedding the query and every document, then ranking by similarity.

**Pooling (`mean` / `cls`)** — how a per-token encoder output becomes one sentence vector: average all token vectors (`mean`) or take the special first token (`cls`).

**InfoNCE / in-batch negatives** — the contrastive loss behind e5 and bge. In a batch of (query, passage) pairs, each query must rank *its own* passage above every other passage in the batch. Those other passages are the "in-batch negatives" — free negative examples. Bigger batch = more negatives = stronger training.

**Hard negative** — a passage that *looks* relevant but isn't. Far more informative than a random negative, and the main lever for retrieval quality. Foundry can [generate or mine them](synthetic.md).

**nDCG@k / Recall@k** — retrieval metrics over the top k results: Recall asks "did the relevant docs show up?", nDCG also rewards ranking them *higher*.

**QAT (quantization-aware training)** — train the model while simulating int8/int4 precision, so the weights adapt to it. The exported model is ~4× smaller with far less quality loss than quantizing after the fact.

**Mixed precision (`bfloat16` / `float16`)** — run most of the math in 16-bit floats: ~2× faster, half the memory, no meaningful quality change. Set `torch_dtype="bfloat16"` in any config.

**LoRA** — instead of fine-tuning all weights, train tiny low-rank matrices alongside them (<1% of the parameters). Cheap to train, and swappable — the basis of skill packs.

**Checkpoint** — a snapshot of model + optimizer state so training can resume. Foundry loads them with `weights_only=True`, so a checkpoint file can never execute code.

---

## Foundry vocabulary

**Seed → grow → fuse → freeze → extend** — the "model family" lifecycle the library is named for: start from one pretrained model (*seed*), scale it up (*grow*), merge abilities (*fuse*), stop touching the base (*freeze*), and add capabilities as adapters (*extend*).

**Skill pack** — Foundry's name for a detachable LoRA adapter: a small file that snaps onto a frozen base to add one capability, and snaps off again. Round-trips to the standard PEFT format. [Reference →](skillpacks.md)

**Growth** — making a model *deeper* by duplicating layers (the SOLAR up-scaling recipe), as a starting point for further training. [Reference →](growth.md)

**Recipe** — a YAML file describing a whole pipeline (models, data, stages), validated with Pydantic and run with `foundry run`. [Reference →](recipes.md)

**`DataPipeline`** — the one adapter between *any* data source (HF dataset, streaming dataset, list of strings or dicts) and every trainer: tokenises, batches, shuffles, attaches labels. [Reference →](data.md)

**`TeacherRegistry`** — a weighted pool of teachers. With more than one, a [fusion strategy](training/config.md#fusion-strategies) decides how their predictions combine per token.

**`foundry doctor`** — CLI command that reports which optional backends (torch, CUDA, accelerate, peft, …) are installed and working.

---

Next: [Which trainer do I need? →](choosing.md)
