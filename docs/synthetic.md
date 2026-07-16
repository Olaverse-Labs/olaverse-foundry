# Synthetic data

`foundry.synthetic` manufactures the training pairs and **hard negatives** that make [contrastive training](training/contrastive.md) strong — three strategies for three data situations:

| You have | Use | Cost |
|---|---|---|
| Passages, no queries (high-resource language) | `synthesize_pairs` + `generate_hard_negatives` (open LLM) | GPU inference |
| Existing pairs | `mine_hard_negatives` (your encoder) | Cheap, LLM-free |
| No data at all in the target language | `synthesize_parallel` (MT model) | GPU inference |

All outputs are lists of `{"anchor", "positive"[, "negative"]}` dicts, ready for `ContrastiveTrainer`.

!!! note "Keep the data commercially clean"
    Use **open, Apache-licensed** models — Qwen 2.5/3, Mistral, MADLAD-400 — not Claude/GPT, whose terms restrict training on outputs.

```bash
pip install "olaverse-foundry[torch]"
```

---

## LLM generation (high-resource languages)

An open instruct LLM writes a query for each passage, or a plausible-but-wrong passage as a hard negative. Keep this to high-resource languages — LLMs write poor low-resource text.

```python
from foundry import load_generator, synthesize_pairs, generate_hard_negatives

gen = load_generator("Qwen/Qwen2.5-3B-Instruct", device="auto", dtype="bfloat16")

pairs = synthesize_pairs(passages, gen)          # {"anchor": query, "positive": passage}
pairs = generate_hard_negatives(pairs, gen)      # adds "negative" to each pair
```

- Both accept a `(model, tokenizer)` tuple from `load_generator`, **or any callable** `prompts -> list[str]` — plug in your own inference stack.
- The prompts are exported (`HARD_NEG_PROMPT`, `QUERY_PROMPT`) and overridable via the `prompt=` argument.
- `llm_generate(gen, prompts, max_new_tokens=96, batch_size=8, temperature=0.7)` is the underlying batched chat generation, if you need it directly.

---

## Encoder mining (low-resource languages)

Use an encoder to pick, for each anchor, the highest-scoring *other* positive as its hard negative. Cheap, LLM-free, and the right choice for low-resource languages.

```python
from foundry import mine_hard_negatives

pairs = mine_hard_negatives(pairs, model, tokenizer,
                            batch_size=64, device="cuda",
                            skip_top=1)   # skip the very top match — likely a near-duplicate
```

`skip_top` avoids false negatives: the single highest-scoring other positive is often a near-duplicate of the true answer, so the *next* one is taken instead.

---

## Translation synthesis (no-data languages)

For languages with **no parallel data**, translate a high-resource corpus with an open MT model. `google/madlad400-3b-mt` covers 400+ languages and is far better at them than a general LLM.

```python
from foundry import load_translator, synthesize_parallel

tr = load_translator("google/madlad400-3b-mt", device="auto", dtype="bfloat16")

pairs = synthesize_parallel(
    english_sentences, tr,
    target_langs=["yo", "sw", "ha"],      # ISO codes
    batch_size=32, max_new_tokens=128,    # throughput knobs
)
# {"anchor": english, "positive": translation} for every (sentence, language)
```

Cross-lingual anchor/positive pairs like these are exactly what aligns a multilingual embedding space in `ContrastiveTrainer`. Empty translations are dropped. `translate_texts(tr, texts, "yo")` is the underlying primitive; like the LLM helpers, `synthesize_parallel` also accepts any callable `(texts, lang) -> list[str]`.

---

## End-to-end: no-data language → retriever

```python
from foundry import (load_translator, synthesize_parallel, mine_hard_negatives,
                     ContrastiveTrainer, ContrastiveConfig,
                     compare_retrievers, print_retrieval_comparison)

tr    = load_translator()
pairs = synthesize_parallel(english_corpus, tr, ["yo"])
pairs = mine_hard_negatives(pairs, model, tok, device="cuda")

trainer = ContrastiveTrainer(model, tok, ContrastiveConfig(batch_size=64, device="cuda"))
trainer.train(pairs)

results = compare_retrievers({"mine": "./out", "e5": "intfloat/multilingual-e5-base"},
                             queries, corpus, qrels)
print_retrieval_comparison(results)
```
