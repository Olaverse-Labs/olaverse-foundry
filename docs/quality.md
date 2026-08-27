# Data Quality Gates

`foundry.quality` is the check you run **between** synthesis and training.

The flagship pipeline manufactures every example it trains on. That is the point
of it — it exists for languages where real data does not exist. But it means the
usual safety net is gone: there is no human-curated corpus to fall back on, and
the failure modes are quiet.

None of these raise an error:

| What happens | What it looks like |
| --- | --- |
| A segment passes through translation untranslated | `anchor == positive` — a pair that teaches the identity map |
| MT repeats itself across a batch | Duplicate pairs, which become each other's in-batch negatives |
| Mining picks a duplicate passage | A "negative" that is textually the positive |
| The encoder collapses | Every text maps to one vector |

All four produce a training run that looks fine. You find out at benchmark time,
and by then you cannot tell which stage was at fault.

## Cleaning a pair set

```python
from foundry.quality import clean_pairs, quality_report, print_quality_report

pairs = synthesize_parallel(source_texts, translator, ["yo", "ha", "ig"])
pairs = mine_hard_negatives(pairs, model, tokenizer)

print_quality_report(quality_report(pairs))
pairs = clean_pairs(pairs)          # train on these
```

`clean_pairs` deduplicates, then drops pairs that cannot teach anything:

- either side empty or shorter than `min_chars`
- `anchor == positive` after normalisation
- a `negative` equal to the pair's own anchor or positive — stripped from the
  pair rather than dropping it, since the anchor/positive half is still usable

Normalisation is deliberately conservative: casefold and whitespace only. No
punctuation stripping, no unicode folding. Two strings differing only in spacing
are the same example; two differing in punctuation may not be, and in a
low-resource language you cannot assume otherwise.

## Checking the encoder

A collapsed encoder is the quietest failure of all, because **it does not score
zero**. Ranking a set of identical vectors is decided by tie-breaking, which on a
small benchmark returns a plausible mid-range nDCG:

```python
from foundry.quality import embedding_health
from foundry.retrieval import encode_texts

health = embedding_health(encode_texts(model, tokenizer, texts))
if health["collapsed"]:
    raise SystemExit("encoder collapsed — retrieval scores are tie-breaking noise")
```

| Key | Meaning |
| --- | --- |
| `mean_similarity` | Mean off-diagonal cosine. Near 1.0 means collapse. |
| `min_similarity` / `max_similarity` | The spread of the pairwise similarities. |
| `effective_dimensions` | Participation ratio of the singular values — how many directions the embeddings actually use. Near 1 means everything sits on one axis. |
| `collapsed` | `mean_similarity >= collapse_threshold` (default `0.99`). |

Run this on a fresh model before you train and on the trained one after. An
untrained encoder scoring nDCG@5 ≈ 0.25 looks like a weak model; `collapsed:
True` tells you it is a broken one.

## What this does not cover

Deliberately absent: language identification, translation adequacy, and toxicity
scoring. Those need real models, and a heuristic version would be worse than
nothing — it would wave bad data through under a green check. Use a proper
langid or toxicity model for those, and use this module for the failure modes
that are free to catch.

Everything here is numpy-only, so it runs on a core `pip install
olaverse-foundry` with no GPU and no model download.

## API

| Function | Purpose |
| --- | --- |
| `normalise_text(text)` | Casefold + collapse whitespace. Duplicate detection only. |
| `duplicate_rate(texts)` | Fraction of entries that repeat an earlier one. |
| `dedup_pairs(pairs, ...)` | Drop repeated `(anchor, positive)` pairs, keeping the first. |
| `drop_degenerate_pairs(pairs, ...)` | Drop empty/identity pairs; strip false negatives. |
| `clean_pairs(pairs, ...)` | `dedup_pairs` then `drop_degenerate_pairs`. The usual gate. |
| `embedding_health(embeddings, ...)` | Diagnose a collapsed encoder. |
| `quality_report(pairs, ..., embeddings=None)` | Summary dict; pass embeddings for an encoder section. |
| `print_quality_report(report)` | Pretty-print a report, loudest problems first. |

All are importable straight from `foundry`:

```python
from foundry import clean_pairs, embedding_health, quality_report
```
