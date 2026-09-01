"""
Quality gates for synthesised training data.

The flagship pipeline — translate → mine hard negatives → contrastively train —
manufactures every example it trains on. Machine translation repeats itself, LLM
generation drifts out of the target language, and mining can pair a passage with
a textual duplicate of itself. None of that raises an error. It shows up much
later, as a model that trained cleanly and retrieves badly.

These are cheap structural checks meant to run *between* synthesis and training.
They need no network, no GPU and no model — only numpy, which is already a core
dependency.

What this module deliberately does **not** do: judge translation adequacy,
identify the language of a string, or score toxicity. Those need models, and a
half-hearted heuristic version would be worse than none — it would let bad data
through under a green check. Use a real langid/toxicity model for those, and use
this for the failure modes that are free to catch.

Typical use::

    from foundry.quality import clean_pairs, quality_report, print_quality_report

    pairs  = synthesize_parallel(...)
    report = quality_report(pairs)
    print_quality_report(report)

    pairs = clean_pairs(pairs)      # then train on these
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np


__all__ = [
    "normalise_text",
    "duplicate_rate",
    "dedup_pairs",
    "drop_degenerate_pairs",
    "clean_pairs",
    "embedding_health",
    "quality_report",
    "print_quality_report",
]


# ── Text normalisation ─────────────────────────────────────────────────────

def normalise_text(text: Any) -> str:
    """
    Casefold and collapse whitespace, for duplicate detection only.

    Deliberately conservative: no punctuation stripping, no unicode folding, no
    stemming. Two strings that differ only in spacing or case are the same
    training example; two that differ in punctuation may not be, and in a
    low-resource language you cannot assume otherwise.
    """
    return " ".join(str(text).split()).casefold()


def duplicate_rate(texts: Iterable[Any]) -> float:
    """Fraction of ``texts`` that repeat an earlier entry, after normalisation."""
    items = [normalise_text(t) for t in texts]
    if not items:
        return 0.0
    return 1.0 - (len(set(items)) / len(items))


# ── Pair-level filters ─────────────────────────────────────────────────────

def dedup_pairs(pairs: Sequence[dict], anchor_key: str = "anchor",
                positive_key: str = "positive") -> list[dict]:
    """
    Drop pairs whose (anchor, positive) has already been seen, keeping the first.

    Duplicate pairs are not harmless padding. In InfoNCE the other members of a
    batch are the negatives, so a duplicate pair can land in the same batch as
    its twin and the loss is then asked to separate two identical examples.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for pair in pairs:
        key = (normalise_text(pair.get(anchor_key, "")),
               normalise_text(pair.get(positive_key, "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


def drop_degenerate_pairs(pairs: Sequence[dict], anchor_key: str = "anchor",
                          positive_key: str = "positive",
                          negative_key: str = "negative",
                          min_chars: int = 1) -> list[dict]:
    """
    Drop pairs that cannot teach anything, and strip false negatives.

    A pair is dropped when either side is empty/whitespace, either side is
    shorter than ``min_chars``, or the anchor and positive are the same string —
    a "positive" identical to its anchor carries no signal beyond the identity
    map, and translation pipelines emit these whenever a segment passes through
    untranslated.

    A negative equal to the pair's own anchor or positive is a *false* negative:
    it asks the loss to push a text away from itself. Those are removed from the
    pair rather than dropping the whole example, since the anchor/positive part
    is still usable.
    """
    out: list[dict] = []
    for pair in pairs:
        anchor   = normalise_text(pair.get(anchor_key, ""))
        positive = normalise_text(pair.get(positive_key, ""))

        if not anchor or not positive:
            continue
        if len(anchor) < min_chars or len(positive) < min_chars:
            continue
        if anchor == positive:
            continue

        kept = dict(pair)
        if negative_key in kept:
            negative = normalise_text(kept[negative_key])
            if not negative or negative in (anchor, positive):
                del kept[negative_key]
        out.append(kept)
    return out


def clean_pairs(pairs: Sequence[dict], anchor_key: str = "anchor",
                positive_key: str = "positive", negative_key: str = "negative",
                min_chars: int = 1) -> list[dict]:
    """Deduplicate, then drop degenerate pairs. The usual pre-training gate."""
    deduped = dedup_pairs(pairs, anchor_key, positive_key)
    return drop_degenerate_pairs(deduped, anchor_key, positive_key,
                                 negative_key, min_chars)


# ── Encoder health ─────────────────────────────────────────────────────────

def embedding_health(embeddings: np.ndarray,
                     collapse_threshold: float = 0.99) -> dict:
    """
    Diagnose a degenerate encoder from its embeddings.

    A collapsed encoder — one mapping every input to nearly the same vector — is
    the quietest failure in this whole pipeline. It does not raise, and it does
    not score zero: ranking a collapsed set is decided by tie-breaking, which on
    a small benchmark returns a plausible mid-range nDCG. An untrained tiny
    encoder scoring nDCG@5 ≈ 0.25 looks like a weak model, not a broken one.

    Returns ``mean_similarity`` / ``min_similarity`` / ``max_similarity`` over
    the off-diagonal cosine pairs, an ``effective_dimensions`` estimate (the
    participation ratio of the singular values — how many directions the
    embeddings actually use), and a ``collapsed`` flag.

    Args:
        embeddings:         (N, D) array. Assumed L2-normalised, as
                            ``encode_texts(..., normalize=True)`` returns.
        collapse_threshold: mean off-diagonal cosine above which the encoder is
                            reported as collapsed.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"expected a 2-D (N, D) array, got shape {emb.shape}")

    n = emb.shape[0]
    if n < 2:
        raise ValueError("need at least 2 embeddings to measure similarity")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.maximum(norms, 1e-12)

    sims = unit @ unit.T
    off_diagonal = sims[~np.eye(n, dtype=bool)]

    # Participation ratio of the singular values: 1 when all the variance sits
    # in a single direction, D when it is spread evenly.
    singular = np.linalg.svd(unit - unit.mean(axis=0, keepdims=True),
                             compute_uv=False)
    energy = float((singular ** 2).sum())
    effective = float((energy ** 2) / ((singular ** 4).sum())) if energy > 1e-12 else 0.0

    mean_sim = float(off_diagonal.mean())
    return {
        "n":                    int(n),
        "mean_similarity":      mean_sim,
        "min_similarity":       float(off_diagonal.min()),
        "max_similarity":       float(off_diagonal.max()),
        "effective_dimensions": effective,
        "collapsed":            bool(mean_sim >= collapse_threshold),
    }


# ── Report ─────────────────────────────────────────────────────────────────

def quality_report(pairs: Sequence[dict], anchor_key: str = "anchor",
                   positive_key: str = "positive", negative_key: str = "negative",
                   embeddings: Optional[np.ndarray] = None) -> dict:
    """
    Summarise what a synthesised pair set looks like before training on it.

    Pass ``embeddings`` (from ``encode_texts``) to include an encoder-health
    section; omit it for a pure text report.
    """
    pairs = list(pairs)
    total = len(pairs)
    anchors   = [p.get(anchor_key, "") for p in pairs]
    positives = [p.get(positive_key, "") for p in pairs]

    with_negative = sum(1 for p in pairs if p.get(negative_key))
    false_negatives = sum(
        1 for p in pairs
        if p.get(negative_key)
        and normalise_text(p[negative_key]) in (normalise_text(p.get(anchor_key, "")),
                                                normalise_text(p.get(positive_key, "")))
    )
    identity_pairs = sum(
        1 for p in pairs
        if normalise_text(p.get(anchor_key, "")) == normalise_text(p.get(positive_key, ""))
    )
    empty_sides = sum(
        1 for p in pairs
        if not normalise_text(p.get(anchor_key, "")) or not normalise_text(p.get(positive_key, ""))
    )

    kept = clean_pairs(pairs, anchor_key, positive_key, negative_key)
    report: dict[str, Any] = {
        "total_pairs":          total,
        "unique_pairs":         len(dedup_pairs(pairs, anchor_key, positive_key)),
        "usable_pairs":         len(kept),
        "anchor_duplicate_rate":   duplicate_rate(anchors),
        "positive_duplicate_rate": duplicate_rate(positives),
        "identity_pairs":       identity_pairs,
        "empty_sides":          empty_sides,
        "with_negative":        with_negative,
        "false_negatives":      false_negatives,
    }
    if embeddings is not None:
        report["encoder"] = embedding_health(embeddings)
    return report


def print_quality_report(report: dict) -> None:
    """Pretty-print a ``quality_report``, loudest problems first."""
    total = report.get("total_pairs", 0)

    def pct(count: int) -> str:
        return f"{(100.0 * count / total):.1f}%" if total else "—"

    print()
    print("  synthetic data quality")
    print("  " + "─" * 46)
    print(f"  {'pairs in':<28}{total:>10}")
    print(f"  {'unique':<28}{report.get('unique_pairs', 0):>10}"
          f"  {pct(report.get('unique_pairs', 0)):>7}")
    print(f"  {'usable after cleaning':<28}{report.get('usable_pairs', 0):>10}"
          f"  {pct(report.get('usable_pairs', 0)):>7}")
    print()
    for label, key in (("anchor duplicates",   "anchor_duplicate_rate"),
                       ("positive duplicates", "positive_duplicate_rate")):
        print(f"  {label:<28}{report.get(key, 0.0) * 100:>9.1f}%")
    for label, key in (("anchor == positive", "identity_pairs"),
                       ("empty side",         "empty_sides"),
                       ("false negatives",    "false_negatives")):
        count = report.get(key, 0)
        flag = "  ←" if count else ""
        print(f"  {label:<28}{count:>10}{flag}")

    encoder = report.get("encoder")
    if encoder:
        print()
        print(f"  {'encoder mean similarity':<28}{encoder['mean_similarity']:>10.4f}")
        print(f"  {'effective dimensions':<28}{encoder['effective_dimensions']:>10.1f}")
        if encoder["collapsed"]:
            print("  COLLAPSED — this encoder maps every input to nearly one vector.")
            print("  Retrieval scores from it are tie-breaking noise, not quality.")
    print()
