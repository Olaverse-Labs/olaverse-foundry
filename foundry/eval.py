"""
Evaluation harness — measure how good a base encoder is, head-to-head with others.

Fine-tunes the same task head on each model with identical data + config, then
reports accuracy and macro-F1 — apples-to-apples, so "better" is a table not a
vibe. Each model is tokenised with **its own** tokenizer (the fair thing to do),
so you pass *raw rows*, not pre-tokenised ids. Pure-numpy metrics (no sklearn).

Example (sequence classification)::

    from foundry import compare_encoders, print_comparison

    train_rows = [{"text": r["text"], "label": r["label"]} for r in news["train"]]
    eval_rows  = [{"text": r["text"], "label": r["label"]} for r in news["validation"]]

    results = compare_encoders(
        {"Purple Mist Base": "./purple-mist-base",
         "mBERT":            "google-bert/bert-base-multilingual-cased",
         "e5-base":          "intfloat/multilingual-e5-base"},
        train_rows, eval_rows, num_labels=NUM, task="sequence",
    )
    print_comparison(results)

For NER, pass rows of ``{"tokens": [...], "ner_tags": [...]}`` and ``task="token"``.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ── Metrics ──────────────────────────────────────────────────────────────────

def macro_f1(preds: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    """Unweighted mean F1 over classes that appear in ``labels``."""
    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    f1s = []
    for c in range(num_labels):
        in_label = labels == c
        if in_label.sum() == 0:
            continue
        in_pred = preds == c
        tp = int((in_pred & in_label).sum())
        fp = int((in_pred & ~in_label).sum())
        fn = int((~in_pred & in_label).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# ── Row → tokenised batches (per model tokenizer) ────────────────────────────

def _seq_pipe(rows, tok, text_key, label_key, max_length, batch_size):
    from foundry.data import DataPipeline
    enc_rows = []
    for r in rows:
        ids = tok(r[text_key], truncation=True, max_length=max_length)["input_ids"]
        enc_rows.append({"input_ids": ids, "label": int(r[label_key])})
    return DataPipeline(enc_rows, batch_size=batch_size, max_length=max_length,
                        mode="embed", label_column="label")


def _token_pipe(rows, tok, tokens_key, tags_key, max_length, batch_size):
    from foundry.data import DataPipeline
    enc_rows = []
    for r in rows:
        enc = tok(r[tokens_key], is_split_into_words=True,
                  truncation=True, max_length=max_length)
        word_ids, labels, prev = enc.word_ids(), [], None
        for wid in word_ids:
            labels.append(-100 if (wid is None or wid == prev) else int(r[tags_key][wid]))
            prev = wid
        enc_rows.append({"input_ids": enc["input_ids"], "ner": labels})
    return DataPipeline(enc_rows, batch_size=batch_size, max_length=max_length,
                        mode="embed", label_column="ner")


def _tokenizer_for(base, tokenizer):
    if tokenizer is not None:
        return tokenizer
    if not isinstance(base, str):
        raise ValueError("Pass tokenizer= when base is a model object, not a path/id.")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base)


# ── Evaluate one / compare many ──────────────────────────────────────────────

def evaluate_encoder(
    base, train_rows, eval_rows, num_labels: int, task: str = "sequence",
    *, tokenizer=None, text_key="text", label_key="label",
    tokens_key="tokens", tags_key="ner_tags",
    max_length: int = 128, batch_size: int = 16, config=None,
) -> dict:
    """
    Fine-tune a head on ``base`` (tokenising rows with the model's own tokenizer)
    and score it on ``eval_rows``.

    Returns ``{"accuracy", "macro_f1", "params_m"}``.
    """
    from foundry.training.heads import (
        SequenceClassificationTrainer, TokenClassificationTrainer,
        HeadTrainConfig, build_encoder_with_head,
    )

    tok   = _tokenizer_for(base, tokenizer)
    model = base if hasattr(base, "forward") else build_encoder_with_head(base, num_labels, task)

    if task == "sequence":
        train = _seq_pipe(train_rows, tok, text_key, label_key, max_length, batch_size)
        ev    = _seq_pipe(eval_rows,  tok, text_key, label_key, max_length, batch_size)
        Trainer = SequenceClassificationTrainer
    else:
        train = _token_pipe(train_rows, tok, tokens_key, tags_key, max_length, batch_size)
        ev    = _token_pipe(eval_rows,  tok, tokens_key, tags_key, max_length, batch_size)
        Trainer = TokenClassificationTrainer

    cfg = config or HeadTrainConfig(device="auto", torch_dtype="bfloat16", epochs=2,
                                    lr_scheduler="cosine", warmup_steps=20, log_every=50)
    cfg.num_labels = num_labels
    trainer = Trainer(model, config=cfg)
    trainer.train(train, eval_dataset=ev)

    preds, labels = trainer.predict(ev)
    acc = float((preds == labels).mean()) if preds.size else 0.0
    f1  = macro_f1(preds, labels, num_labels)
    n_params = sum(p.numel() for p in model.parameters())
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
            "params_m": round(n_params / 1e6, 1)}


def compare_encoders(
    models, train_rows, eval_rows, num_labels: int, task: str = "sequence", **kw,
) -> dict:
    """
    Evaluate several encoders on the same task/data and return
    ``{name: {accuracy, macro_f1, params_m}}``. ``models`` is a ``{name: base}``
    dict or a list of base ids/paths. Extra kwargs pass through to
    :func:`evaluate_encoder`.
    """
    items = models.items() if isinstance(models, dict) else [(m, m) for m in models]
    results: dict[str, Any] = {}
    for name, base in items:
        print(f"[eval] fine-tuning + scoring: {name} …")
        try:
            results[name] = evaluate_encoder(base, train_rows, eval_rows,
                                             num_labels, task, **kw)
        except Exception as exc:               # one model failing shouldn't sink the table
            print(f"[eval]   {name} failed: {exc}")
            results[name] = {"accuracy": float("nan"), "macro_f1": float("nan"),
                             "params_m": float("nan"), "error": str(exc)}
    return results


def print_comparison(results: dict, sort_by: str = "macro_f1") -> None:
    """Pretty-print the comparison table, best first."""
    def key(kv):
        v = kv[1].get(sort_by)
        return v if isinstance(v, (int, float)) and v == v else -1.0   # NaN/last
    rows = sorted(results.items(), key=key, reverse=True)
    name_w = max((len(n) for n in results), default=5) + 2
    print()
    print(f"  {'model':{name_w}}  {'accuracy':>9}  {'macro_f1':>9}  {'params(M)':>10}")
    print("  " + "─" * (name_w + 34))
    for name, m in rows:
        print(f"  {name:{name_w}}  {str(m.get('accuracy')):>9}  "
              f"{str(m.get('macro_f1')):>9}  {str(m.get('params_m')):>10}")
    print()
