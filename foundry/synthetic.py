"""
Synthetic data for retrieval — hard negatives + query synthesis.

Two ways to get the hard negatives that make contrastive training strong:

* **LLM generation** (``generate_hard_negatives`` / ``synthesize_pairs``) — use an
  **open, Apache-licensed** instruct model (Qwen / Mistral) to write a
  plausible-but-wrong passage, or a query for a passage. Keep it to *high-resource*
  languages: LLMs write poor low-resource text. Use an open model (not Claude/GPT)
  so the generated data stays commercially clean.
* **Mining** (``mine_hard_negatives``) — use an encoder to pick, for each anchor, the
  highest-scoring *other* positive as a hard negative. Cheap, no LLM, and the right
  choice for low-resource languages.

Both return pairs with a ``"negative"`` key, ready for ``ContrastiveTrainer``.
"""
from __future__ import annotations

from typing import Any, Callable


HARD_NEG_PROMPT = (
    "Write ONE short passage (1-3 sentences) on a topic SIMILAR to the text below, "
    "but about a DIFFERENT specific thing — so it looks relevant yet does not answer "
    "or match it. Reply with ONLY the passage, in the same language as the text.\n\n"
    "Text: {text}\n\nPassage:"
)

QUERY_PROMPT = (
    "Write ONE short search query, in the same language, that the passage below would "
    "answer. Reply with ONLY the query.\n\nPassage: {text}\n\nQuery:"
)


# ── Open LLM generator ────────────────────────────────────────────────────────

def load_generator(model_id: str = "Qwen/Qwen2.5-3B-Instruct",
                   device: str = "auto", dtype: str = "bfloat16"):
    """
    Load an open, Apache-licensed instruct LLM for generation. Returns
    ``(model, tokenizer)``. Use Qwen2.5/Qwen3 or Mistral — **not** Claude/GPT, whose
    terms restrict training on outputs.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("transformers + torch required. pip install olaverse-foundry[torch]")
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=td, device_map=device)
    return model, tok


def llm_generate(generator, prompts, max_new_tokens: int = 96, batch_size: int = 8,
                 temperature: float = 0.7, do_sample: bool = True) -> list[str]:
    """Batched chat generation from a list of prompts → list of strings."""
    import torch
    model, tok = generator
    tok.padding_side = "left"            # correct for batched decoder generation
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                 temperature=temperature, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            out.append(tok.decode(new, skip_special_tokens=True).strip())
    return out


def _as_callable(generator) -> Callable[[list], list]:
    """Accept a (model, tokenizer) tuple or a plain ``prompts -> list[str]`` callable."""
    if isinstance(generator, tuple):
        return lambda prompts: llm_generate(generator, prompts)
    if callable(generator):
        return generator
    raise TypeError("generator must be a (model, tokenizer) tuple or a callable.")


# ── LLM-based ─────────────────────────────────────────────────────────────────

def generate_hard_negatives(pairs, generator, positive_key: str = "positive",
                            negative_key: str = "negative", prompt: str = HARD_NEG_PROMPT) -> list[dict]:
    """Add an LLM-generated hard negative to each pair. ``generator`` is a
    ``(model, tokenizer)`` tuple from :func:`load_generator`, or a callable."""
    gen  = _as_callable(generator)
    negs = gen([prompt.format(text=p[positive_key]) for p in pairs])
    out  = []
    for p, neg in zip(pairs, negs):
        q = dict(p); q[negative_key] = neg
        out.append(q)
    return out


def synthesize_pairs(passages, generator, anchor_key: str = "anchor",
                     positive_key: str = "positive", prompt: str = QUERY_PROMPT) -> list[dict]:
    """Generate a query for each passage → list of ``{anchor: query, positive: passage}``."""
    gen     = _as_callable(generator)
    queries = gen([prompt.format(text=t) for t in passages])
    return [{anchor_key: q, positive_key: t} for q, t in zip(queries, passages)]


# ── Translation synthesis (for no-data / low-resource languages) ──────────────

def load_translator(model_id: str = "google/madlad400-3b-mt",
                    device: str = "auto", dtype: str = "bfloat16"):
    """
    Load an open, **Apache-licensed** translation model that covers low-resource
    languages. ``google/madlad400-3b-mt`` handles 400+ languages — far better than a
    general LLM at the languages that *have no parallel data*, and commercially clean.
    Returns ``(model, tokenizer)``.
    """
    try:
        import torch
        from transformers import T5ForConditionalGeneration, AutoTokenizer
    except ImportError:
        raise ImportError("transformers + torch required. pip install olaverse-foundry[torch]")
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=td, device_map=device)
    return model, tok


def translate_texts(translator, texts, target_lang: str,
                    max_new_tokens: int = 200, batch_size: int = 16) -> list[str]:
    """Translate texts into ``target_lang`` (ISO code, e.g. 'sw', 'yo'). MADLAD uses a
    ``<2xx>`` target prefix."""
    import torch
    model, tok = translator
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = [f"<2{target_lang}> {t}" for t in texts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def _as_translate(translator) -> Callable[[list, str], list]:
    if isinstance(translator, tuple):
        return lambda texts, lang: translate_texts(translator, texts, lang)
    if callable(translator):
        return translator
    raise TypeError("translator must be a (model, tokenizer) tuple or a callable (texts, lang)->list.")


def synthesize_parallel(source_texts, translator, target_langs,
                        anchor_key: str = "anchor", positive_key: str = "positive",
                        max_new_tokens: int = 128, batch_size: int = 32) -> list[dict]:
    """
    Create synthetic parallel pairs for languages with no data: translate
    ``source_texts`` (e.g. English) into each ``target_langs`` →
    ``{anchor: source, positive: translation}``. Use an MT model (``load_translator``)
    — not a general LLM — for low-resource languages.

    ``batch_size`` / ``max_new_tokens`` control throughput (bigger batch + fewer
    tokens = faster); they apply when ``translator`` is a ``(model, tokenizer)`` tuple.
    """
    if isinstance(translator, tuple):
        tr = lambda texts, lang: translate_texts(translator, texts, lang,
                                                 max_new_tokens=max_new_tokens, batch_size=batch_size)
    else:
        tr = _as_translate(translator)
    pairs = []
    for lang in target_langs:
        translations = tr(list(source_texts), lang)
        pairs += [{anchor_key: s, positive_key: t}
                  for s, t in zip(source_texts, translations) if t and t.strip()]
    return pairs


# ── Encoder-based mining ──────────────────────────────────────────────────────

def mine_hard_negatives(pairs, model, tokenizer, anchor_key: str = "anchor",
                        positive_key: str = "positive", negative_key: str = "negative",
                        pool: str = "mean", max_length: int = 128, batch_size: int = 64,
                        device=None, skip_top: int = 1) -> list[dict]:
    """
    Mine hard negatives with an encoder: for each anchor, the highest-scoring *other*
    positive (skipping the very top ``skip_top`` to avoid near-duplicate false
    negatives). Cheap, LLM-free, and the right choice for low-resource languages.
    """
    import numpy as np
    from foundry.retrieval import encode_texts
    anchors   = [p[anchor_key] for p in pairs]
    positives = [p[positive_key] for p in pairs]
    a = encode_texts(model, tokenizer, anchors,   pool, True, max_length, batch_size, device)
    c = encode_texts(model, tokenizer, positives, pool, True, max_length, batch_size, device)
    sims = a @ c.T
    out = []
    for i, p in enumerate(pairs):
        order = np.argsort(-sims[i])
        cand  = [int(j) for j in order if int(j) != i]
        q = dict(p)
        if len(cand) > skip_top:
            q[negative_key] = positives[cand[skip_top]]
        out.append(q)
    return out
