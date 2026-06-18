"""
Pure-numpy tests (no torch) for:
  • model-agnostic growth — detect_layer_prefix + build_upscaled_state_dict
    across BERT / GPT-2 / Llama key layouts.
  • DataPipeline label_column — sequence labels and token-label padding.
"""
import numpy as np
import pytest

from foundry.growth import detect_layer_prefix, build_upscaled_state_dict, upscale_layer_map
from foundry.data import DataPipeline


# ── Growth: any architecture ──────────────────────────────────────────────────

def _bert(n_layers=2):
    sd = {"embeddings.word_embeddings.weight": np.ones((10, 4), np.float32),
          "pooler.dense.weight": np.ones((4, 4), np.float32)}
    for i in range(n_layers):
        sd[f"encoder.layer.{i}.attention.self.query.weight"] = np.full((4, 4), i, np.float32)
    return sd


def _gpt2(n_layers=2):
    sd = {"wte.weight": np.ones((10, 4), np.float32)}
    for i in range(n_layers):
        sd[f"transformer.h.{i}.attn.c_attn.weight"] = np.full((4, 4), i, np.float32)
    return sd


def _llama(n_layers=2):
    sd = {"model.embed_tokens.weight": np.ones((10, 4), np.float32),
          "lm_head.weight": np.ones((10, 4), np.float32)}
    for i in range(n_layers):
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = np.full((4, 4), i, np.float32)
    return sd


def test_detect_prefix_bert():
    assert detect_layer_prefix(_bert()) == "encoder.layer"


def test_detect_prefix_gpt2():
    assert detect_layer_prefix(_gpt2()) == "transformer.h"


def test_detect_prefix_llama():
    assert detect_layer_prefix(_llama()) == "model.layers"


def test_detect_prefix_raises_without_layers():
    with pytest.raises(ValueError):
        detect_layer_prefix({"embeddings.weight": np.ones((4, 4))})


def test_build_upscaled_bert_auto_prefix():
    sd = _bert(2)
    grown = build_upscaled_state_dict(sd, upscale_layer_map(2, 4))   # prefix auto-detected
    layer_keys = [k for k in grown if k.startswith("encoder.layer.")]
    assert len(layer_keys) == 4
    assert "pooler.dense.weight" in grown          # non-layer keys preserved


def test_build_upscaled_gpt2_auto_prefix():
    sd = _gpt2(2)
    grown = build_upscaled_state_dict(sd, upscale_layer_map(2, 3))
    assert sum(k.startswith("transformer.h.") for k in grown) == 3


def test_build_upscaled_duplicates_are_independent_copies():
    sd = _llama(2)
    grown = build_upscaled_state_dict(sd, [0, 1, 0, 1])
    # editing a duplicated layer must not alias the source layer
    grown["model.layers.2.self_attn.q_proj.weight"][:] = 99
    assert grown["model.layers.0.self_attn.q_proj.weight"][0, 0] == 0


# ── DataPipeline: labels ──────────────────────────────────────────────────────

def test_sequence_labels():
    rows = [{"input_ids": [1, 2, 3], "label": i % 3} for i in range(6)]
    pipe = DataPipeline(rows, batch_size=3, max_length=5, mode="embed", label_column="label")
    b = next(iter(pipe))
    assert set(b) == {"input_ids", "attention_mask", "labels"}
    assert b["labels"].shape == (3,)
    assert b["labels"].tolist() == [0, 1, 2]


def test_token_labels_padded_with_ignore_index():
    rows = [{"input_ids": [1, 2, 3], "ner": [5, 6, 7]} for _ in range(4)]
    pipe = DataPipeline(rows, batch_size=2, max_length=6, mode="embed", label_column="ner")
    b = next(iter(pipe))
    assert b["labels"].shape == (2, 6)
    assert b["labels"][0].tolist() == [5, 6, 7, -100, -100, -100]


def test_no_label_column_unchanged():
    rows = [{"input_ids": [1, 2]} for _ in range(4)]
    out = next(iter(DataPipeline(rows, batch_size=2, max_length=4, mode="lm")))
    assert isinstance(out, np.ndarray)          # lm mode without labels → bare array
    assert out.shape == (2, 4)


def test_custom_label_pad_id():
    rows = [{"input_ids": [1, 2], "ner": [3, 4]} for _ in range(2)]
    pipe = DataPipeline(rows, batch_size=2, max_length=4, mode="embed",
                        label_column="ner", label_pad_id=0)
    b = next(iter(pipe))
    assert b["labels"][0].tolist() == [3, 4, 0, 0]
