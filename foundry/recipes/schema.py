"""
Pydantic-validated recipe schema — fail fast on bad configs before any GPU spend.

Two recipe types:
  FoundryRecipe — full causal-LM factory (seed → grow → fuse → skill packs)
  EmbedRecipe   — compact embedding distillation (single teacher, MSE/cosine loss)
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Training data source for recipe.run()."""

    source:         str   = Field(..., description="HF dataset name or local path")
    name:           Optional[str] = Field(
        None, description="HF dataset config name, e.g. 'wikitext-2-raw-v1'"
    )
    split:          str   = "train"
    text_column:    str   = "text"
    batch_size:     int   = Field(8, gt=0)
    max_length:     int   = Field(512, gt=0)
    streaming:      bool  = True
    shuffle_buffer: int   = Field(0, ge=0)


class SeedConfig(BaseModel):
    """Where the model starts from."""

    arch:     Optional[str] = None     # custom Student class name / module path
    model:    Optional[str] = None     # HF model ID for warm-start
    init:     Literal["from_scratch", "pretrained"] = "pretrained"
    pretrain: Optional[dict] = None    # e.g. {"tokens": 1.2e12}

    @model_validator(mode="after")
    def _check_source(self) -> "SeedConfig":
        if self.init == "from_scratch" and not self.arch:
            raise ValueError("from_scratch requires 'arch' to be set.")
        if self.init == "pretrained" and not self.model:
            raise ValueError("pretrained requires 'model' to be set.")
        return self


class GrowConfig(BaseModel):
    """Optional depth up-scaling stage."""

    method:    Literal["depth_upscale"] = "depth_upscale"
    to_params: float = Field(..., gt=0, description="Target parameter count, e.g. 15e9")

    @field_validator("to_params", mode="before")
    @classmethod
    def _parse_params(cls, v):
        if isinstance(v, str):
            v = v.upper().replace("B", "e9").replace("M", "e6").replace("T", "e12")
            return float(v)
        return float(v)


class TeacherSpec(BaseModel):
    """One entry in the teachers list."""

    role:   str
    model:  str
    weight: float = Field(1.0, ge=0.0)


class FusionConfig(BaseModel):
    """Fusion/distillation settings."""

    strategy: Literal["min_ce", "mean"] = "min_ce"
    align:    Literal["identity", "em", "min_ed"] = "min_ed"
    cache:    str = "topk_64"           # e.g. "topk_64" or "topk_128"

    @property
    def top_k(self) -> int:
        return int(self.cache.split("_")[1]) if "_" in self.cache else 64


class HealConfig(BaseModel):
    """Continued distillation run after up-scaling."""

    tokens:  float = Field(..., gt=0)
    trainer: Literal["distill"] = "distill"
    alpha:   float = Field(0.3, ge=0.0, le=1.0)

    @field_validator("tokens", mode="before")
    @classmethod
    def _parse_tokens(cls, v):
        if isinstance(v, str):
            v = v.upper().replace("T", "e12").replace("B", "e9").replace("M", "e6")
            return float(v)
        return float(v)


class OutputConfig(BaseModel):
    """What to produce at the end of the recipe."""

    freeze_base: bool       = True
    skillpacks:  list[str]  = Field(default_factory=list)
    save_path:   Optional[str] = None


class FoundryRecipe(BaseModel):
    """
    Full recipe schema — the single config object for a causal-LM factory run.

    Example YAML::

        seed:
          model: meta-llama/Llama-3.1-8B
          init: pretrained
        grow:
          method: depth_upscale
          to_params: 15B
        teachers:
          - {role: reasoning, model: org/reasoning-teacher, weight: 1.0}
        fusion:
          strategy: min_ce
          align: min_ed
          cache: topk_64
        heal:
          tokens: 100B
          alpha: 0.3
        output:
          freeze_base: true
          skillpacks: [ola_math, ola_code]
    """

    seed:     SeedConfig
    data:     Optional[DataConfig]   = None
    grow:     Optional[GrowConfig]   = None
    teachers: list[TeacherSpec]      = Field(default_factory=list)
    fusion:   FusionConfig           = Field(default_factory=FusionConfig)
    heal:     Optional[HealConfig]   = None
    output:   OutputConfig           = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _warn_empty_teachers(self) -> "FoundryRecipe":
        if self.heal and not self.teachers:
            raise ValueError(
                "heal stage requires at least one teacher. "
                "Add a [[teachers]] entry or remove the heal block."
            )
        return self

    @model_validator(mode="after")
    def _warn_grow_without_seed_model(self) -> "FoundryRecipe":
        if self.grow and self.seed.init == "from_scratch":
            raise ValueError(
                "grow (depth_upscale) requires a pretrained seed — "
                "you cannot upscale a randomly-initialised model. "
                "Set seed.init to 'pretrained' and provide seed.model."
            )
        return self


# ── Embedding recipe ───────────────────────────────────────────────────────

class EmbedFusionConfig(BaseModel):
    """Fusion config for embedding distillation."""
    embed_loss: Literal["mse", "cosine"] = "cosine"
    embed_pool: Literal["mean", "cls"]   = "mean"
    normalize:  bool                     = True
    temperature: float                   = Field(1.0, gt=0.0)


class EmbedRecipe(BaseModel):
    """
    Compact embedding-model distillation recipe.

    Use this instead of FoundryRecipe when you want to train a bi-encoder /
    sentence-embedding student (e.g. 200M DeBERTa) by matching a larger
    teacher's pooled representations.

    No skill packs, no logit caching, no token-level fusion.

    Example YAML::

        seed:
          model: microsoft/deberta-v3-base
          init: pretrained
        teachers:
          - {role: embedding_teacher,
             model: BAAI/bge-large-en-v1.5,
             weight: 1.0}
        fusion:
          embed_loss: cosine
          embed_pool: mean
        heal:
          tokens: 1B
          alpha: 0.0
    """

    seed:     SeedConfig
    data:     Optional[DataConfig] = None
    teachers: list[TeacherSpec] = Field(
        ..., min_length=1,
        description="Exactly one embedding teacher is the common case.",
    )
    fusion:   EmbedFusionConfig = Field(default_factory=EmbedFusionConfig)
    heal:     Optional[HealConfig] = None
    output:   OutputConfig      = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _single_teacher_recommended(self) -> "EmbedRecipe":
        if len(self.teachers) > 1:
            import warnings
            warnings.warn(
                f"EmbedRecipe has {len(self.teachers)} teachers. "
                "Embedding distillation normally uses 1 teacher; "
                "multiple teachers will have their embeddings averaged.",
                UserWarning,
                stacklevel=2,
            )
        return self
