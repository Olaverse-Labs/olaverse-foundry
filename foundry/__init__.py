"""
olaverse-foundry — a toolkit for building model families.

One expensive ancestor, many cheap descendants.

    seed → grow → fuse/heal → freeze → skill packs

Quick start::

    from foundry import Recipe

    recipe = Recipe.load("my_recipe.yaml")
    for line in recipe.plan():
        print(line)

    base = recipe.run()
"""
from foundry.contracts import ArchConfig, Student, Teacher, TokenizerAlignment
from foundry.fusion import FusionKernel, IdentityAlignment, EMAlignment, MinEDAlignment
from foundry.fusion import STRATEGY_REGISTRY, min_ce, mean_ce
from foundry.growth import GrowthPlan, upscale_layer_map, layers_for_param_target, plan_growth
from foundry.skillpacks import SkillPack, SkillRegistry
from foundry.teachers import TeacherRegistry, ToyTeacher, LogitCache
from foundry.training import DistillTrainer, TrainConfig
from foundry.recipes import Recipe, FoundryRecipe
from foundry.backends import detect_backend

__version__ = "0.1.0"

__all__ = [
    # Contracts
    "ArchConfig", "Student", "Teacher", "TokenizerAlignment",
    # Fusion
    "FusionKernel", "IdentityAlignment", "EMAlignment", "MinEDAlignment",
    "STRATEGY_REGISTRY", "min_ce", "mean_ce",
    # Growth
    "GrowthPlan", "upscale_layer_map", "layers_for_param_target", "plan_growth",
    # Skill packs
    "SkillPack", "SkillRegistry",
    # Teachers
    "TeacherRegistry", "ToyTeacher", "LogitCache",
    # Training
    "DistillTrainer", "TrainConfig",
    # Recipes
    "Recipe", "FoundryRecipe",
    # Backends
    "detect_backend",
]
