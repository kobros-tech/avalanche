"""Reusable skill-memory primitives for continual-learning strategies."""

from .memory import SkillMemory, SkillMemoryError, SkillNotFoundError, SkillRecord
from .plugin import SkillMemoryPlugin
from .compatibility import (
    AdaptationCompatibilityScorer,
    ProbeCompatibilityScorer,
    mean_baseline_mse,
    uniform_guess_cross_entropy,
)

__all__ = [
    "SkillMemory",
    "SkillMemoryError",
    "SkillNotFoundError",
    "SkillRecord",
    "SkillMemoryPlugin",
    "ProbeCompatibilityScorer",
    "AdaptationCompatibilityScorer",
    "mean_baseline_mse",
    "uniform_guess_cross_entropy",
]
