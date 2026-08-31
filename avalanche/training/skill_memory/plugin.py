"""Avalanche plugin for optional skill reuse between experiences."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from avalanche.core import BasePlugin

from .memory import SkillMemory, SkillRecord


class SkillMemoryPlugin(BasePlugin):
    """Reuse compatible model states before training an experience.

    The plugin is intentionally policy-light. Callers provide:

    * ``skill_name``: a function mapping an experience to the name under which
      the learned skill should be stored;
    * ``compatibility``: a function scoring a stored skill against an
      experience in ``[0, 1]``;
    * ``skill_metadata``: an optional function returning metadata to store with
      the acquired skill.

    A compatible skill is loaded immediately before the experience is trained.
    The resulting model state is then registered after training. This makes the
    mechanism usable with existing Avalanche strategies without replacing
    their training loop.

    Reuse is opt-in: when ``compatibility`` is ``None``, no skill is reused and
    every experience follows the strategy's normal training path.
    """

    def __init__(
        self,
        memory: Optional[SkillMemory] = None,
        *,
        skill_name: Callable[[Any], str] = lambda exp: str(exp.current_experience),
        compatibility: Optional[Callable[[SkillRecord, Any], float]] = None,
        skill_metadata: Optional[Callable[[Any], Mapping[str, Any]]] = None,
        threshold: float = 0.5,
        replace_existing: bool = False,
        reset_optimizer_on_reuse: bool = False,
    ):
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.memory = memory if memory is not None else SkillMemory()
        self.skill_name = skill_name
        self.compatibility = compatibility
        self.skill_metadata = skill_metadata
        self.threshold = threshold
        self.replace_existing = replace_existing
        self.reset_optimizer_on_reuse = reset_optimizer_on_reuse

        self.last_reused_skill: Optional[str] = None
        self.last_compatibility_score: float = 0.0

    def before_training_exp(self, strategy, **kwargs):
        """Attempt to initialize the model from a compatible stored skill."""

        self.last_reused_skill = None
        self.last_compatibility_score = 0.0
        if self.compatibility is None or len(self.memory) == 0:
            return

        record, score = self.memory.best_match(
            strategy.experience,
            self.compatibility,
            threshold=self.threshold,
        )
        self.last_compatibility_score = score
        if record is None:
            return

        self.memory.load_into(record.name, strategy.model)
        self.last_reused_skill = record.name

        if self.reset_optimizer_on_reuse and hasattr(strategy, "optimizer"):
            strategy.optimizer.state.clear()

    def after_training_exp(self, strategy, **kwargs):
        """Register the post-training model state as the current skill."""

        experience = strategy.experience
        name = self.skill_name(experience)
        metadata = dict(self.skill_metadata(experience)) if self.skill_metadata else {}
        metadata.update(
            {
                "reused_from": self.last_reused_skill,
                "compatibility_score": self.last_compatibility_score,
            }
        )

        if self.memory.contains(name) and not self.replace_existing:
            return

        self.memory.register(
            name,
            strategy.model.state_dict(),
            metadata=metadata,
            replace=self.replace_existing,
        )
