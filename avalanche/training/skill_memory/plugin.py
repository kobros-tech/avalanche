"""Avalanche plugin for automatic score-driven skill acquisition."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from .memory import SkillMemory, SkillRecord


class SkillMemoryPlugin(SupervisedPlugin):
    """Automatically choose REUSE, CLONE, or SCRATCH for each experience.

    A learned skill is stored as an independent model-state copy in
    :class:`SkillMemory`. For each new training experience the best stored
    candidate is scored. The score itself is the trigger:

    * score >= ``reuse_threshold`` -> REUSE
    * ``clone_threshold`` <= score < ``reuse_threshold`` -> CLONE
    * score < ``clone_threshold`` or no candidate -> SCRATCH

    REUSE and CLONE both initialize the active learner from an independent
    copy of the selected skill. Training therefore never mutates the stored
    source skill. The distinction is the policy decision and its recorded
    provenance: REUSE treats the candidate as sufficiently compatible for
    direct reuse, while CLONE treats it as useful initialization that must be
    adapted by training on the new experience.
    """

    REUSE = "reuse"
    CLONE = "clone"
    SCRATCH = "scratch"

    def __init__(
        self,
        memory: Optional[SkillMemory] = None,
        *,
        skill_name: Callable[[Any], str] = lambda exp: str(exp.current_experience),
        compatibility: Optional[Callable[[SkillRecord, Any], float]] = None,
        skill_metadata: Optional[Callable[[Any], Mapping[str, Any]]] = None,
        reuse_threshold: float = 0.90,
        clone_threshold: float = 0.30,
        # Kept for compatibility with the prototype API. When supplied, it
        # acts as the clone threshold unless the new thresholds are explicit.
        threshold: Optional[float] = None,
        replace_existing: bool = False,
        reset_optimizer_on_reuse: bool = False,
    ):
        super().__init__()
        if threshold is not None:
            clone_threshold = threshold
        if not 0.0 <= clone_threshold <= 1.0:
            raise ValueError("clone_threshold must be in [0, 1]")
        if not 0.0 <= reuse_threshold <= 1.0:
            raise ValueError("reuse_threshold must be in [0, 1]")
        if clone_threshold > reuse_threshold:
            raise ValueError("clone_threshold must not exceed reuse_threshold")

        self.memory = memory if memory is not None else SkillMemory()
        self.skill_name = skill_name
        self.compatibility = compatibility
        self.skill_metadata = skill_metadata
        self.reuse_threshold = reuse_threshold
        self.clone_threshold = clone_threshold
        self.replace_existing = replace_existing
        self.reset_optimizer_on_reuse = reset_optimizer_on_reuse

        self.last_decision: str = self.SCRATCH
        self.last_selected_skill: Optional[str] = None
        self.last_reused_skill: Optional[str] = None
        self.last_compatibility_score: float = 0.0

    def before_training_exp(self, strategy, **kwargs):
        """Select the acquisition policy and initialize the active learner."""

        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_reused_skill = None
        self.last_compatibility_score = 0.0

        if self.compatibility is None or len(self.memory) == 0:
            return

        # Query at the clone threshold so a candidate in the clone range is
        # still available for the policy decision.
        record, score = self.memory.best_match(
            strategy.experience,
            self.compatibility,
            threshold=self.clone_threshold,
        )
        self.last_compatibility_score = score
        if record is None:
            return

        self.last_selected_skill = record.name
        if score >= self.reuse_threshold:
            self.last_decision = self.REUSE
            self.last_reused_skill = record.name
        else:
            self.last_decision = self.CLONE

        # SkillMemory.load_into performs an independent copy, so subsequent
        # training operates on the active model and cannot mutate the source.
        self.memory.load_into(record.name, strategy.model)

        if self.reset_optimizer_on_reuse and hasattr(strategy, "optimizer"):
            strategy.optimizer.state.clear()

    def after_training_exp(self, strategy, **kwargs):
        """Register the newly acquired model state after every experience."""

        experience = strategy.experience
        name = self.skill_name(experience)
        metadata = dict(self.skill_metadata(experience)) if self.skill_metadata else {}
        metadata.update(
            {
                "acquisition_decision": self.last_decision,
                "selected_skill": self.last_selected_skill,
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
