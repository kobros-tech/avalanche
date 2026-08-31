"""Generic storage and matching primitives for reusable learned skills.

The memory intentionally stores model state independently from Avalanche
strategies. A strategy/plugin decides what constitutes a skill and how a new
experience is matched to stored skills.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import torch
from torch import Tensor


class SkillMemoryError(RuntimeError):
    """Base exception for skill-memory errors."""


class SkillNotFoundError(SkillMemoryError, KeyError):
    """Raised when a requested skill is not present."""


@dataclass
class SkillRecord:
    """A stored learned skill.

    ``state_dict`` is kept on CPU so that storing skills does not retain GPU
    memory. Metadata is deliberately unconstrained: callers can record task
    descriptors, provenance, acquisition cost, or experiment information.
    """

    name: str
    state_dict: dict[str, Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)
    creation_order: int = 0


def _copy_state_dict_to_cpu(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return an independent CPU copy of a model state dictionary."""

    result: dict[str, Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(
                "skill state dictionaries must contain torch.Tensor values; "
                f"got {type(value).__name__} for '{name}'"
            )
        result[name] = value.detach().cpu().clone()
    return result


class SkillMemory:
    """Bounded registry of reusable model states.

    Matching is supplied by the caller through ``best_match``. This keeps the
    memory independent from a particular task representation or compatibility
    metric.
    """

    def __init__(self, max_skills: Optional[int] = None):
        if max_skills is not None and max_skills < 1:
            raise ValueError("max_skills must be positive or None")
        self.max_skills = max_skills
        self._records: dict[str, SkillRecord] = {}
        self._order = 0

    def register(
        self,
        name: str,
        state_dict: Mapping[str, Tensor],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        replace: bool = False,
    ) -> SkillRecord:
        """Store a learned skill and return its record.

        By default registration is immutable: replacing an existing skill must
        be explicitly requested with ``replace=True``.
        """

        if name in self._records and not replace:
            raise SkillMemoryError(f"skill '{name}' is already registered")
        if name not in self._records and self.max_skills is not None:
            if len(self._records) >= self.max_skills:
                raise SkillMemoryError(
                    f"skill memory is at capacity ({self.max_skills} skills)"
                )

        self._order += 1
        record = SkillRecord(
            name=name,
            state_dict=_copy_state_dict_to_cpu(state_dict),
            metadata=dict(metadata or {}),
            creation_order=self._order,
        )
        self._records[name] = record
        return record

    def get(self, name: str) -> SkillRecord:
        try:
            return self._records[name]
        except KeyError:
            raise SkillNotFoundError(name) from None

    def contains(self, name: str) -> bool:
        return name in self._records

    def names(self) -> list[str]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def available_capacity(self) -> Optional[int]:
        if self.max_skills is None:
            return None
        return self.max_skills - len(self._records)

    def best_match(
        self,
        query: Any,
        compatibility: Callable[[SkillRecord, Any], float],
        *,
        threshold: float = 0.0,
    ) -> tuple[Optional[SkillRecord], float]:
        """Return the highest-scoring compatible skill.

        The callback must return a finite numeric score. Ties are resolved by
        creation order, making retrieval deterministic for a fixed memory and
        scorer.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        best_record: Optional[SkillRecord] = None
        best_score = float("-inf")
        for record in self._records.values():
            score = float(compatibility(record, query))
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("compatibility scores must be finite")
            if score > best_score:
                best_record = record
                best_score = score

        if best_record is None or best_score < threshold:
            return None, best_score if best_record is not None else 0.0
        return best_record, best_score

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpointable copy of the complete memory state."""

        return {
            "max_skills": self.max_skills,
            "order": self._order,
            "records": {
                name: {
                    "state_dict": _copy_state_dict_to_cpu(record.state_dict),
                    "metadata": deepcopy(record.metadata),
                    "creation_order": record.creation_order,
                }
                for name, record in self._records.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore memory state produced by :meth:`state_dict`."""

        records = state.get("records")
        if not isinstance(records, Mapping):
            raise ValueError("invalid skill-memory state: missing records")

        max_skills = state.get("max_skills", self.max_skills)
        if max_skills is not None and max_skills < 1:
            raise ValueError("max_skills must be positive or None")
        if max_skills is not None and len(records) > max_skills:
            raise ValueError("checkpoint contains more skills than max_skills")

        restored: dict[str, SkillRecord] = {}
        for name, value in records.items():
            if not isinstance(value, Mapping) or "state_dict" not in value:
                raise ValueError(f"invalid skill record for '{name}'")
            restored[str(name)] = SkillRecord(
                name=str(name),
                state_dict=_copy_state_dict_to_cpu(value["state_dict"]),
                metadata=deepcopy(dict(value.get("metadata", {}))),
                creation_order=int(value.get("creation_order", 0)),
            )

        self.max_skills = max_skills
        self._records = restored
        self._order = int(state.get("order", max(
            (record.creation_order for record in restored.values()), default=0
        )))

    def load_into(self, name: str, model: torch.nn.Module) -> SkillRecord:
        """Load a stored skill into ``model`` and return its record."""

        record = self.get(name)
        model.load_state_dict(deepcopy(record.state_dict))
        return record
