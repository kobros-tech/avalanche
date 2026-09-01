"""Continuous compatibility and adaptation-value scorers for Skill Memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor, nn

from .memory import SkillRecord


@dataclass
class ProbeCompatibilityScorer:
    """Score a stored skill by zero-shot error on a live training-data probe."""

    model_factory: Callable[[], nn.Module]
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    probe_fn: Callable[[Any], Tuple[Tensor, Tensor]]
    reference_fn: Callable[[Any], float]
    min_reference: float = 1e-8

    def __call__(self, record: SkillRecord, query: Any) -> float:
        model = self.model_factory()
        model.load_state_dict(record.state_dict)
        model.eval()
        x, y = self.probe_fn(query)
        with torch.no_grad():
            probe_error = float(self.loss_fn(model(x), y))
        reference_error = float(self.reference_fn(query))
        if reference_error <= self.min_reference:
            return 0.0 if probe_error > self.min_reference else 1.0
        return max(0.0, min(1.0, 1.0 - probe_error / reference_error))


@dataclass
class AdaptationCompatibilityScorer:
    """Measure transfer value against a fair, matched SCRATCH control.

    Candidate and SCRATCH models receive the same adaptation data, optimizer,
    minibatch ordering, and number of optimizer updates. When ``batch_size``
    is omitted, the scorer retains the original single-batch behavior for
    backwards-compatible unit tests and small probes.

    With ``batch_size`` set, the returned adaptation loss is measured after the
    exact requested number of updates. Batches are consumed in deterministic
    order and the sequence wraps to the beginning when more updates than one
    pass over the adaptation data are requested. This makes it possible to
    match an Avalanche learner's minibatch training procedure exactly, e.g.
    64 samples / batch size 16 / 12 updates = 3 complete epochs.

    The adaptation data must come only from the new experience's training
    distribution; it must never be the final evaluation stream.
    """

    model_factory: Callable[[], nn.Module]
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    probe_fn: Callable[[Any], Tuple[Tensor, Tensor]]
    steps: int = 3
    batch_size: Optional[int] = None
    adaptation_fn: Optional[Callable[[Any], Tuple[Tensor, Tensor]]] = None
    optimizer_factory: Callable[[Any], torch.optim.Optimizer] = (
        lambda parameters: torch.optim.SGD(parameters, lr=1e-2)
    )
    min_scratch_loss: float = 1e-8

    def _adapt(self, model: nn.Module, x: Tensor, y: Tensor) -> float:
        optimizer = self.optimizer_factory(model.parameters())
        model.train()

        if self.batch_size is None:
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = self.loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
        else:
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            if len(x) == 0:
                raise ValueError("adaptation data must not be empty")
            if len(x) != len(y):
                raise ValueError("adaptation inputs and targets must have equal length")

            num_batches = (len(x) + self.batch_size - 1) // self.batch_size
            for update in range(self.steps):
                batch_index = update % num_batches
                start = batch_index * self.batch_size
                end = min(start + self.batch_size, len(x))
                batch_x = x[start:end]
                batch_y = y[start:end]

                optimizer.zero_grad()
                loss = self.loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            return float(self.loss_fn(model(x), y))

    def __call__(self, record: SkillRecord, query: Any) -> float:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        data_fn = self.adaptation_fn or self.probe_fn
        x, y = data_fn(query)

        rng_state = torch.random.get_rng_state()
        candidate = self.model_factory()
        candidate.load_state_dict(record.state_dict)
        candidate_loss = self._adapt(candidate, x, y)

        # Recreate the exact same fresh initialization for SCRATCH so the
        # comparison isolates initialization quality from random luck.
        torch.random.set_rng_state(rng_state)
        scratch = self.model_factory()
        scratch_loss = self._adapt(scratch, x, y)

        if scratch_loss <= self.min_scratch_loss:
            return 0.0

        score = 1.0 - candidate_loss / scratch_loss
        return max(0.0, min(1.0, score))


def mean_baseline_mse(y: Tensor) -> float:
    """Reference MSE of predicting the probe-target mean for every sample."""
    mean_prediction = y.mean(dim=0, keepdim=True)
    return float(nn.functional.mse_loss(mean_prediction.expand_as(y), y))


def uniform_guess_cross_entropy(num_classes: int) -> Callable[[Any], float]:
    """Reference cross-entropy of guessing uniformly over ``num_classes``."""
    import math

    reference = math.log(num_classes)

    def _reference_fn(_query: Any) -> float:
        return reference

    return _reference_fn
