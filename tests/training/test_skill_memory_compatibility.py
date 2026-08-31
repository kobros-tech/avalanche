import torch
from torch import nn

from avalanche.training.skill_memory import (
    ProbeCompatibilityScorer,
    SkillMemory,
    mean_baseline_mse,
    uniform_guess_cross_entropy,
)


def linear_factory():
    return nn.Linear(1, 1, bias=False)


def test_perfect_candidate_scores_near_one():
    memory = SkillMemory()
    source = linear_factory()
    with torch.no_grad():
        source.weight.fill_(2.0)
    memory.register("double", source.state_dict())

    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = 2.0 * x

    scorer = ProbeCompatibilityScorer(
        model_factory=linear_factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _query: (x, y),
        reference_fn=lambda _query: mean_baseline_mse(y),
    )

    score = scorer(memory.get("double"), object())
    assert score > 0.99


def test_useless_candidate_scores_near_zero():
    memory = SkillMemory()
    source = linear_factory()
    with torch.no_grad():
        source.weight.fill_(-50.0)
    memory.register("bad", source.state_dict())

    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = 2.0 * x

    scorer = ProbeCompatibilityScorer(
        model_factory=linear_factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _query: (x, y),
        reference_fn=lambda _query: mean_baseline_mse(y),
    )

    score = scorer(memory.get("bad"), object())
    assert score == 0.0


def test_partial_transfer_lands_strictly_between_zero_and_one():
    memory = SkillMemory()
    # Learned to double: y = 2x. Probe task is y = 2x + small offset, so the
    # skill is useful but not perfect -- a legitimate CLONE candidate.
    source = linear_factory()
    with torch.no_grad():
        source.weight.fill_(2.0)
    memory.register("double", source.state_dict())

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * x + 1.5

    scorer = ProbeCompatibilityScorer(
        model_factory=linear_factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _query: (x, y),
        reference_fn=lambda _query: mean_baseline_mse(y),
    )

    score = scorer(memory.get("double"), object())
    assert 0.0 < score < 1.0


def test_degenerate_reference_does_not_divide_by_zero():
    memory = SkillMemory()
    source = linear_factory()
    memory.register("any", source.state_dict())

    x = torch.tensor([[1.0]])
    y = torch.tensor([[5.0]])

    scorer = ProbeCompatibilityScorer(
        model_factory=linear_factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _query: (x, y),
        reference_fn=lambda _query: 0.0,
    )

    score = scorer(memory.get("any"), object())
    assert 0.0 <= score <= 1.0


def test_uniform_guess_cross_entropy_is_log_num_classes():
    import math

    reference_fn = uniform_guess_cross_entropy(10)
    assert reference_fn(object()) == math.log(10)


def test_scoring_does_not_mutate_stored_skill():
    memory = SkillMemory()
    source = linear_factory()
    with torch.no_grad():
        source.weight.fill_(2.0)
    memory.register("double", source.state_dict())
    stored_before = memory.get("double").state_dict["weight"].clone()

    x = torch.tensor([[1.0]])
    y = torch.tensor([[2.0]])
    scorer = ProbeCompatibilityScorer(
        model_factory=linear_factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _query: (x, y),
        reference_fn=lambda _query: mean_baseline_mse(y),
    )

    scorer(memory.get("double"), object())

    assert torch.allclose(memory.get("double").state_dict["weight"], stored_before)
