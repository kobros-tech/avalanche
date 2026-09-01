import torch
from torch import nn

from avalanche.training.skill_memory import (
    AdaptationCompatibilityScorer,
    SkillMemory,
    SkillMemoryPlugin,
)


class Experience:
    current_experience = 1


class Strategy:
    def __init__(self, model):
        self.model = model
        self.experience = Experience()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.train_epochs = 1


def factory():
    return nn.Linear(1, 1, bias=False)


def test_adaptation_scorer_detects_useful_initialization():
    memory = SkillMemory()
    source = factory()
    with torch.no_grad():
        source.weight.fill_(2.0)
    memory.register("double", source.state_dict())

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = 3.0 * x
    scorer = AdaptationCompatibilityScorer(
        model_factory=factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _: (x, y),
        steps=2,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
    )

    score = scorer(memory.get("double"), Experience())
    assert score > 0.0
    assert score <= 1.0


def test_adaptation_scorer_matches_minibatch_update_structure():
    """12 updates consume four 16-sample batches for three epochs."""
    model = factory()
    x = torch.arange(1.0, 65.0).reshape(-1, 1)
    y = 3.0 * x
    seen_batch_sizes = []
    seen_first_targets = []

    def recording_loss(prediction, target):
        seen_batch_sizes.append(len(target))
        seen_first_targets.append(float(target[0]))
        return nn.functional.mse_loss(prediction, target)

    scorer = AdaptationCompatibilityScorer(
        model_factory=factory,
        loss_fn=recording_loss,
        probe_fn=lambda _: (x, y),
        adaptation_fn=lambda _: (x, y),
        batch_size=16,
        steps=12,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
    )

    scorer._adapt(model, x, y)

    # Twelve optimizer updates use 12 minibatches of 16. The batches advance
    # through all four minibatches and then wrap for the second and third
    # epochs; the scorer must never silently repeat one 16-sample batch.
    assert seen_batch_sizes == [16] * 12 + [64]
    assert seen_first_targets == [3.0, 51.0, 99.0, 147.0] * 3 + [3.0]


def test_adaptation_scorer_rejects_invalid_batch_size():
    scorer = AdaptationCompatibilityScorer(
        model_factory=factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _: (torch.ones(2, 1), torch.ones(2, 1)),
        batch_size=0,
    )

    with torch.no_grad():
        record_model = factory()
    memory = SkillMemory()
    memory.register("skill", record_model.state_dict())

    try:
        scorer(memory.get("skill"), Experience())
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected invalid batch_size to raise ValueError")


def test_automatic_policy_uses_adaptation_value_for_clone():
    memory = SkillMemory()
    source = factory()
    with torch.no_grad():
        source.weight.fill_(2.0)
    memory.register("double", source.state_dict())

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = 3.0 * x
    clone_scorer = AdaptationCompatibilityScorer(
        model_factory=factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _: (x, y),
        steps=2,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
    )

    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: "target",
        clone_compatibility=clone_scorer,
    )
    strategy = Strategy(factory())
    plugin.before_training_exp(strategy)

    assert plugin.last_decision == plugin.CLONE
    assert plugin.last_selected_skill == "double"
    assert plugin.last_clone_value > 0.0
    assert torch.allclose(strategy.model.weight, source.weight)


def test_automatic_policy_scratches_when_candidate_does_not_beat_scratch():
    memory = SkillMemory()
    source = factory()
    with torch.no_grad():
        source.weight.fill_(100.0)
    memory.register("bad", source.state_dict())

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = 3.0 * x
    clone_scorer = AdaptationCompatibilityScorer(
        model_factory=factory,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda _: (x, y),
        steps=0,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
    )

    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: "target",
        clone_compatibility=clone_scorer,
    )
    strategy = Strategy(factory())
    initial = {k: v.clone() for k, v in strategy.model.state_dict().items()}
    plugin.before_training_exp(strategy)

    assert plugin.last_decision == plugin.SCRATCH
    assert plugin.last_selected_skill is None
    for key, value in initial.items():
        assert torch.equal(strategy.model.state_dict()[key], value)
