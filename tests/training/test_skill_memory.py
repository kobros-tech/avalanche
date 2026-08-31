import pytest
import torch
from torch import nn

from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin


class DummyExperience:
    def __init__(self, current_experience):
        self.current_experience = current_experience


class DummyStrategy:
    def __init__(self, model):
        self.model = model
        self.experience = None
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def test_register_copies_state_to_cpu_and_is_independent():
    memory = SkillMemory(max_skills=2)
    model = nn.Linear(2, 1)

    record = memory.register("a", model.state_dict(), metadata={"origin": "scratch"})
    assert record.name == "a"
    assert record.metadata["origin"] == "scratch"
    assert memory.names() == ["a"]

    with torch.no_grad():
        model.weight.add_(1.0)

    assert not torch.equal(record.state_dict["weight"], model.weight)
    assert all(t.device.type == "cpu" for t in record.state_dict.values())


def test_best_match_selects_highest_score_above_threshold():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    memory.register("low", model.state_dict())
    memory.register("high", model.state_dict())

    scores = {"low": 0.2, "high": 0.9}
    record, score = memory.best_match(
        "query", lambda record, _: scores[record.name], threshold=0.5
    )

    assert record is not None
    assert record.name == "high"
    assert score == pytest.approx(0.9)


def test_best_match_rejects_below_threshold():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    memory.register("skill", model.state_dict())

    record, score = memory.best_match(
        "query", lambda _, __: 0.4, threshold=0.5
    )

    assert record is None
    assert score == pytest.approx(0.4)


def test_memory_capacity_is_enforced():
    memory = SkillMemory(max_skills=1)
    model = nn.Linear(1, 1)
    memory.register("first", model.state_dict())

    with pytest.raises(RuntimeError, match="at capacity"):
        memory.register("second", model.state_dict())


def test_plugin_reuses_and_then_registers_skill():
    source = nn.Linear(1, 1)
    target = nn.Linear(1, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(2.0)

    memory = SkillMemory()
    memory.register("source", source.state_dict())

    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: f"task-{exp.current_experience}",
        compatibility=lambda record, exp: 1.0 if exp.current_experience == 1 else 0.0,
        threshold=0.5,
    )
    strategy = DummyStrategy(target)
    strategy.experience = DummyExperience(1)

    plugin.before_training_exp(strategy)
    assert plugin.last_reused_skill == "source"
    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)

    plugin.after_training_exp(strategy)
    assert memory.contains("task-1")
    assert memory.get("task-1").metadata["reused_from"] == "source"


def test_plugin_falls_back_when_no_skill_is_compatible():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    memory.register("source", model.state_dict())

    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: f"task-{exp.current_experience}",
        compatibility=lambda _, __: 0.1,
        threshold=0.5,
    )
    strategy = DummyStrategy(model)
    strategy.experience = DummyExperience(2)

    plugin.before_training_exp(strategy)

    assert plugin.last_reused_skill is None
    assert plugin.last_compatibility_score == pytest.approx(0.1)
