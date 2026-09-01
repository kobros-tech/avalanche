import pytest
import torch
from torch import nn

from avalanche.models import SimpleMLP
from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin
from avalanche.training.supervised import Naive
from tests.unit_tests_utils import get_fast_benchmark


class DummyExperience:
    def __init__(self, current_experience):
        self.current_experience = current_experience


class DummyStrategy:
    def __init__(self, model):
        self.model = model
        self.experience = None
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def test_empty_memory_returns_no_match():
    memory = SkillMemory()
    record, score = memory.best_match("query", lambda _, __: 1.0, threshold=0.5)
    assert record is None
    assert score == pytest.approx(0.0)


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
    record, score = memory.best_match("query", lambda record, _: scores[record.name], threshold=0.5)
    assert record is not None
    assert record.name == "high"
    assert score == pytest.approx(0.9)


def test_best_match_tie_break_is_deterministic():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    memory.register("first", model.state_dict())
    memory.register("second", model.state_dict())
    for _ in range(3):
        record, score = memory.best_match("query", lambda _, __: 0.8, threshold=0.5)
        assert record is not None
        assert record.name == "first"
        assert score == pytest.approx(0.8)


def test_best_match_rejects_below_threshold():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    memory.register("skill", model.state_dict())
    record, score = memory.best_match("query", lambda _, __: 0.4, threshold=0.5)
    assert record is None
    assert score == pytest.approx(0.4)


def test_best_match_rejects_invalid_threshold_and_nonfinite_score():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    memory.register("skill", model.state_dict())
    with pytest.raises(ValueError, match="threshold"):
        memory.best_match("query", lambda _, __: 1.0, threshold=1.1)
    with pytest.raises(ValueError, match="finite"):
        memory.best_match("query", lambda _, __: float("nan"))


def test_memory_capacity_is_enforced():
    memory = SkillMemory(max_skills=1)
    model = nn.Linear(1, 1)
    memory.register("first", model.state_dict())
    with pytest.raises(RuntimeError, match="at capacity"):
        memory.register("second", model.state_dict())


def test_memory_state_round_trip_preserves_skills_and_metadata():
    source = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(2.0)
    original = SkillMemory(max_skills=3)
    original.register("a", source.state_dict(), metadata={"task": 1})
    original.register("b", source.state_dict(), metadata={"task": 2})
    restored = SkillMemory()
    restored.load_state_dict(original.state_dict())
    assert restored.max_skills == 3
    assert restored.names() == ["a", "b"]
    assert restored.get("a").metadata == {"task": 1}
    for key, value in original.get("a").state_dict.items():
        assert torch.equal(value, restored.get("a").state_dict[key])


def test_plugin_reuse_does_not_register_a_new_skill():
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
        skill_metadata=lambda exp: {"task_id": exp.current_experience},
        compatibility=lambda record, exp: 1.0 if exp.current_experience == 1 else 0.0,
        threshold=0.5,
    )
    strategy = DummyStrategy(target)
    strategy.experience = DummyExperience(1)
    plugin.before_training_exp(strategy)
    assert plugin.last_decision == plugin.REUSE
    assert plugin.last_reused_skill == "source"
    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)
    plugin.after_training_exp(strategy)
    assert not memory.contains("task-1")
    assert memory.names() == ["source"]


def test_plugin_clone_registers_new_skill_and_resets_optimizer():
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
        compatibility=lambda _, __: 0.5,
        threshold=0.3,
    )
    strategy = DummyStrategy(target)
    strategy.experience = DummyExperience(1)
    parameter = next(target.parameters())
    strategy.optimizer.state[parameter] = {"momentum_buffer": torch.ones_like(parameter)}
    plugin.before_training_exp(strategy)
    assert plugin.last_decision == plugin.CLONE
    assert plugin.last_selected_skill == "source"
    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)
    assert len(strategy.optimizer.state) == 0
    with torch.no_grad():
        target.weight.add_(10.0)
        target.bias.add_(10.0)
    plugin.after_training_exp(strategy)
    assert memory.contains("task-1")
    assert memory.get("task-1").metadata["acquisition_decision"] == plugin.CLONE
    assert torch.allclose(memory.get("source").state_dict["weight"], source.weight)
    assert not torch.allclose(
        memory.get("task-1").state_dict["weight"],
        memory.get("source").state_dict["weight"],
    )


def test_plugin_does_not_overwrite_existing_skill_by_default():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    memory.register("task-1", model.state_dict())
    with torch.no_grad():
        model.weight.fill_(9.0)
    plugin = SkillMemoryPlugin(memory, skill_name=lambda exp: "task-1", skill_metadata=lambda _: {"version": 2})
    strategy = DummyStrategy(model)
    strategy.experience = DummyExperience(1)
    plugin.after_training_exp(strategy)
    assert torch.allclose(memory.get("task-1").state_dict["weight"], torch.ones_like(model.weight))
    assert memory.get("task-1").metadata == {}


def test_plugin_can_replace_existing_skill_explicitly():
    memory = SkillMemory()
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    memory.register("task-1", model.state_dict())
    with torch.no_grad():
        model.weight.fill_(9.0)
    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: "task-1",
        skill_metadata=lambda _: {"version": 2},
        replace_existing=True,
    )
    strategy = DummyStrategy(model)
    strategy.experience = DummyExperience(1)
    plugin.after_training_exp(strategy)
    assert torch.allclose(memory.get("task-1").state_dict["weight"], model.weight)
    assert memory.get("task-1").metadata["version"] == 2


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
    assert plugin.last_decision == plugin.SCRATCH


def test_plugin_runs_across_sequential_experiences():
    """Check reuse and acquisition decisions across two real experiences."""
    benchmark = get_fast_benchmark()
    model = SimpleMLP(input_size=6, hidden_size=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    memory = SkillMemory()
    seen = []

    def compatibility(record, exp):
        seen.append(exp.current_experience)
        if exp.current_experience == 1 and record.name == "experience-0":
            return 1.0
        return 0.0

    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: f"experience-{exp.current_experience}",
        compatibility=compatibility,
        threshold=0.5,
    )
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=16,
        train_epochs=1,
        eval_every=-1,
        plugins=[plugin],
    )
    strategy.train(benchmark.train_stream[0])
    assert plugin.last_reused_skill is None
    assert memory.contains("experience-0")
    assert seen == []
    strategy.train(benchmark.train_stream[1])
    assert plugin.last_reused_skill == "experience-0"
    assert not memory.contains("experience-1")
    assert seen == [1]


def test_plugin_does_not_query_memory_during_eval():
    """Compatibility checks happen on training experiences, not test data."""
    benchmark = get_fast_benchmark()
    model = SimpleMLP(input_size=6, hidden_size=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    memory = SkillMemory()
    source = SimpleMLP(input_size=6, hidden_size=10)
    memory.register("seed-skill", source.state_dict())
    calls = []

    def compatibility(record, exp):
        calls.append(exp.current_experience)
        return 0.0

    plugin = SkillMemoryPlugin(memory, compatibility=compatibility, threshold=0.5)
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=16,
        train_epochs=1,
        eval_every=-1,
        plugins=[plugin],
    )
    strategy.train(benchmark.train_stream[0])
    train_call_count = len(calls)
    assert train_call_count > 0
    strategy.eval(benchmark.test_stream[0])
    assert len(calls) == train_call_count


def test_plugin_runs_inside_real_avalanche_training_loop():
    """Check that the plugin is invoked by Naive, not only by direct calls."""
    benchmark = get_fast_benchmark()
    model = SimpleMLP(input_size=6, hidden_size=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    memory = SkillMemory()
    source = SimpleMLP(input_size=6, hidden_size=10)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    memory.register("seed-skill", source.state_dict())
    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: f"experience-{exp.current_experience}",
        compatibility=lambda record, exp: (
            1.0 if record.name == "seed-skill" and exp.current_experience == 0 else 0.0
        ),
        threshold=0.5,
    )
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=16,
        train_epochs=1,
        eval_every=-1,
        plugins=[plugin],
    )
    strategy.train(benchmark.train_stream[0])
    assert plugin.last_reused_skill == "seed-skill"
    assert not memory.contains("experience-0")
    assert memory.names() == ["seed-skill"]
