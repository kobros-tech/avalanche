import torch
from torch import nn

from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin


class Experience:
    def __init__(self, index):
        self.current_experience = index


class Strategy:
    def __init__(self, model, experience):
        self.model = model
        self.experience = experience
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def make_memory():
    memory = SkillMemory()
    source = nn.Linear(1, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(2.0)
    memory.register("source", source.state_dict())
    return memory, source


def run_decision(score):
    memory, source = make_memory()
    model = nn.Linear(1, 1)
    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: f"task-{exp.current_experience}",
        compatibility=lambda _, __: score,
    )
    strategy = Strategy(model, Experience(1))
    plugin.before_training_exp(strategy)
    return plugin, strategy, memory, source


def test_empty_memory_is_scratch():
    plugin = SkillMemoryPlugin(SkillMemory(), compatibility=lambda _, __: 1.0)
    strategy = Strategy(nn.Linear(1, 1), Experience(0))

    plugin.before_training_exp(strategy)

    assert plugin.last_decision == plugin.SCRATCH
    assert plugin.last_selected_skill is None


def test_score_below_clone_threshold_is_scratch():
    plugin, strategy, _, source = run_decision(0.29)

    plugin.before_training_exp(strategy)

    assert plugin.last_decision == plugin.SCRATCH
    assert plugin.last_selected_skill is None
    assert not torch.allclose(strategy.model.weight, source.weight)


def test_score_in_clone_range_selects_clone():
    plugin, strategy, _, source = run_decision(0.30)

    plugin.before_training_exp(strategy)

    assert plugin.last_decision == plugin.CLONE
    assert plugin.last_selected_skill == "source"
    assert torch.allclose(strategy.model.weight, source.weight)
    assert torch.allclose(strategy.model.bias, source.bias)


def test_score_at_reuse_threshold_selects_reuse():
    plugin, strategy, _, source = run_decision(0.90)

    plugin.before_training_exp(strategy)

    assert plugin.last_decision == plugin.REUSE
    assert plugin.last_selected_skill == "source"
    assert plugin.last_reused_skill == "source"
    assert torch.allclose(strategy.model.weight, source.weight)
    assert torch.allclose(strategy.model.bias, source.bias)


def test_highest_candidate_is_selected_and_policy_uses_its_score():
    memory, source = make_memory()
    other = nn.Linear(1, 1)
    memory.register("other", other.state_dict())
    scores = {"source": 0.40, "other": 0.95}
    plugin = SkillMemoryPlugin(
        memory,
        compatibility=lambda record, _: scores[record.name],
    )
    strategy = Strategy(nn.Linear(1, 1), Experience(2))

    plugin.before_training_exp(strategy)

    assert plugin.last_selected_skill == "other"
    assert plugin.last_decision == plugin.REUSE
    assert plugin.last_compatibility_score == 0.95


def test_training_after_clone_does_not_mutate_stored_source():
    plugin, strategy, memory, source = run_decision(0.50)
    plugin.before_training_exp(strategy)
    stored_before = {k: v.clone() for k, v in memory.get("source").state_dict.items()}

    with torch.no_grad():
        strategy.model.weight.add_(10.0)
        strategy.model.bias.add_(10.0)

    assert plugin.last_decision == plugin.CLONE
    assert torch.allclose(memory.get("source").state_dict["weight"], stored_before["weight"])
    assert torch.allclose(memory.get("source").state_dict["bias"], stored_before["bias"])
    assert not torch.allclose(memory.get("source").state_dict["weight"], strategy.model.weight)


def test_training_after_reuse_does_not_mutate_stored_source():
    plugin, strategy, memory, source = run_decision(0.95)
    plugin.before_training_exp(strategy)
    stored_before = {k: v.clone() for k, v in memory.get("source").state_dict.items()}

    with torch.no_grad():
        strategy.model.weight.sub_(7.0)
        strategy.model.bias.sub_(7.0)

    assert plugin.last_decision == plugin.REUSE
    assert torch.allclose(memory.get("source").state_dict["weight"], stored_before["weight"])
    assert torch.allclose(memory.get("source").state_dict["bias"], stored_before["bias"])


def test_new_skill_is_registered_after_every_acquisition():
    memory, _ = make_memory()
    plugin = SkillMemoryPlugin(
        memory,
        skill_name=lambda exp: f"task-{exp.current_experience}",
        compatibility=lambda _, __: 0.1,
    )
    model = nn.Linear(1, 1)
    strategy = Strategy(model, Experience(2))

    plugin.before_training_exp(strategy)
    plugin.after_training_exp(strategy)

    assert plugin.last_decision == plugin.SCRATCH
    assert memory.contains("task-2")
    assert memory.get("task-2").metadata["acquisition_decision"] == plugin.SCRATCH


def test_thresholds_are_configurable():
    memory, _ = make_memory()
    plugin = SkillMemoryPlugin(
        memory,
        reuse_threshold=0.80,
        clone_threshold=0.20,
        compatibility=lambda _, __: 0.80,
    )
    strategy = Strategy(nn.Linear(1, 1), Experience(1))

    plugin.before_training_exp(strategy)

    assert plugin.last_decision == plugin.REUSE
    assert plugin.reuse_threshold == 0.80
    assert plugin.clone_threshold == 0.20
