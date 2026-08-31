"""Minimal Skill Memory example.

The example keeps task/skill compatibility outside the memory itself. This is
intentional: different continual-learning settings can provide different task
descriptors and compatibility estimators without changing the registry.
"""

import torch
from torch import nn
from torch.optim import SGD

from avalanche.training import Naive
from avalanche.training.skill_memory import SkillMemory, SkillMemoryPlugin


class TaskDescriptor:
    def __init__(self, task_id: int):
        self.task_id = task_id


def compatibility(record, experience):
    """Example compatibility function for task IDs.

    Real applications should use an independently defined task/skill
    compatibility estimator rather than inspecting target test labels.
    """
    return 1.0 if record.metadata.get("task_id") == experience.current_experience else 0.0


model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))
optimizer = SGD(model.parameters(), lr=0.01)
memory = SkillMemory(max_skills=10)

plugin = SkillMemoryPlugin(
    memory=memory,
    skill_name=lambda exp: f"task-{exp.current_experience}",
    compatibility=compatibility,
    threshold=0.8,
)

strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    train_mb_size=32,
    train_epochs=1,
    eval_mb_size=32,
    plugins=[plugin],
)

# Train normally on an Avalanche scenario:
# strategy.train(benchmark.train_stream)
# strategy.eval(benchmark.test_stream)
#
# The plugin will query the memory before each experience and register the
# resulting model state afterward.
