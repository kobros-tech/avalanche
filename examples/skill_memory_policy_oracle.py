"""Oracle comparison: does the automatic policy pick what actually helps?

The multi-seed arithmetic benchmark (``skill_memory_benchmark.py``) measures
whether Skill Memory beats a plain baseline. That is necessary but not
sufficient evidence for the automatic REUSE/CLONE/SCRATCH policy itself: a
good *average* result could still hide an automatic policy that regularly
picks the wrong strategy for a given task while getting bailed out elsewhere.

This experiment asks the sharper question directly. For every (seed, task)
pair it runs four conditions with an otherwise identical setup:

* ``scratch_only``:  never touch memory, always train from scratch.
* ``clone_only``:    always CLONE from the best candidate (if one exists).
* ``reuse_only``:    always REUSE the best candidate directly (if one exists).
* ``automatic``:     the real score-driven SkillMemoryPlugin policy.

and reports, per task, which of scratch/clone/reuse achieved the lowest MSE
(the "oracle" choice) and whether ``automatic`` matched it. This is the
"Step B" experiment from the PR review: it directly measures whether the
compatibility score is a reliable predictor of which strategy helps, rather
than only reporting final accuracy/MSE.

Uses the same probe-based ``ProbeCompatibilityScorer`` as
``skill_memory_benchmark.py`` (see that module for the arithmetic task
definitions, which are duplicated here in miniature to keep this script
runnable on its own).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from avalanche.training import Naive
from avalanche.training.skill_memory import (
    ProbeCompatibilityScorer,
    SkillMemory,
    SkillMemoryPlugin,
    mean_baseline_mse,
)

from skill_memory_benchmark import (  # noqa: E402  (local example module)
    ArithmeticExperience,
    EPOCHS,
    EVAL_SEED_BASE,
    EVAL_SAMPLES,
    PROBE_SAMPLES,
    PROBE_SEED_BASE,
    TASKS,
    TRAIN_SEED_BASE,
    TRAIN_SAMPLES,
    evaluate,
    make_dataset,
    make_experience,
    make_model,
)

SEEDS = list(range(15))
CONDITIONS = ["scratch_only", "clone_only", "reuse_only", "automatic"]
FORCE_DECISION = {
    "scratch_only": SkillMemoryPlugin.SCRATCH,
    "clone_only": SkillMemoryPlugin.CLONE,
    "reuse_only": SkillMemoryPlugin.REUSE,
    "automatic": None,
}


def probe_batch(experience: ArithmeticExperience, seed: int):
    dataset = make_dataset(
        experience.operation,
        PROBE_SAMPLES,
        PROBE_SEED_BASE + seed * len(TASKS) + experience.current_experience,
    )
    return dataset.tensors


def run_condition(condition: str, seed: int) -> Dict:
    torch.manual_seed(seed)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    memory = SkillMemory(max_skills=8)

    compatibility = ProbeCompatibilityScorer(
        model_factory=make_model,
        loss_fn=nn.functional.mse_loss,
        probe_fn=lambda exp: probe_batch(exp, seed),
        reference_fn=lambda exp: mean_baseline_mse(probe_batch(exp, seed)[1]),
    )

    plugin = SkillMemoryPlugin(
        memory=memory,
        skill_name=lambda exp: exp.operation,
        skill_metadata=lambda exp: {
            "operation": exp.operation,
            "prerequisites": list(exp.prerequisites),
        },
        compatibility=compatibility,
        reuse_threshold=0.90,
        clone_threshold=0.30,
        force_decision=FORCE_DECISION[condition],
    )

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        train_mb_size=16,
        train_epochs=EPOCHS,
        eval_mb_size=16,
        eval_every=-1,
        plugins=[plugin],
    )

    experiences = [
        make_experience(0, "multiply", (), seed),
        make_experience(1, "add", (), seed),
        make_experience(2, "square", ("multiply",), seed),
        make_experience(3, "divide", (), seed),
    ]

    per_task: Dict[str, Dict[str, object]] = {}
    for experience in experiences:
        strategy.train(experience, eval_streams=[])
        mse = evaluate(strategy.model, experience.operation, seed)
        per_task[experience.operation] = {
            "mse": mse,
            "decision": plugin.last_decision,
            "selected_skill": plugin.last_selected_skill,
            "compatibility_score": plugin.last_compatibility_score,
        }

    return {"condition": condition, "seed": seed, "tasks": per_task}


def main() -> None:
    all_results: List[Dict] = []
    for seed in SEEDS:
        print(f"Running oracle comparison seed {seed}...")
        for condition in CONDITIONS:
            all_results.append(run_condition(condition, seed))

    per_seed_task_comparison = []
    match_count = 0
    comparable_count = 0

    for seed in SEEDS:
        by_condition = {r["condition"]: r for r in all_results if r["seed"] == seed}
        for task in TASKS:
            mse_by_strategy = {
                "scratch": by_condition["scratch_only"]["tasks"][task]["mse"],
                "clone": by_condition["clone_only"]["tasks"][task]["mse"],
                "reuse": by_condition["reuse_only"]["tasks"][task]["mse"],
            }
            oracle_strategy = min(mse_by_strategy, key=mse_by_strategy.get)

            automatic_info = by_condition["automatic"]["tasks"][task]
            automatic_decision = automatic_info["decision"]

            # The first experience always starts from an empty memory, so
            # "clone"/"reuse" oracle conditions cannot act any differently
            # from scratch there; skip it from the match statistics.
            has_candidate = automatic_info["selected_skill"] is not None or task == TASKS[0]
            is_first_task = task == TASKS[0]

            matched = automatic_decision == oracle_strategy
            if not is_first_task:
                comparable_count += 1
                match_count += int(matched)

            per_seed_task_comparison.append(
                {
                    "seed": seed,
                    "task": task,
                    "mse_by_forced_strategy": mse_by_strategy,
                    "oracle_strategy": oracle_strategy,
                    "automatic_decision": automatic_decision,
                    "automatic_compatibility_score": automatic_info["compatibility_score"],
                    "automatic_matches_oracle": matched,
                    "skipped_first_task": is_first_task,
                }
            )

    match_rate = match_count / comparable_count if comparable_count else float("nan")

    by_task_match_rate = {}
    for task in TASKS[1:]:
        rows = [r for r in per_seed_task_comparison if r["task"] == task]
        by_task_match_rate[task] = sum(r["automatic_matches_oracle"] for r in rows) / len(rows)

    result = {
        "benchmark": "skill_memory_arithmetic_policy_oracle_v1",
        "description": (
            "For each (seed, task), compares SCRATCH-only, CLONE-only, "
            "REUSE-only, and the automatic score-driven policy, and reports "
            "whether the automatic decision matched whichever forced "
            "strategy actually achieved the lowest MSE (the oracle choice). "
            "This directly tests the Issue #3 hypothesis -- that the "
            "compatibility score reliably predicts the beneficial strategy "
            "-- rather than only comparing Skill Memory to a plain baseline."
        ),
        "seeds": SEEDS,
        "policy": {"reuse_threshold": 0.90, "clone_threshold": 0.30},
        "overall_match_rate": match_rate,
        "match_rate_by_task": by_task_match_rate,
        "comparisons": per_seed_task_comparison,
    }

    output_path = Path("results/skill_memory_arithmetic_policy_oracle.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nOverall automatic-vs-oracle match rate: {match_rate:.2%}")
    print("By task:")
    print(json.dumps(by_task_match_rate, indent=2))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
