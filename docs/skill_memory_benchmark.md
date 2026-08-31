# Skill Memory benchmark

This benchmark is a controlled proof-of-benefit experiment for the Skill Memory integration.
It is intentionally small and should not be presented as a general continual-learning result.

## Conditions

Two otherwise identical Avalanche `Naive` runs are compared:

- **baseline**: normal sequential training with no Skill Memory plugin;
- **skill_memory**: the same model, optimizer, data, seeds, batch size, and epoch budget,
  with `SkillMemoryPlugin` enabled.

The model and optimizer are initialized from the same random seed for each condition.

## Task sequence

| Experience | Operation | Expected Skill Memory decision |
| --- | --- | --- |
| 0 | multiplication | acquire |
| 1 | addition | acquire |
| 2 | square | reuse `multiply` |
| 3 | division | no compatible prerequisite, then acquire |

The compatibility function is deliberately explicit: it matches a stored operation only
when that operation appears in the current training experience's prerequisite list.
Square declares `multiply` as its prerequisite. Division declares no prerequisites, so it
cannot reuse an existing skill in this controlled benchmark. This isolates the memory/reuse
mechanism; it does **not** claim that the compatibility function is a learned task-similarity model.

## Evaluation protocol

Training and evaluation datasets use different deterministic seeds. Evaluation data are
created only after training and are never passed to the compatibility function.
Therefore the reuse decision cannot inspect evaluation targets.

For every experience, the benchmark evaluates all tasks seen so far. The recorded metrics are:

- final MSE for each task;
- MSE history after each experience;
- per-task forgetting, measured as `max(0, final MSE - minimum MSE observed after any experience for that task)`;
- Skill Memory decision and compatibility score for every experience;
- stored skill names.

The 15-seed benchmark additionally reports mean, sample standard deviation, and an approximate
95% confidence interval for each condition. It also reports paired differences (`Skill Memory - baseline`)
for each seed, along with the number of seeds won by each condition. Pairing by seed is important because
the same seed defines the corresponding baseline and Skill Memory run.

## Multi-seed benchmark

The development benchmark uses fixed seeds `0..14`. It should be interpreted as a robustness check
for the controlled arithmetic result, not as evidence of general continual-learning superiority.

The most important comparison is the paired forgetting difference. Negative values mean Skill Memory
has less forgetting than baseline. Final MSE is reported separately because lower forgetting does not
necessarily imply lower final task error.

The benchmark output is written to `results/skill_memory_benchmark_multiseed.json`.

## Interpretation

A successful run proves that the integration works and records the observed result; it does **not**
by itself establish a statistically significant research claim.

The arithmetic experiment is designed to answer two separate questions:

1. Does Skill Memory reliably make the intended reuse decision across seeds?
2. Does that reuse mechanism change continual-learning behavior, especially forgetting?

A positive forgetting result should not be described as a general improvement until it is reproduced
on a standard Avalanche continual-learning benchmark with an appropriate baseline.

The next step after this controlled multi-seed analysis is a broader benchmark in a standard Avalanche
scenario.

## Running

From the repository root:

```bash
python examples/skill_memory_benchmark.py
```

The script is deterministic with respect to the configured seeds and writes a machine-readable JSON
result suitable for paired analysis across the baseline and Skill Memory conditions.
