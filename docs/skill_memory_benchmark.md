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
| 3 | division | reject incompatible skills, then acquire |

The compatibility function is deliberately explicit: it matches a stored operation only
when that operation appears in the current training experience's prerequisite list.
This isolates the memory/reuse mechanism; it does **not** claim that the compatibility
function is a learned task-similarity model.

## Evaluation protocol

Training and evaluation datasets use different deterministic seeds. Evaluation data are
created only after training and are never passed to the compatibility function.
Therefore the reuse decision cannot inspect evaluation targets.

For every experience, the benchmark evaluates all tasks seen so far. The recorded metrics are:

- final MSE for each task;
- MSE history after each experience;
- per-task forgetting, measured as the increase from that task's best observed MSE to its final MSE;
- Skill Memory decision and compatibility score for every experience;
- stored skill names.

The output is written to `results/skill_memory_benchmark.json`.

## Interpretation

The benchmark is useful only if it produces a measurable difference under the same training
budget. A successful run proves that the integration works and records the observed result;
it does **not** by itself establish a statistically significant research claim.

If the result is promising, the next step is a multi-seed benchmark with confidence intervals,
followed by broader task families in the research repository. If there is no measurable benefit,
that result should be recorded rather than forcing an upstream contribution.

## Running

From the repository root:

```bash
python examples/skill_memory_benchmark.py
```

The script is deterministic with respect to the configured seed and writes a machine-readable
JSON result suitable for later aggregation across seeds.
