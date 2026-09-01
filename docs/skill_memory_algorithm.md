# Skill Memory Algorithm

## Status

This document is the normative definition of the Skill Memory algorithm investigated in this branch.

The algorithm is **model-agnostic at the registry level**. A learned skill is represented by an independently stored model state and its metadata. The generic algorithm does **not** require a single neural network to be partitioned into neuron regions or reserved parameter blocks.

The compatibility score is evidence used to identify promising prior skills. The acquisition decision is made by comparing the expected outcomes of REUSE, CLONE, and SCRATCH, with SCRATCH as the fallback/reference. Score calibration and outcome estimation are experimental components and must be evaluated separately from the underlying skill-memory mechanism.

## 1. Objective

Skill Memory is a continual skill-acquisition mechanism. Its purpose is to let a learner acquire new skills sequentially while preserving previously acquired skills and exploiting useful prior skills when a later experience is compatible.

The central invariant is:

> **A learned skill is never modified by training for a later skill.**

A skill may be **reused** without modification, or **cloned** into a new independent skill instance that may then be modified by training.

Skill Memory is not defined as replay, and it is not merely a checkpoint registry: the stored states are the persistent representations of acquired skills used by the acquisition policy.

## 2. Terminology

### Skill instance

A learned capability represented by an independently stored model state together with the metadata needed to identify, evaluate, and trace it.

A skill instance is independent of other stored skill instances. Training a newly acquired skill must not modify the stored state of its source skill or any other previously acquired skill.

The generic Skill Memory layer does not assume how many parameters a model has or how those parameters are internally organized.

### Skill capacity

`max_skills` is the maximum number of independently stored skill instances that the memory can contain.

Capacity therefore counts **stored skill instances**, not neurons or parameter regions inside one shared network.

For example:

```text
SkillMemory(max_skills=8)

skill_0 = independent model state
skill_1 = independent model state
skill_2 = independent model state
...
skill_7 = independent model state
```

The size of an individual skill is determined by the model architecture used by the experiment. The generic registry does not prescribe a fixed parameter count or slot width.

A model-specific implementation may choose to partition a shared model into parameter regions, but that is an optional architecture/adapter decision and is **not part of the generic Skill Memory contract**.

### REUSE

Use an existing skill directly for the current experience.

The selected skill's stored state remains unchanged. No gradient update is applied to it and no new acquired skill record is created.

REUSE is a **use operation, not an acquisition operation**.

### CLONE

Create an independent copy of the selected source skill's model state. The copied state becomes the starting point for the new experience and may then be modified by training.

The source skill remains unchanged. The clone receives its own independent optimizer/training state before adaptation.

After successful acquisition, the trained copy is stored as a new skill instance with its own identity and lineage metadata.

CLONE is an **acquisition operation**.

### SCRATCH

Create a new independent model state using the normal fresh initialization. Train it on the new experience.

After successful acquisition, store the resulting state as a new skill instance.

SCRATCH is an **acquisition operation** and the reference/fallback whenever neither REUSE nor CLONE is expected to provide an advantage.

## 3. Sequential algorithm

For experiences `E_1, E_2, ..., E_T`:

1. Start with an empty Skill Memory.
2. Receive the next experience through the normal Avalanche experience lifecycle.
3. For each stored skill, compute a compatibility score using only information permitted for training-time decision making. Never use held-out final evaluation/test labels.
4. Use the scores to identify the most promising prior skill candidates. If no stored skill is sufficiently promising according to the predefined candidate-selection procedure, SCRATCH may be selected directly.
5. For each candidate retained for consideration, estimate the outcome of the available actions:
   - **REUSE:** use the candidate unchanged and estimate whether it already satisfies the target criterion;
   - **CLONE:** copy the candidate into a new independent model state and estimate the learning outcome after adaptation under the same acquisition budget;
   - **SCRATCH:** use a freshly initialized independent model state and estimate the learning outcome under the same acquisition budget.
6. Select the **best expected action** among REUSE, CLONE, and SCRATCH. The comparison must use the same predefined objective and budget for all alternatives.
7. If **REUSE** is selected, activate/use the selected existing skill without training and without creating a new skill instance.
8. If **CLONE** is selected, create an independent copy of the selected source state, reset the clone's optimizer/training state, train the copy, and store it as a new skill instance. The source remains protected.
9. If **SCRATCH** is selected, initialize a fresh independent model state, train it, and store it as a new skill instance.
10. Continue to the next experience.

### Core decision rule

The defining policy is **best action versus SCRATCH**, not fixed score thresholds:

> **Choose REUSE or CLONE only when the available evidence indicates that it is better than the SCRATCH alternative under the predefined objective. Otherwise choose SCRATCH.**

Among multiple viable prior skills, select the action with the best expected outcome. A high compatibility score alone is not sufficient to justify REUSE or CLONE.

For example:

```text
candidate A: high compatibility
candidate B: medium compatibility
scratch:     strong expected result

If the expected outcomes are:
    REUSE(A)   = 82%
    CLONE(A)   = 96%
    CLONE(B)   = 91%
    SCRATCH    = 90%

choose CLONE(A).

If instead:
    REUSE(A)   = 70%
    CLONE(A)   = 83%
    CLONE(B)   = 86%
    SCRATCH    = 92%

choose SCRATCH.
```

The numerical values above are illustrative only; they are not algorithm thresholds or required experimental results.

### Acquisition policy and calibration

The compatibility score answers:

> **How promising is this stored skill as a candidate for the new experience?**

The action evaluation answers:

> **Will REUSE or CLONE actually outperform learning from SCRATCH?**

These questions must not be conflated.

A score threshold may be used only as an experimental candidate-filtering heuristic. It is not the final decision rule and no universal values such as `0.90` or `0.30` are prescribed.

The action-selection procedure must be calibrated using data independent of the final evaluation set. A calibration experiment should estimate REUSE, CLONE, and SCRATCH outcomes under matched budgets. The resulting candidate-selection and action-selection procedure must be fixed before final test evaluation.

## 4. State-transition model

```text
                         new experience
                               |
                               v
                       score stored skills
                               |
                               v
                    promising candidates
                               |
                               v
                estimate REUSE / CLONE / SCRATCH
                               |
                               v
                       compare outcomes
                               |
             +-----------------+-----------------+
             |                 |                 |
       REUSE is best     CLONE is best      SCRATCH is best
             |                 |                 |
             v                 v                 v
       use existing       copy model state   fresh model state
       skill unchanged    -> independent     -> independent
             |               copy                 |
             |                 |                 |
             |                 v                 v
             |            train clone        train new skill
             |                 |                 |
             |                 +--------+--------+
             |                          |
             |                          v
             |                   register new skill
             |                          |
             +--------------------------+
```

The defining fallback is:

```text
if no reuse/clone action is expected to beat SCRATCH:
    choose SCRATCH
```

## 5. The critical REUSE/CLONE distinction

The following behavior is **forbidden**:

```text
Skill A -> load Skill A -> train new task -> Skill A changes
```

That is not REUSE. It destroys the immutability guarantee.

The required behavior is:

```text
Stored Skill A
       |
       +---- REUSE ----> use A, do not update A
       |
       +---- CLONE ----> independent copy of A -> train copy
```

After CLONE:

```text
Skill A = original model state       [protected]
Skill B = adapted copy of Skill A   [trainable during acquisition]
```

The adapted skill is a distinct acquired skill and must have its own memory record and identity.

## 6. Why independent skill instances are the generic representation

The current `skill-memory` design uses a separate model per skill. Cloning is a deep copy of the complete model state, and the clone can then be trained independently while the source remains unchanged.

This gives the generic mechanism a simple capacity model:

```text
capacity = maximum number of independent stored skill models
```

It does not require the generic registry to know which neurons or parameter ranges belong to a skill.

A whole-model state copy is therefore sufficient **when the underlying model represents one skill per independently stored model state**, which is the reference representation for this prototype.

If a future Avalanche integration wants to pack multiple skills into one shared model, it must provide an explicit model/strategy adapter defining parameter ownership, isolation, activation, and optimizer routing. That is a separate extension and must not be assumed by the generic registry.

## 7. Skill-memory invariants

1. **Immutability:** a registered source skill never changes as a side effect of another skill's training.
2. **Independent representation:** each stored acquired skill has its own model state.
3. **Clone independence:** modifying a clone must not modify its source.
4. **Reuse isolation:** REUSE does not create a new trainable version of the selected skill.
5. **Explicit acquisition:** only CLONE and SCRATCH create new skills.
6. **Capacity accounting:** every newly acquired skill consumes one entry of the configured skill capacity.
7. **Monotonic acquisition:** a successful acquisition adds a new skill record rather than overwriting an existing skill.
8. **No test leakage:** compatibility, candidate selection, and action selection must not inspect final test/evaluation labels.
9. **Sequential processing:** decisions are made as experiences arrive, not after seeing the complete future stream.
10. **Clone optimizer isolation:** a clone begins adaptation with independent optimizer/training state rather than inheriting the source skill's optimizer history.
11. **Scratch fallback:** if no REUSE or CLONE action is expected to beat SCRATCH, SCRATCH is selected.
12. **Matched comparison:** action estimates use the same predefined target criterion and acquisition budget when comparing REUSE, CLONE, and SCRATCH.

## 8. Compatibility score

Compatibility is a candidate-selection signal, not the final definition of success. The scorer should estimate how promising a stored skill is for the new experience using independent training/calibration information.

A continuous score is useful for experiments, but the algorithm does not require a particular numeric scale. The important requirement is that the score have a defined interpretation and be computed without final test/evaluation labels.

A binary prerequisite lookup can be used as an oracle/control, but it is not itself evidence that an automatic compatibility scorer works.

If a normalized error-based score is used, one possible definition is:

```text
score = clamp(1 - L_skill / L_ref, 0, 1)
```

This is an example, not a mandated formula. The exact scoring function must be reported with the experiment.

The score may be used to rank or filter candidate skills. It must not be interpreted as a universal probability that REUSE or CLONE will beat SCRATCH unless that interpretation has been independently calibrated.

## 9. Training semantics

The generic registry stores model states; the active Avalanche strategy is responsible for training the currently selected model.

For REUSE, the selected stored state is loaded/activated for inference or use, but no training update is applied to that skill. REUSE must not silently turn into adaptation.

For CLONE, the source state is copied into an independent destination model state. The destination starts as an exact copy of the source model parameters, while optimizer/training state is initialized independently. Training then modifies only the destination acquisition state.

For SCRATCH, the destination receives fresh model initialization and independent optimizer/training state.

For model architectures that intentionally share parameters across skills, an explicit adapter must define which parameters are shared and how those parameters are protected. Such sharing is outside the generic registry contract.

## 10. Registration semantics

A new memory record is created only after CLONE or SCRATCH completes acquisition. Its metadata should include at least:

- skill name/identifier;
- acquisition mode (`clone` or `scratch`);
- source skill identifier, if cloned;
- compatibility score used for candidate selection;
- estimated/selected action;
- experience/task metadata;
- creation order.

A REUSE event may be logged for evaluation, but it must not replace the stored source state or create an adapted skill record.

## 11. Scientific success criteria and evaluation

The algorithm should not be judged by whether a compatibility score exceeds an arbitrary number. The score is a mechanism for finding promising prior skills. The scientific question is whether the resulting acquisition decision improves learning relative to SCRATCH while preserving previously learned skills.

### 11.1 Scratch baseline

For every target experience, train the target from a fresh model state using the same architecture, training budget, optimizer configuration, data, and comparable random seeds.

SCRATCH is not merely a baseline reported at the end. It is the reference action against which REUSE and CLONE are judged for the current task.

### 11.2 Transfer benefit

For a target with a genuinely useful prior skill, CLONE is successful when it provides measurable benefit over SCRATCH. Benefit should be measured primarily by:

- lower final target error under a fixed training budget; and/or
- fewer training steps/epochs to reach the same target criterion.

A result should be reported across multiple seeds with uncertainty, not from a single run.

### 11.3 REUSE validity and benefit

REUSE is successful only when using the existing skill unchanged is sufficient for the new task. Evidence should include:

- target performance is adequate without updating the skill; and
- the reused skill remains unchanged.

If adaptation is required, REUSE is not a valid success case; CLONE should be considered because the new task needs its own trainable copy.

### 11.4 Preservation

After acquiring each new skill, evaluate previously acquired skills again. Previously learned skills should not materially degrade relative to their performance immediately after acquisition.

The strongest mechanical test is state isolation: a source skill's stored model state must remain unchanged when its clone is trained.

### 11.5 Automatic-policy success

The automatic Skill Memory policy is successful when, across multiple seeds and target tasks, it provides statistically supported improvement over the scratch baseline on transfer-appropriate tasks **without causing unacceptable degradation on non-transfer tasks or previously learned skills**.

For each target, the decision should be compared with the scratch reference using the same predefined objective. The policy should not receive credit merely for choosing a high-scoring skill; it receives credit when the selected action produces a better acquisition outcome than SCRATCH.

Report:

- mean and dispersion/confidence intervals across seeds;
- final target performance;
- training steps to target performance;
- decision distribution (REUSE/CLONE/SCRATCH);
- selected source skills;
- compatibility scores;
- preservation of prior skills;
- number of stored skill instances / capacity usage;
- parameter and memory overhead;
- runtime overhead.

The candidate-selection, score-calibration, and action-selection procedures must be fixed before final test evaluation.

## 12. Experimental hierarchy

Experiments should be performed in the following order so that policy errors are not confused with mechanism errors.

### Level 1 — Mechanism validity

Prove with unit tests that:

```text
REUSE   -> source unchanged
CLONE   -> source unchanged + clone changes
SCRATCH -> new independent skill state
```

### Level 2 — Oracle transfer

Provide the correct acquisition decision externally. Demonstrate that when CLONE is appropriate, a cloned skill can learn the target faster and/or better than SCRATCH.

Also demonstrate that when no useful prior skill exists, SCRATCH is at least as good as the transfer alternatives under the matched budget.

This isolates the value of the cloning mechanism from the compatibility scorer.

### Level 3 — Automatic policy

Use the compatibility score to identify promising candidates, then use the calibrated action-selection procedure to choose among REUSE, CLONE, and SCRATCH. The policy must be fixed before final testing.

Evaluate the automatic policy against SCRATCH and appropriate control policies on unseen target tasks.

This isolates the quality of the decision mechanism.

## 13. Avalanche integration boundary

Avalanche should provide the normal sequential experience lifecycle, optimizer, model, datasets, evaluation, and plugin mechanism.

Skill Memory should provide:

- skill registry;
- independent model-state storage;
- compatibility lookup;
- candidate selection;
- action evaluation/policy;
- acquisition bookkeeping and lineage;
- integration hooks for activating and training the selected skill instance.

The implementation should remain small and model-agnostic at the registry/policy level. Model-specific parameter partitioning belongs behind an explicit adapter rather than being assumed by the generic registry.

## 14. Minimal reference pseudocode

```python
for experience in stream:
    scores = memory.score_all(experience)
    candidates = policy.select_candidates(scores)

    actions = []
    for candidate in candidates:
        actions.append(estimate_reuse(candidate, experience))
        actions.append(estimate_clone(candidate, experience))

    actions.append(estimate_scratch(experience))
    decision = policy.choose_best_action(actions)

    if decision.kind == SCRATCH:
        destination = model_factory()
        reset_optimizer(destination)
        train(destination, experience)
        memory.register(
            destination,
            source=None,
            acquisition_mode="scratch",
        )

    elif decision.kind == CLONE:
        destination = deepcopy_model(decision.source)
        reset_optimizer(destination)
        train(destination, experience)
        memory.register(
            destination,
            source=decision.source,
            acquisition_mode="clone",
        )

    elif decision.kind == REUSE:
        activate(decision.source)
        # No training and no new skill record.
```

The pseudocode describes the generic semantics. An Avalanche implementation may use the strategy's existing model object and optimizer rather than constructing separate Python model objects, provided that it preserves the same observable invariants: independent skill states, source immutability, independent clone optimizer state, and no training during REUSE.

## 15. Implementation checklist

Before treating the mechanism as complete, verify:

- [ ] A skill is represented by an independently stored model state.
- [ ] `max_skills` counts stored skill instances, not neurons.
- [ ] REUSE never trains the selected stored skill.
- [ ] REUSE does not create a new acquired skill record.
- [ ] CLONE starts from an exact copy of the source model state.
- [ ] CLONE uses independent optimizer/training state.
- [ ] Training a clone cannot modify its source skill.
- [ ] SCRATCH starts from fresh model initialization.
- [ ] Only successful CLONE and SCRATCH acquisitions add new skills.
- [ ] Skill lineage records the source for cloned skills.
- [ ] Compatibility scoring does not use final evaluation/test labels.
- [ ] Action selection compares REUSE/CLONE with SCRATCH under matched criteria and budgets.
- [ ] Previous skills are re-evaluated after subsequent acquisitions.
- [ ] Capacity, parameter/memory cost, and runtime overhead are reported.
- [ ] Avalanche's normal sequential training lifecycle remains the integration mechanism.
