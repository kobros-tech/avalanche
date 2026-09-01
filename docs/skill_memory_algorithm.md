# Skill Memory Algorithm

## Status

This document is the normative definition of the Skill Memory algorithm investigated in this branch. It takes precedence over prototype wording that treats REUSE and CLONE as equivalent model initialization operations.

**The algorithm does not prescribe fixed compatibility-score values or fixed score thresholds.** The compatibility score is evidence used to identify promising prior skills. The acquisition decision is made by comparing the expected outcomes of REUSE, CLONE, and SCRATCH, with SCRATCH as the fallback/reference. Score calibration and outcome estimation are experimental components and must be evaluated separately from the underlying skill-memory mechanism.

## 1. Objective

Skill Memory is a continual skill-acquisition mechanism. Its purpose is to let a learner acquire new skills sequentially while preserving previously acquired skills and exploiting useful prior skills when a later experience is compatible.

The central invariant is:

> **A learned skill is never modified by training for a later skill.**

A skill may be **reused** without modification, or **cloned** into a new reserved region and the clone may then be modified by training. These are different operations.

Skill Memory is not defined as replay, and it is not merely a checkpoint registry.

## 2. Terminology

### Skill

A learned capability represented by a model parameter/neuron region (a reserved skill slot) together with the metadata needed to identify and score it.

### Reserved skill slot

A region of model parameters/neuron capacity assigned to exactly one acquired skill. Once a slot has been committed to a learned skill, its parameters are protected from subsequent training.

Slots are allocated in deterministic order: slot 0, slot 1, slot 2, and so on. The algorithm therefore grows the collection of acquired skills through successive reserved regions rather than rewriting an earlier skill.

### REUSE

Use an existing skill directly for the current experience. The selected skill's reserved parameters remain frozen. No gradient update is applied to that skill and no replacement copy is created.

REUSE is a **use operation, not an acquisition operation**.

### CLONE

Copy the selected source skill from its reserved slot into the next unused reserved slot. The destination slot becomes the active skill for the new experience. Training is allowed to modify only the destination slot (and any explicitly designated shared parameters, if the model architecture declares them). The source slot remains unchanged.

CLONE is an **acquisition operation**.

### SCRATCH

Allocate the next unused reserved slot and initialize it from the normal fresh initialization. Train the new skill in that slot.

SCRATCH is an **acquisition operation** and the reference/fallback whenever neither REUSE nor CLONE is expected to provide an advantage.

## 3. Sequential algorithm

For experiences `E_1, E_2, ..., E_T`:

1. Start with an empty Skill Memory and the first unused reserved slot.
2. Receive the next experience through the normal Avalanche experience lifecycle.
3. For each stored skill, compute a compatibility score using only information permitted for training-time decision making. Never use held-out final evaluation/test labels.
4. Use the scores to identify the most promising prior skill candidates. If no stored skill is sufficiently promising according to the predefined candidate-selection procedure, select SCRATCH directly.
5. For each candidate retained for consideration, estimate the outcome of the available actions:
   - **REUSE:** use the candidate unchanged and estimate whether it already satisfies the target criterion;
   - **CLONE:** copy the candidate into a new slot and estimate the learning outcome after adaptation under the same acquisition budget;
   - **SCRATCH:** use a fresh slot and estimate the learning outcome under the same acquisition budget.
6. Select the **best expected action** among REUSE, CLONE, and SCRATCH. The comparison must use the same predefined objective and budget for all alternatives.
7. If **REUSE** is selected, activate the selected existing skill and keep its reserved parameters frozen. Do not train it and do not create a new skill record.
8. If **CLONE** is selected, copy the selected source slot into the next unused slot, activate the destination slot, and train only the destination. The source remains protected.
9. If **SCRATCH** is selected, initialize the next unused slot normally and train it.
10. After a successful CLONE or SCRATCH acquisition, register the destination slot as a new skill in memory.
11. Continue to the next experience.

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
       existing slot    copy source slot   fresh new slot
       stays frozen     -> next slot       -> next slot
             |                 |                 |
             |                 v                 v
             |              train clone      train new skill
             |                 |                 |
             |                 +--------+--------+
             |                          |
             |                          v
             |                   register new slot
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
Skill A (slot 0)
       |
       +---- REUSE ----> use slot 0, do not update slot 0
       |
       +---- CLONE ----> copy slot 0 -> slot 1 -> train slot 1
```

After CLONE:

```text
slot 0 = original Skill A       [protected]
slot 1 = adapted Skill A/B      [trainable]
```

The adapted skill is a distinct acquired skill and must have its own memory record and slot identity.

## 6. Why reserved slots are part of the algorithm

A whole-model checkpoint copy is not sufficient to express the intended algorithm. A whole-model copy can provide an initialization, but it does not establish which neurons belong to which skill or protect those neurons during later training.

The implementation therefore needs a model/strategy adapter that knows how a model is partitioned into reserved skill regions. The generic Skill Memory layer stores and indexes skills; the adapter owns the actual parameter/neuron routing and protection.

The first implementation may use a simple explicit slot architecture for the arithmetic proof of concept. It must not silently pretend that arbitrary unpartitioned Avalanche models already provide isolated neuron slots.

## 7. Skill-memory invariants

1. **Immutability:** a registered source skill never changes as a side effect of another skill's training.
2. **Unique ownership:** one reserved slot represents one acquired skill version.
3. **Monotonic allocation:** once a slot has been assigned, the next newly acquired skill uses the next unused slot; an older slot is not overwritten.
4. **Clone independence:** modifying a clone must not modify its source.
5. **Reuse isolation:** REUSE must not create a new trainable version of the selected skill.
6. **Explicit acquisition:** only CLONE and SCRATCH create new skills.
7. **No test leakage:** compatibility, candidate selection, and action selection must not inspect final evaluation/test labels.
8. **Sequential processing:** decisions are made as experiences arrive, not after seeing the complete future stream.
9. **Deterministic slot identity:** slot allocation is reproducible for a fixed experience sequence.
10. **Capacity accounting:** every newly acquired skill consumes a distinct reserved slot; experiments must report the resulting capacity/parameter cost.
11. **Scratch fallback:** if no REUSE or CLONE candidate is expected to beat SCRATCH, SCRATCH is selected.
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

The active optimizer must update only parameters belonging to the active trainable skill region, plus any parameters explicitly declared shared by the model adapter.

For REUSE, the selected skill region is frozen. The implementation must not rely solely on loading a copied state dictionary and then continuing ordinary training, because ordinary training would turn that operation into adaptation and therefore violate REUSE semantics.

For CLONE, the source region is frozen and the destination region is trainable. The destination starts as an exact copy of the source region before training.

For SCRATCH, the destination region receives fresh initialization.

## 10. Registration semantics

A new memory record is created only after CLONE or SCRATCH completes acquisition. Its metadata should include at least:

- skill name/identifier;
- slot identifier;
- acquisition mode (`clone` or `scratch`);
- source skill identifier, if cloned;
- compatibility score used for candidate selection;
- estimated/selected action;
- experience/task metadata;
- creation order.

A REUSE event may be logged for evaluation, but it must not replace the stored source state or create an adapted record.

## 11. Scientific success criteria and evaluation

The algorithm should not be judged by whether a compatibility score exceeds an arbitrary number. The score is a mechanism for finding promising prior skills. The scientific question is whether the resulting acquisition decision improves learning relative to SCRATCH while preserving previously learned skills.

The primary scientific target is:

> **Improve target-task learning relative to a scratch baseline when prior skills are useful, while preserving previously learned skills, and fall back to scratch when prior skills do not provide an advantage.**

### 11.1 Scratch baseline

For every target experience, train the target from a fresh reserved slot using the same architecture, capacity budget, training budget, optimizer, data, and comparable random seeds.

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

The strongest mechanical test is parameter/state isolation: a source slot must remain bitwise unchanged when its clone is trained, subject only to explicitly documented shared parameters.

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
- parameter/capacity overhead;
- runtime overhead.

The candidate-selection, score-calibration, and action-selection procedures must be fixed before final test evaluation.

## 12. Experimental hierarchy

Experiments should be performed in the following order so that policy errors are not confused with mechanism errors.

### Level 1 — Mechanism validity

Prove with unit tests that:

```text
REUSE   -> source unchanged
CLONE   -> source unchanged + clone changes
SCRATCH -> new slot independent of all previous slots
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
- compatibility lookup;
- candidate selection;
- action evaluation/policy;
- slot allocation/state isolation;
- integration hooks for activating, freezing, and training reserved regions.

The implementation should remain small and model-agnostic at the registry/policy level. Model-specific neuron partitioning belongs behind an explicit adapter rather than being guessed by the generic registry.

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
        destination = slots.next_free()
        slots.initialize_fresh(destination)
        slots.activate_trainable(destination)

    elif decision.kind == CLONE:
        destination = slots.next_free()
        slots.copy(decision.source.slot, destination)
        slots.freeze(decision.source.slot)
        slots.activate_trainable(destination)

    elif decision.kind == REUSE:
        destination = decision.source.slot
        slots.freeze(destination)
        slots.activate(destination)

    train_current_experience_only_if(decision.kind != REUSE)

    if decision.kind in (CLONE, SCRATCH):
        memory.register(
            slot=destination,
            source=decision.source if decision.kind == CLONE else None,
            metadata=experience.metadata,
        )
```

The exact mechanism used to estimate an action's outcome may differ between experiments. The comparison must remain fair: REUSE, CLONE, and SCRATCH must be evaluated under the same predefined target criterion and acquisition budget, and final test information must not be used to choose the action.

## 15. Definition of done

An implementation is faithful to this algorithm only when all of the following are true:

- [ ] Previously acquired skill parameters are protected from later skill training.
- [ ] REUSE does not update the reused skill.
- [ ] CLONE creates a physically independent destination parameter region.
- [ ] Training a clone changes the clone and not its source.
- [ ] New skills consume reserved slots monotonically.
- [ ] SCRATCH creates a new slot rather than overwriting an old skill.
- [ ] Only CLONE and SCRATCH create new learned-skill records.
- [ ] The compatibility score has a documented interpretation.
- [ ] The score is used for candidate selection rather than as an arbitrary universal decision threshold.
- [ ] The action-selection policy explicitly compares viable transfer actions against SCRATCH.
- [ ] REUSE is selected only when the existing skill is sufficient without adaptation.
- [ ] CLONE is selected only when adapting the copied skill is expected to beat SCRATCH.
- [ ] SCRATCH is selected whenever neither REUSE nor CLONE is expected to beat SCRATCH.
- [ ] Score/policy calibration is performed without final test/evaluation labels.
- [ ] No arbitrary score thresholds are treated as universal algorithm constants.
- [ ] The arithmetic proof of concept demonstrates source preservation and clone adaptation.
- [ ] Oracle transfer demonstrates measurable improvement over scratch where transfer is expected.
- [ ] The automatic policy is evaluated against scratch across multiple seeds.
- [ ] Previous skills are re-evaluated after subsequent acquisitions.
- [ ] Capacity and runtime costs are reported.
- [ ] Avalanche's normal sequential training lifecycle remains the integration mechanism.
