# ML Research Loop Artifacts

Use these records for multi-run or multi-turn work. Keep the current state compact; raw logs remain
separate and immutable.

## Research state

```markdown
# Research state — [question] — updated [UTC date]

## Contract
Target claim:
Metrics and tradeoffs:
Regime and constraints:
Success / kill thresholds:
Locked final test:

## Baseline
Command and run ID:
Result and uncertainty:
Expected result:
Discrepancy status:

## Current evidence
- [Observation, run IDs, confidence]

## Live hypotheses
- [H-ID and next discriminating experiment]

## Killed hypotheses
- [H-ID, failure mechanism, evidence, reopening condition]

## Next actions
1.
2.
3.
```

## Hypothesis card

```markdown
# H-[ID] [name]
Parent / inspiration:
Exact causal delta:
Mechanism hypothesis:
Unique prediction:
Simplest alternative explanation:
Closest precedent:
Smallest discriminating experiment:
Required controls:
Estimated cost:
Success threshold:
Kill criterion:
Status: proposed | implemented | screening | confirming | killed | supported
```

## Experiment record

```markdown
# E-[ID]
Hypothesis:
Question this run answers:
Code revision / diff:
Command and config:
Environment, data version, hardware:
Seed/resample and run ID:
Changed variable:
Held constant:
Predeclared expected observations:
Raw artifact paths:
Result, uncertainty, resource use:
Diagnostics and failures:
Interpretation:
Alternative explanation status:
Decision: promote | revise | combine | kill | rerun
Next experiment:
```

## Branch portfolio

```markdown
| H-ID | Causal delta | Load-bearing assumption | Prediction | Latest evidence | Cost | Status | Next test |
|---|---|---|---|---|---|---|---|
```

Cluster candidates sharing the same causal delta. A hyperparameter variation remains one branch
unless it changes the hypothesized mechanism.

## Failure taxonomy

Classify a failed branch before closing it:

- implementation/correctness;
- evaluator or metric defect;
- baseline discrepancy;
- proxy fidelity failure;
- optimization failure;
- mechanism prediction falsified;
- positive metric but confounded;
- effect too small relative to variance;
- resource infeasibility;
- novelty overlap;
- claim no longer valuable.

Record the evidence and reopening condition. Do not use “didn't work.”

## Claim–evidence matrix

```markdown
| C-ID | Claim | Direct experiment/run | Controls and alternatives | Tested scope | Effect/uncertainty | Confidence | Allowed wording |
|---|---|---|---|---|---|---|---|
```

Examples of calibrated wording:

- “improved mean validation accuracy on datasets X and Y under matched training compute”;
- “is consistent with the proposed variance-reduction mechanism, but the intervention was not
  decisive”;
- “did not improve within the tested learning-rate and batch-size regime”;
- “no close implementation was found in the sources searched through [date].”

Avoid “better,” “robust,” “efficient,” “general,” or “novel” without the column that supplies its
comparison, dimension, scope, or search basis.

## Handoff rule

Before pausing, refresh the state, link every conclusion to run IDs or sources, and write the exact
next command or decision. On resumption, reproduce the baseline/evaluator assumptions before adding
new branches.

Benchmarks of autonomous research agents show why this discipline matters: agents often struggle
with long-horizon experimentation and can produce invalid results. See
[MLAgentBench](https://openreview.net/pdf?id=1Fs1LvjYQW),
[ScienceAgentBench](https://arxiv.org/abs/2410.05080), and
[MLR-Bench](https://arxiv.org/abs/2505.19955).
