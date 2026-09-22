# Experimental Research Record

Use this record for sustained research. Keep raw data immutable and link conclusions to exact
artifacts.

## Current state

```markdown
# Research state — [question] — updated [UTC date]

## Contract
Phenomenon and regime:
Intended use:
Primary response and meaningful effect:
Constraints and safety requirements:

## Competing models
- M1:
- M2:
- M0/artifact:

## Measurement status
Measurand and instrument:
Calibration version:
Dominant uncertainties:
Detection/resolution limits:

## Established observations
- [Observation | evidence IDs | uncertainty | scope]

## Discrepancies and failures
- [Name | mechanism | evidence | reopening condition]

## Next decisive tests
1.
2.
3.
```

## Competing-model card

```markdown
# M-[ID]
Structure / causal graph / equations:
Assumptions:
Parameters and identifiability:
Evidence already explained:
Distinctive prediction:
Null/reversal/boundary prediction:
Strongest evidence against:
Falsifier or downgrade criterion:
Current confidence:
```

## Measurement card

```markdown
# Q-[ID] [measurand]
Operational definition:
Raw signal and reduction pipeline:
Reference/calibration:
Random uncertainty components:
Systematic/correlated components:
Resolution, saturation, censoring, missingness:
Validity evidence:
Propagation method:
Current uncertainty budget:
```

## Experiment card

```markdown
# E-[ID]
Question and competing predictions:
Protocol version:
Experimental unit and assignment:
Factors, levels, blocks, and controls:
Primary response and analysis rule:
Exclusions and stopping rule:
Power/design analysis:
Apparatus/simulation/code revision:
Calibration and environment:
Raw artifact paths:
Deviations from protocol:
Result with uncertainty:
Diagnostics and residuals:
Model update:
Next decision:
```

## Evidence and claim ledger

```markdown
| ID | Claim/observation | Data or simulation artifact | Analysis | Uncertainty | Alternatives tested | Scope | Status |
|---|---|---|---|---|---|---|---|
```

Status values: exploratory, confirmatory, reproduced, independently replicated, contradicted,
inconclusive.

## Failure ledger

Classify failures as:

- protocol or handling;
- calibration or instrument;
- data integrity;
- numerical implementation;
- convergence/resolution;
- parameter or boundary-condition uncertainty;
- conceptual model inadequacy;
- analysis assumption;
- competing model not separated;
- safety/resource infeasibility.

Record mechanism and evidence. “Bad run” is not enough to prevent recurrence or reveal new science.

## Handoff

Before pausing, refresh the state and write the exact next setup, command, or observation. On
resumption, verify calibration/protocol/code versions before combining new evidence with old.
