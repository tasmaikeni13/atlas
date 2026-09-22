---
name: ml-research
description: >-
  Conduct empirical machine-learning research in a codebase: establish a reproducible baseline,
  generate a diverse hypothesis portfolio, implement minimal method changes, run controlled
  experiments, and report evidence-bounded claims. Use when asked to invent, test, or improve an ML
  optimizer, architecture, loss, training method, representation, or algorithm, or to carry an ML
  research problem through experiments.
---

# ML Research

Produce an executable, falsifiable ML result—not a list of impressive-sounding ideas. The default
unit of progress is a completed learning loop: a precise claim, a controlled implementation, an
observed result, and an updated belief.

## Operating rules

- Inspect the repository, data path, evaluator, and prior runs before proposing changes.
- Search current primary literature and the closest implementations. Memory is not a novelty check.
- Do not claim a breakthrough, state of the art, mechanism, or generalization beyond the evidence.
- Never invent metrics or report a run that did not complete. Keep failed and null results.
- Separate the optimization target from the scientific claim. A higher score can come from leakage,
  tuning budget, compute, or variance rather than the proposed mechanism.
- When code, data, and compute are available within scope, implement and run. Do not stop at a
  proposal unless the user asked only for one or a concrete resource barrier prevents execution.
- Keep the test set or final evaluation locked during search. Iterate on train/validation evidence.

## 1. Establish the research contract

Write a compact contract before changing code:

- research question and target failure;
- candidate claim type: performance, efficiency, robustness, understanding, or theory;
- primary and secondary metrics, including direction and acceptable tradeoffs;
- data, distribution, model class, and operating regime;
- strongest relevant baselines;
- compute, time, memory, and dependency limits;
- success threshold, kill threshold, and final confirmation test.

Prefer a claim that can lose. “Explore better architectures” is not ready; “reduce validation loss
under the same parameter and training-compute budget without worsening calibration” is.

## 2. Audit and reproduce the starting point

Before invention:

1. locate the actual training and evaluation entry points;
2. verify data splits, preprocessing, metric implementation, seed handling, and checkpoint selection;
3. run the cheapest valid baseline reproduction;
4. compare it with the documented result and explain any discrepancy;
5. create a fast evaluator or smoke test that catches broken candidates early.

Read [references/experiment-protocol.md](references/experiment-protocol.md) for the full audit and
comparison standard. A research branch cannot inherit credibility from a baseline that was never
made to run.

## 3. Ground the opportunity

Map the closest work by technical facets: representation, information flow, objective, update rule,
state, training budget, inference budget, data regime, and evaluation. Extract known failures and
negative results, not just headline scores.

State the unresolved delta in one sentence: “Existing methods do X under Y; none of the closest
work tests or achieves Z under constraint C.” Treat this as provisional until the final novelty
pass.

## 4. Build a hypothesis portfolio

Generate enough structurally distinct hypotheses to cover the plausible mechanism classes,
normally four to eight. Two candidates are distinct only if their causal change or load-bearing
assumption differs—not merely their name or hyperparameters.

For each hypothesis record:

- exact code or mathematical change;
- proposed mechanism;
- unique observable prediction;
- simplest alternative explanation;
- smallest discriminating experiment;
- expected resource cost;
- kill criterion;
- closest precedent.

Cluster near-duplicates and retain some high-uncertainty candidates so early noisy scores do not
collapse the search. Read [references/method-search.md](references/method-search.md) when generating
optimizers, architectures, objectives, or training algorithms.

## 5. Make the evaluator trustworthy before scaling search

Create tests for invariants, tensor shapes, data leakage, metric direction, and deterministic toy
cases. Ensure the evaluator returns a rich signal when possible: primary metric plus cost, stability,
and failure diagnostics. A single noisy scalar invites metric gaming.

Pin or record environment, code state, config, seed, data version, commands, and outputs. Use a
unique run identifier. Store raw results; tables and prose are derived artifacts.

## 6. Run a staged search loop

For each live branch:

1. **Implement the minimum causal delta.** Avoid unrelated cleanup or bundled tricks.
2. **Smoke test.** Verify execution and expected limiting behavior on a tiny case.
3. **Low-fidelity screen.** Use reduced data, steps, or model size only if this proxy preserves the
   ordering relevant to the claim.
4. **Analyze, do not merely score.** Compare predicted and observed signatures; inspect failures,
   learning curves, resource use, and per-slice behavior.
5. **Update the portfolio.** Promote, revise, combine, or kill branches based on evidence. Preserve
   lineage so failed mutations are not rediscovered.

Balance exploitation of strong branches with exploration of structurally different ones. A weak
candidate with a correct unique prediction may deserve more attention than a high score caused by
extra compute.

Maintain the artifacts in [references/research-loop.md](references/research-loop.md) for work that
spans multiple runs or turns.

## 7. Confirm a surviving result

Screening results are not research conclusions. On the finalists:

- compare against strong and simple baselines under matched resource and tuning budgets;
- run multiple independent seeds or resamples appropriate to the variability;
- report effect sizes and uncertainty, not only the best run or a p-value;
- ablate each claimed component and include a parameter/compute-matched control;
- test the regime where the proposed mechanism predicts failure or reversal;
- evaluate distribution shift or new tasks when the claim includes generalization;
- measure wall time, memory, parameters, training/inference compute, and tuning cost when relevant;
- repeat the final protocol from a clean command or environment.

Choose statistical treatment for the experimental unit and dependence structure. Do not blindly
apply a fixed seed count or test.

## 8. Audit novelty and convert evidence into claims

After the method is concrete, search its equation, update rule, component combination, and predicted
behavior. Compare the closest work facet by facet. Then construct a claim–evidence matrix:

| Claim | Direct evidence | Alternative explanation tested | Scope | Confidence |
|---|---|---|---|---|

Label mechanistic explanations as supported only when discriminating interventions or predictions
were tested. Otherwise call them hypotheses or interpretations. A benchmark improvement alone
supports a benchmark claim, not a universal statement about learning.

## Completion states

A valid completion may be:

- a positive result that survives confirmation;
- a negative result that rules out a meaningful hypothesis under stated conditions;
- a reproducible baseline plus a sharply localized blocker and the decisive next experiment;
- a formal or implementation bug in the original premise, demonstrated by evidence.

In every case deliver the reproducible commands or artifacts, results with uncertainty, killed and
live hypotheses, closest-work comparison, honest claim scope, and the next highest-information test.
