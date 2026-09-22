---
name: experimental-research
description: Design and execute empirical or computational research outside ordinary ML training, including physics, chemistry, biology, engineering, simulation, and quantitative social science. Use when a question requires competing hypotheses, measurement design, controlled experiments, numerical-model verification, uncertainty analysis, or iterative data collection; do not use for a purely literature-only review or a purely formal proof.
---

# Experimental Research

Produce a result that survives contact with measurement, alternative explanations, and the limits
of the apparatus or simulation. A successful outcome may support a model, falsify it, reveal a
measurement defect, or narrow the next experiment. It need not be a positive discovery.

## Non-negotiable distinctions

- **Observation is not mechanism.** A pattern can motivate a hypothesis but does not identify its
  cause without discriminating evidence.
- **Exploration is not confirmation.** Use exploratory data to generate models; confirm on new or
  held-out evidence with the decision rule fixed.
- **Precision is not accuracy.** Repeated measurements can agree while sharing calibration bias.
- **Code verification is not model validation.** Correctly solving the chosen equations does not
  show that those equations represent reality for the intended use.
- **Reproducibility is not replication.** Re-running the same data and code tests a different risk
  from collecting independent evidence.
- **A null result is not “no effect.”** State the interval and effects the design could or could not
  exclude.

Respect domain safety, ethics, sample handling, and institutional requirements. Do not execute wet-
lab, human-subject, hazardous, or high-consequence physical procedures without the required expert
oversight and authorization.

## 1. Write the research contract

Specify:

- phenomenon and scientific question;
- system boundary, population, or physical regime;
- intended use of the result;
- response variables and minimum scientifically meaningful effect;
- controllable factors, observed covariates, and likely confounders;
- resource, safety, temporal, and instrumentation constraints;
- what observations would support, weaken, or fail to distinguish the main hypotheses.

Replace vague goals such as “understand turbulence” with a falsifiable local question under a named
geometry, Reynolds-number range, observable, and accuracy requirement.

## 2. Build competing models

Represent at least the leading hypothesis and the strongest plausible alternative. Add a null or
artifact model when measurement error could explain the effect. For each model record:

- causal or mathematical structure;
- assumptions and nuisance parameters;
- predictions already explained;
- risky prediction that differs from competitors;
- regime where it should fail;
- evidence that would change confidence.

Do not generate alternatives merely for variety. They must explain the current observations through
different load-bearing mechanisms.

When the governing form itself is unknown, read
[references/model-discovery.md](references/model-discovery.md). Treat symbolic regression, sparse
system identification, dimensional analysis, and learned latent models as hypothesis generators.
Select among them using physical constraints, held-out regimes, and interventions—not fit alone.

## 3. Establish prior evidence and unresolved delta

Search current primary literature, data, methods, and standard reference results. Extract the exact
regime, apparatus or simulation, uncertainty, controls, and negative evidence. Identify the closest
experiment and the specific uncertainty the new work will reduce.

Treat reported values as measurements with conditions, not context-free constants. Reconcile unit,
definition, calibration, population, and boundary-condition differences before comparing them.

## 4. Define the measurement model

Before choosing sample size or analysis, define:

- the measurand or construct;
- how raw signals become the reported response;
- calibration and traceability chain;
- resolution, detection limit, missingness, saturation, and censoring;
- random and systematic uncertainty sources;
- spatial, temporal, batch, subject, or instrument dependence;
- validity evidence that the proxy represents the intended quantity.

Build an uncertainty budget and propagate it to the final comparison. Read
[references/simulation-and-measurement.md](references/simulation-and-measurement.md) for physical
measurement and computational-model requirements.

## 5. Design the most discriminating feasible study

Choose the observation or intervention that maximizes separation between competing predictions,
not merely the easiest measurement. Use controls, randomization, blocking, blinding, counterbalancing,
or matched sampling when the domain permits. Predefine exclusions, stopping, transformations,
primary outcomes, and the analysis decision rule for confirmation.

Read [references/study-design.md](references/study-design.md) when selecting observational,
interventional, factorial, sequential, or multiscale designs. Pilot to test the apparatus and
variance model; do not quietly turn pilot choices into confirmed findings on the same data.

## 6. For simulations, verify before validating

Separate:

1. **conceptual model** — selected physics and omissions;
2. **mathematical model** — equations, closures, initial and boundary conditions;
3. **numerical method** — discretization, solver, tolerances;
4. **implementation** — code;
5. **physical comparison** — validation data for the intended use.

Test units, conservation, manufactured or analytic solutions, convergence with resolution and
tolerance, and independent benchmarks. Quantify numerical and parameter uncertainty. Only then
compare with physical observations not used to calibrate the model.

## 7. Execute with provenance

Record sample, apparatus, operator, time, environment, calibration, protocol version, code revision,
parameters, exclusions, and raw outputs. Randomize execution order where drift could alias with a
factor. Monitor predefined quality controls without repeatedly changing the scientific hypothesis.

If the protocol changes, version it and mark which evidence is exploratory after the change.

## 8. Analyze against predictions

- visualize raw and residual data before aggregation;
- use an analysis matched to the experimental unit and dependence structure;
- report effect sizes, intervals, sensitivity to reasonable analysis choices, and uncertainty
  contributions;
- compare posterior or predictive implications of competing models where appropriate;
- inspect outliers as possible errors and possible new regimes—never delete them only because they
  hurt the result;
- distinguish model inadequacy, parameter error, numerical error, measurement error, and random
  variation.

Avoid interpreting a threshold crossing as a discontinuous change in truth. Avoid fitting and
evaluating a flexible model on the same information without accounting for selection.

## 9. Choose the next experiment by information gain

Update model confidence and the failure ledger. The next run should target the largest decision-
relevant uncertainty or the prediction on which live models disagree most. Do not repeat an
uninformative experiment only at larger scale.

When a probabilistic model is defensible, estimate expected information or decision gain across
candidate designs and include measurement cost and risk. Also reserve designs that can reveal that
all current models are wrong; an experiment optimized only within a misspecified model class can be
efficiently misleading.

Maintain [references/research-record.md](references/research-record.md) for multi-turn or multi-run
work. Preserve failed apparatus configurations and model discrepancies with their mechanisms.

## 10. Confirm, replicate, and scope

Before a strong claim:

- repeat the analysis from raw data and a clean computational environment;
- confirm the frozen prediction on new or untouched evidence;
- vary operators, instruments, sites, batches, initial conditions, or regimes relevant to the
  claim;
- compare with a simpler explanation and a calibrated null/artifact model;
- state the domain of validity and what remains untested.

## Required deliverable

Provide the research contract, competing-model table, measurement and uncertainty model, protocol
and controls, raw-data or simulation provenance, analysis linked to predictions, model-validation
status, failures and anomalies, calibrated conclusion, and the next highest-information test.
