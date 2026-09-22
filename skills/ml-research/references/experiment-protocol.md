# Controlled ML Experiment Protocol

Use this protocol to distinguish a method effect from implementation error, tuning advantage,
variance, data leakage, or resource mismatch.

## 1. Repository and environment audit

Identify and record:

- training, validation, test, and export entry points;
- configuration precedence and default values;
- package, compiler, accelerator, and driver versions;
- dataset source, version, checksum when feasible, preprocessing, and split generation;
- random-number generators and where seeds enter;
- checkpoint selection, early stopping, and metric aggregation;
- existing tests, experiment trackers, cached artifacts, and undocumented prior runs.

Run a tiny end-to-end job before editing. Verify that the metric moves in the documented direction
and that evaluation reads the intended checkpoint and split.

## 2. Baseline reproduction gate

A baseline is usable when:

- it completes from a recorded command in the current environment;
- its result is within a plausible range of the documented value or the discrepancy is explained;
- preprocessing, augmentation, training length, and checkpoint selection match the intended setup;
- raw outputs and logs are retained;
- at least one trivial control behaves as expected.

Examples of trivial controls: shuffled labels collapse performance, a zero learning rate leaves
weights fixed, disabling augmentation changes the expected input path, or a known overfit subset can
be memorized.

## 3. Protect the evaluation boundary

- Use training data to fit parameters.
- Use validation data to select ideas, hyperparameters, checkpoints, and stopping rules.
- Access the final test only after the protocol is frozen, or explicitly treat it as validation and
  obtain another final evaluation.
- Fit preprocessing statistics, vocabulary, normalization, feature selection, and augmentation
  policies without test information.
- Check duplicates, temporal leakage, group leakage, and contamination from pretrained models.

Repeatedly inspecting a public leaderboard is test-set adaptation even if the code never reads test
labels.

## 4. Define the experimental unit

The unit may be an independent training run, dataset, task, subject, environment, time period, or
physical sample. Identify dependencies before choosing a statistical summary. Multiple checkpoints
from one run are not independent seeds; multiple folds sharing most data are not independent
datasets.

Report:

- individual observations when the count is small;
- effect size and an interval estimate appropriate to the unit;
- the aggregation rule across tasks or datasets;
- missing or failed runs and their treatment;
- the selection rule used before final evaluation.

Choose seeds or resamples using expected variability, minimum meaningful effect, and compute—not a
ritual fixed count. Avoid highlighting the best seed.

## 5. Make comparisons fair

Match the resources relevant to the claim:

- training examples, updates, tokens, environment interactions, or wall-clock budget;
- parameters, active parameters, memory, and inference compute;
- pretrained data and external retrieval;
- hyperparameter search space, trial count, and human tuning attention;
- augmentation, optimizer, schedule, and early-stopping access when not part of the proposed change.

Include both a strong closest baseline and a simple baseline. If the new method receives additional
compute or information, either give the baseline the same resource or make the gain–cost tradeoff
the claim.

## 6. Use staged fidelity carefully

Low-fidelity screens can reduce cost using fewer steps, smaller models, subsets, lower resolution,
or cheaper simulations. Validate that the proxy ranks several known methods similarly to the full
setting. If it does not, use it only for bug detection, not branch selection.

Recommended gates:

1. syntax and unit tests;
2. deterministic toy problem;
3. overfit a tiny real subset;
4. short validation screen;
5. medium confirmation;
6. frozen full protocol.

Terminate candidates early for correctness violations or predeclared kill criteria, not simply
because an early noisy score is low.

## 7. Attribute the gain

For each claimed component, include:

- removal ablation;
- replacement with a simpler alternative;
- parameter- and compute-matched control;
- tuning-only control when a novel architecture or optimizer is claimed;
- intervention on the internal quantity named by the mechanism;
- boundary regime where the effect should disappear or reverse.

Bundle ablations may establish necessity of a group but not of individual components. An ablation
that changes optimization difficulty can confound the component’s conceptual role.

## 8. Test robustness in the dimensions named by the claim

Examples:

- optimizer: curvature, gradient scale, noise, batch size, delay, conditioning;
- architecture: depth, width, sequence length, resolution, topology, missing inputs;
- representation: nuisance transformation, distribution shift, label scarcity;
- efficiency: batch size, hardware, compiler, memory pressure, end-to-end latency;
- generalization: untouched datasets, tasks, temporal splits, or environments.

Do not append unrelated benchmark breadth. Each robustness dimension should test a stated scope or
alternative explanation.

## 9. Reproducibility bundle

Retain:

- clean command or script for every reported result;
- environment specification and hardware description;
- code revision and diff for the proposed method;
- configs, seeds, data version, and split identifiers;
- raw metrics, logs, and failure traces;
- analysis script that regenerates tables and figures;
- total search and confirmation compute, including failed branches when feasible.

## Research basis

- The NeurIPS reproducibility program emphasizes code, robust workflows, and explicit reporting:
  [Improving Reproducibility in Machine Learning Research](https://www.jmlr.org/papers/v22/20-303.html).
- Finite-run uncertainty can reverse apparent algorithm rankings; report interval estimates and
  robust aggregates: [Deep Reinforcement Learning at the Edge of the Statistical
  Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
- Baselines, hyperparameters, variation, and experimenter bias require deliberate design:
  [Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/v25/23-0183.html).
- Random search is a strong reference because only a subset of hyperparameters often matters:
  [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html).
- ML scholarship can mistake tuning or bundled changes for a new mechanism:
  [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341).
