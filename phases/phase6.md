# Phase 6: Hyperparameter Sweep Comparison

## Purpose

This phase tests whether the projected Taylor-jet diagnostics improve
learning-rate and weight-decay selection under a fixed compute budget.
Curvature is a heuristic for AdamW: the top eigenvalue of a two-dimensional
projected Hessian is not the top eigenvalue of the full Hessian, and
\(2/\lambda_{\max}\) is not an optimal AdamW learning rate theorem.

## Implemented smoke protocol

Run from the repository root with the optional HPO dependency installed:

    .venv/bin/python -m pip install -e '.[hpo]'
    JAX_PLATFORMS=cpu .venv/bin/python experiments/10_hpo_peer_benchmark.py --smoke_test

The CPU smoke run uses a 16-step training budget per method. Each method spends
8 steps selecting a configuration and then retrains its selected configuration
from common initial weights for 8 steps. The final loss uses a shared batch
that was not used to select hyperparameters. Shared JIT warmup is excluded
from timing; ATLAS jet work is included. The methods are:

1. ATLAS: one 8-step exploratory trajectory, a projected Taylor jet, and one
   recommended configuration.
2. Random Search: four log-uniform candidates trained for 2 steps each.
3. Optuna TPE: four candidates from Optuna's seeded TPE sampler, also trained
   for 2 steps each. Three trials are startup samples.
4. Successive Halving: four candidates with synchronous 1-, 2-, and 4-step
   promotion rungs and retained optimizer state.

| Method | Final validation loss | Training steps |
| --- | ---: | ---: |
| ATLAS | 5.0034 | 16 |
| Random Search | 4.6427 | 16 |
| Optuna TPE | 4.7084 | 16 |
| Successive Halving | 4.6755 | 16 |

These values are smoke evidence only. The synthetic images and labels are
independent, so the task has no learnable signal. There is one seed and one
final evaluation batch. The measured ATLAS time includes jet compilation,
and the report does not convert that cost into equivalent training steps.
The scheduler is synchronous successive halving, not asynchronous ASHA.
No method diverged. The phase is incomplete: the run does not support a
target-loss speedup, an optimal learning-rate claim, or a divergence-prevention
rate.

## Next validation

Use a learnable task with a separate validation set, multiple seeds, matched
aggregate compute including diagnostic cost, and a common final horizon.
Report target-loss time and confidence intervals alongside final validation
loss. Compare against an actual asynchronous scheduler if claiming ASHA.
Update the paper and downstream phase evidence from those measured results.
