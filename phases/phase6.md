# Phase 6: Hyperparameter Sweep Comparison

## Purpose

This phase tests whether projected Taylor-jet diagnostics improve
learning-rate and weight-decay selection under a fixed compute budget.
Curvature is a heuristic for AdamW: the top eigenvalue of a two-dimensional
projected Hessian is not the top eigenvalue of the full Hessian, and
\(2/\lambda_{\max}\) is not an optimal AdamW learning rate theorem.

## Implemented smoke protocol

Run from the repository root with the optional HPO dependency installed:

    .venv/bin/python -m pip install -e '.[hpo]'
    JAX_PLATFORMS=cpu .venv/bin/python experiments/10_hpo_peer_benchmark.py --smoke_test
    JAX_PLATFORMS=cpu .venv/bin/python experiments/10_hpo_peer_benchmark.py --smoke_test --seed 43 --output_dir /tmp/atlas-hpo-smoke-43
    JAX_PLATFORMS=cpu .venv/bin/python experiments/10_hpo_peer_benchmark.py --smoke_test --seed 44 --output_dir /tmp/atlas-hpo-smoke-44

The CPU smoke run uses a 16-step training budget per method. Each method spends
8 steps selecting a configuration and then retrains its selected configuration
from common initial weights for 8 steps. The balanced two-class synthetic
color task has a recoverable signal in image channel means; training and
validation batches use independent seeds. Final loss uses a shared batch
that was not used to select hyperparameters. Shared JIT warmup is excluded
from timing; ATLAS jet work is included. The methods are:

1. ATLAS: one 8-step exploratory trajectory, a projected Taylor jet, and one
   recommendation. When a recent optimizer update is sufficiently captured
   by the plane and descends in the jet's local quadratic model, the advisor
   proposes a capped learning-rate scale along that update. Otherwise it
   uses the existing projected-curvature heuristic.
2. Random Search: four log-uniform candidates trained for 2 steps each.
3. Optuna TPE: four candidates from Optuna's seeded TPE sampler, also trained
   for 2 steps each. Three trials are startup samples.
4. Successive Halving: four candidates with synchronous 1-, 2-, and 4-step
   promotion rungs and retained optimizer state.

| Seed | ATLAS | Random Search | Optuna TPE | Successive Halving |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.000924 | 0.000191 | 0.002549 | 0.001401 |
| 43 | 0.002083 | 0.004582 | 0.000343 | 0.349314 |
| 44 | 0.039716 | 0.455166 | 0.000288 | 0.144875 |

The raw reports are in runs/hpo_benchmark/hpo_benchmark_report.json,
runs/hpo_benchmark/smoke_seed_43.json, and
runs/hpo_benchmark/smoke_seed_44.json.

These archived smoke runs did not calibrate projected-gradient noise. Their
reported stochastic SNR and noise-based batch-size advice came from a fixed
placeholder and are marked invalid in the JSON reports. The current advisor
leaves SNR unavailable without a measured gradient-noise scale.

Each row is a 16-step CPU smoke comparison with one held-out final batch.
The ranking changes by seed; ATLAS does not dominate all peers. The color
task is deliberately simple and does not establish performance on real data.
ATLAS took roughly 8 seconds per seed, versus roughly 0.1--0.2 seconds for
each peer; its measured time includes jet compilation, and the report does
not convert that cost into equivalent training steps.
The scheduler is synchronous successive halving, not asynchronous ASHA.
No method diverged. The phase is incomplete: these runs do not support a
target-loss speedup, an optimal learning-rate claim, or a divergence-prevention
rate.

## Next validation

Use a realistic task with a separate validation set, more seeds, matched
aggregate compute including diagnostic cost, and a common final horizon.
Report target-loss time and uncertainty alongside final validation loss.
Compare against an actual asynchronous scheduler if claiming ASHA.
Update the paper and downstream phase evidence from those measured results.
