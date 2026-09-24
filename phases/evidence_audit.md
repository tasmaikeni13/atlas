# Evidence audit (2026-09-24)

This audit uses the current repository and CPU smoke tests. The user restricted
execution to smoke tests, so archived large-run results have not been repeated.
`phases/run_phase.py` currently checks file presence and a short command; its
`COMPLETED` result alone does not establish each phase's scientific criteria.

| Phase | Evidence and remaining gap |
| --- | --- |
| 1 | Lean build passed after precompiled Mathlib cache recovery. The AM-GM statement is now described as a surrogate optimum. Calibrated scalar-loss variance and radial Hessian slope are empirical proxies, not proof of the global derivative and gradient-noise assumptions. Peer-rate claims remain unproved. |
| 2 | The archived certificates use 14 deterministic Halton holdouts. DKW requires iid coordinates, and a two-sided 95%/95% bound needs at least 738. The clipped debiased residual in the phase text is not unbiased. The archived sharpness ratios are 0.13–0.45 at $h=0.05$ with the same batch, contradicting the claimed 11–15-fold inflation and not testing independent-noise scaling. |
| 3 | Import checks cover API availability, not TPU/GPU timing or numerical accuracy. The ViT and 125M CPU smoke tests passed. Jet calibration now waits for completed outputs, warms every batch shape, and rejects unresolved cost fits; a small CPU recorder render passed. TPU v4 checks remain unverified. The random-slice baseline evaluates a different plane, so its loss values are not direct reconstruction errors on the trajectory plane. |
| 4 | ViT and 125M CPU smoke tests passed. The FineWeb synthetic path now runs without the optional tokenizer. Real dataset ingestion and TPU v4 execution remain unverified. |
| 5 | The sweep report records three diagnostic trials but no labeled stability outcomes or measured prediction accuracy. The claimed 100% EoS prediction accuracy is unverified. |
| 6 | The HPO smoke allocates 8 selection and 8 fresh final-training steps per method and evaluates on a separate shared batch. TPE uses Optuna. A learnable color-pattern task was run for three CPU smoke seeds; rankings vary, and ATLAS does not dominate all peers. Its jet compilation makes measured CPU time much higher. No target-loss comparison or divergent peer trial proves a speedup or prevention rate. Raw reports for seeds 42--44 are tracked in runs/hpo_benchmark/. |
| 7 | All three archived ATLAS landscape reports used a marginal cost floor of 1e-8 from timing queued work; their nominal allocations are marked invalid. The archived ViT ImageNet benchmark also contradicts universal peer domination: at a nominal 2-second setting, ATLAS has L2 error 0.000508 versus grid 0.000499, and Spearman 0.715 versus grid 0.727. Recorded Lanczos and Hutchinson times exceed that setting. No large run was repeated. |
| 8 | The phase asserts OOD accuracy correlation 0.842 and RBF ablation outcomes, but raw OOD paired observations and ablation telemetry are absent from tracked artifacts. The implementation uses Shepard weights, not Wendland. The certificate plots are empirical only. |
| 9 | The paper builds, but its publication claims depend on the unresolved phase 1, 2, 6, 7 and 8 evidence. This phase cannot be marked complete on PDF existence alone. |

Next checks are target-hardware cost calibration and smoke-only peer
benchmarks using completed execution timing. Large-scale superiority and OOD
claims require new evidence and cannot be inferred from existing file names.
