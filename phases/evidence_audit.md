# Evidence audit (2026-09-24)

This audit uses the current repository and CPU smoke tests. The user restricted
execution to smoke tests, so archived large-run results have not been repeated.
`phases/run_phase.py` currently checks file presence and a short command; its
`COMPLETED` result alone does not establish each phase's scientific criteria.

| Phase | Evidence and remaining gap |
| --- | --- |
| 1 | Lean build passed after precompiled Mathlib cache recovery. The AM-GM statement is now described as a surrogate optimum. The proposed spatial and stochastic terms still need assumptions and evidence sufficient to bound reconstruction risk; peer-rate claims are unproved. |
| 2 | The archived certificates use 14 deterministic Halton holdouts. DKW requires iid coordinates, and a two-sided 95%/95% bound needs at least 738. The clipped debiased residual in the phase text is not unbiased. The archived sharpness ratios are 0.13–0.45 at $h=0.05$ with the same batch, contradicting the claimed 11–15-fold inflation and not testing independent-noise scaling. |
| 3 | Import checks cover API availability, not TPU/GPU timing or numerical accuracy. The ViT and 125M CPU smoke tests passed. TPU v4 checks remain unverified. The random-slice baseline evaluates a different plane, so its loss values are not direct reconstruction errors on the trajectory plane. |
| 4 | ViT and 125M CPU smoke tests passed. The FineWeb synthetic path now runs without the optional tokenizer. Real dataset ingestion and TPU v4 execution remain unverified. |
| 5 | The sweep report records three diagnostic trials but no labeled stability outcomes or measured prediction accuracy. The claimed 100% EoS prediction accuracy is unverified. |
| 6 | The revised HPO smoke allocates 8 selection and 8 fresh final-training steps to each method, then evaluates on a separate shared batch. TPE now uses Optuna. On this one-seed synthetic run, ATLAS final loss was 5.0034 versus 4.6427 (random), 4.7084 (TPE), and 4.6755 (synchronous successive halving); its measured time was also higher because the jet was compiled. Independent random images and labels offer no learnable task signal. No target-loss step comparison or divergent peer trial proves a speedup or prevention rate. |
| 7 | The archived ViT ImageNet benchmark contradicts universal peer domination: at a 2-second budget, ATLAS has L2 error 0.000508 versus grid 0.000499, and Spearman 0.715 versus grid 0.727. Its recorded Lanczos and Hutchinson times also exceed the nominal budget. No large run was repeated. |
| 8 | The phase asserts OOD accuracy correlation 0.842 and RBF ablation outcomes, but raw OOD paired observations and ablation telemetry are absent from tracked artifacts. The implementation uses Shepard weights, not Wendland. The certificate plots are empirical only. |
| 9 | The paper builds, but its publication claims depend on the unresolved phase 1, 2, 6, 7 and 8 evidence. This phase cannot be marked complete on PDF existence alone. |

Next checks are the 125M CPU smoke, Lean proof build after cache recovery,
and smoke-only peer benchmarks. Large-scale superiority and OOD claims require
new evidence and cannot be inferred from existing file names.
