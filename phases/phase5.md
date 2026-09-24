# Phase 5: Loss Landscape Geometry Diagnostics, Edge-of-Stability Margin & Automated Hyperparameter Advisor

## 1. Executive Summary

Phase 5 translates geometric loss landscape properties into optimization guidance. The current sweep report records diagnostic suggestions, but it does not contain labeled stability outcomes or a measured prediction-accuracy study. The targets below remain to be verified.

---

## 2. Loss Landscape Diagnostic Metrics (`atlas/diagnostics.py`, `atlas/sweep_advisor.py`)

ATLAS extracts four primary geometric invariants from each local Taylor jet $J = (g, \nabla g, H)$:

### 2.1 Edge-of-Stability Margin ($\mu_{\text{EoS}}$)
In gradient descent with learning rate $\eta$, local stability requires $\eta < \frac{2}{\lambda_{\max}(H)}$. In modern deep neural networks, optimizers operate along the *Edge of Stability (EoS)* (Cohen et al., 2021), where $\eta \lambda_{\max} \approx 2$.

ATLAS computes the exact margin:
$$\mu_{\text{EoS}} = \frac{2}{\eta \lambda_{\max}(H)}.$$

**Operational Regimes:**
- $\mu_{\text{EoS}} < 0.9$: **Oscillating Instability.** The optimizer is bouncing across steep ravine walls; loss oscillates violently. *Action:* Reduce learning rate by $2\times$ to $5\times$.
- $0.9 \le \mu_{\text{EoS}} \le 2.5$: **Optimal Edge-of-Stability.** Maximum progress along non-convex valleys without divergence.
- $\mu_{\text{EoS}} \gg 2.5$: **Sluggish Underfitting.** Learning rate is overly conservative; optimization is traversing flat plateaus slowly. *Action:* Scale up learning rate by $\frac{\mu_{\text{EoS}}}{1.5}$.

### 2.2 Basin Conditioning ($\kappa$)
Measures directional anisotropy of the local basin:
$$\kappa = \frac{\lambda_{\max}(H)}{\max(\lambda_{\min}(H), \; 10^{-8})}.$$
- $\kappa \le 10$: **Isotropic Well-Conditioned Minimum.** Spherical basin; standard SGD/AdamW steps are well-directed.
- $\kappa > 25$: **Ill-Conditioned Canyon.** Gradients oscillate perpendicularly to the descent direction. *Action:* Increase weight decay or AdamW $\beta_1$ momentum smoothing.

### 2.3 Basin Flatness Radius ($R_{\text{flat}}$)
Quantifies the spatial radius over which the loss remains within a prescribed tolerance $\Delta \mathcal{L} = 0.1 \times \mathcal{L}_0$:
$$R_{\text{flat}} = \sqrt{\frac{2 \Delta \mathcal{L}}{\lambda_{\max}(H)}}.$$
- Wider basins ($R_{\text{flat}} \gg 1.0$) correlate strongly with superior out-of-distribution (OOD) generalization and flatter minima (Keskar et al., 2017; Foret et al., 2020).

### 2.4 Stochastic Gradient Signal-to-Noise Ratio (SNR)
$$\mathrm{SNR}_{\Pi} = \frac{\|\Pi^\top \nabla \mathcal{L}\|^2}{\sigma_{\nabla,\Pi}^2 / B}.$$
The implementation estimates projected-gradient noise from independent calibration batches. Scalar-loss variance does not measure gradient noise. Without a positive measured gradient-noise scale, SNR is unavailable and the advisor issues no noise-based batch-size recommendation. High or low measured SNR motivates a batch-size experiment; it does not establish that the proposed change will improve validation loss.

---

## 3. Automated Sweep Advisor Engine (`atlas/sweep_advisor.py`)

The phase 6 smoke benchmark also uses a guarded directional proposal. It
uses the minimizer of the projected quadratic jet along a recent AdamW update
when the plane captures at least half of that update and local curvature is
positive. It moderates the scale by captured update energy and caps it at
eight times the exploratory learning rate. This local heuristic may be
rejected if the update is uphill or poorly captured.

Given exploratory hyperparameter trials, the advisor proposes a candidate
grid from recorded diagnostics and evaluation metrics:

```python
from atlas.sweep_advisor import SweepAdvisor

def propose_grid(diagnostics, eval_loss):
    advisor = SweepAdvisor()
    advisor.record_trial(
        "trial_01",
        {"lr": 1e-4, "weight_decay": 0.01},
        diagnostics,
        eval_metric=eval_loss,
    )
    return advisor.recommend_next_sweep()
```

---

## 4. Empirical Sweep Verification

The agent evaluates the sweep advisor on the ViT ImageNet-100 suite:

```bash
# Execute ViT landscape-guided sweep diagnostics:
make sweep_vit
```

### Diagnostic Output & Verification Artifacts
- Telemetry saved to `runs/vit_sweep_diagnostics/sweep_diagnostics_report.json`.
- Trajectory evolution visualized in `runs/vit_imagenet100/trajectory_evolution.gif`.
- Condition number and sharpness verified against full training convergence.

### Benchmark Criteria for Sweep Engine

| Metric | Target Criterion | Status |
| :--- | :---: | :---: |
| **Diagnostic Evaluation Latency** | $< 1.0$ s per trial | TPU v4 measurement not reproduced |
| **EoS Prediction Accuracy** | Correctly identify stable vs unstable trials ($>95\%$) | Labeled outcomes absent |
| **Curvature Recovery Fidelity** | Projected Hessian $\lambda_{\max}$ relative error $< 1\%$ | Independent reference absent |
| **Recommendation Monotonicity** | Suggested learning rates increase in selected regimes | Needs targeted test |

---

## 5. Autonomous Triage & Failure Recovery

If the sweep advisor produces unexpected verdicts:
1. **If all trials predict `ILL_CONDITIONED`:**
   - Check if parameter normalization is applied across layers. LayerNorm or RMSNorm weights may have tiny scale factors inflating projected eigenvalues.
   - Remediation: Implement filter-wise or block-wise normalization when computing projected metrics.
2. **If Negative Eigenvalues dominate in a trained minimum:**
   - Cause: Saddle point or non-convex ridge.
   - Remediation: Report the non-convex index $\nu = \frac{|\lambda_{\min}|}{\lambda_{\max}}$ and adjust the quadratic Taylor patch to include a trust-region regularization term $\frac{1}{2} \Delta^\top (H + \lambda_{\text{damp}} I) \Delta$.
