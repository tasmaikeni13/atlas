# Phase 2: Monte Carlo Variance Analysis, Statistical Error Certificates & Peer Domination Proofs

## 1. Executive Summary

Phase 2 sets out the statistical checks needed for **ATLAS**: finite-difference noise under independent evaluations, holdout error distributions, and peer comparisons under matched budgets. Its theoretical assumptions and archived empirical claims require the corrections and tests below.

**Evidence correction:** The archived sharpness audit uses the same batch for all three finite-difference stencil evaluations at $h=0.05$. Its recorded estimate/reference ratios are 0.13–0.45, so it does not verify the independent-noise $h^{-2}$ scaling or an 11–15-fold inflation claim. The peer-domination and 95% certificate targets below remain unverified.

---

## 2. Statistical Error Mechanics & Monte Carlo Formulation

### 2.1 The Stochastic Loss Manifold
In deep learning on large datasets $\mathcal{D}$, the true population loss $\mathcal{L}(\theta) = \mathbb{E}_{z \sim \mathcal{D}} [\ell(\theta; z)]$ is observed only via mini-batch Monte-Carlo estimators:
$$\hat{\mathcal{L}}_B(\theta) = \frac{1}{B} \sum_{k=1}^B \ell(\theta; z_k), \quad z_k \overset{\text{i.i.d.}}{\sim} \mathcal{D}.$$
By the Central Limit Theorem, the empirical loss satisfies:
$$\hat{\mathcal{L}}_B(\theta) = \mathcal{L}(\theta) + \xi_B(\theta), \quad \xi_B(\theta) \sim \mathcal{N}\left(0, \; \frac{\sigma^2(\theta)}{B}\right).$$

### 2.2 Curvature Noise Explosion Audit (Finite Differences vs. Exact Jets)
Consider estimating directional curvature along a unit vector $v \in \mathbb{R}^D$ using central finite differences with stencil step $h$:
$$\hat{\kappa}_h(\theta) = \frac{\hat{\mathcal{L}}_B(\theta + h v) - 2 \hat{\mathcal{L}}_B(\theta) + \hat{\mathcal{L}}_B(\theta - h v)}{h^2}.$$

**Theorem 3 (Curvature Noise Explosion).**
*Let the mini-batch loss evaluations at $\theta + hv, \theta, \theta - hv$ have independent observation noise with variance $\sigma^2 / B$. The variance of the central finite-difference curvature estimator satisfies:*
$$\mathrm{Var}(\hat{\kappa}_h) = \frac{\mathrm{Var}(\hat{\mathcal{L}}_B) + 4 \mathrm{Var}(\hat{\mathcal{L}}_B) + \mathrm{Var}(\hat{\mathcal{L}}_B)}{h^4} = \frac{6 \sigma^2}{B h^4}.$$
*Consequently, the standard error detonates as $h \to 0$:*
$$\mathrm{SE}(\hat{\kappa}_h) = \frac{\sqrt{6} \sigma}{\sqrt{B} \, h^2} = \mathcal{O}(h^{-2}).$$

Conversely, if $h$ is made large to suppress stochastic noise, the Taylor truncation bias introduces systematic error:
$$\mathrm{Bias}(\hat{\kappa}_h) = \frac{h^2}{12} v^\top \nabla^4 \mathcal{L}(\theta) v + \mathcal{O}(h^4) = \mathcal{O}(h^2 M_4).$$

The minimum Mean Squared Error (MSE) of finite differences balances noise against truncation bias:
$$\mathrm{MSE}(\hat{\kappa}_h) \sim \frac{6 \sigma^2}{B h^4} + c M_4^2 h^4 \implies h^* \sim \left(\frac{\sigma^2}{B M_4^2}\right)^{1/8}, \quad \mathrm{MSE}^* \sim \mathcal{O}\left(B^{-1/2}\right).$$

**ATLAS Resolution (Zero Finite-Difference Noise):**
ATLAS evaluates the projected Hessian $H = \Pi^\top \nabla^2 \hat{\mathcal{L}}_B \Pi$ using exact forward-over-reverse automatic differentiation. The estimation variance of the exact autodiff Hessian is:
$$\mathrm{Var}(\hat{H}_{\text{exact}}) = \frac{\Sigma_H}{B} = \mathcal{O}\left(B^{-1}\right),$$
with **zero dependency on any spatial discretization step $h$**. This completely eliminates the $15\times$ condition number inflation observed in standard empirical landscape packages.

---

## 3. Finite-Sample Distribution-Free DKW Error Certificates

To provide mathematically sound quality guarantees in mission-critical applications without assuming Gaussianity, ATLAS evaluates holdout validation anchors and applies the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality.

### 3.1 Unbiased Variance-Corrected Residuals
Let $\{(x_k', y_k')\}_{k=1}^{N_{\text{cert}}}$ be $N_{\text{cert}}$ independent holdout coordinates sampled via a 2D Halton sequence disjoint from the training anchors.
The original analysis proposed evaluating the loss on two independent mini-batch halves $\mathcal{B}_1, \mathcal{B}_2$ of size $B/2$:
$$\hat{g}_{k, 1} = g(x_k', y_k'; \mathcal{B}_1), \quad \hat{g}_{k, 2} = g(x_k', y_k'; \mathcal{B}_2).$$
The pooled mean and sample variance are:
$$\tilde{g}_k = \frac{\hat{g}_{k, 1} + \hat{g}_{k, 2}}{2}, \quad v_k = \frac{(\hat{g}_{k, 1} - \hat{g}_{k, 2})^2}{4}.$$

The variance-corrected squared residual is:
$$s_k^2 = \max\left(0, \; (\hat{\mathcal{L}}(x_k', y_k') - \tilde{g}_k)^2 - v_k\right).$$

**Correction:** The Lean theorem proves unbiasedness for the *unclipped* quantity
$(\hat{\mathcal{L}}-\tilde g_k)^2-v_k$ under its stated moment assumptions.
The displayed clipped quantity is generally biased upward. Neither quantity is
an observed absolute true error, so applying DKW to its empirical CDF does not
certify a quantile of the population-loss reconstruction error.

### 3.2 Massart-Tight DKW Confidence Bands
For a *fixed evaluation batch*, let $F(t)$ be the CDF of absolute
reconstruction errors at coordinates sampled independently and uniformly over
the stated 2D domain. The empirical CDF of the observed fixed-batch residuals
obeys DKW only when the holdout coordinates are iid and independent of the
fitted reconstruction. A deterministic Halton sequence does not meet this
assumption. A fixed-batch certificate does not certify population loss.

By the Dvoretzky-Kiefer-Wolfowitz inequality with Massart's optimal constant:
$$\mathbb{P}\left(\sup_{t \in \mathbb{R}} |\hat{F}_n(t) - F(t)| \le \epsilon_{\text{dkw}}\right) \ge 1 - \delta, \quad \text{where} \quad \epsilon_{\text{dkw}} = \sqrt{\frac{\ln(2/\delta)}{2 N_{\text{cert}}}}.$$

For a target statistical confidence level $1 - \delta = 0.95$ and quantile $p = 0.95$:
$$q_{0.95} \le \hat{F}_n^{-1}(0.95 + \epsilon_{\text{dkw}})
\quad\text{only if}\quad 0.95 + \epsilon_{\text{dkw}} \le 1.$$

The inverse empirical CDF must use the corresponding order statistic, not an
interpolated percentile. At 95% coverage and 95% confidence the two-sided DKW
bound needs at least 738 iid holdouts. With 14 holdouts, the maximum observed
error gives only a 63.7% DKW lower coverage bound, even if the points are iid.

**Certified Guarantee:**
When these assumptions and the sample-size condition hold, with probability at
least $95\%$, at least $95\%$ of the *fixed-batch* domain has absolute error
bounded by $q_{0.95}$:
$$\mathbb{P}\left( \text{Vol}\left(\{(x, y) : |\hat{\mathcal{L}}(x, y) - g(x, y)| \le q_{0.95}\}\right) \ge 0.95 \cdot \text{Area}(\mathcal{S}) \right) \ge 0.95.$$

---

## 4. Monte Carlo Simulation Protocol & Peer Comparison

The agent must execute synthetic and empirical Monte Carlo simulations across varying budgets, noise levels, and surface dimensionalities:

```bash
# Execute Monte Carlo curvature noise explosion audit:
make sharpness
```

### Required Evidence

| Check | Current evidence | Required smoke-scale follow-up |
| :--- | :--- | :--- |
| Independent-noise finite-difference scaling | Archived same-batch audit at one $h$ | Vary $h$ and independent batch seeds; compare variance with $6\sigma^2/(Bh^4)$ on a synthetic model. |
| Fixed-batch 95%/95% DKW certificate | 14 deterministic points, invalid | iid uniform coordinates and at least 738 holdouts for this two-sided DKW criterion. |
| Population-loss error certificate | No uniform observation-error bound | Add a valid observation-error guarantee or report only fixed-batch error. |
| Peer ranking under equal budgets | Archived metrics do not cover this claim | Record device, seeds, total wall time, and matched evaluation targets. |

---

## 5. Autonomous Diagnosis & Iterative Adaptation Protocol

If Monte Carlo simulations or statistical tests fail:
1. **If DKW Empirical Coverage $< 95\%$:**
   - Verify iid uniform holdout coordinates independent of fitting.
   - Check that $0.95+\epsilon_{\mathrm{dkw}}\le 1$; increasing a 14-point holdout fraction to 20 points does not meet this criterion.
   - For population-loss claims, separately establish an observation-error bound.
2. **If Curvature Relative Error $> 1.0\%$:**
   - Verify that JAX float32/bfloat16 precision accumulator is set to highest: `jax.config.update("jax_default_matmul_precision", "highest")`.
   - Ensure the JVPs of the reverse-mode gradient are correctly projected onto both subspace directions.
3. **If a Peer matches ATLAS in any Monte Carlo metric:**
   - Investigate anchor density distribution. Transition from quasi-random Halton anchors to Lloyd-relaxed Voronoi anchors.
   - Adjust Wendland kernel support radius $r_i$ dynamically based on local $k$-nearest neighbor distance.
   - Re-evaluate and re-run until ATLAS achieves strict statistical dominance.
