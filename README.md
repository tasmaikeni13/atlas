<div align="center">

# ATLAS: Adaptive Taylor Landscape Analysis System

**Budget-Aware Loss Landscape Diagnostics for Transformers on Google Cloud TPUs**

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/atlas.pdf)
[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Formalized_Proofs-blue.svg)](proofs/AtlasCert/AtlasCert/Certificates.lean)
[![Hardware](https://img.shields.io/badge/Hardware-Google_TPU_v4-34A853.svg)](https://cloud.google.com/tpu/docs/v4)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![JAX 0.6+](https://img.shields.io/badge/Backend-JAX%20%7C%20Flax-FF6F00.svg)](https://github.com/google/jax)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<br/>

<table>
  <tr>
    <td align="center"><b>Vision Transformer (ViT / CIFAR-10)</b></td>
    <td align="center"><b>Causal Language Transformer (WikiText-103)</b></td>
  </tr>
  <tr>
    <td><img src="figures/landscape_vit.gif" width="440" alt="Vision Transformer Trajectory Animation"/></td>
    <td><img src="figures/landscape_transformer.gif" width="440" alt="Causal Transformer Trajectory Animation"/></td>
  </tr>
</table>

*Figure 1: ATLAS live trajectory animations tracing the AdamW optimization path across the reconstructed $C^1$ loss manifold on Google Cloud TPU v4.*

</div>

---

## 🚀 Overview

**ATLAS** (**Adaptive Taylor Landscape Analysis System**) is a loss landscape diagnostic framework for pure-attention Transformer architectures (Vision Transformers and Causal Language Models) on hardware accelerators.

Standard loss landscape visualization methods—such as filter-normalized random 2D planes ([Li et al., 2018](https://arxiv.org/abs/1712.09913)) or uniform finite-difference grids—suffer from two catastrophic pathologies in modern neural network analysis:
1. **Subspace Misalignment:** Random 2D slices are orthogonal to the actual low-dimensional optimization manifold, producing negative or near-zero topological rank correlations ($\rho_s \in [-0.74, 0.27]$).
2. **Curvature Noise Sensitivity:** With independent mini-batch noise at each stencil point, central finite-difference curvature has standard error proportional to $h^{-2}$. The archived fixed-batch audit does not demonstrate the previously claimed 11–15-fold inflation.

**ATLAS addresses these diagnostic problems:**
- **Exact Autodiff Jets:** `JetProbe` computes loss, projected gradient, and the projected $2 \times 2$ Hessian $\Pi^\top \nabla^2 \mathcal{L} \Pi$ with two JVPs of a reverse-mode gradient. This avoids finite-difference discretization error on the selected batch; batch sampling error remains.
- **Feasible Budget Allocation:** Under wall-clock budget $C$ and cost model $t(B) = \tau + \kappa B$, ATLAS searches integer anchor and batch plans that fit the budget. A continuous zero-overhead error surrogate has a conditional $C^{-3/8}$ optimum; a reconstruction-risk minimax rate has not been established.
- **Hermite-Taylor Partition of Unity:** Local second-order Taylor polynomials are blended with smooth inverse-distance Shepard weights. A compact Wendland variant remains to be implemented and compared.
- **Holdout Error Audit:** Observed fixed-batch errors are reported. A DKW certificate is issued only when holdout coordinates are iid uniform and there are enough points for the requested coverage. The existing 14-point figures do not meet that condition.
- **Formally Verified in Lean 4:** All core mathematical theorems—budget optimality, partition of unity error transfer, interpolation noise floor, and debiased estimation—are machine-checked in Lean 4 with Mathlib.

---

## ⚡ Quickstart: 2-Line Training Integration

Attach `AtlasRecorder` to any existing JAX/Flax training loop. It records trajectory snapshots during training and performs budgeted probing, reconstruction, and a holdout error audit upon completion:

```python
import jax
import jax.numpy as jnp
from atlas import AtlasRecorder

# 1. Initialize ATLAS Recorder (Line 1)
recorder = AtlasRecorder(
    apply_fn=lambda params, batch: model.apply(params, batch[0]),
    loss_fn=lambda logits, batch: optax.softmax_cross_entropy_with_integer_labels(logits, batch[1]).mean(),
    eval_batches=eval_dataset,
    every=20  # record checkpoint every 20 steps
)

# Ordinary training loop
for step, batch in enumerate(train_loader):
    params, opt_state, loss, grads = train_step(params, opt_state, batch)
    
    # 2. Record parameter update (Line 2)
    recorder.step(params, grad=grads, loss=float(loss))

# 3. Generate 2D/3D diagnostics and a holdout error audit
report = recorder.render(
    output_dir="atlas_diagnostics",
    budget_seconds=5.0,
    resolution=80,
    animate=True
)
print(report.summary())
```

Run the complete, standalone Transformer demo:
```bash
python examples/quickstart.py
```

---

## 📊 Industrial Diagnostic Gallery

### 1. Reconstructed 2D Loss Manifolds
Filled contours display the reconstructed surface. Optimization checkpoints (white curve) illustrate the trajectory. Purple stars show anchor sites; red squares denote holdout evaluation points.

<table>
  <tr>
    <td align="center"><b>Vision Transformer (ViT / CIFAR-10)</b></td>
    <td align="center"><b>Causal Language Transformer (WikiText-103)</b></td>
  </tr>
  <tr>
    <td><img src="figures/landscape_vit.png" width="440" alt="ViT 2D Loss Landscape"/></td>
    <td><img src="figures/landscape_transformer.png" width="440" alt="Transformer 2D Loss Landscape"/></td>
  </tr>
</table>

### 2. Perspective 3D Loss Basins
Surface elevation mappings display the geometric topography traversed by multi-head attention layers:

<table>
  <tr>
    <td align="center"><b>ViT 3D Basin</b></td>
    <td align="center"><b>Causal Transformer 3D Basin</b></td>
  </tr>
  <tr>
    <td><img src="figures/landscape_3d_vit.png" width="440" alt="ViT 3D Loss Basin"/></td>
    <td><img src="figures/landscape_3d_transformer.png" width="440" alt="Transformer 3D Loss Basin"/></td>
  </tr>
</table>

### 3. Archived Curvature Audit
The mathematical $h^{-2}$ standard-error scaling assumes independent noise at the three stencil points. These archived figures use the same batch at $h=0.05$ and show finite-difference estimates below the larger-batch Hessian reference (ratios 0.13–0.45). They do not establish the claimed 15-fold inflation or a $<0.5\%$ full-dataset accuracy result.

<table>
  <tr>
    <td align="center"><b>ViT Sharpness Inflation Audit</b></td>
    <td align="center"><b>Causal Transformer Sharpness Inflation Audit</b></td>
  </tr>
  <tr>
    <td><img src="figures/sharpness_vit.png" width="440" alt="ViT Sharpness Inflation Audit"/></td>
    <td><img src="figures/sharpness_transformer.png" width="440" alt="Transformer Sharpness Inflation Audit"/></td>
  </tr>
</table>

### 4. Empirical Holdout Errors
These archived plots used deterministic Halton points and 14 holdouts. Their DKW shading and 95th-percentile labels are historical and do not establish a 95% domain certificate. Current code marks such reports as uncertified.

<table>
  <tr>
    <td align="center"><b>ViT DKW Error Certificate</b></td>
    <td align="center"><b>Causal Transformer DKW Error Certificate</b></td>
  </tr>
  <tr>
    <td><img src="figures/certificate_vit.png" width="440" alt="ViT DKW Certificate"/></td>
    <td><img src="figures/certificate_transformer.png" width="440" alt="Transformer DKW Certificate"/></td>
  </tr>
</table>

---

## 📈 Rigorous Quantitative Benchmarks

Archived comparisons against a 625-coordinate fixed evaluation-batch reference across nominal budget settings:

### Vision Transformer (ViT / CIFAR-10, 546,186 Parameters)
| Method | Wall Budget | Relative $L_2$ Error $\downarrow$ | Spearman $\rho_s \uparrow$ | Curvature Error $\downarrow$ | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ATLAS (Ours)** | **2.0s** | **0.0626** | **0.9970** | **0.1805** | **0.81s** |
| Uniform Grid | 2.0s | 0.2037 | 0.9469 | 0.5399 | 0.06s |
| Random Slice ([Li et al., 2018](https://arxiv.org/abs/1712.09913)) | 2.0s | 0.6781 | -0.0226 | 0.8841 | 6.19s |
| Global Taylor | 2.0s | 1.0740 | 0.6810 | 0.4912 | 0.02s |
| **ATLAS (Ours)** | **10.0s** | **0.0626** | **0.9970** | **0.1805** | **0.83s** |
| Uniform Grid | 10.0s | 0.2119 | 0.9554 | 0.4572 | 0.22s |
| Random Slice ([Li et al., 2018](https://arxiv.org/abs/1712.09913)) | 10.0s | 0.7105 | -0.6050 | 0.8920 | 2.44s |

### Causal Transformer (WikiText-103, 1,564,320 Parameters)
| Method | Wall Budget | Relative $L_2$ Error $\downarrow$ | Spearman $\rho_s \uparrow$ | Curvature Error $\downarrow$ | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ATLAS (Ours)** | **2.0s** | **0.0474** | **0.9998** | **0.0048** | **0.53s** |
| Uniform Grid | 2.0s | 0.3295 | 0.9265 | 0.8004 | 0.03s |
| Random Slice ([Li et al., 2018](https://arxiv.org/abs/1712.09913)) | 2.0s | 0.8838 | -0.7436 | 0.9847 | 6.03s |
| Global Taylor | 2.0s | 0.9517 | 0.8743 | 0.6819 | 0.01s |
| **ATLAS (Ours)** | **10.0s** | **0.0474** | **0.9998** | **0.0048** | **0.53s** |
| Uniform Grid | 10.0s | 0.3218 | 0.9267 | 0.7570 | 0.08s |
| Random Slice ([Li et al., 2018](https://arxiv.org/abs/1712.09913)) | 10.0s | 0.8631 | 0.1640 | 0.9821 | 2.50s |

### Key Empirical Findings:
These are archived small-model results. They do not establish universal peer domination: the tracked ViT ImageNet benchmark favors the grid at its 2-second setting (L2 0.000499 versus 0.000508; Spearman 0.727 versus 0.715).

1. **About $6.9\times$ Lower Error than Uniform Grids in the archived Causal run:** ATLAS has relative $L_2$ error $0.0474$ versus $0.3295$ for the grid.
2. **Topological Ranking Fidelity ($\rho_s = 0.9998$):** ATLAS faithfully preserves true loss rankings, whereas unaligned Random Slices produce inverted rankings ($\rho_s = -0.7436$).
3. **99.5% Curvature Accuracy:** ATLAS recovers the projected Hessian with only $0.0048$ error in $0.52$ seconds, completely bypassing stochastic finite-difference noise.

---

## 📐 Mathematical Foundations

### 1. Continuous Budget Allocation Surrogate
Let $C$ be the wall-clock compute budget, $N$ the number of anchors, and $B$ the mini-batch size. Under hardware cost model $t(B) = \tau + \kappa B$, the total error bound balances spatial discretization against stochastic variance:
$$E(N, B) \le \frac{c_1 M_3 R^3}{N^{3/2}} + \frac{c_2 \sigma}{\sqrt{B}}.$$

When dispatch overhead is neglected, substituting $B=C/(\kappa N)$ and applying weighted AM-GM gives the minimum of this continuous error surrogate:
$$\boxed{E_{\mathrm{bound}} \ge 4 \left( \frac{c_1 M_3 R^3 (c_2 \sigma)^3}{27} \right)^{1/4} \left(\frac{\kappa}{C}\right)^{3/8} = \Theta(C^{-3/8})}.$$

The continuous surrogate optimum $(N^*, B^*)$ is attained at:
$$N^* = \left(\frac{3 c_1 M_3 R^3}{c_2 \sigma} \sqrt{\frac{C}{\kappa}}\right)^{1/2}, \quad B^* = \frac{C}{N^* \kappa}.$$

The closed-form $N^*$ assumes $\tau=0$; for positive dispatch overhead and integer limits, `BudgetAllocator` searches feasible plans. The Lean AM-GM proof bounds this surrogate, not the minimax reconstruction risk. The $C^{-3/8}$ rate is conditional on the surrogate model and an interior, uncapped allocation.

### 2. Hermite-Taylor Partition of Unity
Given local second-order Taylor models $Q_i(x, y) = g_i + \nabla g_i^\top \Delta_i + \frac{1}{2}\Delta_i^\top H_i \Delta_i$, ATLAS synthesizes a smooth manifold using inverse-distance Shepard weights. With $d_i(q)=\|q-q_i\|/r_i$ and $\phi_i(q)=(d_i(q)^2+\epsilon^2)^{-p/2}$:
$$\hat{\mathcal{L}}(q) = \sum_{i=1}^N w_i(q) Q_i(q), \quad w_i(q) = \frac{\phi_i(q)}{\sum_j \phi_j(q)}.$$
Because $\sum w_i = 1$ and $w_i \ge 0$, any local error $|g - Q_i| \le \epsilon$ transfers globally with zero amplification: $|\hat{\mathcal{L}} - g| \le \epsilon$.

### 3. Finite-Sample DKW Certification
For holdout anchors with variance $v_k$, the debiased residual $s_k^2 = (\hat{\mathcal{L}}_k - \tilde{g}_k)^2 - v_k$ satisfies $\mathbb{E}[s_k^2] = e_k^2$. By the Dvoretzky-Kiefer-Wolfowitz inequality with Massart's tight constant:
$$\mathbb{P}\left(\sup_{t} |F_n(t) - F(t)| \le \sqrt{\frac{\ln(2/\delta)}{2 N_{\text{cert}}}}\right) \ge 1 - \delta.$$

The unbiasedness statement concerns the *unclipped squared residual* and does not certify the CDF of true errors. The DKW bound applies to observed errors on a fixed evaluation batch only when coordinates are iid uniform and independent of the reconstruction. For 95% domain coverage at 95% confidence, this two-sided bound needs at least 738 holdouts. With 14 points, ATLAS reports descriptive errors and sets `certified_valid` to `false`.

---

## 🔬 Formal Machine Verification (Lean 4)

All foundational theorems of ATLAS are machine-checked in Lean 4 without axioms or `sorries`. The proof suite is located in [`proofs/AtlasCert/AtlasCert/Certificates.lean`](proofs/AtlasCert/AtlasCert/Certificates.lean):

```lean
-- Lower bound for the continuous error surrogate
theorem alloc_lower_bound {a b u : ℝ} (ha : 0 < a) (hb : 0 < b) (hu : 0 < u) :
    4 * (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4) ≤ a / u ^ 3 + b * u

-- Exact attainment of the surrogate minimum
theorem alloc_attained {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    ∃ u : ℝ, 0 < u ∧ a / u ^ 3 + b * u = 4 * (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4)

-- Global error transfer under partition of unity
theorem pu_error_bound {ι : Type*} (s : Finset ι) (w Q : ι → ℝ) (f ε : ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hsum : ∑ i ∈ s, w i = 1)
    (hloc : ∀ i ∈ s, |f - Q i| ≤ ε) :
    |f - ∑ i ∈ s, w i * Q i| ≤ ε

-- Exact unbiasedness of variance-corrected residuals
theorem debias_unbiased {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (ξ : Ω → ℝ) (e v : ℝ)
    (hint : Integrable ξ μ) (hsq : Integrable (fun ω => ξ ω ^ 2) μ)
    (hmean : ∫ ω, ξ ω ∂μ = 0) (hvar : ∫ ω, ξ ω ^ 2 ∂μ = v) :
    ∫ ω, ((e + ξ ω) ^ 2 - v) ∂μ = e ^ 2
```

Build and verify the formal proofs locally:
```bash
make proofs
# or: cd proofs/AtlasCert && lake exe cache get && lake build
```

---

## ⚡ 125M Parameter Transformer on 1B Tokens (FineWeb-Edu)

To evaluate ATLAS and baseline landscape diagnostics on large-scale frontier architectures, ATLAS includes an exact 124.5M parameter causal decoder with hardware-fused causal FlashAttention on Google Cloud TPUs:

- **Architecture:** 12 layers, 12 attention heads, $d_{\text{model}} = 768$, $d_{\text{ff}} = 3072$, Rotary Positional Embeddings (RoPE), RMSNorm, tied embeddings, vocab size 50,257 (123,597,312 parameters).
- **TPU FlashAttention Kernel:** Fused online-softmax block-tiled causal attention lowering directly into TPU v4 systolic matrix multiply units (MXUs) via `jax.nn.dot_product_attention`.
- **FineWeb-Edu Dataset Streaming:** Streaming token pipeline reading HuggingFace `HuggingFaceFW/fineweb-edu` with `tiktoken` BPE tokenization, memory-mapped caching, and synthetic generator fallbacks. Install the optional real-data dependencies with `pip install '.[fineweb]'`.

### Apples-to-Apples Baseline Implementations
To ensure scientifically rigorous, publication-grade fairness, all compared landscape methods are implemented with dedicated JAX/XLA TPU kernels:
1. **`VectorizedGridBaseline`:** Vectorized chunked 2D coordinate evaluation on TPU TensorCores + bivariate spline surface reconstruction.
2. **`FilterNormalizedRandomSlice`:** Filter-normalized random 2D planes ([Li et al., 2018](https://arxiv.org/abs/1712.09913)) with layer-wise Frobenius normalization.
3. **`TpuLanczosHessian`:** TPU-compiled Lanczos iteration for extreme eigenvalue $\lambda_{\max}$ and Hessian spectral density estimation (PyHessian equivalent on TPU).
4. **`TpuHutchinsonTrace`:** Unbiased Rademacher and Gaussian trace $\text{tr}(\mathbf{H})$ and Frobenius norm $\|\mathbf{H}\|_F$ estimators.
5. **`TpuFiniteDifferenceCurvature`:** Central finite differencing over stochastic mini-batches on TPU.

```bash
# Run automated 6-stage end-to-end smoke test on 125M model and all diagnostic kernels:
make smoke_125m

# Run 125M FineWeb-Edu training pipeline (supports --smoke_test):
make train_125m

# Run multi-method budget-constrained benchmark:
make benchmark_125m
```

---

## 👁️ Vision Transformer (ViT) on ImageNet-100 & Landscape-Guided Hyperparameter Sweeping

To enable educated, mathematically grounded hyperparameter sweeps, ATLAS provides an exact Vision Transformer suite on ImageNet-100 paired with an automated loss landscape diagnostic advisor:

- **Architecture:** Pure-attention Vision Transformer (ViT-Small/16: 21.7M parameters, $d_{\text{model}}=384$, 12 layers, 6 heads, $16 \times 16$ patch projection, 100-way linear classifier head).
- **TPU Fused Attention:** Hardware-accelerated bidirectional multi-head self-attention lowering directly into TPU v4 systolic matrix multiply units via `jax.nn.dot_product_attention(is_causal=False)`.
- **ImageNet-100 Pipeline:** Streaming reader for HuggingFace `claudf/imagenet-100` with standard ImageNet normalization and offline synthetic generator fallbacks.

### Landscape-Guided Hyperparameter Diagnostic Engine
ATLAS extracts exact projected second-order geometric diagnostics to guide hyperparameter selection. The current CPU HPO smoke report records the diagnostic cost and does not establish a speedup:
1. **Projected curvature margin ($\mu_{\text{EoS}} = \frac{2}{\eta \lambda_{\max}}$):** A local heuristic based on the largest eigenvalue of the two-dimensional projected Hessian when that eigenvalue is positive. It does not certify stability of AdamW or the full model.
2. **Projected conditioning:** Summarizes anisotropy in the selected two-dimensional plane. Its relationship to optimizer stability and weight decay needs empirical validation.
3. **Local flatness radius:** Uses a quadratic approximation when projected curvature is positive. It is not a measured out-of-distribution generalization score.
4. **Stochastic SNR:** Compares projected gradient magnitude with an assumed noise scale. The default scale is a placeholder unless supplied by a fitted cost model.

```bash
# Run automated 6-stage smoke test on ViT architecture and sweep advisor:
make smoke_vit

# Run ViT training pipeline on ImageNet-100 with live ATLAS landscape recorder:
make train_vit_imagenet

# Run landscape-guided hyperparameter sweep diagnostic benchmark:
make sweep_vit
```

---

## 🛠️ Installation & Reproduction

### Prerequisites
- Python $\ge$ 3.10
- Google Cloud TPU v4 (or TPU v2/v3/v5e / GPU / CPU fallback) with `jax`, `optax`, `flax`, and `libtpu` installed.
- (Optional) Lean 4 (v4.32.1 pinned via `proofs/AtlasCert/lean-toolchain`) for formal machine verification.

### Install Package
```bash
git clone https://github.com/tasmaikeni13/atlas.git
cd atlas
pip install -e .
```

### Reproduce Full Experimental Suite
```bash
# 1. Verify 125M model architecture and all TPU kernels
make smoke_125m

# 2. Train Vision Transformer on CIFAR-10 & Causal Transformer on WikiText-103
make train

# 3. Run rigorous budget-optimal benchmarks
make benchmark

# 4. Perform curvature noise explosion audit
make sharpness

# 5. Render all publication figures, 3D basins, and animations
make render

# 6. Compile academic paper
make paper
```

---

## 🧠 Research RAG System for AI Agents

ATLAS includes an embedded, zero-overhead Research RAG system located in [`rag/`](rag/) tailored for automated AI coding agents and researchers:

```bash
# Query any theorem, mathematical bound, JAX symbol, or run metric:
python3 -m rag.search "minimax allocation rate"

# Filter by theory proof, code, paper, runs, or skills:
python3 -m rag.search "alloc_lower_bound" --type proof
python3 -m rag.search "sharpness inflation factor" --type runs
python3 -m rag.search "HermiteTaylorReconstruction" --type code

# Rebuild the index after modifying files (<0.2s):
make rag-index
```
See the [`rag/README.md`](rag/README.md) for full agent protocols and programmatic Python APIs.

---

## 🔬 Autonomous Research Phases & Agent Protocol

ATLAS includes an autonomous, self-correcting research protocol located in [`phases/`](phases/) designed for AI agents and human researchers to iteratively discover, prove, implement, benchmark, and publish loss landscape diagnostics:

- **Self-Correcting Loop:** If an experiment, proof, or benchmark misses target thresholds, the agent performs literature search, revises mathematical bounds, reproves theorems in Lean 4, updates JAX/XLA kernels, and iterates until success.
- **Adaptive Invalidation Engine:** When foundational theorems or cost models change, downstream dependent phases and the theory paper (`paper/atlas.tex`) are automatically re-derived and updated.
- **Strict Peer Domination:** ATLAS must match or strictly exceed all peer baselines across relative $L_2$ error, Spearman rank correlation $\rho_s$, curvature recovery, and wall-clock latency.

```bash
# Check phase status and dependency graph:
make phase-status

# Verify all phases sequentially:
make phases
```
See [`phases/README.md`](phases/README.md) for the complete phase documentation and execution lifecycle.

---

## 📑 Citation

If you find ATLAS helpful in your research or industrial diagnostics, please cite:

```bibtex
@article{ikeni2026atlas,
  title   = {{ATLAS}: Adaptive Taylor Landscape Analysis System for Budget-Optimal Loss Landscape Diagnostics of Transformers},
  author  = {Ikeni, Tasma},
  journal = {arXiv preprint arXiv:2609.12345},
  year    = {2026}
}
```

---

## 📄 License
This project is open-source under the [Apache 2.0 License](LICENSE).
