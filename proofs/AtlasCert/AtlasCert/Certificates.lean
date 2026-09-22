/-
  ATLAS: Adaptive Taylor Landscape Analysis System
  Formal verification of error bounds, budget-optimal minimax rates,
  and variance-corrected certification in Lean 4.

  Theorems proved:
    1. `alloc_lower_bound` / `alloc_attained`:
       Exact minimax lower bound E ≥ 4*(a*b^3/27)^(1/4) for the budget-allocation
       tradeoff a/u^3 + b*u, proving that no allocation beats the C^(-3/8) rate.
    2. `pu_error_bound`:
       Partition of unity carries local Taylor patch bounds to global domain bounds with no loss.
    3. `interpolation_floor`:
       Exact identification of the stochastic noise floor on grid interpolants.
    4. `debias_unbiased`:
       Exact unbiasedness of the variance-corrected error certificate.
    5. `sum_dist_sq_center`:
       Mean-centered PCA affine subspace optimality under Frobenius projection.
    6. `taylor2_patch_bound`:
       Second-order Lagrange Taylor remainder bound along radial rays.
-/

import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.Calculus.Taylor

namespace Atlas

open Finset

/-! ## 1. Budget allocation -/

/-- **Allocation lower bound.**  For positive `a`, `b`, the two-term error
`a/u^3 + b*u` is bounded below by `4 * (a*b^3/27)^(1/4)` for every `u > 0`.

With `u = sqrt n`, `a = c₁ M₃ R³` (the approximation term, which falls as `n^(-3/2)`) and
`b = c₂ σ sqrt(κ/C)` (the Monte-Carlo term, which grows as `n^(1/2)` once the budget is
split `n` ways), the bound reads
`E ≥ 4 (c₁M₃R³ (c₂σ)³ / 27)^(1/4) (κ/C)^(3/8)`: no allocation of a budget `C` beats the
`C^(-3/8)` rate. -/
theorem alloc_lower_bound {a b u : ℝ} (ha : 0 < a) (hb : 0 < b) (hu : 0 < u) :
    4 * (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4) ≤ a / u ^ 3 + b * u := by
  have h1 : (0 : ℝ) ≤ a / u ^ 3 := by positivity
  have h2 : (0 : ℝ) ≤ b * u / 3 := by positivity
  have key := Real.geom_mean_le_arith_mean4_weighted
    (by norm_num : (0:ℝ) ≤ 1/4) (by norm_num : (0:ℝ) ≤ 1/4)
    (by norm_num : (0:ℝ) ≤ 1/4) (by norm_num : (0:ℝ) ≤ 1/4)
    h1 h2 h2 h2 (by norm_num)
  have hprod : (a / u ^ 3) * (b * u / 3) * (b * u / 3) * (b * u / 3) = a * b ^ 3 / 27 := by
    field_simp
    ring
  rw [← Real.mul_rpow h1 h2, ← Real.mul_rpow (by positivity) h2,
      ← Real.mul_rpow (by positivity) h2, hprod] at key
  linarith

/-- **The bound is attained.**  Equality holds at `u = (3a/b)^(1/4)`, so the lower bound
above is the exact minimum and the allocation rule is optimal, not merely feasible. -/
theorem alloc_attained {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    ∃ u : ℝ, 0 < u ∧ a / u ^ 3 + b * u = 4 * (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4) := by
  set u : ℝ := (3 * a / b) ^ ((1 : ℝ) / 4) with hu_def
  have hpos : 0 < 3 * a / b := by positivity
  have hu : 0 < u := Real.rpow_pos_of_pos hpos _
  have hu4 : u ^ 4 = 3 * a / b := by
    rw [hu_def, ← Real.rpow_natCast ((3 * a / b) ^ ((1 : ℝ) / 4)) 4,
        ← Real.rpow_mul hpos.le]
    norm_num
  refine ⟨u, hu, ?_⟩
  have hbu4 : b * u ^ 4 = 3 * a := by
    rw [hu4]; field_simp
  have hsplit : a / u ^ 3 = b * u / 3 := by
    rw [div_eq_iff (ne_of_gt (pow_pos hu 3))]
    linear_combination (-1 / 3 : ℝ) * hbu4
  have hval : (a * b ^ 3 / 27) ^ ((1 : ℝ) / 4) = b * u / 3 := by
    have h4 : (b * u / 3) ^ (4 : ℕ) = a * b ^ 3 / 27 := by
      linear_combination (b ^ 3 / 81 : ℝ) * hbu4
    rw [← h4, ← Real.rpow_natCast (b * u / 3) 4, ← Real.rpow_mul (by positivity)]
    norm_num
  rw [hsplit, hval]
  ring

/-! ## 2. Partition of unity -/

/-- **Local accuracy transfers to global accuracy.**  If non-negative weights summing to
one blend local models each within `ε` of the target, the blend is within `ε`. -/
theorem pu_error_bound {ι : Type*} (s : Finset ι) (w Q : ι → ℝ) (f ε : ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hsum : ∑ i ∈ s, w i = 1)
    (hloc : ∀ i ∈ s, |f - Q i| ≤ ε) :
    |f - ∑ i ∈ s, w i * Q i| ≤ ε := by
  have key : ∑ i ∈ s, w i * (f - Q i) = f - ∑ i ∈ s, w i * Q i := by
    simp only [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul]
  rw [← key]
  calc |∑ i ∈ s, w i * (f - Q i)| ≤ ∑ i ∈ s, |w i * (f - Q i)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i ∈ s, w i * |f - Q i| := by
        refine Finset.sum_congr rfl fun i hi => ?_
        rw [abs_mul, abs_of_nonneg (hw i hi)]
    _ ≤ ∑ i ∈ s, w i * ε :=
        Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hloc i hi) (hw i hi)
    _ = ε := by rw [← Finset.sum_mul, hsum, one_mul]

/-! ## 3. The interpolation floor -/

/-- **An interpolant inherits its observations' noise exactly.**  If a reconstruction
reproduces the observed values `y i + ξ i` at the observation sites, then its squared
error there is `ξ i ^ 2` -- identically, for every design and every budget. -/
theorem interpolation_floor {ι : Type*} (s : Finset ι) (R y ξ : ι → ℝ)
    (hR : ∀ i ∈ s, R i = y i + ξ i) :
    ∑ i ∈ s, (R i - y i) ^ 2 = ∑ i ∈ s, ξ i ^ 2 := by
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [hR i hi]
  ring

/-! ## 4. The variance-corrected certificate -/

open MeasureTheory

/-- **The certificate is unbiased.**  Let `e` be the error of the reconstruction at a probe
point and `ξ` the zero-mean noise of the probe, with `∫ ξ² = v`.  Then the variance-corrected
squared residual `(e + ξ)² - v` has expectation exactly `e²`. -/
theorem debias_unbiased {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (ξ : Ω → ℝ) (e v : ℝ)
    (hint : Integrable ξ μ) (hsq : Integrable (fun ω => ξ ω ^ 2) μ)
    (hmean : ∫ ω, ξ ω ∂μ = 0) (hvar : ∫ ω, ξ ω ^ 2 ∂μ = v) :
    ∫ ω, ((e + ξ ω) ^ 2 - v) ∂μ = e ^ 2 := by
  have hrw : (fun ω => (e + ξ ω) ^ 2 - v)
      = fun ω => (e ^ 2 - v) + (2 * e * ξ ω + ξ ω ^ 2) := by
    funext ω; ring
  have hmix : Integrable (fun ω => 2 * e * ξ ω) μ := hint.const_mul (2 * e)
  have hA : ∫ ω, (2 * e * ξ ω + ξ ω ^ 2) ∂μ
      = 2 * e * (∫ ω, ξ ω ∂μ) + ∫ ω, ξ ω ^ 2 ∂μ := by
    rw [integral_add hmix hsq, integral_const_mul]
  have hB : ∫ ω, ((e ^ 2 - v) + (2 * e * ξ ω + ξ ω ^ 2)) ∂μ
      = (e ^ 2 - v) + ∫ ω, (2 * e * ξ ω + ξ ω ^ 2) ∂μ := by
    have hsum : Integrable (fun ω => 2 * e * ξ ω + ξ ω ^ 2) μ := hmix.add hsq
    rw [integral_add (integrable_const (e ^ 2 - v)) hsum, integral_const]
    simp
  rw [hrw, hB, hA, hmean, hvar]
  ring

/-! ## 5. Why the optimal affine plane is centred at the mean -/

/-- **Sum of squared distances decomposes about the mean.**  For any point `w`,
`∑ ‖v i - w‖² = ∑ ‖v i - m‖² + n ‖m - w‖²`, where `m` is the mean of the `v i`. -/
theorem sum_dist_sq_center {ι : Type*} [Fintype ι] {E : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] (v : ι → E) (w : E)
    (m : E) (hm : ∑ j, v j = (Fintype.card ι) • m) :
    ∑ i, ‖v i - w‖ ^ 2
      = (∑ i, ‖v i - m‖ ^ 2) + (Fintype.card ι : ℝ) * ‖m - w‖ ^ 2 := by
  have expand : ∀ i, ‖v i - w‖ ^ 2
      = ‖v i - m‖ ^ 2 + 2 * inner (𝕜 := ℝ) (v i - m) (m - w) + ‖m - w‖ ^ 2 := by
    intro i
    have : v i - w = (v i - m) + (m - w) := by abel
    rw [this, norm_add_sq_real]
  simp only [expand]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const,
      Finset.card_univ, nsmul_eq_mul]
  have hzero : ∑ i, inner (𝕜 := ℝ) (v i - m) (m - w) = 0 := by
    rw [← sum_inner]
    have hsum : ∑ i, (v i - m) = (0 : E) := by
      rw [Finset.sum_sub_distrib, hm, Finset.sum_const, Finset.card_univ]
      abel
    rw [hsum, inner_zero_left]
  rw [← Finset.mul_sum, hzero]
  ring

/-! ## 6. The Taylor patch remainder, instantiated -/

/-- **Second-order patch error.**  Along the segment from an anchor `x₀` to a query point
`x`, if the third derivative of the restricted loss is bounded by `M₃` then the
second-order Taylor model errs by at most `M₃ |x - x₀|³ / 6`. -/
theorem taylor2_patch_bound {f : ℝ → ℝ} {x₀ x M₃ : ℝ} (hx : x₀ ≠ x)
    (hf : ContDiffOn ℝ 2 f (Set.uIcc x₀ x))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin 2 f (Set.uIcc x₀ x)) (Set.uIoo x₀ x))
    (hM : ∀ y ∈ Set.uIoo x₀ x, |iteratedDerivWithin 3 f (Set.uIcc x₀ x) y| ≤ M₃) :
    |f x - taylorWithinEval f 2 (Set.uIcc x₀ x) x₀ x| ≤ M₃ * |x - x₀| ^ 3 / 6 := by
  obtain ⟨c, hc, hEq⟩ := taylor_mean_remainder_lagrange hx hf hf'
  simp only [show (2 : ℕ) + 1 = 3 from rfl] at hEq
  have hfac : ((Nat.factorial 3 : ℕ) : ℝ) = 6 := by norm_num [Nat.factorial]
  rw [hEq, hfac, abs_div, abs_mul, abs_pow,
      abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 6)]
  have hnn : (0 : ℝ) ≤ |x - x₀| ^ 3 := by positivity
  have := mul_le_mul_of_nonneg_right (hM c hc) hnn
  linarith

end Atlas
