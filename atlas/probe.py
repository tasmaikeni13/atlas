"""TPU-accelerated analytical Taylor jet probing."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from .basis import SubspaceBasis
from .device import flatten_params, unflatten_params

@dataclasses.dataclass
class Jet:
    """The 2nd-order Taylor jet of a scalar loss function at an affine plane coordinate."""
    x: float
    y: float
    loss: float
    grad: np.ndarray  # shape (2,)
    hess: np.ndarray  # shape (2, 2)
    variance: float = 0.0

    @property
    def hessian(self) -> np.ndarray:
        """Alias for hess (projected 2x2 Hessian matrix)."""
        return self.hess

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "loss": float(self.loss),
            "grad": [float(g) for g in self.grad],
            "hess": [[float(h) for h in row] for row in self.hess],
            "variance": float(self.variance),
        }

    def evaluate_polynomial(self, qx: float, qy: float) -> float:
        """Evaluates the 2nd-order local Taylor polynomial at query coordinate (qx, qy)."""
        dx = qx - self.x
        dy = qy - self.y
        d = np.array([dx, dy], dtype=np.float32)
        quad = 0.5 * float(d @ self.hess @ d)
        lin = float(self.grad @ d)
        return float(self.loss + lin + quad)


@dataclasses.dataclass
class CostModel:
    """Empirical cost and smoothness characteristics of the loss surface."""
    tau: float       # Base kernel/launch overhead (seconds)
    kappa: float     # Marginal cost per evaluation example (seconds/example)
    sigma2: float    # Stochastic batch variance of the loss
    m3: float        # Estimated 3rd-derivative Lipschitz bound
    loss_relief: float  # Observed surface dynamic range max(L) - min(L)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau": float(self.tau),
            "kappa": float(self.kappa),
            "sigma2": float(self.sigma2),
            "m3": float(self.m3),
            "loss_relief": float(self.loss_relief),
        }


class JetProbe:
    """Evaluates exact 2nd-order Jets on Google Cloud TPUs via forward-over-reverse autodiff."""

    def __init__(
        self,
        apply_fn: Callable[[Any, Any], jnp.ndarray],
        loss_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
        basis: SubspaceBasis
    ):
        self.apply_fn = apply_fn
        self.loss_fn = loss_fn
        self.basis = basis

        u = self.basis.u
        v = self.basis.v
        origin = self.basis.origin
        meta = self.basis.meta

        u_tree = unflatten_params(u, meta)
        v_tree = unflatten_params(v, meta)
        origin_tree = unflatten_params(origin, meta)

        def total_loss(p: Any, b: Any) -> jnp.ndarray:
            logits = apply_fn(p, b)
            return loss_fn(logits, b)

        def tree_dot(t1: Any, t2: Any) -> jnp.ndarray:
            leaves1 = jax.tree_util.tree_leaves(t1)
            leaves2 = jax.tree_util.tree_leaves(t2)
            dots = [jnp.sum(l1 * l2) for l1, l2 in zip(leaves1, leaves2)]
            return sum(dots)

        def get_params(coords: jnp.ndarray) -> Any:
            x, y = coords[0], coords[1]
            return jax.tree_util.tree_map(
                lambda p0, ul, vl: p0 + x * ul + y * vl,
                origin_tree, u_tree, v_tree
            )

        # JIT-compiled exact Jet evaluator
        @jax.jit
        def _jet_step(coords: jnp.ndarray, batch: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            p_tree = get_params(coords)
            
            # Forward-backward pass: scalar loss and full gradient
            loss, grad_tree = jax.value_and_grad(total_loss)(p_tree, batch)

            # Projected 2D gradient: <grad, u> and <grad, v>
            gx = tree_dot(grad_tree, u_tree)
            gy = tree_dot(grad_tree, v_tree)
            grad_2d = jnp.array([gx, gy], dtype=jnp.float32)

            # Exact 2D projected Hessian via two JVP passes (Pearlmutter 1994)
            def grad_fn(p):
                return jax.grad(total_loss)(p, batch)

            _, hvp_u = jax.jvp(grad_fn, (p_tree,), (u_tree,))
            _, hvp_v = jax.jvp(grad_fn, (p_tree,), (v_tree,))

            h00 = tree_dot(hvp_u, u_tree)
            h01 = tree_dot(hvp_u, v_tree)
            h11 = tree_dot(hvp_v, v_tree)
            hess_2d = jnp.array([[h00, h01], [h01, h11]], dtype=jnp.float32)

            return loss, grad_2d, hess_2d

        self._jit_jet = _jet_step

        # JIT-compiled loss-only evaluator for quick point queries
        @jax.jit
        def _loss_step(coords: jnp.ndarray, batch: Any) -> jnp.ndarray:
            p_tree = get_params(coords)
            return total_loss(p_tree, batch)

        self._jit_loss = _loss_step

    def evaluate_jet(self, x: float, y: float, batch: Any) -> Jet:
        """Evaluates exact loss, 2D gradient, and 2x2 Hessian at (x, y)."""
        coords = jnp.array([x, y], dtype=jnp.float32)
        loss, grad, hess = self._jit_jet(coords, batch)
        # Block until TPU ready
        loss_val = float(np.array(loss))
        grad_val = np.array(grad, dtype=np.float32)
        hess_val = np.array(hess, dtype=np.float32)

        return Jet(
            x=x,
            y=y,
            loss=loss_val,
            grad=grad_val,
            hess=hess_val,
            variance=0.0
        )

    def evaluate_loss(self, x: float, y: float, batch: Any) -> float:
        """Evaluates scalar loss at (x, y)."""
        coords = jnp.array([x, y], dtype=jnp.float32)
        loss = self._jit_loss(coords, batch)
        return float(np.array(loss))

    def calibrate(
        self,
        sample_batches: Sequence[Any],
        batch_sizes: Sequence[int],
        radius: float = 1.0
    ) -> CostModel:
        """Profiles TPU throughput and measures loss variance and Lipschitz constants."""
        # 1. Warm-up JIT compilation
        coords0 = jnp.array([0.0, 0.0], dtype=jnp.float32)
        _ = self._jit_jet(coords0, sample_batches[0])

        # 2. Measure execution time across batch sizes
        times = []
        for batch in sample_batches[:len(batch_sizes)]:
            t0 = time.perf_counter()
            for _ in range(5):
                _ = self._jit_jet(coords0, batch)
                jax.block_until_ready(coords0)
            t1 = time.perf_counter()
            times.append((t1 - t0) / 5.0)

        # Fit linear model: t(B) = tau + kappa * B
        B_arr = np.array(batch_sizes[:len(times)], dtype=np.float64)
        T_arr = np.array(times, dtype=np.float64)
        if len(B_arr) > 1:
            poly = np.polyfit(B_arr, T_arr, 1)
            kappa = max(float(poly[0]), 1e-8)
            tau = max(float(poly[1]), 1e-5)
        else:
            tau = float(T_arr[0] * 0.2)
            kappa = float((T_arr[0] * 0.8) / max(B_arr[0], 1))

        # 3. Measure batch variance sigma2 at origin
        losses = [self.evaluate_loss(0.0, 0.0, b) for b in sample_batches[:10]]
        sigma2 = float(np.var(losses, ddof=1)) if len(losses) > 1 else 1e-4

        # 4. Measure M3 (third derivative) along radial test points
        h = 0.1 * radius
        j_pos = self.evaluate_jet(h, 0.0, sample_batches[0])
        j_neg = self.evaluate_jet(-h, 0.0, sample_batches[0])
        j_zero = self.evaluate_jet(0.0, 0.0, sample_batches[0])
        # Second difference of Hessian approximates 3rd derivative
        diff_h = np.linalg.norm(j_pos.hess - j_neg.hess) / (2.0 * h + 1e-12)
        m3 = max(float(diff_h), 0.01)

        relief = max(max(losses) - min(losses), 0.1)

        return CostModel(
            tau=tau,
            kappa=kappa,
            sigma2=sigma2,
            m3=m3,
            loss_relief=relief
        )
