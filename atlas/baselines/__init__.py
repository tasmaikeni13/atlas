"""Hardware-optimized baseline landscape diagnostic methods and XLA kernels for fair benchmarking."""

from .grid import VectorizedGridBaseline, fit_2d_spline_surface
from .random_slice import FilterNormalizedRandomSlice
from .hvp_lanczos import TpuLanczosHessian, TpuHutchinsonTrace, hessian_vector_product
from .finite_difference import TpuFiniteDifferenceCurvature

__all__ = [
    "VectorizedGridBaseline",
    "fit_2d_spline_surface",
    "FilterNormalizedRandomSlice",
    "TpuLanczosHessian",
    "TpuHutchinsonTrace",
    "hessian_vector_product",
    "TpuFiniteDifferenceCurvature",
]
