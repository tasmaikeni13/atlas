"""ATLAS: Adaptive Taylor Landscape Analysis System.

Budget-optimal, certified loss landscape diagnostics and visualization
for deep neural networks on high-performance accelerators (Google Cloud TPUs & GPUs).
"""

from .api import AtlasRecorder, RenderReport
from .device import get_tpu_device, setup_tpu_runtime
from .models import VisionTransformer, CausalTransformer
from .basis import SubspaceBasis, trajectory_pca, gradient_subspace, filter_normalized_random
from .probe import JetProbe, Jet
from .design import BudgetAllocator, OptimalAllocation
from .reconstruct import HermiteTaylorReconstruction
from .certify import Certificate, certify_reconstruction
from .diagnostics import LandscapeDiagnostics

__version__ = "1.0.0"
__all__ = [
    "AtlasRecorder",
    "RenderReport",
    "get_tpu_device",
    "setup_tpu_runtime",
    "VisionTransformer",
    "CausalTransformer",
    "SubspaceBasis",
    "trajectory_pca",
    "gradient_subspace",
    "filter_normalized_random",
    "JetProbe",
    "Jet",
    "BudgetAllocator",
    "OptimalAllocation",
    "HermiteTaylorReconstruction",
    "Certificate",
    "certify_reconstruction",
    "LandscapeDiagnostics",
]
