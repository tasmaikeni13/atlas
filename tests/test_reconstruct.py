"""Check smooth blending at anchor coordinates."""

import unittest

import numpy as np

from atlas.probe import Jet
from atlas.reconstruct import HermiteTaylorReconstruction


class ReconstructionTests(unittest.TestCase):
    def test_value_is_continuous_at_anchor(self):
        jets = [
            Jet(0.0, 0.0, 0.0, np.zeros(2), np.zeros((2, 2))),
            Jet(1.0, 0.0, 1.0, np.zeros(2), np.zeros((2, 2))),
        ]
        reconstruction = HermiteTaylorReconstruction(jets, eps=0.25)
        at_anchor = reconstruction.evaluate_at(0.0, 0.0)
        near_anchor = reconstruction.evaluate_at(1e-8, 0.0)
        self.assertAlmostEqual(at_anchor, near_anchor, places=6)
        self.assertGreater(at_anchor, 0.0)


if __name__ == "__main__":
    unittest.main()
