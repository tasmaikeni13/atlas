"""Checks that batch advice uses measured projected-gradient noise."""

import unittest

import numpy as np

from atlas.probe import CostModel, Jet
from atlas.sweep_advisor import LandscapeDiagnosticEngine


class SweepNoiseTests(unittest.TestCase):
    def setUp(self):
        self.engine = LandscapeDiagnosticEngine(
            probe=None, current_lr=0.5, batch_size=4, max_lr=1.0
        )
        self.jet = Jet(
            x=0.0, y=0.0, loss=1.0,
            grad=np.array([1.0, 0.0]),
            hess=np.diag([1.0, 2.0]),
        )

    def cost(self, gradient_sigma2=None):
        return CostModel(
            tau=0.001, kappa=0.001, sigma2=1000.0,
            m3=1.0, loss_relief=1.0,
            gradient_sigma2=gradient_sigma2,
        )

    def test_loss_noise_alone_does_not_drive_batch_advice(self):
        result = self.engine.analyze_jet(self.jet, self.cost())
        self.assertIsNone(result.stochastic_snr)
        self.assertIsNone(result.to_dict()["stochastic_snr"])
        self.assertEqual(result.recommended_batch_scale, 1.0)

    def test_projected_gradient_noise_drives_batch_advice(self):
        high_snr = self.engine.analyze_jet(self.jet, self.cost(0.01))
        low_snr = self.engine.analyze_jet(self.jet, self.cost(100.0))
        self.assertAlmostEqual(high_snr.stochastic_snr, 400.0)
        self.assertEqual(high_snr.recommended_batch_scale, 0.5)
        self.assertAlmostEqual(low_snr.stochastic_snr, 0.04)
        self.assertEqual(low_snr.recommended_batch_scale, 2.0)


if __name__ == "__main__":
    unittest.main()
