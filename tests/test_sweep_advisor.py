"""Checks guarded directional learning-rate proposals."""

import unittest

import numpy as np

from atlas.probe import Jet
from atlas.sweep_advisor import directional_step_scale


class DirectionalStepScaleTests(unittest.TestCase):
    def test_quadratic_line_minimum_with_full_capture(self):
        jet = Jet(
            x=0.0, y=0.0, loss=5.0,
            grad=np.array([-4.0, 0.0]),
            hess=np.diag([2.0, 1.0]),
        )
        self.assertAlmostEqual(
            directional_step_scale(jet, np.array([1.0, 0.0]), 1.0),
            2.0,
        )

    def test_rejects_uphill_or_poorly_captured_direction(self):
        jet = Jet(
            x=0.0, y=0.0, loss=5.0,
            grad=np.array([1.0, 0.0]),
            hess=np.diag([2.0, 1.0]),
        )
        self.assertIsNone(
            directional_step_scale(jet, np.array([1.0, 0.0]), 1.0)
        )
        jet.grad[0] = -1.0
        self.assertIsNone(
            directional_step_scale(jet, np.array([1.0, 0.0]), 0.2)
        )


if __name__ == "__main__":
    unittest.main()
