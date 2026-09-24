"""Checks cost calibration waits for work and separates spatial relief."""

import time
import unittest

import jax.numpy as jnp
import numpy as np

from atlas.probe import Jet, JetProbe


class PendingResult:
    def __init__(self, counters, delay):
        self.counters = counters
        self.delay = delay

    def block_until_ready(self):
        self.counters["blocked"] += 1
        time.sleep(self.delay)
        return self


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.probe = object.__new__(JetProbe)
        self.counters = {"launched": 0, "blocked": 0}

        def fake_jet(coords, batch):
            size = batch.shape[0]
            self.counters["launched"] += 1
            delay = 0.001 + 0.0002 * size
            return (
                PendingResult(self.counters, delay),
                PendingResult(self.counters, 0),
                PendingResult(self.counters, 0),
            )

        def fake_eval_jet(x, y, batch):
            return Jet(
                x=x, y=y, loss=(x - 1.0) ** 2,
                grad=np.zeros(2),
                hess=np.diag([2.0 + 6.0 * x, 1.0]),
            )

        losses = {4: 1.0, 8: 2.0, 12: 4.0}
        self.probe._jit_jet = fake_jet
        self.probe.evaluate_loss = lambda x, y, batch: losses[batch.shape[0]]
        self.probe.evaluate_jet = fake_eval_jet
        self.batches = [jnp.zeros((size,)) for size in (4, 8, 12)]

    def test_calibration_times_completed_jets_and_uses_fixed_batch_relief(self):
        cost = self.probe.calibrate(self.batches, [4, 8, 12])

        self.assertEqual(self.counters["launched"], 18)
        self.assertEqual(self.counters["blocked"], 54)
        self.assertGreater(cost.kappa, 5e-5)
        self.assertAlmostEqual(cost.m3, 6.0, places=5)
        self.assertAlmostEqual(cost.loss_relief, 0.4, places=5)
        self.assertAlmostEqual(cost.sigma2, 17.6666666667, places=5)

    def test_rejects_mismatched_batch_metadata(self):
        with self.assertRaises(ValueError):
            self.probe.calibrate(self.batches, [4, 8])
        with self.assertRaises(ValueError):
            self.probe.calibrate(self.batches, [4, 8, 11])

    def test_rejects_unresolved_marginal_cost(self):
        def decreasing_cost_jet(coords, batch):
            delay = 0.006 - 0.0003 * batch.shape[0]
            return (
                PendingResult(self.counters, delay),
                PendingResult(self.counters, 0),
                PendingResult(self.counters, 0),
            )

        self.probe._jit_jet = decreasing_cost_jet
        with self.assertRaisesRegex(ValueError, "Marginal batch cost"):
            self.probe.calibrate(self.batches, [4, 8, 12])


if __name__ == "__main__":
    unittest.main()
