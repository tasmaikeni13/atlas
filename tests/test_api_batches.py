"""Checks that recorder budgets use batches available for probing."""

import unittest

import jax.numpy as jnp
import numpy as np

from atlas.api import _prepare_probe_batches


class ProbeBatchPreparationTests(unittest.TestCase):
    def test_equal_sized_inputs_yield_disjoint_calibration_shapes(self):
        source = [
            (jnp.full((4, 2), value), jnp.full((4,), value))
            for value in range(4)
        ]
        calibration, sizes, pool, available = _prepare_probe_batches(source)

        self.assertEqual(sizes, [2, 4, 8])
        self.assertEqual(available, 16)
        self.assertEqual(pool[0].shape[0], 16)
        np.testing.assert_array_equal(calibration[0][1], [0, 0])
        np.testing.assert_array_equal(calibration[1][1], [1] * 4)
        np.testing.assert_array_equal(calibration[2][1], [2] * 4 + [3] * 4)

    def test_preserves_distinct_input_shapes(self):
        source = [jnp.zeros((2, 3)), jnp.ones((4, 3))]
        calibration, sizes, pool, available = _prepare_probe_batches(source)
        self.assertEqual(sizes, [2, 4])
        self.assertEqual(available, 6)
        self.assertEqual(pool.shape, (6, 3))
        self.assertEqual(calibration[0].shape, (2, 3))

    def test_rejects_insufficient_independent_batches(self):
        with self.assertRaises(ValueError):
            _prepare_probe_batches([jnp.zeros((4, 2))] * 2)


if __name__ == "__main__":
    unittest.main()
