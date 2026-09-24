"""Checks the conditions under which a DKW quantile claim is valid."""

import math
import unittest

import numpy as np

from atlas.certify import certify_reconstruction
from atlas.probe import Jet


class ZeroReconstruction:
    def evaluate_batch(self, points):
        return np.zeros(len(points))


def make_jets(count):
    return [
        Jet(float(i) / count, 0.0, float(i), np.zeros(2), np.zeros((2, 2)))
        for i in range(count)
    ]


class CertificateTests(unittest.TestCase):
    def test_small_holdout_cannot_claim_95_percent_coverage(self):
        cert = certify_reconstruction(
            ZeroReconstruction(), make_jets(14), 14.0, iid_uniform_coords=True
        )
        self.assertFalse(cert.certified_valid)
        self.assertEqual(cert.q95_upper_bound, cert.max_error)
        self.assertLess(cert.coverage_lower_bound, 0.95)
        self.assertIn("not certified", cert.summary())

    def test_deterministic_coordinates_never_get_dkw_certificate(self):
        cert = certify_reconstruction(ZeroReconstruction(), make_jets(800), 800.0)
        self.assertFalse(cert.certified_valid)
        self.assertEqual(cert.coverage_lower_bound, 0.0)

    def test_valid_bound_uses_order_statistic(self):
        count = 800
        cert = certify_reconstruction(
            ZeroReconstruction(), make_jets(count), 800.0, iid_uniform_coords=True
        )
        rank = math.ceil(
            count * (0.95 + math.sqrt(math.log(40.0) / (2.0 * count)))
        ) - 1
        self.assertTrue(cert.certified_valid)
        self.assertEqual(cert.q95_upper_bound, float(rank))
        self.assertGreaterEqual(cert.coverage_lower_bound, 0.95)


if __name__ == "__main__":
    unittest.main()
