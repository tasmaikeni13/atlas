"""Budget feasibility and optimality for the integer allocation surrogate."""

import unittest

from atlas.design import BudgetAllocator
from atlas.probe import CostModel


class BudgetAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.cost = CostModel(
            tau=0.01, kappa=0.001, sigma2=0.04, m3=12.0, loss_relief=1.0
        )

    def test_fails_when_minimum_plan_is_unaffordable(self):
        allocator = BudgetAllocator(self.cost)
        with self.assertRaisesRegex(ValueError, "No allocation"):
            allocator.solve(0.1, min_batch=16, max_batch=32,
                            min_anchors=8, max_anchors=12)

    def test_matches_brute_force_feasible_minimum(self):
        allocator = BudgetAllocator(self.cost, cert_fraction=0.25)
        plan = allocator.solve(0.6, radius=1.0, min_batch=4,
                               max_batch=16, min_anchors=8, max_anchors=12)
        self.assertEqual(plan.n_est + plan.n_cert, plan.n_total)
        self.assertLessEqual(
            plan.n_total * (self.cost.tau + self.cost.kappa * plan.batch_size),
            plan.budget_seconds,
        )

        errors = []
        for n_total in range(8, 13):
            n_cert = max(round(n_total * 0.25), 4)
            n_est = n_total - n_cert
            if n_est < 4:
                continue
            for batch in range(4, 17):
                if n_total * (self.cost.tau + self.cost.kappa * batch) <= 0.6:
                    errors.append(
                        2.0 * n_est ** (-1.5) + 0.2 * batch ** (-0.5)
                    )
        self.assertAlmostEqual(plan.expected_error, min(errors))


if __name__ == "__main__":
    unittest.main()
