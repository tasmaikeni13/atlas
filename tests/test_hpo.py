"""Checks that the HPO baseline uses Optuna's ask/tell sampler."""

import importlib.util
import unittest

from atlas.baselines.hpo import OptunaTPEBaseline


@unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna is optional")
class OptunaTPEBaselineTests(unittest.TestCase):
    def test_trials_are_recorded_and_best_is_selected(self):
        sampler = OptunaTPEBaseline(n_startup_trials=2, seed=17)
        observed = []
        for loss in (0.5, 0.2, 0.8):
            config = sampler.sample_config()
            observed.append(config)
            sampler.record_result(config, loss)

        self.assertEqual(len(sampler.study.trials), 3)
        self.assertEqual(sampler.get_best()["config"], observed[1])
        self.assertAlmostEqual(sampler.get_best()["loss"], 0.2)

    def test_pending_trial_requires_matching_result(self):
        sampler = OptunaTPEBaseline(seed=17)
        config = sampler.sample_config()
        with self.assertRaises(RuntimeError):
            sampler.sample_config()
        with self.assertRaises(ValueError):
            sampler.record_result({**config, "lr": config["lr"] * 2}, 0.1)
        sampler.record_result(config, 0.1)
        self.assertEqual(len(sampler.study.trials), 1)


if __name__ == "__main__":
    unittest.main()
