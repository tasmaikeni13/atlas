"""Checks the offline HPO task has a reproducible, recoverable signal."""

import unittest

import numpy as np

from atlas.synthetic import ColorPatternDataset


class ColorPatternDatasetTests(unittest.TestCase):
    def test_labels_are_balanced_and_recoverable(self):
        first = ColorPatternDataset(img_size=64, seed=7)
        second = ColorPatternDataset(img_size=64, seed=7)
        images, labels = first.generate_batch(20)
        replay_images, replay_labels = second.generate_batch(20)

        np.testing.assert_array_equal(images, replay_images)
        np.testing.assert_array_equal(labels, replay_labels)
        self.assertEqual(np.bincount(np.asarray(labels)).tolist(), [10, 10])

        channel_means = np.asarray(images).mean(axis=(1, 2))
        predictions = (channel_means[:, 2] > channel_means[:, 0]).astype(int)
        np.testing.assert_array_equal(predictions, labels)

    def test_rejects_unbalanced_batch_size(self):
        with self.assertRaises(ValueError):
            ColorPatternDataset().generate_batch(3)


if __name__ == "__main__":
    unittest.main()
