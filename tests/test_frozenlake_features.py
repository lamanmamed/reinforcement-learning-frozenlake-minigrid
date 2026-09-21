import unittest

import numpy as np

from rl_experiments.frozenlake.dqn import FrozenLakeImageWrapper
from rl_experiments.frozenlake.environment import FrozenLake
from rl_experiments.frozenlake.linear_control import LinearWrapper


LAKE = [
    ["&", "."],
    ["#", "$"],
]


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.env = FrozenLake(LAKE, slip=0.0, max_steps=8, seed=0)

    def test_linear_features_are_one_hot_per_action(self):
        wrapped = LinearWrapper(self.env)
        features = wrapped.encode_state(0)
        self.assertEqual(features.shape, (4, self.env.n_states * self.env.n_actions))
        np.testing.assert_array_equal(features.sum(axis=1), np.ones(4))

    def test_image_wrapper_has_four_channels(self):
        wrapped = FrozenLakeImageWrapper(self.env)
        image = wrapped.encode_state(0)
        self.assertEqual(image.shape, (4, 2, 2))
        self.assertEqual(image[0].sum(), 1.0)
        self.assertEqual(image[1].sum(), 1.0)
        self.assertEqual(image[2].sum(), 1.0)
        self.assertEqual(image[3].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
