import unittest

import numpy as np

from rl_experiments.frozenlake.environment import FrozenLake
from rl_experiments.frozenlake.tabular_control import q_learning, sarsa


SMALL_LAKE = [
    ["&", ".", "."],
    [".", "#", "."],
    [".", ".", "$"],
]


class TabularControlTests(unittest.TestCase):
    def test_sarsa_returns_finite_policy_and_values(self):
        env = FrozenLake(SMALL_LAKE, slip=0.0, max_steps=20, seed=2)
        policy, value = sarsa(env, 100, eta=0.5, gamma=0.9, epsilon=0.5, seed=2)
        self.assertEqual(policy.shape, (env.n_states,))
        self.assertTrue(np.isfinite(value).all())

    def test_q_learning_returns_finite_policy_and_values(self):
        env = FrozenLake(SMALL_LAKE, slip=0.0, max_steps=20, seed=3)
        policy, value = q_learning(env, 100, eta=0.5, gamma=0.9, epsilon=0.5, seed=3)
        self.assertEqual(policy.shape, (env.n_states,))
        self.assertTrue(np.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
