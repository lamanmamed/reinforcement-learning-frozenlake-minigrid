import unittest

import numpy as np

from rl_experiments.frozenlake import BIG_LAKE, FrozenLake, policy_iteration, value_iteration


class DynamicProgrammingTests(unittest.TestCase):
    def setUp(self):
        self.env = FrozenLake(BIG_LAKE, slip=0.1, max_steps=16, seed=0)

    def test_policy_iteration_counts_are_reported_separately(self):
        _, _, improvement_rounds, evaluation_sweeps = policy_iteration(
            self.env, gamma=0.9, theta=0.001
        )
        self.assertEqual(improvement_rounds, 15)
        self.assertEqual(evaluation_sweeps, 194)

    def test_value_iteration_takes_21_sweeps(self):
        _, _, sweeps = value_iteration(self.env, gamma=0.9, theta=0.001)
        self.assertEqual(sweeps, 21)

    def test_model_based_methods_return_same_solution(self):
        value_pi, policy_pi, _, _ = policy_iteration(self.env, gamma=0.9, theta=0.001)
        value_vi, policy_vi, _ = value_iteration(self.env, gamma=0.9, theta=0.001)
        np.testing.assert_array_equal(policy_pi, policy_vi)
        np.testing.assert_allclose(value_pi, value_vi)


if __name__ == "__main__":
    unittest.main()
