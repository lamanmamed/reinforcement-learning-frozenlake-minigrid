import unittest

from rl_experiments.minigrid.reward_shaping import ShapingConfig, ShapingState, compute_shaping_bonus


class RewardShapingTests(unittest.TestCase):
    def test_key_door_and_target_rewards_are_one_time_events(self):
        state = ShapingState()
        config = ShapingConfig(progress_reward=0.0)

        self.assertAlmostEqual(
            compute_shaping_bonus(state, carrying_type="key", any_door_open=False, target_distance=None, config=config),
            0.10,
        )
        self.assertAlmostEqual(
            compute_shaping_bonus(state, carrying_type="key", any_door_open=False, target_distance=None, config=config),
            0.0,
        )
        self.assertAlmostEqual(
            compute_shaping_bonus(state, carrying_type="key", any_door_open=True, target_distance=3, config=config),
            0.20,
        )
        self.assertAlmostEqual(
            compute_shaping_bonus(state, carrying_type="box", any_door_open=True, target_distance=0, config=config),
            0.30,
        )

    def test_progress_reward_decreases_with_distance(self):
        near_state = ShapingState(door_opened=True)
        far_state = ShapingState(door_opened=True)
        config = ShapingConfig(progress_reward=0.002)
        near = compute_shaping_bonus(
            near_state, carrying_type=None, any_door_open=True, target_distance=1, config=config
        )
        far = compute_shaping_bonus(
            far_state, carrying_type=None, any_door_open=True, target_distance=5, config=config
        )
        self.assertGreater(near, far)


if __name__ == "__main__":
    unittest.main()
