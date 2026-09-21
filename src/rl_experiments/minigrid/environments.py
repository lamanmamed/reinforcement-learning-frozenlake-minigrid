"""Environment builders for the MiniGrid experiments."""

from __future__ import annotations

from .reward_shaping import ShapingConfig, wrap_blocked_unlock_env


def make_blocked_unlock_env(seed: int = 0, shaped: bool = False, shaping: ShapingConfig = ShapingConfig()):
    """Create the flat-observation BlockedUnlockPickup environment."""
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        from minigrid.wrappers import FlatObsWrapper
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError("Install this project with the 'minigrid' extra.") from exc

    env = gym.make("MiniGrid-BlockedUnlockPickup-v0")
    if shaped:
        env = wrap_blocked_unlock_env(env, shaping)
    env = FlatObsWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_unlock_image_env(seed: int = 0, full_observation: bool = False, shaped_key_reward: bool = False):
    """Create an image-observation UnlockPickup environment for DQN or PPO trials."""
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        from gymnasium.wrappers import ResizeObservation
        from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError("Install this project with the 'minigrid' extra.") from exc

    env = gym.make("MiniGrid-UnlockPickup-v0")
    if shaped_key_reward:
        env = _wrap_key_pickup_reward(env)
    if full_observation:
        env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = ResizeObservation(env, (84, 84))
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def _wrap_key_pickup_reward(env, bonus: float = 0.2):
    from gymnasium import Wrapper

    class KeyPickupReward(Wrapper):
        def __init__(self, wrapped):
            super().__init__(wrapped)
            self.reward_given = False

        def reset(self, **kwargs):
            self.reward_given = False
            return self.env.reset(**kwargs)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            carrying = self.env.unwrapped.carrying
            if carrying is not None and carrying.type == "key" and not self.reward_given:
                reward += bonus
                self.reward_given = True
            return obs, reward, terminated, truncated, info

    return KeyPickupReward(env)
