"""Reward shaping used for the BlockedUnlockPickup transfer stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapingConfig:
    key_reward: float = 0.10
    door_reward: float = 0.20
    target_reward: float = 0.30
    progress_reward: float = 0.002
    step_penalty: float = 0.0
    target_type: str = "box"


@dataclass
class ShapingState:
    had_key: bool = False
    door_opened: bool = False
    picked_target: bool = False


def compute_shaping_bonus(
    state: ShapingState,
    *,
    carrying_type: str | None,
    any_door_open: bool,
    target_distance: int | None,
    config: ShapingConfig = ShapingConfig(),
) -> float:
    """Update event state and return the extra reward for one environment step."""
    bonus = config.step_penalty

    if carrying_type == "key" and not state.had_key:
        bonus += config.key_reward
        state.had_key = True

    if any_door_open and not state.door_opened:
        bonus += config.door_reward
        state.door_opened = True

    if carrying_type == config.target_type and not state.picked_target:
        bonus += config.target_reward
        state.picked_target = True

    if state.door_opened and not state.picked_target and target_distance is not None:
        bonus += config.progress_reward / (target_distance + 1)

    return bonus


def wrap_blocked_unlock_env(env, config: ShapingConfig = ShapingConfig()):
    """Wrap a MiniGrid environment with the training-time shaping rewards.

    Gymnasium is imported only when this function is called so the FrozenLake
    part of the repository can run without MiniGrid dependencies installed.
    """
    try:
        from gymnasium import Wrapper
    except ImportError as exc:
        raise ImportError("Install the 'minigrid' optional dependencies to use this wrapper.") from exc

    class RewardShapingWrapper(Wrapper):
        def __init__(self, wrapped_env):
            super().__init__(wrapped_env)
            self.shaping_state = ShapingState()

        def reset(self, **kwargs):
            self.shaping_state = ShapingState()
            return self.env.reset(**kwargs)

        def _door_is_open(self) -> bool:
            grid_env = self.env.unwrapped
            for x in range(grid_env.width):
                for y in range(grid_env.height):
                    obj = grid_env.grid.get(x, y)
                    if obj is not None and obj.type == "door" and obj.is_open:
                        return True
            return False

        def _target_distance(self) -> int | None:
            grid_env = self.env.unwrapped
            agent_x, agent_y = tuple(grid_env.agent_pos)
            for x in range(grid_env.width):
                for y in range(grid_env.height):
                    obj = grid_env.grid.get(x, y)
                    if obj is not None and obj.type == config.target_type:
                        return abs(agent_x - x) + abs(agent_y - y)
            return None

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            carrying = self.env.unwrapped.carrying
            carrying_type = carrying.type if carrying is not None else None
            reward += compute_shaping_bonus(
                self.shaping_state,
                carrying_type=carrying_type,
                any_door_open=self._door_is_open(),
                target_distance=self._target_distance(),
                config=config,
            )
            return obs, reward, terminated, truncated, info

    return RewardShapingWrapper(env)
