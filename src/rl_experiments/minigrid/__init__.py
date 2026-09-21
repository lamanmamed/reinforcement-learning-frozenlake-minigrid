"""MiniGrid training, reward shaping, and evaluation utilities."""

from .reward_shaping import ShapingConfig, ShapingState, compute_shaping_bonus

__all__ = ["ShapingConfig", "ShapingState", "compute_shaping_bonus"]
