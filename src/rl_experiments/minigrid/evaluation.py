"""Deterministic evaluation helpers for recurrent MiniGrid policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    successes: int
    episodes: int
    mean_reward: float
    mean_length: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes


def evaluate_recurrent_model(model_path: str, episodes: int = 50, seed: int = 0) -> EvaluationResult:
    """Evaluate a saved RecurrentPPO model on the original unshaped task."""
    try:
        import numpy as np
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise ImportError("Install this project with the 'minigrid' extra.") from exc

    from .environments import make_blocked_unlock_env

    env = make_blocked_unlock_env(seed=seed, shaped=False)
    model = RecurrentPPO.load(model_path, env=env)
    rewards = []
    lengths = []
    successes = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_start = np.array([done], dtype=bool)
            total_reward += float(reward)
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        successes += int(total_reward > 0)

    env.close()
    return EvaluationResult(
        successes=successes,
        episodes=episodes,
        mean_reward=float(np.mean(rewards)),
        mean_length=float(np.mean(lengths)),
    )
