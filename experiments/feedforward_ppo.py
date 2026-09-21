"""Feed-forward PPO ablations on MiniGrid UnlockPickup.

These experiments did not solve the partially observable task reliably. They
are retained because they motivated the move to recurrent PPO.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    timesteps: int
    entropy_coefficient: float
    rollout_steps: int
    full_observation: bool = False
    shaped_key_reward: bool = False


EXPERIMENTS = [
    Experiment("higher_entropy", 200_000, 0.05, 2048),
    Experiment("longer_rollout", 200_000, 0.01, 4096),
    Experiment("combined_tuning", 500_000, 0.05, 4096),
    Experiment("full_observation", 200_000, 0.01, 2048, full_observation=True),
    Experiment("full_observation_tuned", 500_000, 0.05, 4096, full_observation=True),
    Experiment("key_reward_shaping", 200_000, 0.01, 2048, shaped_key_reward=True),
]


def run(experiment: Experiment, seed: int = 0):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise SystemExit("Install the project with the 'minigrid' extra before running PPO experiments.") from exc

    from rl_experiments.minigrid.environments import make_unlock_image_env

    set_random_seed(seed)
    env = make_unlock_image_env(
        seed=seed,
        full_observation=experiment.full_observation,
        shaped_key_reward=experiment.shaped_key_reward,
    )
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=experiment.entropy_coefficient,
        n_steps=experiment.rollout_steps,
        batch_size=64,
        verbose=1,
    )
    model.learn(total_timesteps=experiment.timesteps)
    return model


if __name__ == "__main__":
    for experiment in EXPERIMENTS:
        print(f"Running {experiment.name}")
        model = run(experiment)
        model.save(f"{experiment.name}.zip")
