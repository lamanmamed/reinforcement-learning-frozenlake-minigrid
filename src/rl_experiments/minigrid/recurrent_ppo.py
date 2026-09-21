"""Transfer Recurrent PPO from UnlockPickup to BlockedUnlockPickup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .environments import make_blocked_unlock_env
from .reward_shaping import ShapingConfig


@dataclass(frozen=True)
class RecurrentPPOConfig:
    total_timesteps: int = 250_000
    n_envs: int = 4
    seed: int = 0
    learning_rate: float = 1e-4
    entropy_coefficient: float = 0.003
    eval_episodes: int = 50
    eval_frequency: int = 25_000
    checkpoint_frequency: int = 25_000


def train_transfer_model(
    source_checkpoint: str | Path,
    output_dir: str | Path,
    config: RecurrentPPOConfig = RecurrentPPOConfig(),
    shaping: ShapingConfig = ShapingConfig(),
) -> Path:
    """Continue an UnlockPickup recurrent policy on BlockedUnlockPickup.

    The source checkpoint is an UnlockPickup transfer checkpoint. The final
    BlockedUnlockPickup checkpoint is written under ``output_dir``.
    """
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        raise ImportError("Install this project with the 'minigrid' extra.") from exc

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    best_dir = output_dir / "best_model"
    eval_dir = output_dir / "eval"
    tensorboard_dir = output_dir / "tensorboard"
    for directory in [checkpoint_dir, best_dir, eval_dir, tensorboard_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    def factory(rank: int, shaped: bool):
        return lambda: make_blocked_unlock_env(
            seed=config.seed + rank,
            shaped=shaped,
            shaping=shaping,
        )

    train_env = SubprocVecEnv([factory(i, True) for i in range(config.n_envs)])
    eval_env = factory(999, False)()

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=config.checkpoint_frequency,
            save_path=str(checkpoint_dir),
            name_prefix="rppo_blocked_unlockpickup",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(eval_dir),
            eval_freq=config.eval_frequency,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
        ),
    ])

    model = RecurrentPPO.load(
        str(source_checkpoint),
        env=train_env,
        seed=config.seed,
        verbose=1,
        custom_objects={
            "learning_rate": config.learning_rate,
            "ent_coef": config.entropy_coefficient,
        },
        tensorboard_log=str(tensorboard_dir),
    )

    final_path = output_dir / "rppo_blocked_unlockpickup_final"
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
    finally:
        model.save(str(final_path))
        train_env.close()
        eval_env.close()

    return final_path.with_suffix(".zip")
