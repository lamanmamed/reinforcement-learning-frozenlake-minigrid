"""Exploratory Double-DQN-style training on MiniGrid UnlockPickup.

This experiment was part of the algorithm search and is not the final MiniGrid
training pipeline.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    env_id: str = "MiniGrid-UnlockPickup-v0"
    seed: int = 0
    total_steps: int = 300_000
    buffer_size: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_update_every: int = 2_000
    start_learning: int = 10_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150_000
    max_episode_steps: int = 512
    key_pickup_bonus: float = 0.2


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, observation_shape, n_actions: int):
        super().__init__()
        channels, height, width = observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.encoder(torch.zeros(1, channels, height, width)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        return self.head(self.encoder(x))


def epsilon(step: int, config: Config) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    fraction = step / config.epsilon_decay_steps
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def main(config: Config = Config()) -> None:
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        from minigrid.wrappers import ImgObsWrapper
    except ImportError as exc:
        raise SystemExit("Install the project with the 'minigrid' extra before running this experiment.") from exc

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ImgObsWrapper(gym.make(config.env_id, max_episode_steps=config.max_episode_steps))
    observation, _ = env.reset(seed=config.seed)
    height, width, channels = observation.shape
    observation_shape = (channels, height, width)
    n_actions = env.action_space.n

    online = DQN(observation_shape, n_actions).to(device)
    target = DQN(observation_shape, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.buffer_size)

    def tensorize(obs):
        return torch.from_numpy(np.transpose(obs, (2, 0, 1)).astype(np.float32))

    key_reward_given = False
    for step in range(1, config.total_steps + 1):
        eps = epsilon(step, config)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(online(tensorize(observation).unsqueeze(0).to(device)).argmax(dim=1).item())

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        carrying = env.unwrapped.carrying
        if carrying is not None and carrying.type == "key" and not key_reward_given:
            reward += config.key_pickup_bonus
            key_reward_given = True

        replay.push((observation, action, reward, next_observation, done))
        observation = next_observation

        if step >= config.start_learning and len(replay) >= config.batch_size:
            states, actions, rewards, next_states, dones = replay.sample(config.batch_size)
            state_tensor = torch.stack([tensorize(x) for x in states]).to(device)
            next_state_tensor = torch.stack([tensorize(x) for x in next_states]).to(device)
            action_tensor = torch.from_numpy(actions).long().to(device)
            reward_tensor = torch.from_numpy(rewards).to(device)
            done_tensor = torch.from_numpy(dones).to(device)

            selected_q = online(state_tensor).gather(1, action_tensor[:, None]).squeeze(1)
            with torch.no_grad():
                next_actions = online(next_state_tensor).argmax(dim=1)
                next_q = target(next_state_tensor).gather(1, next_actions[:, None]).squeeze(1)
                target_q = reward_tensor + config.gamma * next_q * (1.0 - done_tensor)

            loss = nn.functional.smooth_l1_loss(selected_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()

        if step % config.target_update_every == 0:
            target.load_state_dict(online.state_dict())

        if done:
            observation, _ = env.reset()
            key_reward_given = False

    env.close()


if __name__ == "__main__":
    main()
