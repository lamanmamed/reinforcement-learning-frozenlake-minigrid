"""Convolutional DQN used for the image-based FrozenLake experiment."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FrozenLakeImageWrapper:
    """Encode a lake state as four binary image channels."""

    def __init__(self, env):
        self.env = env
        self.n_actions = env.n_actions
        lake = env.lake
        self.state_shape = (4, lake.shape[0], lake.shape[1])
        static = [(lake == symbol).astype(float) for symbol in ["&", "#", "$"]]
        empty_agent = np.zeros(lake.shape, dtype=float)
        self.state_images = {env.absorbing_state: np.stack([empty_agent] + static)}

        for state in range(lake.size):
            agent = np.zeros(lake.shape, dtype=float)
            row, col = np.unravel_index(state, lake.shape)
            agent[row, col] = 1.0
            self.state_images[state] = np.stack([agent] + static)

    def encode_state(self, state: int) -> np.ndarray:
        return self.state_images[state]

    def reset(self) -> np.ndarray:
        return self.encode_state(self.env.reset())

    def step(self, action: int):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    def decode_policy(self, network):
        states = np.asarray([self.encode_state(s) for s in range(self.env.n_states)])
        with torch.no_grad():
            q_values = network(states).cpu().numpy()
        return np.argmax(q_values, axis=1), np.max(q_values, axis=1)


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.RandomState):
        self.buffer = deque(maxlen=capacity)
        self.rng = rng

    def append(self, transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=True)
        return [self.buffer[index] for index in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class DeepQNetwork(nn.Module):
    def __init__(self, state_shape, n_actions: int, learning_rate: float, kernel_size: int = 3, conv_channels: int = 4, hidden_features: int = 8, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        channels, height, width = state_shape
        self.conv = nn.Conv2d(channels, conv_channels, kernel_size=kernel_size)
        conv_height = height - kernel_size + 1
        conv_width = width - kernel_size + 1
        self.hidden = nn.Linear(conv_height * conv_width * conv_channels, hidden_features)
        self.output = nn.Linear(hidden_features, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        x = F.relu(self.conv(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.hidden(x))
        return self.output(x)

    def train_step(self, transitions, gamma: float, target_network) -> float:
        states = np.asarray([t[0] for t in transitions])
        actions = torch.as_tensor([t[1] for t in transitions], dtype=torch.long)
        rewards = torch.as_tensor([t[2] for t in transitions], dtype=torch.float32)
        next_states = np.asarray([t[3] for t in transitions])
        dones = torch.as_tensor([t[4] for t in transitions], dtype=torch.float32)

        selected_q = self(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = target_network(next_states).max(dim=1).values * (1.0 - dones)
            target = rewards + gamma * next_q

        loss = F.mse_loss(selected_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


def train_dqn(env, max_episodes: int = 2000, learning_rate: float = 1e-3, gamma: float = 0.9, epsilon: float = 0.2, batch_size: int = 32, target_update_frequency: int = 50, buffer_size: int = 1000, seed: int = 4):
    """Train the compact CNN DQN configuration used in the FrozenLake stage."""
    rng = np.random.RandomState(seed)
    replay = ReplayBuffer(buffer_size, rng)
    online = DeepQNetwork(env.state_shape, env.n_actions, learning_rate, seed=seed)
    target = DeepQNetwork(env.state_shape, env.n_actions, learning_rate, seed=seed)
    target.load_state_dict(online.state_dict())
    global_step = 0

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        while not done:
            global_step += 1
            if rng.rand() < epsilon:
                action = int(rng.choice(env.n_actions))
            else:
                with torch.no_grad():
                    q_values = online(np.asarray([state]))[0].numpy()
                best = np.flatnonzero(np.isclose(q_values, q_values.max()))
                action = int(rng.choice(best))

            next_state, reward, done = env.step(action)
            replay.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay) >= batch_size and global_step % 2 == 0:
                online.train_step(replay.sample(batch_size), gamma, target)

        if (episode + 1) % target_update_frequency == 0:
            target.load_state_dict(online.state_dict())

    return online
