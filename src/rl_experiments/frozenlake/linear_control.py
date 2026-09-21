"""Linear SARSA and Q-learning with one-hot state-action features."""

from __future__ import annotations

import numpy as np


class LinearWrapper:
    """Represent each state-action pair with a one-hot feature vector."""

    def __init__(self, env):
        self.env = env
        self.n_actions = env.n_actions
        self.n_states = env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, state: int) -> np.ndarray:
        features = np.zeros((self.n_actions, self.n_features), dtype=float)
        for action in range(self.n_actions):
            index = np.ravel_multi_index((state, action), (self.n_states, self.n_actions))
            features[action, index] = 1.0
        return features

    def reset(self) -> np.ndarray:
        return self.encode_state(self.env.reset())

    def step(self, action: int):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    def decode_policy(self, theta: np.ndarray):
        policy = np.zeros(self.n_states, dtype=int)
        value = np.zeros(self.n_states, dtype=float)
        for state in range(self.n_states):
            q_values = self.encode_state(state) @ theta
            policy[state] = int(np.argmax(q_values))
            value[state] = float(np.max(q_values))
        return policy, value


def _choose(q_values: np.ndarray, epsilon: float, rng: np.random.RandomState) -> int:
    if rng.rand() < epsilon:
        return int(rng.randint(len(q_values)))
    return int(rng.choice(np.flatnonzero(q_values == q_values.max())))


def linear_sarsa(env: LinearWrapper, max_episodes: int, eta: float, gamma: float, epsilon: float, seed=None):
    rng = np.random.RandomState(seed)
    learning_rates = np.maximum(0.1, np.linspace(eta, 0.0, max_episodes))
    epsilons = np.maximum(0.2, np.linspace(epsilon, 0.0, max_episodes))
    theta = np.full(env.n_features, 0.5, dtype=float)

    for episode in range(max_episodes):
        features = env.reset()
        q_values = features @ theta
        action = _choose(q_values, epsilons[episode], rng)
        done = False

        while not done:
            next_features, reward, done = env.step(action)
            next_q = next_features @ theta
            if done:
                target = reward
                next_action = 0
            else:
                next_action = _choose(next_q, epsilons[episode], rng)
                target = reward + gamma * next_q[next_action]

            td_error = np.clip(target - q_values[action], -1.0, 1.0)
            theta += learning_rates[episode] * td_error * features[action]
            features = next_features
            q_values = next_q
            action = next_action

    return theta


def linear_q_learning(env: LinearWrapper, max_episodes: int, eta: float, gamma: float, epsilon: float, seed=None):
    rng = np.random.RandomState(seed)
    learning_rates = np.maximum(0.05, np.linspace(eta, 0.0, max_episodes))
    epsilons = np.maximum(0.1, np.linspace(epsilon, 0.0, max_episodes))
    theta = np.full(env.n_features, 0.5, dtype=float)

    for episode in range(max_episodes):
        features = env.reset()
        done = False
        while not done:
            q_values = features @ theta
            action = _choose(q_values, epsilons[episode], rng)
            next_features, reward, done = env.step(action)
            next_q = next_features @ theta
            target = reward if done else reward + gamma * np.max(next_q)
            td_error = np.clip(target - q_values[action], -1.0, 1.0)
            theta += learning_rates[episode] * td_error * features[action]
            features = next_features

    return theta
