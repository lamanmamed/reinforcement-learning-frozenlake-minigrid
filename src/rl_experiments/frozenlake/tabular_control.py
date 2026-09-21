"""Tabular SARSA and Q-learning for FrozenLake."""

from __future__ import annotations

import numpy as np


def _greedy_action(q_values: np.ndarray, rng: np.random.RandomState) -> int:
    candidates = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(candidates))


def sarsa(env, max_episodes: int, eta: float, gamma: float, epsilon: float, seed: int | None = None):
    """Train an on-policy SARSA agent with persistent exploration."""
    rng = np.random.RandomState(seed)
    learning_rates = np.maximum(0.05, np.linspace(eta, 0.0, max_episodes))
    epsilons = np.maximum(0.1, np.linspace(epsilon, 0.0, max_episodes))
    q = np.full((env.n_states, env.n_actions), 0.5, dtype=float)

    for episode in range(max_episodes):
        state = env.reset()
        action = rng.randint(env.n_actions) if rng.rand() < epsilons[episode] else _greedy_action(q[state], rng)
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            if done:
                target = reward
                next_action = 0
            else:
                next_action = rng.randint(env.n_actions) if rng.rand() < epsilons[episode] else _greedy_action(q[next_state], rng)
                target = reward + gamma * q[next_state, next_action]

            q[state, action] += learning_rates[episode] * (target - q[state, action])
            state, action = next_state, next_action

    policy = np.argmax(q, axis=1)
    value = np.max(q, axis=1)
    policy[env.absorbing_state] = 0
    value[env.absorbing_state] = 0.0
    return policy, value


def q_learning(env, max_episodes: int, eta: float, gamma: float, epsilon: float, seed: int | None = None):
    """Train an off-policy Q-learning agent with persistent exploration."""
    rng = np.random.RandomState(seed)
    learning_rates = np.maximum(0.1, np.linspace(eta, 0.0, max_episodes))
    epsilons = np.maximum(0.2, np.linspace(epsilon, 0.0, max_episodes))
    q = np.full((env.n_states, env.n_actions), 0.5, dtype=float)

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        while not done:
            action = rng.randint(env.n_actions) if rng.rand() < epsilons[episode] else _greedy_action(q[state], rng)
            next_state, reward, done = env.step(action)
            target = reward if done else reward + gamma * np.max(q[next_state])
            q[state, action] += learning_rates[episode] * (target - q[state, action])
            state = next_state

    policy = np.argmax(q, axis=1)
    value = np.max(q, axis=1)
    policy[env.absorbing_state] = 0
    value[env.absorbing_state] = 0.0
    return policy, value
