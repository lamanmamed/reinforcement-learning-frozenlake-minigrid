"""Policy Iteration and Value Iteration for finite FrozenLake models."""

from __future__ import annotations

import numpy as np


def _action_value(env, value: np.ndarray, state: int, action: int, gamma: float) -> float:
    total = 0.0
    for next_state in range(env.n_states):
        probability = env.p(next_state, state, action)
        if probability == 0:
            continue
        reward = env.r(next_state, state, action)
        continuation = 0.0 if next_state == env.absorbing_state else gamma * value[next_state]
        total += probability * (reward + continuation)
    return total


def policy_evaluation(
    env,
    policy: np.ndarray,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, int]:
    """Evaluate a fixed policy with iterative Bellman expectation updates."""
    value = np.zeros(env.n_states, dtype=float)

    for sweep in range(1, max_iterations + 1):
        delta = 0.0
        for state in range(env.n_states):
            old_value = value[state]
            if state == env.absorbing_state:
                value[state] = 0.0
            else:
                value[state] = _action_value(env, value, state, int(policy[state]), gamma)
            delta = max(delta, abs(old_value - value[state]))
        if delta < theta:
            return value, sweep

    return value, max_iterations


def policy_iteration(
    env,
    gamma: float = 0.9,
    theta: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return value, policy, improvement rounds, and total evaluation sweeps."""
    policy = np.zeros(env.n_states, dtype=int)
    total_evaluation_sweeps = 0
    improvement_rounds = 0

    while True:
        improvement_rounds += 1
        value, sweeps = policy_evaluation(env, policy, gamma, theta)
        total_evaluation_sweeps += sweeps
        stable = True

        for state in range(env.n_states):
            if state == env.absorbing_state:
                continue
            old_action = int(policy[state])
            q_values = [
                _action_value(env, value, state, action, gamma)
                for action in range(env.n_actions)
            ]
            policy[state] = int(np.argmax(q_values))
            stable &= int(policy[state]) == old_action

        if stable:
            return value, policy, improvement_rounds, total_evaluation_sweeps


def value_iteration(
    env,
    gamma: float = 0.9,
    theta: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run Bellman optimality updates until the value function converges."""
    value = np.zeros(env.n_states, dtype=float)

    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for state in range(env.n_states):
            if state == env.absorbing_state:
                value[state] = 0.0
                continue
            old_value = value[state]
            q_values = [
                _action_value(env, value, state, action, gamma)
                for action in range(env.n_actions)
            ]
            value[state] = max(q_values)
            delta = max(delta, abs(old_value - value[state]))
        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for state in range(env.n_states):
        if state == env.absorbing_state:
            continue
        q_values = [
            _action_value(env, value, state, action, gamma)
            for action in range(env.n_actions)
        ]
        policy[state] = int(np.argmax(q_values))

    return value, policy, sweeps
