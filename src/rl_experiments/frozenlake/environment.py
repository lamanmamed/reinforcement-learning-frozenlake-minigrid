"""Small FrozenLake environment used by the original experiments."""

from __future__ import annotations

import numpy as np


BIG_LAKE = [
    ["&", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", "#", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", "#", ".", "."],
    [".", ".", ".", "#", ".", ".", ".", "."],
    [".", "#", "#", ".", ".", ".", "#", "."],
    [".", "#", ".", ".", "#", ".", "#", "."],
    [".", ".", ".", "#", ".", ".", ".", "$"],
]


class FrozenLake:
    """Finite-state lake with stochastic action failure and an absorbing state.

    Symbols follow the original project: ``&`` is the start, ``#`` is a hole,
    ``$`` is the goal, and ``.`` is a normal tile. With probability ``slip``
    the agent remains in its current state instead of moving.
    """

    ACTIONS = ((-1, 0), (0, -1), (1, 0), (0, 1))

    def __init__(self, lake, slip: float = 0.1, max_steps: int = 16, seed: int | None = None):
        self.lake = np.asarray(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = float(slip)
        self.max_steps = int(max_steps)
        self.n_actions = 4
        self.n_states = self.lake.size + 1
        self.absorbing_state = self.n_states - 1
        self.random_state = np.random.RandomState(seed)

        self.initial_distribution = np.zeros(self.n_states, dtype=float)
        starts = np.where(self.lake_flat == "&")[0]
        if len(starts) != 1:
            raise ValueError("Lake must contain exactly one '&' start tile.")
        self.initial_distribution[starts[0]] = 1.0

        self.state = int(starts[0])
        self.n_steps = 0

    def p(self, next_state: int, state: int, action: int) -> float:
        """Return the transition probability P(next_state | state, action)."""
        if state == self.absorbing_state:
            return 1.0 if next_state == self.absorbing_state else 0.0

        row, col = divmod(state, self.lake.shape[1])
        tile = self.lake[row, col]
        if tile in {"$", "#"}:
            return 1.0 if next_state == self.absorbing_state else 0.0

        dr, dc = self.ACTIONS[action]
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < self.lake.shape[0] and 0 <= new_col < self.lake.shape[1]:
            target = new_row * self.lake.shape[1] + new_col
        else:
            target = state

        probability = 0.0
        if next_state == target:
            probability += 1.0 - self.slip
        if next_state == state:
            probability += self.slip
        return probability

    def r(self, next_state: int, state: int, action: int) -> float:
        """Return the reward for a transition.

        The submitted environment gives reward 1 when the agent acts while on
        the goal tile. Hole transitions have reward 0 and then enter the
        absorbing state.
        """
        if state < self.lake.size:
            row, col = divmod(state, self.lake.shape[1])
            if self.lake[row, col] == "$":
                return 1.0
        return 0.0

    def draw(self, state: int, action: int) -> tuple[int, float]:
        probabilities = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = int(self.random_state.choice(self.n_states, p=probabilities))
        reward = self.r(next_state, state, action)

        if state < self.lake.size:
            row, col = divmod(state, self.lake.shape[1])
            if self.lake[row, col] in {"$", "#"}:
                next_state = self.absorbing_state
        return next_state, reward

    def reset(self) -> int:
        self.n_steps = 0
        self.state = int(self.random_state.choice(self.n_states, p=self.initial_distribution))
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be in [0, {self.n_actions - 1}].")
        self.n_steps += 1
        self.state, reward = self.draw(self.state, action)
        done = reward in {-1, 1} or self.state == self.absorbing_state or self.n_steps >= self.max_steps
        return self.state, reward, done
