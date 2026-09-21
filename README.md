# Reinforcement Learning: FrozenLake to MiniGrid

This group project implements reinforcement learning methods across two kinds of environments. The first stage uses a custom FrozenLake environment where the full state and transition model are available. The second stage moves to MiniGrid tasks where the agent sees only part of the environment and has to learn longer action sequences involving a key, a locked door, and a target object.

The progression matters because each stage removes information that the earlier methods relied on. On FrozenLake, Policy Iteration and Value Iteration can use the transition model directly. SARSA and Q-learning learn from interaction instead. The later MiniGrid task is partially observable, so the final agent uses Recurrent PPO with an LSTM to carry information from earlier observations through the episode.

The recorded final MiniGrid evaluation was **43 successful episodes out of 50, or 86%**, on the original unshaped `MiniGrid-BlockedUnlockPickup-v0` environment. Reward shaping was used during training only.

## Project progression

```text
FrozenLake
    ↓
Policy Iteration and Value Iteration
    ↓
SARSA and Q-learning
    ↓
Linear SARSA and Linear Q-learning
    ↓
Convolutional DQN
    ↓
MiniGrid UnlockPickup
    ↓
DQN and feed-forward PPO experiments
    ↓
Recurrent PPO with LSTM
    ↓
transfer to BlockedUnlockPickup with training-time reward shaping
```

This repository is a cleaned version of a group project. It presents the shared implementation and experiments as one technical project and does not assign individual authorship to specific components.

## FrozenLake environment

The FrozenLake stage uses a custom 8×8 lake rather than Gymnasium's built-in environment.

```text
&  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .
.  .  .  #  .  .  .  .
.  .  .  .  .  #  .  .
.  .  .  #  .  .  .  .
.  #  #  .  .  .  #  .
.  #  .  .  #  .  #  .
.  .  .  #  .  .  .  $
```

`&` is the start, `#` is a hole, and `$` is the goal. There are four actions: up, left, down, and right. With `slip=0.1`, an intended movement succeeds with probability 0.9 and the agent remains in place with probability 0.1.

The implementation also has a separate absorbing state. Once the agent acts from a hole or the goal tile, the episode moves into that absorbing state.

## Policy Iteration and Value Iteration

These methods have access to the full transition model. They do not have to discover transition probabilities by trial and error.

**Policy Iteration** alternates between two operations:

1. evaluate the current policy by repeatedly applying the Bellman expectation update
2. replace each action with the action that has the highest expected return under the current value function

**Value Iteration** applies the Bellman optimality update directly and extracts the greedy policy after the values converge.

With `gamma=0.9` and `theta=0.001` on the 8×8 lake, the cleaned implementation reproduces:

| Method | Convergence count |
| --- | ---: |
| Policy Iteration | **15 policy-improvement rounds** |
| Policy evaluation inside Policy Iteration | **194 total sweeps** |
| Value Iteration | **21 sweeps** |

Policy Iteration and Value Iteration return the same final policy and value function.

The original coursework notes described the 194 policy-evaluation sweeps as 194 policy-improvement iterations. Those are different quantities. The repository keeps them separate so the result is reported correctly.

## Learning without the transition model

The next stage removes direct access to the optimal Bellman backup and learns action values from episodes.

### SARSA

SARSA is on-policy. Its update uses the next action that the current epsilon-greedy policy actually selects:

```text
Q(s, a) ← Q(s, a) + α [r + γQ(s', a') - Q(s, a)]
```

The implementation keeps a minimum amount of exploration and a non-zero learning rate rather than decaying both all the way to zero.

### Q-learning

Q-learning is off-policy. Its target uses the best next action according to the current action-value table:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

Both implementations use optimistic initial action values and random tie-breaking between equally valued actions.

## Linear value approximation

The project then rewrites SARSA and Q-learning using a parameter vector `theta` and state-action features.

Each state-action pair is represented by a one-hot feature vector. The action value is therefore:

```text
Q(s, a) = phi(s, a) · theta
```

Because the features are one-hot, different states do not share parameters. This stage demonstrates the mechanics of linear function approximation, but it does not yet generalize between similar states the way a richer feature representation would.

## Convolutional DQN on FrozenLake

For the deep-learning stage, each FrozenLake state is converted into a four-channel image:

```text
channel 0: agent position
channel 1: start tile
channel 2: hole tiles
channel 3: goal tile
```

A small convolutional network maps that image to four Q-values, one for each action. Training uses an experience replay buffer and a separate target network.

This connects the earlier Q-learning update to a setting where the action-value function is produced by a neural network rather than stored in a table.

## Why MiniGrid is harder

The second part of the project moves to `MiniGrid-UnlockPickup-v0` and then `MiniGrid-BlockedUnlockPickup-v0`.

These tasks require a sequence of dependent actions. The agent may need to find and pick up a key, return to a locked door, open it, navigate around an obstacle, and pick up the target object.

The agent is also **partially observable**. It does not receive a complete view of the environment at every step. A feed-forward policy only receives the current observation, so information that disappeared from view is not directly available at the next decision.

That creates a reason to use memory rather than simply making the feed-forward network larger.

## MiniGrid DQN experiment

One exploratory approach used an image-based DQN on `UnlockPickup` with:

- an online convolutional Q-network
- a separate target network
- experience replay
- epsilon-greedy exploration from 1.0 down to 0.05
- Huber loss
- gradient clipping
- a one-time `+0.2` reward for picking up the key

The target selects the next action with the online network and evaluates that action with the target network, which follows the Double-DQN idea.

This experiment is retained under `experiments/` because it was part of the search for a method that could deal with the task. It is not presented as the final MiniGrid solution.

## Feed-forward PPO experiments

Feed-forward PPO was also tested on `UnlockPickup` with several changes:

- higher entropy to encourage exploration
- longer rollout windows
- longer training budgets
- full-observation ablations
- key-pickup reward shaping

Those experiments did not produce reliable task completion in the saved project material. Their main role in the repository is to show why the later pipeline switched to a recurrent policy.

## Recurrent PPO with an LSTM

The final MiniGrid pipeline uses `RecurrentPPO` from `sb3-contrib`.

An LSTM carries a hidden state from one timestep to the next. This means the policy can base the current action on both the current observation and information encoded from earlier observations. In a partially observable task, that memory can preserve information about objects or locations that are no longer visible.

The BlockedUnlockPickup training script starts from a Recurrent PPO checkpoint trained on the simpler `UnlockPickup` task and continues training on `BlockedUnlockPickup`.

The saved configuration uses:

```text
training timesteps:     250,000
parallel environments:  4
learning rate:          1e-4
entropy coefficient:    0.003
evaluation episodes:    50
evaluation frequency:   every 25,000 steps
checkpoint frequency:   every 25,000 steps
```

## Reward shaping during training

BlockedUnlockPickup has sparse rewards, so the training environment adds small intermediate rewards for events that are useful for completing the task.

The cleaned implementation preserves the original shaping values:

| Event | Extra reward |
| --- | ---: |
| First key pickup | `+0.10` |
| First door opening | `+0.20` |
| First target pickup | `+0.30` |
| After the door is open, before target pickup | `0.002 / (distance + 1)` |

The progress term is larger when the agent is closer to the target object.

The important evaluation detail is that **reward shaping is not used for the final test episodes**. The recorded 43/50 result comes from deterministic evaluation on the original unshaped BlockedUnlockPickup environment.

## Recorded BlockedUnlockPickup result

The submitted project records:

| Evaluation | Result |
| --- | ---: |
| Environment | `MiniGrid-BlockedUnlockPickup-v0` |
| Policy | Recurrent PPO with LSTM |
| Evaluation mode | deterministic |
| Episodes | 50 |
| Successful episodes | **43** |
| Success rate | **86%** |
| Reward shaping during evaluation | **none** |

This result has not been rerun from this cleaned repository because the archive does not contain the final BlockedUnlockPickup checkpoint.

The large `best_model.zip` file in the original archive is the **UnlockPickup checkpoint used as the starting point for transfer learning**. The BlockedUnlockPickup script loads that checkpoint and saves a later model to a different path. That later checkpoint was not included in the archive, so this repository does not relabel or publish `best_model.zip` as the final model.

## Repository structure

```text
reinforcement-learning-frozenlake-minigrid/
├── src/rl_experiments/
│   ├── frozenlake/
│   │   ├── environment.py
│   │   ├── dynamic_programming.py
│   │   ├── tabular_control.py
│   │   ├── linear_control.py
│   │   └── dqn.py
│   └── minigrid/
│       ├── environments.py
│       ├── reward_shaping.py
│       ├── recurrent_ppo.py
│       └── evaluation.py
├── experiments/
│   ├── minigrid_dqn.py
│   └── feedforward_ppo.py
├── scripts/
│   └── train_blocked_unlockpickup.py
├── examples/
│   ├── frozenlake_demo.py
│   └── minigrid_evaluate.py
├── tests/
│   ├── test_dynamic_programming.py
│   ├── test_tabular_control.py
│   ├── test_frozenlake_features.py
│   └── test_reward_shaping.py
├── pyproject.toml
└── README.md
```

The unfinished starter `policy_improvement.py`, generated logs, macOS metadata, saved training outputs, and the transfer checkpoint have been left out of the cleaned repository.

## Running the FrozenLake code

Python 3.10 or newer is recommended.

Install the base package:

```bash
pip install -e .
```

Run the model-based demo:

```bash
python examples/frozenlake_demo.py
```

Expected output includes:

```text
Policy Iteration improvement rounds: 15
Policy Iteration evaluation sweeps: 194
Value Iteration sweeps: 21
Policies match: True
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Running the MiniGrid code

The MiniGrid part has additional dependencies:

```bash
pip install -e ".[minigrid]"
```

Training BlockedUnlockPickup requires a compatible Recurrent PPO checkpoint from the earlier UnlockPickup stage:

```bash
python scripts/train_blocked_unlockpickup.py \
    path/to/unlockpickup_checkpoint.zip \
    output/blocking_run
```

If a trained BlockedUnlockPickup checkpoint is available, evaluate it on the original unshaped environment with:

```bash
python examples/minigrid_evaluate.py \
    path/to/blocked_unlockpickup_checkpoint.zip \
    --episodes 50
```

## What this project showed

The FrozenLake section makes the difference between model-based and model-free reinforcement learning concrete. Policy Iteration and Value Iteration can compute an optimal solution because the transition model is known. SARSA and Q-learning have to estimate useful action values from interaction instead.

The progression to MiniGrid changes the problem again. Sparse rewards make exploration harder, and partial observability means that the current image is not always enough to choose the right action. The unsuccessful feed-forward experiments are useful in that context because they show why memory became part of the final design.

The transfer stage also shows the difference between training assistance and evaluation conditions. Intermediate shaping rewards make useful events easier to learn during training, but the reported success rate is measured without those extra rewards. That keeps the evaluation tied to the original task rather than to the shaped training objective.