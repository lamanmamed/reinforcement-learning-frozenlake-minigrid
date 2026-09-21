from rl_experiments.frozenlake import BIG_LAKE, FrozenLake, policy_iteration, value_iteration


env = FrozenLake(BIG_LAKE, slip=0.1, max_steps=16, seed=0)
_, policy_pi, improvement_rounds, evaluation_sweeps = policy_iteration(
    env,
    gamma=0.9,
    theta=0.001,
)
_, policy_vi, value_sweeps = value_iteration(env, gamma=0.9, theta=0.001)

print(f"Policy Iteration improvement rounds: {improvement_rounds}")
print(f"Policy Iteration evaluation sweeps: {evaluation_sweeps}")
print(f"Value Iteration sweeps: {value_sweeps}")
print(f"Policies match: {(policy_pi == policy_vi).all()}")
