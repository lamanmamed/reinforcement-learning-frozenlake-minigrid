import argparse

from rl_experiments.minigrid.evaluation import evaluate_recurrent_model


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to a BlockedUnlockPickup RecurrentPPO checkpoint")
parser.add_argument("--episodes", type=int, default=50)
args = parser.parse_args()

result = evaluate_recurrent_model(args.checkpoint, episodes=args.episodes)
print(f"Successes: {result.successes}/{result.episodes}")
print(f"Success rate: {result.success_rate:.1%}")
print(f"Mean reward: {result.mean_reward:.3f}")
print(f"Mean episode length: {result.mean_length:.1f}")
