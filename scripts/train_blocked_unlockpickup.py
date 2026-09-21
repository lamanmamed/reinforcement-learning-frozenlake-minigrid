import argparse

from rl_experiments.minigrid.recurrent_ppo import train_transfer_model


parser = argparse.ArgumentParser()
parser.add_argument("source_checkpoint", help="RecurrentPPO checkpoint trained on UnlockPickup")
parser.add_argument("output_dir", help="Directory for BlockedUnlockPickup checkpoints and logs")
args = parser.parse_args()

final_checkpoint = train_transfer_model(args.source_checkpoint, args.output_dir)
print(f"Saved final checkpoint to {final_checkpoint}")
