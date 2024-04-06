import argparse

from rl_traffic_controller import train, evaluate


parser = argparse.ArgumentParser()

parser.add_argument("mode", type=str)
parser.add_argument("image_paths", type=str, nargs="*", default=[], action="extend")
parser.add_argument("-c", "--continue", action="store_true", dest="load_nets")
parser.add_argument("-e", "--episodes", type=int, default=50)

args = parser.parse_args()

if args.mode.lower() == "train":
    train(
        checkpoints=True,
        load_nets=args.load_nets,
        num_episodes=args.episodes
    )
elif args.mode.lower() == "eval":
    evaluate(
        agent=None,
        image_paths=args.image_paths
    )
