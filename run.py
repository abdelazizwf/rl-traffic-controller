import argparse
import logging

from rl_traffic_controller.main import demo, evaluate, train
from rl_traffic_controller.networks import stacks

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "mode", type=str, help="train or eval or demo or dry-run"
)
parser.add_argument(
    "stack", type=str, help="ID of the network architecture to use", choices=list(stacks.keys())
)
parser.add_argument(
    "image_paths", type=str, nargs="*", default=[], action="extend",
    help="paths of images (observations), and/or directories containing images, to test the agent on"
)
parser.add_argument(
    "-c", "--continue", action="store_true", dest="load_nets",
    help="load the saved network and continue training"
)
parser.add_argument(
    "-s", "--save", action="store_true", dest="save",
    help="save the network after every training episode"
)
parser.add_argument(
    "-e", "--episodes", type=int, default=50, metavar="N",
    help="number of episodes sampled during training (default: %(default)s)"
)

args = parser.parse_args()

if args.mode.lower() == "train":
    train(
        stack_name=args.stack,
        load_nets=args.load_nets,
        save=args.save,
        num_episodes=args.episodes,
        image_paths=args.image_paths
    )
if args.mode.lower() == "dry-run":
    train(
        stack_name=args.stack,
        stub=True,
        load_nets=args.load_nets,
        save=args.save,
        num_episodes=args.episodes,
        image_paths=args.image_paths
    )
elif args.mode.lower() == "eval":
    evaluate(
        stack_name=args.stack,
        agent=None,
        image_paths=args.image_paths
    )
elif args.mode.lower() == "demo":
    demo(stack_name=args.stack)
