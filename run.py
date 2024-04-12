import argparse
import logging
from vncdotool import api

from rl_traffic_controller import train, evaluate
from rl_traffic_controller.networks import stacks

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "mode", type=str, help="train or eval"
)
parser.add_argument(
    "stack", type=str, help="layer stack to use", choices=list(stacks.keys())
)
parser.add_argument(
    "image_paths", type=str, nargs="*", default=[], action="extend",
    help="paths of images (observations) to test the agent on"
)
parser.add_argument(
    "-c", "--continue", action="store_true", dest="load_nets",
    help="load the saved network and continue training"
)
parser.add_argument(
    "-r", "--remote", action="store_true",
    help="setup the VNC client connection"
)
parser.add_argument(
    "-e", "--episodes", type=int, default=50,
    help="number of episodes sampled during training (default: %(default)s)"
)

args = parser.parse_args()

if args.remote is True:
    vnc_server = "localhost::5901"
    password = "abcabc"
        
    try:
        client = api.connect(vnc_server, password=password)
        logger.info(f"VNC connection to {vnc_server} established.")
    except Exception:
        logger.exception("VNC connection error.")
        exit()

if args.mode.lower() == "train":
    train(
        stack_name=args.stack,
        load_nets=args.load_nets,
        num_episodes=args.episodes,
        image_paths=args.image_paths
    )
elif args.mode.lower() == "eval":
    evaluate(
        stack_name=args.stack,
        agent=None,
        image_paths=args.image_paths
    )

if args.remote is True:
    api.shutdown()
    logger.info("VNC connection closed.")
