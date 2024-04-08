import argparse
import logging
from vncdotool import api

from rl_traffic_controller import train, evaluate

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("mode", type=str)
parser.add_argument("image_paths", type=str, nargs="*", default=[], action="extend")
parser.add_argument("-c", "--continue", action="store_true", dest="load_nets")
parser.add_argument("-r", "--remote", action="store_true")
parser.add_argument("-e", "--episodes", type=int, default=50)

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
        checkpoints=True,
        load_nets=args.load_nets,
        num_episodes=args.episodes
    )
elif args.mode.lower() == "eval":
    evaluate(
        agent=None,
        image_paths=args.image_paths
    )

if args.remote is True:
    api.shutdown()
    logger.info("VNC connection closed.")
