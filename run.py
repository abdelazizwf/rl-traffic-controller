#!/usr/bin/bash python3

import argparse
from functools import partial

from rich import print

from rl_traffic_controller.main import demo, evaluate, train

parser = argparse.ArgumentParser()

parser.add_argument(
    "mode", type=str, help="train or eval or demo or dry-run"
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
    "-e", "--episodes", type=int, default=1, metavar="N",
    help="number of episodes sampled during training (default: %(default)s)"
)
parser.add_argument(
    "--images", type=str, nargs="+", action="extend", metavar="", dest="image_paths", default=[],
    help="paths of images (observations), and/or directories containing images, to test the agent on"
)

args = parser.parse_args()

train = partial(
    train,
    load_nets=args.load_nets,
    save=args.save,
    num_episodes=args.episodes,
    image_paths=args.image_paths,
)

mode = args.mode.lower()
if mode == "train":
    train()
elif mode == "dry-run":
    train(stub=True)
elif mode == "eval":
    if args.image_paths is None:
        print(
            "[red]ERROR[/red]: You must use the '--images' option when using 'eval' mode.",
            "Use 'python3.11 run.py --help' to know more."
        )
        exit(-6)
    evaluate(
        agent=None,
        image_paths=args.image_paths
    )
elif mode == "demo":
    demo()
else:
    print(
        f"[red]ERROR[/red]: Invalid mode '{mode}'. Use 'python3.11 run.py --help' to know more."
    )
    exit(-1)
