import logging
import sys

from rl_traffic_controller.main import train, evaluate


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
datefmt = "%H:%M:%S"
formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

file_handler = logging.FileHandler("logs/run.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
