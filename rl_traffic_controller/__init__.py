import logging
import os

from rich.logging import RichHandler

# Create required directories
for directory in ["models/", "logs/"]:
    os.makedirs(os.path.join(".", directory), exist_ok=True)

# Setup the parent logger of the entire package.

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
datefmt = "%H:%M:%S"
file_formatter = logging.Formatter(fmt=file_fmt, datefmt=datefmt)
file_handler = logging.FileHandler("logs/run.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

stream_fmt = "%(message)s"
stream_formatter = logging.Formatter(fmt=stream_fmt)
stream_handler = RichHandler(
    omit_repeated_times=False,
    show_path=False,
    rich_tracebacks=True,
    log_time_format="[%X]",
)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
