import os
import sys
import logging
from pathlib import Path

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = Path("logs")
log_filepath = Path.joinpath(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("textSummarizerLogger")
