import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from box import ConfigBox
from pathlib import Path


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """[summary]

    Args:
        path_to_yaml (Path): [description]

    Raises:
        BoxError: [description]

    Returns:
        ConfigBox: [description]
    """
    try:
        with open(path_to_yaml, mode="r") as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(config_dict)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


def create_directories(path_to_dirs: list, verbose=True):
    for path in path_to_dirs:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"directory created at: {path}")


def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
