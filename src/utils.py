"""
Utility functions

# TODO: Add log handler for both stdout and write to log file.
"""
import os
import git
import yaml
import logging
from time import time
from functools import wraps

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info(f"func:{f.__name__,} took: {round(te - ts, 4)} seconds")
        return result

    return wrap


@timeit
def load_config(directory: str = None, filename: str = "config.yaml"):
    """
    Load configuration from a YAML file.
    """
    directory = (
        directory
        if directory
        else git.Repo(".", search_parent_directories=True).working_tree_dir
    )
    filename = filename if filename else "config.yaml"
    assert os.path.isfile(os.path.join(directory, filename)), f"Not a file: {filename}"
    path = os.path.join(directory, filename)

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f"Config file loaded successfully")
    return config


class Logger:
    def __init__(self, directory: str, filename: str = ""):
        """ """
        self.directory = directory
        self.filename = f"{filename}.logs" if filename else "logs.logs"
        self.log_path = os.path.join(directory, self.filename)
        logger.info("Logger Initialized successfully")

    def _get_root_logger(self):
        """ """
        self.root_logger = logging.getLogger()
        return self

    def _get_log_formatter(self):
        """ """
        self.log_formatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        )
        return self

    def _get_file_handler(self):
        """ """
        self.file_handler = logging.FileHandler(self.log_path)
        self.file_handler.setFormatter(self.log_formatter)
        return self

    def _add_handlers(self):
        """ """
        self.root_logger.addHandler(self.file_handler)
        return self

    def get_logger(self):
        """ """
        logging.basicConfig(level=logging.INFO)
        self._get_root_logger()
        self._get_log_formatter()
        self._get_file_handler()
        self._add_handlers()
        return self.root_logger

    def log_dict(self, obj: dict, level: str = "INFO"):
        pass
