import logging
import sys
from pathlib import Path
from typing import Optional

ROOT_LOGGER_NAME = "vlmeval"


def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    console: bool = True,
    reset: bool = False,
) -> logging.Logger:
    """
    Setup the project logger system (Only use in entry scripts)

    Args:
        log_file: Log file path. If None, avoid output to file.
        log_level: Log level.
        console: Whether to print to the console.
        reset: Force to reset the project root logger.

    Example:
        logger = setup_logger(log_file='outputs/run.log', log_level='INFO')
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    if logger.handlers and not reset:
        return logger
    elif reset:
        logger.handlers.clear()

    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(file_handler)

    # Avoid to propagate to the root logger.
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Please use `__name__` of the module.

    Example:
        logger.info("Hello")
    """
    if name:
        # Add ROOT_LOGGER_NAME if needed.
        if name.startswith(ROOT_LOGGER_NAME):
            logger_name = name
        else:
            logger_name = f"{ROOT_LOGGER_NAME}.{name}"
    else:
        logger_name = ROOT_LOGGER_NAME

    logger = logging.getLogger(logger_name)

    # Avoid warning if `setup_logger` is not called.
    # (If vlmeval package is used as a third-party module)
    if not logging.getLogger(ROOT_LOGGER_NAME).handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def setup_subprocess_logger(log_file: str):
    """
    Setup log system of child processes.

    Use in child processes, to redirect the log file to a standalone file.

    Args:
        log_file: The log file path.

    Example:
        def subprocess_task(log_file):
            setup_subprocess_logger(log_file)
            logger.info('This goes to the subprocess log file')
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger
