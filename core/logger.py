"""Main logging module."""
import logging


def get_logger() -> logging.Logger:
    """
    Get a global pytorch lightning logger.

    Returns
    -------
    logging.Logger
        Custom logger
    """
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Define format for logs
    log_format = "%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)2d | %(message)s"

    # Set up logging to a file
    file_handler = logging.FileHandler("classification.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Set up logging to a console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Set up a PL logger too, but no need to return it as the main logger
    pl_logger = logging.getLogger("pytorch_lightning")
    pl_logger.setLevel(logging.DEBUG)
    pl_logger.addHandler(file_handler)

    return logger


logger = get_logger()
