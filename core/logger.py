import logging


def get_logger() -> logging.Logger:
    """
    Get a global pytorch lightning logger.

    Returns
    -------
    logging.Logger
        Custom logger
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("lightning")

    while logger.handlers:
        logger.handlers.pop()

    return logger
