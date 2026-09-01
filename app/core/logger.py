import logging
import sys
import warnings

_logging_configured = False

def setup_logging(loglevel: str):
    """
    Configures the root logger.
    """
    global _logging_configured
    if _logging_configured:
        return

    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set level from config
    root_logger.setLevel(loglevel.upper())

    # Ignore future warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    _logging_configured = True

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with a specified name.
    Assumes setup_logging has been called.
    """
    return logging.getLogger(name)
