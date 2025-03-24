import logging
import sys

def get_logger(name: str = __name__, log_file: str = None, level=logging.INFO):
    """Configure and return a logger that writes to console and optionally to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # If logger already has handlers, return it (to avoid duplicate logs)
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
