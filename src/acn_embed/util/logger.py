import logging
import os
import sys
from datetime import datetime

from pytz import timezone, utc


def get_logger(name):
    """Get a logger that reads the level from the env var LOG_LEVEL"""

    # pylint: disable=unused-argument
    def my_local_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone(os.getenv("LOG_TIME_ZONE", "America/Los_Angeles"))
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)s:%(message)s")
    formatter.converter = my_local_time
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.handlers = []

    handlers = [logging.StreamHandler(sys.stderr)]

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    env_log_level = os.getenv("LOG_LEVEL", "WARN").strip().upper()
    logger.setLevel(env_log_level)
    return logger


def get_log_level_str(logger):
    """Return the logging level as a string, like "INFO", "WARNING", etc."""
    return logging.getLevelName(logger.getEffectiveLevel())
