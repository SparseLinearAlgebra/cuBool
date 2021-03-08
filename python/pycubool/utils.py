"""
Utils for accessing cuBool features.

Features:
- Allows to setup logging to custom file with filter settings
- Allows to setup default log
- Allows to create default log file name (for user purposes)
"""

from . import wrapper
from . import bridge
import ctypes
import pathlib
import datetime

__all__ = [
    "setup_logger",
    "setup_default_logger",
    "get_default_log_name"
]


def get_default_log_name():
    """
    Creates default log name, composed from current datetime info.

    :return: String log file
    """
    return "cubool-" + datetime.datetime.now().strftime("%d-%m-%y--%H-%M-%S") + ".textlog"


def setup_logger(file_path: str, default=True, error=False, warning=False):
    """
    Allows to setup logging into user defined logging file.

    :param file_path: Full/relative path to the file to save logged messages
    :param default: Set in true to use default (all) log filter
    :param error: Set in true to log errors
    :param warning: Set in true to log warnings
    :return: None
    """

    status = wrapper.loaded_dll.cuBool_SetupLogging(
        file_path.encode("utf-8"),
        ctypes.c_uint(bridge.get_log_hints(default, error, warning))
    )

    bridge.check(status)


def setup_default_logger():
    """
    Setups default logger, with automatically selected log file
    and default logged messages filter settings.

    :return:
    """

    here = pathlib.Path(__file__).parent
    log_path = here / get_default_log_name()

    setup_logger(str(log_path), default=True)
