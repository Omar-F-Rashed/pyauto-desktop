import logging
rapid_logger = logging.getLogger("RapidOCR")
rapid_logger.handlers.clear()
rapid_logger.propagate = False
rapid_logger.disabled = True
from .functions import Session
from . import dpi_manager
from .window_control import (
    find_window,
    move_window,
    resize_window,
    focus_window,
    maximize_window,
    minimize_window,
    get_window_info
)

dpi_manager.enable_dpi_awareness()

def inspector():
    from .main import run_inspector
    run_inspector()

# Add the new functions to __all__ so they are exported cleanly
__all__ = [
    'Session',
    'inspector',
    'find_window',
    'move_window',
    'resize_window',
    'focus_window',
    'maximize_window',
    'minimize_window',
    'get_window_info'
]