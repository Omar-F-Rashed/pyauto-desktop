from .functions import locateOnScreen, locateAllOnScreen

# Import the GUI runner from main.py
from .main import run_inspector as inspector

__all__ = ['locateOnScreen', 'locateAllOnScreen', 'inspector']