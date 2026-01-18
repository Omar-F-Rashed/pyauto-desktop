import time
import platform
import pywinctl

if platform.system() == "Windows":
    import ctypes
    user32 = ctypes.windll.user32


def _get_window_pid(window_obj):
    """
    Helper to get PID from a pywinctl Window object on Windows.
    """
    if platform.system() == "Windows":
        try:
            hwnd = window_obj.getHandle()
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return pid.value
        except Exception:
            return None
    return None


def find_window(target):
    """
    Finds a window by Title (string) or PID (int).
    Returns the first matching pywinctl Window object or None.
    """
    all_windows = pywinctl.getAllWindows()

    if isinstance(target, int):
        target_pid = target
        for win in all_windows:
            if _get_window_pid(win) == target_pid:
                return win
        return None

    elif isinstance(target, str):
        matches = pywinctl.getWindowsWithTitle(target)
        if matches:
            return matches[0]

        target_lower = target.lower()
        for win in all_windows:
            if target_lower in win.title.lower():
                return win

    return None


def move_window(target, x, y):
    """
    Moves the top-left corner of the window to (x, y).
    """
    win = find_window(target)
    if win:
        try:
            win.moveTo(int(x), int(y))
            return True
        except Exception as e:
            print(f"Failed to move window: {e}")
    else:
        print(f"Window '{target}' not found.")
    return False


def resize_window(target, width, height):
    """
    Resizes the window to the specified width and height.
    """
    win = find_window(target)
    if win:
        try:
            win.resizeTo(int(width), int(height))
            return True
        except Exception as e:
            print(f"Failed to resize window: {e}")
    return False


def focus_window(target):
    """
    Brings the window to the foreground.
    Automatically un-minimizes (restores) it if hidden.
    """
    win = find_window(target)
    if win:
        try:
            if hasattr(win, "isMinimized") and win.isMinimized:
                win.restore()
                time.sleep(0.1)

            win.activate()
            return True
        except Exception as e:
            print(f"Failed to focus window: {e}")
    return False


def maximize_window(target):
    win = find_window(target)
    if win:
        win.maximize()
        return True
    return False


def minimize_window(target):
    win = find_window(target)
    if win:
        win.minimize()
        return True
    return False


def get_window_info(target):
    """
    Returns a dictionary of window properties (x, y, width, height, title, pid).
    """
    win = find_window(target)
    if win:
        return {
            "title": win.title,
            "x": win.left,
            "y": win.top,
            "width": win.width,
            "height": win.height,
            "pid": _get_window_pid(win)
        }
    return None