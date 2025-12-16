# Updated import to point to the new 'functions.py' inside the package
from . import functions as pyauto_desktop
from PyQt6.QtCore import QThread, pyqtSignal


class DetectionWorker(QThread):
    """
    One-shot thread for image recognition.
    Refactored to use the 'pyauto-desktop' module to ensure
    preview behavior matches generated code behavior.
    """
    result_signal = pyqtSignal(list, int)  # list of rects, count

    def __init__(self, template_img, confidence, grayscale, region, overlap_threshold=0.5):
        super().__init__()
        self.template_img = template_img
        self.confidence = confidence
        self.grayscale = grayscale
        self.region = region
        self.overlap_threshold = overlap_threshold

    def run(self):
        try:
            # We use the library function directly.
            # Note: locateAllOnScreen returns Global coordinates.
            rects = pyauto_desktop.locateAllOnScreen(
                image=self.template_img,
                region=self.region,
                grayscale=self.grayscale,
                confidence=self.confidence,
                overlap_threshold=self.overlap_threshold
            )

            final_rects = []
            if self.region:
                rx, ry, _, _ = self.region
                for (x, y, w, h) in rects:
                    final_rects.append((x - rx, y - ry, w, h))
            else:
                final_rects = rects

            self.result_signal.emit(final_rects, len(final_rects))

        except Exception as e:
            print(f"Error in detection worker: {e}")
            import traceback
            traceback.print_exc()
            self.result_signal.emit([], 0)