from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen


class Overlay(QWidget):
    """Transparent overlay to draw bounding boxes."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool |
                            Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Logical size to cover screen
        geom = QApplication.primaryScreen().geometry()
        self.setGeometry(geom)

        self.rects = []
        self.scale_factor = 1.0

    def update_rects(self, rects, scale_factor):
        self.rects = rects  # Physical coordinates
        self.scale_factor = scale_factor
        self.update()

    def paintEvent(self, event):
        if not self.rects:
            return
        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setBrush(QColor(0, 255, 0, 50))

        # Convert physical rects to logical for drawing
        for (x, y, w, h) in self.rects:
            lx = int(x / self.scale_factor)
            ly = int(y / self.scale_factor)
            lw = int(w / self.scale_factor)
            lh = int(h / self.scale_factor)
            painter.drawRect(lx, ly, lw, lh)