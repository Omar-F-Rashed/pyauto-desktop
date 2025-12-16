from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QPen


class Snipper(QWidget):
    """Widget for selecting a region on the screen."""
    finished = pyqtSignal(tuple, float)  # Emits ((x, y, w, h), scale_factor)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.start_point = None
        self.end_point = None
        self.is_snipping = False

        # Handle High DPI
        self.screen = QApplication.primaryScreen()
        self.scale_factor = self.screen.devicePixelRatio()

        # Grab physical pixels for the overlay image
        self.screen_pixmap = self.screen.grabWindow(0)

        # Set geometry to logical size so it covers the screen correctly in Qt
        geom = self.screen.geometry()
        self.setGeometry(geom)

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.setFocus()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Draw the physical screenshot scaled to the logical widget size
        painter.drawPixmap(self.rect(), self.screen_pixmap)

        # Draw semi-transparent black overlay
        painter.setBrush(QColor(0, 0, 0, 100))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

        if self.start_point and self.end_point:
            # Clear the selected area (make it bright)
            rect = QRect(self.start_point, self.end_point).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.drawRect(rect)

            # Draw border
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(0, 120, 255), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        self.start_point = event.pos()
        self.end_point = event.pos()
        self.is_snipping = True
        self.update()

    def mouseMoveEvent(self, event):
        if self.is_snipping:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.is_snipping = False
        rect = QRect(self.start_point, self.end_point).normalized()
        self.close()
        self.finished.emit((rect.x(), rect.y(), rect.width(), rect.height()), self.scale_factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.finished.emit((0, 0, 0, 0), self.scale_factor)