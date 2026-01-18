from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(35)
        self.parent_window = parent
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 0, 0)
        self.layout.setSpacing(10)

        self.lbl_title = QLabel("Desktop Inspector")
        self.lbl_title.setStyleSheet("color: #0d6efd; font-weight: bold; font-size: 14px;")
        self.layout.addWidget(self.lbl_title)
        self.layout.addStretch()

        btn_style = """
            QPushButton { background: transparent; color: #aaa; border: none; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background: #333; color: white; }
        """
        btn_close_style = """
            QPushButton { background: transparent; color: #aaa; border: none; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background: #dc3545; color: white; }
        """

        self.btn_min = QPushButton("—")
        self.btn_min.setFixedSize(40, 35)
        self.btn_min.setStyleSheet(btn_style)
        self.btn_min.clicked.connect(self.minimize_window)
        self.layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("☐")
        self.btn_max.setFixedSize(40, 35)
        self.btn_max.setStyleSheet(btn_style)
        self.btn_max.clicked.connect(self.maximize_window)
        self.layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(40, 35)
        self.btn_close.setStyleSheet(btn_close_style)
        self.btn_close.clicked.connect(self.close_window)
        self.layout.addWidget(self.btn_close)

        self.start_pos = None

    def minimize_window(self):
        self.parent_window.showMinimized()

    def maximize_window(self):
        if self.parent_window.isMaximized():
            self.parent_window.showNormal()
        else:
            self.parent_window.showMaximized()

    def close_window(self):
        self.parent_window.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            delta = event.globalPosition().toPoint() - self.start_pos
            self.parent_window.move(self.parent_window.pos() + delta)
            self.start_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.start_pos = None


class ClickableDropLabel(QLabel):
    clicked = pyqtSignal()
    file_dropped = pyqtSignal(str)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("border: 2px dashed #555; padding: 10px; background-color: #222; color: #aaa;")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._original_pixmap = None

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        super().setPixmap(self._scale_pixmap(pixmap))

    def _scale_pixmap(self, pixmap):
        if not pixmap or pixmap.isNull(): return pixmap
        return pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        if self._original_pixmap:
            super().setPixmap(self._scale_pixmap(self._original_pixmap))
        super().resizeEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            fpath = urls[0].toLocalFile()
            if fpath and fpath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.file_dropped.emit(fpath)
                event.accept()
                return
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class RegionButton(QPushButton):
    reset_clicked = pyqtSignal()

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.btn_close = QPushButton("X", self)
        self.btn_close.setCursor(Qt.CursorShape.ArrowCursor)
        self.btn_close.setStyleSheet(
            "QPushButton { background: transparent; color: red; font-weight: bold; font-size: 14px; padding: 0px; text-align: center; }"
            "QPushButton:hover { color: #ffcccc; border-color: #ffcccc; }"
        )
        self.btn_close.setFixedSize(30, 30)
        self.btn_close.clicked.connect(self.reset_clicked.emit)
        self.btn_close.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.btn_close.move(self.width() - self.btn_close.width() - 5, (self.height() - self.btn_close.height()) // 2)

    def set_active(self, active):
        if active:
            self.setStyleSheet(
                "QPushButton { background-color: #198754; text-align: left; padding-left: 15px; } QPushButton:hover { background-color: #157347; }")
            self.btn_close.show()
        else:
            self.setStyleSheet("")
            self.btn_close.hide()