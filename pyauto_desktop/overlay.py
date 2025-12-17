from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush


class Overlay(QWidget):
    """Transparent overlay to draw bounding boxes and click targets."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool |
                            Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Track the top-left corner of the specific screen we are detecting on
        self.target_offset_x = 0
        self.target_offset_y = 0

        # Click Visualization Settings
        self.show_click = False
        self.click_offset_x = 0
        self.click_offset_y = 0

        # Calculate bounding box to cover all screens (The Giant Canvas)
        self._update_geometry()

        self.rects = []
        self.scale_factor = 1.0

    def _update_geometry(self):
        """Recalculate overlay geometry to cover all screens."""
        screens = QApplication.screens()
        if screens:
            # Union of all screen geometries is safer than manual min/max
            # to ensure we match Qt's internal understanding of the virtual desktop
            full_rect = screens[0].geometry()
            for screen in screens[1:]:
                full_rect = full_rect.united(screen.geometry())

            self.setGeometry(full_rect)
        else:
            self.setGeometry(QApplication.primaryScreen().geometry())

    def showEvent(self, event):
        """Recalculate geometry when shown in case screens changed."""
        super().showEvent(event)
        self._update_geometry()

    def set_target_screen_offset(self, x, y):
        """
        Update the offset for the screen currently being scanned.
        x, y: The Global Logical coordinates of the target screen's top-left corner.
        """
        self.target_offset_x = x
        self.target_offset_y = y

    def set_click_config(self, show, off_x, off_y):
        """Update click visualization settings and trigger a repaint."""
        self.show_click = show
        self.click_offset_x = off_x
        self.click_offset_y = off_y
        self.update()

    def update_rects(self, rects, scale_factor):
        self.rects = rects  # Now receiving Screen-Local Logical coordinates
        self.scale_factor = scale_factor
        self.update()

    def paintEvent(self, event):
        if not self.rects:
            return

        try:
            painter = QPainter(self)

            # 1. Draw Detection Boxes (Green)
            pen_box = QPen(QColor(0, 255, 0), 2)
            brush_box = QColor(0, 255, 0, 50)
            painter.setPen(pen_box)
            painter.setBrush(brush_box)

            # Pre-calculate click visualization tools if needed
            if self.show_click:
                pen_dot = QPen(QColor(255, 0, 0), 2)
                brush_dot = QBrush(QColor(255, 0, 0))

            for x, y, w, h in self.rects:
                # --- Draw Box ---
                # Logic:
                # 1. Start with Local Rect (x, y) -> Relative to the Screen being scanned
                # 2. Add Target Screen Offset -> Becomes Global Coordinate
                global_x = x + self.target_offset_x
                global_y = y + self.target_offset_y

                # 3. Use Qt's built-in mapper to find where this global point
                # sits inside this specific Overlay widget.
                top_left_local = self.mapFromGlobal(QPoint(int(global_x), int(global_y)))

                draw_x = top_left_local.x()
                draw_y = top_left_local.y()
                draw_w = int(round(w))
                draw_h = int(round(h))

                # Reset to box style
                painter.setPen(pen_box)
                painter.setBrush(brush_box)
                painter.drawRect(draw_x, draw_y, draw_w, draw_h)

                # --- Draw Click Target (Red Dot) ---
                if self.show_click:
                    # Logic: Calculate position relative to the BOX, not the screen global.
                    # Since draw_x/draw_y are already mapped to the overlay's local coordinate system,
                    # we can simply add the offset here. This avoids "mapFromGlobal" wrapping artifacts
                    # when coordinates go off-screen (negative values), because QPainter handles
                    # out-of-bounds local coordinates by simply clipping them (not drawing them).

                    local_center_x = draw_x + (draw_w / 2)
                    local_center_y = draw_y + (draw_h / 2)

                    target_local = QPoint(
                        int(local_center_x + self.click_offset_x),
                        int(local_center_y + self.click_offset_y)
                    )

                    painter.setPen(pen_dot)
                    painter.setBrush(brush_dot)
                    painter.drawEllipse(target_local, 4, 4)

        except Exception as e:
            print(f"Overlay paint error: {e}")