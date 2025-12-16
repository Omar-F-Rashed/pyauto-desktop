import sys
import os
import time
from PIL import ImageGrab
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QSlider,
                             QCheckBox, QTextEdit, QFileDialog, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

# --- UPDATED IMPORTS (Relative imports for package support) ---
from .style import DARK_THEME
from .capture_tool import Snipper
from .overlay import Overlay
from .detection import DetectionWorker
from .editor import MagicWandEditor

# --- Windows-specific setup to ignore overlay in screenshots ---
if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

    user32 = ctypes.windll.user32

    # Constants for SetWindowDisplayAffinity
    WDA_NONE = 0x00000000
    WDA_MONITOR = 0x00000001
    WDA_EXCLUDEFROMCAPTURE = 0x00000011  # Excludes window from capture


    def set_window_display_affinity(hwnd, affinity):
        """
        Sets the display affinity of a window.
        WDA_EXCLUDEFROMCAPTURE makes it visible to users but invisible to screen capture.
        """
        try:
            user32.SetWindowDisplayAffinity(hwnd, affinity)
        except Exception as e:
            print(f"Failed to set display affinity: {e}")

else:
    # Dummy function for non-Windows platforms
    def set_window_display_affinity(hwnd, affinity):
        pass


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
        # Position X button on the right, vertically centered
        self.btn_close.move(self.width() - self.btn_close.width() - 5, (self.height() - self.btn_close.height()) // 2)

    def set_active(self, active):
        if active:
            self.setStyleSheet("QPushButton { background-color: #198754; text-align: left; padding-left: 15px; } QPushButton:hover { background-color: #157347; }")
            self.btn_close.show()
        else:
            self.setStyleSheet("")  # Revert to default/stylesheet
            self.btn_close.hide()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop Inspector")
        self.resize(550, 700)
        self.setStyleSheet(DARK_THEME)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        # State
        self.template_image = None
        self.search_region = None  # (x, y, w, h) in Physical Pixels
        self.current_scale = 1.0

        # Initialize Overlay
        self.overlay = Overlay()

        # --- APPLY THE FIX HERE ---
        # We set the affinity so the screen capture sees "through" the overlay.
        if sys.platform == "win32":
            # Force creation of the window handle (WId) so we can pass it to Windows API
            self.overlay.winId()
            # Apply the exclusion flag
            set_window_display_affinity(int(self.overlay.winId()), WDA_EXCLUDEFROMCAPTURE)

        # New state for timer-based detection
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.detection_step)
        self.is_detecting = False
        self.worker = None  # To hold the current worker
        self.worker_running = False
        self.last_fps_time = 0

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- Header ---
        lbl_title = QLabel("AutoMate Studio")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #0d6efd;")
        layout.addWidget(lbl_title, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- 1. Capture Section ---
        grp_cap = QGroupBox("1. Template & Region")
        cap_layout = QVBoxLayout()

        hbox_btns = QHBoxLayout()
        self.btn_snip = QPushButton("Snip Template")
        self.btn_snip.clicked.connect(self.start_snip_template)

        self.btn_region = RegionButton("Set Search Region")
        self.btn_region.clicked.connect(self.start_snip_region)
        self.btn_region.setObjectName("secondary_btn")
        self.btn_region.reset_clicked.connect(self.reset_region)

        hbox_btns.addWidget(self.btn_snip)
        hbox_btns.addWidget(self.btn_region)

        # New "Edit Template" button
        self.btn_reedit = QPushButton("Edit Template")
        self.btn_reedit.clicked.connect(self.reedit_template)
        self.btn_reedit.setEnabled(False)  # Disabled until we have a template
        self.btn_reedit.setObjectName("secondary_btn")

        self.lbl_preview = QLabel("No template selected")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet(
            "border: 2px dashed #555; padding: 10px; min-height: 100px; background-color: #222;")

        self.lbl_region_status = QLabel("Region: Full Screen")
        self.lbl_region_status.setStyleSheet("color: #888; font-size: 12px;")

        cap_layout.addLayout(hbox_btns)
        cap_layout.addWidget(self.btn_reedit)  # Add here
        cap_layout.addWidget(self.lbl_preview)
        cap_layout.addWidget(self.lbl_region_status)
        grp_cap.setLayout(cap_layout)
        layout.addWidget(grp_cap)

        # --- 2. Parameters & Test ---
        grp_test = QGroupBox("2. Live Test")
        test_layout = QVBoxLayout()

        # Confidence
        hbox_conf = QHBoxLayout()
        hbox_conf.addWidget(QLabel("Confidence:"))
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(50, 99)  # 0.5 to 0.99
        self.slider_conf.setValue(90)
        self.slider_conf.valueChanged.connect(self.update_conf_label)
        self.lbl_conf_val = QLabel("0.90")

        hbox_conf.addWidget(self.slider_conf)
        hbox_conf.addWidget(self.lbl_conf_val)

        # Overlap Threshold
        hbox_overlap = QHBoxLayout()
        hbox_overlap.addWidget(QLabel("Overlap Threshold:"))
        self.slider_overlap = QSlider(Qt.Orientation.Horizontal)
        self.slider_overlap.setRange(0, 100)  # 0.0 to 1.0
        self.slider_overlap.setValue(50)  # Default 0.5
        self.slider_overlap.valueChanged.connect(self.update_overlap_label)
        self.lbl_overlap_val = QLabel("0.50")

        hbox_overlap.addWidget(self.slider_overlap)
        hbox_overlap.addWidget(self.lbl_overlap_val)

        # Grayscale
        self.chk_gray = QCheckBox("Grayscale (Faster)")
        self.chk_gray.setChecked(True)
        self.chk_gray.setToolTip("Note: Disables transparency checking for preview.")

        # Buttons
        hbox_ctrl = QHBoxLayout()
        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.toggle_detection)
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("background-color: #198754;")  # Green

        self.lbl_status = QLabel("Matches: 0")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #00ff00;")

        test_layout.addLayout(hbox_conf)
        test_layout.addLayout(hbox_overlap)
        test_layout.addWidget(self.chk_gray)
        test_layout.addLayout(hbox_ctrl)
        test_layout.addWidget(self.btn_start)
        test_layout.addWidget(self.lbl_status)
        grp_test.setLayout(test_layout)
        layout.addWidget(grp_test)

        # --- 3. Output ---
        grp_out = QGroupBox("3. Generate Code")
        out_layout = QVBoxLayout()

        self.txt_output = QTextEdit()
        self.txt_output.setPlaceholderText("Generated code will appear here...")
        self.txt_output.setFixedHeight(120)

        btn_gen = QPushButton("Save Image & Generate Code")
        btn_gen.clicked.connect(self.generate_code)

        out_layout.addWidget(btn_gen)
        out_layout.addWidget(self.txt_output)
        grp_out.setLayout(out_layout)
        layout.addWidget(grp_out)

    def update_conf_label(self):
        val = self.slider_conf.value() / 100.0
        self.lbl_conf_val.setText(f"{val:.2f}")

    def update_overlap_label(self):
        val = self.slider_overlap.value() / 100.0
        self.lbl_overlap_val.setText(f"{val:.2f}")

    # --- Snipping Logic ---
    def start_snip_template(self):
        self.hide()
        self.snipper = Snipper()
        self.snipper.finished.connect(self.on_template_snipped)
        self.snipper.show()

    def on_template_snipped(self, rect, scale_factor):
        self.current_scale = scale_factor
        x, y, w, h = rect

        if w < 5 or h < 5:
            self.show()
            return

        # Capture Physical
        phys_x = int(x * scale_factor)
        phys_y = int(y * scale_factor)
        phys_w = int(w * scale_factor)
        phys_h = int(h * scale_factor)

        # Grab image (Switched to ImageGrab)
        captured_img = ImageGrab.grab(bbox=(phys_x, phys_y, phys_x + phys_w, phys_y + phys_h))

        # Open Editor
        self.open_editor(captured_img)

    def reedit_template(self):
        if self.template_image:
            self.open_editor(self.template_image)

    def open_editor(self, pil_img):
        editor = MagicWandEditor(pil_img, self)
        if editor.exec():
            # User accepted
            self.template_image = editor.get_result()
            self.update_preview()
            self.btn_start.setEnabled(True)
            self.btn_start.setText("Start Detection")
            self.btn_reedit.setEnabled(True)  # Enable edit button

        self.show()

    def update_preview(self):
        if not self.template_image: return
        qim = self.pil2pixmap(self.template_image)
        # Check if alpha exists for display
        self.lbl_preview.setPixmap(
            qim.scaled(self.lbl_preview.width(), self.lbl_preview.height(), Qt.AspectRatioMode.KeepAspectRatio))
        self.lbl_preview.setText("")

    def start_snip_region(self):
        self.hide()
        self.snipper = Snipper()
        self.snipper.finished.connect(self.on_region_snipped)
        self.snipper.show()

    def reset_region(self):
        self.search_region = None
        self.lbl_region_status.setText("Region: Full Screen")
        self.btn_region.set_active(False)

    def on_region_snipped(self, rect, scale_factor):
        self.show()
        self.current_scale = scale_factor
        x, y, w, h = rect

        # Handle cancellation (Escape key or tiny selection)
        if w < 5 or h < 5:
            return

        phys_x = int(x * scale_factor)
        phys_y = int(y * scale_factor)
        phys_w = int(w * scale_factor)
        phys_h = int(h * scale_factor)

        self.search_region = (phys_x, phys_y, phys_w, phys_h)
        self.lbl_region_status.setText(f"Region: {self.search_region}")
        self.btn_region.set_active(True)

    # --- Detection Logic ---
    def toggle_detection(self):
        if self.is_detecting:
            self.is_detecting = False
            self.detection_timer.stop()
            self.overlay.hide()
            self.btn_start.setText("Start Detection")
            self.btn_start.setStyleSheet("background-color: #198754;")
            self.btn_snip.setEnabled(True)
            self.btn_reedit.setEnabled(True)
        else:
            if not self.template_image:
                return

            # Check if user is using Grayscale with a Transparent image
            if self.chk_gray.isChecked() and self.template_image.mode in ('RGBA', 'LA'):
                extrema = self.template_image.getextrema()
                if extrema[-1][0] < 255:
                    ret = QMessageBox.question(self, "Transparency vs Grayscale",
                                               "You have Grayscale enabled with a transparent template.\n"
                                               "Grayscale ignores transparency (treating it as black), which may cause detection failure.\n\n"
                                               "Disable Grayscale for better accuracy?",
                                               QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if ret == QMessageBox.StandardButton.Yes:
                        self.chk_gray.setChecked(False)

            self.is_detecting = True
            self.last_fps_time = time.time()
            # We can enable the overlay immediately since it's now invisible to screenshots
            self.overlay.show()
            self.detection_timer.start(10)  # Check frequently (limited by processing speed)
            self.btn_start.setText("Stop Detection")
            self.btn_start.setStyleSheet("background-color: #dc3545;")  # Red
            self.btn_snip.setEnabled(False)
            self.btn_reedit.setEnabled(False)  # Disable editing while detecting

    def detection_step(self):
        if self.worker_running or not self.is_detecting:
            return

        if not self.template_image:
            self.toggle_detection()  # Stop if template is gone
            return

        self.worker_running = True

        # --- FIX: Removed self.overlay.hide() and processEvents() ---
        # The overlay is now set to WDA_EXCLUDEFROMCAPTURE, so we don't need to hide it.
        # This eliminates the flickering.

        conf = self.slider_conf.value() / 100.0
        gray = self.chk_gray.isChecked()
        overlap = self.slider_overlap.value() / 100.0

        self.worker = DetectionWorker(self.template_image, conf, gray, self.search_region, overlap)
        self.worker.result_signal.connect(self.on_detection_result)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.worker_running = False
        self.worker = None

    def on_detection_result(self, rects, count):
        if not self.is_detecting:
            return

        curr_time = time.time()
        dt = curr_time - self.last_fps_time
        self.last_fps_time = curr_time
        fps = int(1.0 / dt) if dt > 0 else 0

        # Note: DetectionWorker now subtracts the region offset so we get 'relative' coordinates back.
        # We re-add them here for the Overlay to draw correctly on the full screen.
        if self.search_region:
            region_x, region_y = self.search_region[0], self.search_region[1]
            offset_rects = [(r[0] + region_x, r[1] + region_y, r[2], r[3]) for r in rects]
        else:
            offset_rects = rects

        self.overlay.update_rects(offset_rects, self.current_scale)
        self.lbl_status.setText(f"Matches: {count} (FPS: {fps})")
        # Ensure overlay is visible (it should already be, but safe to call)
        self.overlay.show()

    # --- Generation Logic ---
    def generate_code(self):
        if not self.template_image:
            return

        # 1. Save Image (Save as PNG to preserve transparency)
        options = QFileDialog.Option.DontUseNativeDialog
        fname, _ = QFileDialog.getSaveFileName(self, "Save Template Image", "template.png", "Images (*.png)",
                                               options=options)

        if fname:
            if not fname.endswith('.png'):
                fname += '.png'
            self.template_image.save(fname)
            filename = os.path.basename(fname)

            # 2. Generate String
            conf = self.lbl_conf_val.text()
            gray = str(self.chk_gray.isChecked())
            overlap = self.lbl_overlap_val.text()

            # Formulate the parameters string dynamically
            params = [
                f"'{filename}'",
                f"confidence={conf}",
                f"grayscale={gray}",
                f"overlap_threshold={overlap}"
            ]

            # Only add region if it exists
            if self.search_region:
                params.append(f"region={self.search_region}")

            # Join parameters
            params_str = ", ".join(params)

            code_block = (
                f"import pyauto_desktop\n\n"
                f"# Locate single match\n"
                f"match = pyauto_desktop.locateOnScreen({params_str})\n"
                f"if match:\n"
                f"    print(f\"Found at: {{match}}\")\n\n"
                f"# Or locate all matches\n"
                f"matches = pyauto_desktop.locateAllOnScreen({params_str})\n"
                f"for m in matches:\n"
                f"    print(m)"
            )

            self.txt_output.setText(code_block)
            cb = QApplication.clipboard()
            cb.setText(code_block)
            self.lbl_status.setText("Code copied!")

    def pil2pixmap(self, image):
        # Handle RGBA/RGB
        if image.mode == "RGBA":
            data = image.tobytes("raw", "RGBA")
            qim = QImage(data, image.size[0], image.size[1], QImage.Format.Format_RGBA8888)
        else:
            data = image.convert("RGB").tobytes("raw", "RGB")
            qim = QImage(data, image.size[0], image.size[1], QImage.Format.Format_RGB888)

        return QPixmap.fromImage(qim)

    def closeEvent(self, event):
        self.is_detecting = False
        self.detection_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait()
        self.overlay.close()
        event.accept()

# --- ENTRY POINT FUNCTION ---
def run_inspector():
    """Entry point to run the GUI from pip install."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_inspector()