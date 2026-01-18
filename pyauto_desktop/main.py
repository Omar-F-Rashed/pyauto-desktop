import sys
import os
import time
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage

from .capture_tool import SnippingController
from .overlay import Overlay
from .detection import DetectionWorker
from .editor import MagicWandEditor
from .utils import logical_to_physical, physical_to_logical
from .functions import get_monitors_safe, Session
from .ui_setup import InspectorUI

if sys.platform == "win32":
    import ctypes

    user32 = ctypes.windll.user32
    WDA_EXCLUDEFROMCAPTURE = 0x00000011


    def set_window_display_affinity(hwnd, affinity):
        try:
            user32.SetWindowDisplayAffinity(hwnd, affinity)
            pass
        except Exception as e:
            print(f"Failed to set display affinity: {e}")
else:
    def set_window_display_affinity(hwnd, affinity):
        pass


class MainWindow(InspectorUI):
    def __init__(self):
        super().__init__()

        self.primary_specs = None

        # Image Mode Data
        self.template_image = None
        self.search_region = None
        self.current_scale = 1.0
        self.is_image_unsaved = False
        self.current_filename = None
        self.last_save_dir = ""

        # Text Mode Data
        self.text_rect = None  # (x, y, w, h) physical
        self.text_offsets = [0, 0, 0, 0]  # Top, Bottom, Left, Right

        # Anchor Data (Global)
        self.anchor_image = None
        self.anchor_rect = None
        self.target_rect = None  # For Image Mode
        self.text_anchor_rect = None  # For Text Mode
        self.anchor_filename = None
        self.is_anchor_unsaved = False

        self.snip_controller = SnippingController()
        self.snip_controller.finished.connect(self.on_snip_finished)
        self.active_snip_mode = None

        self.overlay = Overlay()
        if sys.platform == "win32":
            self.overlay.winId()
            set_window_display_affinity(int(self.overlay.winId()), WDA_EXCLUDEFROMCAPTURE)

        # Main Detection Timer
        self.detection_timer = QTimer(self)
        self.detection_timer.timeout.connect(self.detection_step)

        # Live Preview Timer (Passive)
        self.live_preview_timer = QTimer(self)
        self.live_preview_timer.timeout.connect(self.update_live_preview)
        self.live_preview_timer.start(100)  # 10 FPS for passive preview

        self.is_detecting = False
        self.worker = None
        self.worker_running = False
        self.last_fps_time = 0

        self.detection_context = {}

        # Lists to manage granular enabling/disabling
        self.disable_on_run = []  # Widgets to disable during detection

        self.initUI()

        primary = QApplication.primaryScreen()
        if primary:
            self.primary_specs = {
                'dpr': primary.devicePixelRatio(),
                'res': (primary.geometry().width(), primary.geometry().height())
            }

    @pyqtSlot(int)
    def on_tab_changed(self, index):
        try:
            # 0 = Image, 1 = Text
            # Hide Image Match parameters when in Text Mode
            self.container_image_params.setVisible(index == 0)
            self.btn_save.setVisible(index == 0)
            # Disable Anchor Mode completely when in Text Extract tab
            is_text_mode = (index == 1)

            if is_text_mode:
                # Force uncheck and disable
                self.chk_anchor_mode.setChecked(False)
                self.chk_anchor_mode.setEnabled(False)
                self.chk_anchor_mode.setToolTip("Anchor mode is disabled in Text Extract mode.")
            else:
                # Enable controls
                self.chk_anchor_mode.setEnabled(True)
                self.chk_anchor_mode.setToolTip("")

            self.check_gen_enable()
            # Clear overlay artifacts when switching tabs
            if self.overlay.isVisible():
                self.overlay.update_rects([], [], [], 1.0)
        except Exception as e:
            print(f"Error checking gen enable on tab change: {e}")

    def on_text_offset_changed(self):
        self.text_offsets = [
            self.spin_text_top.value(),
            self.spin_text_bottom.value(),
            self.spin_text_left.value(),
            self.spin_text_right.value()
        ]

    def populate_screens(self):
        monitor_rects = get_monitors_safe()
        q_screens = QApplication.screens()
        self.cbo_screens.clear()

        for i, (mx, my, mw, mh) in enumerate(monitor_rects):
            matched_q_screen = None
            for qs in q_screens:
                geo = qs.geometry()
                if geo.x() == mx and geo.y() == my:
                    matched_q_screen = qs
                    break
            if not matched_q_screen and i < len(q_screens):
                matched_q_screen = q_screens[i]

            label = f"Screen {i} [Pos: {mx},{my}] ({mw}x{mh})"
            self.cbo_screens.addItem(label, matched_q_screen)

    def toggle_anchor_ui(self, state):
        is_anchor_on = (state == Qt.CheckState.Checked.value)
        self.frm_anchor.setVisible(is_anchor_on)
        self.check_gen_enable()
        if is_anchor_on:
            self.btn_snip.setText("2. Snip Target Image")
            self.btn_snip_anchor.setText("1. Snip Anchor")
        else:
            self.btn_snip.setText("Snip Target Image")
            self.btn_snip_anchor.setText("Snip Anchor")

    def toggle_margin_inputs(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.spin_margin_x.setEnabled(enabled)
        self.spin_margin_y.setEnabled(enabled)

    def update_conf_label(self):
        val = self.slider_conf.value() / 100.0
        self.lbl_conf_val.setText(f"{val:.2f}")

    def update_overlap_label(self):
        val = self.slider_overlap.value() / 100.0
        self.lbl_overlap_val.setText(f"{val:.2f}")

    def update_overlay_click_settings(self):
        self.overlay.set_click_config(
            self.chk_click.isChecked(),
            self.spin_off_x.value(),
            self.spin_off_y.value()
        )

    def start_snip_template(self):
        self.hide()
        self.active_snip_mode = 'template'
        self.snip_controller.start()

    def start_snip_anchor(self):
        self.hide()
        self.active_snip_mode = 'anchor'
        self.snip_controller.start()

    def start_snip_region(self):
        self.hide()
        self.active_snip_mode = 'region'
        self.snip_controller.start()

    def start_snip_text(self):
        self.hide()
        self.active_snip_mode = 'text'
        self.snip_controller.start()

    def _optimize_image(self, img):
        if img.mode == 'RGBA':
            extrema = img.getextrema()
            if extrema[3][0] == 255:
                return img.convert('RGB')
        return img

    def on_snip_finished(self, pixmap, physical_rect, target_screen):
        self.show()
        if not target_screen:
            self.active_snip_mode = None
            return

        x, y, w, h = physical_rect
        if w < 5 or h < 5:
            self.active_snip_mode = None
            return

        captured_dpr = target_screen.devicePixelRatio()
        captured_res = (target_screen.geometry().width() * captured_dpr,
                        target_screen.geometry().height() * captured_dpr)

        is_anchor_mode = self.chk_anchor_mode.isChecked()
        is_primary = False
        primary_name = "Primary Element"

        if is_anchor_mode:
            primary_name = "Anchor Image"
            if self.active_snip_mode == 'anchor':
                is_primary = True
        else:
            if self.active_snip_mode == 'template' or self.active_snip_mode == 'text':
                is_primary = True

        if is_primary:
            specs_changed = False
            if self.primary_specs:
                if (self.primary_specs['dpr'] != captured_dpr) or (self.primary_specs['res'] != captured_res):
                    specs_changed = True
            else:
                specs_changed = True

            self.primary_specs = {'dpr': captured_dpr, 'res': captured_res}

            if specs_changed:
                self.reset_secondary_elements(is_anchor_mode)

        else:
            if self.primary_specs:
                m_dpr = self.primary_specs['dpr']
                m_res = self.primary_specs['res']

                if (captured_dpr != m_dpr) or (captured_res != m_res):
                    QMessageBox.critical(
                        self,
                        "Monitor Mismatch",
                        f"Capture Rejected!\n\n"
                        f"You are trying to capture a secondary element on a different monitor than your {primary_name}.\n\n"
                        f"Required Monitor: {m_res} @ DPR {m_dpr}\n"
                        f"Current Monitor: {captured_res} @ DPR {captured_dpr}\n\n"
                        f"To switch monitors, please re-capture the {primary_name} on the new monitor first."
                    )
                    self.active_snip_mode = None
                    return
            else:
                self.primary_specs = {'dpr': captured_dpr, 'res': captured_res}

        if self.active_snip_mode == 'template':
            pil_image = self.qpixmap_to_pil(pixmap)
            edited_img = self.open_editor(pil_image)

            if edited_img:
                edited_img = self._optimize_image(edited_img)
                self.template_image = edited_img
                self.is_image_unsaved = True
                self.current_filename = None
                self.target_rect = physical_rect
                self.update_preview()
                self.btn_reedit.setEnabled(True)
                self.btn_save.setEnabled(True)

        elif self.active_snip_mode == 'anchor':
            pil_image = self.qpixmap_to_pil(pixmap)
            edited_img = self.open_editor(pil_image)
            if edited_img:
                edited_img = self._optimize_image(edited_img)
                self.anchor_image = edited_img
                self.anchor_rect = physical_rect
                self.is_anchor_unsaved = True
                self.anchor_filename = None
                self.update_anchor_preview()
                self.btn_save_anchor.setEnabled(True)

        elif self.active_snip_mode == 'text':
            self.text_rect = physical_rect
            self.lbl_text_region.setText(f"Region Set: {physical_rect}")

        elif self.active_snip_mode == 'region':
            self.search_region = physical_rect
            self.lbl_region_status.setText(f"Region: {self.search_region}")
            self.btn_region.set_active(True)

        self.active_snip_mode = None
        self.check_gen_enable()

    def reset_secondary_elements(self, is_anchor_mode):
        self.reset_region()

        if is_anchor_mode:
            self.template_image = None
            self.target_rect = None
            self.current_filename = None
            self.is_image_unsaved = False
            self.lbl_preview.setText("Click or Drop Target Image Here\n(PNG, JPG, BMP)")
            self.lbl_preview.setPixmap(QPixmap())

            self.text_rect = None
            self.lbl_text_region.setText("No Region Selected")

            self.btn_reedit.setEnabled(False)
            self.btn_save.setEnabled(False)

    def reset_region(self):
        self.search_region = None
        self.lbl_region_status.setText("Region: Full Screen")
        self.btn_region.set_active(False)

    def request_upload_image(self, mode='target'):
        current_img = self.template_image if mode == 'target' else self.anchor_image
        label = "Target" if mode == 'target' else "Anchor"

        if current_img:
            if QMessageBox.question(self, "Replace?", f"Replace current {label} image?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
                return

        fname, _ = QFileDialog.getOpenFileName(self, f"Open {label} Image", "", "Images (*.png *.jpg *.bmp)")
        if fname:
            self.process_loaded_image(fname, mode)

    def handle_dropped_image(self, path, mode='target'):
        current_img = self.template_image if mode == 'target' else self.anchor_image
        label = "Target" if mode == 'target' else "Anchor"

        if current_img:
            if QMessageBox.question(self, "Replace?", f"Replace current {label} image?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
                return

        self.process_loaded_image(path, mode)

    def process_loaded_image(self, path, mode='target'):
        try:
            img = Image.open(path)
            edited_img = self.open_editor(img)
            if not edited_img: return

            filename = os.path.basename(path)

            self.primary_specs = None

            if mode == 'target':
                self.template_image = edited_img
                self.current_filename = filename
                self.target_rect = None
                self.is_image_unsaved = False
                self.current_scale = QApplication.primaryScreen().devicePixelRatio()

                self.update_preview()
                self.btn_reedit.setEnabled(True)
                self.btn_save.setEnabled(True)

            elif mode == 'anchor':
                self.anchor_image = edited_img
                self.anchor_filename = filename
                self.anchor_rect = None
                self.is_anchor_unsaved = False

                self.update_anchor_preview()
                self.btn_save_anchor.setEnabled(True)

            self.check_gen_enable()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def reedit_template(self):
        if self.template_image:
            edited_img = self.open_editor(self.template_image)
            if edited_img:
                self.template_image = edited_img
                self.update_preview()

    def open_editor(self, pil_img):
        editor = MagicWandEditor(pil_img, self)
        if editor.exec():
            return editor.get_result()
        return None

    def update_preview(self):
        if not self.template_image: return
        qim = self.pil2pixmap(self.template_image)
        self.lbl_preview.setPixmap(qim)

    def update_anchor_preview(self):
        if not self.anchor_image: return
        qim = self.pil2pixmap(self.anchor_image)
        self.lbl_anchor_preview.setPixmap(qim)
        self.lbl_anchor_preview.setText("")

    def toggle_detection(self):
        if self.is_detecting:
            self.is_detecting = False
            self.detection_timer.stop()
            self.overlay.hide()
            self.btn_start.setText("Start Detection")
            self.btn_start.setStyleSheet("background-color: #198754;")
            self.set_controls_enabled(True)
        else:
            current_mode = self.tabs.currentIndex()  # 0 = Image, 1 = Text
            use_anchor = self.chk_anchor_mode.isChecked()

            if use_anchor:
                if not self.anchor_image:
                    QMessageBox.warning(self, "Anchor Missing", "Please snip an anchor image first.")
                    return
                if not self.anchor_rect:
                    QMessageBox.warning(self, "Anchor Data",
                                        "Anchor has no coordinates (loaded from disk?). Cannot use Relative mode without a snip.")
                    return

            if current_mode == 0:  # Image Mode
                if not self.template_image: return
                if use_anchor and not self.target_rect:
                    if QMessageBox.warning(self, "Spatial Data Missing",
                                           "Target not snipped. Cannot calc offset.\nContinue with zero offset?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
                        return

                if self.chk_gray.isChecked() and self.template_image.mode in ('RGBA', 'LA'):
                    if self.template_image.getextrema()[-1][0] < 255:
                        if QMessageBox.question(self, "Warning",
                                                "Grayscale with transparency enabled. Disable grayscale?",
                                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                            self.chk_gray.setChecked(False)

            elif current_mode == 1:  # Text Mode
                if not self.text_rect:
                    QMessageBox.warning(self, "Region Missing", "Please snip the Text Region first.")
                    return
                # Ensure we clear previous Image Match overlays when starting Text Mode
                self.overlay.update_rects([], [], [], 1.0)

            self.is_detecting = True
            self.update_overlay_click_settings()
            self.last_fps_time = time.time()
            self.overlay.show()
            # 10ms = 100 FPS
            self.detection_timer.start(10)
            self.btn_start.setText("Stop Detection")
            self.btn_start.setStyleSheet("background-color: #dc3545;")
            self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled):
        # Explicitly handled list
        for widget in self.disable_on_run:
            widget.setEnabled(enabled)
        self.tabs.tabBar().setEnabled(enabled)
        # Also re-check Gen button logic if enabled=True
        if enabled:
            self.check_gen_enable()

    def update_live_preview(self):
        """Passive update for Text Tab when detection is NOT running."""
        # Only run if:
        # 1. Not detecting (Worker handles it then)
        # 2. Text Tab is selected
        # 3. We have a valid text_rect
        if self.is_detecting: return
        if self.tabs.currentIndex() != 1: return
        if not self.text_rect: return

        try:
            # Determine Screen
            screen_idx = self.cbo_screens.currentIndex()
            if screen_idx < 0: screen_idx = 0

            # Simple Capture Logic
            # We need absolute coordinates of the region
            # Text Rect is physical coordinates

            x, y, w, h = self.text_rect

            # Apply Offsets
            top, bot, left, right = self.text_offsets
            x -= left
            y -= top
            w += (left + right)
            h += (top + bot)

            if w <= 0 or h <= 0: return

            # For passive preview, we can use Session to grab
            session = Session(screen=screen_idx)

            # Fast Capture using MSS via Session
            img_data, _, _, _ = session._prepare_capture((x, y, w, h))

            if img_data is not None and img_data.size > 0:
                # Convert BGRA to RGBA for Qt
                import cv2
                img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGRA2RGBA)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)

                pix = QPixmap.fromImage(q_img)
                # Scale to label
                lbl_w = self.lbl_text_preview.width()
                lbl_h = self.lbl_text_preview.height()
                if lbl_w > 0 and lbl_h > 0:
                    self.lbl_text_preview.setPixmap(pix.scaled(lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio))

        except Exception as e:
            # print(f"Preview Error: {e}")
            pass

    def detection_step(self):
        if self.worker_running or not self.is_detecting: return

        mode_idx = self.tabs.currentIndex()
        mode = 'image' if mode_idx == 0 else 'text'

        target_screen_idx = self.cbo_screens.currentIndex()
        if target_screen_idx < 0: target_screen_idx = 0

        target_screen_obj = self.cbo_screens.currentData()
        if not target_screen_obj: target_screen_obj = QApplication.primaryScreen()

        source_dpr = self.primary_specs['dpr'] if self.primary_specs else target_screen_obj.devicePixelRatio()
        source_res = self.primary_specs['res'] if self.primary_specs else (target_screen_obj.geometry().width(),
                                                                           target_screen_obj.geometry().height())

        selected_scaling = self.cbo_scaling.currentData()

        conf = self.slider_conf.value() / 100.0
        gray = self.chk_gray.isChecked()
        overlap = self.slider_overlap.value() / 100.0
        use_anchor = self.chk_anchor_mode.isChecked()

        # Prepare Params
        img_to_pass = None
        local_region = None
        anchor_img = None
        anchor_conf = None

        # ANCHOR SETUP
        if use_anchor and mode == 'image':  # Disabled in Text mode via GUI logic
            anchor_img = self.anchor_image
            use_margin = self.chk_anchor_margin.isChecked()
            mx_log = self.spin_margin_x.value() if use_margin else 0
            my_log = self.spin_margin_y.value() if use_margin else 0
            mx_phys, my_phys, _, _ = logical_to_physical((mx_log, my_log, 0, 0), source_dpr)

            anchor_conf = {
                'margin_x': mx_phys,
                'margin_y': my_phys,
                'w': 0, 'h': 0,
                'offset_x': 0, 'offset_y': 0
            }

            if self.anchor_rect and self.target_rect:
                ax, ay, _, _ = self.anchor_rect
                tx, ty, tw, th = self.target_rect
                anchor_conf['offset_x'] = tx - ax
                anchor_conf['offset_y'] = ty - ay
                anchor_conf['w'] = tw
                anchor_conf['h'] = th

        # IMAGE MODE PARAMS
        if mode == 'image':
            img_to_pass = self.template_image
            local_region = self.search_region

        # TEXT MODE PARAMS
        ocr_lang = 'en'  # Default to EN since we removed the selector
        ocr_mode = 'clean'
        use_det = False
        text_rect_param = None
        text_offsets = None

        if mode == 'text':
            text_offsets = self.text_offsets
            use_det = self.chk_use_det.isChecked()

            # Determine OCR Mode from Radio Buttons
            if self.rdo_ocr_dyn.isChecked():
                ocr_mode = 'binarize'
            elif self.rdo_ocr_raw.isChecked():
                ocr_mode = 'restore'
            else:
                ocr_mode = 'clean'

            text_rect_param = self.text_rect  # Absolute (Anchor disabled for text)

        self.detection_context = {
            'screen_geo': target_screen_obj.geometry(),
            'dpr': target_screen_obj.devicePixelRatio(),
            'source_dpr': source_dpr
        }

        self.worker_running = True

        self.worker = DetectionWorker(
            mode=mode,
            template_img=img_to_pass,
            screen_idx=target_screen_idx,
            confidence=conf,
            grayscale=gray,
            overlap_threshold=overlap,
            anchor_img=anchor_img,
            anchor_config=anchor_conf,
            search_region=local_region,
            source_dpr=source_dpr,
            source_resolution=source_res,
            scaling_type=selected_scaling,
            ocr_lang=ocr_lang,
            ocr_mode=ocr_mode,
            use_det=use_det,
            text_rect=text_rect_param,
            text_offsets=text_offsets
        )

        if mode == 'image':
            self.worker.result_signal.connect(self.on_detection_result)
        else:
            self.worker.text_signal.connect(self.on_text_result)

        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.worker_running = False
        self.worker = None

    def on_text_result(self, q_img, text):
        if not self.is_detecting: return

        curr_time = time.time()
        dt = curr_time - self.last_fps_time
        self.last_fps_time = curr_time
        fps = int(1.0 / dt) if dt > 0 else 0
        self.lbl_status.setText(f"Scanning... (FPS: {fps})")

        # Clear overlay if text mode
        if self.overlay.isVisible():
            self.overlay.update_rects([], [], [], 1.0)

        if not q_img.isNull():
            pix = QPixmap.fromImage(q_img)
            h = self.lbl_text_preview.height()
            w = self.lbl_text_preview.width()
            scaled_pix = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl_text_preview.setPixmap(scaled_pix)
        else:
            self.lbl_text_preview.setText(text if text else "Error/No Image")

        self.txt_extracted_result.setText(text)

    def on_detection_result(self, rects, anchors, regions, count):
        if not self.is_detecting: return

        curr_time = time.time()
        dt = curr_time - self.last_fps_time
        self.last_fps_time = curr_time
        fps = int(1.0 / dt) if dt > 0 else 0

        ctx = self.detection_context
        screen_geo = ctx['screen_geo']
        dpr = ctx['dpr']
        source_dpr = ctx.get('source_dpr', dpr)

        self.overlay.set_target_screen_offset(screen_geo.x(), screen_geo.y())

        def map_rects(raw_rects):
            mapped = []
            for r in raw_rects:
                lx, ly, lw, lh = physical_to_logical(r, dpr)
                mapped.append((lx, ly, lw, lh))
            return mapped

        mapped_rects = map_rects(rects)
        mapped_anchors = map_rects(anchors)
        mapped_regions = map_rects(regions)

        self.overlay.update_rects(mapped_rects, mapped_anchors, mapped_regions, source_dpr)

        status_msg = f"Matches: {count} (FPS: {fps})"
        if anchors:
            status_msg += f" | Anchors: {len(anchors)}"
        self.lbl_status.setText(status_msg)

    def save_image(self):
        if not self.template_image: return
        name = self._save_image_dialog(self.template_image, "target.png")
        if name:
            self.current_filename = name
            self.is_image_unsaved = False
            self.check_gen_enable()

    def save_anchor_image(self):
        if not self.anchor_image: return
        name = self._save_image_dialog(self.anchor_image, "anchor.png")
        if name:
            self.anchor_filename = name
            self.is_anchor_unsaved = False
            self.btn_save_anchor.setEnabled(False)
            self.check_gen_enable()

    def _save_image_dialog(self, img, default_name):
        start_path = os.path.join(self.last_save_dir, default_name) if self.last_save_dir else default_name
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", start_path, "Images (*.png)")
        if fname:
            self.last_save_dir = os.path.dirname(fname)
            if not fname.endswith('.png'): fname += '.png'
            try:
                img.save(fname)
                self.lbl_status.setText(f"Saved: {os.path.basename(fname)}")
                return os.path.basename(fname)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")
        return None

    def check_gen_enable(self, return_bool=False):
        # Crash Fix: Ensure this always returns boolean and doesn't rely on ambiguous states
        enabled = True

        # Check Anchor Mode
        if self.chk_anchor_mode.isChecked():
            if not self.anchor_image or self.is_anchor_unsaved:
                enabled = False

        # Check Tab specific
        if self.tabs.currentIndex() == 0:  # Image
            if not self.template_image or self.is_image_unsaved:
                enabled = False
        else:  # Text
            if not self.text_rect:
                enabled = False

        if return_bool:
            return bool(enabled)

        # Safe setEnabled call
        self.btn_gen.setEnabled(bool(enabled))

        can_start = False
        if self.tabs.currentIndex() == 0:
            if self.template_image: can_start = True
        else:
            if self.text_rect: can_start = True

        if self.is_detecting:
            self.btn_start.setEnabled(True)
        else:
            self.btn_start.setEnabled(bool(can_start))

    def generate_code(self):
        mode = self.tabs.currentIndex()  # 0=Image, 1=Text

        screen_idx = self.cbo_screens.currentIndex()
        m_res = self.primary_specs['res'] if self.primary_specs else (1920, 1080)
        dpr = self.primary_specs['dpr'] if self.primary_specs else 1.0

        scaling_val = self.cbo_scaling.currentData()
        scaling_str = f", scaling_type='{scaling_val}'" if scaling_val else ""

        code_lines = []
        code_lines.append(
            f'screen{screen_idx} = pyauto_desktop.Session(screen={screen_idx}, source_resolution=({int(m_res[0])}, {int(m_res[1])}), source_dpr={dpr}{scaling_str})')

        # --- Helper for Click Logic ---
        def get_click_line(var_name, indent="    "):
            if not self.chk_click.isChecked(): return None
            off_x = self.spin_off_x.value()
            off_y = self.spin_off_y.value()
            if off_x == 0 and off_y == 0:
                return f"{indent}screen{screen_idx}.click({var_name})"
            return f"{indent}screen{screen_idx}.click({var_name}, offset=({off_x}, {off_y}))"

        use_anchor = self.chk_anchor_mode.isChecked()

        # --- IMAGE MODE GENERATION ---
        if mode == 0:
            # Handle Target Names
            # original_name: used for the file path (e.g. "Hello World")
            # var_name: used for the python variable (e.g. "Hello_World")
            original_target_name = self.current_filename.replace('.png', '') if self.current_filename else 'target'
            target_var = original_target_name.replace(' ', '_').replace('-', '_')

            # 1. ANCHOR LOGIC
            if use_anchor:
                original_anchor_name = self.anchor_filename.replace('.png', '') if self.anchor_filename else 'anchor'
                anchor_var = original_anchor_name.replace(' ', '_').replace('-', '_')

                # Anchor Params
                p_anch = [f"'images/{original_anchor_name}.png'"]
                if self.search_region: p_anch.append(f"region={self.search_region}")

                code_lines.append(f"{anchor_var}_matches = screen{screen_idx}.locateAllOnScreen({', '.join(p_anch)})")
                code_lines.append(f"for {anchor_var} in {anchor_var}_matches:")
                code_lines.append(f"    # Use current anchor in loop")
                code_lines.append(f"    ax, ay, aw, ah = {anchor_var}[:4]")

                if not self.anchor_rect or not self.target_rect:
                    code_lines.append("    # ERROR: Coords missing, cannot calc relative offset")
                else:
                    ax_orig, ay_orig, _, _ = self.anchor_rect
                    tx_orig, ty_orig, tw, th = self.target_rect

                    rel_x = tx_orig - ax_orig
                    rel_y = ty_orig - ay_orig
                    mx = self.spin_margin_x.value()
                    my = self.spin_margin_y.value()
                    sign_x = "+" if rel_x >= 0 else "-"
                    sign_y = "+" if rel_y >= 0 else "-"

                    code_lines.append(
                        f"    target_region = (ax {sign_x} {abs(rel_x)} - {mx}, ay {sign_y} {abs(rel_y)} - {my}, {tw + mx * 2}, {th + my * 2})")

                    p = [f"'images/{original_target_name}.png'", "region=target_region"]
                    if self.chk_gray.isChecked(): p.append("grayscale=True")
                    p.append(f"confidence={self.slider_conf.value() / 100.0}")

                    # Single vs Loop (Inner Loop)
                    if self.rdo_single.isChecked():
                        code_lines.append(f"    {target_var} = screen{screen_idx}.locateOnScreen({', '.join(p)})")
                        code_lines.append(f"    if {target_var}:")
                        code_lines.append(f"        print('Found {original_target_name}')")
                        click_cmd = get_click_line(target_var, "        ")
                        if click_cmd: code_lines.append(click_cmd)
                        code_lines.append(f"        break # Stop after first match")
                    else:
                        code_lines.append(
                            f"    {target_var}_matches = screen{screen_idx}.locateAllOnScreen({', '.join(p)})")
                        code_lines.append(f"    for {target_var} in {target_var}_matches:")
                        code_lines.append(f"        print('Found {original_target_name}')")
                        click_cmd = get_click_line(target_var, "        ")
                        if click_cmd: code_lines.append(click_cmd)

            # 2. STANDARD IMAGE LOGIC (No Anchor)
            else:
                p = [f"'images/{original_target_name}.png'"]
                if self.search_region: p.append(f"region={self.search_region}")
                if self.chk_gray.isChecked(): p.append("grayscale=True")
                p.append(f"confidence={self.slider_conf.value() / 100.0}")

                # Single vs Loop
                if self.rdo_single.isChecked():
                    code_lines.append(f"{target_var} = screen{screen_idx}.locateOnScreen({', '.join(p)})")
                    code_lines.append(f"if {target_var}:")
                    code_lines.append(f"    print('Found {original_target_name}')")
                    click_cmd = get_click_line(target_var, "    ")
                    if click_cmd: code_lines.append(click_cmd)
                else:
                    code_lines.append(f"{target_var}_matches = screen{screen_idx}.locateAllOnScreen({', '.join(p)})")
                    code_lines.append(f"for {target_var} in {target_var}_matches:")
                    code_lines.append(f"    print('Found {original_target_name}')")
                    click_cmd = get_click_line(target_var, "    ")
                    if click_cmd: code_lines.append(click_cmd)

        # --- TEXT MODE GENERATION ---
        elif mode == 1:
            offsets = self.text_offsets
            use_det = self.chk_use_det.isChecked()

            # Get OCR Mode
            ocr_mode = 'clean'
            if self.rdo_ocr_dyn.isChecked():
                ocr_mode = 'binarize'
            elif self.rdo_ocr_raw.isChecked():
                ocr_mode = 'restore'

            # Static Region Logic (Anchor disabled)
            x, y, w, h = self.text_rect
            top, bot, left, right = offsets
            final_x = x - left
            final_y = y - top
            final_w = w + left + right
            final_h = h + top + bot

            code_lines.append(f"# Static Text Region with offsets applied")
            code_lines.append(f"region = ({final_x}, {final_y}, {final_w}, {final_h})")

            code_lines.append(
                f"text = screen{screen_idx}.read_text(region=region, mode='{ocr_mode}', use_det={use_det})")
            code_lines.append(f"print(f'Extracted: {{text}}')")

        code_block = "\n".join(code_lines)
        self.txt_output.setText(code_block)
        QApplication.clipboard().setText(code_block)
        self.lbl_status.setText("Code copied to clipboard!")

    def pil2pixmap(self, image):
        if image.mode == "RGBA":
            data = image.tobytes("raw", "RGBA")
            qim = QImage(data, image.size[0], image.size[1], QImage.Format.Format_RGBA8888)
        else:
            data = image.convert("RGB").tobytes("raw", "RGB")
            stride = image.size[0] * 3
            qim = QImage(data, image.size[0], image.size[1], stride, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qim.copy())

    def qpixmap_to_pil(self, pixmap):
        try:
            if pixmap.isNull(): raise ValueError("Pixmap is null")
            qimg = pixmap.toImage()
            qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
            width = qimg.width()
            height = qimg.height()
            ptr = qimg.constBits()
            if ptr is None: raise ValueError("Failed to get image bits")
            try:
                ptr.setsize(height * width * 4)
                data_bytes = ptr.asstring()
            except (AttributeError, TypeError):
                data_bytes = qimg.bits().asstring(height * width * 4)

            return Image.frombytes("RGBA", (width, height), data_bytes, "raw", "RGBA", 0, 1)
        except Exception as e:
            print(f"Error converting QPixmap to PIL: {e}")
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    def closeEvent(self, event):
        self.is_detecting = False
        self.detection_timer.stop()
        self.live_preview_timer.stop()
        if self.worker: self.worker.wait()
        self.overlay.close()
        event.accept()


def run_inspector():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    run_inspector()