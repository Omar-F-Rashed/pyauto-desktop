from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QCheckBox, QTextEdit,
                             QGroupBox, QComboBox, QSpinBox, QRadioButton, QFrame,
                             QTabWidget, QScrollArea, QSizePolicy, QButtonGroup)
from PyQt6.QtCore import Qt

from .widgets import CustomTitleBar, ClickableDropLabel, RegionButton
from .style import DARK_THEME


class InspectorUI(QMainWindow):
    """
    Handles the UI construction and layout for the Main Window.
    Logic and Event handling is deferred to the main class.
    """

    def initUI(self):
        self.setWindowTitle("Desktop Inspector")
        self.resize(1000, 590)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(DARK_THEME + """
            QMainWindow { background-color: #1e1e1e; border: 1px solid #444; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self)
        root_layout.addWidget(self.title_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        root_layout.addWidget(self.scroll_area)

        content_widget = QWidget()
        self.scroll_area.setWidget(content_widget)

        columns_layout = QHBoxLayout(content_widget)
        columns_layout.setContentsMargins(10, 10, 10, 10)
        columns_layout.setSpacing(15)
        columns_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        left_col_widget = QWidget()
        left_col_layout = QVBoxLayout(left_col_widget)
        left_col_layout.setContentsMargins(0, 0, 0, 0)
        left_col_layout.setSpacing(10)
        left_col_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        right_col_widget = QWidget()
        right_col_layout = QVBoxLayout(right_col_widget)
        right_col_layout.setContentsMargins(0, 0, 0, 0)
        right_col_layout.setSpacing(10)
        right_col_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        columns_layout.addWidget(left_col_widget, stretch=5)
        columns_layout.addWidget(right_col_widget, stretch=5)

        grp_config = QGroupBox("1. Configuration")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(5)

        self.chk_anchor_mode = QCheckBox("Use Anchor Image (Relative Search)")
        self.chk_anchor_mode.setStyleSheet("font-weight: bold; color: #ffc107;")
        self.chk_anchor_mode.stateChanged.connect(self.toggle_anchor_ui)
        config_layout.addWidget(self.chk_anchor_mode)
        self.disable_on_run.append(self.chk_anchor_mode)

        self.frm_anchor = QFrame()
        self.frm_anchor.setVisible(False)
        anchor_layout = QVBoxLayout(self.frm_anchor)
        anchor_layout.setContentsMargins(0, 0, 0, 0)

        hbox_anchor_btn = QHBoxLayout()
        self.btn_snip_anchor = QPushButton("Snip Anchor")
        self.btn_snip_anchor.clicked.connect(self.start_snip_anchor)
        hbox_anchor_btn.addWidget(self.btn_snip_anchor)
        self.disable_on_run.append(self.btn_snip_anchor)

        self.btn_save_anchor = QPushButton("Save Anchor")
        self.btn_save_anchor.clicked.connect(self.save_anchor_image)
        self.btn_save_anchor.setEnabled(False)
        hbox_anchor_btn.addWidget(self.btn_save_anchor)
        anchor_layout.addLayout(hbox_anchor_btn)

        hbox_anchor_margin = QHBoxLayout()
        self.chk_anchor_margin = QCheckBox("Search Margin")
        self.chk_anchor_margin.setChecked(True)
        self.chk_anchor_margin.stateChanged.connect(self.toggle_margin_inputs)
        self.disable_on_run.append(self.chk_anchor_margin)

        self.spin_margin_x = QSpinBox()
        self.spin_margin_x.setRange(0, 9999)
        self.spin_margin_x.setValue(20)
        self.spin_margin_x.setSuffix(" px")
        self.spin_margin_x.setToolTip("Horizontal Margin (Left/Right)")
        self.disable_on_run.append(self.spin_margin_x)

        self.spin_margin_y = QSpinBox()
        self.spin_margin_y.setRange(0, 9999)
        self.spin_margin_y.setValue(20)
        self.spin_margin_y.setSuffix(" px")
        self.spin_margin_y.setToolTip("Vertical Margin (Top/Bottom)")
        self.disable_on_run.append(self.spin_margin_y)

        hbox_anchor_margin.addWidget(self.chk_anchor_margin)
        hbox_anchor_margin.addWidget(QLabel("X:"))
        hbox_anchor_margin.addWidget(self.spin_margin_x)
        hbox_anchor_margin.addWidget(QLabel("Y:"))
        hbox_anchor_margin.addWidget(self.spin_margin_y)
        anchor_layout.addLayout(hbox_anchor_margin)

        self.lbl_anchor_preview = ClickableDropLabel("Click or Drop Target Image Here\n(PNG, JPG, BMP)")
        self.lbl_anchor_preview.clicked.connect(lambda: self.request_upload_image(mode='anchor'))
        self.lbl_anchor_preview.file_dropped.connect(lambda p: self.handle_dropped_image(p, mode='anchor'))
        self.lbl_anchor_preview.setMinimumHeight(80)
        self.lbl_anchor_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.disable_on_run.append(self.lbl_anchor_preview)
        anchor_layout.addWidget(self.lbl_anchor_preview)

        config_layout.addWidget(self.frm_anchor)

        self.tabs = QTabWidget()

        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; top: -1px; }
            QTabBar::tab { background: #2b2b2b; color: #888; padding: 6px 20px; border: 1px solid #333; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; min-width: 80px; }
            QTabBar::tab:selected { background: #3c3c3c; color: #fff; border-bottom: 2px solid #0d6efd; font-weight: bold; }
            QTabBar::tab:hover { background: #333; }
        """)
        self.tabs.tabBar().setDocumentMode(True)
        self.tabs.tabBar().setExpanding(True)

        self.tab_image = QWidget()
        self.init_image_tab()
        self.tabs.addTab(self.tab_image, "Image Match")

        self.tab_text = QWidget()
        self.init_text_tab()
        self.tabs.addTab(self.tab_text, "Text Extract")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        config_layout.addWidget(self.tabs)
        grp_config.setLayout(config_layout)

        left_col_layout.addWidget(grp_config)

        grp_test = QGroupBox("2. Live Test & Action")
        test_layout = QVBoxLayout()
        test_layout.setSpacing(5)

        self.container_image_params = QWidget()
        img_params_layout = QVBoxLayout(self.container_image_params)
        img_params_layout.setContentsMargins(0, 0, 0, 0)
        img_params_layout.setSpacing(5)

        hbox_conf = QHBoxLayout()
        hbox_conf.addWidget(QLabel("Confidence:"))
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(50, 99)
        self.slider_conf.setValue(90)
        self.slider_conf.valueChanged.connect(self.update_conf_label)
        self.lbl_conf_val = QLabel("0.90")
        hbox_conf.addWidget(self.slider_conf)
        hbox_conf.addWidget(self.lbl_conf_val)
        img_params_layout.addLayout(hbox_conf)

        hbox_overlap = QHBoxLayout()
        hbox_overlap.addWidget(QLabel("Overlap:"))
        self.slider_overlap = QSlider(Qt.Orientation.Horizontal)
        self.slider_overlap.setRange(0, 100)
        self.slider_overlap.setValue(50)
        self.slider_overlap.valueChanged.connect(self.update_overlap_label)
        self.lbl_overlap_val = QLabel("0.50")
        hbox_overlap.addWidget(self.slider_overlap)
        hbox_overlap.addWidget(self.lbl_overlap_val)
        img_params_layout.addLayout(hbox_overlap)

        self.chk_gray = QCheckBox("Grayscale (Faster)")
        self.chk_gray.setChecked(True)
        img_params_layout.addWidget(self.chk_gray)

        hbox_click = QHBoxLayout()
        self.chk_click = QCheckBox("Simulate Click")
        self.chk_click.stateChanged.connect(self.update_overlay_click_settings)

        self.spin_off_x = QSpinBox()
        self.spin_off_x.setRange(-9999, 9999)
        self.spin_off_x.setValue(0)
        self.spin_off_x.setSuffix(" px")
        self.spin_off_x.valueChanged.connect(self.update_overlay_click_settings)

        self.spin_off_y = QSpinBox()
        self.spin_off_y.setRange(-9999, 9999)
        self.spin_off_y.setValue(0)
        self.spin_off_y.setSuffix(" px")
        self.spin_off_y.valueChanged.connect(self.update_overlay_click_settings)

        hbox_click.addWidget(self.chk_click)
        hbox_click.addWidget(QLabel("X:"))
        hbox_click.addWidget(self.spin_off_x)
        hbox_click.addWidget(QLabel("Y:"))
        hbox_click.addWidget(self.spin_off_y)
        img_params_layout.addLayout(hbox_click)

        test_layout.addWidget(self.container_image_params)

        hbox_screen = QHBoxLayout()
        self.cbo_screens = QComboBox()
        hbox_screen.addWidget(QLabel("Detect On:"))
        hbox_screen.addWidget(self.cbo_screens)
        self.populate_screens()
        self.disable_on_run.append(self.cbo_screens)

        hbox_scaling = QHBoxLayout()
        hbox_scaling.addWidget(QLabel("Scaling Strategy:"))
        self.cbo_scaling = QComboBox()
        self.cbo_scaling.addItem("DPR Awareness", "dpr")
        self.cbo_scaling.addItem("Resolution Matching", "resolution")
        self.cbo_scaling.setToolTip(
            "DPR: For non-full screen applications (browser, office apps, etc) \nResolution: Full screen applications like games.")
        hbox_scaling.addWidget(self.cbo_scaling)
        self.disable_on_run.append(self.cbo_scaling)

        hbox_ctrl = QHBoxLayout()
        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.toggle_detection)
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("background-color: #198754;")

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #00ff00;")

        test_layout.addLayout(hbox_screen)
        test_layout.addLayout(hbox_scaling)
        test_layout.addLayout(hbox_ctrl)
        test_layout.addWidget(self.btn_start)
        test_layout.addWidget(self.lbl_status)
        grp_test.setLayout(test_layout)

        right_col_layout.addWidget(grp_test)

        grp_out = QGroupBox("3. Generate Code")
        out_layout = QVBoxLayout()
        out_layout.setSpacing(5)

        hbox_mode = QHBoxLayout()
        self.rdo_single = QRadioButton("Best Match (Single)")
        self.rdo_single.setChecked(True)
        self.rdo_all = QRadioButton("All Matches (Loop)")
        hbox_mode.addWidget(self.rdo_single)
        hbox_mode.addWidget(self.rdo_all)
        out_layout.addLayout(hbox_mode)

        hbox_gen = QHBoxLayout()

        self.btn_save = QPushButton("Save Target")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        self.disable_on_run.append(self.btn_save)

        self.btn_gen = QPushButton("Generate Code")
        self.btn_gen.clicked.connect(self.generate_code)
        self.btn_gen.setEnabled(False)
        self.disable_on_run.append(self.btn_gen)

        hbox_gen.addWidget(self.btn_save)
        hbox_gen.addWidget(self.btn_gen)

        out_layout.addLayout(hbox_gen)

        self.txt_output = QTextEdit()
        self.txt_output.setPlaceholderText("Generated code will appear here...")
        self.txt_output.setFixedHeight(120)
        out_layout.addWidget(self.txt_output)

        grp_out.setLayout(out_layout)

        right_col_layout.addWidget(grp_out)

        right_col_layout.addStretch()

    def init_image_tab(self):
        layout = QVBoxLayout(self.tab_image)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        hbox_btns = QHBoxLayout()
        self.btn_snip = QPushButton("Snip Target Image")
        self.btn_snip.clicked.connect(self.start_snip_template)
        self.disable_on_run.append(self.btn_snip)

        self.btn_region = RegionButton("Set Search Region")
        self.btn_region.clicked.connect(self.start_snip_region)
        self.btn_region.setObjectName("secondary_btn")
        self.btn_region.reset_clicked.connect(self.reset_region)
        self.disable_on_run.append(self.btn_region)

        hbox_btns.addWidget(self.btn_snip)
        hbox_btns.addWidget(self.btn_region)

        self.btn_reedit = QPushButton("Edit Target Image")
        self.btn_reedit.clicked.connect(self.reedit_template)
        self.btn_reedit.setEnabled(False)
        self.btn_reedit.setObjectName("secondary_btn")
        self.disable_on_run.append(self.btn_reedit)

        self.lbl_preview = ClickableDropLabel("Click or Drop Target Image Here\n(PNG, JPG, BMP)")
        self.lbl_preview.clicked.connect(lambda: self.request_upload_image(mode='target'))
        self.lbl_preview.file_dropped.connect(lambda p: self.handle_dropped_image(p, mode='target'))

        self.lbl_preview.setFixedHeight(250)

        self.disable_on_run.append(self.lbl_preview)

        self.lbl_region_status = QLabel("Region: Full Screen")
        self.lbl_region_status.setStyleSheet("color: #888; font-size: 12px;")

        layout.addLayout(hbox_btns)
        layout.addWidget(self.btn_reedit)
        layout.addWidget(self.lbl_preview)
        layout.addWidget(self.lbl_region_status)

    def init_text_tab(self):
        layout = QVBoxLayout(self.tab_text)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        self.btn_snip_text = QPushButton("Snip Text Region")
        self.btn_snip_text.clicked.connect(self.start_snip_text)
        self.disable_on_run.append(self.btn_snip_text)
        layout.addWidget(self.btn_snip_text)

        self.lbl_text_region = QLabel("No Region Selected")
        self.lbl_text_region.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.lbl_text_region)

        hbox_det = QHBoxLayout()
        self.chk_use_det = QCheckBox("Enable Smart Detection (Slower)")
        self.chk_use_det.setToolTip("Uses deep learning detection model for more accurate but slower results.")
        hbox_det.addWidget(self.chk_use_det)
        layout.addLayout(hbox_det)

        hbox_ocr = QHBoxLayout()
        hbox_ocr.addWidget(QLabel("Mode:"))

        self.rdo_ocr_std = QRadioButton("Clean")
        self.rdo_ocr_dyn = QRadioButton("Binarize")
        self.rdo_ocr_raw = QRadioButton("Restore")
        self.rdo_ocr_std.setChecked(True)
        self.rdo_ocr_std.setToolTip("Standard cleanup (Denoise, Upscale)")
        self.rdo_ocr_dyn.setToolTip("High contrast/Color splitting (For colored text)")
        self.rdo_ocr_raw.setToolTip("Heavy restoration (For faded/damaged text)")

        self.grp_ocr_mode = QButtonGroup(self)
        self.grp_ocr_mode.addButton(self.rdo_ocr_std)
        self.grp_ocr_mode.addButton(self.rdo_ocr_dyn)
        self.grp_ocr_mode.addButton(self.rdo_ocr_raw)

        hbox_ocr.addWidget(self.rdo_ocr_std)
        hbox_ocr.addWidget(self.rdo_ocr_dyn)
        hbox_ocr.addWidget(self.rdo_ocr_raw)
        layout.addLayout(hbox_ocr)

        hbox_fine = QHBoxLayout()
        hbox_fine.setContentsMargins(5, 5, 5, 5)

        self.spin_text_top = QSpinBox()
        self.spin_text_bottom = QSpinBox()
        self.spin_text_left = QSpinBox()
        self.spin_text_right = QSpinBox()

        for s in [self.spin_text_top, self.spin_text_bottom, self.spin_text_left, self.spin_text_right]:
            s.setRange(-500, 500)
            s.setSuffix("")
            s.valueChanged.connect(self.on_text_offset_changed)

        hbox_fine.addWidget(QLabel("T:"))
        hbox_fine.addWidget(self.spin_text_top)
        hbox_fine.addWidget(QLabel("B:"))
        hbox_fine.addWidget(self.spin_text_bottom)
        hbox_fine.addWidget(QLabel("L:"))
        hbox_fine.addWidget(self.spin_text_left)
        hbox_fine.addWidget(QLabel("R:"))
        hbox_fine.addWidget(self.spin_text_right)

        layout.addWidget(QLabel("Live Region Preview:"))
        self.lbl_text_preview = QLabel("Waiting for Detection...")
        self.lbl_text_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_text_preview.setStyleSheet("background-color: #000; color: #fff; border: 1px solid #555;")

        self.lbl_text_preview.setFixedHeight(120)
        self.lbl_text_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        layout.addWidget(self.lbl_text_preview)

        layout.addWidget(QLabel("Extracted Text Result:"))
        self.txt_extracted_result = QTextEdit()
        self.txt_extracted_result.setReadOnly(True)
        self.txt_extracted_result.setFixedHeight(100)
        layout.addWidget(self.txt_extracted_result)

        layout.addStretch()