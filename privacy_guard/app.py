from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, QEventLoop
from PySide6.QtGui import (
    QColor, QPainter, QPen, QPixmap, QIcon, QFont, QFontDatabase, QCloseEvent,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QComboBox, QDoubleSpinBox,
    QSystemTrayIcon, QMenu, QStackedWidget, QSizePolicy, QScrollArea, QFileDialog,
    QInputDialog, QLineEdit, QMessageBox,
)

from .detection import CameraWorker, OWNER_MATCH_THRESHOLD
from .policy import AlertPolicy, HoldTrigger, ArmState
from .recording import AudioRecorder, ScreenRecorder, grab_screen_gdi, save_image
from . import storage

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
MODEL = ROOT / "assets" / "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_MODEL = ROOT / "assets" / "face_recognition_sface_2021dec.onnx"

ENROLL_TARGET = 12  # owner embeddings to collect before guard mode is usable

STYLE = """
QWidget { color: #e7edf2; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 13px; }
QMainWindow, QWidget#root, QWidget#page { background: #10151d; }
QScrollBar:vertical { background: #10151d; width: 10px; margin: 0; }
QScrollBar:horizontal { background: #10151d; height: 10px; margin: 0; }
QScrollBar::handle { background: #3c4c60; border-radius: 5px; min-width: 24px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QFrame#sidebar { background: #141c26; border-right: 1px solid #25303d; }
QFrame#card { background: #18212d; border: 1px solid #2b3746; border-radius: 14px; }
QFrame#inner { background: #111923; border: 1px solid #2b3746; border-radius: 10px; }
QLabel { background: transparent; }
QLabel#muted { color: #a0afbf; }
QLabel#small { color: #8b9aaa; font-size: 11px; }
QLabel#eyebrow { color: #69d7bd; font-size: 11px; font-weight: 600; }
QLabel#heading { font-size: 25px; font-weight: 700; }
QLabel#number { font-size: 38px; font-weight: 700; }
QLabel#status { color: #6cdac0; font-size: 20px; font-weight: 600; }
QLabel#rec { color: #f08a8a; font-size: 13px; font-weight: 600; }
QLabel#error { color: #f08a8a; font-size: 12px; }
QLabel#path { color: #cfe3dc; background: #111923; border: 1px solid #2b3746; border-radius: 7px; padding: 8px 10px; }
QPushButton { background: #243142; border: 1px solid #39485b; border-radius: 8px; padding: 11px 16px; font-weight: 600; }
QPushButton:hover { background: #304056; }
QPushButton:pressed { background: #1e2b39; }
QPushButton:disabled { color: #607082; background: #1b2633; border-color: #263443; }
QPushButton#primary { background: #6fdfc0; color: #102821; border: 0; }
QPushButton#primary:hover { background: #92edd2; }
QPushButton#primary:disabled { background: #315b51; color: #90aaa2; }
QPushButton#toggle { background: #1b2633; border: 1px solid #334256; text-align: left; padding-left: 14px; }
QPushButton#toggle:checked { background: #244b44; border: 1px solid #4fb89b; color: #9cf0d8; }
QPushButton#toggle:disabled { color: #566573; }
QPushButton#nav { border: 0; text-align: left; background: transparent; color: #a3b3c4; padding: 13px; }
QPushButton#nav:checked { background: #253a3b; color: #84e1cb; }
QPushButton#quiet { background: transparent; border: 0; color: #adbcca; }
QComboBox, QDoubleSpinBox { background: #111923; border: 1px solid #38495b; border-radius: 7px; padding: 8px 10px; min-height: 20px; }
QComboBox:disabled, QDoubleSpinBox:disabled { color: #738294; }
QComboBox QAbstractItemView { background: #1b2836; selection-background-color: #34524e; }
QToolTip { background: #243344; color: white; border: 1px solid #4a5d71; padding: 8px; }
"""


def label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    item = QLabel(text)
    if name:
        item.setObjectName(name)
    item.setWordWrap(wrap)
    return item


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def toggle(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("toggle")
    button.setCheckable(True)
    button.setMinimumHeight(40)
    return button


def make_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#6fdfc0"))
    p.setPen(Qt.PenStyle.NoPen)
    from PySide6.QtGui import QPainterPath
    path = QPainterPath()
    path.moveTo(32, 5)
    path.lineTo(54, 14)
    path.lineTo(51, 36)
    path.cubicTo(49, 48, 38, 56, 32, 60)
    path.cubicTo(26, 56, 15, 48, 13, 36)
    path.lineTo(10, 14)
    path.closeSubpath()
    p.drawPath(path)
    p.setPen(QPen(QColor("#122c26"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(22, 31, 29, 38)
    p.drawLine(29, 38, 42, 24)
    p.end()
    return QIcon(pix)


class CameraView(QLabel):
    def __init__(self):
        super().__init__()
        self.source = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #0d141e; border: 1px solid #2c3a4c; border-radius: 12px; color: #899bad;")
        self.set_idle()

    def set_idle(self, message="摄像头尚未开启\n\n点击「开始监测」启用本地检测"):
        self.source = None
        self.clear()
        self.setText(message)

    def set_image(self, image):
        self.source = QPixmap.fromImage(image)
        self.refresh()

    def refresh(self):
        if self.source is not None:
            self.setPixmap(self.source.scaled(self.size() - QSize(8, 8), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


class BlackoutOverlay(QWidget):
    """A full-screen opaque blackout showing only the four characters 注意隐私.

    It reveals nothing about the scene (no count, no identity). Esc disarms
    until the user re-arms; any other key dismisses and leaves re-arming to the
    owner's policy. Keyboard input is grabbed so a key always reaches it.
    """

    def __init__(self, owner, geometry):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.owner = owner
        self.setObjectName("overlay")
        self.setGeometry(geometry)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.BlankCursor)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch()
        title = QLabel("注意隐私")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        size = max(72, min(geometry.width(), geometry.height()) // 6)
        font = QFont("Microsoft YaHei UI")
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, size * 0.14)
        title.setFont(font)
        # font-size must come from the stylesheet: the global QWidget rule would
        # otherwise override a size set via setFont().
        title.setStyleSheet(
            f"color: #f2f5f7; background: transparent;"
            f" font-size: {size}px; font-weight: 700;"
        )
        outer.addWidget(title)
        outer.addStretch()
        hint = QLabel("Esc 撤防（需手动重新布防） · 其他任意键暂离")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #39434d; font-size: 13px; background: transparent;")
        outer.addWidget(hint)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))

    def present(self):
        self.show()
        self.raise_()
        self.activateWindow()
        try:
            self.grabKeyboard()
        except Exception:
            pass

    def release(self):
        try:
            self.releaseKeyboard()
        except Exception:
            pass

    def keyPressEvent(self, event):
        self.owner.handle_blackout_key(event.key())

    def closeEvent(self, event):
        event.ignore()
        self.owner.handle_blackout_key(Qt.Key.Key_Space)


class RecordingController:
    """Start and stop a single silent recording session (camera, screen, audio).

    A session writes into one timestamped folder under the recordings folder so
    the camera clip, screen clip and microphone track from the same incident sit
    together. ``ensure`` makes the live session match what the policies want,
    restarting only when the shape of the video request (reason, camera, screen)
    changes; audio is added to or removed from the running session in place.
    """

    def __init__(self, window):
        self.window = window
        self.active = False
        self.reason = None
        self.dir = None
        self.screen = None
        self.audio = None
        self.has_camera = False
        self.started_at = 0.0

    def elapsed(self, now=None):
        if not self.active:
            return 0.0
        return (now if now is not None else time.monotonic()) - self.started_at

    def split(self, reason, camera, screen, audio=False):
        """Close the current session and open a fresh one (auto-segmentation)."""
        self.stop()
        self.start(reason, camera, screen)
        self._sync_audio(audio)

    def _has_screen(self):
        return self.screen is not None

    def ensure(self, want, reason, camera, screen, audio=False):
        if not want:
            if self.active:
                self.stop()
            return
        if not (
            self.active
            and self.reason == reason
            and self.has_camera == bool(camera and self.window.worker is not None)
            and self._has_screen() == bool(screen)
        ):
            if self.active:
                self.stop()
            self.start(reason, camera, screen)
        self._sync_audio(audio)

    def _sync_audio(self, want):
        if want and self.audio is None and self.dir is not None:
            self.audio = AudioRecorder(storage._unique(self.dir / "audio.wav"))
            self.audio.start()
        elif not want and self.audio is not None:
            self.audio.stop()
            self.audio = None

    def start(self, reason, camera, screen):
        base = self.window.recordings_base()
        storage.ensure_dirs(base)
        self.dir = storage.new_session_dir(reason, base)
        self.reason = reason
        self.has_camera = bool(camera and self.window.worker is not None)
        if self.has_camera:
            self.window.worker.set_camera_recording(True, self.dir)
        if screen:
            self.screen = ScreenRecorder(
                self.dir / "screen.mp4", grab_fn=self.window.screen_grab_fn
            )
            self.screen.start()
        self.started_at = time.monotonic()
        self.active = True

    def stop(self):
        if self.window.worker is not None:
            self.window.worker.set_camera_recording(False)
        if self.screen is not None:
            self.screen.stop()
            self.screen = None
        if self.audio is not None:
            self.audio.stop()
            self.audio = None
        self.active = False
        self.reason = None
        self.dir = None
        self.has_camera = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = storage.load_settings()
        self.worker = None
        self.active = False
        self.closing = False
        self.error_message = ""
        self.overlays = []
        self.overlay_is_test = False
        self.latest_count = 0
        self.alert_count = 0
        self.policy = AlertPolicy()

        # Arming, recording and guard state.
        self.arm = ArmState()
        self.rec = RecordingController(self)
        self.screen_grab_fn = None  # None -> ScreenRecorder uses the Windows GDI
        self.record_2p_off = float(self.settings.get("record_2p_off", 3.0))
        self.guard_off = float(self.settings.get("guard_off", 5.0))
        self.max_segment_min = float(self.settings.get("max_segment_min", 0.0))
        self.manual_stop_min = float(self.settings.get("manual_stop_min", 0.0))
        self.record_2p_trigger = HoldTrigger(on_seconds=0.6, off_seconds=self.record_2p_off)
        self.guard_trigger = HoldTrigger(on_seconds=2.0, off_seconds=self.guard_off)
        # Owner recognized alone for 1.5 s while a blackout is up -> close it.
        self.owner_clear_trigger = HoldTrigger(on_seconds=1.5, off_seconds=0.0)
        self.auto_close_owner = bool(self.settings.get("auto_close_owner", True))
        self.guard_snapshot = bool(self.settings.get("guard_snapshot", True))
        self._auto_closed = False
        self.owner_features = storage.load_owner()
        self.owner_present = None
        self.manual_recording = False
        self.enrolling = False
        self.enroll_samples = []

        self.auto_record_2p = bool(self.settings.get("auto_record_2p", True))
        self.record_screen_on_2p = bool(self.settings.get("record_screen_on_2p", False))
        self.guard_enabled = bool(self.settings.get("guard_enabled", False)) and bool(self.owner_features)
        self.rearm_seconds = float(self.settings.get("rearm_seconds", 60.0))
        self.record_audio = bool(self.settings.get("record_audio", False))
        self.snapshot_2p = bool(self.settings.get("snapshot_2p", True))
        self.snapshot_screen = bool(self.settings.get("snapshot_screen", True))
        self.snapshot_count = 0
        self.last_snapshot = ""
        self.paths_revealed = False  # file paths are hidden until unlocked
        self._io_threads = []
        # A hand-edited or stale profile must never merge the two folders.
        for key in ("snapshots_dir", "recordings_dir"):
            if storage.validate_dirs(self.recordings_base(), self.snapshots_base()):
                self.settings[key] = ""

        self.setWindowTitle("隐私卫士 · Privacy Guard")
        self.setWindowIcon(make_icon())
        self.resize(1180, 840)
        self.setMinimumSize(860, 560)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(196)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 30, 20, 24)
        brand = label("隐私卫士")
        brand.setStyleSheet("font-size: 23px; font-weight: 700;")
        side.addWidget(brand)
        side.addWidget(label("PRIVACY GUARD", "eyebrow"))
        side.addSpacing(37)
        self.nav_monitor = QPushButton("实时监测")
        self.nav_media = QPushButton("录像与设置")
        self.nav_help = QPushButton("使用说明")
        self.nav_buttons = [self.nav_monitor, self.nav_media, self.nav_help]
        for button in self.nav_buttons:
            button.setObjectName("nav")
            button.setCheckable(True)
            side.addWidget(button)
        self.nav_monitor.setChecked(True)
        side.addStretch()
        side.addWidget(label("●  离线检测", "eyebrow"))
        side.addSpacing(8)
        side.addWidget(label("画面仅在本机处理\n录像只存本机，不上传", "small", True))
        if storage.PORTABLE:
            side.addSpacing(8)
            side.addWidget(label("●  便携模式", "eyebrow"))
            side.addWidget(label("数据保存在程序目录的\ndata 文件夹", "small", True))
        side.addSpacing(20)
        side.addWidget(label("v2.4 · YuNet + SFace", "small"))
        shell.addWidget(sidebar)
        self.pages = QStackedWidget()
        shell.addWidget(self.pages, 1)
        for page in (self.build_monitor(), self.build_media(), self.build_help()):
            page.setObjectName("page")
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setStyleSheet("QScrollArea { background: #10151d; border: 0; }")
            scroll.setWidget(page)
            self.pages.addWidget(scroll)
        for index, button in enumerate(self.nav_buttons):
            button.clicked.connect(lambda _=False, i=index: self.navigate(i))
        self.build_tray()
        app = QApplication.instance()
        app.screenAdded.connect(self.on_screen_added)
        app.screenRemoved.connect(self.on_screen_removed)

        self.arm_timer = QTimer(self)
        self.arm_timer.setInterval(500)
        self.arm_timer.timeout.connect(self.arm_tick)
        self.arm_timer.start()
        self.enroll_timer = QTimer(self)
        self.enroll_timer.setSingleShot(True)
        self.enroll_timer.timeout.connect(lambda: self.finish_enroll())
        self.refresh_security_ui()

    def navigate(self, index):
        self.pages.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)

    def build_monitor(self):
        page = QWidget()
        content = QVBoxLayout(page)
        content.setContentsMargins(30, 27, 30, 22)
        content.setSpacing(19)
        header = QHBoxLayout()
        headings = QVBoxLayout()
        headings.addWidget(label("让屏幕上的隐私，多一份守护", "heading"))
        headings.addSpacing(5)
        headings.addWidget(label("多人时全屏「注意隐私」，并可无感录像、录音与抓拍取证。", "muted"))
        header.addLayout(headings)
        header.addStretch()
        self.tray_button = QPushButton("收起到托盘")
        self.tray_button.setObjectName("quiet")
        self.tray_button.clicked.connect(self.hide_to_tray)
        header.addWidget(self.tray_button)
        content.addLayout(header)

        metrics = QHBoxLayout()
        metrics.setSpacing(13)
        status_card = card()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 16, 20, 16)
        status_layout.addWidget(label("监测状态", "muted"))
        self.status_label = label("待开启", "status")
        status_layout.addWidget(self.status_label)
        self.status_hint = label("准备好后，开启摄像头", "small", True)
        status_layout.addWidget(self.status_hint)
        self.rec_label = label("", "rec")
        status_layout.addWidget(self.rec_label)
        metrics.addWidget(status_card, 2)
        for heading, value, note, attr in [
            ("当前人脸", "—", "两张及以上触发", "count_label"),
            ("本次提醒", "0", "全屏黑屏提示次数", "alerts_label"),
        ]:
            item = card()
            layout = QVBoxLayout(item)
            layout.setContentsMargins(20, 12, 20, 12)
            layout.addWidget(label(heading, "muted"))
            number = label(value, "number")
            setattr(self, attr, number)
            layout.addWidget(number)
            layout.addWidget(label(note, "small"))
            metrics.addWidget(item, 1)
        content.addLayout(metrics)

        middle = QHBoxLayout()
        middle.setSpacing(17)
        preview_card = card()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 15, 16, 13)
        row = QHBoxLayout()
        row.addWidget(label("摄像头预览"))
        row.addStretch()
        row.addWidget(label("LOCAL ONLY", "eyebrow"))
        preview_layout.addLayout(row)
        preview_layout.addSpacing(5)
        self.camera_view = CameraView()
        preview_layout.addWidget(self.camera_view, 1)
        preview_layout.addWidget(label("绿框为本人、红框为陌生人（开启卫兵模式后）。预览本身不保存。", "small", True))
        middle.addWidget(preview_card, 3)
        middle.addWidget(self.build_controls())
        content.addLayout(middle, 1)

        foot = card()
        foot_layout = QHBoxLayout(foot)
        foot_layout.setContentsMargins(17, 13, 17, 13)
        foot_layout.addWidget(label("i", "eyebrow"))
        foot_layout.addSpacing(7)
        foot_layout.addWidget(label("黑屏时：按 Esc 撤防（需手动重新布防），按其他任意键暂离并在设定时间后自动布防。\n录像为无感模式，不在桌面显示任何提示；文件带时间戳，仅保存在本机。", "muted", True), 1)
        content.addWidget(foot)
        return page

    def build_controls(self):
        controls = card()
        controls.setFixedWidth(286)
        settings = QVBoxLayout(controls)
        settings.setContentsMargins(19, 18, 19, 18)
        settings.setSpacing(10)

        settings.addWidget(label("监测"))
        settings.addWidget(label("摄像头", "muted"))
        self.camera_combo = QComboBox()
        for i in range(5):
            self.camera_combo.addItem(f"摄像头 {i}" + ("（默认）" if i == 0 else ""), i)
        self.camera_combo.setCurrentIndex(int(self.settings.get("camera_index", 0)) % 5)
        settings.addWidget(self.camera_combo)

        settings.addSpacing(6)
        settings.addWidget(label("布防", "eyebrow"))
        self.arm_label = label("布防中", "muted", True)
        settings.addWidget(self.arm_label)
        self.rearm_button = QPushButton("重新布防")
        self.rearm_button.clicked.connect(self.manual_arm)
        settings.addWidget(self.rearm_button)

        settings.addSpacing(6)
        settings.addWidget(label("卫兵模式（本人识别）", "eyebrow"))
        self.enroll_button = QPushButton("录入本人面孔")
        self.enroll_button.clicked.connect(self.start_enroll)
        settings.addWidget(self.enroll_button)
        self.enroll_status = label("", "small", True)
        settings.addWidget(self.enroll_status)
        self.guard_toggle = toggle("卫兵模式：非本人独自在前即录像")
        self.guard_toggle.clicked.connect(self.on_guard_toggled)
        settings.addWidget(self.guard_toggle)

        settings.addSpacing(6)
        settings.addWidget(label("快捷操作", "eyebrow"))
        self.manual_rec_toggle = toggle("手动录像（摄像头＋屏幕）")
        self.manual_rec_toggle.clicked.connect(self.on_manual_rec_toggled)
        settings.addWidget(self.manual_rec_toggle)
        self.snap_button = QPushButton("立即抓拍")
        self.snap_button.clicked.connect(self.manual_snapshot)
        settings.addWidget(self.snap_button)
        settings.addWidget(label("置信度、录像停止、录音、抓拍与保存位置见「录像与设置」页。", "small", True))

        settings.addSpacing(8)
        self.start_button = QPushButton("开始监测")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(43)
        self.start_button.clicked.connect(self.toggle_monitoring)
        settings.addWidget(self.start_button)
        self.test_button = QPushButton("测试黑屏提示")
        self.test_button.clicked.connect(lambda: self.show_alert(2, is_test=True))
        settings.addWidget(self.test_button)
        settings.addStretch()
        return controls

    def _spin(self, value, low, high, step, decimals, handler, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(handler)
        return spin

    def _field(self, text, widget):
        box = QVBoxLayout()
        box.setSpacing(4)
        box.addWidget(label(text, "muted"))
        box.addWidget(widget)
        return box

    def build_media(self):
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(30, 27, 30, 22)
        box.setSpacing(16)
        box.addWidget(label("录像与设置", "heading"))
        box.addWidget(label("可调参数与录像/抓拍选项都在这里；录像和抓拍分开存放，全部只保存在本机。", "muted"))

        # --- detection & arming knobs ---
        detect_card = card()
        detect = QVBoxLayout(detect_card)
        detect.setContentsMargins(20, 16, 20, 16)
        detect.setSpacing(10)
        detect.addWidget(label("检测与布防", "eyebrow"))
        grid = QHBoxLayout()
        grid.setSpacing(16)
        self.duration_combo = QComboBox()
        for text, value in [("0.8 秒 · 灵敏", 0.8), ("1.5 秒 · 标准", 1.5), ("2.5 秒 · 稳定", 2.5)]:
            self.duration_combo.addItem(text, value)
        target = float(self.settings.get("confirm_seconds", 0.8))
        for index in range(self.duration_combo.count()):
            if abs(self.duration_combo.itemData(index) - target) < 1e-6:
                self.duration_combo.setCurrentIndex(index)
        grid.addLayout(self._field("多人持续多久后提示", self.duration_combo), 1)
        self.confidence = self._spin(
            float(self.settings.get("confidence", 0.80)), 0.60, 0.95, 0.05, 2, self.on_confidence_changed)
        self.confidence.setToolTip("数值越高，误报越少，也可能漏检。默认 0.80。")
        grid.addLayout(self._field("检测置信度", self.confidence), 1)
        self.rearm_spin = self._spin(self.rearm_seconds, 5.0, 1800.0, 10.0, 0, self.on_rearm_changed, " 秒")
        grid.addLayout(self._field("黑屏暂离后自动布防", self.rearm_spin), 1)
        detect.addLayout(grid)
        self.autoclose_toggle = toggle("人走后只剩本人时自动关闭黑屏")
        self.autoclose_toggle.clicked.connect(self.on_autoclose_toggled)
        detect.addWidget(self.autoclose_toggle)
        detect.addWidget(label(
            "置信度与提示时长在下次「开始监测」时生效；自动布防时间即时生效。"
            "自动关闭需已录入本人面孔：其他人离开、画面中只剩被识别为本人的一人并持续约 1.5 秒后关闭；"
            "若只剩陌生人或无人，黑屏保持。", "small", True))
        box.addWidget(detect_card)

        row = QHBoxLayout()
        row.setSpacing(16)
        rec_card = card()
        rec = QVBoxLayout(rec_card)
        rec.setContentsMargins(20, 16, 20, 16)
        rec.setSpacing(10)
        rec.addWidget(label("无感录像", "eyebrow"))
        self.auto2p_toggle = toggle("检测到两人自动录像（摄像头）")
        self.auto2p_toggle.clicked.connect(self.on_auto2p_toggled)
        rec.addWidget(self.auto2p_toggle)
        self.screen2p_toggle = toggle("两人时同时录屏")
        self.screen2p_toggle.clicked.connect(self.on_screen2p_toggled)
        rec.addWidget(self.screen2p_toggle)
        self.audio_toggle = toggle("录像时同时录音（麦克风）")
        self.audio_toggle.clicked.connect(self.on_audio_toggled)
        rec.addWidget(self.audio_toggle)
        stops = QHBoxLayout()
        stops.setSpacing(14)
        self.rec2p_off_spin = self._spin(self.record_2p_off, 0.5, 60.0, 0.5, 1, self.on_rec2p_off_changed, " 秒")
        self.rec2p_off_spin.setToolTip("人数少于两人持续这么久后，停止两人录像。")
        stops.addLayout(self._field("两人录像停止延迟", self.rec2p_off_spin), 1)
        self.guard_off_spin = self._spin(self.guard_off, 0.5, 120.0, 0.5, 1, self.on_guard_off_changed, " 秒")
        self.guard_off_spin.setToolTip("独自陌生人消失持续这么久后，停止卫兵录像。")
        stops.addLayout(self._field("卫兵录像停止延迟", self.guard_off_spin), 1)
        rec.addLayout(stops)
        lengths = QHBoxLayout()
        lengths.setSpacing(14)
        self.max_seg_spin = self._spin(self.max_segment_min, 0.0, 240.0, 5.0, 0, self.on_max_segment_changed, " 分钟")
        self.max_seg_spin.setSpecialValueText("不限")
        self.max_seg_spin.setToolTip("录像达到该时长自动新建文件分段，避免单个文件过大。0 为不限。")
        lengths.addLayout(self._field("单段最长录像", self.max_seg_spin), 1)
        self.manual_stop_spin = self._spin(self.manual_stop_min, 0.0, 240.0, 5.0, 0, self.on_manual_stop_changed, " 分钟")
        self.manual_stop_spin.setSpecialValueText("手动")
        self.manual_stop_spin.setToolTip("手动录像达到该时长自动停止。0 为一直录到再次点击。")
        lengths.addLayout(self._field("手动录像自动停止", self.manual_stop_spin), 1)
        rec.addLayout(lengths)
        rec.addWidget(label(
            "卫兵模式与手动录像始终录摄像头＋屏幕；录音保存为同一文件夹内的 audio.wav，"
            "约在录像开始 0.5 秒后接入。首次使用请在 Windows 设置中允许访问麦克风。", "small", True))
        rec.addStretch()
        row.addWidget(rec_card, 1)

        snap_card = card()
        snap = QVBoxLayout(snap_card)
        snap.setContentsMargins(20, 16, 20, 16)
        snap.setSpacing(10)
        snap.addWidget(label("抓拍", "eyebrow"))
        self.snap2p_toggle = toggle("检测到两人自动抓拍")
        self.snap2p_toggle.clicked.connect(self.on_snapshot2p_toggled)
        snap.addWidget(self.snap2p_toggle)
        self.snapscreen_toggle = toggle("抓拍时同时截屏")
        self.snapscreen_toggle.clicked.connect(self.on_snapshot_screen_toggled)
        snap.addWidget(self.snapscreen_toggle)
        self.guardsnap_toggle = toggle("无人值守（卫兵）触发：先抓拍再录像")
        self.guardsnap_toggle.clicked.connect(self.on_guard_snapshot_toggled)
        snap.addWidget(self.guardsnap_toggle)
        snap.addWidget(label(
            "每次两人事件抓拍一次：摄像头照片（不带检测框）＋可选屏幕截图，均带时间戳。"
            "截屏在黑屏出现之前完成；撤防或暂离时照常抓拍。"
            "卫兵触发时先拍摄像头照片并截屏（始终截屏），之后才开始录像。", "small", True))
        self.snap_status = label("本次尚未抓拍", "muted", True)
        snap.addWidget(self.snap_status)
        snap.addStretch()
        row.addWidget(snap_card, 1)
        box.addLayout(row)

        paths_card = card()
        paths = QVBoxLayout(paths_card)
        paths.setContentsMargins(20, 16, 20, 16)
        paths.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label("保存位置", "eyebrow"))
        head.addStretch()
        self.reveal_button = QPushButton("显示路径")
        self.reveal_button.setObjectName("quiet")
        self.reveal_button.clicked.connect(self.toggle_reveal_paths)
        head.addWidget(self.reveal_button)
        paths.addLayout(head)
        for title, attr, choose, opener in [
            ("录像（摄像头 / 屏幕 / 录音）", "rec_path_label", self.choose_recordings_dir, self.open_recordings_folder),
            ("抓拍（摄像头照片 / 屏幕截图）", "snap_path_label", self.choose_snapshots_dir, self.open_snapshots_folder),
        ]:
            paths.addWidget(label(title, "muted"))
            line = QHBoxLayout()
            path_label = label("", "path", True)
            path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            setattr(self, attr, path_label)
            line.addWidget(path_label, 1)
            change = QPushButton("更改…")
            change.clicked.connect(choose)
            line.addWidget(change)
            open_button = QPushButton("打开")
            open_button.clicked.connect(opener)
            line.addWidget(open_button)
            paths.addLayout(line)
            paths.addSpacing(4)
        self.paths_status = label("", "error", True)
        paths.addWidget(self.paths_status)
        paths.addWidget(label(
            "正常状态下路径隐藏。两个文件夹必须不同；更改后新的录像与抓拍写入新位置，正在进行的录像仍写入原位置。", "small", True))
        box.addWidget(paths_card)

        sec_card = card()
        sec = QVBoxLayout(sec_card)
        sec.setContentsMargins(20, 16, 20, 16)
        sec.setSpacing(8)
        sec.addWidget(label("安全（二级密码）", "eyebrow"))
        self.pw_status = label("", "muted", True)
        sec.addWidget(self.pw_status)
        line = QHBoxLayout()
        self.pw_button = QPushButton("设置二级密码")
        self.pw_button.clicked.connect(self.set_or_change_password)
        line.addWidget(self.pw_button)
        self.clear_button = QPushButton("一键清理录像与截图")
        self.clear_button.clicked.connect(self.clear_media_action)
        line.addWidget(self.clear_button)
        line.addStretch()
        sec.addLayout(line)
        sec.addWidget(label(
            "设置后，更改/查看保存路径、打开文件夹、重新录入本人、关闭卫兵模式均需二级密码；"
            "「一键清理」额外需要人脸验证（须先开始监测、已录入本人）。清理会永久删除全部录像与截图，不可恢复。"
            "此为应用内保护，无法阻止直接访问磁盘文件的人。", "small", True))
        box.addWidget(sec_card)
        box.addStretch()
        return page

    def build_help(self):
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(34, 32, 34, 32)
        box.setSpacing(18)
        box.addWidget(label("使用说明", "heading"))
        for title, text in [
            ("01  开始监测", "选择摄像头并点击「开始监测」。确认摄像头面向使用者及身后的环境；首次使用请允许 Windows 桌面应用访问摄像头。"),
            ("02  两人触发全屏黑屏", "检测到两张或更多人脸并持续设定时长后，所有屏幕立即变为纯黑并显示「注意隐私」，不显示人数或任何画面内容。"),
            ("03  退出黑屏与布防", "已录入本人面孔时，其他人离开、画面中只剩被识别为本人的一人并持续约 1.5 秒后，黑屏自动关闭并保持布防（可在「录像与设置」关闭此功能）；只剩陌生人或无人时黑屏保持。也可手动：按 Esc 为撤防，之后需手动点击「重新布防」才会再次生效；按其他任意键为暂离，经设定时间（默认 60 秒，可调）后自动重新布防。"),
            ("04  无感录像", "开启「检测到两人自动录像」后，多人出现时自动录制摄像头；可选同时录屏。录像不在桌面显示任何提示，文件带时间戳并仅保存在本机。也可用「手动录像」随时开始。"),
            ("05  卫兵模式", "先「录入本人面孔」建立本人模板，再开启卫兵模式。当画面中只有一个非本人独自在屏幕前时（无人值守），先拍一张摄像头照片并截屏，再开始录制摄像头与屏幕，用于离开时取证。"),
            ("06  录音与抓拍", "在「录像与设置」页可开启录像时同时录音（audio.wav，与视频同一文件夹）。检测到两人时自动抓拍摄像头照片，并可同时截屏；截屏在黑屏出现前完成。也可点「立即抓拍」手动抓拍。"),
            ("07  录像何时停止", "两人录像在人数少于两人持续设定秒数后停止，卫兵录像在独自陌生人消失持续设定秒数后停止，手动录像需再次点击（或设定自动停止时间）。可设「单段最长录像」在超时后自动分段，避免单个文件过大。以上均在「录像与设置」页调节。"),
            ("08  保存位置", "录像与抓拍分开存放，均可在「录像与设置」页更改，两个文件夹必须不同。默认位于用户目录下 PrivacyGuard\\recordings 与 PrivacyGuard\\snapshots。"),
            ("09  二级密码与一键清理", "在「录像与设置」页可设置二级密码。设置后：更改/查看保存路径、打开文件夹、重新录入本人、关闭卫兵模式都需要密码；路径正常状态下隐藏，点「显示路径」并验证后才显示。「一键清理」会永久删除全部录像与截图，需要密码＋人脸验证（须先开始监测、已录入本人）。此为应用内保护，不能阻止直接访问磁盘文件的人。"),
            ("10  后台与退出", "点击「收起到托盘」可继续后台运行。右键托盘图标可恢复窗口、停止监测或退出。"),
            ("隐私与边界", "使用 OpenCV Zoo 的 YuNet 检测与 SFace 识别，全部在本机离线运行。本人面孔仅保存为本地特征向量，不保存原图，不上传。录音会录下他人声音，请遵守当地关于录音的法律规定。侧脸、口罩、遮挡、暗光可能漏检或影响识别；软件提供隐私辅助，不能保证阻止窥视。"),
        ]:
            section = card()
            layout = QVBoxLayout(section)
            layout.setContentsMargins(20, 15, 20, 15)
            layout.addWidget(label(title, "eyebrow"))
            layout.addSpacing(4)
            layout.addWidget(label(text, "muted", True))
            box.addWidget(section)
        box.addStretch()
        source = QLabel('<a style="color:#80dfc7" href="https://github.com/opencv/opencv_zoo">识别项目：OpenCV Zoo / YuNet + SFace（MIT 许可证） ↗</a>')
        source.setOpenExternalLinks(True)
        box.addWidget(source)
        return page

    def build_tray(self):
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_button.hide()
            return
        self.tray = QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip("隐私卫士 · 已停止")
        menu = QMenu(self)
        menu.addAction("显示主窗口", self.restore)
        menu.addAction("停止监测", self.stop_monitoring)
        menu.addSeparator()
        menu.addAction("退出", self.close)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.restore() if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick) else None)
        self.tray.show()

    def hide_to_tray(self):
        if self.tray:
            self.hide()

    def restore(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def set_controls(self, monitoring):
        for control in (self.camera_combo, self.duration_combo, self.confidence):
            control.setEnabled(not monitoring)

    # ---- settings persistence --------------------------------------------
    def save_current_settings(self):
        self.settings.update({
            "camera_index": int(self.camera_combo.currentData()),
            "confidence": float(self.confidence.value()),
            "confirm_seconds": float(self.duration_combo.currentData()),
            "auto_record_2p": bool(self.auto_record_2p),
            "record_screen_on_2p": bool(self.record_screen_on_2p),
            "guard_enabled": bool(self.guard_enabled),
            "rearm_seconds": float(self.rearm_seconds),
            "record_audio": bool(self.record_audio),
            "record_2p_off": float(self.record_2p_off),
            "guard_off": float(self.guard_off),
            "max_segment_min": float(self.max_segment_min),
            "manual_stop_min": float(self.manual_stop_min),
            "snapshot_2p": bool(self.snapshot_2p),
            "snapshot_screen": bool(self.snapshot_screen),
            "auto_close_owner": bool(self.auto_close_owner),
            "guard_snapshot": bool(self.guard_snapshot),
        })
        storage.save_settings(self.settings)

    # ---- monitoring lifecycle --------------------------------------------
    def toggle_monitoring(self):
        if self.worker is not None:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        if self.worker is not None or self.closing:
            return
        if not MODEL.is_file() or hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
            self.on_error("识别模型缺失或校验失败。请重新解压完整软件包。")
            return
        self.dismiss_alert(mark_incident=False)
        self.policy = AlertPolicy(confirm_seconds=float(self.duration_combo.currentData()))
        self.record_2p_trigger.reset()
        self.guard_trigger.reset()
        self.error_message = ""
        self.active = True
        self.set_controls(True)
        self.start_button.setText("停止监测")
        self.start_button.setEnabled(True)
        self.status_label.setText("正在连接")
        self.status_hint.setText("正在打开摄像头，请稍候")
        self.camera_view.set_idle("正在连接摄像头…")
        recognizer_path = SFACE_MODEL if SFACE_MODEL.is_file() else None
        self.worker = CameraWorker(
            int(self.camera_combo.currentData()), MODEL, self.confidence.value(), recognizer_path
        )
        self.worker.set_guard(self.guard_enabled, self.owner_features, OWNER_MATCH_THRESHOLD)
        self.worker.opened.connect(self.on_opened)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.analysis.connect(self.on_analysis)
        self.worker.owner_sample.connect(self.on_owner_sample)
        self.worker.snapshot_saved.connect(self.on_snapshot_saved)
        self.worker.failed.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.save_current_settings()
        self.refresh_security_ui()

    def on_opened(self):
        if not self.active:
            return
        self.status_label.setText("监测中")
        self.status_hint.setText("画面仅在本机分析")
        if self.tray:
            self.tray.setToolTip("隐私卫士 · 正在监测")

    def on_frame(self, image, count, timestamp):
        if not self.active or time.monotonic() - timestamp > 1.0:
            return
        self.latest_count = count
        self.count_label.setText(str(count))
        self.camera_view.set_image(image)
        should_alert = self.policy.update(count, timestamp)
        if should_alert:
            if self.snapshot_2p:
                # Before the blackout opens, so the screenshot shows the real screen.
                self.take_snapshot("多人")
            if self.arm.armed:
                if self.overlays:
                    self.dismiss_alert(mark_incident=False)
                self.show_alert(count)
            else:
                # Disarmed or snoozed: acknowledge silently, no blackout shown.
                self.policy.dismiss()
        elif self.overlays and not self.overlay_is_test and self.auto_close_owner and self.owner_features:
            # The other person left and the one at the screen is the owner:
            # close the blackout once that holds steadily.
            owner_alone = count == 1 and self.owner_present is True
            if self.owner_clear_trigger.update(owner_alone, timestamp):
                self.auto_close_blackout()
        if self.auto_record_2p:
            self.record_2p_trigger.update(count >= 2, timestamp)
        self.update_recording()
        self.refresh_status_text()

    def on_analysis(self, owner_present, stranger_alone, timestamp):
        if not self.active or time.monotonic() - timestamp > 1.0:
            return
        self.owner_present = owner_present
        if self.guard_enabled:
            was_active = self.guard_trigger.active
            self.guard_trigger.update(bool(stranger_alone), timestamp)
            if self.guard_trigger.active and not was_active and self.guard_snapshot:
                # Unattended: photo + screenshot first, recording afterwards.
                self.take_snapshot("卫兵", screen=True)
            self.update_recording()
        self.refresh_status_text()

    def auto_close_blackout(self):
        """Close a live blackout because only the verified owner remains."""
        self.dismiss_alert(mark_incident=True)
        self.owner_clear_trigger.reset()
        self._auto_closed = True

    def on_owner_sample(self, feature):
        if not self.enrolling or feature is None:
            return
        self.enroll_samples.append(np.asarray(feature, dtype=np.float32).reshape(1, -1))
        self.enroll_status.setText(f"正在录入本人面孔… {len(self.enroll_samples)}/{ENROLL_TARGET}")
        if len(self.enroll_samples) >= ENROLL_TARGET:
            self.finish_enroll()

    def on_error(self, message):
        self.error_message = message
        self.active = False
        self.policy.reset()
        self.record_2p_trigger.reset()
        self.guard_trigger.reset()
        self.rec.stop()
        self.manual_recording = False
        self.dismiss_alert(mark_incident=False)
        self.count_label.setText("—")
        self.status_label.setText("监测未启动")
        self.status_hint.setText(message)
        self.status_hint.setToolTip(message)
        self.camera_view.set_idle(message)
        if self.tray:
            self.tray.setToolTip("隐私卫士 · 监测不可用")
            if not self.isVisible():
                self.restore()
        if self.worker:
            self.start_button.setEnabled(False)
            self.worker.stop()
        else:
            self.set_controls(False)
            self.start_button.setText("重新开始")
            self.start_button.setEnabled(True)
        self.refresh_security_ui()

    def stop_monitoring(self):
        self.active = False
        self.policy.reset()
        self.record_2p_trigger.reset()
        self.guard_trigger.reset()
        self.rec.stop()
        self.manual_recording = False
        self.dismiss_alert(mark_incident=False)
        if self.enrolling:
            self.finish_enroll()
        if self.worker is not None:
            if not self.error_message:
                self.status_label.setText("正在停止")
                self.status_hint.setText("正在释放摄像头，请稍候")
            self.start_button.setText("正在停止…")
            self.start_button.setEnabled(False)
            self.worker.stop()
        self.refresh_security_ui()

    def on_worker_finished(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.active = False
        self.policy.reset()
        self.record_2p_trigger.reset()
        self.guard_trigger.reset()
        self.rec.stop()
        self.manual_recording = False
        self.dismiss_alert(mark_incident=False)
        self.set_controls(False)
        self.start_button.setText("重新开始" if self.error_message else "开始监测")
        self.start_button.setEnabled(True)
        self.count_label.setText("—")
        if not self.error_message:
            self.status_label.setText("已停止")
            self.status_hint.setText("摄像头已释放")
            self.camera_view.set_idle()
        if self.tray:
            self.tray.setToolTip("隐私卫士 · 已停止" if not self.error_message else "隐私卫士 · 监测不可用")
        self.refresh_security_ui()
        if self.closing:
            self.close()

    # ---- blackout overlay -------------------------------------------------
    def show_alert(self, count, is_test=False):
        if self.overlays:
            for overlay in self.overlays:
                overlay.raise_()
            return
        self.overlay_is_test = is_test
        if not is_test:
            self.alert_count += 1
            self.alerts_label.setText(str(self.alert_count))
            # Start fresh: identity must be re-confirmed after this blackout opens.
            self.owner_present = None
            self.owner_clear_trigger.reset()
            self._auto_closed = False
            if self.worker is not None and self.owner_features and self.auto_close_owner:
                self.worker.set_identify(True)
        for screen in QApplication.screens():
            self.add_overlay(screen)
        if self.overlays:
            self.overlays[0].present()

    def add_overlay(self, screen):
        overlay = BlackoutOverlay(self, screen.geometry())
        screen.geometryChanged.connect(overlay.setGeometry)
        self.overlays.append(overlay)
        overlay.present()

    def on_screen_added(self, screen):
        if self.overlays:
            self.add_overlay(screen)

    def on_screen_removed(self, screen):
        if self.overlays:
            is_test = self.overlay_is_test
            self.dismiss_alert(mark_incident=False)
            self.overlay_is_test = is_test
            for remaining in QApplication.screens():
                self.add_overlay(remaining)

    def handle_blackout_key(self, key):
        is_test = self.overlay_is_test
        if not is_test:
            now = time.monotonic()
            if key == Qt.Key.Key_Escape:
                self.arm.disarm()
            else:
                self.arm.snooze(now, self.rearm_seconds)
        self.dismiss_alert(mark_incident=not is_test)
        self.refresh_arm_status()

    def dismiss_alert(self, mark_incident=True):
        if mark_incident and not self.overlay_is_test:
            self.policy.dismiss()
        for overlay in self.overlays:
            overlay.release()
            overlay.hide()
            overlay.deleteLater()
        self.overlays.clear()
        self.overlay_is_test = False
        if self.worker is not None:
            self.worker.set_identify(False)

    # ---- arming -----------------------------------------------------------
    def manual_arm(self):
        self.arm.arm()
        self.refresh_arm_status()

    def arm_tick(self):
        self.arm.tick(time.monotonic())
        self.refresh_arm_status()
        self.maintain_recording()

    def on_rearm_changed(self, value):
        self.rearm_seconds = float(value)
        self.save_current_settings()

    def on_confidence_changed(self, value):
        self.save_current_settings()

    # ---- recording stop / segmentation settings ---------------------------
    def on_rec2p_off_changed(self, value):
        self.record_2p_off = float(value)
        self.record_2p_trigger.off_seconds = self.record_2p_off
        self.save_current_settings()

    def on_guard_off_changed(self, value):
        self.guard_off = float(value)
        self.guard_trigger.off_seconds = self.guard_off
        self.save_current_settings()

    def on_max_segment_changed(self, value):
        self.max_segment_min = float(value)
        self.save_current_settings()

    def on_manual_stop_changed(self, value):
        self.manual_stop_min = float(value)
        self.save_current_settings()

    def maintain_recording(self):
        """Enforce the max-length split and the manual auto-stop (timer-driven)."""
        if not self.rec.active:
            return
        now = time.monotonic()
        elapsed = self.rec.elapsed(now)
        if (
            self.manual_recording
            and self.rec.reason == "手动"
            and self.manual_stop_min > 0
            and elapsed >= self.manual_stop_min * 60
        ):
            self.manual_recording = False
            if hasattr(self, "manual_rec_toggle"):
                self.manual_rec_toggle.setChecked(False)
            self.update_recording()
            return
        if self.max_segment_min > 0 and elapsed >= self.max_segment_min * 60:
            want, reason, camera, screen = self.desired_recording()
            if want:
                self.rec.split(reason, camera, screen, audio=self.record_audio)
                self.refresh_record_status()

    # ---- recording orchestration -----------------------------------------
    def desired_recording(self):
        if self.manual_recording:
            return True, "手动", True, True
        if self.guard_enabled and self.guard_trigger.active:
            return True, "卫兵", True, True
        if self.auto_record_2p and self.record_2p_trigger.active:
            return True, "多人", True, self.record_screen_on_2p
        return False, None, False, False

    def update_recording(self):
        want, reason, camera, screen = self.desired_recording()
        self.rec.ensure(want, reason, camera, screen, audio=self.record_audio)
        self.refresh_record_status()

    def on_audio_toggled(self):
        self.record_audio = self.audio_toggle.isChecked()
        self.update_recording()
        self.save_current_settings()

    # ---- snapshots --------------------------------------------------------
    def take_snapshot(self, reason, screen=None):
        """Photograph the camera (via the worker) and optionally the screen.

        ``screen`` forces the screenshot on/off; ``None`` follows the
        「抓拍时同时截屏」 toggle. Returns whether anything was captured. The
        screen is grabbed right here, synchronously, so a blackout or a
        recording started afterwards comes strictly later; only the JPEG
        encoding and disk write run on a background thread.
        """
        camera_path, screen_path = storage.snapshot_paths(reason, self.snapshots_base())
        took = False
        if self.worker is not None and self.active:
            self.worker.request_snapshot(camera_path, "CAM")
            took = True
        if self.snapshot_screen if screen is None else screen:
            try:
                image = (self.screen_grab_fn or grab_screen_gdi)()
            except Exception:
                image = None
            if image is not None:
                self._run_io(save_image, screen_path, image, "SCREEN")
                took = True
        if took:
            self.snapshot_count += 1
            self.last_snapshot = datetime.now().strftime("%H:%M:%S")
        self.refresh_snapshot_status()
        return took

    def manual_snapshot(self):
        if not self.take_snapshot("手动"):
            self.snap_status.setText("未抓拍：请先「开始监测」，或开启「抓拍时同时截屏」。")

    def on_snapshot_saved(self, path, ok):
        if not ok:
            self.snap_status.setText("摄像头照片保存失败，请检查抓拍文件夹是否可写。")

    def on_snapshot2p_toggled(self):
        self.snapshot_2p = self.snap2p_toggle.isChecked()
        self.save_current_settings()

    def on_snapshot_screen_toggled(self):
        self.snapshot_screen = self.snapscreen_toggle.isChecked()
        self.save_current_settings()

    def on_guard_snapshot_toggled(self):
        self.guard_snapshot = self.guardsnap_toggle.isChecked()
        self.save_current_settings()

    def on_autoclose_toggled(self):
        self.auto_close_owner = self.autoclose_toggle.isChecked()
        if self.worker is not None and self.overlays and not self.overlay_is_test:
            self.worker.set_identify(self.auto_close_owner and bool(self.owner_features))
        self.owner_clear_trigger.reset()
        self.save_current_settings()

    def refresh_snapshot_status(self):
        if not hasattr(self, "snap_status"):
            return
        if self.snapshot_count:
            self.snap_status.setText(f"本次已抓拍 {self.snapshot_count} 次 · 最近 {self.last_snapshot}")
        else:
            self.snap_status.setText("本次尚未抓拍")

    def _run_io(self, fn, *args):
        def job():
            try:
                fn(*args)
            except Exception:
                pass
        self._io_threads = [t for t in self._io_threads if t.is_alive()]
        thread = threading.Thread(target=job, name="snapshot-writer", daemon=True)
        thread.start()
        self._io_threads.append(thread)

    def wait_io(self, timeout=5.0):
        for thread in self._io_threads:
            thread.join(timeout)
        self._io_threads = []

    # ---- save locations ---------------------------------------------------
    def recordings_base(self):
        return Path(self.settings.get("recordings_dir") or storage.RECORDINGS_DIR)

    def snapshots_base(self):
        return Path(self.settings.get("snapshots_dir") or storage.SNAPSHOTS_DIR)

    def set_recordings_dir(self, folder):
        return self._set_dir("recordings_dir", Path(folder), self.snapshots_base())

    def set_snapshots_dir(self, folder):
        return self._set_dir("snapshots_dir", Path(folder), self.recordings_base())

    def _set_dir(self, key, folder, other):
        error = storage.validate_dirs(folder, other) or storage.check_writable(folder)
        if error:
            self.paths_status.setText(error)
            return False
        self.settings[key] = str(folder)
        self.paths_status.setText("")
        self.save_current_settings()
        self.refresh_paths()
        return True

    def choose_recordings_dir(self):
        if not self.require_password("更改保存位置 · 请输入二级密码"):
            return
        folder = QFileDialog.getExistingDirectory(self, "选择录像保存位置", str(self.recordings_base()))
        if folder:
            self.set_recordings_dir(folder)

    def choose_snapshots_dir(self):
        if not self.require_password("更改保存位置 · 请输入二级密码"):
            return
        folder = QFileDialog.getExistingDirectory(self, "选择抓拍保存位置", str(self.snapshots_base()))
        if folder:
            self.set_snapshots_dir(folder)

    def refresh_paths(self):
        if not hasattr(self, "rec_path_label"):
            return
        for path_label, base in ((self.rec_path_label, self.recordings_base()),
                                 (self.snap_path_label, self.snapshots_base())):
            if not self.paths_revealed:
                path_label.setText("●●●●●●（已隐藏）")
                path_label.setToolTip("")
                continue
            text = str(base)
            # Zero-width spaces let long paths wrap at separators.
            path_label.setText(text.replace("\\", "\\​").replace("/", "/​"))
            path_label.setToolTip(text)
        if hasattr(self, "reveal_button"):
            self.reveal_button.setText("隐藏路径" if self.paths_revealed else "显示路径")

    def toggle_reveal_paths(self):
        if self.paths_revealed:
            self.paths_revealed = False
        else:
            if not self.require_password("显示保存路径 · 请输入二级密码"):
                return
            self.paths_revealed = True
        self.refresh_paths()

    # ---- secondary password & protected actions ---------------------------
    def _info(self, message, title="隐私卫士"):
        QMessageBox.information(self, title, message)

    def _confirm(self, message, title="请确认"):
        reply = QMessageBox.question(
            self, title, message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def has_password(self):
        return storage.has_password(self.settings)

    def ask_password(self, title):
        text, ok = QInputDialog.getText(
            self, "二级密码", title, QLineEdit.EchoMode.Password)
        return text if ok else None

    def require_password(self, reason):
        """Gate a sensitive action behind the secondary password."""
        if not self.has_password():
            self._info("此操作受二级密码保护。请先在「录像与设置」页设置二级密码。")
            return False
        password = self.ask_password(reason)
        if password is None:
            return False
        if not storage.verify_password(self.settings, password):
            self._info("二级密码错误。")
            return False
        return True

    def set_or_change_password(self):
        if self.has_password():
            current = self.ask_password("修改二级密码 · 请输入当前密码")
            if current is None:
                return
            if not storage.verify_password(self.settings, current):
                self._info("当前密码错误。")
                return
        first, ok = QInputDialog.getText(
            self, "二级密码", "请输入新的二级密码（至少 4 位）：", QLineEdit.EchoMode.Password)
        if not ok:
            return
        if len(first) < 4:
            self._info("密码至少需要 4 位。")
            return
        second, ok = QInputDialog.getText(
            self, "二级密码", "请再次输入新密码：", QLineEdit.EchoMode.Password)
        if not ok:
            return
        if first != second:
            self._info("两次输入不一致。")
            return
        storage.set_password(self.settings, first)
        self.save_current_settings()
        self.refresh_security_ui()
        self._info("二级密码已设置。敏感操作（更改/查看路径、录入本人、关闭卫兵、一键清理）将需要此密码。")

    def verify_owner_live(self, timeout_ms=6000):
        """Confirm the enrolled owner is on camera now; None if it can't run."""
        if self.worker is None or not self.active or not self.owner_features:
            return None
        loop = QEventLoop()
        outcome = {"ok": None}

        def on_verified(ok):
            outcome["ok"] = bool(ok)
            loop.quit()

        self.worker.owner_verified.connect(on_verified)
        self.worker.request_verify()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        loop.exec()
        try:
            self.worker.owner_verified.disconnect(on_verified)
        except Exception:
            pass
        return outcome["ok"]

    def clear_media_action(self):
        """One-click clear of all recordings and snapshots (password + face)."""
        if not self.require_password("一键清理 · 请输入二级密码"):
            return
        verified = self.verify_owner_live()
        if verified is None:
            self._info("一键清理需要人脸验证：请先「开始监测」并确认已录入本人面孔。")
            return
        if not verified:
            self._info("人脸验证未通过，已取消清理。")
            return
        if not self._confirm("将永久删除全部录像与截图，且无法恢复。确定清理吗？"):
            return
        self.manual_recording = False
        if hasattr(self, "manual_rec_toggle"):
            self.manual_rec_toggle.setChecked(False)
        self.rec.stop()
        removed, failed = storage.clear_media(self.recordings_base(), self.snapshots_base())
        message = f"已清理 {removed} 项录像/截图。"
        if failed:
            message += f"（{failed} 项删除失败，可能正被占用）"
        self._info(message)

    def on_auto2p_toggled(self):
        self.auto_record_2p = self.auto2p_toggle.isChecked()
        if not self.auto_record_2p:
            self.record_2p_trigger.reset()
        self.update_recording()
        self.save_current_settings()

    def on_screen2p_toggled(self):
        self.record_screen_on_2p = self.screen2p_toggle.isChecked()
        self.update_recording()
        self.save_current_settings()

    def on_manual_rec_toggled(self):
        self.manual_recording = self.manual_rec_toggle.isChecked()
        self.update_recording()

    def open_recordings_folder(self):
        if not self.require_password("打开录像文件夹 · 请输入二级密码"):
            return
        self.open_folder(self.recordings_base())

    def open_snapshots_folder(self):
        if not self.require_password("打开抓拍文件夹 · 请输入二级密码"):
            return
        self.open_folder(self.snapshots_base())

    def open_folder(self, folder):
        storage.ensure_dirs(folder)
        path = str(folder)
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    # ---- guard enrollment -------------------------------------------------
    def start_enroll(self):
        if not SFACE_MODEL.is_file():
            self.enroll_status.setText("缺少人脸识别模型，无法使用卫兵模式。")
            return
        if self.worker is None or not self.active:
            self.enroll_status.setText("请先「开始监测」，再录入本人面孔。")
            return
        # Re-enrolling replaces the owner template, so protect it once a
        # password exists (an intruder must not swap in their own face).
        if self.owner_features and self.has_password():
            if not self.require_password("重新录入本人面孔 · 请输入二级密码"):
                return
        self.enrolling = True
        self.enroll_samples = []
        self.worker.set_enrolling(True)
        self.enroll_status.setText("正在录入本人面孔…请正对摄像头，缓慢转动头部。")
        self.enroll_button.setEnabled(False)
        self.enroll_timer.start(15000)

    def finish_enroll(self):
        if not self.enrolling:
            return
        self.enrolling = False
        self.enroll_timer.stop()
        if self.worker is not None:
            self.worker.set_enrolling(False)
        if len(self.enroll_samples) >= 3:
            self.owner_features = self.enroll_samples[-ENROLL_TARGET:]
            storage.save_owner(self.owner_features)
            if self.worker is not None:
                self.worker.set_guard(self.guard_enabled, self.owner_features, OWNER_MATCH_THRESHOLD)
            self.enroll_status.setText(f"本人面孔已录入（{len(self.owner_features)} 张特征）。")
        else:
            self.enroll_status.setText("录入失败：未能稳定检测到正脸，请在光线良好处重试。")
        self.enroll_samples = []
        self.enroll_button.setEnabled(True)
        self.refresh_security_ui()

    def on_guard_toggled(self):
        want = self.guard_toggle.isChecked()
        if want and not self.owner_features:
            self.guard_toggle.setChecked(False)
            self.enroll_status.setText("请先「录入本人面孔」，再开启卫兵模式。")
            return
        # Turning protection off is sensitive: gate it behind the password.
        if not want and self.guard_enabled and self.has_password():
            if not self.require_password("关闭卫兵模式 · 请输入二级密码"):
                self.guard_toggle.setChecked(True)
                return
        self.guard_enabled = want
        if self.worker is not None:
            self.worker.set_guard(want, self.owner_features, OWNER_MATCH_THRESHOLD)
        if not want:
            self.guard_trigger.reset()
            self.update_recording()
        self.save_current_settings()
        self.refresh_security_ui()

    # ---- status refresh ---------------------------------------------------
    def refresh_arm_status(self):
        if not hasattr(self, "arm_label"):
            return
        if self.arm.state == ArmState.ARMED:
            self.arm_label.setText("布防中 · 多人将触发黑屏")
            self.rearm_button.setEnabled(False)
        elif self.arm.state == ArmState.DISARMED:
            self.arm_label.setText("已撤防 · 需手动重新布防")
            self.rearm_button.setEnabled(True)
        else:
            remaining = int(round(self.arm.remaining(time.monotonic())))
            self.arm_label.setText(f"暂离 · 约 {remaining} 秒后自动布防")
            self.rearm_button.setEnabled(True)

    def refresh_record_status(self):
        if not hasattr(self, "rec_label"):
            return
        if self.rec.active:
            parts = [self.rec.reason]
            audio = self.rec.audio
            if audio is not None:
                parts.append("录音不可用" if audio.error else "录音")
            self.rec_label.setText(f"● 正在无感录像（{'·'.join(parts)}）")
            self.rec_label.setToolTip(audio.error if audio is not None and audio.error else "")
        else:
            self.rec_label.setText("")
            self.rec_label.setToolTip("")

    def refresh_status_text(self):
        if not self.active:
            return
        if self.guard_enabled and self.guard_trigger.active:
            self.status_label.setText("卫兵取证中")
            self.status_hint.setText("检测到非本人独自在前，正在录像")
        elif self.policy.alert_visible:
            self.status_label.setText("多人提醒")
        else:
            self.status_label.setText("监测中")
            if self.policy.incident_latched and self._auto_closed:
                self.status_hint.setText("本人已在场 · 黑屏已自动关闭")
            elif self.policy.incident_latched:
                self.status_hint.setText("本次已关闭 · 等待人数恢复")
            elif self.latest_count >= 2:
                self.status_hint.setText("发现多张人脸，请留意隐私")
            else:
                self.status_hint.setText("画面仅在本机分析")

    def refresh_security_ui(self):
        has_owner = bool(self.owner_features)
        if hasattr(self, "guard_toggle"):
            self.guard_toggle.setChecked(self.guard_enabled)
            self.guard_toggle.setEnabled(has_owner or self.guard_enabled)
            if not has_owner and not self.enroll_status.text():
                self.enroll_status.setText("尚未录入本人面孔。")
            elif has_owner and self.enroll_status.text() in ("", "尚未录入本人面孔。"):
                self.enroll_status.setText(f"已录入本人面孔（{len(self.owner_features)} 张特征）。")
        if hasattr(self, "auto2p_toggle"):
            self.auto2p_toggle.setChecked(self.auto_record_2p)
            self.screen2p_toggle.setChecked(self.record_screen_on_2p)
            self.audio_toggle.setChecked(self.record_audio)
            self.snap2p_toggle.setChecked(self.snapshot_2p)
            self.snapscreen_toggle.setChecked(self.snapshot_screen)
            self.guardsnap_toggle.setChecked(self.guard_snapshot)
            self.autoclose_toggle.setChecked(self.auto_close_owner)
            self.manual_rec_toggle.setChecked(self.manual_recording)
        if hasattr(self, "pw_button"):
            if self.has_password():
                self.pw_button.setText("修改二级密码")
                self.pw_status.setText("已设置二级密码 · 敏感操作受保护。")
            else:
                self.pw_button.setText("设置二级密码")
                self.pw_status.setText("尚未设置。设置后可保护路径、清理等敏感操作。")
        self.refresh_arm_status()
        self.refresh_record_status()
        self.refresh_paths()

    def closeEvent(self, event: QCloseEvent):
        if self.worker is not None:
            self.closing = True
            self.stop_monitoring()
            event.ignore()
            return
        self.rec.stop()
        self.wait_io()
        self.save_current_settings()
        self.dismiss_alert(mark_incident=False)
        if self.tray:
            self.tray.hide()
        event.accept()
        QApplication.instance().quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description="隐私卫士 / 本地摄像头多人提醒与无感录像")
    parser.add_argument("--screenshot", type=Path, help="渲染主界面到 PNG 后退出；不会打开摄像头")
    parser.add_argument("--page", type=int, default=0, help="配合 --screenshot：0 实时监测，1 录像与设置，2 使用说明")
    parser.add_argument("--alert-screenshot", type=Path, help="渲染全屏黑屏提示到 PNG 后退出；不会打开摄像头")
    parser.add_argument("--smoke-test", action="store_true", help="加载识别模型和界面后退出；不会打开摄像头")
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    if not QFontDatabase.families() and sys.platform == "win32":
        # Qt's headless platform does not enumerate installed Windows fonts.
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf"):
            QFontDatabase.addApplicationFont(str(fonts / name))
    app.setApplicationName("PrivacyGuard")
    app.setApplicationDisplayName("隐私卫士")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    if args.smoke_test:
        from .detection import FaceCounter
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("Model checksum failed")
        faces = FaceCounter(MODEL).detect(np.zeros((480, 640, 3), dtype=np.uint8))
        if len(faces) != 0:
            raise AssertionError("Blank frame must contain no faces")
        QTimer.singleShot(100, window.close)
    if args.screenshot or args.alert_screenshot:
        def save_screenshot():
            target = args.screenshot or args.alert_screenshot
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.alert_screenshot:
                window.show_alert(2, is_test=True)
                app.processEvents()
                shot = window.overlays[0].grab()
            else:
                window.navigate(max(0, min(args.page, window.pages.count() - 1)))
                app.processEvents()
                shot = window.grab()
            ok = shot.save(str(target))
            window.close()
            if not ok:
                app.exit(1)
        QTimer.singleShot(200, save_screenshot)
    return app.exec()
