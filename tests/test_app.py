"""Offscreen UI integration tests; fakes stand in for camera, screen and disk."""

import os
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from privacy_guard import app as app_module
from privacy_guard import storage as storage_module


class FakeCameraWorker(QObject):
    """Expose the real worker's signals and setters without a thread or device."""

    opened = Signal()
    frame_ready = Signal(object, int, float)
    analysis = Signal(bool, bool, float)
    owner_sample = Signal(object)
    snapshot_saved = Signal(str, bool)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, camera_index, model_path, confidence, recognizer_path=None):
        super().__init__()
        self.camera_index = camera_index
        self.model_path = model_path
        self.confidence = confidence
        self.recognizer_path = recognizer_path
        self.started = False
        self.stop_calls = 0
        self.guard = None
        self.enrolling = False
        self.recording = []  # list of (on, dir)
        self.snapshots = []  # requested camera snapshot paths

    def request_snapshot(self, path, label="CAM"):
        self.snapshots.append(Path(path))

    identify = False

    def set_identify(self, on):
        self.identify = bool(on)

    def start(self):
        self.started = True

    def stop(self):
        self.stop_calls += 1

    def set_guard(self, enabled, owner_features=None, threshold=None):
        self.guard = (enabled, list(owner_features or []), threshold)

    def set_enrolling(self, enrolling):
        self.enrolling = enrolling

    def set_camera_recording(self, on, session_dir=None):
        self.recording.append((bool(on), session_dir))


class FakeScreenRecorder:
    instances = []

    def __init__(self, path, fps=8.0, grab_fn=None, label="屏幕"):
        self.path = path
        self.grab_fn = grab_fn
        self.started = False
        self.stopped = False
        FakeScreenRecorder.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, timeout=3.0):
        self.stopped = True

    @property
    def active(self):
        return self.started and not self.stopped


class FakeAudioRecorder:
    instances = []
    default_error = ""

    def __init__(self, path, sample_rate=16000, channels=1, source_factory=None):
        self.path = Path(path)
        self.error = FakeAudioRecorder.default_error
        self.started = False
        self.stopped = False
        FakeAudioRecorder.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, timeout=3.0):
        self.stopped = True


class FakeScreen(QObject):
    geometryChanged = Signal(QRect)

    def __init__(self, x=0):
        super().__init__()
        self.rect = QRect(x, 0, 800, 600)

    def geometry(self):
        return self.rect


class MainWindowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])
        cls.qt_app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        for name, value in {
            "DATA_DIR": base,
            "RECORDINGS_DIR": base / "recordings",
            "SNAPSHOTS_DIR": base / "snapshots",
            "OWNER_FILE": base / "owner.npy",
            "SETTINGS_FILE": base / "settings.json",
        }.items():
            patcher = patch.object(storage_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

        # Safety net: no modal dialog may block the offscreen suite.
        for name in ("information", "warning", "critical"):
            p = patch.object(app_module.QMessageBox, name, return_value=None)
            p.start(); self.addCleanup(p.stop)
        q = patch.object(app_module.QMessageBox, "question",
                         return_value=app_module.QMessageBox.StandardButton.No)
        q.start(); self.addCleanup(q.stop)
        g = patch.object(app_module.QInputDialog, "getText", return_value=("", False))
        g.start(); self.addCleanup(g.stop)

        self.camera_guard = patch(
            "privacy_guard.detection.cv2.VideoCapture",
            side_effect=AssertionError("UI tests must never access a camera"),
        )
        self.camera_guard.start()
        self.addCleanup(self.camera_guard.stop)
        self.worker_patch = patch.object(app_module, "CameraWorker", FakeCameraWorker)
        self.worker_patch.start()
        self.addCleanup(self.worker_patch.stop)
        FakeScreenRecorder.instances = []
        self.screen_patch = patch.object(app_module, "ScreenRecorder", FakeScreenRecorder)
        self.screen_patch.start()
        self.addCleanup(self.screen_patch.stop)
        FakeAudioRecorder.instances = []
        FakeAudioRecorder.default_error = ""
        self.audio_patch = patch.object(app_module, "AudioRecorder", FakeAudioRecorder)
        self.audio_patch.start()
        self.addCleanup(self.audio_patch.stop)

        def no_system_tray(window):
            window.tray = None
            window.tray_button.hide()

        self.tray_patch = patch.object(app_module.MainWindow, "build_tray", no_system_tray)
        self.tray_patch.start()
        self.addCleanup(self.tray_patch.stop)
        self.window = self.make_window()
        self.image = QImage(64, 48, QImage.Format.Format_RGB888)
        self.image.fill(Qt.GlobalColor.black)

    def make_window(self):
        window = app_module.MainWindow()
        # Synthetic screen: record whether a blackout already existed at grab time.
        self.grabs = []
        self.rec_at_grab = []

        def fake_grab():
            self.grabs.append(bool(window.overlays))
            self.rec_at_grab.append(window.rec.active)
            return np.full((90, 160, 3), 60, np.uint8)

        window.screen_grab_fn = fake_grab
        window.show()
        self.qt_app.processEvents()
        return window

    def tearDown(self):
        self.window.wait_io()
        if self.window.worker is not None:
            worker = self.window.worker
            self.window.stop_monitoring()
            worker.finished.emit()
        self.window.dismiss_alert(mark_incident=False)
        self.window.close()
        self.window.deleteLater()
        self.qt_app.processEvents()

    # ---- helpers ----------------------------------------------------------
    def start(self):
        self.window.start_monitoring()
        self.assertIsInstance(self.window.worker, FakeCameraWorker)
        self.assertTrue(self.window.active)
        self.window.worker.opened.emit()
        self.assertEqual(self.window.status_label.text(), "监测中")
        return self.window.worker

    def frame(self, count, timestamp):
        with patch("privacy_guard.app.time.monotonic", return_value=timestamp):
            self.window.worker.frame_ready.emit(self.image, count, timestamp)

    def analysis(self, owner_present, stranger_alone, timestamp):
        with patch("privacy_guard.app.time.monotonic", return_value=timestamp):
            self.window.worker.analysis.emit(owner_present, stranger_alone, timestamp)

    def trigger(self, start=0.0):
        self.frame(2, start)
        self.assertFalse(self.window.overlays)
        self.frame(2, start + 0.8)
        self.assertTrue(self.window.overlays)

    def overlay_text(self):
        return "\n".join(
            lbl.text()
            for overlay in self.window.overlays
            for lbl in overlay.findChildren(QLabel)
        )

    def press(self, key):
        overlay = self.window.overlays[0]
        QTest.keyClick(overlay, key)

    # ---- blackout ---------------------------------------------------------
    def test_two_people_show_blackout_with_only_four_characters(self):
        self.start()
        self.trigger()
        self.assertEqual(self.window.alert_count, 1)
        self.assertIn("注意隐私", self.overlay_text())
        self.assertNotIn("2", self.overlay_text())  # no count leaked
        self.assertTrue(self.window.policy.alert_visible)

    def test_other_key_snoozes_and_auto_rearms_after_delay(self):
        self.window.rearm_seconds = 60.0
        self.start()
        self.trigger()
        with patch("privacy_guard.app.time.monotonic", return_value=100.0):
            self.window.handle_blackout_key(Qt.Key.Key_Space)
        self.assertFalse(self.window.overlays)
        self.assertEqual(self.window.arm.state, app_module.ArmState.SNOOZED)
        # Still snoozed: more faces do not reopen the blackout.
        self.frame(2, 2.0)
        self.frame(2, 2.8)
        self.assertFalse(self.window.overlays)
        # Before the deadline it stays snoozed; at the deadline the tick re-arms.
        with patch("privacy_guard.app.time.monotonic", return_value=159.0):
            self.window.arm_tick()
        self.assertEqual(self.window.arm.state, app_module.ArmState.SNOOZED)
        with patch("privacy_guard.app.time.monotonic", return_value=161.0):
            self.window.arm_tick()
        self.assertEqual(self.window.arm.state, app_module.ArmState.ARMED)

    def test_escape_disarms_until_manual_rearm(self):
        self.start()
        self.trigger()
        self.press(Qt.Key.Key_Escape)
        self.assertFalse(self.window.overlays)
        self.assertEqual(self.window.arm.state, app_module.ArmState.DISARMED)
        # Disarmed: sustained faces never reopen the blackout.
        for ts in (2.0, 2.8, 3.6, 4.4):
            self.frame(2, ts)
        self.assertFalse(self.window.overlays)
        self.assertEqual(self.window.alert_count, 1)
        # Manual re-arm restores protection.
        self.window.manual_arm()
        self.assertEqual(self.window.arm.state, app_module.ArmState.ARMED)

    def test_disarmed_state_does_not_open_blackout_but_stays_consistent(self):
        self.start()
        self.window.arm.disarm()
        self.frame(2, 0.0)
        self.frame(2, 0.8)
        self.assertFalse(self.window.overlays)
        self.assertEqual(self.window.alert_count, 0)
        self.assertFalse(self.window.policy.alert_visible)

    def test_test_overlay_does_not_count_or_touch_arming(self):
        self.start()
        self.window.show_alert(2, is_test=True)
        self.assertTrue(self.window.overlay_is_test)
        self.assertEqual(self.window.alert_count, 0)
        self.press(Qt.Key.Key_Escape)
        self.assertFalse(self.window.overlays)
        self.assertEqual(self.window.arm.state, app_module.ArmState.ARMED)

    def test_key_closes_all_screen_overlays(self):
        screens = [FakeScreen(), FakeScreen(800)]
        with patch.object(app_module.QApplication, "screens", return_value=screens):
            self.window.show_alert(2, is_test=True)
            self.assertEqual(len(self.window.overlays), 2)
            QTest.keyClick(self.window.overlays[1], Qt.Key.Key_Space)
            self.assertFalse(self.window.overlays)

    # ---- recording --------------------------------------------------------
    def test_two_people_auto_record_camera_only_by_default(self):
        worker = self.start()
        self.assertTrue(self.window.auto_record_2p)
        self.frame(2, 0.0)
        self.frame(2, 0.7)
        self.assertTrue(self.window.rec.active)
        self.assertEqual(self.window.rec.reason, "多人")
        self.assertTrue(any(on for on, _ in worker.recording))
        self.assertEqual(FakeScreenRecorder.instances, [])  # screen off by default
        # Scene clears: recording stops after the off-window.
        for ts in (1.0, 2.0, 3.0, 4.0, 5.0):
            self.frame(0, ts)
        self.assertFalse(self.window.rec.active)
        self.assertTrue(any(not on for on, _ in worker.recording))

    def test_screen_recording_for_two_people_when_enabled(self):
        self.start()
        self.window.screen2p_toggle.setChecked(True)
        self.window.on_screen2p_toggled()
        self.frame(2, 0.0)
        self.frame(2, 0.7)
        self.assertTrue(self.window.rec.active)
        self.assertEqual(len(FakeScreenRecorder.instances), 1)
        self.assertTrue(FakeScreenRecorder.instances[0].started)

    def test_disabling_auto_record_prevents_recording(self):
        worker = self.start()
        self.window.auto2p_toggle.setChecked(False)
        self.window.on_auto2p_toggled()
        self.frame(2, 0.0)
        self.frame(2, 0.7)
        self.assertFalse(self.window.rec.active)
        self.assertFalse(any(on for on, _ in worker.recording))

    def test_manual_recording_runs_without_any_detection(self):
        worker = self.start()
        self.window.manual_rec_toggle.setChecked(True)
        self.window.on_manual_rec_toggled()
        self.assertTrue(self.window.rec.active)
        self.assertEqual(self.window.rec.reason, "手动")
        self.assertTrue(any(on for on, _ in worker.recording))
        self.assertEqual(len(FakeScreenRecorder.instances), 1)
        self.window.manual_rec_toggle.setChecked(False)
        self.window.on_manual_rec_toggled()
        self.assertFalse(self.window.rec.active)

    def test_stop_monitoring_stops_recording(self):
        worker = self.start()
        self.frame(2, 0.0)
        self.frame(2, 0.7)
        self.assertTrue(self.window.rec.active)
        self.window.stop_monitoring()
        self.assertFalse(self.window.rec.active)
        self.assertTrue(any(not on for on, _ in worker.recording))
        worker.finished.emit()

    # ---- guard mode -------------------------------------------------------
    def test_guard_mode_records_lone_stranger(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.window.guard_enabled = True
        worker = self.start()
        self.assertEqual(worker.guard[0], True)
        # A non-owner alone, sustained past the on-window, starts a 卫兵 session.
        self.analysis(False, True, 0.0)
        self.analysis(False, True, 1.0)
        self.analysis(False, True, 2.0)
        self.assertTrue(self.window.rec.active)
        self.assertEqual(self.window.rec.reason, "卫兵")
        self.assertEqual(len(FakeScreenRecorder.instances), 1)  # screen recorded too
        # Owner returns; after the off-window (5s) recording stops.
        for ts in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            self.analysis(True, False, ts)
        self.assertFalse(self.window.rec.active)

    def test_unattended_trigger_snapshots_before_recording(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.window.guard_enabled = True
        self.window.snapshot_screen = False  # guard screenshot is taken regardless
        worker = self.start()
        self.analysis(False, True, 0.0)
        self.analysis(False, True, 1.0)
        self.assertEqual(worker.snapshots, [])
        self.analysis(False, True, 2.0)  # guard trigger turns on here
        self.assertEqual(len(worker.snapshots), 1)
        self.assertIn("卫兵", worker.snapshots[0].name)
        self.assertEqual(self.rec_at_grab, [False])  # screenshot before recording began
        self.assertTrue(self.window.rec.active)
        self.assertEqual(self.window.rec.reason, "卫兵")
        # Only once per trigger, not on every frame while it stays on.
        self.analysis(False, True, 3.0)
        self.assertEqual(len(worker.snapshots), 1)
        self.window.wait_io()
        self.assertEqual(len(list(storage_module.SNAPSHOTS_DIR.glob("*卫兵_screen.jpg"))), 1)

    def test_unattended_snapshot_can_be_turned_off(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.window.guard_enabled = True
        self.window.guardsnap_toggle.setChecked(False)
        self.window.on_guard_snapshot_toggled()
        worker = self.start()
        for ts in (0.0, 1.0, 2.0):
            self.analysis(False, True, ts)
        self.assertTrue(self.window.rec.active)
        self.assertEqual(worker.snapshots, [])
        self.assertEqual(self.grabs, [])

    # ---- blackout auto-close when only the owner remains ------------------
    def owner_scene(self, owner_present, count, ts):
        # The worker emits analysis just before frame_ready for the same frame.
        self.analysis(owner_present, False, ts)
        self.frame(count, ts)

    def test_blackout_auto_closes_when_only_owner_remains(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        worker = self.start()
        self.trigger()
        self.assertTrue(worker.identify)  # recognition runs during the blackout
        for ts in (1.0, 1.5, 2.0):
            self.owner_scene(True, 1, ts)
        self.assertTrue(self.window.overlays)  # not yet 1.5 s
        self.owner_scene(True, 1, 2.5)
        self.assertFalse(self.window.overlays)
        self.assertFalse(worker.identify)
        self.assertEqual(self.window.arm.state, app_module.ArmState.ARMED)  # still armed
        self.assertFalse(self.window.policy.alert_visible)
        self.assertIn("自动关闭", self.window.status_hint.text())

    def test_blackout_stays_when_remaining_person_is_stranger(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.start()
        self.trigger()
        for ts in (1.0, 2.0, 3.0, 4.0):
            self.owner_scene(False, 1, ts)
        self.assertTrue(self.window.overlays)

    def test_blackout_stays_when_owner_still_with_someone_or_nobody(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.start()
        self.trigger()
        for ts in (1.0, 2.0, 3.0):
            self.owner_scene(True, 2, ts)  # owner present but not alone
        for ts in (4.0, 5.0, 6.0):
            self.owner_scene(False, 0, ts)  # everyone left
        self.assertTrue(self.window.overlays)

    def test_owner_must_be_steady_for_auto_close(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.start()
        self.trigger()
        self.owner_scene(True, 1, 1.0)
        self.owner_scene(True, 1, 2.0)
        self.owner_scene(False, 1, 2.3)  # recognition blip restarts the window
        self.owner_scene(True, 1, 2.6)
        self.owner_scene(True, 1, 3.5)
        self.assertTrue(self.window.overlays)
        self.owner_scene(True, 1, 4.2)  # 1.6 s since the window restarted at 2.6
        self.assertFalse(self.window.overlays)

    def test_no_auto_close_without_enrollment_or_when_disabled(self):
        worker = self.start()
        self.trigger()  # no owner enrolled
        self.assertFalse(worker.identify)
        for ts in (1.0, 2.0, 3.0):
            self.owner_scene(True, 1, ts)
        self.assertTrue(self.window.overlays)
        self.window.dismiss_alert()
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.window.autoclose_toggle.setChecked(False)
        self.window.on_autoclose_toggled()
        self.clear_and_retrigger(4.0)
        self.assertTrue(self.window.overlays)
        self.assertFalse(worker.identify)
        for ts in (9.5, 10.5, 11.5, 12.5):
            self.owner_scene(True, 1, ts)
        self.assertTrue(self.window.overlays)

    def test_test_blackout_never_auto_closes(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.start()
        self.window.show_alert(2, is_test=True)
        for ts in (1.0, 2.0, 3.0):
            self.owner_scene(True, 1, ts)
        self.assertTrue(self.window.overlays)

    def test_guard_toggle_requires_enrollment(self):
        self.window.owner_features = []
        self.window.guard_toggle.setChecked(True)
        self.window.on_guard_toggled()
        self.assertFalse(self.window.guard_enabled)
        self.assertFalse(self.window.guard_toggle.isChecked())

    def test_enrollment_collects_samples_and_enables_guard(self):
        worker = self.start()
        self.window.start_enroll()
        self.assertTrue(self.window.enrolling)
        self.assertTrue(worker.enrolling)
        for _ in range(app_module.ENROLL_TARGET):
            worker.owner_sample.emit(np.random.rand(1, 128).astype(np.float32))
        self.assertFalse(self.window.enrolling)
        self.assertEqual(len(self.window.owner_features), app_module.ENROLL_TARGET)
        self.assertTrue(storage_module.OWNER_FILE.is_file())
        self.assertTrue(self.window.guard_toggle.isEnabled())

    def test_enrollment_requires_monitoring(self):
        self.window.start_enroll()
        self.assertFalse(self.window.enrolling)
        self.assertIn("开始监测", self.window.enroll_status.text())

    # ---- snapshots ----------------------------------------------------------
    def snapshot_dir(self):
        return storage_module.SNAPSHOTS_DIR

    def test_two_people_snapshot_camera_and_screen_before_blackout(self):
        worker = self.start()
        self.trigger()
        self.assertEqual(len(worker.snapshots), 1)
        camera = worker.snapshots[0]
        self.assertEqual(camera.parent, self.snapshot_dir())
        self.assertTrue(camera.name.endswith("_camera.jpg"))
        self.assertIn("多人", camera.name)
        self.assertEqual(self.grabs, [False])  # screen grabbed before the blackout
        self.window.wait_io()
        screens = list(self.snapshot_dir().glob("*_screen.jpg"))
        self.assertEqual(len(screens), 1)
        self.assertGreater(screens[0].stat().st_size, 0)
        self.assertEqual(self.window.snapshot_count, 1)
        # Snapshots never land in the recordings folder.
        self.assertFalse(list(storage_module.RECORDINGS_DIR.rglob("*.jpg")))

    def test_snapshot_once_per_incident(self):
        worker = self.start()
        self.trigger()
        for ts in (1.0, 1.5, 2.0):
            self.frame(2, ts)
        self.assertEqual(len(worker.snapshots), 1)

    def test_snapshot_still_taken_when_disarmed(self):
        worker = self.start()
        self.window.arm.disarm()
        self.frame(2, 0.0)
        self.frame(2, 0.8)
        self.assertFalse(self.window.overlays)
        self.assertEqual(len(worker.snapshots), 1)

    def test_snapshot_toggles(self):
        worker = self.start()
        self.window.snapscreen_toggle.setChecked(False)
        self.window.on_snapshot_screen_toggled()
        self.trigger()
        self.assertEqual(len(worker.snapshots), 1)  # camera only
        self.assertEqual(self.grabs, [])
        self.window.snap2p_toggle.setChecked(False)
        self.window.on_snapshot2p_toggled()
        self.window.dismiss_alert()
        self.clear_and_retrigger(5.0)
        self.assertEqual(len(worker.snapshots), 1)  # auto snapshot off

    def clear_and_retrigger(self, start):
        for step in range(8):
            self.frame(1, start + step * 0.5)
        self.frame(2, start + 4.1)
        self.frame(2, start + 5.0)

    def test_manual_snapshot_without_monitoring_saves_screen_only(self):
        self.window.manual_snapshot()
        self.window.wait_io()
        self.assertEqual(len(list(self.snapshot_dir().glob("*手动_screen.jpg"))), 1)
        self.assertEqual(self.window.snapshot_count, 1)
        self.window.snapscreen_toggle.setChecked(False)
        self.window.on_snapshot_screen_toggled()
        self.window.manual_snapshot()
        self.assertIn("未抓拍", self.window.snap_status.text())
        self.assertEqual(self.window.snapshot_count, 1)

    def test_manual_snapshot_while_monitoring_requests_camera(self):
        worker = self.start()
        self.window.manual_snapshot()
        self.assertEqual(len(worker.snapshots), 1)
        self.assertIn("手动", worker.snapshots[0].name)

    # ---- audio --------------------------------------------------------------
    def test_audio_added_and_removed_within_one_session(self):
        self.start()
        self.window.audio_toggle.setChecked(True)
        self.window.on_audio_toggled()
        self.window.manual_rec_toggle.setChecked(True)
        self.window.on_manual_rec_toggled()
        session = self.window.rec.dir
        self.assertEqual(len(FakeAudioRecorder.instances), 1)
        first = FakeAudioRecorder.instances[0]
        self.assertTrue(first.started)
        self.assertEqual(first.path.parent, session)
        self.assertEqual(first.path.name, "audio.wav")
        self.assertIn("录音", self.window.rec_label.text())
        # Turning audio off keeps the same video session running.
        self.window.audio_toggle.setChecked(False)
        self.window.on_audio_toggled()
        self.assertTrue(first.stopped)
        self.assertTrue(self.window.rec.active)
        self.assertEqual(self.window.rec.dir, session)
        self.assertNotIn("录音", self.window.rec_label.text())

    def test_audio_off_by_default_and_stops_with_session(self):
        self.start()
        self.window.manual_rec_toggle.setChecked(True)
        self.window.on_manual_rec_toggled()
        self.assertEqual(FakeAudioRecorder.instances, [])
        self.window.audio_toggle.setChecked(True)
        self.window.on_audio_toggled()
        audio = FakeAudioRecorder.instances[0]
        self.window.manual_rec_toggle.setChecked(False)
        self.window.on_manual_rec_toggled()
        self.assertTrue(audio.stopped)
        self.assertIsNone(self.window.rec.audio)

    def test_microphone_failure_is_reported_without_stopping_video(self):
        FakeAudioRecorder.default_error = "无法打开麦克风"
        self.start()
        self.window.record_audio = True
        self.window.manual_rec_toggle.setChecked(True)
        self.window.on_manual_rec_toggled()
        self.assertTrue(self.window.rec.active)
        self.assertIn("录音不可用", self.window.rec_label.text())
        self.assertIn("麦克风", self.window.rec_label.toolTip())

    # ---- recording stop / segmentation settings ---------------------------
    def test_stop_delay_spins_update_triggers_live_and_persist(self):
        self.window.rec2p_off_spin.setValue(8.0)
        self.window.guard_off_spin.setValue(12.5)
        self.assertEqual(self.window.record_2p_trigger.off_seconds, 8.0)
        self.assertEqual(self.window.guard_trigger.off_seconds, 12.5)
        stored = storage_module.load_settings()
        self.assertEqual(stored["record_2p_off"], 8.0)
        self.assertEqual(stored["guard_off"], 12.5)

    def test_configured_2p_stop_delay_keeps_recording_until_elapsed(self):
        # A 6 s delay: recording must survive a 4 s gap below threshold.
        self.window.rec2p_off_spin.setValue(6.0)
        self.start()
        self.frame(2, 0.0)
        self.frame(2, 0.7)
        self.assertTrue(self.window.rec.active)
        # Off-window counts from the first below-threshold frame (t=1.0).
        for ts in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            self.frame(0, ts)  # 5 s elapsed since 1.0 < 6 s delay
        self.assertTrue(self.window.rec.active)
        self.frame(0, 7.0)  # 6 s since 1.0 -> stops
        self.assertFalse(self.window.rec.active)

    def test_max_segment_splits_into_a_new_session(self):
        self.window.max_seg_spin.setValue(10.0)  # 10 minutes
        worker = self.start()
        self.window.manual_rec_toggle.setChecked(True)
        self.window.on_manual_rec_toggled()
        first_dir = self.window.rec.dir
        with patch("privacy_guard.app.time.monotonic", return_value=self.window.rec.started_at + 1.0):
            self.window.arm_tick()
        self.assertEqual(self.window.rec.dir, first_dir)  # not yet
        with patch("privacy_guard.app.time.monotonic", return_value=self.window.rec.started_at + 601.0):
            self.window.arm_tick()
        self.assertTrue(self.window.rec.active)
        self.assertNotEqual(self.window.rec.dir, first_dir)  # new segment folder
        self.assertEqual(self.window.rec.reason, "手动")

    def test_manual_recording_auto_stops_after_configured_minutes(self):
        self.window.manual_stop_spin.setValue(5.0)
        self.start()
        self.window.manual_rec_toggle.setChecked(True)
        self.window.on_manual_rec_toggled()
        self.assertTrue(self.window.rec.active)
        with patch("privacy_guard.app.time.monotonic", return_value=self.window.rec.started_at + 301.0):
            self.window.arm_tick()
        self.assertFalse(self.window.manual_recording)
        self.assertFalse(self.window.manual_rec_toggle.isChecked())
        self.assertFalse(self.window.rec.active)

    def test_zero_max_segment_never_splits(self):
        self.window.max_seg_spin.setValue(0.0)
        self.start()
        self.window.manual_rec_toggle.setChecked(True)
        self.window.on_manual_rec_toggled()
        first_dir = self.window.rec.dir
        with patch("privacy_guard.app.time.monotonic", return_value=self.window.rec.started_at + 99999.0):
            self.window.arm_tick()
        self.assertEqual(self.window.rec.dir, first_dir)
        self.assertTrue(self.window.rec.active)

    # ---- save locations ---------------------------------------------------
    def test_recordings_and_snapshots_use_separate_chosen_folders(self):
        base = Path(self.tmp.name)
        recordings, snapshots = base / "我的录像", base / "我的抓拍"
        self.assertTrue(self.window.set_recordings_dir(recordings))
        self.assertFalse(self.window.set_snapshots_dir(recordings))
        self.assertIn("分开存放", self.window.paths_status.text())
        self.assertTrue(self.window.set_snapshots_dir(snapshots))
        self.assertEqual(self.window.paths_status.text(), "")
        worker = self.start()
        self.trigger()  # starts a 多人 recording and a snapshot
        self.assertEqual(self.window.rec.dir.parent, recordings)
        self.assertEqual(worker.snapshots[0].parent, snapshots)
        self.window.wait_io()
        self.assertEqual(len(list(snapshots.glob("*_screen.jpg"))), 1)
        stored = storage_module.load_settings()
        self.assertEqual(stored["recordings_dir"], str(recordings))
        self.assertEqual(stored["snapshots_dir"], str(snapshots))
        # Paths are hidden by default; revealed only after unlocking.
        self.assertEqual(self.window.snap_path_label.toolTip(), "")
        self.window.paths_revealed = True
        self.window.refresh_paths()
        self.assertIn("我的抓拍", self.window.snap_path_label.toolTip())

    def test_identical_folders_in_profile_are_reset_on_load(self):
        same = Path(self.tmp.name) / "same"
        storage_module.save_settings({**storage_module.DEFAULT_SETTINGS,
                                      "recordings_dir": str(same), "snapshots_dir": str(same)})
        window = self.make_window()
        try:
            self.assertNotEqual(window.recordings_base(), window.snapshots_base())
            self.assertEqual(window.recordings_base(), same)
        finally:
            window.close()
            window.deleteLater()

    # ---- secondary password & protected actions ---------------------------
    def set_pw(self, password="1234"):
        storage_module.set_password(self.window.settings, password)
        self.window.refresh_security_ui()

    def test_paths_hidden_by_default_and_reveal_requires_password(self):
        self.assertFalse(self.window.paths_revealed)
        self.assertIn("已隐藏", self.window.rec_path_label.text())
        # No password set yet: reveal is refused with guidance.
        with patch.object(self.window, "_info") as info:
            self.window.toggle_reveal_paths()
        self.assertFalse(self.window.paths_revealed)
        info.assert_called_once()
        # With a password, a correct entry reveals; wrong entry does not.
        self.set_pw("1234")
        with patch.object(self.window, "ask_password", return_value="nope"), \
             patch.object(self.window, "_info"):
            self.window.toggle_reveal_paths()
        self.assertFalse(self.window.paths_revealed)
        with patch.object(self.window, "ask_password", return_value="1234"):
            self.window.toggle_reveal_paths()
        self.assertTrue(self.window.paths_revealed)
        self.assertNotIn("已隐藏", self.window.rec_path_label.text())

    def test_require_password_paths(self):
        with patch.object(self.window, "_info"):
            self.assertFalse(self.window.require_password("x"))  # no password set
        self.set_pw("abcd")
        with patch.object(self.window, "ask_password", return_value=None):
            self.assertFalse(self.window.require_password("x"))  # cancelled
        with patch.object(self.window, "ask_password", return_value="bad"), \
             patch.object(self.window, "_info"):
            self.assertFalse(self.window.require_password("x"))
        with patch.object(self.window, "ask_password", return_value="abcd"):
            self.assertTrue(self.window.require_password("x"))

    def test_clear_media_requires_password_and_face(self):
        base = Path(self.tmp.name)
        rec, snap = base / "recordings", base / "snapshots"
        (rec).mkdir(parents=True); (rec / "clip.mp4").write_bytes(b"x")
        (snap).mkdir(parents=True); (snap / "a_camera.jpg").write_bytes(b"x")
        # No password -> refused, nothing deleted.
        with patch.object(self.window, "_info"):
            self.window.clear_media_action()
        self.assertTrue((rec / "clip.mp4").exists())
        self.set_pw("1234")
        # Password ok but face verification fails -> nothing deleted.
        with patch.object(self.window, "require_password", return_value=True), \
             patch.object(self.window, "verify_owner_live", return_value=False), \
             patch.object(self.window, "_info"):
            self.window.clear_media_action()
        self.assertTrue((rec / "clip.mp4").exists())
        # Face cannot run (no monitoring) -> informed, nothing deleted.
        with patch.object(self.window, "require_password", return_value=True), \
             patch.object(self.window, "verify_owner_live", return_value=None), \
             patch.object(self.window, "_info") as info:
            self.window.clear_media_action()
        self.assertTrue((snap / "a_camera.jpg").exists())
        self.assertIn("人脸验证", info.call_args[0][0])
        # Password + face + confirm -> deleted.
        with patch.object(self.window, "require_password", return_value=True), \
             patch.object(self.window, "verify_owner_live", return_value=True), \
             patch.object(self.window, "_confirm", return_value=True), \
             patch.object(self.window, "_info"):
            self.window.clear_media_action()
        self.assertEqual(list(rec.iterdir()), [])
        self.assertEqual(list(snap.iterdir()), [])
        self.assertTrue(rec.is_dir() and snap.is_dir())

    def test_clear_media_cancelled_at_confirm_keeps_files(self):
        base = Path(self.tmp.name)
        rec = storage_module.RECORDINGS_DIR
        rec.mkdir(parents=True, exist_ok=True)
        (rec / "clip.mp4").write_bytes(b"x")
        self.set_pw("1234")
        with patch.object(self.window, "require_password", return_value=True), \
             patch.object(self.window, "verify_owner_live", return_value=True), \
             patch.object(self.window, "_confirm", return_value=False):
            self.window.clear_media_action()
        self.assertTrue((rec / "clip.mp4").exists())

    def test_turning_guard_off_requires_password(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.window.guard_enabled = True
        self.window.refresh_security_ui()
        self.set_pw("1234")
        self.window.guard_toggle.setChecked(False)
        with patch.object(self.window, "require_password", return_value=False):
            self.window.on_guard_toggled()
        self.assertTrue(self.window.guard_enabled)  # stayed on
        self.assertTrue(self.window.guard_toggle.isChecked())
        self.window.guard_toggle.setChecked(False)
        with patch.object(self.window, "require_password", return_value=True):
            self.window.on_guard_toggled()
        self.assertFalse(self.window.guard_enabled)

    def test_verify_owner_live_returns_none_without_monitoring(self):
        self.window.owner_features = [np.ones((1, 128), np.float32)]
        self.assertIsNone(self.window.verify_owner_live())

    # ---- preserved detector/plumbing behavior -----------------------------
    def test_stale_results_do_not_change_counts_or_open_blackout(self):
        self.start()
        with patch("privacy_guard.app.time.monotonic", return_value=10.0):
            self.window.worker.frame_ready.emit(self.image, 2, 0.0)
            self.window.worker.frame_ready.emit(self.image, 2, 0.8)
        self.assertEqual(self.window.count_label.text(), "—")
        self.assertFalse(self.window.overlays)
        self.frame(2, 10.0)
        self.frame(2, 10.8)
        self.assertTrue(self.window.overlays)

    def test_missing_model_does_not_create_or_start_worker(self):
        with patch.object(app_module.MODEL.__class__, "is_file", return_value=False):
            self.window.start_monitoring()
        self.assertIsNone(self.window.worker)
        self.assertFalse(self.window.active)
        self.assertIn("模型缺失", self.window.status_hint.text())

    def test_failed_camera_start_recovers_controls(self):
        worker = self.start()
        worker.failed.emit("测试摄像头不可用")
        self.assertFalse(self.window.active)
        self.assertEqual(worker.stop_calls, 1)
        worker.finished.emit()
        self.assertIsNone(self.window.worker)
        self.assertEqual(self.window.start_button.text(), "重新开始")
        self.assertIn("测试摄像头不可用", self.window.status_hint.text())

    def test_stop_cleans_overlay_and_ignores_late_frames(self):
        worker = self.start()
        self.trigger()
        self.window.stop_monitoring()
        self.assertFalse(self.window.active)
        self.assertFalse(self.window.overlays)
        self.assertEqual(worker.stop_calls, 1)
        worker.frame_ready.emit(self.image, 3, 1.6)
        self.assertFalse(self.window.overlays)
        worker.finished.emit()
        self.assertIsNone(self.window.worker)
        self.assertEqual(self.window.count_label.text(), "—")

    def test_overlay_follows_screen_geometry_change(self):
        screen = FakeScreen()
        with patch.object(app_module.QApplication, "screens", return_value=[screen]):
            self.window.show_alert(2, is_test=True)
        screen.rect = QRect(100, 20, 1024, 768)
        screen.geometryChanged.emit(screen.rect)
        self.assertEqual(self.window.overlays[0].geometry(), screen.rect)

    def test_settings_persist_across_windows(self):
        self.window.rearm_spin.setValue(120)
        self.window.auto2p_toggle.setChecked(False)
        self.window.on_auto2p_toggled()
        self.window.save_current_settings()
        stored = storage_module.load_settings()
        self.assertEqual(stored["rearm_seconds"], 120.0)
        self.assertFalse(stored["auto_record_2p"])


if __name__ == "__main__":
    unittest.main()
