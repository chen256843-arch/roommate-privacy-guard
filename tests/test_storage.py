"""Separate recording/snapshot folders, collision-safe names, folder checks."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from privacy_guard import storage


class FolderValidationTests(unittest.TestCase):
    def test_identical_folders_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "证据"
            self.assertIn("分开存放", storage.validate_dirs(folder, folder))
            self.assertIn("分开存放", storage.validate_dirs(folder, folder / "x" / ".."))

    def test_distinct_folders_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(storage.validate_dirs(Path(tmp) / "录像", Path(tmp) / "抓拍"), "")

    def test_check_writable_creates_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "新建" / "抓拍"
            self.assertEqual(storage.check_writable(folder), "")
            self.assertTrue(folder.is_dir())
            self.assertEqual(list(folder.iterdir()), [])  # probe file removed

    def test_check_writable_reports_unusable_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "file.txt"
            blocker.write_text("x")
            self.assertTrue(storage.check_writable(blocker / "sub"))


class PortableModeTests(unittest.TestCase):
    def test_marker_beside_program_keeps_data_in_its_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "隐私卫士便携版"
            app_dir.mkdir()
            (app_dir / storage.PORTABLE_MARKER).write_text("portable", encoding="utf-8")
            data, portable = storage.resolve_data_dir(app_dir, frozen=True)
            self.assertTrue(portable)
            self.assertEqual(data, app_dir / "data")
            self.assertTrue(data.is_dir())

    def test_without_marker_uses_user_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, portable = storage.resolve_data_dir(Path(tmp), frozen=True)
            self.assertFalse(portable)
            self.assertEqual(data, Path.home() / "PrivacyGuard")
            self.assertFalse((Path(tmp) / "data").exists())

    def test_unwritable_portable_folder_falls_back_to_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp)
            (app_dir / storage.PORTABLE_MARKER).write_text("portable", encoding="utf-8")
            (app_dir / "data").write_text("not a folder")  # data cannot be created
            data, portable = storage.resolve_data_dir(app_dir, frozen=True)
            self.assertFalse(portable)
            self.assertEqual(data, Path.home() / "PrivacyGuard")


class NamingTests(unittest.TestCase):
    def test_snapshot_paths_are_paired_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera, screen = storage.snapshot_paths("多人", Path(tmp))
            self.assertEqual(camera.parent, Path(tmp))
            self.assertEqual(camera.name.replace("_camera", "_screen"), screen.name)
            camera.write_bytes(b"x")
            camera2, screen2 = storage.snapshot_paths("多人", Path(tmp))
            self.assertNotEqual(camera2, camera)
            self.assertFalse(camera2.exists() or screen2.exists())

    def test_session_dirs_in_same_second_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = storage.new_session_dir("卫兵", Path(tmp))
            second = storage.new_session_dir("卫兵", Path(tmp))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())


class PasswordTests(unittest.TestCase):
    def test_set_verify_and_clear(self):
        settings = {}
        self.assertFalse(storage.has_password(settings))
        self.assertFalse(storage.verify_password(settings, "anything"))
        storage.set_password(settings, "s3cret")
        self.assertTrue(storage.has_password(settings))
        self.assertTrue(storage.verify_password(settings, "s3cret"))
        self.assertFalse(storage.verify_password(settings, "wrong"))
        self.assertNotIn("s3cret", str(settings))  # only a salted hash is stored
        storage.clear_password(settings)
        self.assertFalse(storage.has_password(settings))

    def test_salt_is_random_per_set(self):
        a, b = {}, {}
        storage.set_password(a, "same")
        storage.set_password(b, "same")
        self.assertNotEqual(a["pw_hash"], b["pw_hash"])


class ClearMediaTests(unittest.TestCase):
    def test_deletes_contents_but_keeps_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "录像"
            snap = Path(tmp) / "抓拍"
            (rec / "2026_多人").mkdir(parents=True)
            (rec / "2026_多人" / "camera.mp4").write_bytes(b"x")
            (snap).mkdir()
            (snap / "a_camera.jpg").write_bytes(b"x")
            (snap / "a_screen.jpg").write_bytes(b"x")
            removed, failed = storage.clear_media(rec, snap)
            self.assertEqual(failed, 0)
            self.assertEqual(removed, 3)  # one session dir + two jpgs
            self.assertTrue(rec.is_dir() and snap.is_dir())
            self.assertEqual(list(rec.iterdir()), [])
            self.assertEqual(list(snap.iterdir()), [])

    def test_missing_folder_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            removed, failed = storage.clear_media(Path(tmp) / "nope")
            self.assertEqual((removed, failed), (0, 0))


if __name__ == "__main__":
    unittest.main()
