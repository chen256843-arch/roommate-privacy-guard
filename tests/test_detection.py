"""Offline detector, frame fitting, and preview-buffer checks.

Run: python -m unittest discover -s tests -p test_detection.py -v

Optional real-image check, without accessing a camera or downloading anything:
    python tests/test_detection.py --face-fixture PATH_TO_ONE_FACE_IMAGE -v

The optional image must be a close crop containing one visible face. It stays
local and is not included in the distribution. The test creates one- and
two-face 640 x 480 frames in memory, using a mirrored copy for the second face.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy_guard.detection import (
    FaceCounter, FaceRecognizer, _fit_frame, _preview_image, largest_face_index,
)

MODEL = ROOT / "assets" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = ROOT / "assets" / "face_recognition_sface_2021dec.onnx"
FACE_FIXTURE = os.environ.get("PRIVACY_GUARD_FACE_FIXTURE")


class FaceCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.counter = FaceCounter(MODEL)

    def test_blank_frames_count_zero_at_multiple_sizes(self) -> None:
        for height, width in ((480, 640), (320, 320), (96, 128)):
            with self.subTest(size=(width, height)):
                faces = self.counter.detect(np.zeros((height, width, 3), np.uint8))
                self.assertEqual(faces.shape, (0, 15))
                self.assertEqual(faces.dtype, np.float32)

    def test_invalid_frame_is_rejected_before_inference(self) -> None:
        invalid = (
            None,
            [],
            np.zeros((0, 64, 3), np.uint8),
            np.zeros((64, 0, 3), np.uint8),
            np.zeros((64, 64), np.uint8),
            np.zeros((64, 64, 4), np.uint8),
            np.zeros((64, 64, 3), np.float32),
        )
        for frame in invalid:
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaisesRegex(ValueError, "BGR uint8"):
                    self.counter.detect(frame)

    def test_non_contiguous_frame_is_accepted(self) -> None:
        frame = np.zeros((320, 640, 3), np.uint8)[:, ::2]
        self.assertFalse(frame.flags.c_contiguous)
        self.assertEqual(len(self.counter.detect(frame)), 0)

    def test_invalid_confidence_is_rejected(self) -> None:
        for confidence in (0, -0.1, 1.01, float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    FaceCounter(MODEL, confidence)

    def test_missing_model_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            FaceCounter(ROOT / "assets" / "does-not-exist.onnx")


class FrameAndPreviewTests(unittest.TestCase):
    def test_fit_preserves_aspect_ratio_and_preview_bounds(self) -> None:
        for height, width in ((1080, 1920), (1920, 1080), (240, 320), (1, 2000)):
            with self.subTest(size=(width, height)):
                frame = _fit_frame(np.zeros((height, width, 3), np.uint8))
                fitted_height, fitted_width = frame.shape[:2]
                self.assertLessEqual(fitted_width, 640)
                self.assertLessEqual(fitted_height, 480)
                self.assertGreaterEqual(fitted_width, 1)
                self.assertGreaterEqual(fitted_height, 1)
                # Rounding an extremely narrow image may add at most one pixel.
                scale = min(640 / width, 480 / height)
                self.assertLessEqual(abs(fitted_width - width * scale), 1)
                self.assertLessEqual(abs(fitted_height - height * scale), 1)

    def test_preview_owns_pixels_and_converts_bgr_to_rgb(self) -> None:
        frame = np.full((240, 320, 3), (11, 27, 213), dtype=np.uint8)
        preview = _preview_image(frame, np.empty((0, 15), np.float32))
        self.assertEqual((preview.width(), preview.height()), (640, 480))
        self.assertEqual(preview.pixelColor(320, 240).getRgb(), (213, 27, 11, 255))
        self.assertEqual(preview.pixelColor(0, 0).getRgb(), (16, 19, 24, 255))
        frame.fill(255)
        del frame
        gc.collect()
        # Allocations encourage reuse of temporary numpy buffers after return.
        scratch = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(preview.pixelColor(320, 240).getRgb(), (213, 27, 11, 255))
        self.assertEqual(scratch.sum(), 0)

    def test_outside_face_boxes_are_clipped_without_changing_input(self) -> None:
        frame = np.zeros((480, 640, 3), np.uint8)
        faces = np.zeros((2, 15), np.float32)
        faces[0, :4] = (-40, -40, 100, 100)
        faces[1, :4] = (630, 470, 100, 100)
        faces[:, -1] = 0.95
        preview = _preview_image(frame, faces)
        self.assertFalse(preview.isNull())
        self.assertEqual(np.count_nonzero(frame), 0)
        self.assertEqual(preview.pixelColor(0, 0).getRgb()[:3], (140, 222, 94))


class LargestFaceTests(unittest.TestCase):
    def test_empty_returns_minus_one(self):
        self.assertEqual(largest_face_index(np.empty((0, 15), np.float32)), -1)
        self.assertEqual(largest_face_index(None), -1)

    def test_picks_biggest_box(self):
        faces = np.zeros((3, 15), np.float32)
        faces[0, :4] = (0, 0, 10, 10)
        faces[1, :4] = (0, 0, 40, 40)  # largest
        faces[2, :4] = (0, 0, 20, 20)
        self.assertEqual(largest_face_index(faces), 1)


class FaceRecognizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = FaceRecognizer(SFACE_MODEL)

    def test_missing_model_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            FaceRecognizer(ROOT / "assets" / "nope.onnx")

    def test_identical_features_match_and_best_match_takes_max(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((1, 128)).astype(np.float32)
        b = rng.standard_normal((1, 128)).astype(np.float32)
        self.assertAlmostEqual(self.recognizer.cosine(a, a), 1.0, places=4)
        self.assertLess(self.recognizer.cosine(a, b), 0.9)
        self.assertAlmostEqual(self.recognizer.best_match(a, [b, a]), 1.0, places=4)


class WorkerSnapshotTests(unittest.TestCase):
    def test_pending_snapshot_is_saved_once_from_the_clean_frame(self):
        import tempfile
        from privacy_guard.detection import CameraWorker

        worker = CameraWorker(0, MODEL)  # never started; no camera is opened
        saved = []
        worker.snapshot_saved.connect(lambda path, ok: saved.append((path, ok)))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "抓拍" / "多人_camera.jpg"
            frame = np.full((480, 640, 3), 70, np.uint8)
            worker._take_pending_snapshot(frame)  # nothing requested yet
            self.assertEqual(saved, [])
            worker.request_snapshot(target)
            worker._take_pending_snapshot(frame)
            worker._take_pending_snapshot(frame)  # request is consumed
            self.assertEqual(saved, [(str(target), True)])
            self.assertTrue(target.is_file())


class OptionalRealImageTests(unittest.TestCase):
    def test_zero_one_and_two_real_faces(self) -> None:
        if not FACE_FIXTURE:
            self.skipTest("Pass --face-fixture or PRIVACY_GUARD_FACE_FIXTURE for real-image inference")
        fixture = Path(FACE_FIXTURE).resolve()
        self.assertTrue(fixture.is_file(), f"Missing fixture: {fixture}")
        crop = cv2.imread(str(fixture))
        self.assertIsNotNone(crop, f"Cannot decode image: {fixture}")
        height, width = crop.shape[:2]
        scale = min(240 / width, 360 / height)
        tile = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))))
        tile_height, tile_width = tile.shape[:2]
        top = (480 - tile_height) // 2
        one = np.full((480, 640, 3), 24, np.uint8)
        one[top : top + tile_height, 40 : 40 + tile_width] = tile
        two = one.copy()
        two[top : top + tile_height, 360 : 360 + tile_width] = cv2.flip(tile, 1)
        counter = FaceCounter(MODEL)
        for name, frame, expected in (
            ("blank", np.zeros((480, 640, 3), np.uint8), 0),
            ("one face", one, 1),
            ("two faces", two, 2),
        ):
            with self.subTest(name=name):
                start = time.perf_counter()
                faces = counter.detect(frame)
                elapsed_ms = (time.perf_counter() - start) * 1000
                scores = [round(float(score), 5) for score in faces[:, -1]]
                print(f"\n{name}: count={len(faces)}, confidence={scores}, inference={elapsed_ms:.1f} ms")
                self.assertEqual(len(faces), expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--face-fixture")
    args, remaining = parser.parse_known_args()
    if args.face_fixture:
        FACE_FIXTURE = args.face_fixture
    unittest.main(argv=[sys.argv[0], *remaining])
