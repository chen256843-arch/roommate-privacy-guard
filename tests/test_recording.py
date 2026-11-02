"""Recording helpers: timestamped sinks and the background screen recorder.

These never touch a real screen; a synthetic ``grab_fn`` feeds frames.
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wave

import cv2

from privacy_guard.recording import (
    AudioRecorder, VideoSink, ScreenRecorder, draw_timestamp, save_image, session_timestamp,
)


class FakeMicrophone:
    """Deliver 0.1 s chunks of a 16 kHz mono tone, then a short tail on close."""

    def __init__(self, sample_rate, channels):
        self.chunk = (np.arange(sample_rate // 10, dtype=np.int16) % 200 * 50).tobytes()
        self.closed = False

    def read(self):
        time.sleep(0.01)
        return self.chunk

    def close(self):
        self.closed = True
        return self.chunk[:320]


class TimestampTests(unittest.TestCase):
    def test_session_timestamp_is_filesystem_safe(self):
        stamp = session_timestamp()
        self.assertTrue(all(ch.isalnum() or ch in "-_" for ch in stamp))

    def test_draw_timestamp_modifies_frame_in_place(self):
        frame = np.zeros((120, 200, 3), np.uint8)
        returned = draw_timestamp(frame, "摄像头")
        self.assertIs(returned, frame)
        self.assertGreater(int(frame.sum()), 0)  # text was burned in


class VideoSinkTests(unittest.TestCase):
    def test_writes_a_playable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            sink = VideoSink(path, fps=10.0, label="摄像头")
            for _ in range(5):
                sink.write(np.full((48, 64, 3), 120, np.uint8))
            self.assertTrue(sink.opened)
            sink.close()
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_resizes_frames_to_first_size_and_tolerates_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            sink = VideoSink(Path(tmp) / "clip.mp4", fps=8.0)
            sink.write(np.zeros((40, 60, 3), np.uint8))
            sink.write(np.zeros((80, 100, 3), np.uint8))  # different size, resized
            sink.write(np.zeros((0, 0, 3), np.uint8))  # ignored, no crash
            sink.close()


class ScreenRecorderTests(unittest.TestCase):
    def test_records_from_injected_grab(self):
        grabs = {"n": 0}

        def fake_grab():
            grabs["n"] += 1
            return np.full((60, 80, 3), 90, np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.mp4"
            recorder = ScreenRecorder(path, fps=30.0, grab_fn=fake_grab, label="屏幕")
            recorder.start()
            time.sleep(0.25)
            recorder.stop()
            self.assertFalse(recorder.active)
            self.assertGreater(recorder.frames_written, 0)
            self.assertTrue(path.is_file())

    def test_capture_failures_stop_cleanly(self):
        def broken_grab():
            raise RuntimeError("no display")

        with tempfile.TemporaryDirectory() as tmp:
            recorder = ScreenRecorder(Path(tmp) / "s.mp4", fps=60.0, grab_fn=broken_grab)
            recorder.start()
            time.sleep(0.2)
            recorder.stop()
            self.assertEqual(recorder.frames_written, 0)


class SaveImageTests(unittest.TestCase):
    def test_writes_stamped_jpeg_to_non_ascii_path(self):
        # cv2.imwrite silently fails on such paths on Windows; save_image must not.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "抓拍目录" / "多人_camera.jpg"
            frame = np.full((120, 160, 3), 50, np.uint8)
            save_image(target, frame, "CAM")
            self.assertTrue(target.is_file())
            decoded = cv2.imdecode(np.frombuffer(target.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (120, 160, 3))
            self.assertEqual(int(frame.max()), 50)  # caller's frame left unchanged


class AudioRecorderTests(unittest.TestCase):
    def test_writes_pcm_wav_with_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "录音" / "audio.wav"
            recorder = AudioRecorder(path, source_factory=FakeMicrophone)
            recorder.start()
            time.sleep(0.15)
            recorder.stop()
            self.assertEqual(recorder.error, "")
            self.assertFalse(recorder.active)
            with wave.open(str(path)) as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnframes() * 2, recorder.bytes_written)
                self.assertGreater(wav.getnframes(), 0)

    def test_microphone_failure_sets_error_and_writes_nothing(self):
        def broken(sample_rate, channels):
            raise RuntimeError("无法打开麦克风（错误码 2）")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            recorder = AudioRecorder(path, source_factory=broken)
            recorder.start()
            recorder.stop()
            self.assertIn("麦克风", recorder.error)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
