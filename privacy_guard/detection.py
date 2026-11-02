"""Local YuNet face counting, optional SFace recognition, and a camera worker.

Live camera images stay in memory for detection and preview. Frames are
written to disk only while recording is explicitly switched on (see
:mod:`privacy_guard.recording`); nothing here uploads anything.
"""

from __future__ import annotations

import math
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .recording import VideoSink, save_image

# OpenCV's recommended cosine threshold for "same identity" with SFace.
OWNER_MATCH_THRESHOLD = 0.363


class FaceCounter:
    """Thin, synchronous wrapper around OpenCV Zoo's 2023mar YuNet model.

    ``detect`` accepts a BGR uint8 image and returns one 15-value row per face:
    x, y, width, height, five landmark pairs, confidence.
    """

    def __init__(self, model_path: Path, confidence: float = 0.8) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到人脸检测模型：{model_path.name}")
        if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
            raise ValueError("人脸检测置信度必须大于 0 且不超过 1。")
        self._input_size = (320, 320)
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            self._input_size,
            float(confidence),
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Count candidates on one image; no camera or GUI is required."""
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] < 1
            or frame.shape[1] < 1
            or frame.dtype != np.uint8
        ):
            raise ValueError("检测图像必须是非空的 BGR uint8 三通道图像。")
        image_size = (frame.shape[1], frame.shape[0])
        if image_size != self._input_size:
            self._detector.setInputSize(image_size)
            self._input_size = image_size
        _, faces = self._detector.detect(np.ascontiguousarray(frame))
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return faces


class FaceRecognizer:
    """Wrap OpenCV Zoo's SFace model for 1:1 owner verification.

    ``feature`` aligns and embeds one detected face (a 15-value YuNet row) into
    a 128-D vector. ``cosine`` compares two embeddings; values at or above
    :data:`OWNER_MATCH_THRESHOLD` indicate the same person.
    """

    def __init__(self, model_path: Path) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到人脸识别模型：{model_path.name}")
        self._recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")

    def feature(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        row = np.ascontiguousarray(face_row, dtype=np.float32).reshape(1, -1)
        aligned = self._recognizer.alignCrop(np.ascontiguousarray(frame), row)
        return self._recognizer.feature(aligned).copy()

    def cosine(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        return float(
            self._recognizer.match(
                feature_a, feature_b, cv2.FaceRecognizerSF_FR_COSINE
            )
        )

    def best_match(self, feature: np.ndarray, owner_features) -> float:
        """Highest cosine similarity between ``feature`` and any owner sample."""
        best = -1.0
        for sample in owner_features:
            best = max(best, self.cosine(feature, sample))
        return best


def largest_face_index(faces: np.ndarray) -> int:
    """Index of the face with the biggest box; ``-1`` when there are none."""
    if faces is None or len(faces) == 0:
        return -1
    areas = [float(face[2]) * float(face[3]) for face in faces]
    return int(np.argmax(areas))


def _fit_frame(frame: np.ndarray) -> np.ndarray:
    """Keep inference and preview within 640×480 without distorting faces."""
    height, width = frame.shape[:2]
    scale = min(640.0 / width, 480.0 / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if new_size == (width, height):
        return frame
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, new_size, interpolation=interpolation)


def _canvas(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Letterbox a fitted frame onto the fixed 640×480 preview/record canvas."""
    height, width = frame.shape[:2]
    canvas = np.full((480, 640, 3), (24, 19, 16), dtype=np.uint8)
    left, top = (640 - width) // 2, (480 - height) // 2
    canvas[top : top + height, left : left + width] = frame
    return canvas, left, top


def _preview_image(
    frame: np.ndarray, faces: np.ndarray, box_colors=None
) -> QImage:
    """Draw boxes in the detector's coordinates and letterbox into a QImage.

    ``box_colors`` optionally supplies a per-face BGR color; by default every
    box is drawn in the standard green used across the preview.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for index, face in enumerate(faces):
        x, y, box_width, box_height = [float(value) for value in face[:4]]
        left = max(0, min(width - 1, round(x)))
        top = max(0, min(height - 1, round(y)))
        right = max(0, min(width - 1, round(x + box_width)))
        bottom = max(0, min(height - 1, round(y + box_height)))
        if right <= left or bottom <= top:
            continue
        color = (94, 222, 140)
        if box_colors is not None and index < len(box_colors) and box_colors[index]:
            color = box_colors[index]
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        score = f"{float(face[-1]):.0%}"
        cv2.putText(
            annotated,
            score,
            (left, max(15, top - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    canvas, _, _ = _canvas(annotated)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # The copy owns its bytes after numpy's temporary buffer goes out of scope.
    return QImage(rgb.data, 640, 480, rgb.strides[0], QImage.Format_RGB888).copy()


class CameraWorker(QThread):
    """Read and process the selected camera entirely off the Qt GUI thread.

    Frames are emitted at no more than 10 fps. The timestamp represents the
    fresh camera sample, before inference; consumers can ignore stale results.

    When guard mode is enabled via :meth:`set_guard`, each frame is also matched
    against the enrolled owner and an :attr:`analysis` signal reports whether the
    owner is present and whether a single stranger is alone on camera. While
    :meth:`set_enrolling` is on, the largest face's embedding is emitted through
    :attr:`owner_sample`. Silent camera recording is toggled with
    :meth:`set_camera_recording` and a one-off photo is requested with
    :meth:`request_snapshot`; frames are written in this thread so the GUI
    never handles raw video.

    ``stop`` requests interruption. Some camera drivers can block inside read;
    the GUI must retain this worker until ``finished`` is emitted.
    """

    frame_ready = Signal(object, int, float)
    analysis = Signal(bool, bool, float)  # owner_present, stranger_alone, timestamp
    owner_sample = Signal(object)  # 128-D embedding of the largest enrolled face
    snapshot_saved = Signal(str, bool)  # path, success
    owner_verified = Signal(bool)  # result of a one-shot live owner face check
    failed = Signal(str)
    opened = Signal()

    # Owner/guest preview box colors (BGR).
    _OWNER_COLOR = (120, 222, 94)
    _GUEST_COLOR = (70, 110, 240)

    def __init__(
        self,
        camera_index: int,
        model_path: Path,
        confidence: float = 0.8,
        recognizer_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.camera_index = int(camera_index)
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.recognizer_path = Path(recognizer_path) if recognizer_path else None
        self._lock = threading.Lock()
        self._guard_enabled = False
        self._owner_features: list[np.ndarray] = []
        self._match_threshold = OWNER_MATCH_THRESHOLD
        self._enrolling = False
        self._rec_on = False
        self._rec_dir: Path | None = None
        self._snapshot: tuple[Path, str] | None = None
        self._verify_requested = False
        self._identify = False  # recognize faces while a blackout is shown

    # ---- GUI-thread controls (thread-safe) --------------------------------
    def set_guard(self, enabled: bool, owner_features=None, threshold=None) -> None:
        with self._lock:
            self._guard_enabled = bool(enabled)
            if owner_features is not None:
                self._owner_features = list(owner_features)
            if threshold is not None:
                self._match_threshold = float(threshold)

    def set_enrolling(self, enrolling: bool) -> None:
        with self._lock:
            self._enrolling = bool(enrolling)

    def set_camera_recording(self, on: bool, session_dir: Path | None = None) -> None:
        with self._lock:
            self._rec_on = bool(on)
            self._rec_dir = Path(session_dir) if session_dir else None

    def request_snapshot(self, path: Path, label: str = "CAM") -> None:
        """Save the next clean camera frame (no boxes) to ``path``."""
        with self._lock:
            self._snapshot = (Path(path), label)

    def set_identify(self, on: bool) -> None:
        """Keep owner recognition running (e.g. to auto-close a blackout)."""
        with self._lock:
            self._identify = bool(on)

    def request_verify(self) -> None:
        """Verify on the next frame whether the enrolled owner is present."""
        with self._lock:
            self._verify_requested = True

    def stop(self) -> None:
        self.requestInterruption()

    def _take_pending_snapshot(self, frame: np.ndarray) -> None:
        with self._lock:
            pending, self._snapshot = self._snapshot, None
        if pending is None:
            return
        path, label = pending
        try:
            save_image(path, frame, label)
        except Exception:
            self.snapshot_saved.emit(str(path), False)
        else:
            self.snapshot_saved.emit(str(path), True)

    def _open_camera(self) -> cv2.VideoCapture | None:
        backends = (
            (cv2.CAP_DSHOW, cv2.CAP_MSMF)
            if platform.system() == "Windows"
            else (cv2.CAP_ANY,)
        )
        for backend in backends:
            if self.isInterruptionRequested():
                return None
            capture = cv2.VideoCapture()
            try:
                if capture.open(self.camera_index, backend) and capture.isOpened():
                    return capture
            except cv2.error:
                pass
            capture.release()
        if not self.isInterruptionRequested():
            raise RuntimeError(
                "无法打开摄像头。请检查摄像头编号、Windows 摄像头权限，"
                "并关闭正在占用摄像头的应用。"
            )
        return None

    def _wait_until(self, deadline: float) -> None:
        while not self.isInterruptionRequested():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.msleep(max(1, min(25, math.ceil(remaining * 1000))))

    def _load_recognizer(self) -> FaceRecognizer | None:
        if self.recognizer_path is None or not self.recognizer_path.is_file():
            return None
        try:
            return FaceRecognizer(self.recognizer_path)
        except Exception:
            return None

    def _analyze(self, recognizer, frame, faces):
        """Return (owner_present, stranger_alone, box_colors, largest_feature)."""
        with self._lock:
            owner_features = list(self._owner_features)
            threshold = self._match_threshold
            enrolling = self._enrolling
        box_colors = [None] * len(faces)
        largest = largest_face_index(faces)
        largest_feature = None
        if recognizer is None:
            return None, False, box_colors, None
        owner_present = False
        matched = [False] * len(faces)
        if owner_features:
            for index, face in enumerate(faces):
                try:
                    feature = recognizer.feature(frame, face)
                except Exception:
                    continue
                if recognizer.best_match(feature, owner_features) >= threshold:
                    matched[index] = True
                    owner_present = True
                box_colors[index] = (
                    self._OWNER_COLOR if matched[index] else self._GUEST_COLOR
                )
        if enrolling and largest >= 0:
            try:
                largest_feature = recognizer.feature(frame, faces[largest])
            except Exception:
                largest_feature = None
        stranger_alone = len(faces) == 1 and not owner_present and bool(owner_features)
        return owner_present, stranger_alone, box_colors, largest_feature

    def _record_frame(self, sink, frame) -> "VideoSink | None":
        """Open/continue/close the camera sink to match the recording flag."""
        with self._lock:
            rec_on = self._rec_on
            rec_dir = self._rec_dir
        if rec_on and rec_dir is not None:
            if sink is None:
                sink = VideoSink(rec_dir / "camera.mp4", fps=10.0, label="CAM")
            canvas, _, _ = _canvas(frame)
            sink.write(canvas)
            return sink
        if sink is not None:
            sink.close()
        return None

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        sink: VideoSink | None = None
        try:
            if self.isInterruptionRequested():
                return
            counter = FaceCounter(self.model_path, self.confidence)
            # The SFace model (~38 MB) is loaded lazily, only once guard mode,
            # enrollment or a verification first needs it, so pure multi-person
            # monitoring never pays its memory or startup cost.
            recognizer: FaceRecognizer | None = None
            recognizer_tried = False
            capture = self._open_camera()
            if capture is None or self.isInterruptionRequested():
                return
            for prop, value in (
                (cv2.CAP_PROP_FRAME_WIDTH, 640),
                (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                (cv2.CAP_PROP_FPS, 10),
                (cv2.CAP_PROP_BUFFERSIZE, 1),
            ):
                try:
                    capture.set(prop, value)
                except cv2.error:
                    # Camera drivers may ignore optional format requests.
                    pass
            if self.isInterruptionRequested():
                return
            self.opened.emit()
            next_sample = time.monotonic()
            failed_reads = 0
            while not self.isInterruptionRequested():
                self._wait_until(next_sample)
                if self.isInterruptionRequested():
                    break
                ok, frame = capture.read()
                sampled_at = time.monotonic()
                next_sample = sampled_at + 0.1
                if self.isInterruptionRequested():
                    break
                if not ok or frame is None or frame.size == 0:
                    failed_reads += 1
                    if failed_reads >= 5:
                        raise RuntimeError(
                            "摄像头连续读取失败。请重新连接摄像头，并检查是否被其他应用占用。"
                        )
                    continue
                failed_reads = 0
                mirrored = _fit_frame(cv2.flip(frame, 1))
                faces = counter.detect(mirrored)
                if self.isInterruptionRequested():
                    break

                with self._lock:
                    verify = self._verify_requested
                    self._verify_requested = False
                    need_analysis = (
                        self._guard_enabled or self._enrolling or self._identify or verify
                    )
                if need_analysis and not recognizer_tried:
                    recognizer = self._load_recognizer()
                    recognizer_tried = True
                box_colors = None
                if need_analysis and recognizer is not None:
                    owner_present, stranger_alone, box_colors, sample = self._analyze(
                        recognizer, mirrored, faces
                    )
                    if sample is not None:
                        self.owner_sample.emit(sample)
                    if verify:
                        self.owner_verified.emit(bool(owner_present))
                    if owner_present is not None:
                        self.analysis.emit(
                            bool(owner_present), bool(stranger_alone), sampled_at
                        )
                elif verify:
                    # No recognizer or no owner template: cannot confirm identity.
                    self.owner_verified.emit(False)

                # Photo first, then recording: a snapshot requested together with
                # a new recording is taken from the frame before writing starts.
                self._take_pending_snapshot(mirrored)
                sink = self._record_frame(sink, mirrored)
                preview = _preview_image(mirrored, faces, box_colors)
                self.frame_ready.emit(preview, len(faces), sampled_at)
        except Exception as exc:
            if not self.isInterruptionRequested():
                detail = str(exc).strip() or type(exc).__name__
                self.failed.emit(f"摄像头检测未能继续：{detail}")
        finally:
            if sink is not None:
                sink.close()
            if capture is not None:
                capture.release()
