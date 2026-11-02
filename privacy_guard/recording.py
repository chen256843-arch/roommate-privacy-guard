"""Silent, timestamped camera, screen and microphone recording, plus snapshots.

Recordings are written locally as timestamped ``.mp4`` / ``.wav`` files and
snapshots as ``.jpg``; nothing is uploaded. Every written video frame and image
carries a burned-in ``YYYY-MM-DD HH:MM:SS`` stamp so clips can be trusted as
evidence. Nothing here draws on the live desktop, so recording stays
unobtrusive ("无感录像").

Screen capture uses the Windows GDI and microphone capture the Windows winmm
``waveIn`` API, both through ``ctypes`` with no extra dependency. ``grab_fn``
and ``source_factory`` can be injected for other platforms or tests.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def session_timestamp() -> str:
    """Filesystem-safe stamp for one recording session folder."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def draw_timestamp(frame: np.ndarray, label: str | None = None) -> np.ndarray:
    """Burn the wall-clock time (and optional label) into a BGR frame in place."""
    text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if label:
        text = f"{text}  {label}"
    height = frame.shape[0]
    scale = max(0.5, min(0.9, height / 640.0))
    thickness = 1 if scale < 0.75 else 2
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x, y = 10, height - 12
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - baseline - 4),
        (x + text_w + 5, y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        frame, text, (x, y - baseline + 1),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return frame


class VideoSink:
    """A fixed-size ``cv2.VideoWriter`` that stamps and normalizes each frame.

    The first written frame fixes the output size; later frames are resized to
    match. ``write`` and ``close`` tolerate a writer that failed to open so a
    recording problem never crashes detection.
    """

    def __init__(self, path: Path, fps: float = 10.0, label: str | None = None) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 10.0
        self.label = label
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None

    @property
    def opened(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            # Even dimensions keep common codecs happy.
            self._size = (width - (width % 2), height - (height % 2))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.path), fourcc, self.fps, self._size
            )
        if self._size is not None and (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        stamped = draw_timestamp(np.ascontiguousarray(frame), self.label)
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(stamped)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def save_image(path: Path, frame: np.ndarray, label: str | None = None, quality: int = 92) -> Path:
    """Stamp a copy of a BGR frame and save it as JPEG.

    ``cv2.imwrite`` cannot open non-ASCII paths on Windows (it fails silently),
    so the image is encoded in memory and written with Python file I/O.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = draw_timestamp(np.ascontiguousarray(frame).copy(), label)
    ok, data = cv2.imencode(".jpg", stamped, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("图像编码失败")
    path.write_bytes(data.tobytes())
    return path


def grab_screen_gdi() -> np.ndarray:
    """Capture the full virtual desktop as a BGR image using the Windows GDI."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN) or user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN) or user32.GetSystemMetrics(1)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    hdc = user32.GetDC(0)
    memdc = gdi32.CreateCompatibleDC(hdc)
    bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(memdc, bitmap)
    SRCCOPY = 0x00CC0020
    gdi32.BitBlt(memdc, 0, 0, width, height, hdc, left, top, SRCCOPY)
    header = BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    header.biWidth = width
    header.biHeight = -height  # top-down rows
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = 0
    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(memdc, bitmap, 0, height, buffer, ctypes.byref(header), 0)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(0, hdc)
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
    return image[:, :, :3].copy()


class ScreenRecorder:
    """Record the desktop to a timestamped file on a background thread.

    Grabbing, stamping and encoding all run off the GUI thread so the desktop
    stays responsive. Capture failures are counted and stop the recorder rather
    than raising. ``grab_fn`` is injectable for non-Windows hosts and tests.
    """

    def __init__(
        self,
        path: Path,
        fps: float = 8.0,
        grab_fn=None,
        label: str = "SCREEN",
    ) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 8.0
        self.grab_fn = grab_fn or grab_screen_gdi
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_written = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="screen-recorder", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        sink = VideoSink(self.path, self.fps, label=self.label)
        interval = 1.0 / self.fps
        failures = 0
        try:
            next_at = time.monotonic()
            while not self._stop.is_set():
                try:
                    frame = self.grab_fn()
                except Exception:
                    failures += 1
                    if failures >= 5:
                        break
                    self._stop.wait(interval)
                    continue
                failures = 0
                if frame is not None and getattr(frame, "size", 0):
                    sink.write(np.ascontiguousarray(frame))
                    self.frames_written += 1
                next_at += interval
                delay = next_at - time.monotonic()
                if delay < 0:
                    next_at = time.monotonic()
                    delay = 0
                self._stop.wait(delay)
        finally:
            sink.close()


class WinMMAudioSource:
    """Default Windows microphone through winmm ``waveIn`` (16-bit PCM).

    A small ring of buffers is queued to the driver; :meth:`read` returns the
    bytes of the next completed buffer (or ``None`` when none is ready yet) and
    re-queues it. :meth:`close` stops capture and returns any final audio.
    """

    WAVE_MAPPER = 0xFFFFFFFF
    WHDR_DONE = 0x00000001
    WHDR_PREPARED = 0x00000002

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 buffer_seconds: float = 0.25, buffers: int = 4) -> None:
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        winmm = ctypes.WinDLL("winmm")
        self._winmm = winmm

        class WAVEFORMATEX(ctypes.Structure):
            _fields_ = [
                ("wFormatTag", wintypes.WORD),
                ("nChannels", wintypes.WORD),
                ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD),
                ("nBlockAlign", wintypes.WORD),
                ("wBitsPerSample", wintypes.WORD),
                ("cbSize", wintypes.WORD),
            ]

        class WAVEHDR(ctypes.Structure):
            _fields_ = [
                ("lpData", ctypes.c_void_p),
                ("dwBufferLength", wintypes.DWORD),
                ("dwBytesRecorded", wintypes.DWORD),
                ("dwUser", ctypes.c_size_t),
                ("dwFlags", wintypes.DWORD),
                ("dwLoops", wintypes.DWORD),
                ("lpNext", ctypes.c_void_p),
                ("reserved", ctypes.c_size_t),
            ]

        self._hdr_size = ctypes.sizeof(WAVEHDR)
        winmm.waveInOpen.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.POINTER(WAVEFORMATEX),
            ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD,
        ]
        winmm.waveInOpen.restype = ctypes.c_uint
        for name in ("waveInPrepareHeader", "waveInUnprepareHeader", "waveInAddBuffer"):
            fn = getattr(winmm, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
            fn.restype = ctypes.c_uint
        for name in ("waveInStart", "waveInReset", "waveInClose"):
            fn = getattr(winmm, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_uint

        fmt = WAVEFORMATEX()
        fmt.wFormatTag = 1  # WAVE_FORMAT_PCM
        fmt.nChannels = channels
        fmt.nSamplesPerSec = sample_rate
        fmt.wBitsPerSample = 16
        fmt.nBlockAlign = channels * 2
        fmt.nAvgBytesPerSec = sample_rate * fmt.nBlockAlign
        fmt.cbSize = 0

        self._headers = []
        self._buffers = []
        self._next = 0
        self._handle = ctypes.c_void_p()
        result = winmm.waveInOpen(ctypes.byref(self._handle), self.WAVE_MAPPER,
                                  ctypes.byref(fmt), 0, 0, 0)
        if result != 0:
            self._handle = None
            raise RuntimeError(
                f"无法打开麦克风（错误码 {result}）。请检查麦克风连接，"
                "并在 Windows 设置中允许桌面应用访问麦克风。"
            )
        size = max(fmt.nBlockAlign, int(sample_rate * buffer_seconds) * fmt.nBlockAlign)
        try:
            for _ in range(buffers):
                buffer = ctypes.create_string_buffer(size)
                header = WAVEHDR()
                header.lpData = ctypes.cast(buffer, ctypes.c_void_p)
                header.dwBufferLength = size
                self._check(winmm.waveInPrepareHeader(self._handle, ctypes.byref(header), self._hdr_size))
                self._buffers.append(buffer)
                self._headers.append(header)
                self._check(winmm.waveInAddBuffer(self._handle, ctypes.byref(header), self._hdr_size))
            self._check(winmm.waveInStart(self._handle))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(result: int) -> None:
        if result != 0:
            raise RuntimeError(f"麦克风录音失败（错误码 {result}）")

    def read(self) -> bytes | None:
        if self._handle is None or not self._headers:
            return None
        header = self._headers[self._next]
        if not header.dwFlags & self.WHDR_DONE:
            return None
        data = self._ct.string_at(self._buffers[self._next], header.dwBytesRecorded)
        header.dwBytesRecorded = 0
        header.dwFlags = self.WHDR_PREPARED
        self._winmm.waveInAddBuffer(self._handle, self._ct.byref(header), self._hdr_size)
        self._next = (self._next + 1) % len(self._headers)
        return data

    def close(self) -> bytes:
        if self._handle is None:
            return b""
        tail = []
        self._winmm.waveInReset(self._handle)  # returns every queued buffer as done
        for offset in range(len(self._headers)):
            index = (self._next + offset) % len(self._headers)
            header = self._headers[index]
            if header.dwFlags & self.WHDR_DONE and header.dwBytesRecorded:
                tail.append(self._ct.string_at(self._buffers[index], header.dwBytesRecorded))
        for header in self._headers:
            self._winmm.waveInUnprepareHeader(self._handle, self._ct.byref(header), self._hdr_size)
        self._winmm.waveInClose(self._handle)
        self._handle = None
        return b"".join(tail)


class AudioRecorder:
    """Record the microphone to a 16-bit PCM WAV file on a background thread.

    If the microphone cannot be opened, :attr:`error` explains why and no file
    is created; video recording in the same session continues unaffected.
    ``source_factory(sample_rate, channels)`` is injectable for tests.
    """

    def __init__(self, path: Path, sample_rate: int = 16000, channels: int = 1,
                 source_factory=None) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.source_factory = source_factory or WinMMAudioSource
        self.error = ""
        self.bytes_written = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="audio-recorder", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        import wave

        try:
            source = self.source_factory(self.sample_rate, self.channels)
        except Exception as exc:
            self.error = str(exc).strip() or type(exc).__name__
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(self.path), "wb") as wav:
                wav.setnchannels(self.channels)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                while not self._stop.is_set():
                    data = source.read()
                    if data:
                        wav.writeframes(data)
                        self.bytes_written += len(data)
                    else:
                        self._stop.wait(0.02)
                tail, source = source.close(), None
                if tail:
                    wav.writeframes(tail)
                    self.bytes_written += len(tail)
        except Exception as exc:
            self.error = str(exc).strip() or type(exc).__name__
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
