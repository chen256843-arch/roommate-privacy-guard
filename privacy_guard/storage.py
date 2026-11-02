"""Local, per-user storage for settings, recordings, snapshots and the owner template.

By default everything lives under ``~/PrivacyGuard`` and never leaves the
machine. In portable mode (a ``portable.ini`` file beside the executable) it
lives in a ``data`` folder next to the program instead, so the whole folder can
be moved or carried on a USB drive. Recordings (video/audio sessions) and snapshots (photos/screenshots)
are kept in two separate folders, each of which the user may relocate. The
owner face template is a small set of 128-D embeddings; the original images
are not kept. Reads tolerate missing or corrupt files and fall back to
defaults so a damaged profile never blocks the app.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PORTABLE_MARKER = "portable.ini"
_PW_ITERATIONS = 200_000


def _pw_hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _PW_ITERATIONS
    ).hex()


def has_password(settings: dict) -> bool:
    return bool(settings.get("pw_hash")) and bool(settings.get("pw_salt"))


def set_password(settings: dict, password: str) -> None:
    salt = os.urandom(16).hex()
    settings["pw_salt"] = salt
    settings["pw_hash"] = _pw_hash(password, salt)


def verify_password(settings: dict, password: str) -> bool:
    salt, stored = settings.get("pw_salt"), settings.get("pw_hash")
    if not salt or not stored:
        return False
    try:
        return hmac.compare_digest(_pw_hash(password, salt), stored)
    except Exception:
        return False


def clear_password(settings: dict) -> None:
    settings.pop("pw_salt", None)
    settings.pop("pw_hash", None)


def clear_media(*folders: Path) -> tuple[int, int]:
    """Delete everything *inside* each folder, keeping the folders themselves.

    Only the contents of the given recordings/snapshots folders are touched.
    Returns (removed, failed) item counts.
    """
    removed = failed = 0
    for folder in folders:
        path = Path(folder)
        if not path.is_dir():
            continue
        for item in path.iterdir():
            try:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                removed += 1
            except Exception:
                failed += 1
    return removed, failed


def check_writable(folder: Path) -> str:
    """Create ``folder`` if needed and confirm a file can be written there."""
    try:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".privacy_guard_write_test"
        probe.write_bytes(b"ok")
        probe.unlink()
    except Exception as exc:
        return f"无法写入该文件夹：{exc}"
    return ""


def resolve_data_dir(app_dir: Path | None = None, frozen: bool | None = None) -> tuple[Path, bool]:
    """Return (data folder, portable?).

    Portable mode needs the marker beside the program and a writable ``data``
    folder there; otherwise (e.g. a read-only location) data falls back to the
    user profile so the app still works.
    """
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    if app_dir is None:
        app_dir = Path(sys.executable).resolve().parent if frozen else Path(__file__).resolve().parents[1]
    app_dir = Path(app_dir)
    if (app_dir / PORTABLE_MARKER).is_file():
        data = app_dir / "data"
        if not check_writable(data):
            return data, True
    return Path.home() / "PrivacyGuard", False


DATA_DIR, PORTABLE = resolve_data_dir()
RECORDINGS_DIR = DATA_DIR / "recordings"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
OWNER_FILE = DATA_DIR / "owner.npy"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "camera_index": 0,
    "confidence": 0.80,
    "confirm_seconds": 0.8,
    "auto_record_2p": True,
    "record_screen_on_2p": False,
    "record_audio": False,
    "record_2p_off": 3.0,   # stop 2-person recording after N s below threshold
    "guard_off": 5.0,       # stop guard recording after N s without a lone stranger
    "max_segment_min": 0.0,  # auto-split a recording every N minutes (0 = no split)
    "manual_stop_min": 0.0,  # auto-stop manual recording after N minutes (0 = off)
    "snapshot_2p": True,
    "snapshot_screen": True,
    "auto_close_owner": True,  # close the blackout once only the owner remains
    "guard_snapshot": True,    # unattended trigger: snapshot first, then record
    "guard_enabled": False,
    "rearm_seconds": 60.0,
    "recordings_dir": "",  # empty -> RECORDINGS_DIR
    "snapshots_dir": "",  # empty -> SNAPSHOTS_DIR
    "pw_salt": "",  # secondary password (salted PBKDF2 hash; never plaintext)
    "pw_hash": "",
}


def _safe(text: str, fallback: str) -> str:
    return "".join(ch for ch in text if ch.isalnum() or ch in "._-") or fallback


def _unique(path: Path) -> Path:
    """Append _2, _3… so two events in the same second never overwrite."""
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def ensure_dirs(*bases: Path) -> None:
    for base in bases or (RECORDINGS_DIR, SNAPSHOTS_DIR):
        Path(base).mkdir(parents=True, exist_ok=True)


def new_session_dir(reason: str, base: Path | None = None) -> Path:
    """Create a timestamped folder for one recording session."""
    from .recording import session_timestamp

    root = Path(base) if base else RECORDINGS_DIR
    path = _unique(root / f"{session_timestamp()}_{_safe(reason, '录像')}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_paths(reason: str, base: Path | None = None) -> tuple[Path, Path]:
    """Matching (camera, screen) JPEG paths for one snapshot event."""
    from .recording import session_timestamp

    root = Path(base) if base else SNAPSHOTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{session_timestamp()}_{_safe(reason, '抓拍')}"
    camera = root / f"{stem}_camera.jpg"
    screen = root / f"{stem}_screen.jpg"
    index = 2
    while camera.exists() or screen.exists():
        camera = root / f"{stem}_{index}_camera.jpg"
        screen = root / f"{stem}_{index}_screen.jpg"
        index += 1
    return camera, screen


def validate_dirs(recordings: Path, snapshots: Path) -> str:
    """Return an error message, or "" when the two folders are usable and distinct."""
    try:
        rec = Path(recordings).expanduser().resolve()
        snap = Path(snapshots).expanduser().resolve()
    except Exception:
        return "无法识别所选文件夹。"
    if rec == snap:
        return "录像和抓拍需要分开存放，请选择两个不同的文件夹。"
    return ""


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_FILE.is_file():
            stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for key in DEFAULT_SETTINGS:
                if key in stored:
                    settings[key] = stored[key]
    except Exception:
        pass
    return settings


def save_settings(settings: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def load_owner() -> list[np.ndarray]:
    try:
        if OWNER_FILE.is_file():
            data = np.load(OWNER_FILE)
            if data.ndim == 2 and data.shape[0] >= 1:
                return [np.ascontiguousarray(row.reshape(1, -1)) for row in data]
    except Exception:
        pass
    return []


def save_owner(features: list[np.ndarray]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        stacked = np.vstack([np.asarray(f).reshape(1, -1) for f in features])
        np.save(OWNER_FILE, stacked)
    except Exception:
        pass


def clear_owner() -> None:
    try:
        OWNER_FILE.unlink(missing_ok=True)
    except Exception:
        pass
