# Third-party notices

Application source is MIT licensed. Third-party components retain their own licenses.
No third-party source code or Qt libraries were modified.

| Component | Version | License / source |
| --- | --- | --- |
| YuNet model | OpenCV Zoo f12e12798e8314f7c074a6656816c048dcc95b7a | MIT; https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet |
| SFace model | OpenCV Zoo face_recognition_sface_2021dec.onnx | Apache-2.0; https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface |
| OpenCV Python headless | 4.11.0.86 | Apache-2.0 / bundled third-party terms; https://github.com/opencv/opencv-python and https://github.com/opencv/opencv/tree/4.11.0 |
| NumPy | 2.2.6 | BSD-3-Clause / bundled third-party terms; https://github.com/numpy/numpy/tree/v2.2.6 |
| PySide6 Essentials, Shiboken6, Qt | 6.8.3 | LGPLv3 / applicable Qt module terms; https://code.qt.io/cgit/pyside/pyside-setup.git/tree/?h=v6.8.3 and https://code.qt.io/cgit/qt/qtbase.git/tree/?h=v6.8.3 |
| Python | 3.12.14 | PSF License and third-party terms; https://www.python.org/ and https://github.com/python/cpython/tree/v3.12.14 |
| PyInstaller bootloader | 6.16.0 | GPL with bootloader exception permitting bundled applications; https://github.com/pyinstaller/pyinstaller/tree/v6.16.0 |

License copies are under `assets/licenses` (or `_internal/assets/licenses` in the executable package). The YuNet license is also in `assets/YuNet-LICENSE.txt` and the SFace license in `assets/SFace-LICENSE.txt`.

Screen recording uses the Windows GDI and microphone recording uses the Windows winmm `waveIn` API, both through Python's standard-library `ctypes`; no additional third-party capture library is bundled.

Qt/PySide6/Shiboken6 are dynamically linked. The executable distribution keeps DLLs in the `_internal/PySide6` and `_internal/shiboken6` directories. You may replace these with compatible modified libraries and may reverse engineer this application to debug changes to those libraries as permitted by LGPLv3. This application imposes no restriction on those rights. Complete application source is provided separately with the software, and a compatible environment can be rebuilt with `requirements.txt` and `build.ps1`. Obtain original corresponding Qt and PySide sources from the official repositories above at v6.8.3. Build dependencies are listed in `requirements-build.txt`.

The single-file `PrivacyGuard.exe` contains the same dynamically linked Qt/PySide6 libraries packed inside the executable and unpacks them to a temporary folder at each start. To run with modified Qt libraries, use the folder build (`PrivacyGuard-Windows.zip`) or rebuild from the provided source with `build.ps1` (add `-OneFile` for a single file).

No Windows fonts or test photographs are redistributed. The GUI uses fonts already installed on Windows.
