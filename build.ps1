param([string]$Python = 'python', [switch]$OneFile)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    # -OneFile: one self-contained PrivacyGuard.exe (unpacks to a temp folder at each start).
    # Default: a folder build (PrivacyGuard.exe + _internal) that starts faster.
    $layout = if ($OneFile) { '--onefile' } else { '--onedir' }
    & $Python -m PyInstaller --noconfirm --windowed $layout --name PrivacyGuard --icon assets/app.ico --add-data 'assets;assets' --exclude-module PySide6.QtQml --exclude-module PySide6.QtQuick --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtNetwork main.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
    if (-not $OneFile) {
        Copy-Item -LiteralPath README.md,LICENSE,THIRD_PARTY_NOTICES.md -Destination dist/PrivacyGuard
    }
} finally {
    Pop-Location
}
