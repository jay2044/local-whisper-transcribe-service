#!/usr/bin/env python3
"""
Build script for Local Whisper Transcribe Service
Uses PyInstaller to create a standalone executable
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✓ PyInstaller already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed")

def create_spec_file():
    """Create a PyInstaller spec file optimized for this app"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Collect all necessary data files
datas = [
    ('src', 'src'),  # Include source code
]

# Hidden imports for ML libraries
hiddenimports = [
    'torch',
    'torchaudio',
    'torchvision',
    'whisper',
    'resemblyzer',
    'sklearn',
    'sklearn.cluster',
    'sklearn.metrics',
    'sounddevice',
    'soundfile',
    'soundcard',
    'webrtcvad',
    'numpy',
    'scipy',
    'librosa',
    'ffmpeg',
    'cv2',
    'PIL',
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
]

# Exclude unnecessary modules to reduce size
excludes = [
    'matplotlib',
    'jupyter',
    'IPython',
    'pandas',
    'seaborn',
    'plotly',
    'bokeh',
    'dask',
    'distributed',
    'tornado',
    'zmq',
    'notebook',
    'sphinx',
    'pytest',
    'unittest',
    'doctest',
    'pdb',
    'tkinter',
    'wx',
    'PySide',
    'PySide2',
    'PySide6',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LocalWhisperTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True if you want console output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalWhisperTranscriber',
)
'''
    
    with open('LocalWhisperTranscriber.spec', 'w') as f:
        f.write(spec_content)
    print("✓ Created PyInstaller spec file")

def build_app():
    """Build the application using PyInstaller"""
    print("Building application with PyInstaller...")
    
    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    # Run PyInstaller
    cmd = [
        'pyinstaller',
        '--clean',
        '--noconfirm',
        'LocalWhisperTranscriber.spec'
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✓ Build completed successfully!")
        print(f"✓ Executable created in: {os.path.abspath('dist/LocalWhisperTranscriber')}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    return True

def create_installer():
    """Create a simple installer script"""
    installer_content = '''@echo off
echo Installing Local Whisper Transcriber...
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator
) else (
    echo Please run this installer as administrator
    pause
    exit /b 1
)

REM Create installation directory
set INSTALL_DIR=C:\\Program Files\\LocalWhisperTranscriber
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy files
echo Copying files...
xcopy /E /I /Y "LocalWhisperTranscriber" "%INSTALL_DIR%"

REM Create desktop shortcut
echo Creating desktop shortcut...
set DESKTOP=%USERPROFILE%\\Desktop
echo @echo off > "%DESKTOP%\\Local Whisper Transcriber.bat"
echo cd /d "%INSTALL_DIR%" >> "%DESKTOP%\\Local Whisper Transcriber.bat"
echo start LocalWhisperTranscriber.exe >> "%DESKTOP%\\Local Whisper Transcriber.bat"

echo.
echo Installation completed!
echo You can now run the application from your desktop shortcut.
pause
'''
    
    with open('install.bat', 'w') as f:
        f.write(installer_content)
    print("✓ Created installer script")

def main():
    """Main build process"""
    print("=" * 60)
    print("Local Whisper Transcriber - Build Script")
    print("=" * 60)
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Create spec file
    create_spec_file()
    
    # Build the app
    if build_app():
        # Create installer
        create_installer()
        
        print("\n" + "=" * 60)
        print("BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Files created:")
        print(f"  - Executable: dist/LocalWhisperTranscriber/")
        print(f"  - Installer: install.bat")
        print("\nTo distribute:")
        print("  1. Zip the 'dist/LocalWhisperTranscriber' folder")
        print("  2. Include the 'install.bat' script")
        print("  3. Share with users")
    else:
        print("\n❌ Build failed. Check the error messages above.")

if __name__ == "__main__":
    main() 