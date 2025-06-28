#!/usr/bin/env python3
"""
cx_Freeze setup script for Local Whisper Transcribe Service
Alternative to PyInstaller for packaging the application
"""
import sys
import os
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": [
        "torch", "torchaudio", "torchvision",
        "whisper", "resemblyzer", "sklearn",
        "sounddevice", "soundfile", "soundcard",
        "webrtcvad", "numpy", "scipy", "librosa",
        "ffmpeg", "cv2", "PIL", "PyQt6",
        "queue", "threading", "tempfile", "subprocess"
    ],
    "excludes": [
        "matplotlib", "jupyter", "IPython", "pandas",
        "seaborn", "plotly", "bokeh", "dask",
        "distributed", "tornado", "zmq", "notebook",
        "sphinx", "pytest", "unittest", "doctest",
        "pdb", "tkinter", "wx", "PySide", "PySide2", "PySide6"
    ],
    "include_files": [
        ("src", "src"),  # Include source code
        ("README.md", "README.md"),
        ("LICENSE", "LICENSE"),
    ],
    "optimize": 2,
}

# GUI applications require a different base on Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="LocalWhisperTranscriber",
    version="1.0.0",
    description="Local Whisper Transcription Service with GUI",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "main.py",
            base=base,
            target_name="LocalWhisperTranscriber.exe",
            icon=None,  # Add icon path here if you have one
        )
    ]
) 