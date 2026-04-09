"""
Compatibility fix: deepfilternet uses torchaudio.backend.common.AudioMetaData
which was removed in torchaudio >= 2.6. This script creates a shim module.
Run once after pip install: python setup_compat.py
"""
import os
import torchaudio

backend_dir = os.path.join(os.path.dirname(torchaudio.__file__), "backend")
os.makedirs(backend_dir, exist_ok=True)

init_path = os.path.join(backend_dir, "__init__.py")
if not os.path.exists(init_path):
    with open(init_path, "w") as f:
        f.write("")

common_path = os.path.join(backend_dir, "common.py")
if not os.path.exists(common_path):
    with open(common_path, "w") as f:
        f.write(
            "from dataclasses import dataclass\n"
            "from typing import Optional\n\n"
            "@dataclass\n"
            "class AudioMetaData:\n"
            "    sample_rate: int = 0\n"
            "    num_frames: int = 0\n"
            "    num_channels: int = 0\n"
            "    bits_per_sample: int = 0\n"
            "    encoding: str = ''\n"
        )
    print(f"Created torchaudio compatibility shim at {common_path}")
else:
    print(f"Shim already exists at {common_path}")
