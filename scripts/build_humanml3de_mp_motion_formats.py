#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "MotionPatches-main"
    / "scripts"
    / "build_humanml3de_mp_motion_formats.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
