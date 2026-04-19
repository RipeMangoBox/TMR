#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


TARGET = (
    Path(__file__).resolve().parents[1]
    / "MotionPatches-main"
    / "scripts"
    / "run_motion_repr_ablation.py"
)

runpy.run_path(str(TARGET), run_name="__main__")
