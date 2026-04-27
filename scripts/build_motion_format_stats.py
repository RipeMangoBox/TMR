#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

from linked_codebases import MOTIONPATCHES_ROOT

TARGET = MOTIONPATCHES_ROOT / "scripts" / "build_motion_format_stats.py"

runpy.run_path(str(TARGET), run_name="__main__")
