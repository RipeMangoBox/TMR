#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_DIR}/MotionPatches-main/scripts/run_motion_repr_t5_pos66_serial.sh" "$@"
