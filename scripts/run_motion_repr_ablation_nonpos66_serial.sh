#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/linked_codebases.sh"
exec "${MOTIONPATCHES_DIR}/scripts/run_motion_repr_ablation_nonpos66_serial.sh" "$@"
