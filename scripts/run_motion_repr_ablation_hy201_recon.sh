#!/usr/bin/env bash
set -euo pipefail

SCHEMA_OVERRIDE="hy201_recon" \
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-3}" \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_motion_repr_ablation_single.sh" \
  "$@"
