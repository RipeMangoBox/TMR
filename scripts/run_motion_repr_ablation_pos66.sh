#!/usr/bin/env bash
set -euo pipefail

SCHEMA_OVERRIDE="pos66" \
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-1}" \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_motion_repr_ablation_single.sh" \
  "$@"
