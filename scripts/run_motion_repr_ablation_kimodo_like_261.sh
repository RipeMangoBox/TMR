#!/usr/bin/env bash
set -euo pipefail

SCHEMA_OVERRIDE="kimodo_like_261" \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_motion_repr_ablation_single.sh" \
  "$@"
