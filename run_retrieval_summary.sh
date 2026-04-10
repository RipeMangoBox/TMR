#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TMR_EVAL_DIR="RUN_DIR/contrastive_metrics"
DEFAULT_MOTIONPATCHES_EVAL_DIR="MotionPatches-main/checkpoints/pretrained/HumanML3D/contrastive_metrics"
DEFAULT_EVENTT2M_EVAL_DIR="EventT2M-codes-main/checkpoints/pretrained/HumanML3D/eval"
DEFAULT_OUTPUT="retrieval_results_summary.md"

if [[ $# -eq 0 ]]; then
  echo "Using default retrieval metric directories:"
  echo "  TMR: ${DEFAULT_TMR_EVAL_DIR}"
  echo "  MotionPatches: ${DEFAULT_MOTIONPATCHES_EVAL_DIR}"
  echo "  EventT2M: ${DEFAULT_EVENTT2M_EVAL_DIR}"
  echo "Default summary output: ${DEFAULT_OUTPUT}"

  python3 "${SCRIPT_DIR}/summarize_retrieval_results.py" \
    --eval-dir "${DEFAULT_TMR_EVAL_DIR}" \
    --eval-dir "${DEFAULT_MOTIONPATCHES_EVAL_DIR}" \
    --eval-dir "${DEFAULT_EVENTT2M_EVAL_DIR}" \
    --name TMR \
    --name MotionPatches \
    --name EventT2M \
    --output "${DEFAULT_OUTPUT}"
else
  python3 "${SCRIPT_DIR}/summarize_retrieval_results.py" "$@"
fi
