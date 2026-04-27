#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/linked_codebases.sh"
DEFAULT_TMR_EVAL_DIR="RUN_DIR/contrastive_metrics"
DEFAULT_MOTIONPATCHES_EVAL_DIR="${MOTIONPATCHES_DIR}/checkpoints/pretrained/HumanML3D/contrastive_metrics"
DEFAULT_EVENTT2M_EVAL_DIR="${EVENTT2M_DIR}/checkpoints/pretrained/HumanML3D/eval"
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
