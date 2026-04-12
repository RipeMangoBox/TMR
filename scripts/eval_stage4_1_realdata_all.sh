#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_RETRIEVAL=1
FORCE_RETRIEVAL="${FORCE_RETRIEVAL:-0}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-256}"
REPORT_DATE="${REPORT_DATE:-$(date +%F)}"
RUN_PREFIX="${RUN_PREFIX:-stage4_1_realdata_e50_b128}"
DRY_RUN="${DRY_RUN:-0}"

D0_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d0"
D1_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d1"
D1_5_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d1_5"
D2A_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d2a"
D2B_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d2b"
D3_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d3"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_stage4_1_realdata_all.sh [options]

Options:
  --skip-retrieval
  --force-retrieval
  --retrieval-batch-size <int>
  --report-date <YYYY-MM-DD>
  --run-prefix <prefix>
  --dry-run
  -h, --help

This script:
1. runs retrieval.py for any available corrected real-data stage run_dir
2. generates unified D1/D1.5/D2a/D2b/D3 summary docs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-retrieval)
      RUN_RETRIEVAL=0
      shift
      ;;
    --force-retrieval)
      FORCE_RETRIEVAL=1
      shift
      ;;
    --retrieval-batch-size)
      RETRIEVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --report-date)
      REPORT_DATE="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      D0_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d0"
      D1_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d1"
      D1_5_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d1_5"
      D2A_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d2a"
      D2B_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d2b"
      D3_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d3"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run_cmd() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

has_retrieval_metrics() {
  local run_dir="$1"
  for name in normal.yaml threshold_0.95.yaml nsim.yaml guo.yaml; do
    if [[ ! -f "${run_dir}/contrastive_metrics/${name}" ]]; then
      return 1
    fi
  done
  return 0
}

eval_one() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/config.json" ]]; then
    echo "[eval] Skip ${run_dir}: missing config.json"
    return 0
  fi
  if [[ "${RUN_RETRIEVAL}" != "1" ]]; then
    echo "[eval] Retrieval disabled, skip ${run_dir}"
    return 0
  fi
  if [[ "${FORCE_RETRIEVAL}" != "1" ]] && has_retrieval_metrics "${run_dir}"; then
    echo "[eval] Retrieval metrics already exist for ${run_dir}, skip."
    return 0
  fi
  run_cmd conda run -n TMR python retrieval.py \
    run_dir="${run_dir}" \
    protocol=all \
    batch_size="${RETRIEVAL_BATCH_SIZE}"
}

for run_dir in \
  "${D1_RUN_DIR}" \
  "${D1_5_RUN_DIR}" \
  "${D2A_RUN_DIR}" \
  "${D2B_RUN_DIR}"
do
  eval_one "${run_dir}"
done

run_cmd conda run -n TMR python scripts/summarize_stage4_1_realdata.py \
  --report-date "${REPORT_DATE}" \
  --run-prefix "${RUN_PREFIX}" \
  --d0-dir "${D0_RUN_DIR}" \
  --output-dir "${D3_RUN_DIR}"
