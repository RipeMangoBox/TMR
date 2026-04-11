#!/usr/bin/env bash
set -euo pipefail

# This script intentionally runs the corrected real-data rerun in strict serial
# order and stops on the first failing stage.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

START_STAGE="${START_STAGE:-d0}"
END_STAGE="${END_STAGE:-d3}"
REPORT_DATE="${REPORT_DATE:-$(date +%F)}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-256}"
RUN_RETRIEVAL="${RUN_RETRIEVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE_RETRIEVAL="${FORCE_RETRIEVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_stage4_1_realdata_batch.sh [options] [extra Hydra train args...]

Options:
  --start-stage <d0|d1|d1_5|d2a|d2b|d3>
  --end-stage <d0|d1|d1_5|d2a|d2b|d3>
  --report-date <YYYY-MM-DD>
  --epochs <int>
  --batch-size <int>
  --num-workers <int>
  --retrieval-batch-size <int>
  --skip-retrieval
  --skip-summary
  --force-retrieval
  --dry-run
  -h, --help

Environment overrides are also supported:
  START_STAGE, END_STAGE, REPORT_DATE, TRAIN_EPOCHS, BATCH_SIZE, NUM_WORKERS,
  RETRIEVAL_BATCH_SIZE, RUN_RETRIEVAL, RUN_SUMMARY, FORCE_RETRIEVAL, DRY_RUN

Examples:
  bash scripts/run_stage4_1_realdata_batch.sh --report-date 2026-04-11
  bash scripts/run_stage4_1_realdata_batch.sh --start-stage d1 --end-stage d2b --epochs 3
  bash scripts/run_stage4_1_realdata_batch.sh --dry-run trainer.accelerator=gpu
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage)
      START_STAGE="$2"
      shift 2
      ;;
    --end-stage)
      END_STAGE="$2"
      shift 2
      ;;
    --report-date)
      REPORT_DATE="$2"
      shift 2
      ;;
    --epochs)
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --retrieval-batch-size)
      RETRIEVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --skip-retrieval)
      RUN_RETRIEVAL=0
      shift
      ;;
    --skip-summary)
      RUN_SUMMARY=0
      shift
      ;;
    --force-retrieval)
      FORCE_RETRIEVAL=1
      shift
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
      TRAIN_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

STAGE_ORDER=(d0 d1 d1_5 d2a d2b d3)

stage_index() {
  local needle="$1"
  local idx=0
  for stage in "${STAGE_ORDER[@]}"; do
    if [[ "${stage}" == "${needle}" ]]; then
      echo "${idx}"
      return 0
    fi
    idx=$((idx + 1))
  done
  echo "Unknown stage: ${needle}" >&2
  exit 1
}

should_run_stage() {
  local stage="$1"
  local stage_idx start_idx end_idx
  stage_idx="$(stage_index "${stage}")"
  start_idx="$(stage_index "${START_STAGE}")"
  end_idx="$(stage_index "${END_STAGE}")"
  [[ "${stage_idx}" -ge "${start_idx}" && "${stage_idx}" -le "${end_idx}" ]]
}

run_cmd() {
  echo
  echo "[$(date '+%F %T')] + $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

stage_banner() {
  local label="$1"
  echo
  echo "===== ${label} ====="
  echo "[batch] stage=${label} report_date=${REPORT_DATE}"
}

run_d0() {
  run_cmd conda run -n TMR python scripts/d0_humanml3de_event_stats.py \
    --report-date "${REPORT_DATE}"
}

run_train_stage() {
  local stage_label="$1"
  local script_path="$2"
  run_cmd bash "${script_path}" \
    trainer.max_epochs="${TRAIN_EPOCHS}" \
    dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    "${TRAIN_EXTRA_ARGS[@]}"
}

extract_d1_weights() {
  if [[ -d "RUN_DIR/stage4_1_realdata_d1/last_weights" && -n "$(find RUN_DIR/stage4_1_realdata_d1/last_weights -maxdepth 1 -type f -name '*.pt' -print -quit)" ]]; then
    echo
    echo "[batch] D1 last_weights already exist, skip extraction."
    return 0
  fi
  run_cmd conda run -n TMR python - <<'PY'
from src.load import extract_ckpt
extract_ckpt("RUN_DIR/stage4_1_realdata_d1")
PY
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

run_retrieval_stage() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/config.json" ]]; then
    echo
    echo "[batch] Skip retrieval for ${run_dir}: missing config.json"
    return 0
  fi
  if [[ "${FORCE_RETRIEVAL}" != "1" ]] && has_retrieval_metrics "${run_dir}"; then
    echo
    echo "[batch] Retrieval metrics already exist for ${run_dir}, skip."
    return 0
  fi
  run_cmd conda run -n TMR python retrieval.py \
    run_dir="${run_dir}" \
    protocol=all \
    batch_size="${RETRIEVAL_BATCH_SIZE}"
}

run_summary() {
  run_cmd bash scripts/eval_stage4_1_realdata_all.sh \
    --skip-retrieval \
    --report-date "${REPORT_DATE}"
}

echo "[batch] repo=${REPO_DIR}"
echo "[batch] start_stage=${START_STAGE} end_stage=${END_STAGE}"
echo "[batch] report_date=${REPORT_DATE}"
echo "[batch] epochs=${TRAIN_EPOCHS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS}"
echo "[batch] retrieval_batch_size=${RETRIEVAL_BATCH_SIZE}"
echo "[batch] run_retrieval=${RUN_RETRIEVAL} run_summary=${RUN_SUMMARY} force_retrieval=${FORCE_RETRIEVAL} dry_run=${DRY_RUN}"
if [[ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "[batch] extra_train_args=${TRAIN_EXTRA_ARGS[*]}"
fi

if should_run_stage d0; then
  stage_banner "D0"
  run_d0
fi

if should_run_stage d1; then
  stage_banner "D1"
  run_train_stage "D1" "scripts/run_stage4_1_realdata_d1.sh"
  extract_d1_weights
  if [[ "${RUN_RETRIEVAL}" == "1" ]]; then
    run_retrieval_stage "RUN_DIR/stage4_1_realdata_d1"
  fi
fi

if should_run_stage d1_5; then
  stage_banner "D1.5"
  run_train_stage "D1.5" "scripts/run_stage4_1_realdata_d1_5.sh"
  if [[ "${RUN_RETRIEVAL}" == "1" ]]; then
    run_retrieval_stage "RUN_DIR/stage4_1_realdata_d1_5"
  fi
fi

if should_run_stage d2a; then
  stage_banner "D2a"
  extract_d1_weights
  run_train_stage "D2a" "scripts/run_stage4_1_realdata_d2a.sh"
  if [[ "${RUN_RETRIEVAL}" == "1" ]]; then
    run_retrieval_stage "RUN_DIR/stage4_1_realdata_d2a"
  fi
fi

if should_run_stage d2b; then
  stage_banner "D2b"
  extract_d1_weights
  run_train_stage "D2b" "scripts/run_stage4_1_realdata_d2b.sh"
  if [[ "${RUN_RETRIEVAL}" == "1" ]]; then
    run_retrieval_stage "RUN_DIR/stage4_1_realdata_d2b"
  fi
fi

if should_run_stage d3 && [[ "${RUN_SUMMARY}" == "1" ]]; then
  stage_banner "D3"
  run_summary
fi
