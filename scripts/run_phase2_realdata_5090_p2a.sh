#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_PREFIX="${RUN_PREFIX:-phase2_realdata_e50_b128}"
PHASE2_SUFFIX="${PHASE2_SUFFIX:-p2a}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-logs/phase2_realdata_5090}"
WARM_START_WEIGHTS_DIR="${WARM_START_WEIGHTS_DIR:-RUN_DIR/stage4_1_realdata_e50_b128_d2b/last_weights}"
RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-RUN_DIR/${RUN_PREFIX}_${PHASE2_SUFFIX}}"

mkdir -p "${LOG_DIR}" "RUN_DIR"
LOG_FILE="${LOG_DIR}/${RUN_PREFIX}_${PHASE2_SUFFIX}_gpu${GPU_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
  echo "[phase2-5090][P2a] ERROR: conda executable not found" >&2
  exit 1
fi

if [[ -z "$(find "${WARM_START_WEIGHTS_DIR}" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null)" ]]; then
  echo "[phase2-5090][P2a] ERROR: no warm-start weights found in ${WARM_START_WEIGHTS_DIR}" >&2
  exit 1
fi

echo "[phase2-5090][P2a] repo=${REPO_DIR}"
echo "[phase2-5090][P2a] run_prefix=${RUN_PREFIX}"
echo "[phase2-5090][P2a] phase2_suffix=${PHASE2_SUFFIX}"
echo "[phase2-5090][P2a] run_dir=${RUN_DIR_OVERRIDE}"
echo "[phase2-5090][P2a] warm_start=${WARM_START_WEIGHTS_DIR}"
echo "[phase2-5090][P2a] gpu_id=${GPU_ID} epochs=${TRAIN_EPOCHS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
echo "[phase2-5090][P2a] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE}" \
WARM_START_WEIGHTS_DIR_OVERRIDE="${WARM_START_WEIGHTS_DIR}" \
bash scripts/run_phase2_realdata_p2a.sh \
  trainer.max_epochs="${TRAIN_EPOCHS}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  "$@"

echo "[phase2-5090][P2a] completed successfully"
