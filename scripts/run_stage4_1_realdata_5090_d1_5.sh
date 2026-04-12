#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_PREFIX="${RUN_PREFIX:-stage4_1_realdata_e50_b128}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_ID="1"
LOG_DIR="${LOG_DIR:-logs/stage4_1_realdata_5090}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi

if ! "${CONDA_BIN}" --version >/dev/null 2>&1; then
  echo "[stage4.1-5090][D1.5] ERROR: unable to resolve conda executable (${CONDA_BIN})" >&2
  exit 1
fi

D1_5_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d1_5"

mkdir -p "${LOG_DIR}" "RUN_DIR"
LOG_FILE="${LOG_DIR}/${RUN_PREFIX}_d1_5_gpu${GPU_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[stage4.1-5090][D1.5] repo=${REPO_DIR}"
echo "[stage4.1-5090][D1.5] run_prefix=${RUN_PREFIX}"
echo "[stage4.1-5090][D1.5] gpu_id=${GPU_ID} epochs=${TRAIN_EPOCHS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
echo "[stage4.1-5090][D1.5] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[stage4.1-5090][D1.5] conda=${CONDA_BIN}"
echo "[stage4.1-5090][D1.5] run_dir=${D1_5_RUN_DIR}"

RUN_DIR_OVERRIDE="${D1_5_RUN_DIR}" \
bash scripts/run_stage4_1_realdata_d1_5.sh \
  trainer.max_epochs="${TRAIN_EPOCHS}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}"

echo "[stage4.1-5090][D1.5] completed successfully"
