#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_PREFIX="${RUN_PREFIX:-stage4_1_realdata_e50_b128}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_ID="${GPU_ID:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-logs/stage4_1_realdata_5090}"

D1_LAST_WEIGHTS_DIR="RUN_DIR/${RUN_PREFIX}_d1/last_weights"
D2B_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d2b"

mkdir -p "${LOG_DIR}" "RUN_DIR"
LOG_FILE="${LOG_DIR}/${RUN_PREFIX}_d2b_gpu${GPU_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

has_weights() {
  [[ -n "$(find "${D1_LAST_WEIGHTS_DIR}" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null)" ]]
}

echo "[stage4.1-5090][D2b] repo=${REPO_DIR}"
echo "[stage4.1-5090][D2b] run_prefix=${RUN_PREFIX}"
echo "[stage4.1-5090][D2b] gpu_id=${GPU_ID} epochs=${TRAIN_EPOCHS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
echo "[stage4.1-5090][D2b] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[stage4.1-5090][D2b] waiting for warm start in ${D1_LAST_WEIGHTS_DIR}"

until has_weights; do
  echo "[stage4.1-5090][D2b] $(date '+%F %T') waiting ${WAIT_SECONDS}s for D1 last_weights..."
  sleep "${WAIT_SECONDS}"
done

echo "[stage4.1-5090][D2b] detected D1 weights, starting training"

RUN_DIR_OVERRIDE="${D2B_RUN_DIR}" \
WARM_START_WEIGHTS_DIR_OVERRIDE="${D1_LAST_WEIGHTS_DIR}" \
bash scripts/run_stage4_1_realdata_d2b.sh \
  trainer.max_epochs="${TRAIN_EPOCHS}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}"

echo "[stage4.1-5090][D2b] completed successfully"
