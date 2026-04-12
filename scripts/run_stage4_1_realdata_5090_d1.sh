#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_PREFIX="${RUN_PREFIX:-stage4_1_realdata_e50_b128}"
REPORT_DATE="${REPORT_DATE:-$(date +%F)}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_ID="0"
LOG_DIR="${LOG_DIR:-logs/stage4_1_realdata_5090}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi

if ! "${CONDA_BIN}" --version >/dev/null 2>&1; then
  echo "[stage4.1-5090][D1] ERROR: unable to resolve conda executable (${CONDA_BIN})" >&2
  exit 1
fi

D0_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d0"
D1_RUN_DIR="RUN_DIR/${RUN_PREFIX}_d1"
D1_LAST_WEIGHTS_DIR="${D1_RUN_DIR}/last_weights"

mkdir -p "${LOG_DIR}" "RUN_DIR"
LOG_FILE="${LOG_DIR}/${RUN_PREFIX}_d1_gpu${GPU_ID}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[stage4.1-5090][D1] repo=${REPO_DIR}"
echo "[stage4.1-5090][D1] run_prefix=${RUN_PREFIX}"
echo "[stage4.1-5090][D1] report_date=${REPORT_DATE}"
echo "[stage4.1-5090][D1] gpu_id=${GPU_ID} epochs=${TRAIN_EPOCHS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}"
echo "[stage4.1-5090][D1] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[stage4.1-5090][D1] conda=${CONDA_BIN}"
echo "[stage4.1-5090][D1] d0_run_dir=${D0_RUN_DIR}"
echo "[stage4.1-5090][D1] d1_run_dir=${D1_RUN_DIR}"

"${CONDA_BIN}" run -n TMR python scripts/d0_humanml3de_event_stats.py \
  --report-date "${REPORT_DATE}" \
  --output-dir "${D0_RUN_DIR}"

RUN_DIR_OVERRIDE="${D1_RUN_DIR}" \
bash scripts/run_stage4_1_realdata_d1.sh \
  trainer.max_epochs="${TRAIN_EPOCHS}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}"

"${CONDA_BIN}" run -n TMR python -c "from src.load import extract_ckpt; extract_ckpt('${D1_RUN_DIR}')"

if [[ -z "$(find "${D1_LAST_WEIGHTS_DIR}" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null)" ]]; then
  echo "[stage4.1-5090][D1] ERROR: no extracted weights found in ${D1_LAST_WEIGHTS_DIR}" >&2
  exit 1
fi

echo "[stage4.1-5090][D1] completed successfully"
