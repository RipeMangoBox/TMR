#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

SCHEMA="${SCHEMA_OVERRIDE:-}"
if [[ -z "${SCHEMA}" ]]; then
  echo "[motion repr ablation] ERROR: SCHEMA_OVERRIDE is not set" >&2
  exit 1
fi

DATA_ROOT_OVERRIDE="${DATA_ROOT_OVERRIDE:-${MOTIONPATCHES_DATA_ROOT:-${HOME}/Coding/Github/Motion/datasets/HumanML3D-E-MP}}"
FORMAT_ROOT_OVERRIDE="${FORMAT_ROOT_OVERRIDE:-${DATA_ROOT_OVERRIDE}/motion_formats}"
STATS_ROOT_OVERRIDE="${STATS_ROOT_OVERRIDE:-${DATA_ROOT_OVERRIDE}/motion_format_stats}"
TEXT_DIR_OVERRIDE="${TEXT_DIR_OVERRIDE:-${DATA_ROOT_OVERRIDE}/texts}"
SPLIT_DIR_OVERRIDE="${SPLIT_DIR_OVERRIDE:-${DATA_ROOT_OVERRIDE}}"
RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-RUN_DIR/motion_repr_ablation_${SCHEMA}}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-cuda}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-50}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-32}"
NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-8}"
SEED_OVERRIDE="${SEED_OVERRIDE:-42}"
LR_OVERRIDE="${LR_OVERRIDE:-1e-4}"
WEIGHT_DECAY_OVERRIDE="${WEIGHT_DECAY_OVERRIDE:-1e-5}"
PATIENCE_OVERRIDE="${PATIENCE_OVERRIDE:-10}"
MAX_MOTION_LENGTH_OVERRIDE="${MAX_MOTION_LENGTH_OVERRIDE:-224}"
MAX_TEXT_LENGTH_OVERRIDE="${MAX_TEXT_LENGTH_OVERRIDE:-64}"
TEXT_ENCODE_BATCH_SIZE_OVERRIDE="${TEXT_ENCODE_BATCH_SIZE_OVERRIDE:-64}"
LATENT_DIM_OVERRIDE="${LATENT_DIM_OVERRIDE:-256}"
NUM_LAYERS_OVERRIDE="${NUM_LAYERS_OVERRIDE:-2}"
NUM_HEADS_OVERRIDE="${NUM_HEADS_OVERRIDE:-4}"
FF_SIZE_OVERRIDE="${FF_SIZE_OVERRIDE:-512}"
DROPOUT_OVERRIDE="${DROPOUT_OVERRIDE:-0.1}"
TEMPERATURE_OVERRIDE="${TEMPERATURE_OVERRIDE:-0.07}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
  echo "[motion repr ablation][${SCHEMA}] ERROR: conda executable not found" >&2
  exit 1
fi

echo "[motion repr ablation][${SCHEMA}] repo: ${REPO_DIR}"
echo "[motion repr ablation][${SCHEMA}] data_root: ${DATA_ROOT_OVERRIDE}"
echo "[motion repr ablation][${SCHEMA}] format_root: ${FORMAT_ROOT_OVERRIDE}"
echo "[motion repr ablation][${SCHEMA}] stats_root: ${STATS_ROOT_OVERRIDE}"
echo "[motion repr ablation][${SCHEMA}] run_dir: ${RUN_DIR_OVERRIDE}"
if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]]; then
  echo "[motion repr ablation][${SCHEMA}] cuda_visible_devices: ${CUDA_VISIBLE_DEVICES_OVERRIDE}"
fi
echo "[motion repr ablation][${SCHEMA}] device: ${DEVICE_OVERRIDE}"
echo "[motion repr ablation][${SCHEMA}] epochs: ${EPOCHS_OVERRIDE}"
echo "[motion repr ablation][${SCHEMA}] batch_size: ${BATCH_SIZE_OVERRIDE}"

CMD_PREFIX=()
if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]]; then
  CMD_PREFIX=(env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_OVERRIDE}")
fi

"${CMD_PREFIX[@]}" "${CONDA_BIN}" run --live-stream -n TMR python scripts/run_motion_repr_ablation.py \
  --schemas "${SCHEMA}" \
  --format-root "${FORMAT_ROOT_OVERRIDE}" \
  --stats-root "${STATS_ROOT_OVERRIDE}" \
  --text-dir "${TEXT_DIR_OVERRIDE}" \
  --split-dir "${SPLIT_DIR_OVERRIDE}" \
  --output-dir "${RUN_DIR_OVERRIDE}" \
  --epochs "${EPOCHS_OVERRIDE}" \
  --batch-size "${BATCH_SIZE_OVERRIDE}" \
  --lr "${LR_OVERRIDE}" \
  --weight-decay "${WEIGHT_DECAY_OVERRIDE}" \
  --device "${DEVICE_OVERRIDE}" \
  --seed "${SEED_OVERRIDE}" \
  --patience "${PATIENCE_OVERRIDE}" \
  --num-workers "${NUM_WORKERS_OVERRIDE}" \
  --max-motion-length "${MAX_MOTION_LENGTH_OVERRIDE}" \
  --max-text-length "${MAX_TEXT_LENGTH_OVERRIDE}" \
  --text-encode-batch-size "${TEXT_ENCODE_BATCH_SIZE_OVERRIDE}" \
  --latent-dim "${LATENT_DIM_OVERRIDE}" \
  --num-layers "${NUM_LAYERS_OVERRIDE}" \
  --num-heads "${NUM_HEADS_OVERRIDE}" \
  --ff-size "${FF_SIZE_OVERRIDE}" \
  --dropout "${DROPOUT_OVERRIDE}" \
  --temperature "${TEMPERATURE_OVERRIDE}" \
  "$@"
