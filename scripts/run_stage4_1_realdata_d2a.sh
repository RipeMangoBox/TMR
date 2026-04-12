#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-RUN_DIR/stage4_1_realdata_d2a}"
WARM_START_WEIGHTS_DIR_OVERRIDE="${WARM_START_WEIGHTS_DIR_OVERRIDE:-RUN_DIR/stage4_1_realdata_d1/last_weights}"
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi

if ! "${CONDA_BIN}" --version >/dev/null 2>&1; then
  echo "[D2a realdata] ERROR: unable to resolve conda executable (${CONDA_BIN})" >&2
  exit 1
fi

echo "[D2a realdata] repo: ${REPO_DIR}"
echo "[D2a realdata] run_dir: ${RUN_DIR_OVERRIDE}"
echo "[D2a realdata] warm_start: ${WARM_START_WEIGHTS_DIR_OVERRIDE}"
echo "[D2a realdata] conda: ${CONDA_BIN}"

"${CONDA_BIN}" run --live-stream -n TMR python train.py \
  model=tmr_d2a \
  data=humanml3d_e \
  run_dir="${RUN_DIR_OVERRIDE}" \
  model.warm_start_weights_dir="${WARM_START_WEIGHTS_DIR_OVERRIDE}" \
  dataloader.batch_size=128 \
  dataloader.num_workers=8 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=50 \
  trainer.log_every_n_steps=20 \
  "$@"
