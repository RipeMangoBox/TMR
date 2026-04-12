#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-RUN_DIR/stage4_1_realdata_e50_b128_d1}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
  echo "[D1 realdata] ERROR: conda executable not found" >&2
  exit 1
fi

echo "[D1 realdata] repo: ${REPO_DIR}"
echo "[D1 realdata] run_dir: ${RUN_DIR_OVERRIDE}"

"${CONDA_BIN}" run --live-stream -n TMR python train.py \
  model=tmr_d1 \
  data=humanml3d_e \
  run_dir="${RUN_DIR_OVERRIDE}" \
  dataloader.batch_size=128 \
  dataloader.num_workers=8 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=50 \
  trainer.log_every_n_steps=20 \
  "$@"
