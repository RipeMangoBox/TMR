#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

echo "[D2a realdata] repo: ${REPO_DIR}"
echo "[D2a realdata] run_dir: RUN_DIR/stage4_1_realdata_d2a"

conda run -n TMR python train.py \
  model=tmr_d2a \
  data=humanml3d_e \
  run_dir=RUN_DIR/stage4_1_realdata_d2a \
  model.warm_start_weights_dir=RUN_DIR/stage4_1_realdata_d1/last_weights \
  dataloader.batch_size=32 \
  dataloader.num_workers=0 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=100 \
  trainer.log_every_n_steps=20 \
  "$@"
