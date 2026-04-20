#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

GPU_ID="${GPU_ID_OVERRIDE:-1}"
STAGE="${STAGE_OVERRIDE:-all}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-outputs/humanml3d_e_mp_motion_repr_server}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-128}"
NUM_WORKERS="${NUM_WORKERS_OVERRIDE:-8}"
SEED="${SEED_OVERRIDE:-1234}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE_OVERRIDE:-256}"
LOG_DIR="${LOG_DIR_OVERRIDE:-${RUN_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

if [[ -n "${SCHEMAS_OVERRIDE:-}" ]]; then
  read -r -a SCHEMAS <<< "${SCHEMAS_OVERRIDE}"
else
  SCHEMAS=(smpl135 hy201 hml272)
fi

DRY_RUN_FLAG=()
if [[ "${DRY_RUN_OVERRIDE:-0}" == "1" ]]; then
  DRY_RUN_FLAG=(--dry-run)
fi

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
  echo "[tmr-hml3de-mp][gpu${GPU_ID}] ERROR: conda executable not found" >&2
  exit 1
fi

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG_PATH="${LOG_DIR}/gpu${GPU_ID}_${STAGE}_${TIMESTAMP}.log"

echo "[tmr-hml3de-mp][gpu${GPU_ID}] repo=${REPO_DIR}"
echo "[tmr-hml3de-mp][gpu${GPU_ID}] stage=${STAGE}"
echo "[tmr-hml3de-mp][gpu${GPU_ID}] schemas=${SCHEMAS[*]}"
echo "[tmr-hml3de-mp][gpu${GPU_ID}] run_root=${RUN_ROOT}"
echo "[tmr-hml3de-mp][gpu${GPU_ID}] log=${LOG_PATH}"

CMD=(
  env
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "${CONDA_BIN}"
  run
  --live-stream
  -n
  TMR
  python
  scripts/run_tmr_humanml3de_mp_motion_repr.py
  --stage "${STAGE}" \
  --schemas "${SCHEMAS[@]}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device cuda \
  --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}" \
  --run-root "${RUN_ROOT}" \
  "${DRY_RUN_FLAG[@]}" \
  "$@" \
)

printf -v CMD_STR '%q ' "${CMD[@]}"

if command -v script >/dev/null 2>&1 && [[ "${DRY_RUN_OVERRIDE:-0}" != "1" ]]; then
  script -qefc "${CMD_STR}" "${LOG_PATH}"
else
  "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
fi
