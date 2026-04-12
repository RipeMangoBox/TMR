#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

RUN_DIR="${RUN_DIR:-RUN_DIR/phase2_realdata_e50_b128_p2a}"
BASE_DIR="${BASE_DIR:-RUN_DIR/stage4_1_realdata_e50_b128_d2b}"
REPORT_DATE="${REPORT_DATE:-$(date +%F)}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-256}"
FORCE_RETRIEVAL="${FORCE_RETRIEVAL:-0}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
else
  echo "[phase2 realdata][eval] ERROR: conda executable not found" >&2
  exit 1
fi

has_retrieval_metrics() {
  local run_dir="$1"
  for name in normal.yaml threshold_0.95.yaml nsim.yaml guo.yaml; do
    if [[ ! -f "${run_dir}/contrastive_metrics/${name}" ]]; then
      return 1
    fi
  done
  return 0
}

if [[ ! -f "${RUN_DIR}/config.json" ]]; then
  echo "[phase2 realdata][eval] ERROR: missing ${RUN_DIR}/config.json" >&2
  exit 1
fi

if [[ "${FORCE_RETRIEVAL}" == "1" ]] || ! has_retrieval_metrics "${RUN_DIR}"; then
  echo "[phase2 realdata][eval] running retrieval for ${RUN_DIR}"
  "${CONDA_BIN}" run --live-stream -n TMR python retrieval.py \
    run_dir="${RUN_DIR}" \
    protocol=all \
    batch_size="${RETRIEVAL_BATCH_SIZE}"
else
  echo "[phase2 realdata][eval] retrieval metrics already exist for ${RUN_DIR}, skip."
fi

"${CONDA_BIN}" run --live-stream -n TMR python scripts/report_phase2_realdata.py \
  --report-date "${REPORT_DATE}" \
  --run-dir "${RUN_DIR}" \
  --base-dir "${BASE_DIR}"
