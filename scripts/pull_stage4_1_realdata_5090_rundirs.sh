#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

REMOTE_HOST="${REMOTE_HOST:-5090}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/data/public/ripemangobox/Motion/TMR}"
LOCAL_RUN_DIR_ROOT="${LOCAL_RUN_DIR_ROOT:-${REPO_DIR}/RUN_DIR}"
RUN_PREFIX="${RUN_PREFIX:-stage4_1_realdata_e50_b128}"
INCLUDE_D0="${INCLUDE_D0:-1}"
INCLUDE_D3="${INCLUDE_D3:-0}"
INCLUDE_LOGS="${INCLUDE_LOGS:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/pull_stage4_1_realdata_5090_rundirs.sh [options]

Options:
  --remote-host <host>
  --remote-repo-root <path>
  --local-run-dir-root <path>
  --run-prefix <prefix>
  --skip-d0
  --include-d3
  --include-logs
  -h, --help

Defaults:
  remote host: 5090
  remote repo root: /data/public/ripemangobox/Motion/TMR
  local run dir root: <repo>/RUN_DIR
  run prefix: stage4_1_realdata_e50_b128

This script pulls the entire corrected remote run directories back to the
local machine so local retrieval eval and D3 summary can be run safely.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-repo-root)
      REMOTE_REPO_ROOT="$2"
      shift 2
      ;;
    --local-run-dir-root)
      LOCAL_RUN_DIR_ROOT="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --skip-d0)
      INCLUDE_D0=0
      shift
      ;;
    --include-d3)
      INCLUDE_D3=1
      shift
      ;;
    --include-logs)
      INCLUDE_LOGS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOCAL_RUN_DIR_ROOT}"

RUN_DIRS=()
if [[ "${INCLUDE_D0}" == "1" ]]; then
  RUN_DIRS+=("${RUN_PREFIX}_d0")
fi
RUN_DIRS+=(
  "${RUN_PREFIX}_d1"
  "${RUN_PREFIX}_d1_5"
  "${RUN_PREFIX}_d2a"
  "${RUN_PREFIX}_d2b"
)
if [[ "${INCLUDE_D3}" == "1" ]]; then
  RUN_DIRS+=("${RUN_PREFIX}_d3")
fi

echo "[pull] remote_host=${REMOTE_HOST}"
echo "[pull] remote_repo_root=${REMOTE_REPO_ROOT}"
echo "[pull] local_run_dir_root=${LOCAL_RUN_DIR_ROOT}"
echo "[pull] run_prefix=${RUN_PREFIX}"

for run_name in "${RUN_DIRS[@]}"; do
  remote_dir="${REMOTE_REPO_ROOT}/RUN_DIR/${run_name}"
  local_dir="${LOCAL_RUN_DIR_ROOT}/${run_name}"
  echo
  echo "[pull] checking ${REMOTE_HOST}:${remote_dir}"
  if ! ssh "${REMOTE_HOST}" "test -d '${remote_dir}'"; then
    echo "[pull] skip missing remote dir: ${remote_dir}"
    continue
  fi

  mkdir -p "${local_dir}"
  echo "[pull] rsync ${remote_dir} -> ${local_dir}"
  rsync -avP \
    "${REMOTE_HOST}:${remote_dir}/" \
    "${local_dir}/"
done

if [[ "${INCLUDE_LOGS}" == "1" ]]; then
  remote_logs="${REMOTE_REPO_ROOT}/logs/stage4_1_realdata_5090"
  local_logs="${REPO_DIR}/logs/stage4_1_realdata_5090"
  echo
  echo "[pull] checking ${REMOTE_HOST}:${remote_logs}"
  if ssh "${REMOTE_HOST}" "test -d '${remote_logs}'"; then
    mkdir -p "${local_logs}"
    echo "[pull] rsync ${remote_logs} -> ${local_logs}"
    rsync -avP \
      "${REMOTE_HOST}:${remote_logs}/" \
      "${local_logs}/"
  else
    echo "[pull] skip missing remote logs dir: ${remote_logs}"
  fi
fi

echo
echo "[pull] completed"
