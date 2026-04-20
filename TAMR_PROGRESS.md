# TAMR Progress

This file now keeps only the current consolidated state. Detailed metrics and
the `P2a` comparison live in `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`.

## 2026-04-20 HumanML3D-E-MP Launcher Update

### Default Change

- `scripts/run_tmr_humanml3de_mp_gpu0.sh` now defaults to:
  - `MODEL_NAME=tmr_d2b_retrieval_first`
  - `BATCH_SIZE=32`
  - `SEED=1234`
  - `SCHEMAS=(guo263)`
- `scripts/run_tmr_humanml3de_mp_gpu1.sh` now defaults to:
  - `MODEL_NAME=tmr_d2b_retrieval_first`
  - `BATCH_SIZE=32`
  - `SEED=1234`
- `scripts/run_tmr_humanml3de_mp_motion_repr.py` is aligned with the same
  retrieval-first defaults.

### Why

- The old launcher default (`tmr_d2b` + `batch_size=128`) was not comparable to
  the original `tmr_humanml3d_guoh3dfeats` baseline and could drive severe
  global-retrieval collapse on HumanML3D-E-MP.
- The new `tmr_d2b_retrieval_first` config keeps the D2b full-motion setup but
  changes the default loss balance to:
  - `global: 1.0`
  - `evt_align: 0.25`
- The launcher now matches the baseline batch size and seed by default so
  `guo263` comparisons are fairer.

### How To Recover The Old Behavior

- Override the model explicitly, for example:
  - `MODEL_NAME_OVERRIDE=tmr_d2b`
- Override batch or schemas when needed, for example:
  - `BATCH_SIZE_OVERRIDE=128`
  - `SCHEMAS_OVERRIDE="guo263 pos66 kimodo261"`

## 2026-04-12 Current Canonical State

### Final Outcome

| Workstream | Status | Evidence / Output | Current Meaning |
|---|---|---|---|
| Corrected real-data formal rerun | Done | `RUN_DIR/stage4_1_realdata_e50_b128_d0` ... `RUN_DIR/stage4_1_realdata_e50_b128_d3` | This is the only retained formal Stage4.1 family |
| Corrected local eval + D3 closure | Done | `RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md` | Historical Stage4.1 winner is `D2b`; D3 recommended `Go Phase 2 with D2b` |
| Phase2 follow-up `P2a` | Done | `RUN_DIR/phase2_realdata_e50_b128_p2a` | `PrimaryScore = 36.34`, which is `-0.83` vs `D2b`; do not replace `D2b` |
| Consolidated docs | Done | `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`, `STAGE4_1_REALDATA_RUNBOOK.md`, `STAGE4_1_REALDATA_RUNDIRS.md`, `STAGE4_1_NEXT_SESSION_PROMPT.md` | Use these as the maintained docs; standalone Phase2 markdown trackers were removed |

### Retained Artifacts

- `RUN_DIR/stage4_1_realdata_e50_b128_d0`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1_5`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2a`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2b`
- `RUN_DIR/stage4_1_realdata_e50_b128_d3`
- `RUN_DIR/phase2_realdata_e50_b128_p2a`
- `models/tmr_humanml3d_guoh3dfeats`

### Read First In Future Sessions

- `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`
- `STAGE4_1_REALDATA_RUNBOOK.md`
- `STAGE4_1_REALDATA_RUNDIRS.md`
- `STAGE4_1_NEXT_SESSION_PROMPT.md`
- `RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md`

### Working Rules Going Forward

- Do not overwrite the retained `stage4_1_realdata_e50_b128_*` family.
- Do not overwrite `RUN_DIR/phase2_realdata_e50_b128_p2a`; any new follow-up
  must use a fresh explicit suffix.
- Continue using retrieval-first gating on `normal + nsim`.
- Do not resurrect deleted wrong-data, smoke, probe, or profiling families into
  active docs.
