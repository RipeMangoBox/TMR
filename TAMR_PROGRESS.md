# TAMR Progress

This file now keeps only the current effective Stage4.1 state. Older
in-progress notes were removed after the corrected real-data rerun fully
completed and the invalid artifact families were deleted.

## 2026-04-12 Current Canonical State

### Final Outcome

| Workstream | Status | Evidence / Output | Current Meaning |
|---|---|---|---|
| Corrected real-data formal rerun | Done | `RUN_DIR/stage4_1_realdata_e50_b128_d0` ... `RUN_DIR/stage4_1_realdata_e50_b128_d3` | This is the only retained formal Stage4.1 family |
| Corrected local eval + D3 closure | Done | `RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md` | Winner remains `D2b`; recommendation is `Go Phase 2 with D2b` |
| Unified explanation and registry | Done | `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`, `STAGE4_1_REALDATA_RUNBOOK.md`, `STAGE4_1_REALDATA_RUNDIRS.md` | Use these as the only maintained Stage4.1 docs |
| Cleanup of invalid artifacts | Done | wrong-data, smoke, probe, profiling, and legacy unversioned corrected run_dirs removed | Do not recreate deleted families unless explicitly doing forensics |
| Phase2 P2a server handoff | Ready | `PHASE2_REALDATA_RUNBOOK.md`, `PHASE2_REALDATA_RUNDIRS.md`, `PHASE2_REALDATA_5090_TMUX.md`, `scripts/run_phase2_realdata_5090_p2a.sh` | Local temporary Phase2-generated artifacts were deleted; the next step is to launch `P2a` on server |

### Retained Artifacts

- `RUN_DIR/stage4_1_realdata_e50_b128_d0`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1_5`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2a`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2b`
- `RUN_DIR/stage4_1_realdata_e50_b128_d3`
- `models/tmr_humanml3d_guoh3dfeats`
- `RUN_DIR/phase2_realdata_e50_b128_p2a`

### Corrected Data Facts

- Canonical dataset root:
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- EventT2M compatibility symlink:
  - `/home/ripemangobox/Coding/Github/Motion/EventT2M-codes-main/dataset/HumanML3D-E -> /home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Trusted files:
  - `data_train.npy`
  - `data_val.npy`
  - `data_test.npy`
  - `data_test_condition2.npy`
  - `data_test_condition3.npy`
  - `data_test_condition4.npy`
- Corrected D0 gate:
  - `DATA-GATE GO`
- Split / structure summary:
  - entries: `train=24546`, `val=1530`, `test=4646`
  - captions: `83347`
  - canonical `decomposed` coverage: `100%`
  - `nsim_test` overlap: `97/100`

### Corrected Stage Comparison

| Stage | PrimaryScore | normal t2m/R01 | normal m2t/R01 | nsim t2m/R01 | nsim m2t/R01 | normal t2m/R05 | normal m2t/R05 | nsim t2m/R05 | nsim m2t/R05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 9.78 | 0.73 | 0.34 | 11.34 | 8.25 | 2.91 | 1.05 | 34.02 | 19.59 |
| D1.5 | 9.78 | 0.73 | 0.34 | 11.34 | 8.25 | 2.91 | 1.05 | 34.02 | 19.59 |
| D2a | 32.44 | 3.25 | 3.96 | 39.18 | 40.21 | 13.07 | 13.43 | 75.26 | 71.13 |
| D2b | 37.17 | 4.39 | 6.50 | 48.45 | 45.36 | 16.23 | 17.67 | 79.38 | 79.38 |

### Interpretation

- D1 is the Stage4.1 internal baseline on corrected real data.
- D1.5 is the pooling control and ties D1 under the corrected rerun.
- D2a is the first strong gain and shows partial motion unfreezing matters.
- D2b is the strongest corrected result and remains the formal winner.

### Phase2 Follow-Up

- Current Phase2 family prefix:
  - `phase2_realdata_e50_b128`
- Current approved first experiment:
  - `P2a = D2b warm-start + full text encoder unfreeze`
- Warm-start source:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2b/last_weights`
- Gate rule:
  - compare new Phase2 runs back to corrected `D2b` using `normal + nsim`
- Current local state:
  - local temporary Phase2-generated artifacts were deleted on `2026-04-12`
  - server rerun is still pending

### Read First In Future Sessions

- `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`
- `STAGE4_1_REALDATA_RUNBOOK.md`
- `STAGE4_1_REALDATA_RUNDIRS.md`
- `STAGE4_1_NEXT_SESSION_PROMPT.md`
- `RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md`
- `PHASE2_REALDATA_RUNBOOK.md`
- `PHASE2_REALDATA_RUNDIRS.md`
- `PHASE2_REALDATA_5090_TMUX.md`

### Working Rules Going Forward

- Do not overwrite the retained `stage4_1_realdata_e50_b128_*` family.
- New experiments must use a new explicit `run_prefix`.
- Continue using retrieval-first gating on `normal + nsim`.
- Do not resurrect deleted wrong-data or probe families into active docs.
