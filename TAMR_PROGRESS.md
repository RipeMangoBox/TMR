# TAMR Progress

This file now keeps only the current consolidated state. Detailed metrics and
the `P2a` comparison live in `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`.

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
