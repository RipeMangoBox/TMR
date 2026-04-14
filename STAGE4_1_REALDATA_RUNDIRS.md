# Stage4.1 Corrected Real-Data RunDir Registry

This registry reflects the cleaned post-eval state and the first retained
post-D3 follow-up.

## Canonical Final Family

The only retained formal corrected Stage4.1 family is:

- `RUN_DIR/stage4_1_realdata_e50_b128_d0`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1`
- `RUN_DIR/stage4_1_realdata_e50_b128_d1_5`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2a`
- `RUN_DIR/stage4_1_realdata_e50_b128_d2b`
- `RUN_DIR/stage4_1_realdata_e50_b128_d3`

Naming basis:

- corrected real data
- `epochs=50`
- `batch_size=128`
- `num_workers=8`

## Warm-Start Reference

The D1 warm-start reference remains:

- `models/tmr_humanml3d_guoh3dfeats/last_weights`

The D2a / D2b warm-start source inside the formal corrected family is:

- `RUN_DIR/stage4_1_realdata_e50_b128_d1/last_weights`

## Post-D3 Follow-Up

The first retained follow-up run outside the formal Stage4.1 family is:

- `RUN_DIR/phase2_realdata_e50_b128_p2a`

Meaning:

- start from corrected `D2b`
- keep the D2b retrieval recipe
- unfreeze the full text encoder in addition to the full motion encoder
- keep the motion decoder frozen

Status:

- evaluated
- `PrimaryScore = 36.34`
- `Delta vs D2b = -0.83`
- current overall winner remains `D2b`

## Deleted During Cleanup

These categories were intentionally removed because they were invalid or fully
superseded:

- wrong-data Stage4.1 run_dirs
- smoke run_dirs
- local probe run_dirs
- throughput attempt run_dirs
- batch-size sweep run_dirs
- legacy unversioned corrected run_dirs from the intermediate rerun pass
- standalone Phase2 markdown trackers after `P2a` consolidation

## Rule Of Thumb

- If you need the corrected final Stage4.1 evidence, use only the canonical
  final family in the first section.
- If you launch another formal rerun, do not overwrite this family; append a
  new suffix or date explicitly.
- If you launch another post-D3 follow-up, do not overwrite `P2a`; append a new
  suffix explicitly.
- If you need the consolidated explanation of what each stage means and how
  `P2a` compared, read:
  - `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`
