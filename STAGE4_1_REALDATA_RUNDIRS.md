# Stage4.1 Corrected Real-Data RunDir Registry

This registry reflects the cleaned post-eval state.

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

## Deleted During Cleanup

These categories were intentionally removed because they were invalid or fully
superseded:

- wrong-data Stage4.1 run_dirs
- smoke run_dirs
- local probe run_dirs
- throughput attempt run_dirs
- batch-size sweep run_dirs
- legacy unversioned corrected run_dirs from the intermediate rerun pass

## Rule Of Thumb

- If you need the corrected final Stage4.1 evidence, use only the canonical
  final family in the first section.
- If you launch another formal rerun, do not overwrite this family; append a
  new suffix or date explicitly.
- If you need the consolidated explanation of what each stage means, read:
  - `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`
- If you need post-D3 follow-up experiments, read:
  - `PHASE2_REALDATA_RUNDIRS.md`
