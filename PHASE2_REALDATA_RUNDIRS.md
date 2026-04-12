# Phase2 Real-Data RunDir Registry

This registry tracks post-D3 follow-up experiments that start from the
corrected Stage4.1 winner and live outside the retained Stage4.1 family.

## Warm-Start Source

- `RUN_DIR/stage4_1_realdata_e50_b128_d2b/last_weights`

## Current Phase2 Family

- family prefix: `phase2_realdata_e50_b128`
- tracked run dirs:
  - `RUN_DIR/phase2_realdata_e50_b128_p2a`
- current status:
  - local temporary Phase2-generated artifacts were deleted before server migration
  - server rerun remains pending
  - retrieval gate remains pending until a completed rerun generates retrieval metrics

## P2a Meaning

- start from corrected `D2b`
- keep the D2b retrieval recipe
- unfreeze the text encoder in addition to the full motion encoder
- keep the motion decoder frozen

## Guardrails

- Do not overwrite `RUN_DIR/stage4_1_realdata_e50_b128_*`.
- Do not recreate deleted wrong-data, smoke, probe, or profiling families.
- Compare new Phase2 runs back to corrected `D2b` with `normal + nsim`.
