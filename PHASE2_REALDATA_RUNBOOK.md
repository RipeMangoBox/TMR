# Phase2 Real-Data Runbook

This file tracks post-D3 follow-up experiments after the corrected Stage4.1
winner was fixed as `D2b`.

Stage4.1 source of truth remains:

- `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`
- `RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md`

Phase 2 must not overwrite the retained `stage4_1_realdata_e50_b128_*` family.

## Current Phase2 Family

- family prefix: `phase2_realdata_e50_b128`
- current experiment:
  - `RUN_DIR/phase2_realdata_e50_b128_p2a`

## P2a Definition

- initialization:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2b/last_weights`
- trainable parts:
  - full motion encoder
  - full text encoder
  - event projection head
- frozen parts:
  - motion decoder
- objective:
  - same retrieval recipe as D2b
  - masked event InfoNCE + weak global contrastive loss
- retrieval gate:
  - compare against corrected `D2b` using `normal + nsim`
  - do not decide from `evt_align_acc` alone

## Launch

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_phase2_realdata_p2a.sh
```

Explicit launch with the current formal settings:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_phase2_realdata_p2a.sh \
  trainer.max_epochs=50 \
  dataloader.batch_size=128 \
  dataloader.num_workers=8
```

## Eval And Report

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

RUN_DIR=RUN_DIR/phase2_realdata_e50_b128_p2a \
BASE_DIR=RUN_DIR/stage4_1_realdata_e50_b128_d2b \
REPORT_DATE=2026-04-12 \
bash scripts/eval_phase2_realdata.sh
```

This eval script:

- runs `retrieval.py` if the four protocol files are missing
- writes a retrieval-first comparison report into:
  - `RUN_DIR/phase2_realdata_e50_b128_p2a/2026-04-12_phase2_realdata_report.md`

## Server Launch

- tmux helper doc:
  - `PHASE2_REALDATA_5090_TMUX.md`
- server helper script:
  - `scripts/run_phase2_realdata_5090_p2a.sh`

Example:

```bash
cd /data/public/ripemangobox/Motion/TMR

tmux new-session -d -s p2_p2a \
  'cd /data/public/ripemangobox/Motion/TMR && bash scripts/run_phase2_realdata_5090_p2a.sh'
```

## Local State

- The temporary local Phase2-generated run_dir was intentionally deleted before
  server migration.
- Recommended server behavior:
  - start directly on server with the same default family if no conflicting
    remote run exists
  - or provide a fresh explicit `RUN_PREFIX` if you want a clean new server-only
    family

## Rules

- Do not overwrite the retained `stage4_1_realdata_e50_b128_*` family.
- If a second Phase2 branch is added, keep the family prefix and add a new
  suffix such as `_p2b`.
- Keep using `normal + nsim` as the primary decision surface.
