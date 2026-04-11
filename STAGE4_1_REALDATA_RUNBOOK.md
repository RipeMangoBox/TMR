# Stage4.1 Corrected Real-Data Serial Runbook

This markdown is the execution source of truth for the corrected HumanML3D-E rerun.

If a future session is handed only this file, it should execute the commands below in order instead of redesigning the workflow.

## Scope

- Target repo: `/home/ripemangobox/Coding/Github/Motion/TMR`
- Canonical dataset root: `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- EventT2M symlink: `/home/ripemangobox/Coding/Github/Motion/EventT2M-codes-main/dataset/HumanML3D-E -> /home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Valid corrected run dirs:
  - `RUN_DIR/stage4_1_realdata_d0`
  - `RUN_DIR/stage4_1_realdata_d1`
  - `RUN_DIR/stage4_1_realdata_d1_5`
  - `RUN_DIR/stage4_1_realdata_d2a`
  - `RUN_DIR/stage4_1_realdata_d2b`
  - `RUN_DIR/stage4_1_realdata_d3`

## Rules

- Do not reuse the older wrong-data Stage4.1 outputs for decision-making.
- Run the corrected chain in strict serial order: `D0 -> D1 -> D1 retrieval -> D1.5 -> D1.5 retrieval -> D2a -> D2a retrieval -> D2b -> D2b retrieval -> D3 summary`.
- `D2a` and `D2b` must warm-start from `RUN_DIR/stage4_1_realdata_d1/last_weights`.
- Retrieval-first gate remains `normal + nsim`.
- Use the batch script as the default entrypoint:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/scripts/run_stage4_1_realdata_batch.sh`

## Default Serial Launch

Run this unless the progress table shows that a later corrected stage is already complete:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --start-stage d0 \
  --end-stage d3 \
  --report-date 2026-04-11 \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

## Resume Commands

If the server job stops mid-chain, resume from the first unfinished stage.

Resume from `D1`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --start-stage d1 \
  --end-stage d3 \
  --report-date 2026-04-11 \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Resume from `D1.5`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --start-stage d1_5 \
  --end-stage d3 \
  --report-date 2026-04-11 \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Resume from `D2a`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --start-stage d2a \
  --end-stage d3 \
  --report-date 2026-04-11 \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Resume from `D2b`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --start-stage d2b \
  --end-stage d3 \
  --report-date 2026-04-11 \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Run only `D3` summary after all corrected retrieval metrics already exist:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --start-stage d3 \
  --end-stage d3 \
  --report-date 2026-04-11 \
  --skip-retrieval
```

## Unified Eval And Summary

Use this after checkpoints exist and you want to re-run retrieval and refresh all corrected docs in one pass:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/eval_stage4_1_realdata_all.sh \
  --report-date 2026-04-11 \
  --retrieval-batch-size 256
```

Force retrieval refresh for all corrected run dirs:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/eval_stage4_1_realdata_all.sh \
  --report-date 2026-04-11 \
  --retrieval-batch-size 256 \
  --force-retrieval
```

## Local GPU Probe

Use this only to validate memory / startup behavior on a local GPU without touching the formal corrected run dirs.

Reference outcome on the local RTX 3090:

- `D1`, `D1.5`, `D2a`, and `D2b` all completed a probe with:
  - `batch_size=64`
  - `num_workers=4`
  - `max_epochs=1`
  - `limit_train_batches=20`
  - `limit_val_batches=1`
- No OOM was observed.
- A full corrected epoch at `batch_size=64` is still estimated at roughly `26-28 minutes` on that local machine, so formal reruns should stay on server.

Example `D1` local probe:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_d1.sh \
  run_dir=RUN_DIR/stage4_1_realdata_d1_local_probe \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=20 \
  +trainer.limit_val_batches=1 \
  dataloader.batch_size=64 \
  dataloader.num_workers=4 \
  trainer.log_every_n_steps=5
```

## Manual Checkpoints

Check whether D1 warm-start weights already exist:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

find RUN_DIR/stage4_1_realdata_d1/last_weights -maxdepth 1 -type f -name '*.pt'
```

If `last_weights` are missing but D1 `last.ckpt` exists, extract them manually:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

conda run -n TMR python - <<'PY'
from src.load import extract_ckpt
extract_ckpt("RUN_DIR/stage4_1_realdata_d1")
PY
```

## Expected Outputs

- D0 report:
  - `RUN_DIR/stage4_1_realdata_d0/2026-04-11_d0_realdata_report.md`
- Per-stage corrected reports:
  - `RUN_DIR/stage4_1_realdata_d1/2026-04-11_d1_realdata_report.md`
  - `RUN_DIR/stage4_1_realdata_d1_5/2026-04-11_d1_5_realdata_report.md`
  - `RUN_DIR/stage4_1_realdata_d2a/2026-04-11_d2a_realdata_report.md`
  - `RUN_DIR/stage4_1_realdata_d2b/2026-04-11_d2b_realdata_report.md`
- D3 closure:
  - `RUN_DIR/stage4_1_realdata_d3/2026-04-11_d3_stage4_1_closure_summary.md`

## Read Before Running

- Current progress table:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/TAMR_PROGRESS.md`
- Loader and corrected data assumptions:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/src/data/humanml3de_event.py`
- D0 corrected data audit:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_d0/2026-04-11_d0_realdata_report.md`
