# Stage4.1 Corrected Real-Data Serial Runbook

This markdown is the execution source of truth for the retained corrected
HumanML3D-E rerun family.

If a future session is handed only this file, it should use the canonical
`stage4_1_realdata_e50_b128` family instead of recreating deleted legacy
artifacts.

RunDir registry:
- `STAGE4_1_REALDATA_RUNDIRS.md`

Unified explanation:
- `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md`

Phase 2 follow-up:
- `PHASE2_REALDATA_RUNBOOK.md`

## Scope

- Target repo: `/home/ripemangobox/Coding/Github/Motion/TMR`
- Canonical dataset root: `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- EventT2M symlink: `/home/ripemangobox/Coding/Github/Motion/EventT2M-codes-main/dataset/HumanML3D-E -> /home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Canonical final corrected run dirs:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d0`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d1`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d1_5`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2a`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2b`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d3`

## Rules

- Do not reuse the older wrong-data Stage4.1 outputs for decision-making.
- Run the corrected chain in strict serial order: `D0 -> D1 -> D1 retrieval -> D1.5 -> D1.5 retrieval -> D2a -> D2a retrieval -> D2b -> D2b retrieval -> D3 summary`.
- `D2a` and `D2b` must warm-start from `RUN_DIR/stage4_1_realdata_e50_b128_d1/last_weights`.
- Retrieval-first gate remains `normal + nsim`.
- Use the batch script as the default entrypoint:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/scripts/run_stage4_1_realdata_batch.sh`
- If future work continues from corrected `D2b`, keep this retained family
  frozen and launch Phase2 under:
  - `phase2_realdata_e50_b128_*`

## Default Serial Launch

Use this as the canonical serial launch template for future reruns with a fresh
suffix if needed:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --run-prefix stage4_1_realdata_e50_b128 \
  --start-stage d0 \
  --end-stage d3 \
  --report-date 2026-04-12 \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

## Resume Commands

If the server job stops mid-chain, resume from the first unfinished stage.

Resume from `D1`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --run-prefix stage4_1_realdata_e50_b128_rerun2 \
  --start-stage d1 \
  --end-stage d3 \
  --report-date 2026-04-12 \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Resume from `D1.5`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --run-prefix stage4_1_realdata_e50_b128_rerun2 \
  --start-stage d1_5 \
  --end-stage d3 \
  --report-date 2026-04-12 \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Resume from `D2a`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --run-prefix stage4_1_realdata_e50_b128_rerun2 \
  --start-stage d2a \
  --end-stage d3 \
  --report-date 2026-04-12 \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Resume from `D2b`:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --run-prefix stage4_1_realdata_e50_b128_rerun2 \
  --start-stage d2b \
  --end-stage d3 \
  --report-date 2026-04-12 \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 8 \
  --retrieval-batch-size 256
```

Run only `D3` summary after all corrected retrieval metrics already exist:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_batch.sh \
  --run-prefix stage4_1_realdata_e50_b128_rerun2 \
  --start-stage d3 \
  --end-stage d3 \
  --report-date 2026-04-12 \
  --skip-retrieval
```

## Unified Eval And Summary

Use this after checkpoints exist and you want to re-run retrieval and refresh all corrected docs in one pass:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/eval_stage4_1_realdata_all.sh \
  --run-prefix stage4_1_realdata_e50_b128 \
  --report-date 2026-04-12 \
  --retrieval-batch-size 256
```

Force retrieval refresh for all corrected run dirs:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/eval_stage4_1_realdata_all.sh \
  --run-prefix stage4_1_realdata_e50_b128 \
  --report-date 2026-04-12 \
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
- The old probe run_dirs were deleted during cleanup; the numbers above are
  retained as historical evidence only.

## Batch-Size Sweep Result

Representative stage:

- `D2b`

Sweep setting:

- `num_workers=8`
- `max_epochs=1`
- `limit_train_batches=20`
- `limit_val_batches=0`
- warm-start source at the time of profiling: a temporary local D1 probe
  checkpoint that was deleted during cleanup
- the profiling run_dirs themselves were removed during cleanup; only the
  summarized numbers were retained

Sweep outcome:

- `bs=64`: estimated epoch `19.496 min`, avg GPU util `40.02%`, peak GPU util `88%`, peak mem `8400 MiB`
- `bs=128`: estimated epoch `18.218 min`, avg GPU util `40.81%`, peak GPU util `92%`, peak mem `10640 MiB`
- `bs=192`: estimated epoch `18.537 min`, avg GPU util `33.19%`, peak GPU util `100%`, peak mem `13590 MiB`

Recommended corrected launch point:

- `batch_size=128`
- `num_workers=8`

Reason:

- `64 -> 128` still improves throughput.
- `128 -> 192` no longer reduces estimated epoch time.
- `bs=192` already reaches `100%` peak GPU utilization, so the sweep stop condition has been met.

Example `D1` local probe:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

bash scripts/run_stage4_1_realdata_d1.sh \
  run_dir=RUN_DIR/stage4_1_realdata_probe_tmp_d1 \
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

find RUN_DIR/stage4_1_realdata_e50_b128_d1/last_weights -maxdepth 1 -type f -name '*.pt'
```

If `last_weights` are missing but D1 `last.ckpt` exists, extract them manually:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR

conda run -n TMR python - <<'PY'
from src.load import extract_ckpt
extract_ckpt("RUN_DIR/stage4_1_realdata_e50_b128_d1")
PY
```

## Expected Outputs

- D0 report:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d0/2026-04-12_d0_realdata_report.md`
- Per-stage corrected reports:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d1/2026-04-12_d1_realdata_report.md`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d1_5/2026-04-12_d1_5_realdata_report.md`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2a/2026-04-12_d2a_realdata_report.md`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2b/2026-04-12_d2b_realdata_report.md`
- D3 closure:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md`

## Read Before Running

- Current progress table:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/TAMR_PROGRESS.md`
- Loader and corrected data assumptions:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/src/data/humanml3de_event.py`
- D0 corrected data audit:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d0/2026-04-12_d0_realdata_report.md`
