# TAMR Progress

## 2026-04-11 Corrected HumanML3D-E Real-Data Rerun Handoff

This section supersedes the earlier Stage4.1 D0-D3 chain that was run against the wrong HumanML3D-E source.

Direct serial execution runbook:
- `STAGE4_1_REALDATA_RUNBOOK.md`

### Status Table

| Workstream | Status | Evidence / Output | Next Action |
|---|---|---|---|
| Canonical HumanML3D-E data landing | Done | `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E` now contains the six trusted `.npy` files | Keep this directory as the only canonical HumanML3D-E source for this rerun |
| EventT2M symlink wiring | Done | `EventT2M-codes-main/dataset/HumanML3D-E -> /home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E` | Do not overwrite `EventT2M-codes-main/dataset/HumanML3D` |
| TMR data path fix | Done | `configs/data/humanml3d_e.yaml` now points explicitly to `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E` | Keep all corrected reruns on this canonical path |
| HumanML3D-E loader fix | Done | `src/data/humanml3de_event.py` now reads canonical `text[].decomposed[].caption` first and uses official `nsim_test.txt` overlap when available | Use this loader for all corrected reruns |
| D0 corrected data audit | Done | `RUN_DIR/stage4_1_realdata_d0/2026-04-11_d0_realdata_report.md` | Use this D0 as the new data-gate reference; it does not depend on training completion |
| D1 throughput profiling on local 3090 | Done | `RUN_DIR/stage4_1_realdata_d1_attempt_ep10_workers4/` and `RUN_DIR/stage4_1_realdata_d1_attempt_ep2_workers4_bs32/`; TensorBoard probe showed `train_loss_step` reached only step 159/767 in epoch 0 after several minutes with `batch_size=32` | Move full experiments to server; do not keep local `batch_size=32` as the main schedule |
| Local corrected real-data GPU fit probe | Done | `RUN_DIR/stage4_1_realdata_d{1,1_5,2a,2b}_local_probe/` all completed `max_epochs=1`, `limit_train_batches=20`, `limit_val_batches=1`, `batch_size=64`, `num_workers=4` on local RTX 3090 without OOM | Keep `batch_size=64` as a valid launch point; still prefer server for full epochs because local full epoch is estimated at roughly 26-28 minutes |
| D1 corrected full rerun | Pending server | New run dir reserved: `RUN_DIR/stage4_1_realdata_d1` | Launch on server |
| D1.5 corrected full rerun | Pending server | New run dir reserved: `RUN_DIR/stage4_1_realdata_d1_5` | Launch on server after D1 |
| D2a corrected full rerun | Pending server | New run dir reserved: `RUN_DIR/stage4_1_realdata_d2a` | Launch after extracting D1 `last_weights` |
| D2b corrected full rerun | Pending server | New run dir reserved: `RUN_DIR/stage4_1_realdata_d2b` | Launch after extracting D1 `last_weights` |
| D3 Stage4.1 closure summary | Pending server | Not started | Write after D1/D1.5/D2a/D2b retrieval metrics are ready |

### Completed Facts

- Canonical HumanML3D-E directory:
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- EventT2M read path:
  - `/home/ripemangobox/Coding/Github/Motion/EventT2M-codes-main/dataset/HumanML3D-E`
- TMR explicit read path:
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Existing `EventT2M-codes-main/dataset/HumanML3D` was intentionally left untouched because it still points at the full HumanML3D tree.

### Corrected Real-Data D0 Summary

- D0 scope:
  - data audit / launch gate only
- Training dependency:
  - none; D0 statistics are computed directly from the six trusted `.npy` files

- Split entries:
  - `train=24546`
  - `val=1530`
  - `test=4646`
- Total captions across train/val/test:
  - `83347`
- Motion-level text count distribution:
  - `1 -> 2952`
  - `2 -> 2941`
  - `3 -> 24803`
  - `4 -> 26`
- Caption structure:
  - plain natural-language caption: `83347`
  - `action i:` marker caption: `0`
  - canonical `decomposed` coverage: `100%`
- Event count distribution:
  - `K=1 -> 42261 (50.70%)`
  - `K>=2 -> 41086 (49.30%)`
- Overlap cue ratio:
  - `1.81%`
- Gate result:
  - `DATA-GATE GO: corrected real data support launching D1 frozen minimal event-time head`

### nsim_test Note

- TMR official split file exists:
  - `/home/ripemangobox/Coding/Github/Motion/TMR/datasets/annotations/humanml3d/splits/nsim_test.txt`
- Official split size:
  - `100`
- Overlap with corrected real `data_test.npy`:
  - `97`
- Missing keyids:
  - `001052`
  - `008340`
  - `M010392`

### Local RTX 3090 Probe

- Probe device:
  - `NVIDIA GeForce RTX 3090 24GB`
- Probe setting:
  - `batch_size=64`
  - `num_workers=4`
  - `max_epochs=1`
  - `limit_train_batches=20`
  - `limit_val_batches=1`
- Completed without OOM:
  - `D1`
  - `D1.5`
  - `D2a`
  - `D2b`
- Observed probe durations:
  - `D1: about 87s for 20 train batches`
  - `D1.5: about 81s for 20 train batches`
  - `D2a: about 84s for 20 train batches`
  - `D2b: about 84s for 20 train batches`
- Estimated full train epoch at `batch_size=64` on this local 3090:
  - roughly `26-28 minutes/epoch` before full validation overhead
- Conclusion:
  - current `batch_size=64` does not need to be reduced for memory on this machine
  - local machine is acceptable for short probes
  - server remains the right place for the corrected formal D1-D3 rerun

### Server Run Order

1. D1
2. Extract D1 `last_weights`
3. D1 retrieval
4. D1.5
5. D1.5 retrieval
6. D2a
7. D2a retrieval
8. D2b
9. D2b retrieval
10. D3 closure summary

### Recommended Server Commands

Start from:

```bash
cd /home/ripemangobox/Coding/Github/Motion/TMR
```

One-command batch entry for the corrected chain:

```bash
bash scripts/run_stage4_1_realdata_batch.sh \
  --start-stage d0 \
  --end-stage d3 \
  --report-date 2026-04-11 \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 8
```

Unified eval + summary entry after checkpoints are ready:

```bash
bash scripts/eval_stage4_1_realdata_all.sh \
  --retrieval-batch-size 256 \
  --report-date 2026-04-11
```

Run corrected D0 again if needed:

```bash
conda run -n TMR python scripts/d0_humanml3de_event_stats.py
```

Recommended launch baseline for the corrected full-data rerun:

- start with `dataloader.batch_size=64`
- use `dataloader.num_workers=8` on server
- keep `trainer.max_epochs=2` for the first corrected comparison pass
- if server throughput is strong and wall-clock budget allows, extend to `3-5` epochs after the first pass

D1:

```bash
bash scripts/run_stage4_1_realdata_d1.sh \
  trainer.max_epochs=2 \
  dataloader.batch_size=64 \
  dataloader.num_workers=8
```

Extract D1 weights for D2a / D2b warm-start:

```bash
conda run -n TMR python - <<'PY'
from src.load import extract_ckpt
extract_ckpt("RUN_DIR/stage4_1_realdata_d1")
PY
```

D1 retrieval:

```bash
conda run -n TMR python retrieval.py \
  run_dir=RUN_DIR/stage4_1_realdata_d1 \
  protocol=all \
  batch_size=256
```

D1.5:

```bash
bash scripts/run_stage4_1_realdata_d1_5.sh \
  trainer.max_epochs=2 \
  dataloader.batch_size=64 \
  dataloader.num_workers=8
```

D1.5 retrieval:

```bash
conda run -n TMR python retrieval.py \
  run_dir=RUN_DIR/stage4_1_realdata_d1_5 \
  protocol=all \
  batch_size=256
```

D2a:

```bash
bash scripts/run_stage4_1_realdata_d2a.sh \
  trainer.max_epochs=2 \
  dataloader.batch_size=64 \
  dataloader.num_workers=8
```

D2a retrieval:

```bash
conda run -n TMR python retrieval.py \
  run_dir=RUN_DIR/stage4_1_realdata_d2a \
  protocol=all \
  batch_size=256
```

D2b:

```bash
bash scripts/run_stage4_1_realdata_d2b.sh \
  trainer.max_epochs=2 \
  dataloader.batch_size=64 \
  dataloader.num_workers=8
```

D2b retrieval:

```bash
conda run -n TMR python retrieval.py \
  run_dir=RUN_DIR/stage4_1_realdata_d2b \
  protocol=all \
  batch_size=256
```

### Do Not Reuse

- `RUN_DIR/stage4_1_d0_humanml3de`
- `RUN_DIR/stage4_1_d1`
- `RUN_DIR/stage4_1_d1_5`
- `RUN_DIR/stage4_1_d2a`
- `RUN_DIR/stage4_1_d2b`

These earlier Stage4.1 outputs were produced from the wrong HumanML3D-E source and are no longer valid for decision-making.
