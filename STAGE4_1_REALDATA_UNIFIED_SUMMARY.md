# Stage4.1 Corrected Real-Data Unified Summary

This file is the current Stage4.1 source of truth after the corrected
HumanML3D-E rerun, local eval, and cleanup pass.

## Retained Artifacts

- Canonical corrected dataset:
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Warm-start reference model:
  - `models/tmr_humanml3d_guoh3dfeats`
- Final formal corrected family:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d0`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d1`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d1_5`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2a`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d2b`
  - `RUN_DIR/stage4_1_realdata_e50_b128_d3`

## Cleanup Result

The following categories were deleted because they were invalid, partial, or
fully superseded by the final corrected family:

- wrong-data Stage4.1 run_dirs
- smoke run_dirs
- local probe run_dirs
- throughput attempt run_dirs
- batch-size sweep run_dirs and their sidecar logs
- old unversioned corrected run_dirs from the intermediate rerun pass
- superseded helper scripts:
  - `scripts/run_stage4_1_d1.sh`
  - `scripts/run_stage4_1_d1_5.sh`
  - `scripts/run_stage4_1_d2a.sh`
  - `scripts/run_stage4_1_d2b.sh`

The key profiling conclusion that was preserved after cleanup is:

- representative local D2b sweep recommended `batch_size=128`, `num_workers=8`

## Baseline Clarification

There are two different notions of "baseline" in this project:

1. Warm-start backbone baseline:
   - `models/tmr_humanml3d_guoh3dfeats`
   - model target: `src.model.TMR`
   - data target: `src.data.text_motion.TextMotionDataset`
   - role in Stage4.1: provides the pretrained motion/text backbone weights that
     D1 loads before adding the minimal event head
   - important limitation: its retrieval metrics are not directly comparable to
     the corrected Stage4.1 event runs because it was trained/evaluated under
     the regular text-motion dataset path, not `HumanML3DEventDataset`

2. Stage4.1 internal comparison baseline:
   - D1 is the first fair Stage4.1 comparison point on corrected HumanML3D-E
   - D1.5, D2a, and D2b should be interpreted relative to D1 under the same
     corrected data regime

Warm-start reference retrieval metrics, for context only:

| Reference | normal t2m/R01 | normal m2t/R01 | normal t2m/R05 | normal m2t/R05 | nsim t2m/R01 | nsim m2t/R01 | nsim t2m/R05 | nsim m2t/R05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `models/tmr_humanml3d_guoh3dfeats` | 5.84 | 9.19 | 20.53 | 22.54 | 46.00 | 49.00 | 82.00 | 83.00 |

## Experiment Ladder

| Stage | Type | Initialization | Trainable Parts | Frozen Parts | Key Design Choice | Validation Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline warm-start | reference model | pretrained `models/tmr_humanml3d_guoh3dfeats/last_weights` | not retrained inside Stage4.1 | not applicable inside Stage4.1 | standard TMR text-motion retrieval backbone, no event decomposition head | provide pretrained backbone weights for D1 and a non-event reference point |
| D0 | data audit | corrected six trusted `.npy` files | no training | all | verify split size, decomposed coverage, event count distribution, overlap cues, and `nsim_test` handling | decide whether corrected real data justify launching D1 |
| D1 | training stage | baseline warm-start backbone | `event_proj_e`, `event_proj_t` only | motion encoder, text encoder, motion decoder | frozen minimal event-time head with attention pooling + masked InfoNCE event alignment + weak global contrastive loss | test whether a tiny event head alone can improve corrected retrieval |
| D1.5 | training control | same warm-start as D1 | same as D1 | same as D1 | identical to D1 except attention pooling is replaced by uniform masked averaging | isolate whether D1 gains come from attention pooling itself |
| D2a | training stage | D1 `last_weights` | event head + last 2 motion encoder transformer blocks | text encoder, motion decoder, all earlier motion blocks | keep D1 recipe, only partially unfreeze motion encoder; nested tensor fast-path stays disabled | test whether limited motion adaptation improves corrected retrieval over frozen D1 |
| D2b | training stage | D1 `last_weights` | event head + full motion encoder | text encoder, motion decoder | keep D1 recipe, only expand motion unfreeze scope from last-2 blocks to full encoder | test whether full motion adaptation outperforms D2a under the same minimal-head recipe |
| D3 | closure gate | D1/D1.5/D2a/D2b retrieval outputs | no training | all | retrieval-first comparison using `normal + nsim`; do not decide by `evt_align_acc` alone | choose `Keep D2a`, `Go D3`, or `Go Phase 2 with D2b` |
| Phase 2 | follow-up experiment track | corrected D2b `last_weights` | current approved P2a path trains full motion encoder + full text encoder + event head | motion decoder | keep the D2b retrieval recipe and only expand training scope to the text encoder; launch under `phase2_realdata_e50_b128_*` so Stage4.1 stays untouched | compare directly against corrected D2b on `normal + nsim` and decide whether text adaptation is worth keeping |

## Freeze / Objective Summary

| Stage | Motion Encoder | Text Encoder | Motion Decoder | Event Head | Pooling | Losses |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | frozen | frozen | frozen | trainable | attention | masked event InfoNCE + global InfoNCE (`global=0.1`, `evt_align=1.0`) |
| D1.5 | frozen | frozen | frozen | trainable | uniform masked average | same as D1 |
| D2a | last 2 blocks trainable, earlier blocks frozen | frozen | frozen | trainable | attention | same as D1 |
| D2b | full encoder trainable | frozen | frozen | trainable | attention | same as D1 |

Notes:

- `D2a` and `D2b` both warm-start from D1, not from each other.
- `D2a` and `D2b` keep the text encoder and motion decoder frozen.
- `D2a` and `D2b` keep the motion encoder nested tensor fast-path disabled.

## Corrected Data Recap

- Trusted files:
  - `data_train.npy`
  - `data_val.npy`
  - `data_test.npy`
  - `data_test_condition2.npy`
  - `data_test_condition3.npy`
  - `data_test_condition4.npy`
- Corrected D0 result:
  - total train/val/test entries: `30722`
  - total captions: `83347`
  - canonical `decomposed` coverage: `100%`
  - gate: `DATA-GATE GO`
- `nsim_test` handling:
  - official TMR split file is preferred
  - corrected overlap is `97/100`
  - missing keyids: `001052`, `008340`, `M010392`

## Corrected Stage Results

PrimaryScore is the mean of:

- normal `t2m/R01`
- normal `m2t/R01`
- normal `t2m/R05`
- normal `m2t/R05`
- nsim `t2m/R01`
- nsim `m2t/R01`
- nsim `t2m/R05`
- nsim `m2t/R05`

| Stage | PrimaryScore | normal t2m/R01 | normal m2t/R01 | nsim t2m/R01 | nsim m2t/R01 | normal t2m/R05 | normal m2t/R05 | nsim t2m/R05 | nsim m2t/R05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 9.78 | 0.73 | 0.34 | 11.34 | 8.25 | 2.91 | 1.05 | 34.02 | 19.59 |
| D1.5 | 9.78 | 0.73 | 0.34 | 11.34 | 8.25 | 2.91 | 1.05 | 34.02 | 19.59 |
| D2a | 32.44 | 3.25 | 3.96 | 39.18 | 40.21 | 13.07 | 13.43 | 75.26 | 71.13 |
| D2b | 37.17 | 4.39 | 6.50 | 48.45 | 45.36 | 16.23 | 17.67 | 79.38 | 79.38 |

## Interpretation

- D1 and D1.5 are effectively tied under the corrected run, so the attention
  pooling change alone did not create a visible retrieval gap while the
  backbone stayed fully frozen.
- The meaningful gains arrive only after motion encoder unfreezing.
- D2a shows that partial motion adaptation is already useful.
- D2b is the strongest corrected result on both `normal` and `nsim`, so the
  corrected closure still supports `winner = D2b`.

## Final Decision

- D3 recommendation:
  - `Go Phase 2 with D2b`
- Current winner:
  - `D2b`
- Immediate implication:
  - if future work continues, it should start from the corrected D2b branch and
    not from D1 or D2a
  - the first retained Phase2 entry is `P2a`, documented separately under the
    `phase2_realdata_e50_b128_*` family

## Canonical Documents

- RunDir registry:
  - `STAGE4_1_REALDATA_RUNDIRS.md`
- Execution runbook:
  - `STAGE4_1_REALDATA_RUNBOOK.md`
- Final closure:
  - `RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md`
- Progress log:
  - `TAMR_PROGRESS.md`
- Phase2 runbook:
  - `PHASE2_REALDATA_RUNBOOK.md`
- Phase2 run registry:
  - `PHASE2_REALDATA_RUNDIRS.md`
