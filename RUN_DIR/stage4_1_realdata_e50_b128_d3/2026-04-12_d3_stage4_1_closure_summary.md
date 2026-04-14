# D3: Stage4.1 Corrected Real-Data Closure Summary

- This closure supersedes the earlier Stage4.1 chain because the previous D0-D3 results were generated from the wrong HumanML3D-E source.
- Corrected trusted HumanML3D-E source:
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_train.npy`
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_val.npy`
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test.npy`
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test_condition2.npy`
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test_condition3.npy`
  - `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test_condition4.npy`

## D0 Gate Recap

- D0 scope: **data_audit_only**
- D0 training dependency: **none**
- Total train/val/test entries: **30722**
- Total captions: **83347**
- Canonical decomposed coverage: **1.00**
- Gate recommendation: **DATA-GATE GO: corrected real data support launching D1 frozen minimal event-time head.**
- Interpretation: D0 is a corrected real-data audit and launch gate only; it does not depend on whether downstream training finished full epochs.

## nsim Note

- `nsim_test` now prefers the official TMR split file and uses the corrected data overlap.
- Current overlap is `97/100`; missing keyids are `001052`, `008340`, and `M010392`.

## Stage Comparison

| Stage | Status | PrimaryScore | normal t2m/R01 | normal m2t/R01 | nsim t2m/R01 | nsim m2t/R01 | normal t2m/R05 | normal m2t/R05 | nsim t2m/R05 | nsim m2t/R05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | evaluated | 9.78 | 0.73 | 0.34 | 11.34 | 8.25 | 2.91 | 1.05 | 34.02 | 19.59 |
| D1.5 | evaluated | 9.78 | 0.73 | 0.34 | 11.34 | 8.25 | 2.91 | 1.05 | 34.02 | 19.59 |
| D2a | evaluated | 32.44 | 3.25 | 3.96 | 39.18 | 40.21 | 13.07 | 13.43 | 75.26 | 71.13 |
| D2b | evaluated | 37.17 | 4.39 | 6.50 | 48.45 | 45.36 | 16.23 | 17.67 | 79.38 | 79.38 |

## Recommendation

- Final recommendation: **Go Phase 2 with D2b**
- Winner under the current corrected evidence: **D2b**

## Per-Stage Docs

- D1: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d1/2026-04-12_d1_realdata_report.md`
- D1.5: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d1_5/2026-04-12_d1_5_realdata_report.md`
- D2a: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d2a/2026-04-12_d2a_realdata_report.md`
- D2b: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d2b/2026-04-12_d2b_realdata_report.md`
