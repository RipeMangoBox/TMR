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
| D1 | pending | - | - | - | - | - | - | - | - | - |
| D1.5 | pending | - | - | - | - | - | - | - | - | - |
| D2a | pending | - | - | - | - | - | - | - | - | - |
| D2b | pending | - | - | - | - | - | - | - | - | - |

## Recommendation

- Final recommendation: **Pending: wait for corrected D1/D1.5/D2a/D2b retrieval metrics.**
- Winner under the current corrected evidence: **-**

## Per-Stage Docs

- D1: pending
- D1.5: pending
- D2a: pending
- D2b: pending
