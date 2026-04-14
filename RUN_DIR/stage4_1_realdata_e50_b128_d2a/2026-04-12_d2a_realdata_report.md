# D2a: Corrected Real-Data Rerun

- Stage status: **evaluated**
- Run dir: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d2a`
- Dataset root: `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Batch size: `128`
- Num workers: `0`
- Max epochs: `50`
- Last checkpoint: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d2a/tmr_d2a_humanml3d_e_None/version_0/checkpoints/last.ckpt`
- last_weights present: **yes**
- Primary retrieval score (normal+nsim R@1/R@5 mean): **32.44**

## Retrieval Metrics

| Protocol | t2m/R01 | t2m/R05 | t2m/R10 | m2t/R01 | m2t/R05 | m2t/R10 | t2m/MedR | m2t/MedR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 3.25 | 13.07 | 21.42 | 3.96 | 13.43 | 21.03 | 55.00 | 59.50 |
| threshold_0.95 | 7.47 | 20.58 | 30.65 | 6.52 | 17.52 | 25.91 | 34.00 | 45.75 |
| nsim | 39.18 | 75.26 | 84.54 | 40.21 | 71.13 | 82.47 | 2.00 | 2.00 |
| guo | 57.31 | 86.31 | 93.15 | 56.59 | 86.12 | 93.02 | 1.20 | 1.23 |
