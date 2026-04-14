# D2b: Corrected Real-Data Rerun

- Stage status: **evaluated**
- Run dir: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d2b`
- Dataset root: `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Batch size: `128`
- Num workers: `0`
- Max epochs: `50`
- Last checkpoint: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d2b/tmr_d2b_humanml3d_e_None/version_0/checkpoints/last.ckpt`
- last_weights present: **yes**
- Primary retrieval score (normal+nsim R@1/R@5 mean): **37.17**

## Retrieval Metrics

| Protocol | t2m/R01 | t2m/R05 | t2m/R10 | m2t/R01 | m2t/R05 | m2t/R10 | t2m/MedR | m2t/MedR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 4.39 | 16.23 | 26.39 | 6.50 | 17.67 | 26.78 | 36.00 | 36.50 |
| threshold_0.95 | 9.58 | 25.05 | 35.62 | 10.27 | 22.73 | 32.57 | 23.00 | 29.00 |
| nsim | 48.45 | 79.38 | 92.78 | 45.36 | 79.38 | 92.78 | 2.00 | 2.00 |
| guo | 64.29 | 90.06 | 94.96 | 64.74 | 90.02 | 94.98 | 1.06 | 1.03 |
