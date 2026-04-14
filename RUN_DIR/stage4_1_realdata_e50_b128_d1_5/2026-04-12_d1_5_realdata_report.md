# D1.5: Corrected Real-Data Rerun

- Stage status: **evaluated**
- Run dir: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d1_5`
- Dataset root: `/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E`
- Batch size: `128`
- Num workers: `0`
- Max epochs: `50`
- Last checkpoint: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/stage4_1_realdata_e50_b128_d1_5/tmr_d1_5_humanml3d_e_None/version_0/checkpoints/last.ckpt`
- last_weights present: **yes**
- Primary retrieval score (normal+nsim R@1/R@5 mean): **9.78**

## Retrieval Metrics

| Protocol | t2m/R01 | t2m/R05 | t2m/R10 | m2t/R01 | m2t/R05 | m2t/R10 | t2m/MedR | m2t/MedR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 0.73 | 2.91 | 5.79 | 0.34 | 1.05 | 1.66 | 556.50 | 1308.00 |
| threshold_0.95 | 2.20 | 5.06 | 8.70 | 0.56 | 1.42 | 2.09 | 357.00 | 1112.75 |
| nsim | 11.34 | 34.02 | 48.45 | 8.25 | 19.59 | 27.84 | 11.00 | 26.00 |
| guo | 25.43 | 54.01 | 71.27 | 12.11 | 34.35 | 52.82 | 4.85 | 9.68 |
