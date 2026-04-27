# Retrieval Results Summary

## Evaluation directories

- **TMR**: `/home/ripemangobox/Coding/Github/Motion/TMR/RUN_DIR/contrastive_metrics`
- **MotionPatches**: `/data/Life Me/Obsidian Respository/ResearchWY/linkedCodebases/MotionPatches-main/checkpoints/pretrained/HumanML3D/contrastive_metrics`
- **EventT2M**: `/data/Life Me/Obsidian Respository/ResearchWY/linkedCodebases/EventT2M-codes-main/checkpoints/pretrained/HumanML3D/eval`

## Native evaluation markers

- `*` 表示该单元格对应仓库原生支持的评测协议。
- 未标 `*` 的结果表示为了与 TMR 对齐而补充的 retrieval-style 评测。

| Repo | Native protocols | Native files |
| --- | --- | --- |
| TMR | `normal`, `threshold_0.95`, `nsim`, `guo` | `normal` → `normal.yaml`, `threshold_0.95` → `threshold_0.95.yaml`, `nsim` → `nsim.yaml`, `guo` → `guo.yaml` |
| MotionPatches | `normal`, `guo` | `normal` → `normal.yaml`, `guo` → `guo.yaml` |
| EventT2M | `normal` | `normal` → `E-native_normal.yaml`, `native_normal.yaml` |

## Quick view

| Protocol | Metric | TMR | MotionPatches | EventT2M |
| --- | --- | --- | --- | --- |
| normal | t2m/R01 | 5.47 * | 11.66 * | 5.82 * |
| normal | t2m/R05 | 19.71 * | 28.65 * | 20.53 * |
| normal | t2m/R10 | 30.47 * | 40.44 * | 31.32 * |
| normal | t2m/MedR | 27.00 * | 17.00 * | 27.00 * |
| normal | m2t/R01 | 9.53 * | 12.34 * | 9.17 * |
| normal | m2t/R05 | 23.06 * | 28.65 * | 22.54 * |
| normal | m2t/R10 | 32.96 * | 38.89 * | 32.62 * |
| normal | m2t/MedR | 27.00 * | 19.00 * | 25.50 * |
| threshold_0.95 | t2m/R01 | 11.22 * | 14.19 | 12.98 |
| threshold_0.95 | t2m/R05 | 27.24 * | 31.16 | 28.49 |
| threshold_0.95 | t2m/R10 | 38.21 * | 42.81 | 39.03 |
| threshold_0.95 | t2m/MedR | 19.00 * | 15.00 | 18.00 |
| threshold_0.95 | m2t/R01 | 13.41 * | 15.24 | 12.32 |
| threshold_0.95 | m2t/R05 | 27.81 * | 30.31 | 27.58 |
| threshold_0.95 | m2t/R10 | 37.71 * | 39.90 | 37.73 |
| threshold_0.95 | m2t/MedR | 21.50 * | 17.00 | 20.00 |
| nsim | t2m/R01 | 50.00 * | 51.00 | 66.00 |
| nsim | t2m/R05 | 83.00 * | 86.00 | 95.00 |
| nsim | t2m/R10 | 89.00 * | 90.00 | 100.00 |
| nsim | t2m/MedR | 1.50 * | 1.00 | 1.00 |
| nsim | m2t/R01 | 48.00 * | 51.00 | 72.00 |
| nsim | m2t/R05 | 83.00 * | 86.00 | 95.00 |
| nsim | m2t/R10 | 88.00 * | 89.00 | 99.00 |
| nsim | m2t/MedR | 2.00 * | 1.00 | 1.00 |
| guo | t2m/R01 | 67.27 * | 72.88 * | 67.52 |
| guo | t2m/R05 | 91.35 * | 93.59 * | 91.47 |
| guo | t2m/R10 | 95.32 * | 96.88 * | 95.39 |
| guo | t2m/MedR | 1.02 * | 1.00 * | 1.03 |
| guo | m2t/R01 | 68.09 * | 73.81 * | 68.18 |
| guo | m2t/R05 | 91.26 * | 93.75 * | 91.20 |
| guo | m2t/R10 | 95.39 * | 96.81 * | 95.39 |
| guo | m2t/MedR | 1.02 * | 1.00 * | 1.02 |

## normal

| Metric | TMR | MotionPatches | EventT2M |
| --- | --- | --- | --- |
| t2m/R01 | 5.47 * | 11.66 * | 5.82 * |
| t2m/R02 | 10.15 * | 15.83 * | 10.93 * |
| t2m/R03 | 13.34 * | 21.85 * | 14.35 * |
| t2m/R05 | 19.71 * | 28.65 * | 20.53 * |
| t2m/R10 | 30.47 * | 40.44 * | 31.32 * |
| t2m/MedR | 27.00 * | 17.00 * | 27.00 * |
| m2t/R01 | 9.53 * | 12.34 * | 9.17 * |
| m2t/R02 | 12.04 * | 15.17 * | 11.70 * |
| m2t/R03 | 17.13 * | 21.58 * | 16.77 * |
| m2t/R05 | 23.06 * | 28.65 * | 22.54 * |
| m2t/R10 | 32.96 * | 38.89 * | 32.62 * |
| m2t/MedR | 27.00 * | 19.00 * | 25.50 * |

## threshold_0.95

| Metric | TMR | MotionPatches | EventT2M |
| --- | --- | --- | --- |
| t2m/R01 | 11.22 * | 14.19 | 12.98 |
| t2m/R02 | 14.51 * | 17.95 | 16.20 |
| t2m/R03 | 20.00 * | 24.52 | 21.33 |
| t2m/R05 | 27.24 * | 31.16 | 28.49 |
| t2m/R10 | 38.21 * | 42.81 | 39.03 |
| t2m/MedR | 19.00 * | 15.00 | 18.00 |
| m2t/R01 | 13.41 * | 15.24 | 12.32 |
| m2t/R02 | 15.58 * | 16.72 | 14.90 |
| m2t/R03 | 21.49 * | 23.68 | 21.24 |
| m2t/R05 | 27.81 * | 30.31 | 27.58 |
| m2t/R10 | 37.71 * | 39.90 | 37.73 |
| m2t/MedR | 21.50 * | 17.00 | 20.00 |

## nsim

| Metric | TMR | MotionPatches | EventT2M |
| --- | --- | --- | --- |
| t2m/R01 | 50.00 * | 51.00 | 66.00 |
| t2m/R02 | 68.00 * | 73.00 | 84.00 |
| t2m/R03 | 78.00 * | 80.00 | 91.00 |
| t2m/R05 | 83.00 * | 86.00 | 95.00 |
| t2m/R10 | 89.00 * | 90.00 | 100.00 |
| t2m/MedR | 1.50 * | 1.00 | 1.00 |
| m2t/R01 | 48.00 * | 51.00 | 72.00 |
| m2t/R02 | 66.00 * | 67.00 | 86.00 |
| m2t/R03 | 73.00 * | 81.00 | 90.00 |
| m2t/R05 | 83.00 * | 86.00 | 95.00 |
| m2t/R10 | 88.00 * | 89.00 | 99.00 |
| m2t/MedR | 2.00 * | 1.00 | 1.00 |

## guo

| Metric | TMR | MotionPatches | EventT2M |
| --- | --- | --- | --- |
| t2m/R01 | 67.27 * | 72.88 * | 67.52 |
| t2m/R02 | 80.95 * | 86.02 * | 81.14 |
| t2m/R03 | 86.54 * | 90.10 * | 86.27 |
| t2m/R05 | 91.35 * | 93.59 * | 91.47 |
| t2m/R10 | 95.32 * | 96.88 * | 95.39 |
| t2m/MedR | 1.02 * | 1.00 * | 1.03 |
| m2t/R01 | 68.09 * | 73.81 * | 68.18 |
| m2t/R02 | 81.25 * | 86.18 * | 81.84 |
| m2t/R03 | 86.98 * | 90.26 * | 86.88 |
| m2t/R05 | 91.26 * | 93.75 * | 91.20 |
| m2t/R10 | 95.39 * | 96.81 * | 95.39 |
| m2t/MedR | 1.02 * | 1.00 * | 1.02 |
