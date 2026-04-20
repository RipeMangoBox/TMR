# Stage4.1 Next-Session Prompt

Copy the block below into a new session if we want the next agent to continue
from the cleaned and consolidated Stage4.1 state.

```text
你在继续同一个仓库任务，请直接执行，不要先问方案。

工作目录：
/home/ripemangobox/Coding/Github/Motion/TMR

当前仓库状态已经完成这些关键步骤：
1. corrected HumanML3D-E 真实数据链已经完成 D0-D3 正式 rerun。
2. 当前唯一有效的正式实验家族是：
   - RUN_DIR/stage4_1_realdata_e50_b128_d0
   - RUN_DIR/stage4_1_realdata_e50_b128_d1
   - RUN_DIR/stage4_1_realdata_e50_b128_d1_5
   - RUN_DIR/stage4_1_realdata_e50_b128_d2a
   - RUN_DIR/stage4_1_realdata_e50_b128_d2b
   - RUN_DIR/stage4_1_realdata_e50_b128_d3
3. 旧 wrong-data Stage4.1 结果、smoke、probe、profiling ckpt 目录已经清理，不要恢复。
4. 当前 unified summary 已存在：
   - STAGE4_1_REALDATA_UNIFIED_SUMMARY.md
5. D3 corrected closure 已存在：
   - RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md

你必须先阅读这些文件再继续：
- STAGE4_1_REALDATA_UNIFIED_SUMMARY.md
- STAGE4_1_REALDATA_RUNBOOK.md
- STAGE4_1_REALDATA_RUNDIRS.md
- TAMR_PROGRESS.md
- RUN_DIR/stage4_1_realdata_e50_b128_d3/2026-04-12_d3_stage4_1_closure_summary.md

不要再依赖已经删除的 PHASE2_REALDATA_* 零散文档；P2a 的结果已经并入
STAGE4_1_REALDATA_UNIFIED_SUMMARY.md。

当前已经确认的结论：
- corrected D0 gate = GO
- D1 与 D1.5 基本持平
- D2a 明显优于 D1
- D2b 明显优于 D2a
- corrected Stage4.1 winner = D2b
- D3 recommendation = Go Phase 2 with D2b
- P2a 已完成，PrimaryScore = 36.34，低于 D2b 的 37.17
- 当前 overall winner 仍是 D2b

重要实现事实：
- train.py 里不再硬编码把所有训练强制绑到 GPU0；它现在只在外部没有设置时才默认 `CUDA_VISIBLE_DEVICES=0`
- real-data stage scripts 已支持 `CONDA_EXE -> command -v conda -> $HOME/miniconda3/bin/conda` 兜底
- Stage4.1 默认 run_prefix 已切到：
  - stage4_1_realdata_e50_b128
- D2a / D2b 默认 warm-start 应以：
  - RUN_DIR/stage4_1_realdata_e50_b128_d1/last_weights
  为准
- HumanML3D-E-MP motion-repr launcher 现在默认走 retrieval-first 配方：
  - `scripts/run_tmr_humanml3de_mp_gpu0.sh`
  - `scripts/run_tmr_humanml3de_mp_gpu1.sh`
  - 默认 `MODEL_NAME=tmr_d2b_retrieval_first`
  - 默认 `BATCH_SIZE=32`
  - 默认 `SEED=1234`

如果你要继续做 Phase 2 或新实验，请遵守：
1. 不要覆盖当前正式 family；新实验请显式加新后缀。
2. 不要重新引入已删除的 wrong-data / smoke / probe run_dir。
3. 继续坚持 retrieval-first gate，以 `normal + nsim` 为主，不以 `evt_align_acc` 单独决策。
4. 如果你修改任何 Stage4.1 相关默认路径，必须同步更新 runbook / registry / summary 文档。
5. 当前首个已完成的 Phase 2 条目是：
   - `RUN_DIR/phase2_realdata_e50_b128_p2a`
   - 含义是 `D2b warm-start + full text encoder unfreeze`
   - 结论是没有超过 D2b；如果继续请新建后缀，不要覆盖 `p2a`

如果要继续新实验或 Phase 2，请直接执行，不要先做方案讨论；只在 commentary 中做简短进度同步。
开始前只需自行判断：
1. 当前任务是否需要新增 run_prefix
2. 是否会触碰当前正式 family
3. 哪些文档需要同步更新
```
