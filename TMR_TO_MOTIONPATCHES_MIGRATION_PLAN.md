# TMR -> MotionPatches Migration Plan

## 1. 结论先行

基于 `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md` 的当前结论，`D2b` 仍然是 TMR 侧
最强、最稳的保留 winner；`P2a` 说明“继续直接解冻 text encoder”至少在当前配方
下边际收益有限。

因此下一步主线不再是继续深挖 TMR 里的 text encoder 解冻，而是把 **已经被 TMR
验证有效的最小机制** 迁移到 `MotionPatches-main` backbone 上。

这次迁移的目标不是“把 TMR 代码原样搬过去”，而是把下面 5 个行为约束迁过去：

1. GT event decomposition 作为训练输入，而不是只做评测侧附加分析。
2. 先从最小新增 head 开始，不一开始就叠加太多 MotionPatches 特有模块。
3. 先冻结 text side，只允许 motion side 逐步解冻。
4. 继续坚持 retrieval-first gate，主判据只看 strict `TMR-normal + TMR-nsim`。
5. temporal / EVT 指标保留，但只作为 secondary diagnostics，不单独决定方向。

这里的几个词需要区分：

- `strict TMR-normal / TMR-nsim`
  - 指 MotionPatches 在 `scripts/test.py` 里导出的 **TMR 对齐版评测文件**
  - 不是 MotionPatches 原生的 `normal.yaml / nsim.yaml`
  - 而是 `TMR-normal.yaml / TMR-nsim.yaml`
- `strict retrieval gate`
  - 指“用哪组指标当主门槛来筛选路线”
  - 在本文里，gate 就是 strict `TMR-normal + TMR-nsim`
- `决策`
  - 指最终的 keep / go / stop 判断
  - 它依赖 gate，但不等于 gate 本身
  - 更准确地说：**gate 是决策规则，决策是规则作用后的结论**

## 2. 当前事实基础

### 2.1 TMR 侧已经验证过什么

- `D1/D1.5` 说明：只改 pooling、本体全冻住，收益基本不明显。
- `D2a` 说明：有限 motion adaptation 是有效的。
- `D2b` 说明：**motion-only full unfreeze** 是当前真正起量的关键。
- `P2a` 说明：把 text encoder 也一起放开，并没有稳定超过 `D2b`。

换句话说，TMR 给 MotionPatches 的最重要启发不是“加更多 loss”，而是：

- 先用最小 event head 验证信号能不能进入 backbone；
- 再只放开 motion encoder；
- 最后才考虑 text side 或更复杂分支。

### 2.2 MotionPatches 侧已经具备什么

`MotionPatches-main` 目前并不是空白状态，已经具备：

- GT event lookup:
  - `MotionPatches-main/temporal_utils.py`
  - `HumanML3DGTEventResolver`
- event / temporal scaffold:
  - `train.event_temporal.*`
- temporal adapter:
  - `train.temporal_adapter.*`
- event-patch alignment:
  - `train.event_patch_alignment.*`
- strict TMR-aligned evaluation export:
  - `MotionPatches-main/scripts/test.py`
  - 输出 `TMR-normal.yaml`, `TMR-nsim.yaml`, `TMR-threshold_0.95.yaml`, `TMR-guo.yaml`
- EVT temporal diagnostics export:
  - 输出 `EVT-normal.yaml`, `EVT-nsim.yaml`

这意味着迁移重点不在“重写评测”，而在 **训练主线要不要改成 TMR D2b 风格**。

### 2.3 当前参考分数的可比性说明

下面的 `PrimaryScore(strict)` 定义与 TMR Stage4.1 一致，取 8 项均值：

- `TMR-normal`: `t2m/R01`, `m2t/R01`, `t2m/R05`, `m2t/R05`
- `TMR-nsim`: `t2m/R01`, `m2t/R01`, `t2m/R05`, `m2t/R05`

它的用途是做一个 **粗粒度排序分数**，避免只盯单一指标做选择。
但它不是唯一真理，因此后面的 gate 仍然保留 side constraints：

- 不能只因为某一侧 `R@5` 提高，就忽略 `normal` 或 `R@1` 的塌陷
- `PrimaryScore(strict)` 用来做主排序
- `R@1`、`normal` 稳定性、`EVT-*` diagnostics 用来做二次审查

先明确一点：

- `TMR` 的 `D1/D1.5/D2a/D2b` 是在 **corrected HumanML3D-E / `HumanML3DEventDataset`**
  上评测的
- `MotionPatches` 当前导出的 `TMR-normal.yaml / TMR-nsim.yaml` 则是从
  `data/HumanML3D/new_joints + texts + test.txt/nsim_test.txt` 读取 strict retrieval
  样本
- `HumanML3D-E` 的 GT `decomposed` event 在当前 MotionPatches 代码里主要用于
  temporal / EVT 相关构造与诊断，不等于 retrieval 样本已经切到了 HumanML3D-E

这个“不公平”不只是路径名字不同，而是至少有三层差异：

1. 样本集合不同

| Split | MotionPatches 当前 HumanML3D strict | TMR corrected HumanML3D-E | 备注 |
| --- | ---: | ---: | --- |
| `test` | 4384 | 4646 | 主 test 集大小不同 |
| 交集 | 4196 | 4196 | 只有这部分 keyid 真正重合 |
| 仅 MP 侧 | 188 | - | 在当前 HumanML3D strict 中有、HumanML3D-E 中没有 |
| 仅 HumanML3D-E 侧 | - | 450 | 在 corrected HumanML3D-E 中有、MP 当前 strict 中没有 |
| `nsim_test` 覆盖 | 100/100 | 97/100 | HumanML3D-E 缺 `001052`, `008340`, `M010392` |

2. motion 表示不同

- MotionPatches 主干读取 `new_joints/*.npy` 的原始关节序列，再转成 patch image
- TMR 的 corrected HumanML3D-E 主干读取 `data_*.npy` 里的 `(T, 263)` Guo features

3. GT event 的角色不同

- 在 TMR `D1-D2b` 里，HumanML3D-E 是训练与主评测的核心数据通路
- 在 MotionPatches 当前实现里，HumanML3D-E 更像 GT event lookup / temporal
  diagnostics 的辅助来源，而非 retrieval 主样本来源

因此，“让 MotionPatches 切到 HumanML3D-E regime” 的正确含义不是：

- 直接让 MotionPatches 读取 `data_test.npy` 里的 motion tensor

而是：

- 用 HumanML3D-E 的 keyid 集合作为 train/val/test/nsim split
- motion 仍从 `data/HumanML3D/new_joints` 读取
- text 仍从 `data/HumanML3D/texts` 读取
- event supervision 从 HumanML3D-E 的 `data_<split>.npy` 中读取 `decomposed`

所以，下面这张表 **不是公平的跨 backbone 对比表**，只能作为“各自仓库内部量级和
趋势参考”，不能拿来做 `TMR vs MotionPatches` 的正式胜负判断。

| Family | Reference | Dataset Regime | Source | PrimaryScore(strict) | 说明 |
| --- | --- | --- | --- | ---: | --- |
| MotionPatches | `pretrained` | HumanML3D strict retrieval split | `MotionPatches-main/checkpoints/pretrained/HumanML3D/contrastive_metrics` | 43.46 | 从 `data/HumanML3D` 的 `new_joints/texts/test.txt/nsim_test.txt` 评测 |
| MotionPatches | `stage2_mp_gt` | HumanML3D strict retrieval split | `MotionPatches-main/checkpoints/stage2_mp_gt/HumanML3D/contrastive_metrics` | 44.19 | GT event 参与训练/诊断，但 retrieval 样本仍是 HumanML3D 路径 |
| MotionPatches | `stage4_mp_gt_adapter_eval` | HumanML3D strict retrieval split | `MotionPatches-main/checkpoints/stage4_mp_gt_adapter_eval/HumanML3D/contrastive_metrics` | 41.32 | 当前 adapter 主线在 strict gate 下回落 |
| TMR | `D1` | corrected HumanML3D-E | `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md` | 9.78 | 最小 event head，motion/text 全冻结 |
| TMR | `D1.5` | corrected HumanML3D-E | `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md` | 9.78 | uniform pooling 控制组，与 `D1` 持平 |
| TMR | `D2a` | corrected HumanML3D-E | `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md` | 32.44 | event head + last 2 motion blocks |
| TMR | `D2b` | corrected HumanML3D-E | `STAGE4_1_REALDATA_UNIFIED_SUMMARY.md` | 37.17 | 当前 TMR winner，full motion unfreeze |

为了建立真正可用的 MotionPatches 侧公平 gate，`MP-B0` 已经在 **HumanML3D-E keyid strict
regime** 下完成一次重评：

| Family | Reference | Dataset Regime | Source | PrimaryScore(strict) | 说明 |
| --- | --- | --- | --- | ---: | --- |
| MotionPatches | `pretrained_hmle_b0_eval` | HumanML3D-E keyid strict retrieval split | `MotionPatches-main/checkpoints/pretrained_hmle_b0_eval/HumanML3D/contrastive_metrics` | 43.5312 | 对应 native 主分数 `43.85` |
| MotionPatches | `stage2_mp_gt_hmle_b0_eval` | HumanML3D-E keyid strict retrieval split | `MotionPatches-main/checkpoints/stage2_mp_gt_hmle_b0_eval/HumanML3D/contrastive_metrics` | 44.8263 | 当前公平 regime 下的主 gate 锚点；对应 native 主分数 `44.6887` |
| MotionPatches | `stage4_mp_gt_adapter_hmle_b0_eval` | HumanML3D-E keyid strict retrieval split | `MotionPatches-main/checkpoints/stage4_mp_gt_adapter_hmle_b0_eval/HumanML3D/contrastive_metrics` | 40.4475 | 对应 native 主分数 `40.3937`；由于 checkpoint 早于 learned temporal pooling head，引入 `legacy_temporal_pooling` 兼容加载 |

对 MotionPatches 当前资产，仍然有一个 **仓库内部有效结论**：

- 在 MotionPatches 自己当前的 HumanML3D strict retrieval regime 下，
  `stage4_mp_gt_adapter_eval` 没有打赢 `stage2_mp_gt`
- 在对齐后的 HumanML3D-E keyid strict regime 下，
  `stage4_mp_gt_adapter_hmle_b0_eval` 依然没有打赢 `stage2_mp_gt_hmle_b0_eval`

但这个结论不能外推出：

- `MotionPatches > TMR`
- 或 `TMR > MotionPatches`

因为它们当前不在同一个数据集 regime 上。

因此迁移策略应该是：

- 先回到更保守的 TMR D2b 逻辑；
- 先证明“最小 event head + motion-only unfreeze”在 MotionPatches 上也成立；
- 然后再讨论要不要重新叠加 adapter / alignment。

## 3. 迁移目标

### 3.1 主目标

在 MotionPatches backbone 上建立一条新的、可复现实验主线，满足：

- backbone 仍然是 MotionPatches 的 ViT patch encoder；
- event supervision 来自 HumanML3D-E GT `decomposed`；
- **训练与主评测都切到 HumanML3D-E keyid regime**，避免和 TMR 使用不同样本集合；
- 训练策略遵循 TMR `D1 -> D2a -> D2b` 梯度；
- 决策继续使用 strict `TMR-normal + TMR-nsim` 这组主 gate；
- 不覆盖现有 `stage2_mp_gt` / `stage4_mp_gt_adapter*` 资产。

### 3.1.1 MotionPatches 切到 HumanML3D-E 的正确方式

需要显式区分“样本集合对齐”和“motion 表示格式”：

- 对齐的是 **split keyid**
  - train/val/test 由 HumanML3D-E 的 `data_train.npy / data_val.npy / data_test.npy`
    的 keyid 生成
  - `nsim_test` 继续用官方 TMR split，再和 HumanML3D-E keyid 求交
- 保留的是 **MotionPatches 自己的输入表示**
  - motion 继续从 `data/HumanML3D/new_joints/*.npy` 读取
  - text 继续从 `data/HumanML3D/texts/*.txt` 读取
  - patch 化、kinematic chain、伪图像编码流程不变
- 新增的是 **event 监督来源**
  - 从 HumanML3D-E `data_<split>.npy` 读取 `decomposed`

一句话说：**MP 要切到 HumanML3D-E，切的是 keyid regime 和 event supervision，
不是把 backbone 输入从 `new_joints` 改成 `(T,263)` Guo features。**

### 3.2 非目标

这轮迁移先不把下面内容设为主线：

- text encoder partial unfreeze
- differential text/motion lr 作为第一批变量
- adapter inference branch
- event-patch alignment 与 temporal adapter 同时一起上
- rule-based event source 的复杂 ablation

这些都可以放到 `MP-D2b` 成功之后再回头做。

## 4. 设计原则

1. **最小改动先行**
   先把 TMR 里已经验证过的最小 event head 行为迁进去，不在第一轮把 MotionPatches
   的所有扩展 loss 一起打开。

2. **严格单变量推进**
   第一轮只允许控制下面 3 个维度：
   - event head 是否接入
   - motion encoder 解冻范围
   - 是否保留 strict retrieval gate

3. **先统一数据集 regime，再谈跨 backbone 比较**
   如果 MotionPatches 继续沿用当前 `data/HumanML3D` strict retrieval 路径，而 TMR
   用的是 corrected HumanML3D-E，那么所有 `TMR vs MP` 数值比较都只能算参考，
   不能算公平结论。

4. **text side 默认冻结**
   在 `MP-D2b` 没跑赢之前，不把 text encoder 当成主调参方向。

5. **评测与训练解耦**
   训练时可以产出 EVT diagnostics，但最终保留与淘汰决策仍然由 strict retrieval
   主分数决定。

6. **不覆盖已有资产**
   所有新实验使用新的 `exp_name` 前缀，不覆盖现有 checkpoint 目录。

## 5. 模块映射

| TMR D2b 里的有效部件 | MotionPatches 对应落点 | 迁移动作 |
| --- | --- | --- |
| `HumanML3DEventDataset` | `HumanML3DGTEventResolver` + 训练 batch 侧 event sequence 构造 | 复用 GT event lookup，不必重建完整数据集类 |
| `motion_temporal` token 序列 | `ClipModel._split_motion_tokens()` 里的 `time_tokens` | 用 `time_tokens` 作为 event 对齐目标 |
| `event_proj_e`, `event_proj_t` | `models/clip.py` 新增最小 head | 新增一对轻量投影层，保持与 TMR 同构 |
| masked event attention pooling | 基于 `time_tokens` 的 event-conditioned attention | 先实现 TMR 式 masked pooling，不依赖 adapter |
| masked event InfoNCE | `ClipModel.forward(..., return_loss=True)` 新增 loss 分支 | 直接复现 D1/D2a/D2b 的 event alignment loss |
| frozen text branch | `train_text_encoder=false` + text projection 冻结 | 第一轮保持 text 空间稳定 |
| motion-only unfreeze | ViT 最后 2 blocks / 全部 blocks | 对应 `MP-D2a` / `MP-D2b` |
| retrieval-first gate | `scripts/test.py` 已有 `TMR-*` 导出 | 保留现有 strict eval，作为唯一主 gate |

## 6. 推荐实验梯度

建议在 MotionPatches 里新开一条 `MP-D*` 系列，逻辑上完全对齐 TMR：

| Stage | Warm-start | Trainable | Frozen | Loss 主体 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `MP-B0` | 现有 checkpoints | 无训练 | 全部 | 只评测 | 已完成；在 HumanML3D-E keyid regime 下重评 `pretrained / stage2 / stage4`，得到 `43.5312 / 44.8263 / 40.4475` 三个公平锚点 |
| `MP-D1` | `checkpoints/pretrained/HumanML3D/best_model.pt` | `event_proj_e`, `event_proj_t` | motion encoder, text encoder, motion/text projection | `evt_align=1.0`, `global=0.1(log-only or frozen-branch)` | 证明最小 event head 在 MP time tokens 上可工作 |
| `MP-D2a` | `MP-D1` | event head + last 2 ViT blocks | text side，全局其它模块 | 与 TMR D2a 同配方 | 验证有限 motion adaptation 是否有效 |
| `MP-D2b` | `MP-D1` | event head + full motion encoder (+ motion projection) | text side | 与 TMR D2b 同配方 | 主线 winner 候选 |
| `MP-A1` | `MP-D2b` | `MP-D2b` + event_patch_alignment | text side | 在 `D2b` 上单独加 alignment | 只在 `D2b` 稳住后再测 |
| `MP-A2` | `MP-D2b` 或 `MP-A1` | `MP-D2b` + temporal_adapter | text side | 只加 adapter，不混入别的变量 | 判断 adapter 是否真能带来增益 |

### 6.1 为什么不建议直接从当前 `stage4_mp_gt_adapter` 接着调

因为当前证据显示：

- 在旧的 HumanML3D strict regime 下：
  - `stage2_mp_gt` strict 分数 `44.19`
  - `stage4_mp_gt_adapter_eval` strict 分数 `41.32`
- 在已对齐的 HumanML3D-E keyid strict regime 下：
  - `stage2_mp_gt_hmle_b0_eval` strict 分数 `44.8263`
  - `stage4_mp_gt_adapter_hmle_b0_eval` strict 分数 `40.4475`

也就是无论在 **旧 HumanML3D strict regime** 还是 **新 HumanML3D-E 对齐 regime**
下，当前更复杂的 adapter 主线都没有证明自己。

所以更合理的路线不是“继续在 adapter 上堆变量”，而是：

- 先建立一个更接近 TMR `D2b` 的保守主线；
- 如果它赢了，再把 adapter / alignment 一项一项重新加回去。

## 7. 建议新增配置块

建议在 `MotionPatches-main/conf/config.yaml` 里新增一个独立配置块，例如：

```yaml
train:
  tmr_transfer:
    enable: false
    warm_start_event_head: true
    event_align_tau: 0.1
    global_weight: 0.1
    evt_align_weight: 1.0
    max_events: 4
    freeze_text_encoder: true
    freeze_text_projection: true
    freeze_motion_projection: true
    motion_unfreeze_last_n_blocks: 0
    use_attention_pooling: true
    event_source:
      type: gt
      split: train
      use_cache: true
```

使用建议：

- `motion_unfreeze_last_n_blocks=0` 且 `freeze_motion_encoder=true` 对应 `MP-D1`
- `motion_unfreeze_last_n_blocks=2` 对应 `MP-D2a`
- `motion_unfreeze_last_n_blocks=all` 或单独开关对应 `MP-D2b`

同时，第一轮里建议显式关闭：

```yaml
train.event_temporal.enable: false
train.temporal_adapter.enable: false
train.event_patch_alignment.enable: false
```

理由很简单：先把 TMR 的 winner 机制单独验证清楚。

## 8. 文件级实施拆解

### 8.0 `HumanML3D-E` 评测 regime 迁移

这是正式迁移前必须先完成的 0 号步骤。

需要新增一套对 MotionPatches 友好的 HumanML3D-E split 生成逻辑：

1. 从 HumanML3D-E 的 `data_train.npy / data_val.npy / data_test.npy` 提取 keyid。
2. 生成 MotionPatches 可直接读取的 split files，例如：
   - `MotionPatches-main/datasets/annotations/humanml3de/splits/train.txt`
   - `MotionPatches-main/datasets/annotations/humanml3de/splits/val.txt`
   - `MotionPatches-main/datasets/annotations/humanml3de/splits/test.txt`
3. `nsim_test` 继续以官方 TMR `nsim_test.txt` 为起点，但与 HumanML3D-E keyid 求交。
4. strict retrieval 读取样本时：
   - motion 走 `data/HumanML3D/new_joints/*.npy`
   - text 走 `data/HumanML3D/texts/*.txt`
   - split keyid 走上面新生成的 HumanML3D-E split files
5. GT event supervision 与 EVT diagnostics：
   - 继续从 HumanML3D-E 的 `data_<split>.npy` 读取 `decomposed`

需要显式记录和打印：

- HumanML3D-E test keyid 总数
- 可在 `new_joints` / `texts` 中解析成功的 keyid 数
- 与官方 `nsim_test` 的交集和缺失 keyid

当前已知事实：

- HumanML3D strict `test = 4384`
- corrected HumanML3D-E `test = 4646`
- 交集 `= 4196`
- HumanML3D-E `nsim_test` 覆盖 `= 97/100`

`MP-B0` 现已完成，产出的 HumanML3D-E strict 主锚点为：

- `pretrained_hmle_b0_eval`: `43.5312`
- `stage2_mp_gt_hmle_b0_eval`: `44.8263`
- `stage4_mp_gt_adapter_hmle_b0_eval`: `40.4475`

所以 `MP-B0` 的第一职责已经完成：这套 HumanML3D-E strict eval regime 已经建起来，
后续 `MP-D*` 系列都应该默认对齐到这套锚点，而不是回头引用旧的 `44.19 / 43.46`。

### 8.1 `MotionPatches-main/models/clip.py`

新增或调整：

- 新增 TMR-style 最小 event head
  - `event_proj_e`
  - `event_proj_t`
- 新增基于 `time_tokens` 的 masked event attention pooling
- 新增 TMR-style event alignment loss
- 新增按配置冻结 / 解冻 motion blocks 的能力

### 8.2 `MotionPatches-main/scripts/train.py`

新增或调整：

- 解析 `train.tmr_transfer.*`
- 使用 `HumanML3DGTEventResolver` 构造 batch event sequence
- 按 `D1 / D2a / D2b` 控制 trainable 参数
- optimizer param groups 明确拆开
  - motion side
  - text side
  - event head
- 日志中显式打印
  - strict gate 主分数
  - avg events per caption
  - evt_align_acc

### 8.3 `MotionPatches-main/datasets/dataset.py`

可能需要补一个小改动：

- 训练时保留原始 `m_length`
- 明确构造 time-bin valid mask，避免 padded frames 参与 event attention

如果 random crop 会破坏 event 顺序，可增加一个保守选项：

- `crop_mode=fixed_start`
- 或 `crop_mode=deterministic_for_event`

第一轮建议优先保证稳定性，而不是随机 crop 多样性。

### 8.4 `MotionPatches-main/scripts/test.py`

评测主干不需要推翻，但需要新增 HumanML3D-E strict split 支持。

建议补三类增强：

- 允许 strict eval 使用 `humanml3de` split registry，而不是固定绑死 `humanml3d`
- strict retrieval 按 HumanML3D-E keyid 读样本，但 motion / text 仍从
  `data/HumanML3D` 资产读取
- 自动汇总 `PrimaryScore(strict)`
- 在 log 中直接打印相对参考点的 delta

## 9. Gate 与决策规则

### 9.1 主 gate

每个新 run 都必须产出：

- `TMR-normal.yaml`
- `TMR-nsim.yaml`

主分数定义：

```text
PrimaryScore(strict)
= mean(
  normal t2m/R01, normal m2t/R01, normal t2m/R05, normal m2t/R05,
  nsim   t2m/R01, nsim   m2t/R01, nsim   t2m/R05, nsim   m2t/R05
)
```

这个均值的合理性在于：

- 它同时约束 `normal` 和 `nsim`
- 它同时约束 `R@1` 和 `R@5`
- 能避免模型只在某一个协议或某一个 recall 档位上“刷分”

但它的局限也很明确：

- 会弱化某些单项指标的解释力，尤其是 `R@1`
- 不能表达“某一项虽小但更关键”的主观偏好
- 不适合替代逐项读表

因此本文把它定位为：

- **主排序分数**
- 不是唯一决策依据
- 最终结论仍需联动检查 `normal` 是否塌陷、`R@1` 是否显著回落、`EVT-*` 是否只是假繁荣

### 9.2 决策标准

下面的阈值默认以 **`MP-B0` 已经落盘的 HumanML3D-E strict 基线** 为准：

- `pretrained_hmle_b0_eval`: `43.5312`
- `stage2_mp_gt_hmle_b0_eval`: `44.8263`

旧的 `44.19 / 43.46` 只保留为 HumanML3D old-regime 参考，不再作为正式 gate 锚点。

- `GO`
  - `MP-D2b` 明确超过 `44.8263`
  - 且 `normal` 侧没有明显塌陷
- `MAYBE`
  - `MP-D2b` 与 `44.8263` 在 `+-0.5` 内
  - 但 `EVT-*` diagnostics 有明显改善
  - 这时再考虑 `MP-A1` 或 `MP-A2`
- `STOP / DEBUG`
  - `MP-D1` 比 `43.5312` 还低很多
  - 或 `MP-D2a/D2b` 出现 `nsim` 明显退化
  - 先检查 mask、crop、GT lookup 与 freeze 逻辑，不直接加新 loss

### 9.3 次级指标

保留但不作为主判据：

- `EVT-normal.yaml`
- `EVT-nsim.yaml`
- `evt_align_acc`
- ordering / before / after / duration diagnostics

## 10. 命名与资产保护

建议统一使用新前缀：

- `stage5_mp_tmrtransfer_b0`
- `stage5_mp_tmrtransfer_d1`
- `stage5_mp_tmrtransfer_d2a`
- `stage5_mp_tmrtransfer_d2b`
- `stage5_mp_tmrtransfer_a1`
- `stage5_mp_tmrtransfer_a2`

明确不要覆盖：

- `MotionPatches-main/checkpoints/pretrained/*`
- `MotionPatches-main/checkpoints/stage2_mp_gt/*`
- `MotionPatches-main/checkpoints/stage4_mp_gt_adapter*/*`
- `MotionPatches-main/checkpoints/stage4_mp_gt_alignv1*/*`
- TMR 侧所有 `stage4_1_realdata_*` 与 `phase2_realdata_e50_b128_p2a`

## 11. 这轮最值得优先做的事

按投入产出比排序，建议执行顺序是：

1. 在 MotionPatches 里实现独立的 `tmr_transfer` 配置分支。
2. 先跑 `MP-D1`，只验证最小 event head 和 strict eval 流水线。
3. 如果 `MP-D1` 正常，再直接跑 `MP-D2a` 和 `MP-D2b`。
4. 只有当 `MP-D2b` 站稳以后，才回头测试 `alignment` 或 `adapter`。

## 12. 一句话路线图

**先把 TMR 的 winner 机制，以最保守、最少变量的方式嫁接到 MotionPatches backbone；
先赢 strict retrieval，再谈更复杂的 temporal adapter / alignment。**

## 13. Motion Rep 重做计划（2026-04-19）

### 13.1 本轮核对结论

这轮核对的目标不是判断“现有 9MB 实验是否有参考价值”，而是判断它们是否回答了现在真正想问的问题：

- `motion rep` 本身是否按原论文语义构造；
- `kimodo / pos66` 在 T5 下的对比是否发生在 **原始 MotionPatches full ckpt / full ClipModel** 框架里；
- 带旋转表示与 MotionPatches 原生 `pos66` 的对比，是否真的是同一条 full-backbone 路线。

结论是：**当前两组实验都不满足这个目标，需要重做。**

- 现有 `motion_repr_ablation_*` 与 `t5s/*` 走的是 `MotionReprBaseline`，只保存约 `9MB` 的轻量 state dict。
- 它们没有走 MotionPatches 的 `ClipModel + DistilBERT/T5 + ViT motion encoder` 主干。
- 因此它们可以保留为轻量 probe，但**不能再当成与原始 `583M/610MB` MotionPatches ckpt 对应的公平主结论**。

### 13.2 各 motion rep 的判定

#### `guo263`

- 原始 HumanML3D / Guo 表示来自 **position -> IK -> canonical** 流程。
- `src/guofeats/motion_representation.py` 明确复现了这一点：先做统一骨架、落地、`XZ` 归零、初始朝向对齐到 `+Z`，再 `inverse_kinematics_np(...)` 提取 cont6d 与 root velocity。
- 但当前 `build_humanml3de_mp_motion_formats.py` 里的 `guo263` 不是原版：
  - 它先读取打包好的 `guo263`；
  - 然后把 `67:193` 的 rotation 段替换成了 `hml272` 提取出的 `non_root_rot6d`。
- 所以当前落盘的 `guo263` 只能算 **guo263_hml272swap hybrid**，**不再是原论文设定**。

判定：

- `guo263` 的“原始语义”应当保留 **canonical + IK**。
- 当前本地导出结果 **不符合** 原论文设定。

#### `kimodo_like_261`

- Kimodo 原始表示在 `kimodo_motionrep.py` / 文档里定义为：
  - `smooth_root_pos`
  - `global_root_heading`
  - `local_joints_positions`
  - `global_rot_data`
  - `velocities`
  - `foot_contacts`
- 关键点是：
  - rotation 是 **global joint rotations**，不是 local rot6d；
  - position block 保留整套 joints 语义，不是简单丢掉 root 后直接拼 `new_joints[:, 1:]`；
  - smooth root 与 foot contact 也不是当前脚本里的简化 `1-2-1` 卷积和固定阈值差分。
- 当前 `_build_kimodo_like_261(...)` 使用的是：
  - `smooth_root_positions(root_pos)`
  - `forward_heading`
  - `non_root_pos`
  - `joint_velocity(non_root_pos)`
  - `non_root local_rot6d`
  - `foot_contact`

判定：

- 当前 `kimodo_like_261` 是 **kimodo-inspired hybrid**；
- rotation 本该是 **global**，当前却来自 `hml272` 的 local/non-root rot6d；
- 因而 **不符合** Kimodo 原始表示设定。

#### `hml272` / `humanml272`

- 272-dim 仓库与 MotionStreamer 使用的是原生 SMPL 旋转，不走 IK recovery。
- 但当前本地 `build_humanml3d_272_self.py` 只是一个 **self-processed aligned** 版本，不是官方流程原样复现。
- 它与原仓库 `representation_272.py` 的关系是：
  - 表示布局大体对齐；
  - 但数据源改成了本地 HumanML3D clip 对齐版本；
  - `metadata.json` 里也明确写了 `fps=20`，并且 `save_repr272[:-1]` 去对齐本地 `new_joints` 长度。
- 272-dim 官方脚本来自 `amass_process.py -> face_z_transform.py -> infer_get_joints.py -> representation_272.py`，其中 `amass_process.py` 明确写了 `ex_fps = 30`。

判定：

- `hml272` 当前是 **本地对齐版 272**，不是官方 272 repo / MotionStreamer paper 的原样数据资产；
- 它仍然属于 **原生 SMPL rot6d，不是 IK**；
- 但从“是否符合原论文数据处理设定”看，答案仍然是 **不完全符合**。

#### `hy201_recon`

- HY Motion 论文把单帧表示写成 `R201`：
  - `t ∈ R3`
  - `r ∈ R6`
  - `j_r ∈ R21x6`
  - `j_p ∈ R22x3`
- 这意味着它要的是：
  - **global root translation**
  - **global body orientation**
  - **local joint rotations**
  - **local joint positions**
- 当前 `_build_hy201_recon(...)` 的拼接维度是对的，但它的 rotation 来源不是直接 paper-style root/body orientation：
  - `root_rot6d` 来自当前 `hml272` 的 rotation block 提取；
  - 而当前 `hml272` 本身又是本地 aligned 版本，不是官方 30fps 管线；
  - 因此这里的 root orientation 并不能直接等同于 HY Motion paper 里的 `global body orientation`。

判定：

- `hy201_recon` 当前仍然是 **基于 native SMPL rot6d 的重构版**，不是 IK；
- 但它**不是 paper-faithful 的 HY Motion 201 表示**；
- 尤其 root orientation 来源需要改成直接来自对齐后的 canonical SMPL root orient，而不是从 `hml272` 反拆。

#### `smpl_d135_recon`

- 这个表示当前没有找到独立论文定义，它本质上是：
  - `root_rot6d`
  - `root_xz_velocity`
  - `root_y`
  - `21 x rot6d`
- 它同样继承了当前 `hml272` root/local rotation 的来源问题。

判定：

- 它可以保留为“native SMPL rot6d minimal variant”候选；
- 但当前实现仍然属于 **从本地 aligned hml272 反拆出来的 recon schema**；
- 如果它要进入正式主对比，必须先明确其 paper/source 定义，至少要与 `hml272 / HY201` 的 canonical SMPL 源保持一致。

### 13.3 总结性判断

从“是否回答当前研究问题”这个标准看，当前资产应当拆成两类：

- 可以保留的：
  - 现有 `9MB` ablation / T5 结果，可作为轻量 probe 或方向性参考。
- 不能再直接当主结论的：
  - `guo263` 当前 hybrid 版；
  - `kimodo_like_261` 当前 local-rot 简化版；
  - `hml272` 当前 local aligned 版；
  - `hy201_recon` 当前从 `hml272` 反拆的 recon 版；
  - 所有基于这些数据、且仍走 `MotionReprBaseline` 的 T5 结果。

因此本轮正式决策为：

- **motion rep 本身需要重做；**
- **基于这些 motion rep 的 full MotionPatches DistilBERT 实验需要重做；**
- **后续 T5 三组实验也必须建立在新的 phase 1 full-backbone checkpoint 之上。**

### 13.4 Phase 1：Full MotionPatches DistilBERT + ViT 重做

目标：

- 保持 MotionPatches 主体不变；
- `pos66` 继续作为原生基线；
- 只为带旋转表示新增 full-backbone 输入路径；
- 输出的 checkpoint 必须仍然属于 **full ClipModel 家族**，不再落回 `9MB` 轻量基线。

本阶段纳入训练的 schema：

- `guo263`
- `kimodo_like_261`
- `hml272`
- `hy201_recon`
- `smpl_d135_recon`

本阶段不重训的 schema：

- `pos66`

`pos66` 的对照锚点使用两条：

- `plain00_s42`：
  - HumanML3D-E-MP / strict regime 下的 full MotionPatches plain baseline；
  - `PrimaryScore(strict) = 43.86375`；
  - checkpoint 量级与原始 full ckpt 一致，约 `583M`。
- `pretrained_hmle_b0_eval`：
  - 原始 MotionPatches `pretrained/HumanML3D/best_model.pt` 在 HumanML3D-E keyid strict 上的公平重评锚点；
  - `PrimaryScore(strict) = 43.5312`。

#### Phase 1 的初始化策略

推荐策略：

- **不用随机从零训练整个模型；**
- **也不再使用 `MotionReprBaseline` 那条轻量脚本；**
- 采用 full MotionPatches `ClipModel`；
- 以 **`plain00_s42` 的 full checkpoint** 作为首选 warm start；
- 如果某些 motion-side 输入层 shape 不兼容，只重置：
  - rep-specific motion stem / patch embed；
  - 必要的 motion positional adaptation 参数；
  - 其它 shape-compatible 的 ViT blocks、text encoder、projection head 全部沿用。

这样做的原因是：

- 它与当前要比较的 `pos66` full baseline 属于同一 regime；
- 它保留了 MotionPatches 真正有效的 DistilBERT + ViT 主干；
- 它把变量尽量压缩在“motion rep 输入方式”本身；
- 比完全随机初始化更接近用户要的“原始 `583M/610MB` ckpt setting 对应的公平对比”。

#### Phase 1 的实现要求

- 新建一套 **paper-faithful motion-format rebuild** 资产，避免覆盖旧结果：
  - `datasets/HumanML3D-E-MP/motion_formats_paper_v1`
  - `datasets/HumanML3D-E-MP/motion_format_stats_paper_v1`
- 新数据构造原则：
  - `guo263`：保留原始 packaged `guo263`，不再替换 rotation 段。
  - `kimodo_like_261`：按 Kimodo 真实语义重建，尤其要改成 **global rotations**。
  - `hml272`：明确为 canonical SMPL 旋转版本；若继续沿用 local aligned 数据，需在命名和文档中说明，不得再冒充官方 272。
  - `hy201_recon`：root orientation 改为直接来自 canonical SMPL root orient。
  - `smpl_d135_recon`：与上面同源，不再从当前 `hml272` 反拆 root orientation。
- 新训练输出目录：
  - `MotionPatches-main/checkpoints/motion_repr_full_paper_v1/<schema>/...`
- 主指标继续沿用：
  - HumanML3D-E keyid strict retrieval split
  - `PrimaryScore(strict)`
  - 联动检查 `R@1 / R@5 / normal / nsim`

### 13.5 Phase 2：基于 Phase 1 checkpoint 的 T5 三组实验

Phase 2 不再直接复用任何 `t5s/*` 里的 `9MB` 结果。

正确路径应为：

- 先完成 Phase 1；
- 用户从 Phase 1 结果里挑选若干 motion rep；
- 对每个入选 motion rep，都从其 **Phase 1 full checkpoint** 出发做 text-side 替换实验；
- T5 结果仍然保存为 full MotionPatches checkpoint，而不是轻量 baseline state dict。

Phase 2 默认三组 text encoder：

- `flan-t5-base`
- `t5-base`
- `t5-large`

Phase 2 的公平对照：

- 同 motion rep 的 DistilBERT full checkpoint
- 同训练/eval split
- 同 strict retrieval metric

### 13.6 这轮之后的执行顺序

按优先级执行：

1. 先修正 `motion format` 构建脚本，把 `guo263 / kimodo_like_261 / hml272 / hy201_recon / smpl_d135_recon` 的语义问题修正。
2. 落新的 `motion_formats_paper_v1` 与 `motion_format_stats_paper_v1`，不覆盖旧目录。
3. 在 full MotionPatches 路线上接入非 `pos66` 的输入分支，并以 `plain00_s42` 作为 warm start 跑 Phase 1。
4. 汇总 Phase 1 结果后，由用户选择进入 Phase 2 的 motion rep。
5. 再运行基于 full checkpoint 的三组 T5 实验。

### 13.7 当前状态标签

从现在开始，旧实验的状态统一记为：

- `motion_repr_ablation_local_2026-04-18_serial`: `legacy-lightweight-probe`
- `motion_repr_ablation_hml272swap_2026-04-19_serial_run2`: `legacy-lightweight-probe`
- `checkpoints/t5s/*`: `legacy-lightweight-probe`

它们不删除，但不再作为本问题的正式结论来源。
