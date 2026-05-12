# GRU 训练与评估策略

这一页描述当前项目的模型目标、GRU 推荐参数和准确率判定口径。回答模型训练、参数选择、长期预测或评估问题时优先读取这一页。

## 1. 模型目标

当前使用的模型是 GRU。

项目最终目标是让模型在没有任何真实 sensor 数据修正的情况下，使用自身历史预测结果继续向后滚动，稳定闭环预测一个月。

回答时应区分：

- teacher-forcing 或单步预测指标：用于训练过程观察，但不能单独代表最终目标
- closed-loop rolling prediction：用于验证模型是否能在无真实 sensor 修正时长期运行

如果用户问“稳定运行一个月”或“一个月预测”，按闭环滚动预测理解。

## 2. 当前硬件

当前开发环境是火山引擎开发机。

主要硬件：

- GPU：NVIDIA A100 80G
- CPU：Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz

## 3. A100 80G 推荐起始参数

这些参数是 A100 80G 上的推荐起点，不是固定最优值。实际训练时应根据显存占用、收敛速度、验证集闭环误差和长期稳定性调整。

推荐起点：

- `model = GRU`
- `hidden_size = 512`
- `num_layers = 2`
- `dropout = 0.1`
- `seq_len = 1024`
- 第一版直接预测全量 `150` 个归一化 sensor，不预测 delta
- 连续类 sensor 使用 `MAE`
- 二值类 sensor 使用 `BCEWithLogitsLoss`
- 显存允许时试 `seq_len = 2048`
- `batch_size = 256`
- `seq_len = 1024` 且显存充足时可试 `batch_size = 512`
- mixed precision 优先使用 `bf16`
- `num_workers = 8~16`
- `pin_memory = true`
- `gradient_clip = 1.0`
- optimizer 使用 `AdamW`
- `lr = 1e-3`
- `weight_decay = 1e-4`

当前推荐训练流程分两阶段：

1. `stage1_teacher_forcing`：非闭环监督训练，`seq_len=1024`、`batch_size=256`、`lr=1e-3`、`epochs=50`。
2. `stage2_rollout_finetune`：从 stage1 best checkpoint 启动，`context_len=1024`、`rollout_steps=128`、`batch_size=64`、`lr=3e-4`、`rollout_loss_weight=0.2`、`epochs=20`。

Stage 2 的 rollout 训练规则：

- 起始 `context_len` 使用真实历史窗口初始化 GRU hidden state。
- rollout 段不再使用真实 sensor 修正输入，sensor 输入由模型上一时刻预测值回灌。
- action、state、time 仍来自 aligned/cache 序列。
- 真实 sensor 只用于计算 loss。
- 连续类预测回灌前 clamp 到 `[0, 1]`。
- 二值类 loss 用 logits，回灌用 `sigmoid(logits)` 概率值，评估时再按阈值判断类别。
- rollout 输入中的 sensor mask 置为 `0`，source_onehot 置为 `carried_sensor`。

如果出现显存不足：

1. 先降低 `batch_size`
2. 再降低 `seq_len`
3. 最后再降低 `hidden_size` 或 `num_layers`

如果闭环滚动预测发散：

1. 优先检查归一化和反归一化是否一致
2. 检查 action/state 特征在滚动过程中是否正确更新
3. 降低学习率，例如试 `3e-4`
4. 增加 dropout 或 weight decay
5. 缩短训练阶段的预测跨度，再逐步拉长

## 4. 闭环滚动评估

最终评估不能只看下一步预测误差。必须包含闭环滚动预测：

- 起始窗口使用真实历史数据初始化
- 进入预测区间后，不再使用真实 sensor 修正输入
- 后续 sensor 输入由模型自己的预测结果滚动生成
- action 和 IO state 仍按已知或计划中的控制序列更新
- 评估长度至少覆盖一个月目标中的关键时间跨度

回答时应说明：单步误差低不代表一个月闭环稳定，闭环滚动误差和发散情况才是最终目标的核心指标。

## 5. 准确率判定

这些 sensor 数据都是归一化后的数据。对每个有效 sensor 预测点，按以下规则判断预测是否正确。

当 `abs(true) > 0.005`：

`abs(pred - true) / abs(true) < 0.20`

满足则判为正确。

当 `abs(true) <= 0.005`：

`abs(pred) < 0.005`

满足则判为正确。

总准确率：

`accuracy = correct_prediction_points / valid_prediction_points`

说明：

- `true` 表示归一化后的真实值
- `pred` 表示归一化后的预测值
- `abs(true) == 0.005` 归入近零分支
- 统计时只计入有效预测点，是否有效应与项目中的 `mask_*`、`loss_mask` 或当前评估脚本的有效性定义保持一致
