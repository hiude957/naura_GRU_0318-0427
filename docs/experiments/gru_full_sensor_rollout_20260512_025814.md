# gru_full_sensor_rollout_20260512_025814

## 目标

验证第一版 GRU 全量 sensor 预测方案是否能够在固定 `110ms` 对齐数据上稳定训练，并观察从 teacher forcing 单步训练切换到 rollout 闭环微调后，验证集误差和准确率是否改善。

本次训练使用两阶段流程：

- Stage 1：`teacher_forcing`，使用真实 cache 特征窗口做单步全量 sensor 预测。
- Stage 2：`rollout`，从 Stage 1 最优 checkpoint 初始化，使用模型预测 sensor 回填输入，进行闭环微调。

## 数据

- cache 版本：`outputs/cache/fixed_110ms_v1`
- 数据网格：固定 `110ms`
- 训练日期：`20260318` 到 `20260415`
- 验证日期：`20260416` 到 `20260427`
- 是否包含 `20260318` 初始化窗口：包含
- 是否包含空日：cache 中包含；训练采样启用 `skip_empty_target_windows: true`
- Stage 1 验证窗口：2413 个有效窗口，覆盖 7 个有有效 target 的验证日期
- Stage 2 验证 batch：38 个

## 配置

- 模型：GRU
- 输入维度：551
- 输出维度：150，全量 sensor 直接预测
- hidden size：512
- GRU layers：2
- dropout：0.1
- 连续类 loss：MAE
- 二值类 loss：BCEWithLogitsLoss
- precision：`bf16`
- optimizer：AdamW
- weight decay：`0.0001`
- accuracy 规则：
  - `abs(true) > 0.005` 时，`abs(pred - true) / abs(true) < 20%` 判定正确
  - `abs(true) <= 0.005` 时，`abs(pred) <= 0.005` 判定正确

Stage 1：

- `seq_len: 1024`
- `batch_size: 256`
- `epochs: 50`
- `lr: 0.001`

Stage 2：

- `context_len: 1024`
- `rollout_steps: 128`
- `batch_size: 64`
- `epochs: 20`
- `lr: 0.0003`
- `rollout_loss_weight: 0.2`

## 命令

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/03_train_gru.py --config configs/train_gru_a100.yaml
```

## 结果

训练完整跑完了 Stage 1 的 50 个 epoch 和 Stage 2 的 20 个 epoch。用户提出暂停时，训练已经自然完成，因此没有强行终止进程。

输出目录：

- run dir：`runs/gru_full_sensor_rollout_20260512_025814`
- metrics：`runs/gru_full_sensor_rollout_20260512_025814/metrics.jsonl`
- Stage 1 best checkpoint：`runs/gru_full_sensor_rollout_20260512_025814/checkpoints/stage1_teacher_forcing_best.pt`
- Stage 2 best checkpoint：`runs/gru_full_sensor_rollout_20260512_025814/checkpoints/stage2_rollout_finetune_best.pt`

资源和耗时：

- GPU：NVIDIA A100 80G
- 训练期间显存约：`30692 MiB`
- 训练耗时约：36 分钟

### Stage 1 Teacher Forcing

Best val loss：

| epoch | train_loss | val_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 35 | 0.097818 | 0.276894 | 0.415216 | 0.170407 | 0.106487 |

Best val accuracy：

| epoch | val_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: |
| 29 | 0.279183 | 0.426362 | 0.168037 | 0.111147 |

Last epoch：

| epoch | train_loss | val_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.097360 | 0.279110 | 0.411788 | 0.178251 | 0.100859 |

### Stage 2 Rollout Finetune

Stage 2 的 `val_acc` 是闭环 rollout 段上的准确率；`val_loss` 包含 context teacher loss 和加权 rollout loss，因此不要和 Stage 1 的 `val_loss` 直接等价比较。

Best val loss：

| epoch | train_loss | val_loss | rollout_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 0.077732 | 0.309171 | 0.301859 | 0.371097 | 0.175692 | 0.126166 |

Best val accuracy：

| epoch | val_loss | rollout_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.313072 | 0.310590 | 0.389656 | 0.174636 | 0.135953 |

Last epoch：

| epoch | train_loss | val_loss | rollout_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.077009 | 0.313972 | 0.305514 | 0.364993 | 0.176244 | 0.129270 |

## 结论

本次训练流程是有效的：数据 cache、GRU、masked loss、二值 BCE、连续 MAE、checkpoint 和 metrics 输出都正常工作。

但是从结果看，当前模型还不足以支持“一个月不使用真实 sensor 修正”的最终目标。Stage 1 的最好单步验证准确率约为 `42.64%`，Stage 2 闭环 rollout 最好准确率约为 `38.97%`，说明模型进入自回归闭环后误差仍然会放大。

Stage 2 没有明显优于 Stage 1。rollout loss 在后半段略有下降，但 accuracy 没有同步提升，最终 epoch 的 rollout accuracy 只有 `36.50%`。这说明当前模型可能在部分 sensor 上预测更平滑，但并没有让项目定义下的准确率整体改善。

## 下一步

- 先做 per-sensor accuracy 诊断，区分哪些连续 sensor 和二值 sensor 拖累整体指标。
- 单独统计二值 sensor 的正负样本比例，确认 BCE 是否被类别不平衡影响。
- 跑短 horizon 闭环评估，建议先看 `1h`、`6h`、`12h`、`24h`，不要直接上一个月。
- 对比只使用 Stage 1 best checkpoint 和 Stage 2 best checkpoint 的闭环表现，确认 rollout finetune 是否真的带来收益。
- 下一版可优先尝试降低 Stage 2 学习率或缩短 rollout 微调 epoch，避免闭环微调破坏 Stage 1 学到的单步能力。
