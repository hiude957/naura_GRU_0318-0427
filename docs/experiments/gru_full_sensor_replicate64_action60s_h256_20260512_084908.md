# gru_full_sensor_replicate64_action60s_h256_20260512_084908

## 目标

验证“action 发生后以及之后一段时间内，提高 continuous sensor loss 权重”是否能改善 continuous sensor 的闭环表现。

本次沿用上一版复刻旧方案的主体设置，只增强 action 后 continuous 权重：

- GRU hidden size：`256`
- 输出头：`LayerNorm + Linear`
- `context_len=256`
- `rollout_steps=64`
- continuous change-aware loss 保持开启
- action 后 continuous 加权从 `10s / 2.0x` 改为 `60s / 5.0x`

## 数据

- cache 版本：`outputs/cache/fixed_110ms_v1`
- 数据网格：固定 `110ms`
- 训练日期：`20260318` 到 `20260415`
- 验证日期：`20260416` 到 `20260427`
- `mask=0`：保留在 rollout 时间线中，但不参与 loss 或 accuracy
- 有效 target：只在 `target_mask=1` 的 sensor 维度上计算

## 配置

- 配置文件：`configs/train_gru_a100_replicate64_action60s.yaml`
- 模型：GRU
- 输入维度：551
- 输出维度：150，全量 sensor 直接预测
- hidden size：256
- GRU layers：2
- dropout：0.1
- 输出头：`LayerNorm + Linear`
- optimizer：AdamW
- precision：`bf16`
- seed：`20260511`

rollout-only stage：

- `mode: rollout_only`
- `context_len: 256`
- `rollout_steps: 64`
- `batch_size: 128`
- `val_batch_size: 256`
- `epochs: 15`
- `lr: 0.0001`
- `weight_decay: 0.0001`
- `gradient_clip: 1.0`

continuous action weight：

- `window_ms: 60000`
- `weight: 5.0`
- 实际作用：action 后约 `545` 个 110ms 网格内，提高 continuous sensor loss 权重

continuous change-aware loss：

- `active_threshold: 0.01`
- `change_threshold: 0.005`
- `change_loss_weight: 3.0`
- `max_loss_weight: 8.0`

说明：action 权重和 change 权重相乘后，用 `max_loss_weight=8.0` 截断。

## 命令

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/03_train_gru.py --config configs/train_gru_a100_replicate64_action60s.yaml
```

## 结果

训练在 epoch 7 后停止。停止原因是验证集从 epoch 1 后持续恶化，且第 1 轮也弱于上一版 `10s / 2.0x` action 权重配置；按用户要求，本次没有继续跑 4 月 27 日全天闭环评估。

输出：

- run dir：`runs/gru_full_sensor_replicate64_action60s_h256_20260512_084908`
- metrics：`runs/gru_full_sensor_replicate64_action60s_h256_20260512_084908/metrics.jsonl`
- best val loss checkpoint：`runs/gru_full_sensor_replicate64_action60s_h256_20260512_084908/checkpoints/best.pt`
- best val acc checkpoint：`runs/gru_full_sensor_replicate64_action60s_h256_20260512_084908/checkpoints/best_acc.pt`
- last checkpoint：`runs/gru_full_sensor_replicate64_action60s_h256_20260512_084908/checkpoints/last.pt`

指标：

| epoch | train_loss | train_acc | val_loss | val_acc | val_continuous_loss | val_binary_loss | val_action_continuous_loss | val_change_continuous_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.138677 | 0.741576 | 0.340649 | 0.465853 | 0.262901 | 0.077748 | 0.194962 | 0.298936 |
| 2 | 0.064981 | 0.822403 | 0.417554 | 0.457960 | 0.276363 | 0.141191 | 0.204588 | 0.316892 |
| 3 | 0.045770 | 0.834866 | 0.779228 | 0.368412 | 0.262875 | 0.516354 | 0.192905 | 0.261000 |
| 4 | 0.036240 | 0.847775 | 1.152869 | 0.308353 | 0.287155 | 0.865714 | 0.208744 | 0.273276 |
| 5 | 0.033220 | 0.853710 | 1.253563 | 0.294974 | 0.286527 | 0.967036 | 0.211214 | 0.262950 |
| 6 | 0.031654 | 0.861673 | 1.290661 | 0.294397 | 0.296408 | 0.994253 | 0.219564 | 0.277984 |
| 7 | 0.030684 | 0.864429 | 1.356779 | 0.293985 | 0.315141 | 1.041638 | 0.231290 | 0.320246 |

## 对比

上一版 `10s / 2.0x` action 权重配置在 epoch 1 的窗口级验证结果：

- `val_loss=0.300134`
- `val_acc=0.492976`
- `val_continuous_loss=0.236069`
- `val_binary_loss=0.064065`

本次 `60s / 5.0x` action 权重配置在 epoch 1：

- `val_loss=0.340649`
- `val_acc=0.465853`
- `val_continuous_loss=0.262901`
- `val_binary_loss=0.077748`

因此，更强的 action 后 continuous 加权没有改善短窗口验证，反而让 continuous 和 binary 验证指标都变差。

## 结论

这组参数不建议继续作为主线。把 action 后窗口从 `10s` 扩大到 `60s`，并把权重从 `2.0` 提到 `5.0` 后，模型更快过拟合，验证集 loss 从 epoch 1 的 `0.340649` 增长到 epoch 7 的 `1.356779`，验证准确率从 `0.465853` 降到 `0.293985`。

当前结果说明：continuous accuracy 低不一定能通过单纯扩大 action 后加权窗口解决。下一步更值得检查的是 continuous sensor 的长期漂移来源，例如 target 对齐方式、归一化口径、长期闭环中的 continuous sensor 均值漂移，以及是否需要按具体 sensor 做权重而不是统一 action window 权重。
