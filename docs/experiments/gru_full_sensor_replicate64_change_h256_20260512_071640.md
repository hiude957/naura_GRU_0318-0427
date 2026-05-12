# gru_full_sensor_replicate64_change_h256_20260512_071640

## 目标

复刻之前效果较好的 `256 warmup + 64 rollout` 思路，并验证它在当前固定 `110ms` cache、`rollout_only` 训练流程中是否仍然有效。

本次不是回到旧实验的 `feature_dim=398`，而是在当前项目 cache 的 `feature_dim=551` 上复刻主要训练策略：

- GRU hidden size 从 `512` 降到 `256`
- 输出头改为 `LayerNorm + Linear`
- `context_len=256`
- `rollout_steps=64`
- 加入 continuous change-aware loss 权重
- 保留 action 后 `10s` continuous loss 加权

## 数据

- cache 版本：`outputs/cache/fixed_110ms_v1`
- 数据网格：固定 `110ms`
- 训练日期：`20260318` 到 `20260415`
- 验证日期：`20260416` 到 `20260427`
- `mask=0`：保留在 rollout 时间线中，但不参与 loss 或 accuracy
- 有效 target：只在 `target_mask=1` 的 sensor 维度上计算

## 配置

- 配置文件：`configs/train_gru_a100_replicate64_change.yaml`
- 模型：GRU
- 输入维度：551
- 输出维度：150，全量 sensor 直接预测
- hidden size：256
- GRU layers：2
- dropout：0.1
- 输出头：`LayerNorm + Linear`
- 参数量：`1,055,126`
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

continuous change-aware loss：

- `active_threshold: 0.01`
- `change_threshold: 0.005`
- `change_loss_weight: 3.0`
- `max_loss_weight: 5.0`

action 连续类加权：

- 只加权 continuous sensor loss
- action 后 `10s`
- weight：`2.0`
- 与 change-aware 权重相乘后使用 `max_loss_weight=5.0` 截断
- binary BCE 不加权

## 命令

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/03_train_gru.py --config configs/train_gru_a100_replicate64_change.yaml
```

## 结果

训练在 epoch 6 后提前停止。停止原因是 epoch 1 之后验证集持续恶化，继续训练会浪费 GPU。

输出：

- run dir：`runs/gru_full_sensor_replicate64_change_h256_20260512_071640`
- metrics：`runs/gru_full_sensor_replicate64_change_h256_20260512_071640/metrics.jsonl`
- best val loss checkpoint：`runs/gru_full_sensor_replicate64_change_h256_20260512_071640/checkpoints/best.pt`
- best val acc checkpoint：`runs/gru_full_sensor_replicate64_change_h256_20260512_071640/checkpoints/best_acc.pt`
- last checkpoint：`runs/gru_full_sensor_replicate64_change_h256_20260512_071640/checkpoints/last.pt`

资源：

- GPU：NVIDIA A100 80G
- 已完成 epoch：6 / 15

指标：

| epoch | train_loss | train_acc | val_loss | val_acc | val_continuous_loss | val_binary_loss | val_change_continuous_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.133750 | 0.756080 | 0.300134 | 0.492976 | 0.236069 | 0.064065 | 0.282124 |
| 2 | 0.057001 | 0.844020 | 0.408639 | 0.482730 | 0.259035 | 0.149604 | 0.293218 |
| 3 | 0.037360 | 0.847528 | 0.552116 | 0.446900 | 0.251277 | 0.300839 | 0.279198 |
| 4 | 0.030938 | 0.862550 | 0.570539 | 0.446463 | 0.237882 | 0.332657 | 0.236959 |
| 5 | 0.028745 | 0.873527 | 0.647118 | 0.433519 | 0.237461 | 0.409656 | 0.224381 |
| 6 | 0.027698 | 0.881476 | 0.727325 | 0.424246 | 0.231815 | 0.495510 | 0.216135 |

## 结论

这组复刻参数显著优于上一版 `hidden_size=512 + rollout_steps=64 + action continuous weight` 的从零 rollout-only 结果。上一版 64 步第 1 轮验证约为 `val_loss=0.663913`、`val_acc=0.267614`，本次第 1 轮提升到 `val_loss=0.300134`、`val_acc=0.492976`。

但是它仍然存在明显过拟合。训练集 loss 从 `0.133750` 降到 `0.027698`，训练准确率从 `0.756080` 升到 `0.881476`；验证集 loss 反而从 `0.300134` 升到 `0.727325`，验证准确率从 `0.492976` 降到 `0.424246`。

主要恶化仍来自 binary loss：验证集 binary loss 从 epoch 1 的 `0.064065` 增长到 epoch 6 的 `0.495510`。continuous loss 相对稳定，说明 change-aware continuous loss 的方向有效，但 binary 维度仍需要单独处理。

当前最可用 checkpoint 是 epoch 1：

- `runs/gru_full_sensor_replicate64_change_h256_20260512_071640/checkpoints/best.pt`
- `runs/gru_full_sensor_replicate64_change_h256_20260512_071640/checkpoints/best_acc.pt`

## 下一步

- 将 `binary_loss_weight` 降到 `0.2` 或 `0.5`，优先让 continuous sensor 闭环稳定。
- 加 early stopping，`patience=1` 或 `patience=2`，避免 epoch 1 后继续过拟合。
- 单独统计 binary sensor 的类别比例和翻转频率，判断是否需要 focal loss、pos_weight 或只对发生翻转的 binary 点加权。
- 如果要更接近旧实验，需要确认旧 `feature_dim=398` 是否做过 sensor 排序，把 binary sensor 放到了最后 40 维。
