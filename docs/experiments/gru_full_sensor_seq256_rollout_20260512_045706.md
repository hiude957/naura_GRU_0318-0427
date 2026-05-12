# gru_full_sensor_seq256_rollout_20260512_045706

## 目标

验证把 GRU 历史滑动窗口从 `1024` 步缩短到 `256` 步后，模型在单步 teacher forcing 和 128 步闭环 rollout 验证上的表现是否改善。

固定 `110ms` 网格下：

- `256` 步约等于 `28.16s`
- 上一版 `1024` 步约等于 `112.64s`

本次实验重点判断：较短历史窗口是否足够描述当前 action/state 对 sensor 的影响，并观察它是否能改善闭环自回归稳定性。

## 数据

- cache 版本：`outputs/cache/fixed_110ms_v1`
- 数据网格：固定 `110ms`
- 训练日期：`20260318` 到 `20260415`
- 验证日期：`20260416` 到 `20260427`
- 是否包含 `20260318` 初始化窗口：包含
- 是否包含空日：cache 中包含；训练采样启用 `skip_empty_target_windows: true`
- Stage 1 dataloader：104 train batch，19 val batch
- Stage 2 dataloader：419 train batch，76 val batch

## 配置

- 配置文件：`configs/train_gru_a100_seq256.yaml`
- 模型：GRU
- 输入维度：551
- 输出维度：150，全量 sensor 直接预测
- hidden size：512
- GRU layers：2
- dropout：0.1
- 连续类 loss：MAE
- 二值类 loss：BCEWithLogitsLoss
- optimizer：AdamW
- precision：`bf16`
- weight decay：`0.0001`
- gradient clip：`1.0`
- num workers：8
- seed：`20260511`

数据采样：

- `train_stride: 128`
- `val_stride: 256`
- `skip_empty_target_windows: true`

Stage 1：

- `mode: teacher_forcing`
- `seq_len: 256`
- `batch_size: 512`
- `epochs: 50`
- `lr: 0.001`

Stage 2：

- `mode: rollout`
- `init_from: stage1_best`
- `context_len: 256`
- `rollout_steps: 128`
- `batch_size: 128`
- `epochs: 20`
- `lr: 0.0003`
- `rollout_loss_weight: 0.2`

准确率规则：

- `abs(true) > 0.005` 时，`abs(pred - true) / abs(true) < 20%` 判定正确
- `abs(true) <= 0.005` 时，`abs(pred) <= 0.005` 判定正确

## 命令

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/03_train_gru.py --config configs/train_gru_a100_seq256.yaml
```

## 结果

训练已完整跑完，没有中途强制停止：

- Stage 1：50 / 50 epoch
- Stage 2：20 / 20 epoch

输出：

- run dir：`runs/gru_full_sensor_seq256_rollout_20260512_045706`
- log：`runs/train_gru_a100_seq256_20260512_045704.log`
- metrics：`runs/gru_full_sensor_seq256_rollout_20260512_045706/metrics.jsonl`
- Stage 1 best checkpoint：`runs/gru_full_sensor_seq256_rollout_20260512_045706/checkpoints/stage1_teacher_forcing_best.pt`
- Stage 2 best checkpoint：`runs/gru_full_sensor_seq256_rollout_20260512_045706/checkpoints/stage2_rollout_finetune_best.pt`

资源：

- GPU：NVIDIA A100 80G
- 训练期间显存约：`26GB - 27GB`
- 训练耗时约：49 分钟
- run 目录大小：约 `226M`

### Stage 1 Teacher Forcing

Best val loss：

| epoch | train_loss | val_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 46 | 0.055531 | 0.108782 | 0.561275 | 0.077286 | 0.031496 |

Best val accuracy：

| epoch | train_loss | val_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 26 | 0.058376 | 0.114410 | 0.626389 | 0.081332 | 0.033077 |

Last epoch：

| epoch | train_loss | val_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.055785 | 0.111636 | 0.583253 | 0.078209 | 0.033427 |

### Stage 2 Rollout Finetune

Stage 2 的验证是 `context_len=256` 加 `rollout_steps=128`。`val_acc` 是 rollout 段上的准确率；`val_loss` 包含 teacher context loss 和加权 rollout loss，因此不要和 Stage 1 的 `val_loss` 直接等价比较。

Best val loss：

| epoch | train_loss | val_loss | teacher_loss | rollout_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.037163 | 0.089680 | 0.073465 | 0.081075 | 0.739711 | 0.046407 | 0.034668 |

Best val accuracy：

| epoch | train_loss | val_loss | teacher_loss | rollout_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.035730 | 0.093441 | 0.077151 | 0.081449 | 0.749714 | 0.046806 | 0.034643 |

Last epoch：

| epoch | train_loss | val_loss | teacher_loss | rollout_loss | val_acc | val_continuous_loss | val_binary_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.023083 | 0.096785 | 0.078429 | 0.091780 | 0.696139 | 0.055454 | 0.036326 |

注意：当前 checkpoint 监控指标是 `val_loss`，所以 `stage2_rollout_finetune_best.pt` 对应 Stage 2 epoch 2，不是 val accuracy 最高的 epoch 3。

## 对比上一版 1024 步窗口

上一版 `gru_full_sensor_rollout_20260512_025814` 的核心结果：

- Stage 1 best val acc：`0.426362`
- Stage 2 best rollout val acc：`0.389656`
- Stage 2 last val acc：`0.364993`

本次 `256` 步窗口：

- Stage 1 best val acc：`0.626389`
- Stage 2 best rollout val acc：`0.749714`
- Stage 2 last val acc：`0.696139`

本次结果明显优于 `1024` 步窗口，尤其是 Stage 2 闭环 rollout 验证准确率提升很大。

## 结论

`256` 步历史窗口是目前更好的候选参数。它不仅提升了 teacher forcing 单步验证表现，也显著提升了 128 步 rollout 的闭环验证表现。

但是 Stage 2 后期存在过拟合或退化迹象：train loss 从 epoch 3 到 epoch 20 持续下降，但 val accuracy 从最高 `0.749714` 回落到 `0.696139`。这说明当前 Stage 2 固定训练 20 个 epoch 不是最优策略，后续应该引入早停，或者减少 Stage 2 epoch。

当前最可用的 checkpoint 是按 `val_loss` 保存的 Stage 2 epoch 2 checkpoint。如果更关心 accuracy，需要后续改 checkpoint 逻辑，单独保存 best val accuracy。

## 下一步

- 保留 `seq_len=256` 和 `context_len=256` 作为下一版主线参数。
- Stage 2 建议先改成 `epochs=5`，或者加入 early stopping。
- 增加按 `val_acc` 保存 checkpoint 的能力，避免最高 accuracy 的 epoch 没有模型文件。
- 跑短 horizon closed-loop eval：`1h`、`6h`、`12h`、`24h`。
- 做 per-sensor accuracy 统计，确认提升来自多数 sensor，还是少数 sensor 主导。
