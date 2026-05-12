# gru_full_sensor_fromscratch_rollout_only_256_20260512_063603

## 目标

验证从随机初始化开始训练 `rollout_only` GRU 是否可行。

本次实验不加载上一版 checkpoint，不执行 Stage 1 teacher-forcing，只保留 warmup/context 初始化 GRU hidden，并只对 rollout 段计算 loss。

## 数据

- cache 版本：`outputs/cache/fixed_110ms_v1`
- 数据网格：固定 `110ms`
- 训练日期：`20260318` 到 `20260415`
- 验证日期：`20260416` 到 `20260427`
- `mask=0`：保留在 rollout 时间线中，但不参与 loss 或 accuracy
- 有效 target：只在 `target_mask=1` 的 sensor 维度上计算

## 配置

- 配置文件：`configs/train_gru_a100_rollout_only.yaml`
- 模型：GRU
- 输入维度：551
- 输出维度：150，全量 sensor 直接预测
- hidden size：512
- GRU layers：2
- dropout：0.1
- optimizer：AdamW
- precision：`bf16`
- seed：`20260511`

rollout-only stage：

- `mode: rollout_only`
- `context_len: 256`
- `rollout_steps: 256`
- `batch_size: 96`
- `val_batch_size: 128`
- `epochs: 15`
- `lr: 0.0002`
- `weight_decay: 0.0001`
- `gradient_clip: 1.0`

action 连续类加权：

- 只加权 continuous sensor loss
- action 后 `10s`
- weight：`2.0`
- binary BCE 不加权

## 命令

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/03_train_gru.py --config configs/train_gru_a100_rollout_only.yaml
```

## 结果

训练在 epoch 3 后提前停止。停止原因是验证集持续恶化，继续训练会浪费 GPU。

输出：

- run dir：`runs/gru_full_sensor_fromscratch_rollout_only_256_20260512_063603`
- log：`runs/train_gru_a100_rollout_only_20260512_063602.log`
- metrics：`runs/gru_full_sensor_fromscratch_rollout_only_256_20260512_063603/metrics.jsonl`
- best val loss checkpoint：`runs/gru_full_sensor_fromscratch_rollout_only_256_20260512_063603/checkpoints/rollout_only_from_scratch_best.pt`
- best val acc checkpoint：`runs/gru_full_sensor_fromscratch_rollout_only_256_20260512_063603/checkpoints/rollout_only_from_scratch_best_acc.pt`

资源：

- GPU：NVIDIA A100 80G
- 显存约：`25GB`
- 已完成 epoch：3 / 15

指标：

| epoch | train_loss | train_acc | val_loss | val_acc | val_continuous_loss | val_binary_loss | val_action_continuous_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.076306 | 0.793758 | 0.960757 | 0.231482 | 0.195681 | 0.765076 | 0.119128 |
| 2 | 0.039419 | 0.812495 | 2.634591 | 0.226564 | 0.264190 | 2.370402 | 0.150222 |
| 3 | 0.032003 | 0.827472 | 3.017952 | 0.209020 | 0.269573 | 2.748379 | 0.148536 |

## 结论

从随机初始化直接做 `rollout_only` 训练不稳定。训练集 loss 快速下降，训练准确率上升，但验证集 loss 快速恶化，验证准确率从 `0.231482` 下降到 `0.209020`。

主要问题来自 binary loss：验证集 binary loss 从 epoch 1 的 `0.765076` 增长到 epoch 3 的 `2.748379`。这说明模型在从零开始闭环回灌时，很快在二值 sensor 上形成错误累积或分布偏移。

当前代码实现是可运行的：`rollout_only`、action 后 continuous loss 加权、`best_acc` checkpoint 和 `mask=0` 屏蔽都已通过 smoke test。但这组从零训练参数不是可用训练策略。

## 下一步

- 不建议继续从随机初始化直接训练 `rollout_steps=256`。
- 下一版建议改成 `rollout_steps=128`、`lr=0.0001`、`batch_size=128`，并只跑 5 个 epoch 观察。
- 另一个更稳的方案是保留从已有 `seq256` checkpoint 微调，但把 loss 改为 `rollout_only`。
- 需要单独检查 binary sensor 的类别比例和 BCE 退化原因。
- 可尝试降低 `binary_loss_weight`，例如 `0.2` 或 `0.5`，先让连续类 sensor 闭环稳定。
