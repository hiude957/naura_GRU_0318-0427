# NAURA GRU Project

本仓库用于把半导体设备 `log`、`apc`、`livedata` 数据对齐到固定 `110ms` 时间网格，并训练 GRU 模型做长期闭环 sensor 预测。

当前训练方案：

- Stage 1 使用真实 cache 输入做非闭环监督训练，直接预测全量 `150` 个 sensor。
- Stage 2 从 Stage 1 best checkpoint 启动，加入短距离 rollout 微调，让模型学习使用自己的预测 sensor 继续滚动。
- 连续类 sensor 使用 MAE，二值类 sensor 使用 BCEWithLogitsLoss。
- 最终闭环评估不使用真实 sensor 修正输入。

核心规则以项目本地 skill 为准：

- `.agents/skills/semiconductor-trainset-alignment/references/merge-rules.md`
- `.agents/skills/semiconductor-trainset-alignment/references/project-architecture.md`
