# Experiment Index

| run_id | 日期 | 主要改动 | best val loss | accuracy | closed-loop 结论 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| [gru_full_sensor_rollout_20260512_025814](gru_full_sensor_rollout_20260512_025814.md) | 2026-05-12 | 第一版 GRU 全量 sensor 两阶段训练，Stage 1 teacher forcing + Stage 2 rollout finetune | Stage 1: 0.276894; Stage 2: 0.309171 | Stage 1 best: 0.426362; Stage 2 best: 0.389656 | 尚未跑独立 closed-loop eval；Stage 2 验证已是 128 步 rollout，闭环准确率偏低 | 完整跑完 50 + 20 epoch，显存约 30GB |
| [gru_full_sensor_seq256_rollout_20260512_045706](gru_full_sensor_seq256_rollout_20260512_045706.md) | 2026-05-12 | 将历史窗口从 1024 步改为 256 步；Stage 1 batch 512，Stage 2 batch 128 | Stage 1: 0.108782; Stage 2: 0.089680 | Stage 1 best: 0.626389; Stage 2 best: 0.749714 | 尚未跑独立 closed-loop eval；Stage 2 验证为 128 步 rollout，显著优于 1024 窗口 | 完整跑完 50 + 20 epoch，显存约 26-27GB；Stage 2 建议早停 |
