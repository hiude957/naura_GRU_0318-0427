# <run_id>

## 目标

本次调参想验证什么。

## 数据

- 日期范围：
- cache 版本：
- 是否包含 `20260318` 初始化窗口：
- 是否包含空日：

## 配置

- 模型参数：
- 训练参数：
- 数据处理参数：

## 命令

```bash
UV_CACHE_DIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-cache \
TMPDIR=/vepfs-mlp2/mlp-public/250259/lyh/uv-tmp \
uv run python scripts/03_train_gru.py --config configs/train_gru_a100.yaml
```

## 结果

- train loss：
- val loss：
- accuracy：
- closed-loop：
- GPU 显存：
- 训练耗时：

## 结论

本次是否有效，问题是什么。

## 下一步

下一次要改什么参数。

