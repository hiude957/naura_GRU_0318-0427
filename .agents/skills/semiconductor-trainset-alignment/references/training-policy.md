# 训练版输出策略

这一页描述当前训练表的输出口径。与主合并流程相比，这部分更容易在后续迭代中调整。

## 1. 原始版与训练版

每天会生成两张表：

- 原始版：偏核对和追溯
- 训练版：偏模型输入

原始版保留：

- `timestamp_raw`
- `ts_ms`
- `source`
- 所有 sensor 列
- 所有 `mask_*`
- 所有 `evt_*`
- 所有 `state_*`

训练版在原始版基础上继续整理。

## 2. 当前训练版口径

旧版训练版规则是：

- 删除全部 `source=logonly` 行
- 保留 `timestamp_raw`
- 保留 `ts_ms`
- 把 `source` 字符串映射成 `source_code`
- 当前训练版要求没有空值

固定 `110ms` 网格后，训练版规则需要同步改成：

- 不再生成额外 `logonly` 行
- 每天以完整自然日 `24h` 的 `110ms` 网格作为标准时间轴，空日也要传递 `state_*` 与 carried sensor
- 当前数据集首日 `20260318` 也从 `00:00:00.000` 建训练网格；首条 `livedata` 之前的行使用 `sensor_default` 和 `action_default.xlsx` 初始化状态，`mask_* = 0`
- 保留固定网格的 `timestamp_raw` 与 `ts_ms`
- 把新的 `source` 字符串映射成新的 `source_code`
- 对 `source=carried_sensor` 或 `source=sensor_default` 的网格，不能作为有效 sensor target 或 loss 点
- 如果为了矩阵形状填补 sensor 数值，训练输入优先使用因果的 `carried_sensor`；只有没有历史真实 sensor 时才使用 `sensor_default`
- 离线核对表可以保留无效线性补值，但部署一致的 GRU 输入不能依赖未来真实 sensor 做后向填充
- 任何无效补值点的对应 `mask_*` 必须置为 `0`
- 必须保留有效性标记，不能把补出来的值当作真实 sensor
- `since_sensor` 或等价特征应在 `carried_sensor` / `sensor_default` 区间继续增长，让模型知道距离上一次真实 sensor 已经多久

## 3. 当前 `source` 定义

`source` 不再视为封闭集合。

旧版已有示例来源包括：

- `apc`
- `livedata`
- `logapc`
- `loglivedata`
- `logonly`

固定 `110ms` 网格后的推荐来源包括：

- `apc_grid`
- `livedata_grid`
- `carried_sensor`
- `sensor_default`
- `missing_sensor`，仅作为无真实 sensor 的总称；训练时更推荐拆分成上面两类

后续允许继续增加你自定义的规则来源。

回答时应明确：

- 原始版保留 `source` 字符串
- 训练版也需要保留新增来源
- 每新增一种 `source`，训练版就要同步分配新的 `source_code`
- 固定网格后，是否有 action 不应再靠 `source` 表达，而应由 `evt_*`、`has_action` 或 `action_count` 表达

## 4. 当前 `source_code` 示例

旧版已有的训练版映射是：

- `1 = apc`
- `2 = livedata`
- `3 = logapc`
- `4 = loglivedata`

固定 `110ms` 网格后，应同步扩展或替换映射，例如：

- `1 = apc_grid`
- `2 = livedata_grid`
- `3 = carried_sensor`
- `4 = sensor_default`

如果仍保留 `missing_sensor` 这个总称，应分配独立 code，不能和 `carried_sensor` 或 `sensor_default` 混用。

如果增加新的来源：

- 训练版必须继续保留这些来源
- `source_code` 映射表必须同步扩展
- 不应把训练版来源定义写死成永远只有 `1/2/3/4`

## 5. 回答这个部分时的建议口径

如果被问到“哪些规则可能会变”，优先回答：

- 训练版是否保留某些时间字段
- 训练版是否删除某类来源
- `source_code` 的编码表如何扩展

不要把以下内容说成待定：

- 输出对齐到固定 `110ms` 网格
- `apc` 覆盖区优先使用 `apc`，不使用 `livedata` 覆盖
- `livedata` 只在非 `apc` 区域上采样到 `110ms`
- action 按 `110ms` 网格聚合，不再插入额外 action 行
- 同一网格内同一 `io_id` 多次 action 时，只取最后一次 action 的状态
- `state_*` 的跨网格、跨天继承方式
- `carried_sensor` / `sensor_default` 的 `mask_* = 0`，不能参与 sensor target 或 loss

如果完整 `24h` 网格导致夜间无 action、无真实 sensor 的行数过大，训练阶段可以对长段 `mask=0` 且无 action 的区间做采样或压缩；但采样不能改变 `state_*` 跨天继承、`since_sensor` 增长、以及 action 附近窗口的连续性。
