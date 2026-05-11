# 110ms 固定网格合并规则

这一页描述当前项目中已经确定的主流程规则。对于这个 skill，这些规则应按固定标准理解。

当前规则已经从“保留原始 sensor 时间点并插入 action 行”调整为“统一投影到 `110ms` 固定时间网格”。

## 1. 数据检查结论

对当前 `dataset/` 做过只读抽样和全量时间统计：

- `apc`：`1510` 个文件，约 `2299100` 行，采样间隔中位数约 `109ms`，常见间隔是 `109ms`、`100ms`、`110ms`
- `livedata`：`35` 个文件，约 `991006` 行，采样间隔中位数约 `790ms`，常见间隔约 `390ms` 到 `936ms`
- `log`：`40` 个文件，约 `485408` 行，存在大量毫秒级连续 action
- 固定 `110ms` 网格内经常出现多个 action，同一 `io_id` 在同一网格内也会重复出现

因此：

- 固定为 `110ms` 网格是可行的，且与 `apc` 的主采样频率接近
- `livedata` 必须被上采样到 `110ms`
- action 不能再通过“插入额外行”处理，必须归并到对应网格
- 同一网格内 action 聚合是必要规则，不是边缘情况
- `livedata` 通常只覆盖白天约 `12` 小时，不应把没有 `livedata` 的夜间理解成设备关闭
- `20260318` 是当前数据集的开机首日：`livedata` 第一条为 `2026-03-18 11:14:23.321`，`log` 第一条为 `2026-03-18 11:14:44.035`，`apc` 最早为 `2026-03-18 08:53:39.088`
- 当前项目口径下，`20260318 00:00:00.000` 到首条 `livedata` 之前仍建立 `110ms` 网格，但视为初始化阶段：sensor 使用 `sensor_default`，`state_*` 使用 `action_default.xlsx` 初始化并延续，`mask_* = 0`
- `20260406` 是空日，没有 action 和 sensor；如果没有关机或 reset 证据，`20260407` 的 action/state 应从 `20260405` 继续继承
- 远离 action 的 sensor 变化统计整体很小，但不是严格数学常数；建模上可以把“无 action 时状态基本不变”作为工程假设，同时用 `mask_*` 标明没有真实 sensor

## 2. 总体流程

合并顺序是：

1. 按日历日期连续遍历数据区间，包括没有文件、没有 action、没有 sensor 的空日
2. 每天建立完整自然日 `24h` 固定 `110ms` 时间网格
3. 把 `apc` 和 `livedata` 都恢复成绝对时间
4. 把 sensor 值投影或插值到 `110ms` 网格
5. 把没有真实 sensor 的网格补成可输入模型的数值，同时把对应 `mask_*` 置为 `0`
6. 把 `log` action 按 `110ms` 网格归并
7. 在每个网格行上展开全部 `evt_*` 与 `state_*`

最终输出的每一行都必须是一个固定网格点，不再因为 action 单独插入额外 timestamp 行。

## 3. 时间网格

固定网格步长：

`grid_step_ms = 110`

推荐网格锚点：

- 按自然日 `00:00:00.000` 作为当天 `grid_origin`
- 第 `k` 个网格时间为 `grid_ts_ms = day_start_ms + k * 110`
- 每个网格代表半开区间 `[grid_ts_ms, grid_ts_ms + 110)`
- 每个自然日覆盖 `[day_start, next_day_start)`，约 `78.5` 万个网格点
- 即使某一天没有 log、apc、livedata，也不能在状态机上跳过这一天；它仍然负责把上一天的 `state_*` 和 carried sensor 传递到下一天
- 当前数据集从 `20260318 00:00:00.000` 开始建模，不因为首条 `livedata` 在 `11:14:23.321` 才出现而移动当天 `grid_origin`

action 归格时使用：

`grid_id = floor((action_ts_ms - day_start_ms) / 110)`

如果 action 正好落在边界上，归入右侧新网格。

## 4. sensor 时间处理

### `apc`

`apc` 的 `Time` 是相对偏移，必须先换算成绝对时间：

`ts_abs = Process Start Time + Time`

`apc` 的真实采样间隔接近 `110ms`，但不是严格固定，存在 `100ms`、`109ms`、`110ms` 和少量更大间隔。因此不能直接把原始 `apc` 行当作最终时间轴，必须投影到固定网格。

### `livedata`

`livedata` 原本就是绝对时间，直接使用。

`livedata` 明显低于 `110ms` 密度，必须在非 `apc` 覆盖区域上采样到固定网格。

## 5. sensor 补值规则

### 优先级

sensor 来源优先级：

1. 如果网格点落在有效 `apc` 连续片段内，使用 `apc`
2. 否则，如果网格点落在有效 `livedata` 连续片段内，使用 `livedata`
3. 否则，如果历史上已经有过可信真实 sensor，使用最近一次可信真实 sensor 做因果前向延续，来源记为 `carried_sensor`
4. 否则，使用 sensor 默认值，来源记为 `sensor_default`

`apc` 覆盖区内不使用 `livedata` 补值。`livedata` 只负责非 `apc` 覆盖区域。

### 连续片段

不要跨长缺口做线性插值。

推荐阈值：

- `apc` 相邻原始点间隔 `> 250ms` 时，切成两个 `apc` 片段
- `livedata` 相邻原始点间隔 `> 2000ms` 时，切成两个 `livedata` 片段

理由：

- 当前 `apc` 间隔 p99 约 `156ms`，`250ms` 能覆盖正常抖动和少量丢点，同时避免跨过较长缺口造假
- 当前 `livedata` 间隔 p99 约 `1435ms`，`2000ms` 可覆盖大部分正常采集，同时避免跨分钟或跨小时缺口造假

### 数值补值

数值补值和训练有效性要分开理解：

- sensor 数值列必须尽量填满，保证模型输入矩阵形状稳定
- `mask_*` 或有效性标记决定这个 sensor 是否是真实可信观测
- 对无效补值点，sensor 可以有数值，但对应 `mask_*` 必须置为 `0`，训练 `loss_mask` / `valid_target` 也不能把它当真实 sensor

对普通 sensor 列：

- 网格时间点正好命中原始时间点时，直接使用原始值
- 网格时间点位于同一连续片段的两个原始点之间时，做线性插值，并保留真实 `mask_*`
- 网格时间点位于两个真实 sensor 点之间，但不满足有效插值边界时，仍允许为了输入形状做线性插值，但对应 `mask_*` 必须置为 `0`
- 网格时间点只有左侧真实 sensor 点时，做因果前向延续，来源为 `carried_sensor`，并将对应 `mask_*` 置为 `0`
- 网格时间点只有右侧真实 sensor 点、但没有任何历史真实 sensor 时，训练和推理输入不能用未来 sensor 回填，应使用 `sensor_default`，并将对应 `mask_*` 置为 `0`
- 左右两侧都没有任何真实 sensor 时，优先继承历史 carried sensor；如果历史上也没有真实 sensor，再使用 sensor 默认值，并将对应 `mask_*` 置为 `0`

### `carried_sensor` 与 `sensor_default`

`carried_sensor` 表示当前 `110ms` 网格没有真实 sensor，但此前已经出现过可信 `apc` 或 `livedata`，所以用最近一次可信真实 sensor 做前向延续。

- `carried_sensor` 是部署一致的因果填充方式
- 夜间没有 `livedata`、没有 action 的长区间，应使用 `carried_sensor` 维持输入连续
- action 发生在当天 sensor 采集结束之后，也应使用 `carried_sensor`
- `carried_sensor` 的 `mask_*` 必须为 `0`，不能作为 sensor target 或 loss

`sensor_default` 表示当前网格既没有真实 sensor，也没有任何历史真实 sensor 可继承，只能使用默认初始 sensor。

- `sensor_default` 主要发生在整个序列第一天开始、第一次开机、或从中途恢复但没有上一天 sensor checkpoint 的场景
- 对当前数据集，`20260318 00:00:00.000` 到 `2026-03-18 11:14:23.321` 首条 `livedata` 之前属于初始化阶段，应使用 `sensor_default`，即使该区间内存在早于 `livedata` 的 `apc` 文件，也不把它作为训练 target 或可信建模起点
- 不要每天重新使用 `sensor_default`
- 当前 `gas_header.xlsx` 更像 sensor 类型表，不是完整默认值表；如果没有明确的 sensor 默认值列或独立 `sensor_default.xlsx`，应使用统一默认值，并保持 `mask_* = 0`
- 一旦出现第一条可信真实 sensor，后续无真实 sensor 的网格应转为 `carried_sensor`

离线核对表可以额外记录 `invalid_linear_fill` 或 `invalid_backfill` 这类填充方式，但它们不能作为部署一致的 GRU 输入依据。训练输入应优先使用 `carried_sensor` / `sensor_default` 这两类因果数值。

### `apc` 插值边界

对某个 `110ms` 网格点 `grid_ts`，如果它被两个 `apc` 原始点夹住：

`t_left <= grid_ts <= t_right`

计算：

- `gap = t_right - t_left`
- `nearest = min(grid_ts - t_left, t_right - grid_ts)`

只有同时满足以下条件时，才使用 `apc` 线性插值：

- `gap <= 250ms`
- `nearest <= 125ms`

插值公式：

`sensor(grid_ts) = sensor_left + (grid_ts - t_left) / (t_right - t_left) * (sensor_right - sensor_left)`

如果 `grid_ts` 距离最近 `apc` 点约 `50ms`，且左右 `apc` 点属于同一连续片段，则这是正常情况，应使用线性插值。

如果不满足 `apc` 有效插值条件，可以继续用左右 `apc` 点做离线数值线性补值，但该补值不算真实 sensor，对应 `mask_*` 必须置为 `0`。训练和推理输入中，如果缺少左侧 `apc` 点，应退到历史 `carried_sensor` 或 `sensor_default`；如果缺少右侧 `apc` 点，应使用左侧最近可信真实 sensor 做 `carried_sensor`。

无效 `apc` 数值补值只用于维持输入形状，不参与 sensor target 或 loss。

### `livedata` 插值到 `110ms`

`livedata` 不要先插成中间频率再二次插值。应直接从 raw `livedata` 插值到最终 `110ms` 网格。

对某个不在有效 `apc` 片段内的 `110ms` 网格点 `grid_ts`，找到夹住它的两个 `livedata` 原始点：

`t_left <= grid_ts <= t_right`

计算：

`gap = t_right - t_left`

只有满足以下条件时，才使用 `livedata` 线性插值：

- `gap <= 2000ms`

插值公式：

`sensor(grid_ts) = sensor_left + (grid_ts - t_left) / (t_right - t_left) * (sensor_right - sensor_left)`

如果 `gap > 2000ms`，仍允许为了输入形状跨这个缺口做数值线性补值，但中间的 `110ms` 网格必须标为无效 sensor，对应 `mask_*` 必须置为 `0`，不能参与 sensor target 或 loss。

如果 action 发生在 sensor 采集之前，左侧没有真实 sensor，训练和推理输入不能用右侧未来真实 sensor 回填；应使用上一天继承下来的 `carried_sensor`，如果也没有历史 sensor checkpoint，则使用 `sensor_default`，并将对应 `mask_*` 置为 `0`。

如果 action 发生在 sensor 采集之后，右侧没有真实 sensor，应使用左侧最近可信真实 sensor 做 `carried_sensor`，并将对应 `mask_*` 置为 `0`。

最终优先级始终是：

`apc_grid` 有效插值 > `livedata_grid` 有效插值 > `carried_sensor` > `sensor_default`

对 `mask_*` 列：

- 不做线性插值
- 有效 sensor 插值点使用左侧最近真实点的 `mask_*`
- 如果左侧没有真实点，不能为了训练输入使用未来 `mask_*` 伪装成有效观测
- 无效线性补值、`carried_sensor`、`sensor_default` 的对应 `mask_*` 必须置为 `0`

对没有可信 sensor 支撑的网格：

- 可以为了保持矩阵形状填 sensor 数值，包括离线无效线性补值、`carried_sensor` 或 `sensor_default`
- 不应把这些补出来的值当作真实 sensor
- 对应 `mask_*` 或有效性标记必须置为无效
- 训练 target 和 loss 不应使用这些无效 sensor 点

## 6. action 网格聚合

`log` 中每一行代表一次 action：

- `timestamp`
- `io_id`
- `io_value`

所有 action 都归并到固定 `110ms` 网格中，不再插入独立 action 行。

聚合规则：

- 同一网格内不同 `io_id` 的 action 要叠加保留
- 同一网格内同一 `io_id` 出现多次 action 时，只取该网格内最后发生的一条作为该 `io_id` 的最终状态
- `evt_{id}` 只表示该网格内该 `io_id` 是否发生过 action，发生过则为 `1`
- `state_{id}` 使用该网格内该 `io_id` 最后一次 action 的 `io_value`

同一网格内 action 的先后顺序只用于确定同一 `io_id` 的最后状态。不同 `io_id` 之间不需要排序后展开成多行。

## 7. `evt_*` 与 `state_*`

每个全局 `io_id` 都展开两列：

- `evt_{id}`：该网格内该 IO 是否发生 action
- `state_{id}`：该网格结束后该 IO 的持续状态

规则如下：

- `evt_*` 是网格级瞬时事件，发生为 `1`，否则为 `0`
- `state_*` 不是只在 action 当下有效，而是会持续继承
- 某个 `io_id` 在当前网格发生 action 后，从当前网格开始使用新 `state_*`
- 同一网格内同一 `io_id` 多次 action 时，只保留最后一次 `io_value`
- 从未出现过的 `io_id`，初始状态按默认值；没有默认值时按 `0`
- 状态既跨网格继承，也跨天继承
- 整个数据集第一天开始时，`state_*` 从 `dataset/action_default.xlsx` 初始化
- 对当前数据集，`20260318 00:00:00.000` 到首条 `livedata` 之前，`state_*` 使用 `action_default.xlsx` 的初始状态并按网格延续；如果该区间出现 action，则仍按 action 更新后续 `state_*`
- 如果从中途恢复训练或推理，并且存在上一天状态 checkpoint，应优先使用 checkpoint；只有没有 checkpoint 时才回退到 `action_default.xlsx`
- 空日不代表设备关闭。没有 shutdown/reset 证据时，空日的 `state_*` 全部从上一天持续继承到下一天

## 8. `source` 建议

固定网格后，`source` 应表达 sensor 值的来源，而不是表达是否有 action。

推荐来源：

- `apc_grid`：该网格 sensor 来自 `apc` 原始点或 `apc` 片段内插值
- `livedata_grid`：该网格 sensor 来自 `livedata` 原始点或 `livedata` 片段内插值
- `carried_sensor`：该网格没有真实 sensor，用最近一次可信真实 sensor 做因果前向延续
- `sensor_default`：该网格没有真实 sensor，也没有历史 sensor 可继承，只能使用默认初始 sensor
- `missing_sensor`：可作为无真实 sensor 的总称，但训练表中更推荐拆成 `carried_sensor` 与 `sensor_default`

是否有 action 应由 `evt_*`、`has_action` 或 `action_count` 表达，不应再通过 `logapc`、`loglivedata`、`logonly` 这类 source 名称表达。

如果下游仍依赖旧的 `source_code`，需要同步扩展映射表，不要继续假设 `source_code` 永远只有 `apc/livedata/logapc/loglivedata` 四类。
