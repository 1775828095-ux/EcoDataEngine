# WASP 标准数据包主线

普通用户流程已经收敛为：选择 WASP → 上传一个或多个原始数据文件 → 自动分析并生成方案 → 查看自动角色与可生成文件 → 下载 `wasp_package.zip`。

当前输出是可审查的 WASP 上游标准数据，不是 WIF/INP。系统只生成有真实数据支撑的文件；缺失的数据不会产生空壳。

## Narragansett 真实案例结果

- USGS `01113895_daily_2009.rdb` → `tributary_flow.csv`，365 条日均流量，`ft^3/s` 可追溯换算为 `m^3/s`，保留原值、参数代码和质量码。
- NBC 2009 nutrient workbook → `tributary_concentration.csv`，只把 `Waterbody=RIVER` 的记录作为河流浓度候选；保留原始受检出限影响的文本和数值化结果。
- NOAA ISD 2009 → `environmental_forcing.csv`，只输出源文件实际存在且单位明确的气象变量。
- RI DEM fixed-site monitoring → `observations.csv`，作为水质观测；不会自动冒充 initial/boundary condition 或同化数据。
- 当前真实集合没有具备同来源、同位置、同时间且 QA/单位合格的 WWTP 流量与浓度配对，因此不生成 `point_source_load.csv`。
- 没有可靠站点到 segment 映射，`mapping.json` 标记 `NEEDS_CONFIRMATION`，但其他标准化结果仍可生成。

## 状态含义

- `raw_data_recognized`：输入已被 Reader 读取并识别。
- `standardized_preprocessing`：实际生成的数据文件完成单位/字段/时间标准化，没有失败输入。
- `wasp_role_matched`：至少一份输入有足够证据映射到 WASP 数据角色。
- `wasp_data_package_generated`：数据包和质量报告已生成。

开发报告可以保留原生适配研究状态，但普通流程不展示 `model_format_matched` 或 `native_model_input_verified`。
