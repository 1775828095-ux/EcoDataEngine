# 通用模型需求匹配与自动处理框架

日期：2026-10-02。本阶段调整架构和验证复用，不新增研究算法。

## A. 修改哪些现有部分

- `models/scenario.py`：从固定 PM2.5 场景改为通用编排入口。保留 `recognise_inputs / suggest_roles / plan_scenario / execute_scenario` 原 API，原案例和页面继续调用。
- `interface/model_page.py`：所有模型走同一上传、匹配、计划、执行、结果界面；不再对非 CMAQ 模型直接返回。高级选项从场景配置读取，下载文件从实际结果读取。
- `profiles/cmaq.json`：明确选择 `cmaq_auxiliary` Adapter，声明允许派生 wind、仅记录边界依赖及高级参数；输出范围严格标记为标准化辅助数据。
- `recognition.py`：修复 Pandas 时间可能采用微秒精度，而间隔计算误按纳秒解释的问题；统一精度再计算，不改变转换算法。
- `examples/run_cmaq_case.py`：增加可指定新输出目录的选项，避免覆盖旧实验结果。

## B. 直接复用什么

唯一模型库仍是 `RequirementStore`：23 个已有官方需求索引，加用户保存的配置覆盖。没有新建第二个数据库。

继续复用识别模块、字段单位解析、CSV/Excel/NetCDF/HDF5/栅格/矢量读取、StructureConverter、ConversionTaskEngine、时间模块、重网格模块、单位换算、加权融合、残差融合、回读质量检查、ZIP 打包及 Windows 启动脚本。QP 系列和原算法实验代码未改动。

## C. 新增模块与职责

```text
RequirementStore（已有模型配置 + 用户配置）
    ↓
ModelSchema                          contracts.py
    ↓
ModelRequirementMatcher              matching.py
    ↓
ProcessingPlanner                    planning.py
    ↓
配置选择 ModelAdapter                adapters/__init__.py
    ├─ standardized：通用预处理        adapters/standardized.py
    └─ cmaq_auxiliary：专用工程规则     adapters/cmaq.py
    ↓
已有 ConversionTaskEngine / Pipeline
    ↓
实际输出回读 + 输出需求复查 + 完成状态
```

通用核心没有按模型名称分支执行处理。Adapter 注册表按配置键选实现；CMAQ 的 PM2.5 语义、平均窗口、风矢量合成、负值策略和逐时段站点残差融合只存在 CMAQ Adapter。

### ModelSchema

从现有配置生成同一执行视图，包含模型名称、版本、类别、参考来源、输入角色、必需/可选字段、标准名和别名、单位、类型、时间/空间要求、CRS、维度、格式、依赖、`allow_derived`、输出格式和 Adapter。

两种已有配置兼容：

1. 有 `preparation_scenario`：读取明确的工程预处理范围。
2. 只有 `inputs`：生成通用准备视图。仅类别索引且未明确变量/单位的输入不能冒充字段级适配；提示补充配置，不直接断言物理数据缺失。

CRS、时间等自然语言说明保留为说明，不擅自编译为算法参数。可解析的 `temporal_resolution` 和数值 `spatial_resolution_km` 可以生成处理要求；其他处理需 `processing` 明确配置。

### 匹配与计划

- `satisfied`：已满足。
- `convertible`：可进行转换/处理。另有 `can_process` 和原因；单位未知、映射歧义或缺少执行能力时显示“待确认/当前无法处理”，不能假称可执行。
- `missing`：缺少必需数据/字段；不造变量、不补坐标。

逐字段匹配标准名/别名；唯一候选自动推荐，多个同等候选要求确认。相同文件可以服务多个角色，不按文件分配次数限制。

格式不符不是变量缺失；真实格式和结构转换由已有 Pipeline 执行。表格有坐标且目标维度要求规则网格时，自动选择现有 `table_to_grid`，不把稀疏站点表伪装为完整网格。

依赖按拓扑排序；引用不存在或循环依赖被拒绝。前置角色缺失/失败时，不继续执行依赖它的输出。可独立处理的角色可输出 `prepared_<role>`；不完整任务带缺失/失败报告，禁止声称生成完整模型输入。CMAQ 联合融合场景要求四类必要数据完整，不提供不完整的联合文件。

### 配置示例：用户水质准备契约

这是配置表达方式，不是 WASP 官方原生文件字段规范。

```json
{
  "name": "water_quality",
  "standard_name": "water_quality",
  "unit": "variable-specific",
  "data_type": "continuous",
  "required": true,
  "aliases": [],
  "formats": ["CSV", "NetCDF"],
  "dimensions": ["record"],
  "temporal_resolution": null,
  "spatial_resolution_km": null,
  "depends_on": [],
  "fields": [
    {"name": "NH3", "standard_name": "ammonia", "aliases": ["氨氮"], "unit": "mg/L", "data_type": "continuous"},
    {"name": "COD", "aliases": ["化学需氧量"], "unit": "mg/L", "data_type": "continuous"}
  ]
}
```

该输入嵌入完整 Model Schema 的 `inputs`，可使用已有 `RequirementStore.save()` 保存。完整外层模板由 `RequirementStore.template()` 提供。新字段不是硬编码在页面里。

通用处理设置：`processing.structure_operation`、`station_field`、`target_time_frequency`、`target_spatial_resolution_km`、`spatial_method`。表格时间处理需要可靠站点/分组字段；现有宽表通道只允许瞬时值内插，平均/累计量不会偷偷当瞬时值处理。

融合派生角色：声明一个连续物理变量、`allow_derived: true`，并配置：

```json
{"processing": {"fusion": {"method": "weighted", "sources": ["source_a", "source_b"]}}}
```

来源角色自动加入依赖；实际调用已有加权融合/标准化流程。`residual` 同样使用已有接口，其输入必须符合该算法的背景场与站点要求。不虚构 uncertainty、观测可信度或物理变量标签。

### 输出状态与信息追溯

质量报告包含：

| 状态 | 含义 |
| --- | --- |
| `standardized_preprocessing_completed` | 本配置范围的预处理及输出复查完成 |
| `model_format_matched` | 满足明确声明的输出格式契约；不等于模型原生读取验收 |
| `native_model_input_verified` | 由 Adapter 的真实版本/读取验证钩子确认；当前实现均为 false |
| `complete_model_input` | 需求完整且原生验证通过才可为 true；当前为 false |

CMAQ 配置为 `standardized_only`，即使生成 NetCDF，`model_format_matched` 也不会称为 I/O API 格式完成。

识别报告保留原文件名、SHA256、变量属性、维度、类型、单位、CRS、时间及来源元数据。通用预处理记录字段/单位换算参数、Pipeline 历史、输入输出维度及信息损失。CMAQ 记录共同时间选取、空间重采样、残差融合和负值修正的损失说明，保留背景、未截断融合、修正标记；没有独立真值不报告 RMSE。

## D. CMAQ 第一个真实工程案例

仍使用 `examples/cmaq_case/input/` 的公开真实小样本：AirNow PM2.5 站点 CSV、NASA GEOS-CF 气象与背景 NetCDF、USGS 3DEP DEM GeoTIFF。没有重新下载、伪造物理数据或训练新模型。

执行：选择 CMAQ → 读取 Schema → 识别四文件 → 推荐四角色 → 检查字段/单位/时间语义与范围 → 生成计划 → Adapter 调用已有重网格/重投影与残差融合 → 回读输出 → 按四类辅助需求再检查。

本次新结果位于 `examples/cmaq_case/output_generic_framework/`，旧 `output/` 未覆盖：

- `input_analysis.json`
- `processing_plan.json`
- `fused_PM25.nc`
- `model_input.nc`
- `quality_report.json`

`model_input.nc` 这个兼容旧案例的文件名不代表已获得可运行的原生 CMAQ 文件；文件属性和报告都标明辅助数据。

复现命令（在项目目录）：

```powershell
.\.venv\Scripts\python.exe examples/run_cmaq_case.py --output examples/cmaq_case/output_generic_framework
```

页面操作：启动软件 → “模型数据准备” → CMAQ → 上传四文件 → 站点时间语义确认 mean → 检查角色与计划 → 执行 → 看完成状态和下载报告。当前案例原始融合有 30 个负值，默认明确截为 0 并保留原值/标记；切换“阻止输出”会拒绝生成文件。

仍缺原生 CMAQ：具体机制的三维物种初始场、垂直网格、真实边界/排放、完整 WRF/MCIP 气象、I/O API 维度/TFLAG/网格属性及对应版本工具读取验收。地形/气象是配套对齐，未参与复杂同化权重。

## E. 如何扩展模型而不重写系统

23 个配置继续共用库；CMAQ、CALPUFF、WRF、SWAT+、WASP、MODFLOW 6、HEC-RAS 均已测试能进入同一 ModelSchema。除 CMAQ 辅助场景外，不声称已跑通这些真实模型的原生适配。

1. 已有通用能力可满足的新需求：增加/编辑 Schema 的字段、单位、维度、依赖和处理设置，直接使用 `standardized`；无需改页面、匹配器或编排入口。
2. 有特殊原生格式/物理约束：增加 Adapter，在注册表绑定配置键；保留通用匹配、计划和回读检查。
3. 升级到“原生已验证”：不仅改状态，必须实现真实对应版本读取验证，并保存证据；仅给配置加 `verified_sample` 不能让当前 Adapter 输出验证成功。

当前限制：自然语言索引尚不能自动生成完整模型参数；通用 Adapter 未接入任意目标 CRS 重投影的配置执行路径，发现不匹配会说明需先使用已有转换任务，CMAQ DEM 专用重投影仍可执行。所有格式/模型组合不宣称已验证。映射歧义需通过明确别名/输入确认解决；新增原生 Adapter 仍需真实样例测试。这些属于尚未完成范围，不是已有成果。
