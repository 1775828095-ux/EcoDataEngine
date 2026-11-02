# EcoData Engine 项目总交付报告

更新时间：2026-10-13  
本报告是后续开发的唯一总交付文档。代码、官方模型权重、测试数据和机器可读质量报告仍保留在各自目录。

## 1. 本次真正完成的功能

### 1.1 Galileo 官方多组输入

`features.galileo_nano` 已从单组输入升级为多组输入，可在一次官方前向计算中同时使用：

- SRTM：`elevation + slope`
- Sentinel-2 RGB：`B2 + B3 + B4`
- ERA5：`temperature_2m + total_precipitation_sum`（当前只验证单时次区域代表值）

空间输入必须有相同 CRS。不同网格会先匹配到明确的参考网格，所有插值都会记录。随后调用 Galileo 官方 `Encoder`，利用官方 token 注意力和 `apply_mask_and_average_tokens_per_patch` 生成联合空间特征。

输出仍是 `galileo_embedding`，不是目标环境变量，也不是高分辨率预测。插件会阻止把 Galileo 特征冒充 `PM2.5`、`LAI` 等物理量。

### 1.2 语义感知跨尺度特征模块

新增插件：`scale.semantic_stack`。

当前规则：

| 变量性质 | 当前处理 |
|---|---|
| 连续量向更细网格匹配 | 双线性，只改变表达网格，明确不声称增加真实细节 |
| `cell_mean` / `flux` 连续量向更粗网格聚合 | 保守重网格 |
| 土地利用等分类变量向更粗网格聚合 | 各类别面积比例，不对类别编号做连续插值 |
| 分类变量向更细网格匹配 | 最近邻 one-hot 特征 |
| 总量、累计量但语义不明确 | 拒绝自动处理，避免错误求和或插值 |

输出同时包含保留原单位的对齐变量和供后续模型使用的无量纲 `feature_stack`。标准化参数写入元数据。

### 1.3 质量感知数值融合模块

新增插件：`fusion.quality_aware`。

处理过程：

1. 每个来源必须提供同一物理变量、相同单位和相同 CRS。
2. 只用 `train` 观测拟合每个来源的线性偏差校正：`y = a × source + b`。
3. 只用 `validation` 观测计算校正后 RMSE。
4. 按验证误差倒数平方生成来源权重。
5. 输出加权目标数值场和 `fusion_uncertainty` 来源分歧场。
6. `test` 观测不参与校正和权重选择，只用于最终独立检查。

没有可靠训练和验证观测时，插件直接拒绝自动判定数据源优劣，不会生成伪造权重。

### 1.4 自动注册和选择

三个插件均接入现有 `AlgorithmRegistry` 和 `AlgorithmSelector`：

- `features.galileo_nano`
- `scale.semantic_stack`
- `fusion.quality_aware`

输入变量、语义、单位、CRS、观测划分或官方输入组不满足条件时，会在兼容性检查阶段被排除。

## 2. 已接入算法及官方源码

| 算法/模块 | 地址 | 当前状态 |
|---|---|---|
| Galileo | https://github.com/nasaharvest/galileo | **已接入官方 nano 编码器，多组特征提取已运行验证** |
| NeuralOperator FNO | https://github.com/neuraloperator/neuraloperator | 已有项目能力，保留 |
| Thera | https://github.com/prs-eth/thera | 计划接入；官方环境和环境变量适配尚未完成 |
| Sentinel-5P自监督超分 | https://github.com/hyamomar/Sentinel-5P-Super-Resolution | 研究状态；许可证和固定环境仍不明确 |
| DIIN | https://github.com/Song-1205/DIIN | 暂缓；官方仓库仍无可运行代码 |
| Granite WxC | https://github.com/IBM/granite-wxc | 计划做气温/风场专用验证，尚未安装大型权重 |
| PRIMER | https://github.com/sunmoumou1/PRIMER | 仅研究其降水偏差与融合机制；许可和GPU条件未解决 |

本次新增的跨尺度和质量感知模块是平台适配逻辑，不冒充上述论文模型。

## 3. 算法覆盖矩阵

| 数据 | 当前可执行路线 | 当前限制 |
|---|---|---|
| DEM、坡度 | Galileo SRTM特征；保守/双线性尺度匹配 | Galileo特征不等于DEM重构结果 |
| Sentinel-2 RGB | Galileo官方特征编码 | 当前只验证RGB组，不覆盖全部波段任务 |
| ERA5温度+降水 | Galileo官方时间特征（单时次代表值） | 空间气象场和完整时间序列尚未接入Galileo接口 |
| 30m土地利用 | 类别面积比例或最近邻one-hot | 不使用连续插值 |
| FVC、LAI、NPP | 按连续量和明确空间语义做尺度匹配 | 尚无经验证的学习型重构器，不强行使用Galileo |
| 气温 | 语义尺度匹配；现有传统方法/FNO/UNO | Granite WxC尚未实际接入 |
| 降水 | 守恒语义处理、现有传统方法 | PRIMER尚未接入；累计量必须明确时间/空间语义 |
| 多个同变量模拟背景场+监测站 | 质量感知偏差校正与动态权重融合 | 需要明确训练/验证观测；无观测时拒绝自动判优 |
| PM2.5背景场+气象/地形/遥感 | 现有FNO/UNO及新特征准备接口 | Galileo特征尚未连接到经过训练和验证的PM2.5预测头 |

## 4. 实际运行结果和质量检查

### 4.1 实验性质

本次使用小型**合成数据**验证接口和数据隔离，因为尚未建立同时满足 Galileo 官方变量组、目标变量真值及严格训练/验证划分的真实联合样本。合成结果不能作为论文精度或真实环境效果证据。

### 4.2 Galileo多组测试

- 输入：8×8 SRTM高程/坡度、4×4 Sentinel-2 RGB、单时次ERA5温度/降水。
- 尺度匹配：Sentinel-2 RGB记录为双线性特征对齐；SRTM保持原网格。
- 输出：4×4×128 `float32` Galileo联合特征。
- 有限值、单位、CRS、文件重读：PASS。
- 环境变量预测精度：未测试，因为没有预测头和独立真值。

### 4.3 跨尺度测试

- 输入：8×8土地利用、4×4 LAI、2×2气温。
- 目标：4×4分析网格。
- 土地利用：生成2类面积比例；比例和最大误差为 `0.0`。
- LAI：保守聚合。
- 气温：双线性匹配，但报告明确其真实信息仍来源于粗网格。
- `feature_stack`：4个特征，全部有限，PASS。

### 4.4 质量感知融合测试

- 输入：两个带不同系统偏差的合成PM2.5背景场。
- 观测划分：16条训练、8条验证、6条最终测试。
- 权重：背景A `0.90465`，背景B `0.09535`。
- 验证RMSE：`0.86243`，但验证集参与选权，不能作为最终独立精度。
- 独立测试RMSE：`0.92703`；MAE `0.79248`；Bias `0.36732`。
- 最终测试观测参与拟合或选权：`false`。
- 非负检查、有限值、单位、CRS：PASS。

这些数值只证明模块按预期工作，不证明真实PM2.5融合能达到相同误差。

### 4.5 自动测试

- 新增核心相关测试：14项通过。
- 全量回归：**1038 passed，4 skipped，0 failed，0 errors**。
- 全量耗时：134.124秒。
- 固定 `.pytest_tmp` 在本机存在Windows权限问题；本次使用独立 `--basetemp` 完成回归，不修改业务代码掩盖该环境问题。

## 5. 完成状态

### 已实现并执行验证

- Galileo官方nano权重加载。
- SRTM + Sentinel-2 RGB + ERA5多组联合特征提取。
- 不同空间网格的显式特征尺度匹配及记录。
- 分类面积比例、连续量保守/双线性语义路由。
- 跨尺度无量纲特征栈。
- 训练校正、验证选权、测试封存的质量感知融合。
- 融合数值场、不确定性场、NetCDF输出和质量门。
- Registry注册、自动筛选和全量回归。

### 仅完成接口或基础版本

- Galileo ERA5仅验证单时次区域代表值，尚不是空间气象场/长时间序列接口。
- 跨尺度模块是可解释的基础特征准备，不是学习型跨尺度预测网络。
- 质量感知模块采用线性去偏和验证误差权重，是可靠基线，不是最终论文创新算法。
- Galileo特征已联合生成，但还没有经过训练的任务预测头。

### 尚未实现

- 真实FVC、LAI、NPP学习型精细重构。
- Galileo特征对真实目标变量的增益验证。
- 真实多源数据端到端实验。
- Thera + 自监督正式接入。
- Granite WxC气象专用权重运行。
- PRIMER降水模型运行。
- 无观测条件下可证明的数据源质量排序。

## 6. 当前问题与下一阶段建议

1. **优先准备真实联合样本。** 应选择同一时间、区域和CRS的S2/SRTM/ERA5及目标观测，固定训练、验证和最终测试，再判断Galileo特征是否有实际增益。
2. **建立轻量任务头。** 先用简单、可审计的线性或小型MLP预测头比较“无Galileo特征”和“有Galileo特征”，只有独立验证改善后才保留。
3. **扩展时间接口。** Galileo ERA5时间序列需要严格对齐月份、时间窗口和官方输入维度，不应把空间气象场静默求平均。
4. **质量权重增加不确定性。** 后续可加入bootstrap置信区间和权重稳定性检查，避免少量验证站导致权重极端化。
5. **之后再接高级模型。** 先完成真实Galileo基线，再依次评估Thera、Granite WxC和PRIMER，不并行下载大型权重。

## 7. 关键路径

### 代码

- `src/ecodata/algorithms/galileo.py`
- `src/ecodata/algorithms/cross_scale.py`
- `src/ecodata/algorithms/quality_aware.py`
- `src/ecodata/algorithms/plugins.py`
- `src/ecodata/algorithms/scale.py`

### 测试

- `tests/test_galileo_plugin.py`
- `tests/test_cross_scale_stack.py`
- `tests/test_quality_aware_fusion.py`

### 可复现实验

- `examples/advanced_algorithms/multisource_fusion_demo.py`

### 输出

- `output/multisource_framework_demo/01_galileo_multigroup_features.nc`
- `output/multisource_framework_demo/02_cross_scale_features.nc`
- `output/multisource_framework_demo/03_quality_aware_fused_pm25.nc`
- `output/multisource_framework_demo/validation_report.json`
- `output/multisource_framework_demo/full_pytest.xml`

### 官方源码与权重

- `external/galileo/`
- 官方版本：`0f0b5b95ac81acef4b74cf4686dc877202f4541b`

---

## 8. 双层数据质量控制（2026-10-15）

### 8.1 原有能力与本次补充

原有 `PostConversionValidator`、导出回读检查和模型兼容性检查继续保留，负责转换前后信息损失、文件可读性、字段/单位/CRS、守恒和模型需求复检。本次没有重写这些逻辑，而是补上两个此前缺失的控制点：

1. 上传后、算法选择前的 `RawInputQualityGate`；
2. 算法输出后、模型适配前的 `ResultQualityGate`。

质量检查只判断数据能否安全进入下一阶段；偏差校正负责修正有证据支持的系统偏差；算法精度评价必须使用独立验证数据。三者保持分离。

### 8.2 第一道门：原始输入

- 完整性：空变量/空维度、关键变量缺失、缺失率、全缺失。
- 有效性：Inf、疑似乱码、3×IQR疑似异常值；异常值只提示，不自动删除。
- 时间：无法解析、重复时间、间隔不规则。
- 空间：坐标是否存在、经纬度范围、空间任务是否缺CRS。
- 物理：单位是否存在/符合明确目标要求、声明物理范围是否违反。
- 可信度：没有独立参考时只记录“未评价”，不擅自判定数据源优劣。

状态为 `PASS`、`WARN`、`BLOCK`。`WARN` 需要用户确认；`BLOCK` 不进入算法选择。

### 8.3 第二道门：算法或融合结果

- 检查目标变量、NaN/Inf、全无效输出、单位和CRS。
- 按语义检查分类值集合和非负物理约束。
- 有独立验证数据时计算RMSE、MAE、Bias；test角色不能用于选型。
- 模型兼容性复检失败时阻止生成“完整模型输入”。
- 无独立真值时明确写明只通过结构/物理检查，不形成精度结论。

现有 `PostConversionValidator` 继续承担更完整的转换信息损失、守恒、回读及模型字段检查；新门控负责统一算法框架中的准入和阻断。

### 8.4 算法筛选联动

- 输入为 `BLOCK`：所有算法标记为不兼容，候选列表为空。
- 输入为 `WARN`：警告写入选择理由；存在可靠观测且插件适用时，提高质量感知校正方案的优先级。
- 输出为 `BLOCK`：不交付结果；可按候选顺序尝试其他兼容算法。
- 回退最多尝试3个候选，每个候选只执行一次，并保存失败原因，避免无限循环。
- 每次运行报告包含 `quality_control.input`、`quality_control.output` 和决策记录。

### 8.5 界面与JSON

数据转换页和模型数据准备页均显示“通过 / 警告 / 阻止”。警告不会静默修改原数据；模型页存在阻止项时禁用执行。处理报告保存两道质量检查的项目、阈值、异常位置、处理决定和是否使用独立真值。

示例：`output/two_stage_quality_control/quality_control_example.json`。

### 8.6 测试结果

新增测试覆盖：正常数据、部分缺失、全缺失、Inf、错误单位、错误坐标、负值物理约束、模型需求复检、独立验证指标和算法筛选阻断。

- 双层门及算法框架：14项通过。
- 界面、模型场景、五模型场景和双层门组合回归：29项通过。
- 全量回归：**1045 passed，4 skipped，0 failed，0 errors**。若可选ESMF/xESMF依赖未安装，保持原有跳过策略。

### 8.7 尚未解决

- 通用单位等价换算仍由现有单位引擎/转换流程执行；质量门只在目标单位明确时判断是否一致，不通过数值范围猜单位。
- 物理合理范围只有在模型Schema、变量配置或数据属性明确给出时才能严格阻断，不能为所有环境变量硬编码统一阈值。
- 没有独立观测时不能评价数据源准确性，也不能据此自动确定融合权重。

### 8.8 关键路径

- `src/ecodata/evaluation/quality_gates.py`
- `src/ecodata/algorithms/pipeline.py`
- `src/ecodata/algorithms/router.py`
- `src/ecodata/interface/task_page.py`
- `src/ecodata/interface/model_page.py`
- `tests/test_two_stage_quality_control.py`

