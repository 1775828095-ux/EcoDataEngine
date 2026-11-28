# EcoData Engine 项目总交付报告

更新时间：2026-11-04
本报告是后续开发的唯一总交付文档。代码、官方模型权重、测试数据和机器可读质量报告仍保留在各自目录。

最新工程集成与稳定性诊断记录见文末的2026-11-04各节；本次GitHub同步及本机复验见末节。下文第35章等内容保留为历史实验，不代表当前全部能力的完成状态。

最新交付见第35章：已取得LGHAP 2019年6个官方日文件，完成1.5/2/3倍人工退化重构。现有Registry/Pipeline实际执行4种估计器×2种约束方案，验证集穷举后自动选择“双三次＋回聚合一致性”，在冻结测试产品上优于固定双线性及最佳单算法；未证明优于人工固定同一最优流程。数据为融合产品参考，不是独立实测真值。第34章五站PM2.5诊断及所有历史结果保留。

重要更正：第30章前版V1/V3的数值确实执行过，但其适配模式曾将未在34项冻结池内的IDW/RF挂在`fusion.residual`名下，不能作为严格冻结库对照。已撤回这部分合规表述，旧数值保留作历史工程记录；最终严格版本以30.9节为准。14个测试站此前已经评价，任何后续复核都不能重新称为全新盲测。

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

---

## 9. 物理语义驱动的算法能力匹配与GOLEM图搜索（2026-10-16）

### 9.1 本阶段边界与代码归属

| 类别 | 内容 | 本阶段处理 |
|---|---|---|
| 第三方开源 | GOLEM的进化图优化器、图对象、Objective、约束与官方mutation | 直接调用官方API，不复制优化器源码 |
| EcoData Engine已有 | Reader、`inspect_dataset`、`RawInputQualityGate`、Algorithm Registry、已有重网格/融合/时间处理插件 | 原样复用并补充能力元数据 |
| 本阶段新增 | `S(D)`语义提取、可配置`Phi`规则、能力目录、隔离进程适配器、语义引导mutation、A/B/C实验 | 已实现并自动测试 |
| 本项目提出的机制 | 环境数据语义到能力需求映射，以及按能力缺口和成本改变候选生成概率 | 原型已运行；科研新颖性尚未通过文献与更大实验确认 |

本阶段没有接入新模型、没有修改Raw/Result两道质量门、没有用模拟评分声称环境数据精度提高。

### 9.2 GOLEM真实复用证据

- 官方仓库：`https://github.com/aimclub/GOLEM`
- 独立环境：`.venv-golem/`
- 固定版本：`thegolem==0.5.0`
- Python要求：`>=3.10`
- 许可证：BSD-3-Clause；记录见`THIRD_PARTY_LICENSES.md`与`requirements-golem.txt`。
- 官方类真实调用：`EvoGraphOptimizer`、`GraphRequirements`、`GraphGenerationParams`、`GPAlgorithmParameters`、`Objective`、官方`base_mutations`及图约束。
- 官方API最小冒烟：`tools/run_golem_official_smoke.py`。
- EcoData适配入口：`src/ecodata/algorithms/golem_search.py`；隔离后端：`tools/run_golem_search.py`。

GOLEM运行在独立Python环境，主项目通过JSON和子进程连接，避免其依赖改变主环境。`output/semantic_golem_experiment/semantic_golem_experiment.json`保存后端名称、版本、官方类、逐次实验结果与变异轨迹；这不是固定结果或静态演示。

### 9.3 数据语义与能力需求映射

统一语义定义为：

`S(D) = (F, C, dx, dt, U, P, Q)`

- `F`：数据结构、来源与组织形式；
- `C`：连续、分类、计数等变量性质；
- `dx`：空间分辨率、CRS和像元支持域；
- `dt`：时间间隔、采样规则与统计区间；
- `U`：单位、量纲及统计含义；
- `P`：已确认物理规则；
- `Q`：缺失、异常和质量门结果。

`SemanticFeatureExtractor`优先使用现有数据检查、变量属性、CF式属性和用户确认信息。每个结论保存`value/status/evidence/reason`；无法确认写为`UNKNOWN`，冲突写入`conflicts`，不会按数值dtype猜测累计量或瞬时量。

需求映射定义为：

`R = Phi(S(D), T; K)`

其中`T`为目标尺度/变量/模型约束，`K`为`semantic_rules.json`中的可扩展规则。输出严格分为：

1. `required_capabilities`：处理图必须覆盖的能力；
2. `hard_constraints`：不得违反的确定性物理限制；
3. `soft_objectives`：可以比较优化的质量或成本目标；
4. `unknown_conditions`：因证据不足不能确定的条件；
5. `trace`：触发规则、输入证据和推导结果。

已验证的差异化场景包括：

| 场景 | 主要推导结果 |
|---|---|
| 500m LAI到30m | 连续空间尺度处理、尺度一致性检查；不承诺恢复真实30m细节 |
| 土地利用分类尺度转换 | 类别保持/类别比例；硬性禁止双线性连续插值 |
| 小时降水到日降水 | 时间累计与区间守恒；只有统计语义确认后才触发 |
| 多源同变量 | 多输入融合；有可靠观测时增加质量感知需求 |
| 单位或统计语义缺失 | `UNKNOWN`和原因进入搜索前提，不静默猜测 |
| CRS或像元支持域不确定 | 明确未知条件，空间算法不能被无条件视为可用 |
| 物理规则冲突 | 保存冲突并阻止任意选边 |

### 9.4 算法能力图与约束

`AlgorithmSpec`新增但保持向后兼容的描述字段：`capabilities`、`preconditions`、`input_contract`、`output_contract`、`verification_evidence`。`capability_catalog()`只纳入Registry中实际可执行且具有能力描述的插件；Thera、DIIN、PRIMER等未验证算法没有被伪装成可执行节点。

处理图要求：

- 对每个必需能力`k`，所选可执行节点能力并集必须覆盖`k`；
- 同时检查类别/非负/累计量等硬约束；
- 使用GOLEM的无环和无自环约束；
- 缺少能力不是立即淘汰中间图，而是作为高惩罚目标，允许进化继续补全；
- 已知的直接物理冲突在生成或约束检查阶段拒绝；
- 多输入能力可以声明并进入需求映射，但本轮数值实验采用最小链式图，尚未验证复杂多分支DAG的真实数据执行。

新增测试插件只需登记能力描述即可进入搜索，测试中没有修改GOLEM优化器或搜索适配器，验证了扩展成本主要落在插件注册和契约说明。

### 9.5 三种搜索方法的精确定义

统一目标函数为：

`J(G)=1000*(缺失能力数+物理冲突数)+估计计算成本+0.1*图复杂度`

在没有独立真值的本轮实验中，不加入预测RMSE。

- **A：GOLEM默认mutation**。调用官方add/change/drop/edge mutation和统一结构约束；物理缺口只体现在Objective。
- **B：默认mutation+语义硬约束**。候选仍由官方mutation产生，之后由确定性物理规则筛除直接冲突。
- **C：语义引导mutation**。候选产生前计算匹配、能力缺口补全和成本，按下式抽样，并保留随机探索：

`P(Aj|D,G) proportional to exp(alpha*Mj + beta*Hj - gamma*Cj)`

确定性硬冲突候选的概率为0。当前实现支持能力缺口驱动的节点增加、兼容节点替换、边重连尝试与不完整图补全；每次保存变异前后图、操作、候选、原因、能力缺口和硬约束。它改变的是候选生成概率，不是普通GOLEM完成后再做一次后验过滤。

实验参数仅为原型参数：`alpha=1, beta=2, gamma=0.25, exploration=0.1`，没有声称最优。

伪代码：

```text
S <- extract_semantics(data, confirmed_metadata)
R <- map_requirements(S, target, configured_rules)
G <- initialise_with_executable_capability_node()
repeat within equal evaluation budget:
    gap <- required_capabilities - capabilities(G)
    for each executable candidate Aj:
        if Aj violates a confirmed hard constraint: probability <- 0
        else probability <- exp(alpha*match + beta*gap_fill - gamma*cost)
    with exploration probability choose a random legal candidate
    mutate G by add / replace / reconnect
    GOLEM verifies graph structure and evaluates J(G)
return qualified graphs and complete trace
```

### 9.6 A/B/C真实实验结果

实验使用相同算法目录、目标函数、种子`11/22/33/44/55`、population=8、generation=4、timeout=30秒。结果文件：

- `output/semantic_golem_experiment/abc_results.csv`
- `output/semantic_golem_experiment/abc_summary.csv`
- `output/semantic_golem_experiment/semantic_golem_experiment.json`

下表为5次均值；`合格最佳数`为5个种子中最终最佳图合格的次数。

| 场景 | 版本 | 变异提议 | 实际评价 | 无效候选 | 合格图 | 唯一图 | 合格最佳数 | 平均耗时(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 分类尺度 | A | 66.8 | 44.4 | 0.0 | 10.8 | 25.8 | 5 | 1.080 |
| 分类尺度 | B | 73.4 | 44.0 | 5.0 | 13.0 | 25.6 | 5 | 1.065 |
| 分类尺度 | C | 75.6 | 46.2 | 0.0 | 26.0 | 28.6 | 5 | 1.151 |
| 连续多源 | A | 63.4 | 44.4 | 0.0 | 3.6 | 25.8 | 4 | 1.069 |
| 连续多源 | B | 63.4 | 44.4 | 0.0 | 3.6 | 25.8 | 4 | 1.086 |
| 连续多源 | C | 112.0 | 45.8 | 0.0 | 16.2 | 30.2 | 5 | 1.111 |

结论严格限定为图搜索原型：C在相近实际评价次数下发现了更多能力合格图，并在两个场景的5个种子中都得到合格最佳图。B只在存在明确类别冲突的场景产生作用；连续场景中A/B一致是合理结果。C在连续多源场景提出了更多变异，因此不能声称它在所有意义上更高效。结果只证明能力/约束搜索行为发生变化，不证明环境预测精度提高。

### 9.7 测试、失败与修复

新增测试：`tests/test_semantic_capability_golem.py`。

覆盖：LAI/土地利用/降水不同需求、UNKNOWN与冲突、多源需求、新插件零优化器改动参与搜索、真实GOLEM A/B/C、语义mutation轨迹。

开发中发现并修复：最初语义约束把“当前图尚缺能力”误当作硬冲突，导致C无法进化。修复后仅拒绝已确认的直接物理冲突；能力缺口留给目标函数惩罚和后续mutation补全。该修复避免把中间不完整图与最终非法图混淆。

最终验证：本阶段定向测试25项通过；全量收集1049项，执行结果为**1045 passed、4 skipped、0 failed、0 errors**。4项跳过沿用可选ESMF/xESMF环境策略。官方GOLEM冒烟日志保存在`output/semantic_golem_experiment/official_golem_smoke.log`，实际返回图为`read -> process -> validate`。

### 9.8 已完成、未完成与创新风险

**已完成**

- GOLEM 0.5.0独立安装、官方API冒烟和EcoData适配运行；
- `S(D)`证据化语义提取与`UNKNOWN/conflict`；
- 配置驱动`Phi`能力映射与解释轨迹；
- 可执行插件能力目录及硬/软约束；
- 默认、筛选、语义引导三组真实搜索；
- 5种子同预算统计和逐次mutation日志。

**仍未完成/未验证**

- 搜索过程中没有执行每个候选的完整环境数值管线，当前Objective评价能力覆盖、物理冲突、成本和复杂度；
- 复杂多分支、多输入DAG虽可由GOLEM表达，但尚无真实多源数据端到端搜索执行证据；
- 边重连由GOLEM结构约束验证，但尚未证明可自动解决所有接口依赖；
- 没有独立高分辨率真值，因此没有预测精度结论；
- 参数权重、搜索预算和规则覆盖仍需在真实任务上验证；
- 模型适配需求能进入规则系统，但并非24个模型都完成原生输入验证。

**潜在创新与重合风险**

GOLEM图进化、能力描述、约束搜索和概率引导mutation本身均有既有研究基础，不能单独作为创新声明。可能形成项目贡献的是：把可追溯环境物理语义`S(D)`映射为能力/硬约束`R`，再让能力缺口直接控制图mutation候选生成，并与双层质量控制及模型需求衔接。该贡献目前是工程与算法原型，仍需系统文献查重、更多真实任务、强基线和独立精度验证。

### 9.9 下一阶段最小建议

1. 选择一个已有真实数据案例，把候选图真正执行并以独立验证误差加入Objective；
2. 加入明确的多输入端口和依赖契约测试，验证真实融合DAG；
3. 在训练/验证/测试严格隔离下分析`alpha/beta/gamma`与预算敏感性；
4. 扩充规则前先补官方变量语义证据，不按变量名称硬编码算法。

### 9.10 关键路径

- `src/ecodata/algorithms/semantic_requirements.py`
- `src/ecodata/algorithms/semantic_rules.json`
- `src/ecodata/algorithms/capabilities.py`
- `src/ecodata/algorithms/golem_search.py`
- `tools/run_golem_search.py`
- `tools/run_golem_official_smoke.py`
- `tools/run_semantic_golem_experiment.py`
- `tests/test_semantic_capability_golem.py`

---

## 10. Algorithm Library Expansion（2026-10-16）

### 10.1 本次完成范围

本阶段完成了现有库审计、三项轻量依赖接入、已有xarray-regrid能力扩充、逐项高级候选核查、能力矩阵、扩大候选库后的真实GOLEM搜索及返回方案执行。没有训练新增神经网络、没有下载大型模型权重、没有增加环境模型或重建平台。

目前Registry共48项描述，其中35项具有可执行实现，9项BLOCKED、4项研究候选。严格按本轮十项证据核验，14项算法/方法达到VERIFIED，21项为PARTIAL。`available`只表示有执行实现，不能代替十项VERIFIED。部分既有算法因本轮未重新收集全部来源证据保留PARTIAL；这不代表其已有功能或历史验证被删除。

所有方法级状态、输入契约与限制见`output/algorithm_library_expansion/algorithm_capability_matrix.csv`，共90行。模型输入与通用能力关系见同目录`model_data_capability_matrix.csv`，共120行，来自现有24个模型Profile；该表不代表24个原生模型均已验证运行。

### 10.2 先审计，后接入

接入前保存`initial_registry_audit.json`，核对Registry、能力标签、已有Adapter、语义规则和质量门。原有xarray-regrid重网格、GSTools、IDW、QP、FNO/UNO、Galileo特征、残差融合及质量感知融合继续复用。

新增依赖仅为可选组：

```text
pip install ".[environment-algorithms]"
# pykrige==1.7.3, metpy==1.7.1, xsdba==0.5.0
```

官方示例额外使用小型绘图依赖nc-time-axis；没有引入新的大型AI运行环境。第三方数值内核通过官方Python API调用，没有复制并改名实现。

### 10.3 按规定顺序核查的11组候选

| 顺序/项目 | 固定版本或核查commit | 许可证 | 实际状态及原因 |
|---|---|---|---|
| 1. [xESMF](https://github.com/pangeo-data/xESMF) | 0.9.2 / `0ca1ab1a3661` | MIT | BLOCKED。已有Python包，但缺ESMF/ESMPy；当前Windows没有可用Conda/WSL。不能把fallback输出计为xESMF验证。|
| 2. [xarray-regrid](https://github.com/xarray-contrib/xarray-regrid) | 0.4.2 / `d804302a4d25` | Apache-2.0 | 已复用linear/nearest/conservative，新增众数、类别比例、mean/median/min/max与cubic包装；分类/统计方法VERIFIED，cubic实际运行但缺直接对应的官方测试证据，标PARTIAL。|
| 3. [PyKrige](https://github.com/GeoStat-Framework/PyKrige) | 1.7.3 / `5e896fb3398c` | BSD-3-Clause | Ordinary/Universal官方示例、真实站点及Adapter测试通过。Regression、3D与外部漂移分支仍PARTIAL，未声称真实DEM辅助或3D场已完整验证。|
| 4. [MetPy](https://github.com/Unidata/MetPy) | 1.7.1 / `433bdd18cc80` | BSD-3-Clause | linear/natural_neighbor/Barnes/Cressman及垂直插值完成官方测试、真实处理与输出验证。官方inverse_distance API用于Barnes/Cressman，不把它误称为任意power-IDW。|
| 5. [xsdba](https://github.com/Ouranosinc/xsdba) | 0.5.0 / `b2eca92b62ed` | Apache-2.0 | EQM达到VERIFIED。DQM/QDM均有真实输出、隔离调用和测试，但直接对应的官方方法测试未完成，严格标PARTIAL。ExtremeValues未接入。|
| 6. [TopoPyScale](https://github.com/ArcticSnow/TopoPyScale) | `8668a5bf6adb` | MIT | BLOCKED。核查版要求Python>=3.13，本机为3.12.14；现有ERA5仅单层，也缺该流程需要的压力层及匹配地形准备。|
| 7. [Thera](https://github.com/prs-eth/thera) | `141a93707aa1` | Apache-2.0 | GPU_REQUIRED。官方环境为Linux/Python3.10/NVIDIA GPU，本机CPU环境不满足；没有下载checkpoint，没有声称支持所有环境连续变量。|
| 8. [Granite WxC](https://github.com/IBM/granite-wxc) / [Prithvi WxC](https://github.com/NASA-IMPACT/Prithvi-WxC) | `a6c4f3165660` / `79dabfcd17ab` | Apache-2.0 / MIT | RESOURCE_BLOCKED。已读取官方MERRA2推理Notebook；它有CPU分支，但checkpoint本身为8,227,134,337字节，另需多层气象与climatology/normalization文件。本轮未下载这些大文件，未运行推理。|
| 9. [Sentinel-5P SR](https://github.com/hyamomar/Sentinel-5P-Super-Resolution) | `98ee2c8c357d` | 未发现明确仓库许可证 | BLOCKED。只核查文档与目录，不安装/再分发未明确授权的实现。其任务是TROPOMI高光谱观测，不是通用PM2.5/LAI模型。|
| 10. [PRIMER](https://github.com/sunmoumou1/PRIMER) | `07166483591c` | 未发现明确仓库许可证 | BLOCKED。原仓库缺明确许可证与已验证checkpoint，官方训练环境使用2张A100；该配置不等于推理最低要求。README另推荐升级仓库，本轮未把未运行版本伪装接入。|
| 11. [PrecipitationFusion](https://github.com/kgavahi/PrecipitationFusion) | `60bc16dac9e7` | MIT | BLOCKED。官方`train.py`第55行存在IndentationError；检查目录中没有预训练权重和直接推理示例。本轮不重训、不改写PDFN冒充原模型。|

上述仓库的完整SHA、许可证原文、小型官方文件下载URL及SHA256见`upstream/*/source_evidence.json`。运行资源与阻塞证据见`upstream_readiness.json`。没有把公开可读仓库等同于已经允许使用或已经可以运行。

论文/方法来源：TopoPyScale [JOSS 2023](https://doi.org/10.21105/joss.05059)及[TopoSCALE](https://doi.org/10.5194/gmd-7-387-2014)；[Thera](https://arxiv.org/abs/2311.17643)；[Prithvi WxC](https://arxiv.org/abs/2409.13598)；Sentinel-5P [监督方法](https://arxiv.org/abs/2501.17210)与[自监督方法](https://arxiv.org/abs/2604.17652)；[PRIMER](https://doi.org/10.1038/s41467-025-67987-9)；[PrecipitationFusion/PDFN](https://doi.org/10.1016/j.rse.2023.113723)。基础软件以官方项目文档/API为实现来源，不为其编造一篇对应论文。

### 10.4 实际接入的能力与输入限制

**网格和分类**

- 继续使用已有xarray-regrid线性、最近邻和保守重网格，未重复实现数值内核。
- 众数粗化调用官方`most_common`；连续统计调用官方`stat`；cubic调用官方`cubic`。
- 类别比例是显式组合：类别one-hot → 官方保守权重 → 各类别比例。记录为`fractional_categorical_aggregation`，不把它称为“众数恢复比例”。
- 分类聚合拒绝细化任务；众数要求有限整数类别码，不能把缺失类别静默转整数。
- xESMF保守方法必须有源和目标cell bounds，缺失时返回`INPUT_REQUIREMENT_MISSING`；取消了该路径原先隐式猜bounds的行为。xarray-regrid规则网格权重沿用其官方定义，使用前仍需确认像元支持域。

**站点与不规则空间**

- PyKrige Ordinary/Universal输出估计场和模型方差；Regression与3D分支提供真实官方API包装，但证据未齐全，保留PARTIAL。
- MetPy调用`interpolate_to_points`和`interpolate_1d/log_interpolate_1d`，没有重写Barnes/Cressman内核。
- 各方法只读取`split=train`；要求明确单位、观测CRS、唯一有效站点，拒绝把多个时间窗口混为一幅场。
- 经纬度通过pyproj转到局地米制坐标再计算距离；投影坐标不改名伪装地理坐标。
- 报告站点凸包外比例、NaN比例、稀疏站点和外推风险。自然邻域/线性覆盖外保持缺测，不偷偷填值。
- 有DEM/辅助变量时检查CRS、连续语义、单位、共同覆盖和矩阵秩；这不等于证明辅助变量有预测增益。

**统计校正**

- EQM/DQM/QDM使用官方`train`和`adjust`；要求`training_reference`、`historical_simulation`显式标记train，训练与应用日期严格不重叠。
- 单位不明、单位不同、重复时间或训练样本不足时拒绝执行；不从数值范围猜单位。
- 只调整统计分布，不把偏差校正等同于时间聚合、守恒或极值保真。
- xsdba更改共享Pint注册器设置，因此采用子进程调用，主平台的pint-xarray单位流程保持独立。

### 10.5 真实输入、实际输出与质量检查

所有新增数值输出位于`output/algorithm_library_expansion/real_examples/`，共19个NetCDF及对应JSON。每份输出均重新打开，并经过现有ResultQualityGate。原始输入没有覆盖。

| 真实输入 | 处理及输出 | 质量/证据范围 |
|---|---|---|
| 现有NASA GEOS-CF PM2.5示例及京津冀小区域切片 | linear/nearest、cubic、mean/median/min/max NetCDF | 文件可读、单位及CRS保留、有效数值检查；不评价真实高分辨率精度 |
| ESA WorldCover 2021 v200官方COG的240×240窗口 | `spatial.xarray_most_common.nc`、`spatial.xarray_fractions.nc` | 真实土地利用类别；类别集合保持，类别比例和为1。只读取窗口，未下载整幅全球/区域影像 |
| 京津冀2019年8月实际站点单时刻 | Ordinary/Universal及4种MetPy站点方法NetCDF | 只用train站；坐标按原始经纬度近似WGS84，基准仍未最终证实，报告保留限制 |
| NOAA IGRA 72357真实探空剖面 | `vertical.metpy.nc` | 明确Pa/degC，按压力插值，禁止超出已有垂直范围 |
| GEOS-CF与40个训练站的真实小时平均序列 | EQM/DQM/QDM NetCDF | 仅前半月训练站拟合，后半月应用；没有用最终测试站拟合或报告精度 |

WorldCover官方来源：`https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39E114_Map.tif`；数据许可CC-BY-4.0。窗口、原始transform、尺寸及SHA256在`worldcover_source_manifest.json`。该真实案例是名义10m类别数据粗化；精确30m→1km接口另用合成UTM网格测试，不混写来源或尺度。

三种偏差校正最终使用官方支持的乘法型参数`kind="*"`，显式启用PM2.5非负质量检查，输出范围约2.37—62.53 ug m-3。加法型产生负值的三份结果已移到`rejected_outputs/`并标记BLOCK，未截零、未删除失败证据。参数选择依据物理非负条件，没有利用验证或测试成绩调参。

这些运行仅证明接口、物理准入与时间隔离成立。GEOS-CF的RH35定义和站点湿度条件差别、原始站点坐标不确定性仍然存在，不能用一次校正成功证明背景数据定义或预测精度已解决。

### 10.6 扩库后的GOLEM实验

语义映射新增规则：分类变量向粗尺度转换需要`categorical_aggregation`；站点到网格需要`observation_interpolation`；气象与地形共同存在时增加地形辅助软目标。未按“LAI→某算法”硬编码。

GOLEM适配层现在读取每个插件的实际compatibility结果。缺单位/观测/辅助/权重/目标网格等前提时记录原因；不可用算法不进入可执行节点池。A在目标函数中惩罚非法节点，B在普通mutation后过滤，C在选择候选前把不兼容模块的概率设为0。GOLEM核心没有改写。

主实验为5任务×2库规模×3方法×5种子，共150次：

- 基础对照：初始库加共同的质量验证节点，20项描述、15项可执行；
- 扩充库：48项描述、35项可执行；受阻神经模型不算可执行候选；
- 固定种子11/22/33/44/55，population=8，最多8代、30秒、60次实际Objective计算；
- 记录Objective调用、预算后拒评次数、实际评价、变异提议、结构/语义非法提议、合格图和图结构多样性；
- 相同的是评价上限，实际评价次数因早停与可行图空间大小而不同；没有把“少评价”偷换为效率提升。

| 任务 | 输入及结果 | 使用限制 |
|---|---|---|
| A：500m LAI→30m | 合成连续场，选择可执行连续重网格，排除分类聚合、无权重Thera及无站点Kriging | 输出更密不证明恢复真实30m细节 |
| B：30m土地利用→1km | 合成UTM分类网格；基础库缺聚合能力，15次均无合格方案；扩充库15次均选择并执行众数聚合 | 最近邻抽样不冒充统计聚合，类别比例为独立输出能力 |
| C：PM2.5站点→约1km | 真实站点，IDW/PyKrige/MetPy为站点候选；返回方案按目标21×21坐标执行并回读 | 当前目标函数倾向低成本IDW，没有依据宣称其精度胜过Kriging |
| D：1km气温+30m DEM | 合成接口任务，识别气象/地形条件，返回可执行常规重采样 | TopoPyScale/Granite受阻，当前结果没有利用DEM形成已验证地形降尺度；不得算该高级能力已完成 |
| E：网格降水+雨量站 | 合成两来源及train/validation/test观测，现有质量感知融合实际生成数值输出 | 尚无ERA5/IMERG+雨量站真实降水效果验证；PRIMER/PDFN没有偷偷替换为加权平均 |

主实验150次中135次返回合格方案并实际执行、回读、通过质量门；15次不合格均为基础库的土地利用统计聚合能力缺口，未隐藏或补分。

5任务、5种子的均值如下。合法比例同时计入节点/输入条件与环/自环检查；中间能力缺口仍由Objective优化。

| 库规模 | 方法 | 合法提议比例 | 变异提议 | 实际评价 | 合格图数 | 唯一评价图 | 搜索耗时(s) |
|---|---|---:|---:|---:|---:|---:|---:|
| 基础 | A | 22.37% | 95.20 | 60.00 | 3.16 | 47.40 | 1.665 |
| 基础 | B | 28.78% | 3080.28 | 46.68 | 19.40 | 22.20 | 1.518 |
| 基础 | C | 99.62% | 1067.72 | 34.40 | 20.36 | 21.04 | 1.280 |
| 扩充 | A | 17.57% | 94.40 | 60.00 | 2.60 | 52.60 | 1.755 |
| 扩充 | B | 24.01% | 515.68 | 60.00 | 31.88 | 34.92 | 1.723 |
| 扩充 | C | 99.48% | 1097.48 | 56.00 | 37.72 | 38.76 | 1.978 |

结论：扩库后C保持较高合法提议比例并发现更多合格图；但变异提议明显多于A，扩充库搜索耗时也高于B。现有证据不能证明C在所有效率指标上全面优于B。各任务的最佳图大多在共同初始化时已可行，主实验的“首个合格图时间”因此缺乏区分度；不能把接近0的该数值作为搜索创新证据。

### 10.7 真实多节点组合补充验证

另用现有真实GEOS-CF示例，经过Reader识别、语义映射，要求同时满足“连续空间变换+非负检查”。分辨率由pyproj.Geod从坐标和明确CRS计算并记录，不因识别字段UNKNOWN而猜固定公里数。

再执行3方法×5种子、相同60次实际评价上限：A仅1/5找到合格图，B为5/5，C为5/5。C的5次均实际生成并执行`spatial.bilinear → validation.result`；B中部分顺序相反，但Pipeline末端仍自动执行质量门。所有11个合格结果均通过实际数值执行和质量检查。

这是当前小规模真实数据的多节点能力组合证据，不能扩大为复杂多分支DAG、所有物理约束或高分辨率预测精度证明。详情保存在统一实验JSON的`real_composition_case`。

### 10.8 测试与发现问题

- xarray-regrid官方reduction测试：10通过，1个官方预期失败；未改官方测试。
- MetPy官方点和一维插值测试：43通过。
- PyKrige：未改动的Ordinary/Universal官方示例实际运行，退出码记录为0。
- xsdba：直接运行固定commit官方Notebook前两段代码，11315个示例值有限；仅移除IPython显示指令，使用非交互绘图。
- 新增Adapter测试23项：类别/比例区分、统计量、守恒积分、缺bounds、稀疏/重复站点、训练隔离、缺单位、垂直越界、图筛选、IDW目标网格及保守方法前提。

修复了本阶段发现的四个实际问题：

1. **全局单位注册器污染**：xsdba导入把`force_ndarray_like`设为False，曾导致42项既有回归失败。改为子进程运行后，原单位引擎未被修改，新增测试确认调用后单位转换仍正常。
2. **IDW目标网格被忽略**：原插件只用来源网格。现在尊重显式目标坐标，并增加结果回读坐标核对；站点约1km案例确实为目标21×21网格。
3. **保守方法准入缺前提检查**：原compatibility没有检查像元统计语义，能被搜索选中但执行失败。现直接复用既有GridRegridder输入校验；未知cell support不再被当作可执行保守变换。
4. **真实偏差校正的负值风险**：加法型分位数调整最低产生约-19.13 ug m-3。案例导出现在于写文件前和回读后均按任务非负约束检查；失败结果保留为拒绝样本，最终乘法型输出通过检查。这不表示乘法型在独立验证中更准确。

最终全量回归：**1068 passed、4 skipped、0 failed、0 errors**，共1072项，167.46秒。XML为`output/algorithm_library_expansion/full_regression.xml`。4项跳过沿用可选环境策略；警告主要为已有第三方弃用提示。23项新增测试全部通过。

### 10.9 覆盖缺口与模型专用能力

- 已有真实候选覆盖连续/分类处理、类别比例、站点估计、常规时间/累计处理、垂直插值、范围检查、已有守恒/QP、同变量融合与偏差校正。
- C5学习型任意尺度、C7地形气候降尺度、曲线网格xESMF路径、C12极值校正仍有明确阻塞或未验证分支。
- C13不规则时间没有因接入MetPy不规则空间插值而被虚报覆盖。
- C15土壤剖面语义以及C24—C30排放/物种/初边界/作物管理映射，分别标为MODEL_SPECIFIC或缺口，不注册为通用GOLEM候选。
- WPS、WRF-Chem工具、SMOKE、FEST-C仍属于Model Adapter/Preprocessor层，本轮未混入随机算法池。
- 任意高级算法尚未验证时，目录里只能记录BLOCKED/PARTIAL，不能据此宣称30项能力全部完成。

### 10.10 复现与关键路径

```text
python tools/run_algorithm_library_real_examples.py regrid
python tools/run_algorithm_library_real_examples.py landcover
python tools/run_algorithm_library_real_examples.py pykrige
python tools/run_algorithm_library_real_examples.py metpy
python tools/run_algorithm_library_real_examples.py bias
python tools/run_algorithm_selection_expansion.py
python tools/run_real_capability_composition.py
python tools/export_algorithm_capability_matrices.py
pytest -p no:cacheprovider --basetemp=output/pytest_library_reproduce
```

GOLEM仍使用`.venv-golem`；其余Adapter使用主环境及上述可选依赖。真实输入路径沿用项目`examples/`与既有京津冀测试数据目录，没有修改原始数据。

主要代码：

- `src/ecodata/algorithms/open_source_spatial.py`
- `src/ecodata/algorithms/open_source_stations.py`
- `src/ecodata/algorithms/open_source_bias.py`
- `src/ecodata/algorithms/{plugins,capabilities,golem_search,semantic_requirements,pipeline}.py`
- `src/ecodata/algorithms/semantic_rules.json`
- `tools/run_xsdba_adjustment.py`
- `tools/run_golem_search.py`
- `tests/test_algorithm_library_expansion.py`

主要交付均在`output/algorithm_library_expansion/`：

- `algorithm_capability_matrix.csv`
- `model_data_capability_matrix.csv`
- `algorithm_selection_expansion_experiment.json`
- `algorithm_verification_evidence.json`
- `upstream_readiness.json`
- `selection_results.csv`、`selection_summary.csv`
- `real_examples/`、`selected_outputs/`
- 官方测试XML、Adapter测试XML及`full_regression.xml`
- `rejected_outputs/`：3份加法型负浓度失败样本与BLOCK报告，作为质量门反例；不属于最终合格输入。
- `delivery_integrity.json`：165份最终NetCDF的回读、维度、有效数值、单位、SHA256记录。

本项目贡献仍是“物理语义→能力需求→兼容约束→引导图进化→处理组合”。PyKrige、MetPy、xsdba、xarray-regrid等数值方法归原项目，不能计为EcoData原创算法。

下一步优先解决xESMF独立Linux环境和一个真正匹配的地形/气象官方案例；完善多输入端口、逐边契约传播与数值目标评价。暂不以增加模型数量或下载全部权重代替这些验证。

## 11. Algorithm Library Expansion V2

### 11.1 本轮范围与原有状态

本轮只补能力空间，不改变 GOLEM 的 mutation、crossover、selection、graph、objective，不运行新一轮 GOLEM 优选实验。保留原算法、实验、数据转换和模型 Adapter。

检查了 `src/ecodata/algorithms/` 中的 Registry、contracts、plugins、capabilities、semantic_requirements、semantic_rules、pipeline、golem_search，以及 tests、已有两张能力矩阵和本报告。复用现有 AlgorithmSpec/AlgorithmContext、Registry、统一 Pipeline、前后质量门、子进程隔离及官方源码取证工具。

V1 基线：48个注册描述，其中35个执行状态可用；集中证据表为14 VERIFIED、21 PARTIAL、9 BLOCKED、4 UNVERIFIED。C5/C7/C12/C13没有足够的可执行覆盖证据；不把普通插值、普通分位数校正、DEM插值、不规则空间站点插值分别充当这四项能力。

### 11.2 实际新增：3个官方组件，不是8个新模型已经可用

| 组件 | 真正调用的官方接口 | 当前验证范围 | 能力结论 |
|---|---|---|---|
| `bias.xsdba_extreme` | xsdba 0.5.0 `ExtremeValues.train/adjust`，先执行官方EQM，再执行广义帕累托尾部二阶校正 | 5年合成日降水训练期与独立应用期；真实拟合、执行、质量门、NetCDF回读 | C12 有可执行极值专用组件；不等于真实环境峰值精度已证实 |
| `temporal.xclim_statistics` | xclim 0.59.1 `select_resample_op`、`threshold_count`、`spell_length_statistics` | 日数据 mean/sum/min/max、阈值达标天数、最长连续达标天数、rate积分；另执行已有真实GEOS-CF气温 | C10 组件可用；未宣称任意不规则时间、所有日历或任意缺测均已支持 |
| `terrain.fords_pressure` | 固定commit ForDs `offline/topo_factory.py:downscale_air_pressure` | 原始函数未修改；显式气压、两种温度、原/目标高程，共同网格；合成测试 | 地形气压修正组件 VERIFIED；整个 C7 降尺度工作流仍 PARTIAL |

注册数变为59，其中新增3个可执行组件、8个明确不可执行的研究候选。执行可用描述38个；按集中证据统计17 VERIFIED、21 PARTIAL、17 BLOCKED、4 UNVERIFIED。这不是“30项能力全部完成”。ForDs组件的注册级VERIFIED只指气压函数；C7整项在矩阵中保守标为PARTIAL。

### 11.3 官方源码、版本与阻塞证据

以下均保存官方README、许可证（若存在）、目录清单、commit、下载时间和SHA256于 `output/algorithm_library_expansion/upstream/`。没有下载任何大型模型权重。

| 候选 | 官方来源与固定版本 | 许可证/入口 | 当前情况 |
|---|---|---|---|
| GeoFAR | [作者项目库](https://github.com/eceo-epfl/GeoFAR)，72d659d199949f771f00b2ab47253a8651943775；README链接的[实际模型库](https://github.com/eceo-epfl/climate_learn_ds)，276ad83b204fdba88ce242c88868a5db12ee3a9c | 实际模型库MIT；`models/hub/geofar.py` | BLOCKED。不能因主库主要为网页就说完全没有代码；实际代码已找到。但导入的`scipy.special.sph_harm`在本机SciPy已不存在，timm未安装，无匹配权重/任务推理。Geo-INR不自动证明任意目标坐标查询可用 |
| Thera | [prs-eth/thera](https://github.com/prs-eth/thera)，沿用141a93707aa12d6d52e9fc6174e389828703be8a | Apache-2.0，官方Linux/Python3.10/NVIDIA路径 | 保持GPU_REQUIRED/BLOCKED，没有改写为VERIFIED |
| ClimateDiffuse | [robbiewatt1/ClimateDiffuse](https://github.com/robbiewatt1/ClimateDiffuse)，7b8ee181fa3baeacf08a2bf9dc275989048b4030 | 检查目录未找到明确仓库许可证；`examples/inference.ipynb`、`inference/save_test_preds.py` | BLOCKED。仓库确实列出diffusion/unet checkpoint，但没有下载执行；需核对本仓库及其改编EDM的授权。固定训练倍率不等于任意尺度 |
| PhyDL-NWP | [yingtaoluo/PhyDL-NWP](https://github.com/yingtaoluo/PhyDL-NWP)，64f126672abdc604a6a1846c578c76901a382935 | MIT；`main.py`、`train_forecasting_models.py` | BLOCKED。明确输入DEM/经纬度及8气象变量；main.py实际读取`data/ningxia_real_train.npy`，当前main目录没有相应数据。存在CPU分支，不能一概说必须GPU；[历史官方发布](https://zenodo.org/records/15482043)也尚未作为本机执行证据 |
| ForDs完整流程 | [leelew/ForDs](https://github.com/leelew/ForDs)，c533dfa4aa65089da2dcc0035aeda536094bfe53 | MIT；`offline/main.py`、`offline/predict.py`、CoLM地形因子入口 | 完整流程BLOCKED，气压组件已执行。完整流程另有站点工程路径、离线插值、训练好的降水模型和CoLM/地形编译依赖；不以气压函数代表全部 |
| TopoPyScale | [ArcticSnow/TopoPyScale](https://github.com/ArcticSnow/TopoPyScale)，沿用8668a5bf6adb2caf884b61f98857578e7b530c3c | MIT；保留既有要求 | 保持BLOCKED：所查版本Python>=3.13；现有ERA5地面层不等于所需压力层气象 |
| WassDiff | [Way-Yuhao/WassDiff](https://github.com/Way-Yuhao/WassDiff)，59c65f424ba1a77d64084d19dbdcffbe3e35ec99 | 检查目录无明确许可证；`src/eval.py model=wassdiff ...` | BLOCKED。官方确有Colab、HuggingFace权重与CONUS mini-dataset；未执行，不能称没有权重，也不能按mini字样推断下载量很小 |
| QMAPP_extrap | [HYCLEN/QMAPP_extrap](https://github.com/HYCLEN/QMAPP_extrap)，4c4a6ddd38c4dcd647f3ee99123861b2847f8122 | 未找到明确许可证；`Example/Python/Main_QMAPP_extrap_precip.py`等 | BLOCKED于授权/接入验证。方法明确针对新极值外推，与普通EQM不同，暂不复制进入主流程 |
| DualTimesField | [WisdomTogether/DualTimesField](https://github.com/WisdomTogether/DualTimesField)，6a2263fd6b7cee4f0a662406df0eb850543fe086 | README声明MIT，但无完整LICENSE；`interpolation/models.py: DualTimesFieldInterpolator.fit/predict` | BLOCKED于授权核对和实际接入。已读到真实不规则时间/掩膜接口；README声称的ICML2026和EPA-Air/USHCN成绩没有作为本机验证或已核实论文结论 |
| CRU | [boschresearch/Continuous-Recurrent-Units](https://github.com/boschresearch/Continuous-Recurrent-Units)，79723e7772102cae8a78109e18d81beff9c7cac2 | AGPL-3.0；`run_experiment.py`、`lib/CRU.py` | 官方[ICML2022论文](https://proceedings.mlr.press/v162/schirmer22a.html)可核对。旧Torch1.11/Python3.9/geotorch路径未复现，也没有任务训练权重；AGPL并非禁止研究，但再分发/服务需审查 |

候选代码/论文支持的物理量和任务边界各不相同。ClimateDiffuse不能直接当任意连续尺度算子，QMAPP_extrap不是完整空间重构，ForDs气压公式不是整个天气降尺度系统，CRU也不是免训练时间填补器。

### 11.4 约束、质量门及输入输出

- 极值组件要求显式`cluster_thresh`及单位、正当分位阈值、分开的训练/应用日期、规则采样、有限值。每期至少365样本是本适配器最低拒绝条件，不是统计充分性保证。上游明确将ExtremeValues标为experimental，短记录和不足尾部样本仍需专业判断。
- 使用官方EQM作为第一阶，再由官方ExtremeValues修正尾部。相同合成数据对比发现395个值发生二阶变化，最大约3.35757 mm/day；这只证明调用了不同的尾部阶段，不能解释为极值精度提高。
- xclim只接本轮已核对的连续日采样；缺测不静默跳过，时间缺口拒绝处理。`frequency`具体指达标天数，不是“事件发生次数”；`duration`是最长连续达标天数。`sum`要求interval_total，rate积分与interval_total求和分开，气温不能声明为累计量。
- 阈值必须有单位。计数、持续天数和积分会合法改变单位，必须显式声明`model_constraints.output_unit`。现有质量门新增可选预期输出单位；未声明仍按原单位检查，声明与实际不符仍BLOCK，不绕过质量检查。
- ForDs只接受确认的surface_air_pressure/Pa、temperature/K、elevation/m；要求坐标与CRS一致、有限且正气压。缺地形或温度不自动编造，不隐含重投影/外推。
- 两个气候包仍在子进程执行，避免污染主进程Pint注册器。ForDs也隔离官方通用`utils`模块名，逐次核对官方文件SHA256。
- 新候选全部是CatalogOnly，compatibility拒绝执行，不能提供实际覆盖。现有GOLEM核心没有改动。

### 11.5 实际运行与测试资源

`tools/run_capability_v2_examples.py`通过原有UnifiedAlgorithmPipeline执行，前后质量门均开启；输出不是只调用import。共保存6份NetCDF，全部回读、检查shape/坐标/有限值/单位并记录SHA256：

| 输出 | 输入与规模 | 验证结论 |
|---|---|---|
| `extreme_precipitation.nc` | 5年1825个合成日值；独立训练/应用时间 | 非负、有限、单位mm/day、日期保持；不代表真实降水效果 |
| `ordinary_eqm_precipitation.nc` | 与上项相同，已有普通EQM作对照 | 证明普通偏差校正与尾部校正阶段不同，不计作新增算法 |
| `real_temperature_mean/min/max.nc` | 既有NASA GEOS-CF 48个小时平均，先按24小时完整组形成2个UTC日；输出1×4×5 | 真实气象数值可进入xclim；文件历史名称era5_sample.nc但属性明确实际为GEOS-CF。只有2天，不能称整月统计或气候评估 |
| `fords_pressure.nc` | 2×3合成高程/温度/气压 | 与独立静力气压关系核对；零高差恢复原气压；不能外推真实地形降尺度精度 |

位置：`output/algorithm_library_expansion/v2_examples/`。`execution_manifest.json`包含逐输出质量门、来源类型、运行时间和SHA256，`extreme_vs_eqm.json`记录尾部阶段差异。各组件均为秒级小测试，具体实测以manifest为准，不作为固定性能承诺。

新增测试30项，覆盖7类日统计、rate积分、缺失/错误时间/语义/单位、ForDs坐标与必需辅助量、极值真实拟合及回读、训练应用重叠拒绝、候选不能执行、显式衍生单位仍受质量门控制。与V1适配测试合并运行：53 passed、0 failed。`pip check`无损坏依赖。

最终全量回归：**1098 passed、4 skipped、0 failed、0 errors**，共1102项，217.98秒；258条警告主要来自既有第三方弃用提示及刻意失败的优化器测试。测试XML为 `v2_full_regression.xml`。交付回读及GOLEM文件哈希核对记录为 `v2_delivery_check.json`。

4项跳过分别是：2项旧空气质量工程输出前置资源尚未生成；1项ESMF缺失；1项CMAQ官方profile/PseudoNetCDF独立环境未配置。没有把跳过项计为通过或原生模型验证。

### 11.6 能力矩阵与剩余缺口

原`algorithm_capability_matrix.csv`已直接扩展，而非另建替代注册系统。保留原列，新增需求的数据性质、能力编号/名称、候选、类型、commit、官方入口、输入输出、运行状态、验证状态、适用类型/模型及测试证据。共101行，模型/角色矩阵保持120行。

| 能力 | 当前真实程度 |
|---|---|
| C5 | GeoFAR/ClimateDiffuse/PhyDL-NWP为待验证研究候选，Thera仍阻塞；没有VERIFIED的学习型任意尺度环境重构 |
| C7 | ForDs官方气压组件可执行；完整气象/地形辅助降尺度未验证，仍PARTIAL；PhyDL-NWP、TopoPyScale等不能计为完成 |
| C12 | 新增xsdba ExtremeValues实际执行；WassDiff、QMAPP_extrap只列候选。普通EQM仍归C20 |
| C13 | DualTimesField和CRU确有代码候选，但本平台没有已执行的对应Adapter，不计覆盖 |
| C9 | 保留现有传统时间插值，不为求新加入网络；不因C13候选而提升验证状态 |
| C10 | 新增xclim日统计组件，保留原interval-aware时间处理；不混淆mean/sum/rate积分 |
| C23 | 保留已有Galileo特征、semantic_stack、外部漂移/辅助量接口。ForDs确实读取温度与高程；未新增独立编码网络或声称特征等于预测 |
| C15 / C29 | 土壤剖面预测仍缺经过本轮验证的通用候选；土壤/土地利用→模型字段映射继续归Model Adapter，不放入GOLEM随机池 |
| C24—C30 | 排放/物种/初边界/土地与土壤/作物管理仍为模型专用能力或配置依赖，不用无关通用算法凑齐 |

其他C1—C30逐项状态以更新后的矩阵和`v2_delivery_check.json.capability_summary`为准。V1对既有算法采用较严格的集中证据清单，因此部分已有可运行方法仍标PARTIAL；这不是删除能力，也不是宣称不能执行。BLOCKED/UNVERIFIED不计完成覆盖。

现有站点估计C6有6个集中证据VERIFIED候选，真正存在选择空间。C12目前只有1个可用极值专用组件；C5/C13为多个研究候选而非多个可执行候选，C7也不能把局部气压组件与完整流程混计。

### 11.7 复现、代码与下一步

```text
python tools/run_capability_v2_examples.py
python -m pytest tests/test_algorithm_library_v2.py tests/test_algorithm_library_expansion.py -p no:cacheprovider --basetemp=output/pytest_v2_reproduce
python tools/export_algorithm_capability_matrices.py
python tools/verify_capability_expansion_v2.py
```

代码：`src/ecodata/algorithms/{open_source_bias,open_source_time,fords,expansion_v2,contracts}.py`、`expansion_v2_catalog.json`；官方ForDs代码/许可证在`external/fords/`，工具入口在`tools/run_{xsdba_adjustment,xclim_statistics,fords_pressure}.py`。仅对现有Pipeline/质量门增加显式衍生输出单位契约；未改变GOLEM核心。

下一阶段先选一个C5匹配的官方权重/小型任务并解决独立环境，再完成一个C7真实地形气象案例，随后核清DualTimesField许可证或在隔离环境验证CRU。已有数据不足时先报告，不制造高分辨率真值、地形或观测。

**【READY_FOR_GOLEM_EXPERIMENT = NO】**

原因：C5尚无已执行的学习型任意尺度环境重构，C7只验证局部组件，C13尚无执行闭环；关键能力也缺多个真实可用候选。可以继续使用原注册框架，但现在冻结为“核心能力已齐备”的Registry V1会掩盖这些缺口。本轮没有启动新的GOLEM实验。

## 12. Algorithm Library Expansion V3

### 12.1 范围与结论

本轮继续复用Registry、语义识别、能力映射、统一Pipeline及质量门。不修改GOLEM mutation/crossover/selection/objective/graph，不执行新的GOLEM正式实验，不重复开发C12。

实际新增一个可执行官方模型：**DualTimesField不规则时间插值**。新增两个有固定源码证据但尚未执行的研究描述：consistency climate downscaling、DSFNO。C7的PhyDL-NWP进行了官方包获取和实际启动，但所需官方数据缺失，不能声明PDE训练或降尺度成功。

注册表现在61个描述：39个执行状态可用；集中证据18 VERIFIED、21 PARTIAL、16 BLOCKED、2 RESEARCH_CANDIDATE、4 UNVERIFIED。算法描述数量不等于能力覆盖数量，研究候选及BLOCKED不进入完成统计。

### 12.2 修正DualTimesField上轮状态

官方库：[WisdomTogether/DualTimesField](https://github.com/WisdomTogether/DualTimesField)，固定commit **6a2263fd6b7cee4f0a662406df0eb850543fe086**。重新读取后commit未变化，官方README确实声明 **MIT License**，并描述ICML2026、USHCN和EPA-Air插值路径。

**修正V2判断：不能仅因没有单独LICENSE文件就整体标为BLOCKED。** 本轮依照作者README中的MIT声明实际开展研究运行。所查树仍未发现独立LICENSE，忠实记录该证据缺项，不制造一份不存在的官方许可证。原始README、源码、commit、文件SHA256保存在`external/dualtimesfield/`，许可证声明与文件存在性记录在`v3/source_manifest.json`。对外再分发时仍应保留原作者声明并核对完整版权通知。README论文成绩没有作为本机复现成绩。

隔离运行环境为`.venv-capability-v3`：新增包安装在该环境中，复用原环境只读Torch/NumPy等二进制，**不是声称完全独立重新安装了一整套Torch**。已执行官方DualTimesField requirements安装检查；PyDOE等额外依赖也只装在该环境，未降级主环境。

### 12.3 真实USHCN fit → predict → Pipeline

使用官方`interpolation/datasets.py: USHCNDataset`内指定的[USHCN基准下载地址](https://raw.githubusercontent.com/usail-hkust/t-PatchGNN/main/data/ushcn/raw/small_chunked_sporadic.csv)，不是自行生成气象序列。完整下载27,912,979 bytes，350665行、12列、1114个序列ID；SHA256为`671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf`。一次截断下载已识别并移除本轮不完整临时副本，完整来源文件保留。

原样调用官方Loader处理，得到668 train、223 validation、223 test序列；本轮只取train中的一个样本，不用validation/test选择模型。取其中一个已有归一化通道的短窗口，并裁到真实观测支持范围，避免边缘缺测被冒称插值。

| 项目 | 实测 |
|---|---|
| 官方模型 | `DualTimesFieldInterpolator`，源码未修改 |
| 输入 | 123个原始不规则数值时刻；36个本来有效的观测，其中7个仅用于填补检查、不交给模型 |
| 真正参与fit | 29个可见观测；其余位置为NaN mask；`target_mask`显式等于观测mask，绝不以隐藏目标训练 |
| 训练 | 10 epochs、seed 42、CPU；官方fit与predict |
| 时间范围 | 约3.1—87.9，保留官方基准数值时间；原始日历起点不明，不能编造年月日 |
| 输出 | 123个有限值，94个缺测/查询位置得到估计；29个原始观测原值保留 |
| 单位 | `1`，明确是官方归一化基准通道，不把它猜成摄氏度或毫米 |
| 检查 | 时间坐标逐值一致、shape正确、无NaN/Inf、单位保留、mask保存、NetCDF回读成功 |
| 质量流程 | 已经执行原有RawInputQualityGate及ResultQualityGate，不是只import或绕开Pipeline |

输入：`output/algorithm_library_expansion/v3/ushcn_masked_input.nc`；输出：`ushcn_dualtimesfield_output.nc`；完整执行与质量门记录：`dualtimesfield_execution.json`。

这是**VERIFIED的真实环境基准短运行**，不是完整论文复现、不是“万能预测”，也不是气象精度已证明。原始基准已经过预处理，官方Loader归一化也有其自身划分策略；因此不把这7个隐藏值产生的误差包装成严格独立精度结果，本轮不报告优化成绩。平台适配器自己的归一化只拟合29个可见输入。

接口仅允许连续一维时间序列、4—512个输入时刻、明确单位、唯一递增时间；NaN代表未观测，Inf拒绝。查询必须在**有限观测**支持范围内，最多1024个时刻；不允许未来外推。短运行最多100轮，避免用户误触无上限训练。

### 12.4 C5a/C5b：官方候选与尺度边界

新候选：[p-hss/consistency-climate-downscaling](https://github.com/p-hss/consistency-climate-downscaling)，MIT，commit **e47a433ae5752d44973204881eac96d33237e7d5**。对应[论文](https://www.nature.com/articles/s42256-025-00980-5)为2025年Nature Machine Intelligence工作。已保存官方README、LICENSE、`sample.py`、配置、模型/推理文件及测试入口。

- 训练入口`main.py`，采样入口`sample.py`；有`ConsistencyInference.load_model/run`和`Consistency.sample_conditional`。
- `sample.py`实际写有`/results/best_*_model.ckpt`、`/data/datasets/era5_3deg.nc`，默认样本尺寸64×96。所查仓库没有checkpoint，未下载大型权重，未执行受训模型推理。
- 条件采样接口接收`conditioning`、`x_image_size`、`y_image_size`、采样步数/时间；源码断言noise与conditioning形状相同。它不是把任意原始粗网格直接传进来就会自动生成目标网格。
- 配置中的`network_resolution`、`sample_dimension`以及论文scale-adaptive结论，不是本机任意目标坐标/任意倍率已验证的证据。实际不同目标尺度是否共用权重、如何构造条件场及归一化，仍需任务checkpoint和匹配输入检验。
- 因而归为 **C5a RESEARCH_CANDIDATE**，声明`scale_adaptive_climate_downscaling`、`learning_based_spatial_reconstruction`、`uncertainty_aware_downscaling`；不声明C5b已完成。官方Dockerfile给出的基线是Python3.10/Torch2.1 CUDA镜像、diffusers0.14.0及Lightning等，本轮没有同时安装整套大模型环境。

保留[DSFNO官方代码](https://github.com/qy707/DSFNO)，MIT，commit **cd53850b97aff43b545332bcb0f2a89f95c4d841**。已读`Modules.py/main.py/README`：官方发布入口主要为Navier-Stokes，16×16→32×32训练，并比较16/32/64输出。其论文包含气候实验不等于该checkout已经提供可直接运行的ERA5工程。当前为C5b **RESEARCH_CANDIDATE**，未运行或声称气候精度。

GeoFAR、Thera、ClimateDiffuse继续保留，状态不因研究介绍而提升。Thera保持GPU_REQUIRED；GeoFAR保留此前环境/权重缺口；ClimateDiffuse保留许可证与任务推理缺口。

### 12.5 PhyDL-NWP：实际获取、启动与缺口

官方库：[yingtaoluo/PhyDL-NWP](https://github.com/yingtaoluo/PhyDL-NWP)，本轮固定官方publish分支commit **4af6dcd93d34367469fc5a9b93f97ddfd81bd86e**，MIT。README、LICENSE、requirements、main、PDE_Discovery及forecast入口均保存。

优先寻找Ningxia后，实际下载并检查了两个官方发布，而非只看GitHub文件名：

| 官方包 | 大小 | 数据清点 |
|---|---:|---|
| [Zenodo 15482043](https://zenodo.org/records/15482043) | 9,504,667 bytes | 源码、图/海报、LICENSE；没有npy/npz/mat/nc/csv数据成员 |
| [更新发布15540502](https://zenodo.org/records/15540502) | 9,667,834 bytes | 同样没有数据成员；已保存完整ZIP成员清单 |

两个ZIP原件、SHA256与URL保存在`v3/source_manifest.json`。README中提到Ningxia，不代表这些实际下载包包含Ningxia。没有用自己造的数据绕过这个问题。

在隔离环境补齐PyDOE，并使用已有Torch CPU执行了：

```text
.venv-capability-v3/Scripts/python.exe main.py --gpu -1 --batch_size 8 --hidden_size 8
```

实际进入CPU路径，随后报`FileNotFoundError: ./data/ningxia_real_train.npy`；错误位置main.py第366行。标准输出、标准错误、命令和退出码保存为`v3/phydl_execution.json`。因此 **BLOCKED的具体原因是缺少官方数据**，不是泛泛“没有GPU”。没有执行成功PDE discovery/latent force，也没有在main失败后伪称forecast训练成功。

官方requirements还包含`python==3.10.14`及固定Torch2.3.1/NumPy1.23.5；不能把当前Python3.12与共享Torch2.14.1环境说成完整原始环境复现。本轮只进行了兼容导入和明确的CPU启动尝试，没有降级主环境。即使解决软件版本，当前数据文件仍缺失。

`phydl_contract.py`是**预检工具，不是已完成的执行Adapter**：检查DEM、经纬度、8个气象字段、显式标准化单位、CRS声明和共同空间维度。这里的标准单位是平台预处理契约，不是从未拿到的Ningxia文件中核实出的元数据。完整输入覆盖检查、官方模型调用和输出质量闭环仍未完成；因没有真实模型输出，不生成空壳“成功结果”。官方main读取8目标场数组，也不能凭README的11输入描述就证明DEM已经参与了本机计算。

### 12.6 能力分层与GOLEM准入

原矩阵原地更新为逐能力/子能力行，不把大类和组件混为一个状态：142行算法能力、120行模型角色。MODEL_SPECIFIC在矩阵中单列、不注册可执行GOLEM节点。C15土壤剖面算法缺口也与C29模型字段映射区别开。

| 能力 | 候选算法 | 状态 | 真正运行 | GOLEM情况 |
|---|---|---|---|---|
| C5a学习型空间降尺度 | consistency climate / GeoFAR / ClimateDiffuse | RESEARCH_CANDIDATE / BLOCKED | 本轮未执行受训气候模型 | 不进入可执行池 |
| C5b任意/连续尺度 | DSFNO / Thera | RESEARCH_CANDIDATE / GPU_REQUIRED | 未验证气候任意尺度推理 | 不进入可执行池 |
| C7地形辅助 | PhyDL-NWP / ForDs | PhyDL BLOCKED；ForDs仅C7b组件VERIFIED，整个C7仍PARTIAL | PhyDL启动后缺数据；ForDs沿用V2结果 | 只有ForDs气压组件可作候选 |
| C12极值 | xsdba ExtremeValues / WassDiff / QMAPP | VERIFIED / BLOCKED / BLOCKED | 沿用V2实际执行，不重复开发 | 只有xsdba可执行 |
| C13不规则时间 | DualTimesField / CRU | VERIFIED / BLOCKED | DualTimesField真实USHCN fit/predict成功 | DualTimesField可执行候选，未开展GOLEM搜索 |

细分标签：C7a terrain_auxiliary；C7b terrain_pressure_adjustment；C7c terrain_aware_weather_downscaling。ForDs插件现在只提供terrain_pressure_adjustment，不借此提升整个C7。C12a普通校正、C12b极值校正、C12c生成式极值重构分列；C13a传统时间插值、C13b不规则插值、C13c连续时间表示分列。

### 12.7 测试、输出与复现

本轮新增29项测试，覆盖不规则/重复/倒序/非法时间、缺测mask、官方fit/predict、坐标回读、禁止外推、DEM/气象必需项、单位、C5不得错误声明任意尺度、研究候选不可执行及模型适配分层。定向测试合计59项通过。全量1131项：**1127 passed、4 skipped、0 failures、0 errors**，204.151秒。4项跳过仍为旧工程前置资源2项、ESMF1项和CMAQ独立读取环境1项，不计作通过。

源码/入口：

- `src/ecodata/algorithms/dual_times_field.py`：现有Registry/Pipeline适配器；不含自写神经网络。
- `tools/run_dualtimesfield.py`：隔离调用未改写的官方模型。
- `tools/prepare_capability_v3_sources.py`：有下载量限制的官方数据获取与SHA256清点。
- `tools/run_capability_v3_examples.py`：真实USHCN实验及PhyDL启动记录。
- `src/ecodata/algorithms/phydl_contract.py`：仅预检；不得标成原生模型执行。
- `expansion_v3_catalog.json`与既有`expansion_v2.py`：候选描述、历史状态修正。
- `tests/test_algorithm_library_v3.py`；`tools/verify_capability_v3.py`：定向与交付核对。

```text
.venv/Scripts/python.exe tools/prepare_capability_v3_sources.py
.venv-capability-v3/Scripts/python.exe tools/run_capability_v3_examples.py
.venv/Scripts/python.exe -m pytest tests/test_algorithm_library_v3.py tests/test_algorithm_library_v2.py -p no:cacheprovider --basetemp=output/pytest_v3_reproduce
.venv/Scripts/python.exe tools/export_algorithm_capability_matrices.py
.venv/Scripts/python.exe tools/verify_capability_v3.py
```

主要结果统一在`output/algorithm_library_expansion/v3/`：原始USHCN CSV、两个官方PhyDL ZIP、source_manifest、输入/输出NetCDF、模型执行JSON、定向/全量测试XML和delivery_check。没有新增分散Markdown交付报告。

报告更新日期使用客户端2026-10-17；测试日志原本记录的2026-10-16运行时间不篡改。两份既有GOLEM核心文件SHA256与V2一致。

### 12.8 最终状态与下一步

- **C5：研究候选更明确，但尚无本机受训气候降尺度闭环；C5b没有被虚报完成。**
- **C7：ForDs气压修正可用，PhyDL缺官方Ningxia数据，完整天气降尺度未完成。**
- **C12：已有可执行极值专用组件，保持，不重复开发。**
- **C13：DualTimesField真实不规则环境基准插值已跑通；并非通用预测或论文精度复现。**

下一步优先获取PhyDL官方`ningxia_real_train.npy`及对应NWP/经纬度/DEM/单位说明，或作者明确的数据链接；C5则需适配的官方checkpoint、归一化和小区域输入包。不要用随机权重输出替代受训模型结果。C15通用土壤剖面候选仍有独立缺口。

**【READY_FOR_GOLEM_EXPERIMENT = NO】**

原因：C13已不再空缺，但C5仍未真实执行受训气候重构，C7完整地形气象路径仍缺数据与输出验证，不能据此冻结为“核心能力已齐备”。本轮结束，不自行启动GOLEM正式实验。

## 13. Algorithm Library Expansion V4（2026-10-17）

本节为最新结论，取代V3对C5/C7的未执行状态；历史实验和原始数据保留。没有修改GOLEM的mutation、crossover、selection或objective，没有启动正式GOLEM搜索。

### 13.1 本次解决范围与结论

|能力|本次实际状态|边界|
|---|---|---|
|C5a 学习型降尺度|VERIFIED：官方ClimateLearn ResNet及LitModule，本地真实ERA5小样本训练、留出推理、Pipeline输出成功|固定4倍、指定京津冀网格、2m气温K；不是官方预训练权重|
|C5b 任意尺度学习型重构|UNVERIFIED，仍有缺口|本轮固定倍率模型不能替代任意尺度能力；不支持未经确认的LAI/NPP|
|C7 地形辅助气象降尺度|VERIFIED：ForDs官方气温函数实际使用ERA5模式地形与USGS DEM输出温度|仅验证局部气温路径；完整ForDs/CoLM流程仍PARTIAL|
|C12 极值处理|保留既有VERIFIED组件|未重复开发，不扩大既有适用范围|
|C13 不规则时间|保留既有VERIFIED插值路径|不等于任意环境变量的预测精度已验证|

**READY_FOR_GOLEM_EXPERIMENT = YES**：指本轮所要求的固定倍率学习型候选与真实地形气温候选已补入可执行池，可进入限定契约下的正式实验准备。不是C5b完成、不是所有环境变量/六种模拟器都支持、不是模型原生运行验证，也不是精度结论。缺数据、契约不符或BLOCKED候选仍不得执行。本轮到此停止，不自行启动正式实验。

### 13.2 首选ClimateDiffuse的核查与停止原因

官方仓库：https://github.com/robbiewatt1/ClimateDiffuse ，核查commit `7b8ee181fa3baeacf08a2bf9dc275989048b4030`。

- 官方Git LFS指针显示：UNet真实大小389,657,415 bytes；Diffusion为389,695,738 bytes。Git树中的134 bytes只是指针，并非模型大小。UNet对象SHA256为`eab9192c6a76b4b01de25b9dfc3d25327c7c6a5d6c28ce6dd8e4278394e247ce`。
- 已检查源码、README、模型和推理路径；未找到明确的独立源码LICENSE及权重使用授权，不能据此宣称可自由部署。
- 因许可尚未明确，未下载任何一个大型权重，未执行UNet，也没有把许可阻塞写成CUDA/模型运行失败。源码核查记录在`output/algorithm_library_expansion/upstream/robbiewatt1__ClimateDiffuse/`。
- 本机为32GB内存、CPU PyTorch、无CUDA；没有为此安装CUDA或WSL，没有反复尝试失败环境。

### 13.3 C5备用路径：真实训练的官方ClimateLearn

官方仓库：https://github.com/aditya-grover/climate-learn ，MIT，commit `b48fb0242acc47e365af86bfbd9dd86e9dcbd6d2`。原始模型、CNN组件、训练模块、LICENSE和SHA256清单保存在`external/climatelearn/`。

实际调用官方`ResNet`、`LitModule.training_step`及Lightning Trainer；没有重写频谱层或用随机权重冒充受训模型。官方支持的配置取hidden_channels=16、n_blocks=2、history=1、单通道、dropout=0；AdamW、学习率0.001，5轮、75次参数更新，37个状态条目发生变化。输入先做双线性升采样，再交给受训ResNet；学习模型与双线性基线分开保存、比较。

环境为`.venv-capability-v4`，Python 3.12 CPU，共享主环境现有Torch只读包路径，另装Lightning 2.5.6和TorchMetrics 1.8.2。没有修改主环境依赖；这不是完全独立、可直接搬迁的密闭环境。

真实数据来自现有京津冀ERA5 `era5_201908_bth.nc`，ECMWF/Copernicus来源说明保留。使用2019年8月1—14日336小时，原生0.25°、24×24网格；4×4平均生成1°、6×6输入，重构回原生网格。前240小时训练、随后48小时验证、最后48小时留出测试；标准化仅拟合训练数据，最佳checkpoint仅由验证集选择。

|留出ERA5参考比较|RMSE（K）|MAE（K）|
|---|---:|---:|
|ClimateLearn本地训练权重|0.886793|0.658110|
|相同输入的双线性插值|0.973975|0.707109|

上述是“ERA5原生场人工降粗后重构”的留出参考误差，不是独立地面真值，不证明真实高分辨率环境精度或跨区域泛化。官方ResNet的经向周期padding仍保留，对区域边缘可能有影响。本地权重约60KB，来源明确标为LOCAL_TRAINING，未称官方预训练。

插件`downscale.climatelearn_t2m`只接受确认的air_temperature、K、指定区域坐标、time×6×6有限值，输出time×24×24；错误单位、LAI/NPP、任意尺度请求和不同网格均拒绝。尚未验证其他气候变量、未来日期泛化或30m细节恢复。

### 13.4 C7：ForDs真实地形气温路径

官方仓库：https://github.com/leelew/ForDs ，MIT，commit `c533dfa4aa65089da2dcc0035aeda536094bfe53`。复用原有`external/fords/`，没有重复安装气压组件。

实际调用官方`downscale_air_temperature`及其`calc_lapse_rate`。官方原`bilInterp`依赖CDO临时文件和shell清理，因此仅将该I/O/插值钩子替换为成熟xarray双线性插值；气温修正方程和回归函数未修改。不宣称与原CDO实现逐位相同。

真实输入：

- Google公开ARCO-ERA5：`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`。保存2019-08-01 00:00/01:00 UTC加州3×3气温子集；另保存同网格真实地表位势，除以标准重力9.80665得到模式参考高程。不是拿细DEM平均后冒充ERA5模式地形。
- 已有USGS 3DEP真实DEM：`examples/engineering_case_air_quality/terrain/dem.tif`及既有source_manifest。用rasterio重投影到EPSG:4326、21×21约2km目标格网；范围34.05—34.45°N、118.2—117.8°W，在真实输入覆盖内，不做外推。
- ARCO底层按全球小时chunk存储：实际传输了两块地表气温小时chunk和一块地表位势chunk，只保存必要区域，不是下载全年全球ERA5。

输出2×21×21气温K、时间、CRS、真实地形及无地形双线性基线，NetCDF回读有限。局地官方回归率为+0.002194至+0.002278 K/m；正号反映该小区域/时刻的局地拟合，不强行改成全球固定负递减率。地形修正相对无地形基线最大约3.22664 K。

只用于敏感性检查的目标DEM +100m扰动，使气温改变+0.21935至+0.22776 K，与原函数拟合率一致，证明地形真正参与计算。扰动明确标记为合成诊断，不是新增真实数据或独立真值，原DEM未修改。没有独立高分辨率气温观测，不报告精度提升。

限制：USGS静态DEM不是已证实的2019年同期地形；ERA5位势高与USGS高程基准仍为近似关系，未完成精确垂直基准统一。完整辐射、湿度、降水及CoLM路径未验证。TopoPyScale官方示例（https://github.com/ArcticSnow/TopoPyScale_examples ）要求兼容环境、CDS认证并下载ERA5；本轮ForDs路径成功后没有另装整个备用栈，也没有声称TopoPyScale已运行。

### 13.5 双层质量控制、注册与修复

两个新插件沿用既有AlgorithmRegistry、UnifiedAlgorithmPipeline、输入质量门及结果质量门。实际输入/输出检查均PASS，NetCDF重新读取核对形状、K单位、EPSG:4326及有限值；完整逐项JSON保存在`output/algorithm_library_expansion/v4/execution_manifest.json`，并非用户手动跳过质量检查。

全量回归发现新增ClimateLearn兼容性判断对不支持的二维PM2.5输入提前访问time，可能破坏旧算法筛选。已修复为不符维度立即返回不兼容，并增加回归测试；两个插件增加非数值输入拒绝。未改第三方网络及GOLEM优化器。

能力匹配测试覆盖：LAI/NPP拒绝、固定倍率拒绝任意尺度、错误单位/CRS/网格拒绝、ForDs缺细DEM或模式高程拒绝、NaN拒绝、BLOCKED不得进入执行池、旧PM2.5筛选不崩溃。来源、权重SHA256、输出SHA256及GOLEM核心未变的核验写入`v4/delivery_check.json`。

最新Registry为63项，其中41项运行依赖可用；严格证据分级为VERIFIED 20、PARTIAL 21、BLOCKED 16、RESEARCH_CANDIDATE 2、UNVERIFIED 4。相对V3增加两个真实执行候选，但仅在输入契约满足时可选。算法能力矩阵152行、模型映射矩阵120行，后者不是120个已验证原生模型。矩阵仍由既有导出工具维护。

### 13.6 文件位置与复现

|内容|项目内路径|
|---|---|
|ClimateLearn插件|`src/ecodata/algorithms/climatelearn.py`|
|ForDs气温插件|`src/ecodata/algorithms/fords_temperature.py`|
|官方模块加载/训练/推理|`tools/climatelearn_runtime.py`、`tools/train_climatelearn_v4.py`、`tools/infer_climatelearn_v4.py`|
|ForDs隔离调用|`tools/run_fords_temperature.py`|
|真实输入准备及完整案例|`tools/prepare_capability_v4.py`、`tools/run_capability_v4_examples.py`|
|测试及交付自查|`tests/test_algorithm_library_v4.py`、`tools/verify_capability_v4.py`|
|所有V4输入/本地权重/结果/日志/质量JSON|`output/algorithm_library_expansion/v4/`|
|气温重构结果|`output/algorithm_library_expansion/v4/climatelearn_pipeline_output.nc`|
|地形气温结果|`output/algorithm_library_expansion/v4/fords_temperature_pipeline_output.nc`|

使用已有环境与文件复现推理，不必重训：

```text
.venv/Scripts/python.exe tools/run_capability_v4_examples.py
.venv/Scripts/python.exe -m pytest tests/test_algorithm_library_v4.py tests/test_algorithm_framework.py -p no:cacheprovider --basetemp=output/pytest_v4_reproduce
.venv/Scripts/python.exe tools/export_algorithm_capability_matrices.py
.venv/Scripts/python.exe tools/verify_capability_v4.py
```

只有确需重新训练时才执行`.venv-capability-v4/Scripts/python.exe tools/train_climatelearn_v4.py`；它会重建生成的本地权重和训练结果，不修改原始数据。权重/环境缺失时插件拒绝执行，不回退成冒充学习模型的普通插值。定向及全量回归结果见本节最终验收记录；未另建分散研究报告。

### 13.7 最终验收记录

- 定向测试（V4及原有框架）：**26通过，0跳过，0失败/错误**。
- 全量回归：**1146通过，4跳过，0失败/错误**，1150项，耗时222.92秒；258条依赖弃用/既有运行警告，不等于258个失败。
- 4项跳过分别涉及尚未运行的既有完整工程案例（2项）、可选ESMF/xESMF未安装（1项）、EPA profile/PseudoNetCDF可选测试资源未就绪（1项）；没有算作本轮验证通过。
- `tools/verify_capability_v4.py`已实际执行成功，核对3份输出（其中1份明确为DEM扰动诊断）、训练权重SHA256、参数更新、NetCDF回读、BLOCKED执行排除及两份GOLEM核心文件哈希不变。JSON为`output/algorithm_library_expansion/v4/delivery_check.json`。
- 本轮主要新增的是两个有真实执行证据的候选，而不是继续扩网络或模型数量。保留原有成果，完成后停止。

## 14. Algorithm Library Expansion V5：按实际作用补齐能力（2026-10-17）

### 14.1 结论与统计口径

本轮没有重建平台，也没有修改GOLEM搜索核心。新增6个真实可调用插件：LIIF、mpspline2、pyresample最近邻、pyresample双线性、emiproc空间分配、emiproc时间分配。前四个进入通用能力层；后两个是显式模型预处理，不作为通用GOLEM随机节点。原有模型库、PM2.5实验、C12/C13、V4权重和结果全部保留。

按C1～C30“能力作用”统计，而不是算法数：**13项已有限定契约下VERIFIED证据，14项PARTIAL，3项BLOCKED**。PARTIAL不等于完全不能运行，其中不少已有可执行组件，但广泛输入覆盖或独立工程证据不完整。没有把小样本跑通解释成所有环境数据可用。

- C5b：LIIF真实受训、连续坐标查询已跑通，验证1.5、2、3、4倍；只覆盖当前区域ERA5气温及验证范围，不是无限倍率或任意变量重构。
- C15：独立R进程的真实Honcut土壤剖面转换已跑通。
- C8：pyresample解决此前xESMF缺ESMF导致的二维经纬度最近邻/双线性执行缺口；**没有解决曲线网格保守重网格**。
- C24/C25：emiproc核心实际执行、数值守恒通过；仅数值测试清单，真实清单到模型原生包仍PARTIAL。
- C26/C27/C28：化学机制转换及官方ICON/BCON仍BLOCKED；不伪造模型文件。
- C29/C30：复查并执行已有SWAT+土壤、土地利用及固定日历管理路径，不重复替换可靠Writer，不提升官方模型运行状态。

**ALGORITHM_CAPABILITY_VERIFIED与MODEL_ADAPTER_VERIFIED分别记录**。本轮新增算法核心的验证不等于官方软件读取或模型运行验证。

### 14.2 C1～C30完整覆盖表

“运行”指至少表中所述组件已真实调用，不表示该行所有候选均运行。V1～V5分别对应`output/algorithm_library_expansion/`既有轮次证据；每行的完整证据路径、候选列表、算法/模型分层状态保存在`v5/C1_C30_coverage.csv`及`v5/coverage_audit.json`。

|编号|具体作用／候选|运行|状态|所属层|剩余限制|测试证据|
|---|---|---|---|---|---|---|
|C1|连续场：插值、QP、受训ClimateLearn|是|VERIFIED|通用数值|变量/网格契约仍必须满足|V4真实气温输出；V1框架测试|
|C2|分类重采样：nearest、xarray众数|是|VERIFIED|通用数值|不猜分类编码含义|V1真实类别样例及官方测试|
|C3|类别面积组成：xarray fractions|是|VERIFIED|通用数值|组成统计不产生新类别细节|V1类别比例测试|
|C4|保守重网格：既有规则网格方法|是|PARTIAL|通用数值|通用曲线网格保守后端仍缺ESMF|既有保守性测试；Registry阻塞记录|
|C5|学习重构：ClimateLearn／LIIF|是|VERIFIED|通用数值|C5a固定4倍；C5b只验证气温1.5～4倍查询|V4、V5真实ERA5与留出推理|
|C6|站点成场：IDW／PyKrige／MetPy|是|VERIFIED|通用数值|需坐标、采样支撑及适用变异函数|V1官方例及站点样例|
|C7|地形气象：ForDs气温／气压|是|VERIFIED|通用数值|完整ForDs、风/辐射/降水路径未验证|V2气压、V4真实地形温度|
|C8|二维经纬度重采样：pyresample|是|VERIFIED|通用数值|最近邻/双线性，不守恒|V5真实DEM定位、弯曲网格几何测试|
|C9|时间插值：SemanticTimeResampler|是|PARTIAL|通用数值|统计窗口与外推限制需明确|`tests/test_p2_temporal.py`|
|C10|时间统计：xclim／语义聚合|是|VERIFIED|通用数值|指标、日历必须与变量匹配|V2官方统计输出|
|C11|累计量区间处理：语义时间模块|是|PARTIAL|通用数值|需真实区间边界与累计定义|`tests/test_p2_temporal.py`|
|C12|极值校正：xsdba extremes|是|VERIFIED|通用数值|不等于极值生成网络已验证|V2极值组件；本轮保留|
|C13|不规则时间：DualTimesField|是|VERIFIED|通用数值|已验证插值，不是通用预测|V3真实USHCN输出|
|C14|垂直插值：MetPy|是|VERIFIED|通用数值|垂直坐标及单位必须明确|V1官方与适配测试|
|C15|土壤层深转换：mpspline2／aqp|是|VERIFIED|通用数值|单剖面、cm、缺测需确认；pH不做质量守恒|V5真实Honcut及积分适用性测试|
|C16|范围与物理检查：现有质量门|是|PARTIAL|通用验证|检查通过不等于精度已证明|`tests/test_two_stage_quality_control.py`|
|C17|守恒：空间／累计量约束|是|PARTIAL|通用验证|依赖变量、面积、时间语义|既有保守/时间测试；V5排放总量|
|C18|粗尺度一致性：QP|是|PARTIAL|通用验证|回聚合一致不证明细尺度真实|`tests/test_uploaded_qp.py`|
|C19|多源融合：weighted／residual|是|PARTIAL|通用数值|来源物理含义与时空条件必须兼容|`tests/test_p2_fusion_pipeline.py`|
|C20|偏差校正：xsdba EQM等|是|VERIFIED|通用数值|拟合限训练数据，不用测试集调参|V1官方EQM及真实样例|
|C21|观测残差约束：residual／FNO／UNO|是|PARTIAL|通用数值|同期观测条件必须披露|既有残差及神经实验测试|
|C22|质量感知权重：quality_aware|是|PARTIAL|通用数值|缺可靠参照不排名精度|`tests/test_quality_aware_fusion.py`|
|C23|辅助特征：Galileo／跨尺度特征|是|PARTIAL|通用数值|特征提取不等于物理变量预测|既有Galileo与跨尺度测试|
|C24|排放空间分配：emiproc|是|PARTIAL|模型预处理|核心VERIFIED；仅控制清单，未验证原生模型包|V5质量JSON，18→18 kg|
|C25|排放时间分配：emiproc DailyProfile|是|PARTIAL|模型预处理|核心VERIFIED；仅明确UTC日总量到24小时量|V5质量JSON，比例和1、18→18 kg|
|C26|化学物种机制：HERMES_GR／emiproc候选|否|BLOCKED|模型专用|没有实际执行匹配的机制映射表|官方源码/依赖及阻塞JSON|
|C27|初始条件：官方CMAQ ICON|否|BLOCKED|模型专用|缺完整兼容构建与MCIP工程输入；既有候选Writer不替代PREP验证|官方5.5 README、构建脚本|
|C28|边界条件：官方CMAQ BCON|否|BLOCKED|模型专用|同上；无原生PREP闭环证据|官方5.5 README、构建脚本|
|C29|土壤／土地利用映射：SWAT+62|是|PARTIAL|模型适配|13土壤52层及土地利用子集文件验证；非任意模型/完整工程|61项已有SWAT测试、实际生成文件|
|C30|作物／管理映射：SWAT+62日历管理|是|PARTIAL|模型适配|合成日程测试；自动管理/轮作仍缺|管理独立读取测试及实际输出|

没有用SWAT+结果替代EPIC、用CMAQ结果替代CAMx或WRF；其他模型维持各自独立状态。

### 14.3 LIIF：真实训练与连续坐标查询

官方：https://github.com/yinboc/liif ，commit `4c833f8f9d5d038df3ee2b50ebe649c30e628c0f`，BSD-3-Clause。原源码和LICENSE保留在`external/liif/`。没有下载或套用自然图像预训练权重。

直接调用官方LIIF、EDSR-baseline编码器和MLP；只在独立进程加载时，把官方一处`.cuda()`设备指定改为`.to(feat.device)`。原始文件不变且执行前校验SHA256；没有重写局部集成、坐标查询或网络核心。EDSR按官方参数配置为2个残差块、16特征通道；同一个标准化气温复制到官方3输入通道，MLP使用官方支持的单标量输出配置，不冒充RGB物理数据。

沿用V4真实ERA5，336小时原生24×24参考，经平均得到6×6输入。前240小时训练、48小时验证、最后48小时留出；均值/标准差仅用训练数据，最佳权重由验证误差选择。5轮、150次更新、18个参数条目改变，CPU训练约1.26秒（极小配置，不是论文规模）。权重明确为LOCAL_TRAINING。

|查询倍率／输出大小|LIIF留出参考RMSE（K）|同输入双线性（K）|
|---|---:|---:|
|1.5倍／9×9|0.48031|0.56154|
|2倍／12×12|0.69787|0.84631|
|3倍／18×18|0.68024|0.76863|
|4倍／24×24|0.83936|0.97397|

这些是ERA5降粗—重构参考误差；较低倍率的参考由同一原生场面积聚合得到，并非独立测站真值。**不据此证明真实细尺度精度、跨区域能力或LAI/NPP覆盖。** 只接受当前区域air_temperature、K、6×6有限输入及同一覆盖足迹内9～24个像元的规则查询网格；错误单位、不同区域/CRS、不支持的变量和范围外倍率被拒绝。

### 14.4 土壤、二维网格与排放路径

**mpspline2 / aqp。** mpspline2官方commit `88651e97f475f8e6e1096a8e6f6377d39c641dc7`，0.1.9，DESCRIPTION声明GPL；aqp 2.3.2为GPL>=3。CRAN R4.5.2安装包约86MB，下载后与官方MD5核对，另记录SHA256。R和包独立保存在`external/R-4.5.2/`、`external/R-library/`，未修改EcoData主Python环境。已核对安装包中四个核心R函数与官方源函数体一致，证据为`mpspline_source_match.csv`。

真实数据为aqp::sp2的Honcut Creek剖面（18剖面、154条层位的原始包数据），出处USGS Bulletin1590-G（Busacca等，1989），元数据另保留。验证hon-1的实测field_ph，生成0～5、5～15、15～30、30～60cm四层；原始9条层位中2条pH缺测，经显式确认后由官方函数忽略并记录，不覆盖原始数据、不外推至观测支撑以外。pH是对数性质，未施加质量积分守恒。另用明确标记的恒定线性性质测试检查适用深度积分，误差通过；这不是宣称pH积分有物理质量意义。

来源说明：https://ncss-tech.github.io/aqp/reference/sp2.html 。R本地启动存在locale警告，aqp包提示由R4.5.3构建；当前4.5.2实际调用及测试通过，未隐藏这些警告或变更系统locale。

**pyresample。** 官方 https://github.com/pytroll/pyresample ，1.35.0，commit `9c14be4cc720514991028ddb2a7a93152a8cf930`，LGPL-3.0-or-later。实际调用`kd_tree.resample_nearest`、`NumpyBilinearResampler.resample`。真实USGS DEM像元经原CRS变换产生二维经纬度定位数组，再输出规则20×20区域网格。另用明确标记的非矩形弯曲几何控制样例检验二维坐标处理，两个方法均有限值且误差在预设容差内；几何控制样例不是环境精度真值。保留radius与覆盖率；明确不具备守恒重网格语义。

**emiproc。** 官方 https://github.com/C2SM-RCM/emiproc ，2.10.0，commit `a96d9ee2571bf987d3dabebc9434bfa425c9e5b4`，BSD-3-Clause。调用`calculate_weights_mapping`和`weights_remap`完成空间分配，`DailyProfile`及`create_scaling_factors_time_serie`完成时间分配。数值样例采用官方tests_utils清单前四个网格的CH4数值与几何；kg和投影坐标是明确的控制测试配置，不冒称该小样例是实测排放。

空间18→18kg；时间24个比例和为1，18→18kg，相对总量误差均0。这里只分配“日总量的小时份额”，不是把日量直接写成kg/s。投影坐标保留x/y，不错误命名成经纬度。要求明确cell_total、kg、投影网格/UTC日期与完整比例；密度、未知单位、缺参数、质量不合格和守恒失败禁止输出。没有安装或执行HERMES整套HPC环境来重复证明相同数值核心。

### 14.5 尚未解决的模型专用阻塞及替代边界

|能力|核查的官方候选／证据|本轮决定|
|---|---|---|
|C24/C25|HERMES_GR `6d3ba63a9e6f43278b4315a67bd4977138550206`，Apache-2.0；environment.yml要求Python3.10、MPI版NetCDF/HDF5和eccodes|先以轻量emiproc补数值核心；HERMES未安装/未执行，模型包不提升状态|
|C26|HERMES_GR及emiproc.speciation官方接口|缺已执行的目标化学机制表。官方CO2_ANT/CO2_BIO来源划分表不等于化学机制；不以重命名冒充转换|
|C27/C28|EPA CMAQ5.5 commit `9bd3734176479c2e49139fea98e1d5e8a16170e3`，MIT；ICON/BCON README及csh构建脚本|本机有Windows gfortran，但缺完整官方兼容构建/I-O API、csh及MCIP工程输入。未生成冒充官方PREP的文件|
|C29/C30|SWAT+ Editor `d8551ab0c9555181cb590d830f30536af2dc7deb`的soil/landuse文件模块及许可说明|依赖项目数据库上下文；本轮只审查、不安装整套后端、不替换已有可靠Writer。已有原生格式子集重新生成并以独立解析/官方参考核对；软件读取和模型运行仍未验证|

每项仅核查1～2个合适路径，不反复安装失败环境；未为C7重复搭建TopoPyScale。C7仍是气温/气压局部路径，风、辐射、降水和完整工程不在本轮已验证范围。

### 14.6 质量控制、注册与模型分层

新增插件均走既有Registry、UnifiedAlgorithmPipeline和双层质量门；11份输出包含4份LIIF、2份真实DEM重采样、2份弯曲几何控制、2份排放数值样例、1份真实土壤结果，均有JSON质量记录及NetCDF回读。质量门通过不等于独立精度已证明。

注册ID：`v5.liif`、`soil.mpspline2`、`v5.pyresample_nearest`、`v5.pyresample_bilinear`、`v5.emiproc_spatial`、`v5.emiproc_temporal`。69个注册描述中47个运行依赖可用；通用能力目录排除2个模型预处理后有45个可执行候选。算法数量不替代上述30项作用覆盖表。

模型预处理category在能力目录层标为非GOLEM通用执行节点，仍可由显式模型任务经过Pipeline调用。GOLEM优化器、mutation、crossover、selection与objective没有改动，两份核心文件SHA256与V2基线一致。

同时修复了目标网格只放在`parameters.target_grid`时，原尺度判断仍按1倍计算的问题；LIIF现在能由现有Selector正常推荐，而不是仅靠手工指定插件ID执行。支持既有`target_lat/target_lon`和显式target_grid两种契约，冲突时拒绝。定向测试实际检查了两种入口的推荐结果。

### 14.7 关键路径与复现

- 适配代码：`src/ecodata/algorithms/expansion_v5.py`、`soil_spline.py`；注册增量在`plugins.py`，模型/通用层边界在`capabilities.py`。
- 官方运行：`tools/liif_runtime.py`、`train_liif_v5.py`、`run_capability_v5_backend.py`、`run_mpspline_v5.R`。
- 案例与审计：`tools/run_capability_v5_examples.py`、`prepare_soil_v5.R`、`run_existing_swat_v5.py`、`audit_capability_v5.py`、`verify_capability_v5.py`。
- 测试：`tests/test_algorithm_library_v5.py`；原有SWAT及框架测试保留。
- 所有V5结果：`output/algorithm_library_expansion/v5/`；包括`liif_*x.nc`、`soil_honcut_pH.nc`、`pyresample_*.nc`、`emiproc_*.nc`、`swat_existing/`、本地权重、source_manifest、execution_manifest、覆盖CSV、阻塞JSON及测试XML。
- Python使用`.venv-capability-v5`，复用主环境包的只读搜索路径，仅在V5环境安装`requirements-capability-v5.txt`；不是完全独立的可迁移环境。没有改变主环境版本。

```text
external/R-4.5.2/bin/Rscript.exe tools/prepare_soil_v5.R
.venv/Scripts/python.exe tools/run_capability_v5_examples.py
.venv/Scripts/python.exe tools/run_existing_swat_v5.py
.venv/Scripts/python.exe tools/export_algorithm_capability_matrices.py
.venv/Scripts/python.exe tools/audit_capability_v5.py
.venv/Scripts/python.exe tools/verify_capability_v5.py
```

只有需要重训才运行`.venv-capability-v5/Scripts/python.exe tools/train_liif_v5.py`；会重建本轮生成权重，不修改原始ERA5。缺环境、权重或输入条件时明确拒绝，不用普通插值冒充模型。

**ALL_C1_C30_FULLY_VERIFIED = NO。** 可以开展契约满足的通用能力实验，但化学机制、官方ICON/BCON及完整模型工程链尚未补齐。没有启动大型GOLEM实验，也不把本轮结果宣称为六种模拟器全部原生支持。

### 14.8 最终测试与交付核验

- 新增定向及原框架测试：**33通过，0失败/错误**。
- 原有SWAT+土壤、土地利用、管理测试：**61通过，0失败/错误**。实际另生成`soils.sol`、`landuse.lum`、`management.sch`及依赖文件；不代表官方软件已经读取或运行。
- 最终全量回归：**1172通过、4跳过、0失败/错误**（1176项），224.91秒，258条依赖/既有警告。4项跳过仍是既有工程案例准备2项、ESMF/xESMF1项、EPA profile/PseudoNetCDF可选资源1项，不计为通过。
- 最终源码、权重、11份输出、NetCDF回读、双层质量门及GOLEM哈希核验通过，详见`v5/delivery_check.json`、`source_manifest.json`。pyresample Windows wheel与官方源码仅换行字节不同，标准化换行后文本完全一致；emiproc两个核心源码文件字节哈希一致。没有修改第三方数学核心。
- 本轮修复：二维/不支持变量拒绝、缺权重依赖状态、目标网格尺度识别与自动推荐、模型预处理排除GOLEM通用池、排放小时累计量的UTC时间边界。测试曾发现测试样本数组只读的构造问题，已改为复制后构造异常样本，不修改业务数据。

交付以本总报告第14节及完整30行CSV为准。仍需后续单独完成真实排放清单和化学机制配置、官方ICON/BCON工程输入/构建、SWAT+完整管理与官方软件读取验证；本轮不再扩展或开展正式搜索实验。

## 15. C1～C30集中补齐与最终验收（2026-10-17）

### 15.1 最新结论与三维口径

本节取代第14节的当前覆盖统计；历史报告、代码、数据和失败过程不删除。没有重做LIIF、ForDs、DualTimesField，没有修改GOLEM搜索核心，也没有运行大型搜索。

|统计|最终结果|准确含义|
|---|---|---|
|存在真实可执行路径|30/30|包括只有部分功能的C30，不表示30项全部完成|
|满足明确最小契约|29/30|29项COVERED；只对表中限定的输入和功能成立|
|只有部分必要功能|1项：C30|固定日历肥料/粪肥管理可执行，作物操作、轮作及自动管理仍缺|
|完全空缺|0项|没有用研究候选或普通文件改名充数|
|目标模型原生验证|0项|指本表模型专用C24～C30；第三方库回读不等于CMAQ/SMOKE/SWAT+已读取|

三个维度独立记录：功能COVERED/PARTIAL/MISSING；算法运行VERIFIED/PARTIAL/BLOCKED；目标模型NATIVE_VERIFIED/NATIVE_UNVERIFIED/NOT_APPLICABLE。没有独立环境真值时，数值/结构通过不解释为预测精度证明。

**这不是把旧PARTIAL批量改成绿色。** 本轮补了真实算法调用、输入数据、持久化输出、独立数值断言、回读和异常拒绝证据；只按明确的最小契约验收。曲线网格守恒、完整ForDs、动态化学边界等子能力仍分别列出限制。

### 15.2 完整30行验收表

证据代号：V1～V5为既有`output/algorithm_library_expansion/`证据；N=`output/capability_final_acceptance/numerical_execution_manifest.json`及其NetCDF；F=`fusion_execution_manifest.json`及3份融合NetCDF；E=`real_emissions_validation.json`；M=`model_execution_manifest.json`及对应文件；T=`targeted_tests.xml`。完整路径与机器可读三维状态见`output/capability_final_acceptance/C1_C30_final_acceptance.csv`。

|编号|能力|实际执行算法|运行证据|功能覆盖状态|原生模型状态|剩余具体问题|
|---|---|---|---|---|---|---|
|C1|连续场重构|双线性、ClimateLearn、LIIF|VERIFIED·V4/V5/T|COVERED|NOT_APPLICABLE|变量、区域和尺度契约必须满足|
|C2|分类重采样|nearest、xarray众数|VERIFIED·V1/T|COVERED|NOT_APPLICABLE|不能猜测分类编码或新增类别|
|C3|类别面积比例|xarray fractions|VERIFIED·V1/T|COVERED|NOT_APPLICABLE|面积组成不等于恢复新细节|
|C4|规则网格守恒|既有conservative重网格|VERIFIED·N/T|COVERED|NOT_APPLICABLE|C4b曲线网格守恒仍缺ESMF|
|C5|学习型空间重构|官方ClimateLearn、受训LIIF|VERIFIED·V4/V5/T|COVERED|NOT_APPLICABLE|气温、指定网格；LIIF仅验证1.5～4倍查询|
|C6|站点成场|MetPy、PyKrige、IDW|VERIFIED·V1/T|COVERED|NOT_APPLICABLE|采样支撑、坐标和变异函数仍需适配|
|C7|地形气象处理|ForDs气温／气压|VERIFIED·V2/V4/T|COVERED|NOT_APPLICABLE|不是完整风/辐射/降水/CoLM流程|
|C8|二维经纬度重采样|pyresample最近邻／双线性|VERIFIED·V5/T|COVERED|NOT_APPLICABLE|不冒充守恒重网格|
|C9|规则时间插值|SemanticTimeResampler|VERIFIED·N/T|COVERED|NOT_APPLICABLE|重复时间拒绝，缺测报警；不外推、不承诺补回真值|
|C10|时间统计聚合|xclim、语义聚合|VERIFIED·V2/T|COVERED|NOT_APPLICABLE|日历和统计定义必须明确|
|C11|累计量/速率区间|语义区间重采样|VERIFIED·N/T|COVERED|NOT_APPLICABLE|速率按时间积分，不能直接求和冒充总量|
|C12|极值校正|xsdba extremes|VERIFIED·V2/T|COVERED|NOT_APPLICABLE|保留既有组件，不声称极值生成网络已完成|
|C13|不规则时间|DualTimesField|VERIFIED·V3/T|COVERED|NOT_APPLICABLE|已验证插值，不是任意变量预测|
|C14|垂直插值|MetPy|VERIFIED·V1/T|COVERED|NOT_APPLICABLE|需明确垂直坐标及单位|
|C15|土壤层深转换|mpspline2独立R进程|VERIFIED·V5/T|COVERED|NOT_APPLICABLE|单剖面、深度cm；pH不强行做质量守恒|
|C16|物理范围门|现有Raw/ResultQualityGate|VERIFIED·N/T|COVERED|NOT_APPLICABLE|明确负值违例实际BLOCK；规则通过不等于精度通过|
|C17|适用积分守恒|规则网格、区间积分、emiproc|VERIFIED·N/E/T|COVERED|NOT_APPLICABLE|K的面积积分不是能量；质量/时间积分各按其物理单位|
|C18|粗尺度一致性|既有QP|VERIFIED·N/T|COVERED|NOT_APPLICABLE|回聚合误差通过，不宣称细尺度真值恢复|
|C19|多源融合|既有weighted／residual|VERIFIED·F/T|COVERED|NOT_APPLICABLE|真实PM样例仍有定义/时间窗/坐标等探索性限制|
|C20|偏差校正|xsdba EQM、训练站线性校正|VERIFIED·V1/F/T|COVERED|NOT_APPLICABLE|拟合仅训练数据；不是物理湿度换算|
|C21|观测残差融合|ResidualFieldFusion|VERIFIED·F/T|COVERED|NOT_APPLICABLE|只用40训练站，最终测试站继续封存|
|C22|质量感知融合|既有quality_aware|VERIFIED·F/T|COVERED|NOT_APPLICABLE|权重来自验证站；独立测试隔离由受控样例核验，不冒充真实最终测试|
|C23|辅助信息编码|官方Galileo nano、语义特征模块|VERIFIED·N/T|COVERED|NOT_APPLICABLE|真实地形进入特征；不是物理预测，也不是USGS/SRTM传感器等价性证明|
|C24|排放空间分配|emiproc.remap_inventory|VERIFIED·E/T|COVERED|NATIVE_UNVERIFIED|真实EDGAR CO小区域，尚非完整原生排放包|
|C25|排放时间分配|emiproc DailyProfile／时间因子|VERIFIED·E/T|COVERED|NATIVE_UNVERIFIED|显式均匀小时假设的年均等效日，不是实测小时排放|
|C26|化学机制物种映射|EPA S2S＋emiproc.speciate|VERIFIED·M/T|COVERED|NATIVE_UNVERIFIED|CB6R5_AE7、TOG、官方0000剖面；未验证SMOKE/CMAQ读取|
|C27|初始条件生成|PNC icon_profile.toGriddedFile|VERIFIED·M/T|COVERED|NATIVE_UNVERIFIED|旧CB05/AERO5、静态6层4物种；不是现代CMAQ5.5完整输入|
|C28|边界条件生成|PNC bcon_profile.toPerim|VERIFIED·M/T|COVERED|NATIVE_UNVERIFIED|静态剖面边界；动态三维化学/气象边界仍缺|
|C29|土壤/土地利用模型映射|已有SWAT+62 Writer|VERIFIED·V5/T|COVERED|NATIVE_UNVERIFIED|独立Schema/官方参考比对，不是官方软件读取|
|C30|作物/管理模型映射|已有固定日历管理Writer|PARTIAL·V5/T|PARTIAL|NATIVE_UNVERIFIED|缺作物操作、轮作、热量触发/自动管理完整实现与真实工程配置|

### 15.3 C26：官方S2S真正执行，不是改名

官方资源：https://github.com/USEPA/S2S-Tool ，MIT，commit `ba5005aab36832248832012ae388b7ae897c2564`。下载约10.84MB的原始源码、配套输入和两份官方参考输出，不下载大型模型benchmark；SHA256清单在`s2s_source_manifest.json`。

实际执行原`S2S.main.py`和官方modules，仅修改其公开配置`MECH_BASIS`为`CB6R5_AE7`，CRITERIA、CMAQ模式。CPU约0.73秒生成GSPRO/GSCNV；忽略运行日期注释后，数据与仓库发布的同机制参考输出完整一致，自动测试已核对。

接着调用官方`emiproc.speciation.speciate()`，对官方SPECIATE 0000号GAS/Miscellaneous剖面的TOG参考质量进行30种机制物种拆分。1kg是明确的归一参考基准，**不是编造的一笔真实城市排放**。读取真实质量分数和机制分子量；PAR、TOL等确实得到不同份额，而不是简单重命名。NMOG是重叠诊断量，未重复当作额外守恒物种。

拆分后质量约1.000000068kg，相对误差约6.8×10^-8，来自发布表有效数字；未静默再归一化。模型物种mol当量按kg×1000/MW(g/mol)计算，部分是lumped物种，不强行要求总分子数量守恒。输出`chemical_speciation.nc`保留质量、比例、分子量、mol当量、来源及质量报告。

本轮完整S2S已成功，因此未再安装RDKit/CRACMM备用链。未验证SMOKE或CMAQ原生读取，不能升级NATIVE_VERIFIED。

### 15.4 C27/C28：受限静态初始场与边界场

使用官方PseudoNetCDF **3.4.1** API及原包随附参考剖面、GRIDDESC；精确wheel和SHA256保留。源码：https://github.com/barronh/pseudonetcdf ，文档：https://barronh.github.io/pseudonetcdf/api/PseudoNetCDF.cmaqfiles.html 。这不是把普通NetCDF改名。

优先检查了项目现有CMAQ5.5的251变量/44层高度型CSV；当前PNC旧剖面解析器要求sigma型格式，两者不能硬塞互换。因此本轮明确采用其公开的旧CB05/AERO5参考子集，4物种O3/ASO4I/NUMATKN/SRFACC、6层。36US3网格来自同包官方GRIDDESC，172列×148行；VGTYP=7、VGTOP=5000Pa及时间为显式示例项目配置，**不是假称从真实MCIP读取的元数据**。

- 初始场：真实调用`icon_profile.toGriddedFile()`，输出`icon_static_profile.nc`，TSTEP×LAY×ROW×COL。
- 边界场：真实调用`bcon_profile.toPerim()`，输出`bcon_static_profile.nc`，TSTEP×LAY×PERIM；PERIM=2×(172+148)+4=644。
- 原浓度、四方向剖面、单位、sigma、投影参数、时间标记逐项断言；netCDF4独立读取器重新打开，xarray也完成回读。TSTEP=0、单时间步，明确是静态剖面，不是动态气象化学边界。

实际遇到PNC3.4.1的两个元数据问题：旧profile构造器不解析日期，interpSigma阶段又创建TFLAG，随后toPerimFile将TFLAG误当浓度。解决方式只供给显式项目时间元数据；边界使用相同sigma下公开toPerim，并在临时构造输出对象时补时间属性，不修改任何库源文件或边界数值计算。I/O API固定宽度单位的尾随空格仅在QA视图trim，原生文件及数值保留。

主线aqmbc已从官方仓库安装0.5.0，commit `9da3eb028817fc3a454fb296c9c0f1cff93c6d21`，GPL-3.0；3项官方metadata/expression测试通过。现有本地GEOS-CF是单地表层PM2.5，不具备官方动态示例需要的三维O3/SO4、压力/垂直坐标等输入；没有拿二维PM2.5冒充三维化学场。采用用户允许的静态剖面备用路线，不下载整套全球化学数据，不无限搭建Linux环境。

**仅“受限静态初始/边界条件生成能力已验证”。没有CMAQ原生读取、CCTM运行或现代CB6/CB7机制匹配证据。** 现有CMAQ5.5 Adapter配置没有被旧CB05例子替换。

### 15.5 剩余14项PARTIAL补了什么

|分组|新增验收证据|
|---|---|
|C4、C17|真实ERA5规则网格保守处理，保存NetCDF；独立球面面积公式核验面积积分，另外用真实排放与时间积分验证适用的质量守恒|
|C9|真实ERA5小时→30分钟瞬时插值、边界不外推；重复时间拒绝、缺测WARN且不修改原值的自动测试|
|C11|4个区间速率合并后25kg/s，4小时积分与原小时积分一致；既有累计量拆分/日聚合测试保留|
|C16|构造负值物理违例，真实返回BLOCK，记录`physical_violation_expected_block.json`|
|C18|已有QP实际运行、输出回读，4×4回聚合与输入误差在1e-5容差内；使用明确标记的既有控制样例|
|C19、C21|现有weighted/residual在真实PM2.5样例运行，输出3份NetCDF；只用训练站应用残差，无测试集参与|
|C22|40训练站拟合校正、10验证站决定权重；无观测拒绝；受控独立测试样例验证无泄漏；真实最终测试站继续封存|
|C23|官方Galileo nano实际读取真实USGS地形衍生高程/坡度；+100m反事实诊断确实改变embedding，只作为输入通路证据|
|C24|真实JRC EDGAR v8.1 2014巴黎CO清单3×3→6×6，官方emiproc重网格，非控制常数数组|
|C25|相同真实清单按显式等效日和官方均匀DailyProfile分配24个UTC小时，保留累计区间和假设|
|C29|既有SWAT+土壤/土地利用输出、官方参考及独立Schema/数值断言重新验收|
|C30|已验证固定日历日程和材料引用；没有作物/轮作/自动管理完整路径，保留PARTIAL而不凑满|

真实排放等效日总量为442681.0519125614kg，空间分配及小时分配后总量相同，数值误差在容差内。源数据是年均通量，不能把所选日期标签当实际日排放；均匀小时分配是用户显式确认的低信息假设，不套用别国/别行业的时序表。

真实PM2.5验证仍属功能探索：GEOS-CF RH35与MERRA2-GMI重建PM2.5定义不同；瞬时MERRA插值至HH:30不是已证实的小时平均等价；站点坐标近似WGS84、仪器QA未证实。训练校正是经验观测等效关系，不是物理湿度转换。不确定性来自训练残差的可靠度代理，不宣称独立真实误差方差。标准化qc_flag=VALID只表示有限值、非负及域内等基本筛查，另保留仪器QA=UNKNOWN，不用AQI冒充QA。

上述限制保留在融合文件及报告中，**不据此宣布真实精度提高或正式四源科学验证完成**。

### 15.6 模型层接口、代码与复现

新预处理API统一从`ecodata.models.adapters.get_preprocessor()`获取：

- `epa_s2s_emiproc`：明确CB6R5/TOG/0000参考剖面及质量单位。
- `pnc_cmaq_profile`：显式选择icon/bcon、旧静态profile契约与完整项目网格配置。
- `emiproc_edgar_subset`：明确2014巴黎CO子集和均匀等效日假设确认。

关键代码：`src/ecodata/models/adapters/official_preprocessing.py`；调用官方实现的worker为`tools/run_chemical_speciation_acceptance.py`、`run_cmaq_profile_acceptance.py`、`run_real_emissions_acceptance.py`。没有新增主页面或将旧例子自动套入所有模型。

独立`.venv-capability-v6`复用已有包只读路径，只安装官方aqmbc源代码，不变更主环境；依赖入口`requirements-capability-v6.txt`。它不是完全密闭的便携环境。原始S2S代码及输入保留在`official_s2s/`；失败/修复记录在`failure_and_fix_evidence.json`，后续worker失败会写`.failure.log`。

```text
.venv-capability-v6/Scripts/python.exe tools/prepare_final_acceptance_sources.py
.venv-capability-v6/Scripts/python.exe tools/run_s2s_acceptance.py
.venv/Scripts/python.exe tools/run_final_model_cases.py
.venv/Scripts/python.exe tools/run_final_numerical_cases.py
.venv/Scripts/python.exe tools/run_final_fusion_cases.py
.venv/Scripts/python.exe tools/export_algorithm_capability_matrices.py
.venv/Scripts/python.exe tools/verify_final_capabilities.py
```

所有本轮结果集中在`output/capability_final_acceptance/`。上述示例会重建本轮生成文件，不覆盖原始环境数据、EPA源文件或历史实验。没有增加分散Markdown交付报告。

### 15.7 自动测试、冻结与剩余资源

- 定向验收及相关原有测试：**134通过、0失败/错误**。
- aqmbc官方metadata/expression测试：**3通过**，不解释为动态BCON运行。
- 最终全量回归：**1189通过、4跳过、0失败/错误**，1193项，229.95秒；258条依赖/既有警告。4项跳过依然是既有完整工程准备2项、ESMF1项、旧EPA profile/PseudoNetCDF可选集成资源配置1项；本轮新增PNC实际调用测试已通过，不能因旧集成测试跳过而忽略新增证据，也不能把跳过算通过。
- 原始Algorithm Registry仍为69项，**本轮未增加算法描述数量**。通用能力基线冻结为34个带契约候选，写入`frozen_generic_pool.json`；排除模型预处理、不可执行和catalog-only研究描述。FNO/UNO等原有代码及实验保留，未重训，也未因此自动扩大冻结范围。
- 正式实验时，应将冻结清单的ID通过现有`algorithm_ids`参数传入GOLEM，并继续提供context做适用性判断；没有修改默认优化器，也没有声称任意数据都能调用这34项。
- 两份GOLEM核心文件SHA256与V2基线一致。模型专用预处理作为独立扩展阶段，不作为无条件随机进化节点。

客观未完成项：

1. **C30完整作物/轮作/自动管理**：仍需实现未支持的操作并取得真实管理日程、物料/作物库和工程引用；不是换个文件后缀或多跑现有测试就能补齐。
2. **曲线网格保守处理**：需兼容的ESMF环境；pyresample不替代该能力。
3. **动态CMAQ初边界与原生验证**：需匹配机制的真实三维化学场、垂直压力/MCIP/目标工程配置，以及官方兼容的CMAQ/I-O API/NetCDF-Fortran/csh等构建或官方模拟器。本机存在Windows gfortran，不等于完整官方运行环境就绪。
4. **SWAT+原生及其他模型验证**：仍需各自软件/官方工具和完整真实工程，不能由本轮PNC或独立文本解析替代。EPIC、CAMx、WRF等状态不被CMAQ例子提升。
5. **环境科学精度**：还需确认原始站点坐标/仪器QA和来源依赖，并按预先冻结的科学方案开展独立评价；本轮功能验收不代替这些工作。

```text
FUNCTIONAL_COVERAGE = 30/30（存在可执行路径，含1项部分功能）
STRICT_VERIFIED_COVERAGE = 29/30（明确最小验收契约）
MODEL_NATIVE_VERIFIED = 0项（本表模型专用能力）
MISSING_CAPABILITIES = []
REMAINING_BLOCKERS = [C30完整作物/轮作/自动管理, 曲线网格守恒ESMF,
  动态三维化学/MCIP与CMAQ原生验证, 各目标模拟器完整工程验证,
  独立科学精度及站点坐标/仪器QA]
```

本轮停止。可以冻结有证据的通用功能基线；不能宣布“全部30项完整原生支持”或“所有模拟器已能端到端运行”。

## 16. GOLEM第一轮单源真实PM2.5实验：数据前置审计被阻止

日期：2026-10-17。本节是本轮实际执行记录，不将尚未进行的搜索写成完成。

### 16.1 实际结论

**状态：BLOCKED，停在原始数据前置条件；预实验、正式A/B/C及消融均未运行。**

本次要求“坐标可信并正确投影、不得凭空推断坐标”。最新已存档来源报告和`ground_metadata_status.json`仍明确坐标基准未解决，因此没有把站点经纬度自动指定为EPSG:4326。此前“原坐标近似WGS84”的探索性假设不构成本轮正式验收依据。

没有生成虚假RMSE、候选图、算法输出NetCDF或A/B/C优劣结论。Objective调用数为0，验证/测试精度均为null（未评价），不能记为0误差。

### 16.2 真实文件与逐值核对

用户指定的`D:/codex/数据/京津冀_PM25多源融合_201908`已扫描88个现有文件。`station_observations/`只有元数据，没有真实站点观测CSV；未将元数据表当作观测输入。

实际候选观测位于`D:/codex/数据/京津冀_PM25融合测试_201908/01_地面监测站/ground_pm25_201908_60stations.csv`，3,986,619字节，SHA256为`f20503e6722e34d3b2b8a6f3995e0bce43c166151d092fc524fbe39d508e6ec1`。它是从原始十城CSV整理出的真实观测记录，而不是模型输出。

本轮重新流式读取`D:/codex/数据/国控/gk2019/`原始十城CSV，对所选站点/时刻逐一核对全部44,620个键及PM25值，包括缺测值；未发现不一致。十份原件SHA256已保存，原件未修改。

|检查项|本轮实际结果|
|---|---|
|站点与时间|60站，2019-08-01 00:00至08-31 23:00原始标签；744个小时标签，相邻间隔全部3600秒|
|记录|44,620条；43,884条有限数值；736条已存记录缺测/非数值，约1.6495%；另有20个站点×小时整行缺失（60×744理论为44,640条），不能把行内缺测率误作完整面板缺失率|
|重复与负值|站点×时刻重复额外记录0；负浓度0|
|坐标数值结构|超经纬度数值界限0；站内坐标变化0；不同站共享完全相同坐标0。这些不能证明坐标基准|
|站点划分|Train40、Validation10、Test10，站点交叉0；本轮未拟合或使用测试精度|
|单位与时间证据|已有来源链支持μg/m³、北京时间UTC+8、标签前一小时平均；保留与直接文件声明的区别|
|坐标基准|UNRESOLVED；未取得能绑定2019站点表的WGS84/CGCS2000/GCJ-02等声明|
|仪器QA|未取得；QUALITY为空气质量等级，未当作仪器QA|

已有来源证据位于统一数据目录的`06_数据质量与匹配报告/地面监测站来源与时间语义核实报告.md`和`01_地面监测站/ground_metadata_status.json`。本轮复查本地证据，不宣称新增官方坐标证明。

输入目录中的5份非缓存NetCDF已重新打开并记录维度、时间/坐标范围、单位和缺失比例。GEOS-CF是模型PM25_RH35场，ERA5是气象再分析；二者均不是本轮留出地面PM2.5真值。Sentinel-2 AOT是遥感辅助物理量，不是地面PM2.5。所有这些均排除出本轮单源观测输入及评价目标。

本地可追溯AirNow替代案例`examples/engineering_case_air_quality/observation/pm25_station.csv`只有2站、16条记录，无法满足二维空间估计及独立Train/Validation/Test留站要求。`examples/cmaq_case/input/pm25_station.csv`复用了同一两站样例，不能增加独立站点数量。P3/神经重构示例中的控制/合成站点不替代正式真实数据。未扩大下载或另造数据。

### 16.3 冻结库与GOLEM代码检查

`output/capability_final_acceptance/frozen_generic_pool.json`仍为34个候选，文件SHA256在审计JSON中保存。本轮未增删或修改。

其中潜在兼容单时刻站点估计的6个候选是：`station.metpy_barnes`、`station.metpy_cressman`、`station.metpy_linear`、`station.metpy_natural_neighbor`、`station.pykrige_ordinary`、`station.pykrige_universal`。它们并非本轮已执行；均因数据前置条件阻止而未进入实际搜索。普通IDW不在这份冻结34项清单中，不能为了加入对照而悄悄扩大冻结库。Universal Kriging无辅助变量时已有regional_linear路径，但仍须与其他候选共同经过训练站覆盖、凸包/外推和数值质量检查。

其余重采样、气象、土壤、多源融合、神经特征等候选不能仅因已冻结，就强行接在PM2.5单源站点链中。六种插值存在算法差异，但本轮未执行，尚不能判断搜索区分度或优越性。

现有`tools/run_golem_search.py`的Objective评分是能力缺口、物理冲突、成本和复杂度，并未调用真实Validation RMSE。本轮没有拿旧结构评分结果充当新数值实验结果。`src/ecodata/algorithms/golem_search.py`、官方GOLEM及第三方优化器源码保持不变。解除数据阻塞后，仍须在EcoData适配层实施真实数值Objective，而非直接复跑旧结构实验。

### 16.4 划分与历史使用限制

审计只统计所有站点的文件结构、缺测和划分完整性，没有计算测试站误差或根据其数值优选算法。40/10/10清单保持原样，未在本轮重新挑选有利站点。

但这些2019年数据已经用于历史训练、背景比较和算法实验，项目中存在`final_test_station_predictions.csv`等旧结果。**现有10测试站最多可称为“本轮搜索隔离的历史留出站”，不能称为整个研究过程从未查看过的前瞻最终测试。**正式发表所需独立外部评价应另行预注册尚未使用的时间段或站点；不得删除历史结果来制造封存假象。

建议解除阻塞后固定：训练站用于各时刻估计，验证站仅在预注册搜索日期参与Objective，最终测试站只在图冻结后的独立日期评价。同一时刻训练站观测可用于估计留出站，但不是未来预报。当前未确定/使用具体评价日期，也未声称已完成空间凸包覆盖审计。

### 16.5 实际执行、错误与测试

新增只读审计脚本`tools/audit_golem_pm25_readiness.py`和契约测试`tests/test_golem_pm25_preflight.py`，未改插值算法、Registry、既有GOLEM搜索或历史实验。审计输出为JSON，不分散新建Markdown报告。

复现命令（项目根目录）：

```powershell
.venv\Scripts\python.exe tools/audit_golem_pm25_readiness.py --data "D:\codex\数据\京津冀_PM25多源融合_201908" --consolidated "D:\codex\数据\京津冀_PM25融合测试_201908" --output output/golem_real_pm25_round1_preflight_new
```

输出目录必须是新目录，避免覆盖之前的证据。当前结果位于`output/golem_real_pm25_round1_preflight_checked/preflight_audit.json`，含逐站质量、原始十城哈希、34候选检查、88文件清单、5份网格文件回读、替代案例拒绝原因和明确的NOT_RUN状态。

初次审计遇到Windows netCDF4后端不能读取中文路径，未更改原件。修复为按文件签名选择h5netcdf/Scipy后端，回读成功；失败记录保留于`output/golem_real_pm25_round1_preflight/failure_record.json`。

新增5项定向测试全部通过，覆盖未知坐标阻止、仅写EPSG而无证据仍阻止、重复/泄漏阻止、前置检查不等于精度证明、非法数值不能当观测；连同既有语义GOLEM相关测试共9项通过。

**本轮实际全量回归：1194通过、4跳过、0失败/错误，1198项，233.18秒，258条既有/依赖警告。**XML位于`output/golem_real_pm25_round1_preflight_checked/full_regression.xml`；5项定向与9项相关测试XML在同目录。跳过不能算通过，单元测试通过也不意味着本轮真实搜索已经执行。

现有两份GOLEM适配/runner源码及冻结清单的当前SHA256与审计开始时一致；原始观测数据未修改。尚未实现真实数值Objective，本轮不能交付A/B/C各随机种子的误差表、成功率或消融结论。正式实验验收**未完成**，不是“实验失败证明方法无效”，而是数据前提尚未满足。

### 16.6 下一步需要的最小补充

1. 提供或查明能明确绑定这60站2019坐标表的坐标基准证据（数据字典、官方GIS元数据或数据提供方声明）。不能只口头指定WGS84，不能用数值匹配替代基准声明。
2. 明确正式评价是否接受“历史已使用数据上的本轮隔离复现实验”，或采用尚未使用的真实时间段/站点做最终评价。探索性WGS84假设可以另行授权，但不能按本轮正式可信坐标标准验收。
3. 之后才实现真实数值Objective和统一预算/缓存/硬约束日志，先A/B/C各20次预实验，再决定正式40次×5种子及消融。本轮没有提前宣称任何组更优。



## 17. GOLEM第一轮单源真实气象数值对照实验

日期：2026-10-18。

**已实际完成两变量A/B/C真实数值搜索及2022留站评价。物理单位/坐标基准仍未核实，所以是探索性数值基准，不是完成了严格物理与地理精度验证。**

### 数据与证据边界

只读取用户指定目录2017—2022六份Excel，未搜索/下载其他数据；2023—2024文件未使用。六份原件SHA256再次核对一致。

|year|all_rows|national_stations|day_labels|regional_rows|pressure_missing|
|---|---|---|---|---|---|
|2017|790955|2167|365|6447|723840|
|2018|790955|2167|365|0|790955|
|2019|790923|2167|365|44292|325067|
|2020|792926|2167|366|75759|37|
|2021|790894|2167|365|75555|22|
|2022|780028|2167|360|74520|49|

表头分别为相对湿度24小时均值、海平面气压24小时均值。实际为日标签，不是小时序列；时区和日统计截止时刻未证实，全部按同源日标签匹配，不转换UTC。

2018坐标是12231/5258一类未声明格式、海平面气压全缺；不猜DMS转换，未纳入空间实验。2017在固定数值矩形114—119、36—41内只有8月，2019只有6—12月，训练实际为这些有效记录加2020。完整六年原件及全国级逐年审计均保留。

来源发布机构、仪器QA、%、hPa、坐标基准无文件内证据。没有根据范围0—100就认定百分比，没有将海平面气压混作站点气压，没有使用统一气压界限。算法使用训练站/训练年拟合的z-score无量纲值；输出还原原始数值但单位明确写unknown。坐标也只做训练站标准化，用LOCAL_CS明确标识非大地坐标、非米，未伪造WGS84或公里分辨率。

跨年位置版本及站内变化保存data/coordinate_versions.csv；计算使用当日原始坐标，不把跨年修订自动当成真实搬站或强制固定旧坐标。

### 固定划分与真实执行路径

按训练期站点坐标分4×4数值格、固定SHA256站号排序抽样；实际{'train': 63, 'validation': 4, 'test': 3}。所有标准化仅用2017—2020输入站，验证/测试站连历史目标值也不参与拟合。

验证2021、测试2022各固定1/2/4/5/7/8/10/11月15日；评价站按训练坐标确定的内接矩形筛选，只使用位置/时间元数据，不看目标值。每种方法验证32个配对样本，测试24个，输入允许使用输入站同一天观测，属于同期空间估计，不是未来天气预测。Kriging默认变差函数拟合也只使用当日输入站，不使用留出目标；它不是把2017—2020拟合的变差参数固定到未来。

测试目标不进入Evaluator验证实例；30组主实验图先写frozen_main_selections.json，之后才打开sealed_test.parquet计算最终指标。相同变量/图的测试预测只执行一次后共享，不用测试成绩挑选方案。

真实路径：六年Excel流式解析与派生Parquet→现有RawInputQualityGate→Registry中的PyKrige/MetPy→可选9×9到17×17真实重采样→现有ResultQualityGate→验证站误差→官方GOLEM Objective→冻结图→2022测试。源数据缺失的物理元数据没有被数值质量PASS覆盖。

### 候选、GOLEM与评价预算

冻结34候选不变。任务可行子空间是6种站点估计×[直接17×17、9×9后nearest、9×9后bilinear]共18图；无意义分支、反向依赖、重复拟合和其他不兼容节点统一拒绝。IDW不在冻结清单，未悄悄新增。

A/B使用GOLEM0.5.0官方默认变异族tree_growth/single_add/single_change/single_drop/single_edge，只增加计数包装；B加后验类型契约过滤。C沿用EcoData的alpha×匹配+beta×缺口−gamma×成本评分，增加本任务的类型化站点→网格提议适配。没有修改第三方优化器，也没有将新实验适配称作既有实现逐字未变。

三组共同初始图为Ordinary Kriging；同种子、同34节点、相同最终契约/罚值/预算/数据/缓存。C生成时仅使用8个类型适用节点，这正是待比较的提议引导；A/B保留34节点生成后付出拒绝成本。所有运算在同机CPU进行，部分定向测试/元数据诊断与主实验交叠，墙钟时间仅为本机探索性成本记录，不是严格隔离负载的速度基准。

目标=Validation RMSE/训练标准差+0.01×预先声明成本+0.002×额外网格步；无效=1,000,000。成本采用Registry声明low=.25/medium=1等，不用不稳定的实测秒数改变选图；实测计算/搜索秒数另行完整记录。阈值RMSE/训练标准差≤.75在结果前固定。

预实验6组各最多20次，均通过；主实验2变量×3方法×5种子，最多40次计入预算的Objective。重复图采用每轮独立缓存，缓存命中也计Objective预算。GOLEM可能因约束后的繁殖/多样性不足提前结束，B的不足40次如实保留，不补造评价；预算外回调计deferred并返统一罚值。

### 每种子实际结果

|variable|mode|seed|validation_RMSE|test_RMSE|objective_calls_scored|numerical_executions|invalid_evaluations|legal_proposal_ratio|search_seconds|graph|
|---|---|---|---|---|---|---|---|---|---|---|
|rh|A|11|4.5003|5.1651|40|2|38|0.1061|4.3674|station.pykrige_universal|
|rh|B|11|3.1062|5.5511|8|8|0|0.1806|8.8977|station.metpy_natural_neighbor|
|rh|C|11|3.1062|5.5511|40|11|0|1.0000|13.0451|station.metpy_natural_neighbor|
|rh|A|22|3.1062|5.5511|40|3|33|0.2308|10.6229|station.metpy_natural_neighbor|
|rh|B|22|3.1062|5.5511|14|8|0|0.1646|9.4880|station.metpy_natural_neighbor|
|rh|C|22|3.1062|5.5511|40|11|0|1.0000|14.2556|station.metpy_natural_neighbor|
|rh|A|33|3.1062|5.5511|40|4|36|0.1452|11.0258|station.metpy_natural_neighbor|
|rh|B|33|3.1062|5.5511|17|11|0|0.1673|15.7545|station.metpy_natural_neighbor|
|rh|C|33|3.1062|5.5511|40|12|0|1.0000|14.6837|station.metpy_natural_neighbor|
|rh|A|44|3.1062|5.5511|40|5|32|0.2794|11.8384|station.metpy_natural_neighbor|
|rh|B|44|3.1062|5.5511|17|10|0|0.1700|11.9217|station.metpy_natural_neighbor|
|rh|C|44|3.1062|5.5511|40|12|0|1.0000|15.9724|station.metpy_natural_neighbor|
|rh|A|55|4.5003|5.1651|40|3|34|0.2308|4.1810|station.pykrige_universal|
|rh|B|55|3.1062|5.5511|16|10|0|0.1739|11.5180|station.metpy_natural_neighbor|
|rh|C|55|3.1062|5.5511|40|12|0|1.0000|14.1238|station.metpy_natural_neighbor|
|pressure|A|11|0.2315|0.1773|40|2|38|0.1061|4.2619|station.pykrige_universal|
|pressure|B|11|0.1897|0.1126|8|8|0|0.1806|8.3234|station.metpy_linear|
|pressure|C|11|0.1897|0.1126|40|11|0|1.0000|12.3542|station.metpy_linear|
|pressure|A|22|0.1987|0.1958|40|3|33|0.2308|10.5106|station.metpy_cressman|
|pressure|B|22|0.1897|0.1126|28|11|0|0.2077|12.3221|station.metpy_linear|
|pressure|C|22|0.1897|0.1126|40|12|0|1.0000|14.2450|station.metpy_linear|
|pressure|A|33|0.1897|0.1126|40|4|36|0.1452|10.5021|station.metpy_linear|
|pressure|B|33|0.1897|0.1126|40|13|0|0.1536|16.2839|station.metpy_linear|
|pressure|C|33|0.1897|0.1126|40|11|0|1.0000|14.3409|station.metpy_linear|
|pressure|A|44|0.1987|0.1958|40|5|32|0.2794|11.8504|station.metpy_cressman|
|pressure|B|44|0.1897|0.1126|34|11|0|0.1522|13.6437|station.metpy_linear|
|pressure|C|44|0.1897|0.1126|40|12|0|1.0000|16.3300|station.metpy_linear|
|pressure|A|55|0.1987|0.1958|40|3|33|0.2462|4.3966|station.metpy_cressman|
|pressure|B|55|0.1897|0.1126|40|12|0|0.1412|16.1289|station.metpy_linear|
|pressure|C|55|0.1897|0.1126|40|11|0|1.0000|14.3940|station.metpy_linear|

完整RMSE/MAE/Bias、变异提议/拒绝、全部Objective回调、缓存、首次合格/阈值评价序号和执行秒数见per_seed_comparison.csv与各run/search_result.json。最终输出非有限值违例为0；因单位不明，绝对物理范围未检查，其违例记NA而不是伪报0。误差单位为原始数值单位，不能将表中rh误差擅自写成百分比、pressure写成hPa。

### 均值、样本标准差及配对差异

|variable|mode|validation_RMSE_mean|validation_RMSE_std|test_RMSE_mean|test_RMSE_std|legal_proposal_ratio_mean|search_seconds_mean|
|---|---|---|---|---|---|---|---|
|pressure|A|0.2035|0.0162|0.1755|0.0361|0.2015|8.3043|
|pressure|B|0.1897|0.0000|0.1126|0.0000|0.1671|13.3404|
|pressure|C|0.1897|0.0000|0.1126|0.0000|1.0000|14.3328|
|rh|A|3.6638|0.7636|5.3967|0.2114|0.1984|8.4071|
|rh|B|3.1062|0.0000|5.5511|0.0000|0.1713|11.5160|
|rh|C|3.1062|0.0000|5.5511|0.0000|1.0000|14.4161|

相同seed的C−A、C−B逐项差保存paired_differences.csv。五个种子重复同一批站点日期，衡量的是搜索随机性，不是五个独立环境数据集；本轮仅作描述性配对分析，不据此宣称统计显著或通用优势。

### 穷举、单算法与消融

两变量18个可行图均实际执行，含6个直接单算法；exhaustive_baselines.csv保留所有成功/失败及验证分数。该小空间可以穷举，GOLEM并非必要最优工具；若最优是单个插值器，不能包装成复杂算法组合创新。

|variable|best_validation_graph|RMSE|
|---|---|---|
|rh|['station.metpy_natural_neighbor']|3.1062|
|pressure|['station.metpy_linear']|0.1897|

对主任务rh另实跑5种子×[关闭缺口、关闭软成本、仅硬类型契约]共15组，数据/预算保持一致；pressure未追加消融，优先完成两变量主实验。

|ablation|seed|validation_RMSE|graph|
|---|---|---|---|
|no_gap|11|3.1062|['station.metpy_natural_neighbor']|
|no_cost|11|3.1062|['station.metpy_natural_neighbor']|
|hard_only|11|3.1062|['station.metpy_natural_neighbor']|
|no_gap|22|3.1062|['station.metpy_natural_neighbor']|
|no_cost|22|3.1062|['station.metpy_natural_neighbor']|
|hard_only|22|3.1062|['station.metpy_natural_neighbor']|
|no_gap|33|3.1062|['station.metpy_natural_neighbor']|
|no_cost|33|3.1062|['station.metpy_natural_neighbor']|
|hard_only|33|3.1062|['station.metpy_natural_neighbor']|
|no_gap|44|3.1062|['station.metpy_natural_neighbor']|
|no_cost|44|3.1062|['station.metpy_natural_neighbor']|
|hard_only|44|3.1062|['station.metpy_natural_neighbor']|
|no_gap|55|3.1062|['station.metpy_natural_neighbor']|
|no_cost|55|3.1062|['station.metpy_natural_neighbor']|
|hard_only|55|3.1062|['station.metpy_natural_neighbor']|

缺口项在共同初始图已满足站点估计时通常为0，关闭它的相同结果不能被解释为该机制已获支持。类型契约减少无效提议与预测精度改善是两回事；本数据没有完成%/hPa范围的物理约束消融。

### 问题、回读与测试

全部517份实际生成NetCDF已再回读，维度x/y/time_label、有限值、目标变量unknown单位及非大地坐标声明均检查。另有103份输出的kriging_variance辅助变量没有单位属性，保留WARN：它是标准化输入空间的模型方差，不能当成原始物理方差或独立精度。未改写原始输出掩盖该问题。源数据完整，失败图与拒绝计数保留，没有删除不利种子。

实际修复：六年拼接中空2018区域表导致time列object，已显式按已有日期转datetime；2018全缺压力的Infinity审计界限改为JSON null；最终审计原先错误地要求辅助方差与目标变量同单位，已分开检查并记录上述缺失属性警告；这些修复不填补数据、不改变算法数学。Windows主环境复用已有隔离GOLEM包路径，未重新安装模型或修改第三方源码。

结果状态：数值GOLEM端到端已运行；绝对物理单位、地理坐标和原始发布/QA证据仍PARTIAL。不能据此宣称‘物理语义引导在真实气象任务上普遍优于普通/后验GOLEM’。

关键代码：tools/prepare_meteorology_golem.py、tools/run_meteorology_golem_numeric.py、tools/summarize_meteorology_golem.py；所有原始/派生审计、配置、输出、图JSON、CSV、失败记录及环境信息均在output/golem_real_meteorology_round1/。


### 复现命令

使用新的实验目录，勿覆盖本轮输出。主实验已有逐run JSON时支持中断续跑；配置、预实验和最终测试已存在时会阻止静默重写。以下命令均在项目根目录执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$meteoRun='output/golem_real_meteorology_reproduce_new'
.venv\Scripts\python.exe tools/prepare_meteorology_golem.py --data 'D:\codex\数据\气压、湿度数据' --output "$meteoRun/data"
.venv\Scripts\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase prepare
.venv\Scripts\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase pilot
.venv\Scripts\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase main
.venv\Scripts\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase baselines
.venv\Scripts\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase ablation --variables rh
.venv\Scripts\python.exe tools/run_meteorology_golem_numeric.py --output $meteoRun --phase test
.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp=output/pytest_meteorology_reproduce_new
```

随机与NumPy种子固定，输出中的每个图可按已冻结计划重放；这不意味着不同硬件/库版本上的墙钟耗时或浮点末位完全一致。


本轮全量回归：1204通过、4跳过、0失败/错误；256.57秒。XML在同一输出目录。


### 对六个研究问题的直接回答


rh：C的合法变异比例100.00%，A为19.84%，B为17.13%，因此确实减少无效提议。验证RMSE均值A=3.663821，B=3.106159，C=3.106159。最终测试RMSE均值A=5.396704，B=5.551102，C=5.551102。搜索秒数均值A=8.41，B=11.52，C=14.42。


pressure：C的合法变异比例100.00%，A为20.15%，B为16.71%，因此确实减少无效提议。验证RMSE均值A=0.203475，B=0.189722，C=0.189722。最终测试RMSE均值A=0.175461，B=0.112576，C=0.112576。搜索秒数均值A=8.30，B=13.34，C=14.33。


首次合格图/首次预设阈值各组均由共同初始化达到，不能证明C更快；实际C平均墙钟耗时更长，更多有效执行不等于更低时间成本。成功找到合格图的主实验均为5/5，也不构成C成功率提高的证据。

两个变量均未显示C优于B的验证精度；相同最终图的测试预测也完全相同。相对湿度最终测试C均值略差于A，海平面气压最终测试C均值优于A，因此不能说C在独立测试上一致占优。C相对A更稳定降低验证误差，不能证明新的插值算法精度或普遍泛化。

本轮贡献主要可归于类型契约约束的提议机制。缺口引导在已满足任务的初始图上不活跃；软成本和硬类型消融只能按已执行表解读，不能声称全部引导分量均有效。%/hPa、真实坐标及严格气象物理约束的优势仍未验证。

后续应取得单位/坐标/发布与QA证据，并在更大但确有处理必要性的任务空间和更多独立站点日期复验；不得加无意义节点、故意劣化基线或筛选数据制造C优势。

## 18. 第二轮真实多源 PM2.5 GOLEM 实验准备（2026-10-18）

### 18.1 结论：任务已明确，但还不能启动公平的多分支数值搜索

推荐任务：**地面训练站＋GEOS-CF RH35 背景特征＋ERA5 气象特征 → 时间窗口匹配 → 空间协变量匹配 → 观测约束空间估计 → PM2.5 分析网格 → 质量门 → 留出站评价**。

首条可复用算法路径是现有 `station.pykrige_universal` 的指定外部漂移接口。GEOS-CF 与气象变量作为不同协变量参与估计，目标仍是地面监测定义的 PM2.5，不能直接把 RH35 浓度改名为观测浓度。该方法已有实现，但本轮没有执行该真实数据路径，更没有运行 A/B/C 搜索。

目前不能声称已找到“执行能力完备且足够多样”的正式任务：站点坐标基准未闭合；通用 GOLEM 适配层没有多分支数值执行；目前能明确指出的多源主干主要是一种 Universal Kriging 拓扑。参数组合数量不能代替不同处理图数量。**准备审计完成，端到端可行性测试和正式搜索均未启动。**

### 18.2 实际数据审计与历史问题

本轮重新读取原始 NetCDF 和站点 CSV，未仅复制早期报告的状态。请求目录为 `D:/codex/数据/京津冀_PM25多源融合_201908/`。

|数据|本轮读取结果|物理意义、用途与限制|
|---|---|---|
|地面 PM2.5|60站，44,620条存储记录，43,884个有限数值，736缺失；站点/时间重复0，负值0；固定40/10/10站|真实监测目标；单位 μg/m³、北京时间、整点前一小时均值有此前来源链支持。坐标基准仍未知，仪器级 QA 未知；AQI等级不能当仪器QA|
|GEOS-CF|744小时×1层×25纬度×24经度；0.25°；113.5–119.25°E、35.5–41.5°N；2019-08-01 00:30至08-31 23:30 UTC；无非有限值，2.2008–872.0667 μg/m³|`pm25_rh35_gcc`，RH35定义，小时平均中心HH:30；模拟背景不是独立真值。不能与未确认调湿条件的监测PM2.5直接等同|
|ERA5|744小时×25×24；实际整月文件10,815,356字节；t2m/d2m/u10/v10/sp/tp均可读，无非有限值|t2m/d2m为K，风为m/s，sp为Pa，tp为m。前五项为整点瞬时量；文件声明tp为截至valid_time的一小时累计。属于辅助信息，不是PM2.5浓度|
|Sentinel-2 AOT|已读取680个站点—场景配对，280个通过现存QA，覆盖60站、11日|550nm无量纲AOT，不是地面PM2.5。站点抽样受坐标基准不确定性影响；不可复制为全天小时数据。现存SCL={2,4,5,6,7}是历史筛选规则，未证明全部为严格晴空|
|MAIAC|有124粒度目录清单，未把清单当HDF观测|本轮未下载；不计入实际可执行数据源|
|MERRA2-GMI|请求目录内无该整月源文件，其他既有实验目录有引用|保留为范围外可选资产，未静默加入本轮任务或宣称已审计；其重构PM2.5及瞬时插值口径也需单独签订契约|

重要目录问题：请求目录的 `station_observations/` 主要是元数据，不包含实际60站完整观测CSV。本轮只读复用了已有 `D:/codex/数据/京津冀_PM25融合测试_201908/01_地面监测站/ground_pm25_201908_60stations.csv`，没有重新下载、复制或修改原件。其 SHA256 为 `f20503e6722e34d3b2b8a6f3995e0bce43c166151d092fc524fbe39d508e6ec1`。

GEOS-CF SHA256 为 `a7465e77c2df499c71ed3173cba2ebd953753ecb2ba6e6d390c1386de287b43d`；ERA5、卫星站点表及执行代码的指纹、实际变量属性和数值范围见本轮JSON。早期 `data_inventory.csv` 仍称ERA5未下载，该状态已过时；实际整月ERA5来自 Earthmover/AWS 匿名区域子集，不能把Google ARCO单时刻样本误记为整月来源。

来源证据沿用现有源文件及保存的官方材料，不在本轮重复联网搜索或新增来源声明：GEOS-CF源地址保存在NetCDF的`source`中；ERA5保存`source_access`、ECMWF机构及文档链接；卫星保存Earth Search场景ID及真实过境时间。监测来源报告包含CNEMC/QuotSoft归档、1360个同标签逐值匹配，以及UTC+8的Zenodo与论文证据。对应链接为 <https://quotsoft.net/air/>、<https://doi.org/10.5281/zenodo.10911197>、<https://doi.org/10.5194/acp-22-15243-2022>。这些是既有证据的引用，并非本轮重新在线查证的结果。

仍不能由此推导：这60站坐标一定是EPSG:4326、每条观测均通过仪器QA、所有来源统计独立，或GEOS-CF未使用任何与研究相关的观测。现存资料不足以证明“来源完全独立”，正式方案应保留依赖关系不确定性；留站评价是目标隔离，不等于证明来源独立。

另发现GEOS-CF的`lev=72`属性同时带`units=millibar`；下载脚本只选了该产品第一层。不能把这个元数据组合解释成真实72 hPa气压层，也不能仅凭这一字段宣称近地层已复核。其`standard_name`写ambient而`long_name/humidity_basis`明确RH35；应保留RH35专用语义，并在正式使用前补齐产品层定义依据。

### 18.3 本轮实际完成的时间连接检查

地面标签转换为UTC窗口中心：`DATATIME - 8小时 - 30分钟`，并保留原始标签与前一小时边界；GEOS-CF按相同窗口精确连接，不按数组索引匹配。

ERA5 t2m/d2m/u10/v10/sp如采用两相邻整点插值到HH:30，必须标记为瞬时协变量估计，不能称为该小时真实平均。tp应按累计区间匹配，不能沿用瞬时线性插值；本轮仅检查前五项的端点覆盖，没有计算降水特征。

实际有 **735个共同窗口**，中心UTC为 **2019-08-01 00:30至08-31 14:30**。月初9个北京时间标签对应的540条记录超出GEOS-CF现有范围，直接排除，不补齐。735窗口内，两ERA5相邻整点均存在。

|站点角色|窗口内存储记录|有限PM2.5记录|
|---|---:|---:|
|训练40站|29,386|28,878|
|验证10站|7,349|7,239|
|历史测试10站|7,345|7,239|
|合计|44,080|43,356|

上述是时间与有限值连接计数，**不是经空间凸包、协变量秩、站点QA及定义审核后可用于融合的最终样本数**。本轮没有计算验证/测试RMSE，没有训练、预测或据此挑选数据。

### 18.4 必要处理图与可替代路径

```text
地面站 → 数据检查 → UTC小时边界 → 已确认坐标转换 ─────────┐
GEOS RH35 → 同窗口匹配 → 保留RH35语义的空间协变量查询 ───┤
ERA5 → 按变量时间语义处理 → 气象空间协变量查询 ──────────┤
                                                       ↓
                         Universal Kriging（训练站拟合外部漂移）
                                                       ↓
                  PM2.5网格 → 非负/单位/时间/覆盖检查 → 留出站误差
```

空间输出可研究0.1°分析网格，并以0.25°为基线；这是站点约束估计的表达网格，不是宣称获得真实0.1°信息。正式边界需使用已确认坐标下的训练站有效覆盖域，禁止让所有方法在不同站点/区域评价。若网格包含凸包外部分，保留明确有效域mask；不能把没有支持的区域偷偷最近邻补满后算成功。当前质量门是否正确允许“有效域外显式mask、有效域内不得NaN”还需接线验证。

既有Universal Kriging可把背景、温度等作为`specified_drift`，并检查协变量有限性和训练设计矩阵满秩。这条路线并不要求不同协变量具有相同单位；每项单位、时间、坐标须分别可靠。其漂移系数属于统计估计，不是已证明的RH35物理调湿转换。温度、风、AOT不能直接与浓度加权平均。

另一条希望比较的真实任务是“历史训练期校正背景 → 同期训练站残差估计 → 背景＋残差”。但当前冻结组合尚未打通训练状态、残差表与分支端口传递，不计为可执行替代拓扑。也不能先把GEOS改名成地面PM2.5后直接相减。时间标签连接、单位核对、质量检查属于必要数据契约或固定边界步骤，不应人为包装成多个可交换算法节点。

### 18.5 34个冻结候选的实际适配范围

|候选|可复用内容|本任务限制|
|---|---|---|
|`station.pykrige_universal`|已有真实多协变量接口、指定外部漂移、坐标投影和秩检查|最接近三源估计主干；仍需确认站点datum、掩膜策略和真实数据运行|
|`station.pykrige_ordinary`；四种`station.metpy_*`|可作为站点单源基线|不是自动获得GEOS/ERA5融合能力；不能因传入auxiliaries就认定算法使用了辅助变量|
|`spatial.nearest/bilinear/conservative`|有空间转换算法|仅在源网格与目标查询确需变化时使用；不能对已同网格数据反复重采样制造图空间；浓度非无条件总量守恒|
|`temporal.semantic_resample`|已有区间均值/累计量处理|必须提供实际time_bounds与语义；本任务小时均值精确窗口连接不需要伪造时间插值|
|`bias.xsdba_eqm`|已有官方train/adjust、训练应用时间分离|当前稀疏训练站不能伪装为完整网格真值；还需匹配的参考维度、训练/应用状态传递|
|`bias.xsdba_extreme`|极端值组件|现有适配要求长序列，不能拿这个月少数小时强行尾部拟合|
|`fusion.quality_aware`|已有仿射去偏＋观测定权|要求两个二维同目标来源；用其validation字段定权后不能再把同批观测当GOLEM独立验证；当前实现按调用重新拟合，不是已验证的跨时段fit/apply|
|`fusion.residual`|已有背景＋观测残差|只支持单时间二维，且linear凸包外自动最近邻回填；严格任务须额外限制/记录覆盖，不直接准入|
|`fusion.uncertainty_weighted`|已有逆方差加权|需要同定义目标及可信不确定性；训练残差方差不是独立已校准来源误差|
|`scale.semantic_stack`、`features.galileo_nano`|特征准备/特征提取|不等于PM2.5数值预测器；当前PM2.5和ERA5文件不能直接冒充Galileo官方输入组|
|其余土壤、地形、垂直、学习型等候选|保留冻结成果|本任务不满足其变量、训练或结构条件，不能为了34个全使用而强行加入|

`fusion.idw`、FNO/UNO虽然项目中存在，但**不在冻结34池**；本轮没有将其加回。通用候选池及第三方GOLEM没有修改。

执行层核查：

- GOLEM官方图对象能表示多父节点，不代表EcoData已执行这种图。
- `src/ecodata/algorithms/pipeline.py`按`AlgorithmPlan.steps`顺序执行，只替换`sources[0]`；没有按父边装配命名端口、分支结果或训练状态。
- `tools/run_golem_search.py`仍是能力覆盖/冲突/成本Objective，不调用环境数值算法。
- `tools/run_meteorology_golem_numeric.py::chain`明确限制最多两节点、单根、单链；不能接三路数据。上一轮真实数值回调不能直接复用为通用多源DAG执行器。
- `tools/run_final_fusion_cases.py`已有手工准备、顺序调用的真实样例，但手工串联不等于GOLEM图执行。该脚本对来源统一写PM2.5语义及近似WGS84，只能按原报告的探索性限制理解，不能复制成严格正式契约。

### 18.6 搜索空间：不能用理论乘积凑数

预备的Universal Kriging参数配方上限为24：GEOS查询方法2种×ERA5查询方法2种×气象协变量集合2种×变差函数3种。协变量集合预先限定`t2m`或`t2m/d2m/u10/v10/sp`，两者都保留GEOS特征；变差函数限定linear/spherical/exponential。只有目标网格确实不同、对应查询方法可实现且结果不等价时，才保留该因子。

**24只是待审参数配方，不是24种已运行算法组合**。当前主要是一种多源估计拓扑；同网格重复操作、等价重采样路径、秩亏或违反契约的配方还要去重/剔除。严格多分支端到端搜索已验证图数目前为0，不把34^n作为搜索空间。

因此本轮明确：该数据集可以支撑有意义的“多源观测约束估计”问题，但尚未证明冻结池和执行器足以支撑“足够多种必要处理组合”的Mutation机制比较。正式前必须至少打通第二条语义不同、科学合理的估计/校正路径，或证明现有图语法确有多种不等价能力补全路线；若最后只有UK参数搜索，应如实改称参数选型实验，不能据此证明一般算法组合创新。

### 18.7 公平A/B/C预备设计（尚未执行）

三组共享完整冻结34清单和同一任务契约；任务不适用节点在三组均须最终拒绝。A使用官方默认Mutation，B在同样Mutation后做语义过滤，C只改变EcoData候选提议概率。最终合法性、输入、输出、Objective、缓存及预算一律相同，不修改GOLEM核心。

共同初始条件采用相同的、未完成处理的输入图，而不是已含最优完整算法的初始化。原始数据叶节点与处理节点分开；允许合法的未完成图继续演化，但不能作为成功输出。能力缺口按**具体输出端口及其祖先路径**计算，不是对全图能力取并集：例如ERA5完成了时间处理，不代表站点分支的时间已对齐。下一步端口依赖、数据可达性和已完成能力日志必须先实现，才能研究能力缺口引导。

完整候选必须真实执行、通过两道质量门、文件回读，再用相同留出站记录计算RMSE。主要目标为验证RMSE除以训练期观测标准差；硬违例统一拒绝或固定大惩罚。成本、MAE、Bias、节点数作为共享的附加指标/平局规则，不按各组单独缩放；墙钟时间单独报告，避免把硬件噪声当精度。

预实验建议相同种子11、每组最多20次Objective调用；通过后才考虑种子11/22/33/44/55、每组40次。此处仅保存方案，没有启动。缓存每组每种子冷启动，缓存键包含图、参数、数据SHA、划分和代码版本；重复调用也计预算，另报实际数值执行数。过滤前提议数、拒绝数、结构检查、缓存命中、失败及墙钟均记录。增加共同提议次数和墙钟上限，避免B/C在相同Objective次数外无限尝试；正式上限须在预实验耗时后统一冻结，不能看C成绩再设定。

指标：RMSE/MAE/Bias；有预测且质量通过的图比例；首次可交付图的提议/Objective/实际执行次数；首次低于训练内交叉验证固定阈值的次数；达到同一误差阈值的墙钟；全流程时间、峰值内存、源数据访问和失败代价。未达到阈值记右删失/未达到，不填0。固定候选单算法和可行小空间穷举是必要参照；若C只有更高合法率、没有更好精度或更低实际成本，不能宣称综合更优。

### 18.8 数据隔离方案与不能掩盖的历史曝光

预备沿用原有40训练、10验证、10测试站，不按成绩重分。日期按北京时间固定为8月2–14日拟合、15–21日搜索验证、22–31日最终评价；避开月初边界不完整日。具体站号及训练40站内的28拟合/12定权子划分已按种子20261018写入JSON，尚未据此拟合或评价。

- 历史偏差参数/归一化只从训练日期的训练站学习。若需定权，使用训练内部留出或交叉拟合，不使用10个搜索验证站来拟合权重。
- 评价日期允许40个输入站的同期观测参与空间估计，这是观测条件重构，不是未来无人站点预报。验证站目标仅给Objective，测试站目标不进入搜索或训练。
- Universal Kriging当时刻漂移拟合可用40站同期值；历史模型参数和变差函数候选设置按既定训练边界。应清楚区分这种同期条件拟合与未来时间泛化。
- **原10测试站已经在项目旧实验中评价过**。本轮可做到运行级封存，不能声称项目历史上从未查看或完全前瞻独立验证。如需强确认性结论，须另外取得从未参与研究决策的日期/站点，而不是在本轮偷换已有划分。
- 最终评价样本清单只能按输入可用性、坐标、物理契约和预定覆盖规则建立，不能按某方法的误差挑选。因方法失败缺预测不能静默缩小到有利样本；应报告共同支持范围和失败率。

### 18.9 阻塞、边界与下一步最小工作

|类别|具体阻塞|所需最小处理|
|---|---|---|
|数据|60站坐标基准未证实；仪器QA未知|取得能绑定该站点表的坐标声明；QA未知保留，不冒充已质控。可以另经确认做近似坐标探索，但不能记正式通过|
|定义|RH35与地面观测条件不同、GEOS层元数据含歧义|优先作为独立协变量，不直接求差/加权；补层定义证据。若走直接融合须独立审核校正目标及适用性|
|执行|没有端口感知的多分支数值图执行器|在EcoData适配层接真实父边、observations/auxiliaries端口、状态和结果缓存；这不是缺一个新算法，也不要求重写GOLEM|
|隔离|定权观测可能与搜索验证重复使用；旧测试已曝光|训练内部定权并在外层验证选图；标记回顾性评估|
|物理质量|残差线性法凸包外自动最近邻；网格mask与质量门还需联调|固定共同有效域，禁止隐式外推，记录未支持区域；不要无依据裁负值|
|搜索设计|目前多源主干拓扑单一|先验证真实不等价的第二条必要路径；否则报告区分度不足，不强行增加空操作|

本轮不做新的算法、没有下载数据、不改网络、没有修改第三方GOLEM或冻结池。由于上述条件未满足，按任务第8项未运行端到端融合测试；执行的是只读数据回读、时间连接和代码契约审计，实际Objective调用0，RMSE/MAE留空，不生成模拟结果。

### 18.10 复现、检查与交付位置

- 新增只读准备脚本：`tools/prepare_golem_multisource_round2.py`。
- 审计与图需求/划分：`output/golem_multisource_round2_preparation/preparation_audit.json`，包括真实文件指纹、实际NetCDF属性、时间样本计数、固定站号、拟定分支与端口，以及BLOCKED原因。
- 时间契约测试：`tests/test_golem_multisource_preparation.py`。使用的测试日期只是映射单元测试，不是合成融合实验。
- 复现命令：`.venv/Scripts/python.exe tools/prepare_golem_multisource_round2.py --data "D:/codex/数据/京津冀_PM25多源融合_201908" --station "D:/codex/数据/京津冀_PM25融合测试_201908/01_地面监测站/ground_pm25_201908_60stations.csv" --output output/golem_multisource_round2_preparation/recheck_audit.json`。输出已存在时拒绝覆盖。

本轮定向测试 **8通过、0失败、0跳过**，结果见同目录`preparation_tests.xml`。存在一条既有`.pytest_cache`权限警告，不影响测试执行。仅新增准备脚本、时间映射测试并追加本节，没有修改业务源码。上一轮1204通过/4跳过是历史全量结果，不能说成此次重新执行。

**MULTISOURCE_NUMERIC_GOLEM_READY = NO。** 现在最需要补的是坐标证据、多分支数值执行接线和可辨识的真实组合路径，不是再安装新算法。

## 19. 第二轮真实多源PM2.5：多分支DAG执行与GOLEM烟测（2026-10-18）

### 19.1 本轮结论

**EcoData层多分支数值执行已实现，两条不等价的真实处理图均已执行。** 图使用官方GOLEM的`OptGraph/OptNode`表示，执行器沿真实`nodes_from`依赖解析和执行，不是按图名挑选固定数值，也不是把手工计算结果交给Objective。另有官方Mutation实际生成的新图完成数值执行、质量检查和NetCDF回读。

但正式A/B/C研究仍未就绪：本轮最多20次Objective的烟测中，A实际20次，B/C各2次后发生图生成停滞；三组都只数值评价了两个共同初始方案。未得到相同实际评价次数下的新组合搜索对照，**不得宣称C优于A/B**，不启动大规模正式实验。

本轮没有扩充34个冻结算法候选，没有修改第三方GOLEM优化器，没有下载新数据或训练神经网络。沿用第18章数据审计与时间定义，只有本轮明确授权的“原始坐标近似WGS84”探索性工程假设，不宣称正式地理精度通过。

### 19.2 新增的真实图执行能力

新增`src/ecodata/algorithms/dag.py`，与既有线性Pipeline并存，不替换旧流程：

1. `Node/NodeAdapter/Port/Artifact`分别描述节点、执行适配器、命名端口和真实数据/状态。
2. 按父节点依赖拓扑执行，处理共享父节点和多分支汇合；拒绝循环、缺失父节点、未消费分支与错误输出端口。
3. `from_golem/run_golem`直接读取GOLEM图边。端口可显式绑定；仅在类型匹配唯一时自动推断，不能按父节点列表顺序猜含义。
4. 逐边检查数据类型及必要元数据；本轮接口继续检查变量定义、单位、时间中心/窗口、CRS操作假设、训练身份与状态兼容性。
5. 中间结果真实保留于`intermediates`，选定运行另保存分组`intermediates.nc`；每条边记录数据指纹，每节点记录操作、参数、命中缓存与耗时。
6. 训练校正状态以独立`state`端口传递，不放在隐含全局变量；记录训练时段、站号、来源指纹、系数、输入输出语义和单位。状态另存JSON供核查。
7. 内容寻址缓存包含真实输入内容、参数、状态、适配版本和运行版本。复制不会改变科学数据指纹；改变输入、单位或参数会改变缓存键。当前是**进程内缓存**，不声称跨进程长期缓存已完成。
8. 失败保留异常、已完成的逐边轨迹和失败节点，不输出假成功文件；旧Pipeline及Registry仍可继续使用。

本轮专用端口适配在`src/ecodata/algorithms/pm25_dag.py`。数据源、时间选择、目标网格Schema和质量门是边界操作，不算新增候选算法。目标Schema含NaN仅描述所求变量/坐标，不能当作真实输入场或模型输出，数值必须由实际算法生成。

### 19.3 路径一：GEOS-CF＋ERA5外部漂移Universal Kriging

```text
GEOS-CF月文件 → 精确选择HH:30小时均值 → RH35协变量 ───┐
ERA5月文件 → 两真实整点间的瞬时温度插值 → 气温协变量 ─┤
40个输入站 → 同一UTC窗口、数值检查、有效站筛选 ────────┤
                                                     ↓
        station.pykrige_universal（官方PyKrige specified drift）
                                                     ↓
        共同有效域mask → 结果质量门 → NetCDF → 留出站预测
```

真实调用现有`PyKrigePlugin`及官方`UniversalKriging`，GEOS和ERA5分别进入`specified_drift`。GEOS保持RH35标识、气温保持K单位，不把二者当成同一物理变量混合；最终目标是地面站定义的PM2.5。输入协变量必须覆盖站点和目标域、数值有限、训练漂移矩阵满秩。

本轮仅使用ERA5 t2m，不为了凑变量加入风、降水或卫星。ERA5结果标记为“插值得到的瞬时气象协变量”，不是假造一小时观测均值。GEOS层号72及属性歧义仍按第18章列为科学限制，本轮不新增“已证实地表层”的声明。

### 19.4 路径二：历史训练校正＋同期站点残差融合

```text
历史GEOS RH35 ──┐
历史40训练站 ───┴→ 现有quality_aware仿射校正组件 → state
当前GEOS RH35 ────────────────────────────────┘
                          ↓ apply(state)
             地面观测定义下的统计背景估计
                          ↓ spatial.bilinear
当前40输入站 ────────────┴→ fusion.residual（SciPy真实残差插值）
                          ↓
             共同有效域mask → 结果质量门 → NetCDF
```

抽取并共用已有`quality_aware`中的仿射拟合计算，新增`fit_affine_source`供原组件和DAG状态节点共同调用，没有另建一种校正算法。图中的`fusion.quality_aware`使用明确的`mode=fit/apply`执行其**校正子组件**，不是声称跑了原来需要两个来源、用验证站定权的完整加权融合器。

训练仅使用北京时间8月2–14日的原40训练站。有效配对12,234条，拟合系数为`slope=0.06359839006202861`、`intercept=18.33315380191699`。应用时校验定义、单位、训练结束早于目标时刻等条件；10个验证站和10个测试站不参与该拟合。不做数据源权重学习，因此第18章预留的28/12训练内部定权划分本轮没有启用。

校正含义是**从RH35模拟变量到地面观测目标的经验统计估计**，不是已建立的物理调湿换算；不能由此声称RH35与监测条件差异已在物理上消除。原始RH35变量、校正方法和状态完整保留。

现有残差组件增加可选`outside_hull="mask"`，默认仍是旧`nearest`行为，保持旧流程兼容；本轮显式使用mask，不能悄悄把凸包外补满。新的参数由原Registry插件传递，不增加算法ID。残差内核仍是原SciPy实现，没有新写插值核。

**路径二未使用ERA5。** 它是“历史校正＋当前背景/观测残差”的双源路径，不能把未用到的ERA5接成空节点冒充三源。这与路径一的“同一时刻多协变量漂移＋克里金”在算法、拟合时段、所用来源及图结构上均不同，不只是同一算法换参数。

### 19.5 时间、空间与质量处理

- 时间沿用`DATATIME - 8h30m`得到UTC小时中心，原值代表标签前一小时平均；不按索引配对，不补月初超范围小时。GEOS按相同小时中心精确选择。
- 工程测试固定两个验证时刻：2019-08-15 00:30、12:30 UTC。只把这两个时刻的40输入站记录及过去训练历史传给执行器；未来日期的输入站观测不进入本轮执行数据对象。测试站目标不进入算法、校正或搜索。
- 使用0.1°分析表达网格，不等于证实0.1°真实细节。所有输出标记`EXPLORATORY`和`ASSUMED_NOT_GEODETICALLY_VERIFIED`。
- 输出域取输入站在局部等距投影下的凸包与经纬度凸包的共同支持域；域外显式NaN/mask，域内任何非有限值或负浓度均阻止输出。这里的投影仅在已声明坐标假设下成立。
- 原残差组件内部仍使用经纬度平面SciPy三角插值，并非已验证的球面残差核；投影覆盖检查没有把它变成地理精度已证实的算法。
- 原始数据调用`RawInputQualityGate`；缺测筛除前记录缺失警告，返回有效输入计数与排除量。有限/非负/范围检查不是仪器QA，不把AQI类别当QA。
- 最终调用`ResultQualityGate`并增加支持域内有限性硬检查。域外mask导致WARN是预期，不自动补值、不裁掉负值来获得PASS。
- NetCDF重新打开后核对数值、坐标、单位、目标物理名称、时间窗口和CRS假设标识；中间数据也回读核对。没有真实高分辨率全场真值，不报告全场RMSE。

### 19.6 实际图运行结果

最终复核在`output/golem_multisource_round2_execution/verified_execution/`。每个时刻从10验证站中仅4站具有完整支持网格邻域：1012A、1051A、1070A、1075A。另6站按预先统一的几何规则排除，不是按误差筛选。两条路径评价相同4站，共8个时空样本。

|UTC时刻|真实处理路径|验证样本|RMSE μg/m³|MAE μg/m³|Bias μg/m³|回读/缓存重放|
|---|---|---:|---:|---:|---:|---|
|08-15 00:30|外部漂移UK|4|7.2018|6.5847|6.5847|通过；重放全部节点命中|
|08-15 00:30|校正＋残差|4|5.7687|5.6697|5.6697|通过；重放全部节点命中|
|08-15 12:30|外部漂移UK|4|5.7981|5.4486|1.2123|通过；重放全部节点命中|
|08-15 12:30|校正＋残差|4|6.6068|5.9660|-2.5025|通过；重放全部节点命中|

两条路径均输出真实数值且各有优劣；上述极小样本和未确认坐标只能证明工程可执行性，不能证明普遍精度。结果状态均为`WARN_EXPLORATORY`。

### 19.7 官方Mutation生成图的额外功能验收

为区分“手工初始图可运行”和“官方变异图也可运行”，使用未修改的`base_mutations.single_change_mutation`和相同冻结34清单，种子11，最多2,000个提议。

实际产生422个提议；第178个首次得到合法spherical UK图，第422个首次得到合法exponential UK图。选取规则事先限定为各参数的首个合法图，不看预测成绩再挑。两张图均由DAG执行器按其真实父边运行，通过质量门、生成预测和NetCDF回读；该时刻4站RMSE分别为5.5462、5.7347 μg/m³。

这些是同一UK的参数变异，**不能计成第二、第三种算法拓扑**；第二种拓扑的证据来自前述校正＋残差图。该额外探针不计入A/B/C预算，不用于证明搜索效率或挑选正式方案。所有422个提议和失败原因保留在`official_mutation_probe/mutation_probe.json`，两个实际结果及轨迹位于其子目录。

### 19.8 A/B/C预实验及为何必须停止正式搜索

三组均实际调用官方`EvoGraphOptimizer`，固定种子11、最多20次计分Objective、60秒墙钟上限、3,000个变异提议上限、相同冻结池、参数Schema、数据、质量门和冷启动缓存；五类官方Mutation函数没有修改。A默认提议，B后验契约过滤，C在至多4个同样的官方提议中做契约/成本加权，所有候选生成与过滤开销均保留。

**相对第18章的设计偏离：** 本轮是执行烟测，三组共同初始化为已经分别验证的两条完整图，不是原计划的未完成前缀。因此能力缺口项在初始图上为0，本次不能检验“能力缺口引导有效”；C也不被写成已验证的完整机制实验。这个限制在运行前的`pilot_config.json`中记录，没有在看到成绩后隐去。

最终复核文件为`verified_pilot/`，第一次烟测保留在`pilot/`，不是删掉失败记录重挑结果。

|组|实际计分Objective|有效记录/不同有效图|变异尝试/无效尝试|契约检查/拒绝|节点工厂调用数|墙钟秒|最佳验证RMSE|
|---|---:|---|---|---|---:|---:|---:|
|A|20|2 / 2|18 / 18|不做后验过滤|16|3.5151|6.2019|
|B|2|2 / 2|2 / 2|1,004 / 1,002|4,047|1.4046|6.2019|
|C|2|2 / 2|9 / 4|1,005 / 1,001|4,052|1.3177|6.2019|

三组最佳图均来自共同初始化，合计8个验证记录，MAE=5.8178、Bias=1.5836 μg/m³。**没有证据表明本次优化发现了更好的新组合**。B/C墙钟较短是提前停滞，不是相同预算下更高效率；20与2次实际调用不能公平比较成功率或精度收益。

具体阻塞不是数值执行器报错，而是官方`tree_growth`调用通用随机子树工厂时，当前任务缺少共同的“真实数据叶节点＋命名端口”生成语法。工厂只随机生成冻结算法名称，无法凭此生成所需固定数据边界和fit/apply状态链；在B/C契约过滤下出现`Could not generate random graph for 1001 iterations`。异常和已发生代价均保留，不能把这些随机图生成开销藏到“只有2次评价”之外。

表中契约检查包含官方随机图生长的尝试，不全是Mutation提议；节点工厂计数也包含这部分。Mutation函数抛错前已经计入尝试/无效尝试并保存错误。C可行提议比例等诊断量不能当作正式效能结论。

本轮不对测试站计算精度，不启动更多种子或正式搜索。`formal_search_decision.json`明确禁止继续正式实验。下一步需在**所有A/B/C共用的EcoData图生成适配层**补齐数据叶节点、端口与状态的生成规则，再恢复未完成前缀的共同初始化；不能给C独占完整图模板或为证明C修改基线。完成后仍需确认可探索的不等价图足够，而不是仅搜索UK参数。

### 19.9 失败与修复记录

- `attempt1/`中校正结果没有带`data_kind=continuous`，被原重网格语义检查正确拒绝；已给派生变量补齐连续量属性，没有降低质量门。
- 初版用Python对象序列化计算xarray指纹，深拷贝可能改变内部索引表示，导致重放缓存未命中。已改为变量/坐标/属性/数据内容指纹，最终四次图重放均全部命中。
- 官方GOLEM用`content['params']`参与节点标识，现有草案的`parameters`字段不会进入其原生描述。现已写入原生`params`，读取兼容旧字段，防止不同fit/apply或参数图被错误合并。
- 显式端口绑定保存在`params._port_bindings`，同样参与官方图标识；拒绝会被原生图标识忽略的顶层`bindings`。补充测试验证交换同类型的两个输入端口后，图标识与计算结果都会改变，不能误命中同一图缓存。
- 保留残差凸包外旧默认行为供旧调用兼容，新DAG显式启用mask；有效域内NaN不得降成普通WARN。
- B/C随机图生成失败未靠重复补Objective、隐藏拒绝或给C私有图生成器来“修成成功”。这是下一阶段适配缺口，不是本轮已解决事项。

### 19.10 测试、源码与复现

新增`tests/test_numerical_dag.py`覆盖拓扑/共享依赖、循环与错误边、未消费节点、元数据契约、xarray复制稳定指纹、真实GOLEM父边端口解析、校正隔离、显式坐标假设、残差域外mask及冻结池存在性。

全量回归：**1214通过、4跳过、0失败/错误，257条警告，264.63秒**，JUnit共1218项。其后补充一个显式端口绑定/图标识用例，最终定向测试：**13通过、0失败/跳过**。这个新增用例不冒计入前一次全量回归的1214项；前12项定向测试及两条真实图也已重新执行。警告主要来自既有依赖，不将警告数解释为质量已经全部无风险。

|位置|作用|
|---|---|
|`src/ecodata/algorithms/dag.py`|通用命名端口DAG执行、GOLEM图转换、缓存、轨迹|
|`src/ecodata/algorithms/pm25_dag.py`|本轮真实数据契约及已有算法端口适配|
|`src/ecodata/algorithms/quality_aware.py`|共享原仿射校正计算，不新增算法候选|
|`src/ecodata/fusion/residual.py`、`algorithms/plugins.py`|新增可选域外mask及传参，保留旧默认|
|`tools/run_pm25_multisource_dag.py`|两条真实GOLEM处理图、输入隔离、文件与中间数据回读|
|`tools/run_pm25_dag_pilot.py`|官方优化器A/B/C烟测及完整成本记录|
|`tools/probe_pm25_golem_mutation.py`|官方Mutation生成的新图实际执行验证|
|`output/golem_multisource_round2_execution/`|全部成功、失败、预测CSV、NetCDF、状态、图和测试日志|

结果目录重点查看：`verified_execution/execution_results.json`、各子目录`result.nc/intermediates.nc/validation_predictions.csv/execution_trace.json`、校正路径的`calibration_state.json`、`verified_pilot/pilot_summary.json`、`full_regression.xml/log`和最终`ports_directed_tests.xml`。没有另建分散Markdown报告。

复现两条图：`.venv/Scripts/python.exe tools/run_pm25_multisource_dag.py --data "D:/codex/数据/京津冀_PM25多源融合_201908" --output output/round2_replay_new`。烟测与变异探针分别使用同目录另外两个脚本，参数相同；已有结果目录拒绝覆盖。全量测试命令为`.venv/Scripts/python.exe -m pytest -p no:cacheprovider --basetemp=output/pytest_round2_new --junitxml=output/round2_regression_new.xml`。

环境：Windows、Python3.12.14，NumPy2.5.3、SciPy1.18.1、xarray2026.7.0、PyKrige1.7.3、h5netcdf1.8.1；主环境执行并引用既有`.venv-golem`官方包路径。本轮未新装环境。冻结池SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`；原`golem_search.py/run_golem_search.py`哈希与第18章一致。

**MULTIBRANCH_NUMERICAL_EXECUTOR = IMPLEMENTED**  
**TWO_NON_EQUIVALENT_REAL_PATHS = EXECUTED_EXPLORATORY**  
**OFFICIAL_MUTATED_GRAPH_EXECUTION = VERIFIED_EXPLORATORY**  
**FAIR_EQUAL_ACTUAL_BUDGET_ABC_COMPARISON = NOT_ACHIEVED**  
**CAPABILITY_GAP_GUIDANCE_EFFECTIVENESS = UNVERIFIED**  
**FORMAL_GOLEM_EXPERIMENT_READY = NO**

## 20. GOLEM类型化图生成定点修复与低预算验收（2026-10-18）

### 20.1 范围与原始故障

仅修复第19章的共用图生成适配，不改变冻结34候选、第三方GOLEM源码、数值算法、原始数据或历史结果。未安装依赖、未重新扫描原始数据、未运行2000次探针。

原始调用链：官方`tree_growth` → `graph_gen_params.random_graph_factory` → 默认`RandomGrowthGraphFactory` → `random_graph` → verifier。旧工厂只抽算法名，不能生成真实数据边界、端口及校正状态依赖，导致1001次失败。现在显式传入共用`TypedGolemFactory.random_graph`，单次按输出类型反推生产者；不调用默认随机重试循环。缺生产者、循环依赖、无法确定插入端口等请求立即拒绝并记原因。没有隐藏该异常或修改GOLEM上限。

### 20.2 实际修改及边界

- 新增`src/ecodata/algorithms/golem_generation.py`：从现有可执行NodeAdapter输入/输出契约建立有限类型语法；固定source叶节点属于数据边界，不增加算法候选。按端口递归连接真实父边，fit的calibration_state供apply消费。
- 官方加父节点策略只索引新父节点本身；共用检查器将该节点已经持有的真实依赖加入图索引，不凭空增加算法。端口绑定保存在原生`params._port_bindings`，变异后按既有边重新核对。允许缺必需输入的中间图参与进化；完整执行仍要求所有输入齐全。
- 新增`tools/pm25_typed_pilot.py`；`tools/run_pm25_dag_pilot.py`默认入口转到该实现，旧实现命名为`legacy_pilot`仅保留历史，不在当前CLI调用。
- 共用`start_depth=12,max_depth=12`：第一次低预算测试发现官方默认start_depth=3把初始图卡在最大深度，single_add直接返回原图。所有组同样修正，不给C特殊放宽。
- 不知道端口类型的插入及重复输出边界请求返回不适用，拒绝日志保留。规范图key包含参数及命名边语义，重复图不算新数值执行。
- 新增`tests/test_golem_typed_generation.py`六个结构用例：两路线单次补全、fit/apply依赖、禁止默认随机工厂、不可满足契约立即失败、兼容参数交换、正式single_add在明确深度下补全。

本任务的可执行适配语法只产生5种完整科学图（UK三种变异函数、残差两种插值参数），而不是34个候选都适用于PM2.5任务。两种不同拓扑来自UK与历史校正＋残差，参数差别不冒充新拓扑。生成验收中指定待测输出生产者用于逐项覆盖两路线，A/B/C搜索没有此指定。

### 20.3 自动生成后真实执行：两项均通过

共同未完成初始图：quality.result已有站点观测分支，缺field输入。固定种子11，分别调用官方`add_separate_parent_node`（single_add策略），由输入契约补齐整条生产者链，不加载手工完整图。

|路线|生成尝试|能力缺口|真实数值计算及NetCDF回读|验证RMSE μg/m³|
|---|---:|---|---|---:|
|GEOS-CF＋ERA5＋站点 → Universal Kriging|1|1→0|通过|5.899727|
|GEOS-CF历史fit → apply → 网格匹配 → 站点残差 → 输出|1|1→0|通过|6.201914|

证据：`output/golem_generation_fix/generation_v2/generation_result.json`，两路线各自`generation_trace.json`、slice目录的result.nc、intermediates.nc、validation_predictions.csv及执行轨迹。初次生成验收`generation/`也保留。

沿用第19章真实输入、训练历史及共同有效几何域：两个UTC中心时刻、每时刻4个验证站，共8条评价记录。测试站目标未传入。RH35只作有定义限制的背景/统计校正输入，不等同地面PM2.5；原始站点坐标近似WGS84仍属探索性假设。输出通过工程质量检查不代表正式地理精度或科研效能证明。

### 20.4 小预实验：尚未达到三组全部通过

共同设置：种子11、同一不完整初始图、同一34候选与适用契约、相同Objective、每组最多5张唯一完整图真实评价、最多50次外层提议、30秒搜索墙钟上限、冷缓存。A/B用相同五种官方Mutation；B增加后验过滤；C对同类官方提议做现有缺口/成本加权，每个提议均计入50次，未获得额外候选或模板。初始图key三组相同：`99e120689c29f35a4e9f8c3de5fe5d54c9fa6a68abcee4da640ae5ed5a9abee0`。

|组|Objective回调|唯一合格图/真实数值执行|生成检查拒绝|后验拒绝|缺口补齐变异|提议数|墙钟秒|最佳验证RMSE|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|A|40|2 / 2|30|不适用|1|50|17.5191|5.635913|
|B|27|0 / 0|10|10|0|50|7.6458|无|
|C|9|3 / 3|25|0|3|50|12.1549|5.794748|

回调数包括不完整图惩罚及重复记录，不能作为实际数值执行数。C另有3次完整图缓存命中，不计入3次新图执行；节点级缓存也单独记录。生成拒绝与后验拒绝可能指向同一提议，不相加充当独立失败图数。

三组都正常以PROPOSAL_LIMIT停止，无optimizer_error，无1001次随机图异常，均未穷尽5张图，不能标为正常空间穷尽。A/C确实出现从不完整图到新可执行图的Mutation轨迹；B在50次内仍未补齐：single_add共6次、single_change8次、single_drop9次、single_edge10次、tree_growth17次；其27次Objective全部缺口1～4，无完整候选。**不能以共享工厂能单次生成两路线来替代B搜索验收，也不能宣布A/B/C公平数值对照已通过。**

当前唯一未通过的搜索验收项是B在固定预算内没有补齐field/上游依赖。具体观察位置：`tools/pm25_typed_pilot.py`的`small_pilot/legal/propose`及官方Mutation后验接受流程，轨迹在`small_v2/B/pilot_result.json`。这是有界搜索中未找到完整解，不是默认random_graph再次重试1001次。未擅自更换种子、扩大次数或给B/C强塞完整图。下一步仅应据该轨迹检查后验接受与部分图演化行为，不泛化为需重建系统。

进一步逐条核对B的6次single_add：第24次缺口2→1、第28次1→1、第29次4→3、第46次3→2、第47次2→1、第50次3→2，均通过中间契约检查。因此不能声称B的工厂完全不能补输入；实际没有在预算内把最后缺口降至0。另一个明确的适配边界是`golem_generation.py:random_graph`目前按任务最终validated_field建图，而官方tree_growth没有在该接口传入被替换子树要求的输出类型；不能声称所有子树替换都已做到上下文类型匹配。需在共用EcoData tree_growth调用适配中传递所选子树输出契约，或明确拒绝缺上下文的内部替换；本次尚无该项通过证据，不用扩大随机重试掩盖。

第一次小实验`small/`因start_depth限制为三组0次数值执行，日志保留；修正后仅执行上述`small_v2/`。不存在删除失败后只呈报成功记录。所有变异、连接、拒绝和回调明细在各组pilot_result.json；汇总与停止决定位于small_v2/pilot_summary.json、formal_search_decision.json。

依用户低成本策略，**不扩大20次评价，不运行5种子正式实验**。当前证据不支持C优于A/B的结论；有限5图且仅8条验证记录也不足以证明效能。

### 20.5 测试、复现与结论

定向结构测试：6通过。全量回归首次在收集阶段遇到5个examples导入错误：新测试在收集阶段暴露独立GOLEM环境的同名examples包，遮蔽项目命名空间。已将新测试的环境路径限制在所需导入期间，不修改业务模块或第三方包；该失败日志保留在full_regression.log。修复后的全量结果见下方最终回归记录。

复现（必须使用新输出目录，禁止覆盖）：

```text
.venv/Scripts/python.exe tools/run_pm25_dag_pilot.py --data "D:/codex/数据/京津冀_PM25多源融合_201908" --output output/typed_generation_new --stage generation
.venv/Scripts/python.exe tools/run_pm25_dag_pilot.py --data "D:/codex/数据/京津冀_PM25多源融合_201908" --output output/typed_small_new --stage small --acceptance output/typed_generation_new/generation_result.json
```

冻结池SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`。本轮没有创建其他Markdown交付报告。

**DEFAULT_RANDOM_1001_FAILURE = AVOIDED_BY_SHARED_TYPED_FACTORY**  
**TWO_AUTOGENERATED_REAL_ROUTES = PASS_EXPLORATORY**  
**ABC_FROM_INCOMPLETE_GRAPH_ACCEPTANCE = NOT_PASSED_B**  
**FORMAL_EXPERIMENT = PAUSED**

最终全量回归：**1221通过、4跳过、0失败、0错误，257条警告，262.63秒**（总1225项）。证据：`output/golem_generation_fix/full_regression_final.xml`及`.log`。本轮仅在测试导入隔离修复后完成这一次全量执行；之前收集失败未计为通过。警告包括既有求解器精度等风险，不当作全部算法质量无风险。原`golem_search.py`、`run_golem_search.py`哈希与第19章一致，未修改第三方优化器或原搜索核心。

## 21. B组部分图保留与tree_growth端口定点检查（2026-10-18）

本轮只读取第20章、既有B轨迹及指定适配代码；未重新审计数据、未扩库、未运行随机搜索/正式实验/全量回归。以下结构测试不产生新RMSE，也不替代真实数值验收。

### 21.1 B的六次single_add：未发现部分图被错误拒绝的Bug

逐张恢复`output/golem_generation_fix/small_v2/B/pilot_result.json`保存的真实节点、父边和参数，重新运行共用inspect，按规范图key匹配既有Objective记录：

|提议|缺口前→后|相同图已有Objective回调|结果|
|---|---|---|---|
|24|2→1|13|接受，INCOMPLETE，score=1001|
|28|1→1|10|同图已有评价，INCOMPLETE，score=1001|
|29|4→3|14|接受，INCOMPLETE，score=1003|
|46|3→2|22、24|同图已有评价，INCOMPLETE，score=1002|
|47|2→1|27|接受，INCOMPLETE，score=1001|
|50|3→2|22、24|同图已有评价，INCOMPLETE，score=1002|

重复图匹配到的回调可能早于本次提议，不能冒称每次都发生新的独立评价。既有日志没有保存每代所有Individual UID和选择前后种群，故不能追溯证明这六次的每一个实例都存活到下一代；选择也不保证所有合格个体必须存活。该证据限制明确保留。

另以确定性单元测试验证完整机制：固定真实契约的缺两输入quality根节点，按固定节点及输入顺序，仅两次官方加父Mutation（每次最多1次operator attempt），得到缺口**2→1→0**。使用官方Mutation类、B同样的partial后验检查、相同`1000+gap`结构评分、pop_size=3默认generational Inheritance及官方Selection；断言子个体UID、父UID、继承后的个体和下一步选中的父个体连续一致，最终DAG可由现有执行器解析且fit→apply依赖完整。没有用随机多次尝试寻找成功，也没有把结构分数冒充数值RMSE。

结论：**部分图接受、继承、选择及连续补全机制正常；B原50次内未完成应判为有限预算未命中，不据此强行修改B选择或为其加入完整图。** 本结论不意味着所有历史个体都必然被选择，也不证明B在任何种子/预算下必定成功。

证据：`output/pytest_golem_two_checks_final/test_replay_all_six_b_addition0/b_replay.json`、`test_partial_graph_survives_of0/continuous_completion.json`，以及原始B轨迹（未覆盖）。

### 21.2 已复现并修复：tree_growth子树类型与共享依赖

旧官方tree_growth只把requirements/depth传给工厂，没有目标类型。固定选中三个位置各做一次旧实现替换即可复现契约错误，无1001次循环：

- `prepare.geos`应输出geos_covariate，却插入validated_field。
- 校正fit应输出calibration_state供apply消费，却插入validated_field。
- `source.era5`应输出meteo_series，却插入validated_field。

共用修复位于`src/ecodata/algorithms/golem_generation.py`：`random_graph`新增显式output_kind；`typed_tree_growth`从消费者命名端口确定被替换子树的目标类型，再单次调用类型化工厂。保留原节点随机遍历、局部/全局深度和主节点选择规则；主节点替换仅允许相同输出类型的数据叶，不能用随机其他来源代替。上下文不唯一时明确报错，不猜类型。`tools/pm25_typed_pilot.py`的A/B/C共用Mutation列表统一引用此适配函数；不是C专享规则，其他四种官方Mutation及第三方优化器未修改。

测试同时复现了真正多分支DAG的共享边问题：官方tree式`update_subtree`会删除旧子树全部祖先，连另一分支仍在使用的source.geos也会删掉，使prepare.geos丢失series。适配层改为只重定向该目标的消费者边，并通过官方OptGraph按可达依赖重建索引，保留其他分支共享数据。不新增节点补洞、不复用旧结果冒充新输出；在副本上检查失败即拒绝，不破坏原图。

三个固定替换位置修复后全部通过完整端口检查及NumericalDAGExecutor.from_golem解析，其中fit替换后另一分支的GEOS输入仍完整。**本轮仅证明结构契约修复，没有重新运行真实数值搜索，不能宣称A/B/C新预实验已全部成功。** 原第20章数值记录保持原状；工厂源码哈希变化，旧生成验收不能绕过现有哈希门直接用于新搜索。

### 21.3 定向测试及失败记录

最终仅运行`tests/test_golem_typed_generation.py`与`tests/test_numerical_dag.py`：**19通过、0失败、0跳过、0错误，5.085秒**。其中类型生成模块原6例＋本轮5例（六图回放1例、连续补全1例、三个子树位置3例）；另8例为现有DAG定向兼容检查。没有运行全量回归。

每种子树位置最多4轮定向测试，每轮固定一次旧故障替换及一次修复替换，合计不超过8次/结构；连续补全每轮固定2步，没有随机搜索。第一次暴露共享祖先被删除和测试预选图不含所需节点；随后一次暴露GraphDelegate不公开内部重连方法，改用显式现有父边重定向。失败日志均保留：`output/golem_two_checks_test.log`、`_v2.log`；随后`_v3.log`及最终`_final.log`成功。最终JUnit：`output/golem_two_checks/targeted_final.xml`。没有删失败结果，也未安装依赖。

源码改动仅限`golem_generation.py`、`pm25_typed_pilot.py`和`test_golem_typed_generation.py`。冻结34池SHA256保持`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`。本章为唯一新增交付说明，未另建Markdown报告。

### 21.4 是否值得扩大本任务A/B/C实验

**建议停止在当前任务上扩大A/B/C实验。** 当前5张完整图只有两种科学拓扑：3种UK变异函数参数和2种残差插值参数；共享源、prepare及质量边界不是额外科学方法。如此小的空间更适合五方案直接比较/穷举作为功能基线，不能支撑“复杂算法组合搜索创新”的有力证据。当前真实验证仅8条有效记录，坐标与变量定义限制也仍存在；无法由一次种子下拒绝率或找到方案速度推断普遍优势。

本轮不新增参数/无意义节点凑空间、不执行穷举或正式A/B/C搜索。后续只有在独立真实任务本身需要更多**不等价且有必要的数据处理决策**、具备充分独立验证和统一预算时，再评估研究实验；不能为了证明C而扩充当前图数量。

**B_PARTIAL_ACCEPTANCE_AND_CONTINUOUS_COMPLETION = VERIFIED_STRUCTURALLY**  
**B_PREVIOUS_STOP = FINITE_BUDGET_MISS_NOT_PROVEN_SELECTION_BUG**  
**SHARED_SUBTREE_OUTPUT_CONTRACT = FIXED_AND_TARGETED_TESTED**  
**EXPAND_CURRENT_FIVE_GRAPH_ABC_EXPERIMENT = NOT_RECOMMENDED**

## 22. 三层公开数据小规模预审（2026-10-19收尾）

### 22.1 范围、目录与总判定

本轮只下载小范围预审所需公开文件、统计数据及只读检查已有算法接口；**未训练、未运行GOLEM或A/B/C对照、未新增算法、未修改核心算法、未运行全量回归**。数据采集记录的UTC日期为2026-10-18，跨日收尾按用户时区为2026-10-19；沿用原目录，不改名造成路径失效。

统一目录：`D:/codex/数据/公开数据三层预审_20261018/`，包含：

- `01_UCI_Beijing/`：官方ZIP、12站原始CSV、逐站逐变量和缺失段统计。
- `02_EPA_PM25/`：Monitor Listing、2018—2022年度摘要、机构/站点统计；Site Listing和小时ZIP的失败记录。
- `03_MAIAC_ERA5/`：匿名CMR目录及访问条件，**没有HDF/ERA5数据数组**。
- `04_统计与校验/`：固定预审规则、算法接口清单、统一来源表及校验/汇总JSON。

收尾统计约73.59MiB；25份已取得的原始包、原始CSV或目录元数据逐份重新核对SHA256及大小，并设置只读，全部匹配。统一来源清单`source_inventory.csv`共有27条，含两项下载失败，**不是27项全部取得**。原始文件只读不等于无法更改的存储，SHA256用于后续发现内容变化。

|实验|预审结论|已取得的证据|尚缺什么|
|---|---|---|---|
|1 UCI单源时间恢复|PASS，允许后续另行开展实验|12站全部CSV、真实缺失段、完整时间轴|仍需冻结人工遮蔽及时间划分；本轮未恢复/训练|
|2 EPA单源空间重建|NOT_PASSED|官方监测器元数据＋五年摘要；2022所选数据只有10个一小时站|未达20站目标；完整小时数据、逐日18小时完整性及共同日期未验证|
|3 EPA＋MAIAC＋ERA5|NOT_PASSED / BLOCKED|四季目录预查、访问条件、真实接口审计|NASA/CDS凭据、实际QA交集、EPA小时覆盖、两条完整且不等价的任务接口|

### 22.2 实验1：UCI北京12站时间缺失恢复

官方来源：[UCI 501](https://archive.ics.uci.edu/dataset/501/beijing)，DOI `10.24432/C5RK5G`，CC BY 4.0。成功取得[官方PRSA2017 ZIP](https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip)，7,959,991字节，SHA256=`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`。现代下载包装链接曾出现传输截断，失败记录保留；改用同一官方原包路径成功，没有引入第三方镜像。

实际读取12个站点：Aotizhongxin、Changping、Dingling、Dongsi、Guanyuan、Gucheng、Huairou、Nongzhanguan、Shunyi、Tiantan、Wanliu、Wanshouxigong。每站35,064行，共420,768行，2013-03-01 00:00至2017-02-28 23:00。每站时间顺序连续、0重复时间、0缺失时间行；**字段值仍存在实际NA，时间行完整不等于无缺测**。

实际18字段：`No/year/month/day/hour/PM2.5/PM10/SO2/NO2/CO/O3/TEMP/PRES/DEWP/RAIN/wd/WSPM/station`。其中没有经纬度或CRS，没有逐记录仪器QA。官方说明污染物来自北京市环境监测机构，气象匹配最近气象站；不能认为每个空气站都独立测得其气象变量。官方页面未明确时间基准，本实验只用源时间序列，不擅自转为UTC。

|变量|有效记录|缺失记录|缺失率|最长连续缺失小时|
|---|---:|---:|---:|---:|
|PM2.5|412,029|8,739|2.0769%|343|
|PM10|414,319|6,449|1.5327%|343|
|SO2|411,747|9,021|2.1439%|457|
|NO2|408,652|12,116|2.8795%|900|
|CO|400,067|20,701|4.9198%|1,517|
|O3|407,491|13,277|3.1554%|497|
|TEMP|420,370|398|0.0946%|7|
|PRES|420,375|393|0.0934%|7|
|DEWP|420,365|403|0.0958%|7|
|RAIN|420,378|390|0.0927%|6|
|WSPM|420,450|318|0.0756%|5|
|wd（分类）|418,946|1,822|0.4330%|13|

上述为实际文件统计，不是数据集介绍推测。PM2.5单位μg/m³、TEMP/DEWP为℃、PRES为hPa、RAIN为mm、WSPM为m/s来自官方字段说明。缺失不自动补齐；`No`及日期组成字段不是待重构环境变量。

**通过依据：** 每站PM2.5/TEMP/PRES均有充足已知数值，可在有效观测上预先固定点状/连续块遮蔽，评价恢复误差。原始NA没有真值，不能纳入“恢复准确率”。下一步应固定遮蔽、时间隔离及归一化拟合范围；这是恢复问题，需明确是否允许使用遮蔽区间两侧已知观测，不能混称未来预测。本轮不生成掩蔽实验成绩。

完整证据在`01_UCI_Beijing/statistics/`：216行`station_variable_statistics.csv`（12站×18字段）、全部实际缺段`missing_runs.csv`、`station_time_coverage.csv`及`preaudit_result.json`；ZIP CRC、CSV回读及记录数勾稽通过。

### 22.3 实验2：EPA站点空间重建

官方入口：[EPA AirData](https://aqs.epa.gov/aqsweb/airdata/download_files.html)，格式依据[FileFormats](https://aqs.epa.gov/aqsweb/airdata/FileFormats.html)。成功下载并读取`aqs_monitors.zip`及`annual_conc_by_monitor_2018.zip`至`2022.zip`，共六份有效ZIP。Site Listing `aqs_sites.zip`及唯一申请的小时包`hourly_88101_2022.zip`下载发生截断/超时，经过一次有界官方URL重试仍未得到完整ZIP，已停止。小时ZIP官方标示约75,836,309字节，但本地仅有不完整part，**不能当作下载成功**。未下载其他年份全国小时文件，也未用年度摘要代替小时值。

站点筛选以官方Monitor Listing中PQAO、Reporting Agency同时明确为`South Coast Air Quality Management District`及监测有效期为依据，不以四县全域代替管理区。该方法得到的是机构明确的子网，并非完整管区边界内所有其他机构站点；Listing是当前快照，不能由此证实五年间机构归属从未变化。

2022元数据候选20个唯一站、35个监测器POC；候选坐标范围经度[-118.53276,-116.21537]、纬度[33.63003,34.26464]。这是包含人工采样的候选范围，不是20个合格连续站。单位实际为`Micrograms/cubic meter (LC)`；坐标Datum字段实际出现NAD83及WGS84，保留区别，不默认为同一基准。站号按State/County/Site组成，POC为同站同参数不同监测器，不能增加独立站点数。

|年份|机构子网有数据站|一小时站|物理24小时采样站|连续记录计算日均站|不同监测器|
|---|---:|---:|---:|---:|---:|
|2018|19|6|19|6|34|
|2019|19|6|19|6|32|
|2020|19|8|19|8|34|
|2021|19|8|19|8|33|
|2022|20|10|19|10|35|

一小时与24小时列允许同站重叠，不相加；`24 HOUR`人工/滤膜采样与`24-HR BLK AVG`连续数据计算日均严格区分。仅5个站在全部五年的一小时摘要中存在，所审摘要坐标未发现跨年变化；这不证明小时连续完整。年度摘要还包含不同标准/事件等多行，不能把摘要行数当独立观测数。

预先固定的日均规则已经记录：至少18个不同有效小时；每站按有效覆盖、非预测成绩确定主POC，不直接平均POC；原始负值及标志保留，严格派生集合排除负浓度和非空qualifier；合格年度连续站暂取至少270个有效日。实际小时未取得，**没有执行这些派生，也无法测得各站有效日数、重复小时/POC重叠、采样实际间隔、逐日共同站数或四季共同观测日期**，均为null/未测而非0。

另查阅EPA[88101数据修订告示](https://aqs.epa.gov/aqsweb/airdata/Data_Advisory_PM25_88101_Pregen_Files.pdf)：T640/T640X原始与修订方法码需要区分，已在预审规则保留236/238排除、修订/更新方法记录要求；年度`Multiple Methods Used`不能代替逐记录方法审查。AQI不是仪器QA。实际方法、Datum、监测器字段明细见`metadata_schema.json`、`district_monitor_identity.csv`和`official_field_evidence.json`。

**未通过依据：** 即使先不考虑下载故障，已审88101机构子网2022摘要只有10个一小时站，未达到20个合格连续站目标。这不是整个南海岸所有PM2.5站的总数。建议后续另行确认是否扩大有官方地理/机构证据的区域，或将其限定为小样本站点方法预实验；若考虑88502，必须先核实定义和方法质量，不在本轮混入人工或非同定义站点凑20。EPA开放下载不擅自标成CC0，许可说明及快照Last-Modified均保存在来源清单。

### 22.4 实验3：MAIAC/ERA5访问、四季目录与时间匹配

标准凭据位置`.netrc/_netrc/.cdsapirc`及所检CDS/Earthdata环境变量均未配置。只记录存在性布尔，不输出密钥；没有注册账号、绕过认证或下载受限科学数据。

- NASA：[MAIAC官方入口](https://modis-land.gsfc.nasa.gov/MAIAC.html)、[Collection 6.1指南](https://modis-land.gsfc.nasa.gov/pdf/MCD19_UserGuide_C61_V3.1.pdf)、[Earthdata注册](https://urs.earthdata.nasa.gov/users/new)。需要用户配置可访问该产品的Earthdata登录/授权，之后才能按范围获取HDF。本轮匿名CMR目录查询成功不代表数据文件可免认证下载。
- ERA5：[CDS产品](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)、[官方API设置](https://cds.climate.copernicus.eu/how-to-api)。需要用户登录/注册、手动接受产品条款及配置个人API token；不得把密钥发到报告或聊天。缺少配置即停止对应下载。

四季日期在检查覆盖前固定为1月15—21日、4月15—21日、7月15—21日、10月15—21日，各连续7天。CMR先按同日期UTC目录窗口粗查，不能直接视为EPA当地标准日的精确交集；真实下载/对齐还需合法边缘时刻。空间范围取上述候选站范围加0.1°缓冲：[-118.63276,33.53003,-116.11537,34.36464]。实际查询MCD19A2.061集合`C2324689816-LPCLOUD`，返回网格`h08v05`。

|季节|固定日期|目录返回粒度|目录日期情况|QA有效率/站点AOD覆盖|三源共同样本|
|---|---|---:|---|---|---|
|冬|2022-01-15—21|3|本次查询返回17、18、19日|未测|未测|
|春|2022-04-15—21|7|查询窗口每天有目录记录|未测|未测|
|夏|2022-07-15—21|7|查询窗口每天有目录记录|未测|未测|
|秋|2022-10-15—21|7|查询窗口每天有目录记录|未测|未测|

24粒度的CMR体积字段合计约231.00MB，仅为目录标称体积，不是实际下载量，也不是有效覆盖。冬季另四天未在本次查询返回，不能进一步编造云量、产品可用性原因或零有效像元结论。

MAIAC为1km正弦网格、HDF4-EOS的柱AOD，无量纲，**不是地面PM2.5浓度**。后续须读取scale/fill、AOD_QA位字段、云/邻云质量及按orbit的实际时间，才可计算有效像元、站点邻域覆盖和时空交集；不能把目录日期复制成每日24小时数据。当前未读取任何AOD数组，实际QA有效率、协变量缺失率、最终训练样本量全部未测。

ERA5官方产品为小时UTC、0.25°发布网格；拟取2m气温/露点K、10m u/v风m/s、地面气压Pa、总降水m、边界层高度m。气温/露点/风/地面气压是瞬时语义，总降水按CDS小时累计定义及实际文件step/bounds核对；边界层高度也需核对有效时刻。这些是产品文档规格，不是本次文件回读结果。没有取得NetCDF/GRIB，不能声称实际单位/缺失/网格已检查。

时间处理约束：EPA按记录的当地**标准**时间与Date GMT/Time GMT成对校验，不直接套用包含夏令时的America/Los_Angeles转换；MAIAC按真实过境UTC映射目标窗口；ERA5按瞬时或累计区间处理。日均PM2.5、卫星过境AOD及气象日特征是不同统计量，AOD和气象只能作为协变量。列明不同来源不等于已经审完观测依赖或证明统计独立；产品是否使用同批观测的验证/开发依赖仍未完成逐项审计。

### 22.5 两条候选路线：已有核心与具体缺口

只读审计记录：`04_统计与校验/algorithm_routes.json`。现有RawInputQualityGate、时间区间语义聚合、空间匹配、PyKrige指定外漂移、训练仿射校正、站点残差融合和ResultQualityGate均存在真实代码。但“有实现”“本任务有输入输出适配”“本轮已执行”是三个不同状态，本轮没有运行算法。

|能力/路线|已有真实接口|本任务阻塞|
|---|---|---|
|质量控制/时间/空间准备|quality_gates.py、temporal.semantic_resample、spatial.bilinear等|通用质量门不能代替MAIAC位QA、EPA方法码及当地标准日完整性|
|A：AOD＋ERA5指定协变量驱动PM2.5估计|station.pykrige_universal的specified漂移；已有独立RegressionKriging代码|缺MAIAC QA/过境准备节点及EPA/AOD端口；协变量NaN会被拒绝，须定义真实共同有效域。RegressionKriging不在冻结34中，不能本轮扩库强加|
|B：独立背景/训练历史校正＋同期站点残差|fit_affine_source、spatial.bilinear、fusion.residual、fit/apply状态与DAG执行器|这三类数据没有现成PM2.5背景；缺通用多协变量背景模型fit/apply任务接口。不能把AOD改名为浓度或套用京津冀RH35校正|
|结果验证|ResultQualityGate、AlgorithmQualityGate及独立误差接口|没有真实共同样本和冻结留站划分，不能计算真实精度|

现有`PM25DAGAdapters`仍绑定GEOS-CF RH35、ERA5 t2m及京津冀探索性契约，不能原样处理EPA＋MAIAC。路线A的协变量参与指定漂移与路线B的独立背景＋残差原则上可不同；**若B只是将同一批观测/协变量的回归Kriging拆成趋势＋残差节点，可能数学等价，不能按节点多算成第二路线**。当前没有两条已具备合法输入输出且可直接执行的EPA路线，实验3算法条件未通过。本轮只列缺口，没有开发。

### 22.6 数据隔离建议、文件清单及停止决定

实验1可进入后续另行授权的时间恢复实验；真实NA不作真值，遮蔽种子/长度及时间留段应提前冻结。实验2/3暂不进入正式主实验或GOLEM搜索。若后续站点/交集足够，应整站划分训练/验证/最终测试，同时保留时间留段；回归、校正、归一化只拟合训练集，验证用于选择，测试最后一次评价。不能因目前缺文件就声称已有可执行无泄漏划分；本轮未使用验证/测试成绩筛数据。

|文件/脚本（相对统一数据根）|作用|
|---|---|
|`01_UCI_Beijing/source_manifest.json`、`statistics/*.csv/json`|UCI来源/许可/逐文件校验与实际缺失统计|
|`01_UCI_Beijing/preaudit_uci.py`|可复现UCI下载、解压与统计|
|`02_EPA_PM25/source_manifest.json`、`statistics/*.csv/json`|EPA成功/失败获取状态、五年摘要、机构身份、坐标和方法字段|
|项目`tools/preaudit_epa_public.py`|可复现EPA获取/元数据/分析分阶段脚本，不属于核心算法|
|`03_MAIAC_ERA5/preaudit_access.py`、`source_manifest.json`|凭据存在性及有界匿名目录查询|
|`03_MAIAC_ERA5/maiac_catalogue_granules.csv`|24粒度实际目录、URL及范围，非已下载AOD数据|
|`03_MAIAC_ERA5/seasonal_intersection_status.csv`|四季目录状态，未测QA/交集保留空值/null|
|`04_统计与校验/review_plan.json`|固定日期、完整性门槛及证据边界|
|`04_统计与校验/source_inventory.csv`|统一官方URL、版本、许可、UTC获取时间、SHA256及失败状态|
|`04_统计与校验/overall_result.json`、`verification.json`|三层结论、25份原件校验及只读核查|
|`04_统计与校验/summarize_preaudit.py`|只读汇总现有证据，不联网、不训练、不搜索|

原始不完整part保留为失败证据，不列为有效数据；来源表不会用它们充数。所有研究说明只追加本总报告，未新建分散Markdown。

**EXPERIMENT_1_TEMPORAL_DATA = PASS**  
**EXPERIMENT_2_SPATIAL_MAIN_DATA = NOT_PASSED**  
**EXPERIMENT_3_MULTISOURCE_DATA_AND_ROUTES = NOT_PASSED**  
**FORMAL_ABC_SEARCH = NOT_RUN**

## 23. 实验2/3真实数据与任务接口落地（2026-10-19）

### 23.1 本轮做了什么及验收边界

本轮复用第22章数据，未重下六份已取得的EPA元数据/年度摘要，不再请求全国小时ZIP。新增取得官方空气盆地与四县边界、EPA 2022日摘要，运行真实IDW/Ordinary Kriging空间留站烟测；新增独立EPA任务端口、RF训练状态与防泄漏检查及真实数值Objective。**没有取得MAIAC/ERA5科学数组，没有执行两条真实多源路线，没有运行A/B/C搜索。**

本目录无`.git`，无法提供Git差异检查；本轮只新增独立文件并在`implementation_status.json`记录源码SHA256，未编辑原有业务算法、GOLEM第三方源码、已有预审脚本或冻结池。实验1保留原状。新增数据位于`D:/codex/数据/公开数据三层预审_20261018/05_实验落地_20261019/`，约24.36MiB（含派生表）。旧原件、下载失败part及全部历史实验未覆盖。

|验收项|总体状态|有实际证据的部分|未满足部分|
|---|---|---|---|
|EXPERIMENT_2_DATA_READY|BLOCKED|真实2022日摘要、官方多边形、全部机构五年摘要覆盖|四县仍不足20合格站；2018—2022原始小时及逐小时QA未取得|
|EXPERIMENT_2_PIPELINE_READY|PARTIAL|**PASS：单日真实日摘要、7训练站＋2验证站的两基线烟测**|完整五年小时准备/正式时间留段未验收|
|EXPERIMENT_3_DATA_READY|BLOCKED|凭据核查与第22章匿名目录|Earthdata/CDS未配置，真实AOD QA、ERA5与共同样本未取得|
|EXPERIMENT_3_TWO_ROUTES_READY|PARTIAL|两种不同数学机制的命名端口、RF状态和隔离单元测试|真实文件准备、MAIAC QA/投影/过境解析及两路线真实执行未验收|
|NUMERICAL_OBJECTIVE_READY|PARTIAL|**PASS：2张真实EPA基线GOLEM图→执行器→质量门→独立数值指标**|多源Objective尚无真实数据执行证据|

状态由`05_实验落地_20261019/implementation_status.json`保存，PASS子项的有限范围不能提升为整体正式实验准备完成。

### 23.2 AQS授权检查与分区获取路径

检查常用AQS/CDS/Earthdata环境变量及用户目录配置文件，未发现已配置凭据；没有打印密钥、代注册账号、调用公开测试账号或发起受保护数据请求。[AQS官方API](https://aqs.epa.gov/aqsweb/documents/data_api.html)的sampleData支持州/县/站点等范围，但需要用户自己的email/key。

新增`tools/download_epa_county_month.py`：读取用户环境`AQS_EMAIL`＋`AQS_KEY`（兼容命名见代码），或用户目录`.aqs.json`的email/key；显式`--fetch`才请求`sampleData/byCounty`，每次仅一县一月、88101、duration=1、2018—2022范围内，一次有界请求，无循环重试。已有文件不覆盖。错误只记录类型/HTTP状态，不记可能含密钥的requests URL；原始观测Data不改，返回Header仅保留安全状态/行数，防止回显凭据。

实际运行检查：`aqs_partitioned/access_result.json`为`BLOCKED_NEEDS_PERSONAL_AQS_EMAIL_AND_KEY`、api_requests=0。**分区客户端尚未成功下载真实小时数据，不写成已完成小时获取。** API邮件注册需用户本人确认，入口为官方文档Sign Up；本轮未发送注册请求。

匿名官方替代获取方面，改取较小的[2022日摘要](https://aqs.epa.gov/aqsweb/airdata/daily_88101_2022.zip)，一次请求成功，9,687,295字节，SHA256=`82f5035943a447f8edebf1f0b3f774e08b81a250a86295871a8574726d6b2f94`。这是约9.24MiB的单年日摘要兜底，不是分区原始小时，也不证明逐小时qualifier已经审核。没有继续反复下载75MB全国小时文件。日摘要原包只读保留，派生表另存。

### 23.3 官方地理边界与全部机构筛选

采用[CARB官方空气盆地服务](https://gis.carb.arb.ca.gov/hosting/rest/services/California_Air_Resources_Board_Air_Basin_MIL1/MapServer/0)返回的South Coast Air Basin多边形；四县使用[US Census TIGERweb州县服务](https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1)，按州06、县037/059/065/071取得四个县多边形并合并。当前官方边界快照与源URL/校验值保存在`source_manifest.json`，不是凭矩形框或机构名称替代边界。

逐点读取Datum：WGS84明确按EPSG:4326，NAD83按EPSG:4269经pyproj转换后落入官方多边形；不明Datum不自动当WGS84。此为现行官方边界下的覆盖核查，不是对全部历史边界变更的追溯验证。

|年|空气盆地全部机构一小时站|批准四县扩展后一小时站|四县全部采样类型站|四县物理24小时采样站|
|---|---:|---:|---:|---:|
|2018|8|11|24|21|
|2019|8|12|25|21|
|2020|9|14|25|22|
|2021|9|14|25|22|
|2022|11|15|25|22|

该表来自已取得年度摘要，去掉PQAO/Reporting Agency限制，**仍然只是小时站存在性的上界，不是年度合格站数**。与第22章仅机构子网不同，当前已经执行批准的四县扩展，而非继续只看某一个管理机构。

四县仍未达到20站。作为有官方数据依据的替代建议，使用同一CARB边界对“San Joaquin Valley＋Sacramento Valley”相邻两盆地并集核查，2018—2022一小时站候选分别17、18、27、32、31。2022的31站**尚未做18小时/270日完整性验收**；2018、2019也不足20候选，因此不能直接宣称该区域能保持20站贯穿五年。建议用户确认是否改为此区域，并在真实覆盖支持后考虑2020训练、2021验证、2022测试；本轮没有擅自下载该替代区域原始观测或改换研究目标。全部15盆地的数值证据在`alternative_region_metadata_counts.csv`。

### 23.4 实际2022日摘要质量统计

四县原始日摘要共13,989行：1 HOUR 5,095行、24-HR BLK AVG 5,015行、物理24 HOUR 3,879行；全部原始单位为`Micrograms/cubic meter (LC)`，原始负均值44行，原件不改。该记录数包含不同POC/标准/统计定义，不是独立站日数。

派生规则：官方多边形内；单位明确；只取1 HOUR/24-HR BLK AVG连续观测日统计；Observation Count≥18且Observation Percent≥75；均值有限且非负；按既有EPA方法修订告示排除原方法236/238；保留无事件及Included事件（不将其当无效值），不混入Excluded统计或24 HOUR人工采样。相同站/POC/日多标准记录若数值冲突就拒绝，不平均；本批检测到冲突0个。按年度有效覆盖、总有效小时、最低POC的固定规则选主POC，非预测成绩选POC。

满足270有效日的10站及主POC/有效日：

|站号|POC|2022有效日|
|---|---:|---:|
|06-037-1103|3|365|
|06-037-9033|1|348|
|06-059-0007|3|356|
|06-065-0009|1|346|
|06-065-0500|21|353|
|06-065-1016|1|312|
|06-065-8001|3|348|
|06-065-8005|3|359|
|06-071-0306|1|358|
|06-071-8001|3|364|

共3,509个有效站日，365天有记录，359天至少8站同时有效，257天10站同时有效。经纬度与原Datum明细见`qualified_continuous_daily_2022.csv`、`qualified_sites_2022.csv`及`poc_daily_coverage_2022.csv`。

**限制：** 这些是EPA日摘要Observation Count支持的完整性筛选；没有取得并重新聚合原始小时，无法逐条检查小时qualifier、重复小时、修订替换过程与真实采样间隔。2018—2021本轮只提供全部机构年度存在性，不能冒称已完成五年逐日合格统计。缺少这些证据且仅10站，EXPERIMENT_2_DATA_READY仍为BLOCKED，没有降低20站门槛。

### 23.5 真正执行的空间留站与数值Objective

真实烟测选2022-01-15当地标准日：按固定种子11及坐标几何规则划分7个输入站、2个内部验证站、1个封存测试站；验证站`06-065-1016/06-065-8001`，测试站`06-071-8001`。凸包限制先按坐标确定，不按预测误差挑站；测试站目标未进入外部拟合端口或误差评价。`station_split_metadata_only.csv`只保存站号、POC、坐标/Datum、站名及split，不含浓度、AQI或极值。

这是单日同期输入站预测留出站的**空间烟测**，不是已完成2018—2020训练/2021验证/2022测试。数据筛选用到覆盖/非负等质量条件，不能声称未读取原始测试数据做过质量盘点；保证的是未用其目标拟合、调参或评价。没有真实高分辨率全场真值，不报告全场RMSE。

两种算法实际经现有Registry调用，不重写数值核：`fusion.idw`和`station.pykrige_ordinary`。新任务图转为真实GOLEM OptGraph的nodes_from和命名端口，由既有`NumericalDAGExecutor.run_golem`执行。新`EPATask.objective()`在图执行后才读取独立Validation标签，并返回指标、质量结果、trace、执行/失败次数与耗时；拒绝test标签及训练/验证站交叉。

|方法|验证样本|RMSE|MAE|Bias|实际计算与输出秒|质量门|
|---|---:|---:|---:|---:|---:|---|
|IDW|2|4.6380|4.6267|4.6267|0.1021|WARN，凸包外mask|
|Ordinary Kriging|2|4.4187|4.4175|4.4175|0.2569|WARN，凸包外mask|

误差单位μg/m³；同一日期、同一验证样本。上述是2点误差，**不足以证明Kriging优于IDW**。预测先在统一目标网格计算，再同规则采样验证坐标。目标网格只表示分析单元，不是恢复精度声明。两份`result.nc`均重新打开，数值、有限值mask与内存结果一致；质量门保留域外NaN而非默默外推，禁止无穷值/负预测交付。

最终证据：`output/epa_implementation_20261019/release_smoke/`中的results.json、experiment_status.json、各算法result.nc与validation_predictions.csv、完整DAG trace。最终Objective调用2次、实际图执行2次、失败0；这些是数值执行，不是缓存凑数。Objective同时拒绝错误验证单位/日期及封存测试站改标validation。

### 23.6 两条多源路线最小接口及数学区别

新增`src/ecodata/algorithms/epa_task.py`，与旧RH35任务独立；不把AOD改名为PM2.5，不给旧GEOS端口换名冒充EPA。新增的是任务边界/状态适配，不向冻结34添加一串新算法候选。

**路线A（接口已建，真实输入受阻）：** 令z(s,t)为实际匹配的AOD及ERA5协变量，PM2.5满足`y(s,t)=β0+βᵀz(s,t)+ε(s,t)`；调用现有PyKrige UniversalKriging的specified drift，在该日输入站条件下估计目标场。只使用训练角色站；协变量不全、rank不足及日期不匹配应拒绝，不能用虚构AOD填满。当前产品准备仍缺MAIAC实际HDF QA位/正弦投影/Orbit_time_stamp读取验证、EPA当地标准日与ERA5真实区间匹配，不标完成。

**路线B（接口及状态测试已建，真实输入受阻）：** 调用官方scikit-learn RandomForestRegressor，以训练历史拟合`f_RF(z)`，保存状态；按站GroupKFold生成历史OOF预测和残差审计，训练截止日之后禁止再把目标日期加入RF拟合。目标日使用与A相同的训练角色站条件化：`r_i(t)=y_i(t)-f_RF(z_i(t))`，再复用现有IDW插值残差，输出`ŷ(s,t)=f_RF(z(s,t))+IDW(r_i(t))`。由于RF未训练这个日期，当前残差是时间外误差，而非同样本RF拟合残差；当前代码拒绝在训练截止日前应用这条残差路径。历史RF、OOF和当前空间修正都不得使用验证/封存测试站。

两路线区别不由节点数量定义：A为指定漂移的空间随机场估计，B为历史非线性树回归背景＋时间外站点残差IDW，模型函数类、参数拟合及误差空间处理不同，一般不等价。**但B还使用历史样本、A主要逐日条件化，后续实验须让各候选拥有同一可用数据池并明确“历史拟合”和“同期输入站条件化”两种信息用途；如果目标改为不允许同期站点的纯未来预报，A/B都必须重新限定任务，不能直接比较。** 本轮无多源成绩，未宣称公平效能已验证。

节点输出含RF状态、背景、残差修正、最终场及质量报告。协变量必须声明可追溯的真实文件准备状态、相同目标日；日期/物理量声明不能代替实际产品解析与SHA256证据。当前没有MAIAC/ERA5文件，**两条路线真实训练、预测、质量门及文件回读均为UNVERIFIED**。

### 23.7 实验3访问与共同样本状态

Earthdata及CDS仍缺配置，本轮未再次请求受保护文件、未注册账号。沿用四季7天固定窗口，新增EPA已取得的日摘要覆盖如下；这不是三源交集：

|季节窗口|EPA有记录日|EPA有效站日|每日最少EPA站|三源共同样本/AOD QA/ERA5缺测|
|---|---:|---:|---:|---|
|1月15—21日|7|70|10|未测/null|
|4月15—21日|7|66|9|未测/null|
|7月15—21日|7|68|9|未测/null|
|10月15—21日|7|70|10|未测/null|

未测不填0。没有≥20站、≥100个三源有效共同日期的证据，不能冻结实验3。NASA的目录存在仍不等于像元QA有效；ERA5变量的文档单位也不等于实际文件已回读。没有下载新的多年全球数据。

需要用户完成的必要操作：

1. AQS：按[官方注册说明](https://aqs.epa.gov/aqsweb/documents/data_api.html#signup)用本人邮箱申请key，设置本地`AQS_EMAIL/AQS_KEY`或用户目录`.aqs.json`，不要将key发到聊天/报告；随后可单次请求县月样本。
2. NASA：配置本人Earthdata登录及产品访问授权（[注册入口](https://urs.earthdata.nasa.gov/users/new)），本地持久化方式遵循官方文档，不提交到源码包。
3. CDS：登录并接受ERA5产品条款，按[官方设置](https://cds.climate.copernicus.eu/how-to-api)配置用户目录`.cdsapirc`。
4. 由于四县仍不足20站，请确认是否采用上述中央谷地两盆地替代区域及在覆盖核实后调整年份；授权凭据本身不能解决站点数量问题。

### 23.8 冻结池、测试、失败与复现

原34冻结池SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`。本轮实际运行的两种基线中，仅Ordinary Kriging属于该冻结池，IDW是已有Registry基线但不在冻结34；RF也是既有成熟库的任务组件，**本轮未加入冻结候选**。因此本EPA任务当前有真实验证证据的冻结科学处理路线只有1条；A/B多源为接口级2路线，不能合计成3条已验证搜索图。通用执行器能执行端口图不等于现有typed工厂已能自动搜索所有EPA新任务节点。如此小的已验证空间不应启动GOLEM为默认方案，继续建议先做有限直接对照，不造节点或参数凑空间。

定向自动测试：`tests/test_epa_task.py`及原`tests/test_numerical_dag.py`，**18通过、0失败/跳过/错误，3.722秒**，证据`output/epa_implementation_20261019/directed_tests_release.xml`。覆盖两路线依赖解析、GroupKFold站点隔离、未来历史/错误单位/错误split拒绝、封存测试站不得改标train、Objective拒绝test/错误验证单位/错位日期且不记为真实执行、无产品来源声明拒绝。RF相关用的是明确标注`test_covariate`的单元测试夹具，**不是实际AOD，也不作为实验3运行成功证据**。没有运行全量回归。

失败记录保留：smoke1因当时缺足够内部留站几何而在算法执行前停止；随后修复EPA Event Type实际`Included`值处理（不是根据误差筛数据）及确定性几何留站。smoke2两图在输出valid_domain时触发xarray DataArray/tuple类型错误，已改为显式numpy值；smoke3、final_smoke、verified_smoke和release_smoke成功。累计五个有数值调用的烟测目录共10次图执行：2次失败、8次成功（成功含相同两算法重复回读/隔离复核，不冒充8张新图）。最终只引用release_smoke的同一固定样本成绩。另一次证据汇总遭遇Windows默认GBK解码，已显式UTF-8，不改数据。

复现与关键路径：

```text
.venv/Scripts/python.exe tools/download_epa_county_month.py --root "D:/codex/数据/公开数据三层预审_20261018/05_实验落地_20261019/aqs_partitioned"
# 本地个人授权配置好后，仅增加 --fetch 请求默认的一县一月，不会自动获取五年。
.venv/Scripts/python.exe tools/run_epa_daily_smoke.py --output output/epa_smoke_new_directory
.venv/Scripts/python.exe -m pytest tests/test_epa_task.py tests/test_numerical_dag.py -q -p no:cacheprovider --basetemp=output/pytest_epa_new_directory
```

下载/边界脚本`tools/prepare_epa_implementation.py`；独立数值端口`src/ecodata/algorithms/epa_task.py`；数据根下finish_checks.py只做本地校验/统计；原始来源与SHA256在source_manifest.json，站点、日覆盖、替代区域及四季交集状态CSV均在05目录。唯一研究说明仍为本总报告，没有新增分散Markdown。**不冻结三个实验，不启动正式A/B/C；下一步先取得用户本地授权并确认可达到站数要求的区域。**

## 24. 三个递进实验：真实数据、基线和多源处理路线交付（2026-10-19）

### 24.1 本轮结论与状态

本章替代第23章“必须先配置NASA/CDS授权才能继续”的下一步建议，但不改写其历史失败结论。本轮没有新增分散Markdown，没有运行正式A/B/C、修改第三方GOLEM、训练FNO/UNO或扩充冻结34候选。

**实际完成：UCI固定遮蔽基线；44站EPA日均空间基线；114天真实NOAA卫星数组与全年ERA5数组；两条不同多源DAG及真实数值Objective。不能据此宣布正式GOLEM研究条件全部满足。**

|项目|本轮验收状态|范围与限制|
|---|---|---|
|实验1|已真实执行并通过|12站PM2.5，固定随机点及6/24/72小时遮蔽，4个基线；区分双侧恢复与过去信息恢复|
|EXPERIMENT_2_DATA_READY|已真实执行并通过|2022年、44站、15,436站日；不是2018—2022完整小时数据库|
|EXPERIMENT_2_PIPELINE_READY|已真实执行并通过|14个固定日期、53条共同验证记录，IDW/OK均执行、输出及回读|
|EXPERIMENT_3_DATA_READY|已取得真实数组并通过有限范围检查|105个日期有共同样本，但只有16天达到20个共同站点；独立留出覆盖仍稀疏|
|EXPERIMENT_3_TWO_ROUTES_READY|已真实执行并通过，质量WARN|显式非负约束版本，两路线分别训练/执行/回读；无约束负值失败保留|
|NUMERICAL_OBJECTIVE_READY|已真实执行并通过|4个明确图配置；真正调用PyKrige、sklearn RF及现有IDW，不是结构分数|
|正式A/B/C研究是否就绪|否|当前任务适配器只有4个可直接枚举的配置，多源烟测仅3条、1个验证站；不能证明进化搜索创新|

### 24.2 国内外选择与替代依据

只深入复用3个地面方案；评分为预备工作可行性的0—5级判断，不是数据精度测量。权重依次为质量25%、可获取20%、交集20%、路线15%、规模10%、许可/成本10%；硬性缺陷不能被分数抵消。计算表在`delivery_evidence/source_comparison.csv`。

|方案|六项分数|总分/100|决定|
|---|---|---|---|
|既有京津冀60站|2.5/5/4/4/4/2|72.5|不作正式主数据：既有证据仍未确认站点坐标基准及原始再使用许可；保留全部历史数据|
|EPA南海岸/四县|5/4/2/4/2/4.5|74|不作主数据：前章严格连续日均口径只有10站，不降低门槛|
|EPA中央谷地＋旧金山湾区|5/4/3/4/4/4.5|82|选择主方案：官方多边形、Datum、单位和日统计定义可核实，44站达到要求|

辅助组合比较：①既有京津冀Sentinel-2 AOT＋ERA5，有实际文件，但不能消除地面坐标问题且只有一个月；②EPA＋MAIAC C6.1＋CDS ERA5仍作为**有条件备用**，缺Earthdata/CDS授权，不宣称已取得其HDF；③本轮选择EPA＋**NOAA GOES-17 ABI AOD**＋通过Open-Meteo公开访问的明确ERA5模型。第三种无需账户，已读取实际数组。备用不是“已全部可用”。

NOAA AOD与MAIAC都描述柱气溶胶光学厚度，但不是同一产品：GOES为地球静止卫星，550nm、固定网格约2km星下点采样，斜视像元更大；本例取20UTC时段第一幅真实扫描，不是MODIS过境、不是MAIAC 1km每日产品，也不是地面PM2.5。产品替代会改变采样、角度、质量及偏差，不得混称MAIAC。

### 24.3 实际数据、来源与口径

|来源|真实文件与范围|定义、质量及许可边界|
|---|---|---|
|UCI北京|复用7,959,991字节官方ZIP，12×35,064小时，2013-03—2017-02|PM2.5有效412,029、缺失8,739；无坐标，限时间任务；CC BY 4.0|
|EPA AQS|复用9,687,295字节`daily_88101_2022.zip`，不重下全国小时ZIP|88101、`Micrograms/cubic meter (LC)`；官方日统计，不伪称自行从原始小时重聚合|
|CARB|复用官方空气盆地GeoJSON|按Sacramento Valley、San Joaquin Valley、San Francisco Bay三个官方多边形筛选，不用县名或矩形冒充盆地|
|NOAA GOES-17|114个实际NetCDF，共742,468,142字节|原始AOD、DQF、投影、扫描起止时间均读取；官方NODD开放使用、署名、不暗示NOAA背书|
|ERA5公开访问|7份真实小时JSON，42个采样节点，全年8,760小时，5个气象变量|明确`models=era5`、`elevation=nan`；温度/露点°C、气压hPa、风速km/h、小时降水mm；Open-Meteo数据CC BY 4.0，免费API限非商业条件|

官方说明：[UCI](https://archive.ics.uci.edu/dataset/501/beijing)、[EPA下载](https://aqs.epa.gov/aqsweb/airdata/download_files.html)、[EPA字段与日统计定义](https://aqs.epa.gov/aqsweb/airdata/FileFormats.html)、[NOAA官方开放分发](https://registry.opendata.aws/noaa-goes/)、[GOES AOD算法说明](https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Baseline/ATBD_GOES-R_Aerosol_Optical_Depth_v4.2_Feb2018.pdf)、[ERA5访问/变量/许可说明](https://open-meteo.com/en/docs/historical-weather-api)。具体逐文件URL、保存时间及SHA256分别在卫星manifest、ERA5 manifest和`delivery_evidence/raw_file_hashes.json`；复用原件另有`reused_official_sources.json`。不把未注明专门许可的EPA/CARB文件擅自标成CC BY或CC0。

EPA筛选沿用前章规则：连续`1 HOUR`/`24-HR BLK AVG`，有效小时数≥18且覆盖≥75%，浓度有限且非负，不混入物理24小时人工采样；只接受已明确事件口径，不平均POC；同站/POC/日不同标准冲突拒绝；按有效覆盖、小时数及POC固定选择，年有效日期门槛270天。最终44站实际最少309天，每个日历日均有至少20站。监测方法代码170占15,310站日，209占126站日；保留逐行Method Code、POC及事件信息。日摘要没有完整逐小时Qualifier、仪器维护和所有修订日志，**不声称这些检查已完成**。

原始Datum为WGS84或NAD83，按记录分别由EPSG:4326/4269转换至4326；不是从数值相似猜基准。空间算法采用现有米制投影距离，输出保留4326。不承诺厘米级坐标精度。EPA日统计官方定义为当地**标准日午夜到午夜**，本区固定UTC−8，不套夏令时。NOAA扫描约20:01—20:04UTC，对应标准时中午，保留真实起止；只作同日辅助信息，不扩成24小时卫星观测。ERA5瞬时变量分开求日平均；前一小时降水按区间归属求和，不按端点日期错误求和。因未额外取得2023-01-01 00时，最后一天降水缺少完整边界，保留NaN；当前模型不使用降水特征。

ERA5原生0.25°，本次以1°节点取样控制数据量，并关闭API高程统计降尺度；JSON返回的42个中心坐标已逐点核对。后续插值仅是特征匹配，不构成更细的真实气象信息。所有气象变量及原始小时值保留，而本轮两路线只使用AOD、2m气温两个协变量。

卫星只接受DQF=0高质量或1中等质量；保留原DQF及“仅0”的覆盖统计，拒绝2/3、填充值、越界和负AOD。上游检索已有云筛选，**未检索像元不能全部解释成云**，没有虚构独立云覆盖率。原始尺度因子/有效范围通过实际NetCDF编码处理。保留mask和真实时间；区域匹配数组是分析网格，不是新的高分辨率真值。

### 24.4 实验1：真实时间恢复基线

训练2013—2015、验证2016、一次固定测试2017-01—02，随机种子11。仅遮蔽原本有真值的位置，自然缺失不评分；随机点、6/24/72小时连续块分别保存。历史小时中位数及日历KNN仅用训练年份拟合；KNN为24邻居的小时/年内季节坐标回归，不冒充利用其他污染物的多变量恢复。过去信息基线为前值保持；双侧时间线性插值允许使用后续观测，明确不是预测任务。

下表为2016验证RMSE，单位μg/m³；完整RMSE/MAE/Bias、2017固定测试结果、逐站计时和掩码在`uci/`。

|缺失场景|双侧线性|过去前值|训练期小时中位数|训练期日历KNN|
|---|---:|---:|---:|---:|
|随机点|12.031|16.772|77.781|92.254|
|连续6小时|55.018|86.568|176.785|158.515|
|连续24小时|25.350|22.273|46.672|69.724|
|连续72小时|49.110|65.552|50.023|78.502|

不同缺失场景采用不同的固定遮蔽位置，不能把跨场景误差排序当成严格难度单调关系。测试集没有用于挑方法或修改参数。结果说明简单方法已经有代表性，不能先假定必须使用GOLEM。

### 24.5 固定空间/时间划分与防泄漏

主方案44站：**36输入训练站、4验证站、4最终测试站**。训练日期2022-01—03，验证04—08，最终测试09—12。允许用输入站同期观测重建留出站，属于空间重建而非未来预测；RF历史拟合只到3月31日。

初期中央谷地29站试跑已有部分目标暴露，故未把它们重新包装成“从未使用的最终测试”。从同一原始文件增加15个湾区合格站后，固定封存`06-001-0007、06-001-0011、06-013-0002、06-085-0006`，它们未参加本轮任何模型拟合或误差评分。最初纯湾区验证方案因实际QA/覆盖域不足而失败，保留证据；随后改为2个内陆＋2个湾区验证站，**依据输入有效域，不依据预测误差**，最终测试站不变。早期试跑不是正式独立测试证据，未复用其模型状态。

在当前冻结数据中，共同样本1,452站日、105天；只有16天达到20个共同站点。因此“105天有样本”不等于“44站每天完整”。按站点角色及时间窗口的主要可用量：

|用途|站日|日期|实际站数|说明|
|---|---:|---:|---:|---|
|历史训练站×训练期|290|21|26|其余输入站没有这段共同AOD，不补造|
|验证站×验证期|51|40|4|尚需按当天支持域进一步约束|
|最终测试站×测试期|47|25|3|固定4站中1站暂无共同样本；只统计可用性，不计算测试误差|

`prepared_train.csv`、`prepared_validation.csv`及同期输入表已经生成；`test_covariates_no_targets.csv`不输出测试目标值。原始EPA文件仍完整保留。所有模型拟合、特征匹配和数据可用性统计与最终测试评价相区分。

### 24.6 实验2：真实空间重建

固定验证日期为4月15—21日及7月15—21日；36输入站预测4验证站，按同一有效支持域与实际缺测得到共同53条样本。两算法共28次真实DAG执行，全部输出NetCDF、质量报告、预测CSV及回读检查。不是最终测试成绩。

|方法|共同验证样本|RMSE|MAE|Bias|
|---|---:|---:|---:|---:|
|现有IDW|53|1.17093|0.90358|−0.44541|
|PyKrige Ordinary Kriging|53|1.26466|0.96997|−0.32996|

这是空间接近的站点、小型日期样本结果，不外推成全年/跨地区精度；对网格完整性和凸包外mask的WARN不解释为误差已被证明。

### 24.7 实验3：两条真实路线、问题与最终小样本

路线A：当日输入站浓度＋实际AOD/ERA5指定漂移项，调用官方PyKrige UniversalKriging，估计协变量趋势与空间残差。

路线B：用1—3月训练站290条共同记录训练官方sklearn随机森林（50树、深度6、叶最少3、种子11）；GroupKFold按站分组生成历史OOF诊断。应用到4月后，再用**同期输入站、但不在历史RF拟合期**的误差做现有IDW残差校正，预测为RF背景＋残差。不是在RF训练样本上直接取过拟合残差。真实fit站名、时间、样本数及分组在`fit_isolation.json`。

两路线不等价：A按当日站点拟合指定漂移Kriging；B有历史非线性回归状态和后续残差分支。两者共享可用输入资源，但有效利用的信息不同，不把差异全部归因于“网络/图更复杂”。

真实故障及修复：

- 同一坐标的带mask栅格被重复`interp`，扩大NaN区域；改为完全相同网格直接取值，只有确需转换时才插值。训练协变量仍要求完整且满秩；仅UK显式允许目标缺测像元保留mask，不静默填充。
- 某日没有可支持验证站时，原检查触发空数组归约错误；现根据输入QA及凸包/完整读出像元邻域明确拒绝并记录，不按RMSE筛选。
- 无约束UK/残差融合确有负浓度，被原质量门BLOCK。保留原失败图、异常位置及轨迹。另设**默认关闭、显式启用**的非负投影，所有方法采用同一规则；记录投影像元、原最小值及最大修正，质量报告WARN。它不是新的学习算法，也不消除模型偏差或证明守恒。

最终数据齐备后，按预先固定规则取前三个输入QA/空间支持通过的验证日期：4月5、8、17日，4月2、15日因支持邻域不足被拒绝，原因已保存。4图×3日共12次真实GOLEM图数值执行及NetCDF回读通过，**只有3条、同1个验证站**，不能作论文精度排名：

|方法|RMSE|MAE|Bias|
|---|---:|---:|---:|
|IDW|1.57563|1.49533|−1.49533|
|Ordinary Kriging|1.62832|1.61998|−1.61998|
|路线A|1.98138|1.95908|−1.95908|
|路线B|3.87951|3.01246|+2.13298|

**没有观察到多源路线优于简单基线的证据。**路线B三个日期投影95、0、1,523个像元，原始最小值分别约−4.790、+4.320、−8.581μg/m³；路线A在4月17日投影2个像元。负值比例和训练到验证的季节变化提示模型适用性风险，不以投影后的非负通过掩盖。整个场没有独立高分辨率真值，不计算虚假全场RMSE。

### 24.8 Registry、DAG、Objective与GOLEM适用性

沿用现有Registry、named-port DAG、fit→apply状态、双层质量门及官方GOLEM OptGraph。实际执行入口为`EPATask.objective(..., golem_graph=...)`→`NumericalDAGExecutor.run_golem()`，逐边契约/输入指纹/节点时间/结果均入迹线。成功返回真实验证RMSE/MAE/Bias、质量报告、计时和执行计数；失败保留异常/trace，附实际耗时及计数，不将失败伪装成0误差。

当前这个任务适配器明确支持4个图配置：IDW、OK、外部漂移UK、RF＋残差。前两者共享拓扑但算法不同，整体仅3种拓扑族。**这不是冻结34候选的任意组合全部可执行；RF是已有任务适配器中的拟合组件，不新增为冻结通用随机节点。**冻结池SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`。

现有4配置可低成本穷举。应先比较固定规则/穷举与真实误差，不能把这个小空间直接作为“语义Mutation创新”的正式主证据。本轮没有运行A/B/C，也没有虚构合法率、搜索成功率或C优越性。

### 24.9 路径、复现、测试与停止边界

全部新产物根：`output/three_experiments_20261019/`。

- `uci/`：固定遮蔽、384组计时、预测、完整指标和配置。
- `northern_california/`：44站元数据、逐POC覆盖、15,436站日、质量审计。
- `anonymous/original/`：114份原始卫星NetCDF、ERA5小时JSON及目录查询；原始下载/补取失败记录保留在manifest中，完整文件不重复下载。
- `anonymous_north/`：114天一致网格AOD、DQF、真实扫描时间及分站QA统计；不是MAIAC。
- `anonymous/era5_daily_2022.nc`：气象日统计及区间语义；原始分辨率在元数据中保留。
- `north_mixed_release/`：固定14天空间基线28次实际执行；不把其他重放目录算作新独立实验。
- **`complete_data_release/`**：最终114天输入快照、固定站点/时间划分、3日12图、NetCDF、验证预测、fit隔离、图契约及每次质量结果。
- **`delivery_evidence/`**：准备好的训练/验证/同期输入表、无目标测试协变量、逐样本资格、共同覆盖、汇总指标、来源URL/SHA256、环境和代码版本、机器可读readiness。
- 所有旧数据、老实验、无约束失败、访问超时及空支持域失败均保留；没有删除或覆盖原始资料。

主要脚本：`tools/run_uci_temporal_baselines.py`、`tools/run_epa_daily_smoke.py`、`tools/prepare_anonymous_multisource.py`、`tools/fetch_era5_regional_public.py`、`tools/run_three_source_ready_smoke.py`、`tools/summarize_three_experiment_preparation.py`。正式复现使用新输出目录，复用原始文件，不重下GEOS-CF或其他历史数据：

```text
.venv/Scripts/python.exe tools/run_uci_temporal_baselines.py --output output/uci_reproduction_NEW
.venv/Scripts/python.exe tools/run_three_source_ready_smoke.py --fresh-holdouts --nonnegative-projection --satellite-snapshot output/three_experiments_20261019/complete_data_release/source_snapshot.json --output output/three_source_reproduction_NEW
```

必要代码仅涉及上述任务脚本、`epa_task.py`、`open_source_stations.py`及其测试，没有修改第三方优化器。最终定向测试：`tests/test_epa_task.py tests/test_numerical_dag.py tests/test_algorithm_library_expansion.py`，**43通过、0失败、0跳过、0错误**；XML为`targeted_frozen.xml`。只有第三方flox的NumPy弃用提示。验证了未知来源拒绝、训练/验证/测试隔离、单位/日期错误拒绝、带mask UK目标覆盖、相同网格不扩NaN、默认负值BLOCK以及显式投影WARN。没有运行全量回归或正式GOLEM实验。

最终独立文件核对记录为`delivery_evidence/artifact_verification.json`：12份NetCDF全部重读，网格/单位/非负/有效域一致；3份真实RF拟合记录及站点分组无交集；114份卫星快照SHA256一致；准备好的290条训练记录与实际执行所用训练记录及特征逐值一致。原始下载目录全部设为只读；未修改原件内容。

最终决策：实验1可以开展固定方案的正式时间恢复比较；实验2可以开展2022年范围内的正式空间验证；实验3工程入口和两路线已真实跑通，数据存在但留出支持稀疏，非负投影暴露较大模型风险。**不宣布三项研究全部完成，也不启动正式A/B/C。**下一步先在冻结的共同有效验证集上穷举4配置、报告覆盖率及失败率；只有出现科学必要且数量足够的非等价可执行图，再冻结统一预算开展A/B/C。最终测试目标继续封存；需要MAIAC特定产品时才需用户Earthdata授权，本轮公开替代路径不再被授权阻塞。

## 25. 四种冻结配置的固定验证集穷举比较（2026-10-19）

### 25.1 执行范围及公平口径

**已完成真实穷举：IDW、Ordinary Kriging、外部漂移UK、RF＋残差校正，36个验证日期×4配置，共144次实际DAG执行、144份NetCDF回读，无方法执行失败。不是GOLEM搜索，也不是重新选择三个有利日期。**

复用第24章`complete_data_release/`的配置、站点划分、非测试匹配表及卫星快照，没有打开包含最终测试目标的原始地面数据表，没有下载数据、改原始数据、扩充算法库或改第三方GOLEM。36训练站/4验证站/4最终测试站及34项冻结池不变；拟合截止2022-03-31，比较期2022-04-01—08-31。

四配置在每个日期使用**同一批9—19个共同有效输入站、同一目标网格、同一单位、同一物理质量门、同一显式非负投影规则和同一验证样本**。IDW、OK不使用卫星特征，但不额外给它们更多无AOD的输入站。UK使用实际AOD和ERA5气温指定漂移。所有配置获得相同可用历史资源；RF确实使用训练期290条历史目标，其余方法按既有定义使用当日输入站拟合，不伪称四者实际历史信息利用量完全相同，不改变算法来制造形式一致。

RF按原配置50树、深度6、叶最少3、种子11拟合，沿用按站分组OOF检查及跨训练期残差方案。仅给原`EPATask.objective`增加可选的既有DAG执行器参数，默认行为不变；复用DAG内容指纹缓存，真实历史拟合节点执行1次、缓存复用35次，未把缓存命中算成35次新训练。这个fit节点内部包含5折诊断和一次最终RF训练，不是总共只调用一次sklearn.fit。每个日期仍实际计算预测、融合、质量门和输出。缓存状态不跨本次进程复用，不加载不明旧权重。

### 25.2 先冻结共同队列，再执行四种方法

原固定验证数据为**51条、40天、4站**。在查看本轮预测误差之前，按第24章既有输入要求统一检查：至少8个有效输入站、UK漂移矩阵契约、与质量门相同的米制凸包、AOD/气象有效mask及完整网格回读邻域。不是按目标值或算法成绩筛样本。

|资格结果|样本数|处理|
|---|---:|---|
|共同合格|38|4种方法全部参加，样本键相同|
|当天输入站凸包外|8|所有方法统一不纳入精度排名|
|共同回读像元邻域不完整|4|不填补卫星缺测，统一不纳入|
|当天有效输入站不足8|1|保持原门槛，不强行拟合|

资格表为`sample_qualification.csv`，固定队列和哈希分别为`fixed_validation_cohort.csv`、`cohort_freeze.json`，均在候选执行前写入。合格队列**38条、36天、2站**：`06-067-0006`占36条，`06-031-0004`仅2条。其余2个原验证站没有本轮共同覆盖域内的合格记录；没有换站凑数量。

所有配置都完成38/38条，因此本轮排名不存在“某个方法只在自己的成功子集上评分”。程序还明确规定：任一配置若未完成整个冻结队列，则全队列RMSE/MAE/Bias不发布为可排名结果，只报告缺失及失败，不能靠剩余容易样本争第一。13条共同输入不合格记录与实际算法执行失败分开统计，不伪称发生52次算法失败。

### 25.3 完整精度、覆盖和成本对照

下表RMSE、MAE、Bias均为μg/m³，以**同一38条样本合并计算**，不是先计算每日RMSE再平均。单次CPU执行与文件I/O耗时不是多次硬件性能基准。

|配置|RMSE|MAE|Bias|共同队列覆盖|相对原51条覆盖|失败/实际日期|累计执行秒|
|---|---:|---:|---:|---:|---:|---:|---:|
|IDW|1.7081|1.3229|+0.1278|100%|74.51%|0/36|3.241|
|Ordinary Kriging|**1.2879**|**0.9861**|−0.1783|100%|74.51%|0/36|4.308|
|外部漂移UK|2.4576|1.8613|+0.9175|100%|74.51%|0/36|4.610|
|RF＋残差校正|2.7848|2.3616|+1.4640|100%|74.51%|0/36|5.940|

RF累计时间包含一次真实拟合节点0.182秒；其余非拟合墙钟约5.759秒，包括缓存处理、预测、质量门和读写，不称为“纯推理时间”。全流程约33.10秒，另含输入匹配、资格检查及汇总开销。四种方法没有隐藏失败重试，没有根据比较结果重新调参。

**网格覆盖和留出点覆盖不是一回事。**四者均用相同200×160矩形网格、36个日期；按1,152,000个像元时次计：IDW/OK有限输出268,973，占23.35%；UK/RF有限输出164,439，占14.27%。这包括矩形中凸包外的不可外推区域，不是面积加权的有效研究区域覆盖率。UK/RF额外受实际AOD缺测约束，不能把其站点队列100%写成整个区域100%完整。

### 25.4 质量门和非负投影

4配置各36次质量门均为**可继续但需说明的WARN**，不是无警告PASS。凸包外或卫星缺测mask被保留；没有NaN被偷偷填补，没有Inf、错误单位或回读后掩码变化。投影前负值按第24章已启用的同一规则处理并记录，本轮不新增或放宽质量阈值。

|配置|非负投影像元时次|占本配置有限网格输出|投影前最小值|最大修正量|
|---|---:|---:|---:|---:|
|IDW|0|0%|+0.0973|0|
|OK|0|0%|+0.6818|0|
|UK|90|0.0547%|−1.1374|1.1374|
|RF＋残差|9,752|5.9305%|−17.7854|17.7854|

这些是跨日期像元时次，不是9,752个不同地理位置。投影使浓度非负，不证明原模型可靠、守恒或精度提升；RF较大修正量是应保留的适用性风险。逐次`quality_report.json`、`execution.json`保留数值、阈值、真实依赖边、缓存命中和时间。

### 25.5 最佳方法的证据强度与继续/暂停决定

**仅在这个固定队列上，OK是描述性最优。**相对IDW，RMSE低0.4202（24.60%）；38个配对样本中23个平方误差更低、15个更高，不是逐点全面获胜。UK和RF相对IDW的RMSE分别高0.7495、1.0766，没有多源必然优于单源的证据。

但36/38条来自同一站，占94.74%，且时间上重复观测，不能当作38个独立空间样本。另一站只有2条，RF恰在这2条上低于OK，不能据此推导任何站点类型优势。未使用独立同分布t检验、伪造显著性或把重复日期当成大量独立站点；没有计算最终测试指标。**当前只支持“该验证队列暂选OK作参照”，不支持“OK在新站点或地区已可靠胜出”。**

建议补充的是固定验证站的真实共同协变量与空间支持覆盖，尤其当前缺少的两个湾区验证站，而不是降低QA、取消凸包、复制卫星时刻或偷看最终测试目标。追加数据应先固定来源/时段/规则，再单独版本化比较；本轮不下载或修改数据。

**明确结论：本轮穷举任务完成；暂停正式A/B/C。**工程运行与数值接口已验证，但4个配置已能完整穷举，缺少足够大的非等价组合空间，当前空间验证证据也偏弱。继续整理和补强独立验证是合理下一步，不能仅凭这次OK低RMSE宣布物理语义Mutation有效。最终4站目标继续封存。

### 25.6 交付路径、定向测试与复现

结果根目录：`output/fixed_validation_enumeration_20261019/`。

- `analysis_plan.json`：执行前方案、评价分母、缓存规则、冻结输入SHA256；`graph_contracts.json`：沿用4图端口。
- `fixed_validation_cohort.csv`、`sample_qualification.csv`：38条共同队列与原51条资格及排除原因。
- `predictions.csv`：4×38条逐样本预测及真值；`all_validation_outcomes.csv`：原51条×4配置，明确区分152条成功预测与52条共同输入不合格记录；`comparison.csv/json`：全队列指标、失败/覆盖、耗时和修正量。
- `by_station.csv`、`by_date.csv`、`paired_descriptive_differences.csv`：分站、分日及同样本差异，不作独立泛化证明。
- `grid_coverage.csv`：逐图逐日期真实网格有效数；日期/方法目录中为144份NetCDF、质量报告、执行记录、RF拟合隔离证据。
- `verification.json`：144文件再次独立回读、36份RF隔离文件、冻结输入及队列哈希不变、指标从CSV重算一致。
- `code_environment.json`：本轮代码SHA256及实际运行环境；此前冻结数据与历史报告不覆盖。
- `conclusion.json`：完成144次、无失败、RF真实拟合1次/缓存35次、最终测试未打开、正式A/B/C仍为false。

代码仅新增`tools/run_fixed_validation_enumeration.py`，给`epa_task.py`的Objective添加可选执行器复用入口，没有修改算法数学定义、第三方GOLEM或34候选池。后者哈希仍为第24章值。结果复现命令（使用不存在的新输出目录）：

```text
.venv/Scripts/python.exe tools/run_fixed_validation_enumeration.py --output output/fixed_validation_reproduction_NEW
.venv/Scripts/python.exe -m pytest tests/test_fixed_validation_enumeration.py tests/test_epa_task.py tests/test_numerical_dag.py -q -p no:cacheprovider --basetemp=output/pytest_fixed_validation_NEW
```

定向测试最终**26通过、0失败、0跳过、0错误**，见`targeted_tests_delivery.xml`。覆盖资格判定不读取验证目标、NoData不填充、站点不足共同拒绝、方法缺样本不能按成功子集排名、失败图仍计入网格覆盖分母、空队列不得生成0误差、训练内容变化使RF缓存失效及原DAG防泄漏检查。首次新增边界测试发现汇总器在失败记录缺投影字段、空对象类型预测列时有2处错误，已最小修复并重测；失败XML保留为`targeted_tests.xml`。修复仅影响空/失败汇总分支，本轮144次成功计算及指标不变，已从保存的预测独立重算核对，没有为此重跑全部实验。未运行全量回归、下载、正式A/B/C或最终测试。

## 26. 固定台湾2017路线：旧下载清理与真实基线执行（2026-10-19）

### 26.1 本轮结论与路线

固定为：实验一保留UCI北京时间恢复；实验二用台湾2017站点空间插值；实验三用同一批站点及论文配套AHI AOD。**不再使用美国数据作为当前主路线，不加入ERA5，不执行正式A/B/C。**

本轮不是仅完成计划：已删除114份经来源与哈希核对的旧下载，读取真实台湾小时观测和卫星数组，并通过现有GOLEM图桥接、DAG执行器及质量门完成32次真实计算和NetCDF回读。但**本轮尚未达到全部验收条件**：13.44GB官方包未下载完整及通过MD5；配套表的时间约定和坐标基准仍缺直接说明。下面数值只属于明确假设下的小样本工程验证，不是已完成的正式研究结果。

依据：[论文](https://doi.org/10.1029/2020EA001599)、[作者配套数据](https://zenodo.org/records/4420335)。论文台湾部分使用77站、2017年AHI资料、30分钟检索，采用高/中两档质量，并进行小时匹配。论文结果与本次固定留站、短时样本、固定带宽并不具有完全相同条件；不机械比较论文RMSE，也不声称严格复现了论文全年的交叉验证。

### 26.2 实际清理与保留范围

- 删除位置仅为项目内`output/three_experiments_20261019/anonymous/original/OR_ABI-L2-AODC*.nc`中通过审核的114份文件。
- 清理前逐份与旧`satellite_manifest_*.json`的NOAA官方下载URL及SHA256匹配，并检查`src/tools/tests/docs`逐文件名引用；114份均未发现直接逐文件引用。旧下载器的目录级读取及历史报告引用仍保留，代表已废弃路线，**旧路线重新下载原始文件后才能完整复跑**；不宣称所有历史脚本仍可无下载复跑。
- 实际删除文件大小合计 **742,468,142字节（约708.07MiB）**。这是按已删除文件大小统计的清理量，不把新数据下载后的磁盘净变化冒充释放量。清理后又取得新数据和安装小型研究依赖；最后一次D盘剩余约211.53GB。
- 证据：`output/taiwan_2017_preparation_20261019/cleanup_candidates.csv`和`cleanup_execution.json`，含完整路径、来源、哈希、大小、理由和执行结果。删除是物理删除，未放回收站；可按保留的官方下载URL重新取得。
- 保留旧预测CSV、NetCDF结果、质量JSON、历史日志、下载manifest、校验值、旧脚本及总报告。没有删除源代码、配置或已有测试。
- `D:/codex/数据/公开数据三层预审_20261018/`在项目目录外，本轮按用户禁止越界删除的要求保留，其中旧EPA文件未冒充已清理。京津冀原始数据、气压湿度数据、WRF资料、UCI与所有其他用户原始数据均未删除。

### 26.3 下载、许可与真实文件状态

官方API：`https://zenodo.org/api/records/4420335`；实际文件地址：`https://zenodo.org/api/records/4420335/files/aod_pm25_data.tar.gz/content`。

|项目|实际核实结果|
|---|---|
|文件|`aod_pm25_data.tar.gz`，13,435,034,939字节|
|官方校验|MD5 `d7329012a175b3ad52012d1f3d6dd50f`；**本地尚未验证通过**|
|许可|Zenodo元数据声明CC BY 4.0；记录发布日2021-01-05|
|空间|下载前D盘剩余212,522,188,800字节，足以下载及选择性解压|
|读取到的台湾成员|`zhang_data/pm25_taiwan/Taiwan_PM_2017.csv`、`StationInfo.csv`及158份`ahi_aod_taiwan` NetCDF，共160份完整成员|
|排除范围|提取程序只接收这两个台湾子目录，没有把美国成员用作台湾数据|
|失败证据|普通流式下载较慢；随后使用官方HTTP Range断点获取，存在SSL错误、超时及不完整响应；10个分段在有限重试后仍失败，已停止而非无限重试|

数据位置：`data/taiwan_2017_zenodo4420335/`。`official_record.json`保留官方元数据；`source_manifest.json`、`provisional_member_inventory.json`保存来源及逐成员SHA256。**逐成员SHA256是本地完整性指纹，不是作者另行提供的独立校验；不能代替官方整包MD5。**

下载停止时HTTP范围/长度检查已接受1,219,526,656字节，完整连续前缀414,220,288字节；另有后续已完成分段。未完成包明确命名为`aod_pm25_data.tar.gz.part`，不是下载成功的原始包。移除了下载器预留的未使用尾部空间，当前临时文件逻辑长度1,404,076,032字节，不能把原先预分配的13.44GB算成下载量。进度与失败保存在`range_progress.json`及实验输出中的`download_status.json`。这部分未使用预留空间回收不计入旧实验数据清理量。

`provisional_members/`中完整读出的成员保留原值并已设只读；其状态为待整包校验，不是正式冻结数据。断点脚本只重新取未完成分段，有界重试，多个失败后停止；校验成功后才把`.part`改成正式包名。没有注册账户、绕过访问限制或下载大型模型。

### 26.4 数据质量与不能假装已核实的问题

地面表实际字段为`stn_eng, DT, PM10, PM2.5`；673,152行、77站，日期2017-01-01至2017-12-31，PM2.5缺失24,503行，重复站点时间0条，负浓度0条。年内缺行和NaN分别统计在`station_annual_coverage.csv`，不把“每站出现过”解释为全年每小时完整。PM2.5的μg/m³含义由配套论文支持；原CSV不含仪器QA列，不凭空生成仪器质量标记。

站点表为`SiteEngName,Lon,Lat`，77行。**未声明WGS84/TWD97等datum。** 工程测试明确采用原始经纬度近似WGS84，再投影到EPSG:3826计算台湾区域距离；这是假设，不是坐标基准已证实或正式地理精度通过。

卫星实际数组为`Longitude, Latitude, AOD550, QCAll`，534×871。AOD550有效范围[-0.05,5]，QCAll明确0高、1中、2低、3无检索；使用0/1及有效范围和坐标有效性筛选。AOD是无量纲550nm光学厚度，不是PM2.5浓度。QCAll不是独立云分类字段，不能把全部无检索像元称为云。158文件覆盖9个日期；逐文件QA通过像元比例0～39.98%，平均约16.70%，见实际`satellite_qa.csv`。

按论文描述的27.5km站点邻域对真实有效像元求均值，再在小时槽内对实际取得的扫描求均值；保留扫描数，不补齐缺少的半小时扫描。当前匹配表含2,885条训练/验证站样本，63站、9日，其中训练2,245条、验证640条；没有写入14个最终测试站的标签。

**时间阻塞仍存在：** `DT`没有时区；AHI文件没有可直接使用的时间坐标，文件名时标及地面小时均值窗口没有在这两个表内直接定义。本轮仅显式假设地面UTC+8、AHI文件名UTC，并以小时槽匹配，生成探索性工程结果。论文关于台湾当地时间与UTC的讨论不能单独证明作者CSV是否已经转换过。没有用相关性反推时区，没有把本次匹配宣称为已核实的正式小时窗口匹配。

### 26.5 冻结划分、实际算法与防泄漏

在查看误差前按站点坐标排序、种子11排列，固定50训练站、13验证站、14最终测试站；名单在`golem_dag_smoke/station_split.csv`。任何年份的测试站标签均不进入拟合、残差、特征匹配评价表或Objective。此处允许同小时训练站观测预测留出站，不声称是未来时间预测。

- 实验二预定四季日期1月15、4月15、7月15、10月15日当地12时。只用训练站凸包内验证目标，不外推；IDW复用既有`IDWPlugin`，Ordinary Kriging调用既有上游PyKrige。
- 线性回归为同小时训练站AOD→PM2.5，重新拟合系数，不搬用论文系数。
- GWR调用官方PySAL `mgwr==2.2.1`，BSD-3-Clause；固定自适应bisquare邻居数25（训练站不足时为n−1），没有用验证或测试选择带宽。它是成熟实现的小样本基线，**不是论文完整动态带宽/十折站点CV复现**。[官方接口](https://mgwr.readthedocs.io/en/latest/generated/mgwr.gwr.GWR.html)
- UK调用PyKrige UniversalKriging的specified drift，以真实QA通过AOD作外部漂移，估计满足漂移约束的空间权重；不是把AOD直接相加。
- RF＋残差：官方sklearn RF，50树、深度6、最小叶3、种子11；只用训练站AOD拟合。按训练站5折交叉拟合得到未见该站标签的预测，残差为观测−OOF预测，用现有IDW对这些残差进行空间估计，再加完整训练站RF预测。与UK的协方差/漂移约束机制实质不同。
- 各方法统一非负投影并保留修正前预测；当前32次运行修正数均为0。没有根据结果修改算法参数。

质量门、预测和回读均实际执行。NetCDF是留出位置的**点集预测**，不是宣称已经恢复出独立真值验证的高分辨率连续格网，也不是任何环境模型的原生输入。

### 26.6 真实共同验证结果

以下主结果来自`output/taiwan_2017_preparation_20261019/golem_dag_smoke/comparison.csv`；仅用于工程跑通，不具有正式精度结论。四种多源方法按已取得文件中的最早六个小时槽（2017-01-02 00～05 UTC假设时标）运行，未按误差选日期。

|实验|方法|共同样本|RMSE|MAE|Bias|R²|该次方法合计秒|
|---|---|---:|---:|---:|---:|---:|---:|
|二|IDW|35|7.1892|5.4013|−1.0311|0.8656|5.309|
|二|Ordinary Kriging|35|8.8007|6.5688|−2.0461|0.7986|1.115|
|三|线性回归|52|21.0709|17.7260|+3.8233|−0.1138|1.578|
|三|GWR|52|14.1188|12.3863|+0.0749|0.4999|11.311|
|三|外部漂移UK|52|9.0081|7.5377|+1.3680|0.7964|2.026|
|三|RF＋OOF残差IDW|52|15.8238|12.2048|+3.2328|0.3719|6.232|

浓度误差单位μg/m³；R²为`1−SSE/SST`，不是相关系数平方。耗时包含该方法当前图执行、质量检查和输出流程，依赖当前机器及并发下载，不能据此宣称某算法普遍更快。

实验二：9个验证站、4时刻，共35个合格目标；原本13站×4时刻=52，实际非缺失地面值51，凸包内35，所以相对有效地面观测覆盖**35/51=68.63%**。实验三：9个验证站、6时刻，原本78个有效地面目标，AOD匹配76个，凸包内共同52个，实际覆盖**52/78=66.67%**。两个/四个方法在各自合格共同队列上均100%预测，失败率0，但**不能把这个100%当作全部验证站覆盖率**。排除依据只涉及输入质量、AOD存在和几何范围，未依据目标误差选择。

32次图计算、32份NetCDF回读通过；每种方法逐样本及逐站指标、执行耗时、质量报告和完整依赖边保留。所有输出的综合质量状态WARN，原因包括未完成来源校验和时空证据不足，不能记作正式数据质量PASS。UK在这个临时小样本数值更低，**不据此证明全年优于其他方法**。初次前缀提取的另一组29条多源样本及结果保留在输出根目录，未拿它与52条队列混排或冒充精度提升。

### 26.7 GOLEM接口、测试及复现

没有修改第三方GOLEM、搜索核心或冻结34候选池；池SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`。新增的是特定台湾任务输入适配，不把GWR等任务组件悄悄增加成冻结池无条件搜索节点。

`tools/run_taiwan_preparation.py`复用现有`run_three_source_ready_smoke.golem_graph`构建真实OptGraph，再由现有`NumericalDAGExecutor.run_golem()`按父边执行。UK图含训练、查询、UK预测及真实质量节点；RF图另有训练状态fit→apply依赖和训练残差校正。查询端不含验证PM2.5标签；最终评价在图输出后读取validation，生成`objective.json`，包含真实RMSE/MAE/Bias/R²、质量状态和成本；不能把这些调用当作已经执行A/B/C搜索。

当前多源基本配置只有线性、GWR、UK、RF残差四种，直接枚举即可；本轮不凭这四图宣称进化搜索有创新优势。尚未取得正式可靠时空匹配前，也不扩大图或参数凑空间。

定向测试：**14通过、0失败、0跳过、0错误**，见`targeted_tests_v3.xml`。验证真实GOLEM边执行、验证/测试标签不得进入Objective拟合、查询不接受目标值、时间不匹配拒绝、RF训练站交叉拟合及已有DAG契约。新增测试初次发现工具模块导入路径问题，已修复并复测；没有用随机重试掩盖故障，没有运行全量回归。合成测试夹具只验证契约，不作为环境精度证据。

UCI保留原结果；从已保存预测重新计算32行指标，与原`metrics.csv`逐项一致，没有重训或重跑基线，见`uci_reproducibility_check.json`。

主要文件：

- 数据及断点：`data/taiwan_2017_zenodo4420335/`。
- 清理、来源、年度覆盖、UCI核对、下载失败及验收JSON：`output/taiwan_2017_preparation_20261019/`。
- 当前实际图结果：其下`golem_dag_smoke/`的`analysis_plan.json`、`station_split.csv`、`matched_train_validation_only.csv`、`predictions.csv`、`executions.csv`、`comparison.csv`、`station_metrics.csv`、`coverage_denominators.csv`及逐日期/算法NC、Objective、质量与轨迹。
- 下载/选择性提取：`tools/prepare_taiwan_archive.py`、`resume_taiwan_ranges.py`、`extract_taiwan_prefix.py`。数据读取后才运行实验，不把下载器退出成功等同于官方MD5通过。
- 实验与回读：`tools/run_taiwan_preparation.py`、`audit_taiwan_outputs.py`；可选依赖`tools/requirements_taiwan_preparation.txt`；定向测试`tests/test_taiwan_preparation.py`。

复现现有前缀小样本（必须使用全新输出目录；仍为探索性假设，不是正式数据）：

```text
.venv/Scripts/python.exe tools/run_taiwan_preparation.py --output output/taiwan_reproduction_NEW
.venv/Scripts/python.exe -m pytest tests/test_taiwan_preparation.py tests/test_numerical_dag.py -q -p no:cacheprovider --basetemp=output/pytest_taiwan_NEW
```

### 26.8 最终状态与剩余条件

|项目|状态|证据与边界|
|---|---|---|
|旧项目下载清理|已执行|114份物理删除，逐份来源与SHA核对；越界数据未动|
|实验一|保留并核对通过|32行指标重算一致，不重复运行|
|台湾数据完整取得|未通过|网络分段失败，官方包MD5尚未验证|
|实验二实际基线|已执行，WARN|35个共同留站目标，2方法8次图执行；时空来源限制仍在|
|实验三四方法|已执行，WARN|52个共同目标，4方法24次图执行；不是全年论文复现|
|真实图数值Objective|小样本可执行|32次真实图、质量门、回读；正式数据契约未通过|
|最终测试评价|未执行|14站封存，未用来挑算法|
|正式A/B/C|暂停|来源校验及时间/坐标证据未完成；当前4图宜枚举|

后续必须完成官方整包校验、补时间窗口及datum依据，再冻结完整匹配数据及独立测试方案。当前不需要冒用账户或编造授权，也不通过降低验收标准获得READY。**本轮有真实清理及基线产物，但不能声称用户要求的全部正式实验准备已经完成。**

## 27. 台湾固定路线最终执行与完整数据验收（2026-10-19）

### 27.1 本轮范围和最终结论

继续UCI时间恢复、台湾2017站点空间重建、台湾地面PM2.5＋AHI AOD三层路线；不切换美国、不引入ERA5、不扩充冻结34候选、不启动正式A/B/C。第26章的失败进度和探索性结果全部保留，本章补充最新状态，不覆盖历史证据。

**已解除整包下载阻塞：官方13,435,034,939字节文件全部下载，官方MD5完全一致；台湾数据已完整选择性提取，全部7231份卫星数组已实际读取。** UCI旧结果和32次台湾真实DAG输出已独立回读核对。发现并修复一处非负质量门输出端口错误及一处测试导入冲突，全量回归最终1257通过、4跳过、0失败、0错误。

**正式数据冻结仍不通过。** 完整包、作者README、论文和全部卫星头信息仍不能证实作者CSV的DT时区/转换与小时窗口、定制AHI文件名的时钟定义、2017作者站表的坐标基准。不得把当前工程假设升级为正式科研事实；因此没有重新生成所谓正式匹配数据、没有重跑未获可靠时空契约支持的正式实验、没有开启最终测试标签评价或正式搜索。这是完成可独立验收工作后保留的来源证据阻塞，不是程序运行失败或需要新的研究方向。

### 27.2 官方完整下载、校验与选择性提取

官方来源：[Zenodo记录4420335](https://zenodo.org/records/4420335)，[官方API](https://zenodo.org/api/records/4420335)，论文[10.1029/2020EA001599](https://doi.org/10.1029/2020EA001599)。记录发布日期2021-01-05，许可CC-BY-4.0；使用须注明作者、数据集和修改/处理过程。配套数据是作者整理的观测和检索产品，不能称为未经处理的仪器原始读数。

|校验项目|实际结果|
|---|---|
|完整文件|`data/taiwan_2017_zenodo4420335/aod_pm25_data.tar.gz`，已设只读|
|实际文件大小|13,435,034,939字节，与官方一致|
|官方与实算MD5|`d7329012a175b3ad52012d1f3d6dd50f`，完全一致|
|实算SHA256|`3131b19497ed5eff69f372c21659810293cda9dbe078f772aed32b196c472dd5`|
|整包校验完成时间|2026-10-19 08:48:23 UTC，即北京时间16:48:23；后续又独立复算整包MD5通过|
|完整归档成员扫描|18,901项，逐项记录名称、大小、类型及是否提取|
|台湾文件提取|7,233份，共12,262,791,338字节：2份地面/站表CSV＋7,231份AHI NetCDF|
|必要作者说明|1份625字节`zhang_data/readme.txt`，另行保存；未解压美国大型数据|
|原有前缀文件|160份逐一SHA256核对未变；原manifest保留，不用其代替整包MD5|

下载没有重复获取已接受范围。沿用280,002,560字节前缀及16MiB分段，最终785段全部完成。实际尝试过API入口、官方网页直链、低并发requests、curl HTTP Range；每段最多2次尝试，各阶段有墙钟和失败上限。故障证据包括连接超时、DNS解析失败和个别分段传输超时，均保留在`range_progress.json`。历史46条失败范围记录不等于当前仍缺46段：最后仅对未完成段执行低并发修复，当前缺段为0。

经系统DNS和显式DNS查询核对官方地址后，使用curl单请求`--resolve zenodo.org:443:188.185.45.63`避免本机间歇DNS失败；域名、HTTPS证书校验和官方服务器均未更换，未修改系统DNS、安全策略，未使用镜像、账号或认证绕过。此地址是本轮实测记录，今后应重新解析，不应视为永久下载配置。

关键证据均在原数据根目录：`official_record.json`、`range_progress.json`、`download_verification.json`、`full_archive_member_inventory.csv`、`verified_taiwan_source_manifest.json`、`archive_documentation/zhang_data/readme.txt`。新完整manifest含实际提取路径、成员大小、SHA256、整包校验状态；旧`source_manifest.json`及失败记录没有覆盖。HTTP响应长度、逻辑文件大小和成员SHA没有被当作官方整包MD5的替代证据。

### 27.3 全年数据实际读取与质量结果

地面CSV已在第26章实际检查，本轮核对原文件指纹后直接复用：77站、673,152条记录、24,503条PM2.5缺失、站点－时间重复0、负浓度0；列为`stn_eng, DT, PM10, PM2.5`。站表为`SiteEngName, Lon, Lat`，没有独立仪器QA、时区、坐标基准字段。缺失行和不存在的小时记录不是同一个概念，年度覆盖详表继续保留。

新增完整卫星核验不是目录存在性检查：7,231份NetCDF的`AOD550`、`QCAll`、`Longitude`、`Latitude`数组均实际读取，逐份检查形状、有限坐标、有效范围、质量码、填充值和尺度属性，输出`output/taiwan_final_20261019/header_audit_verified_archive.json`。

|项目|全年实际结果|
|---|---|
|文件名日期范围|20170101—20171231，共365个日期标签；日期标签不替代已证实的观测时刻|
|数组形状|全部534×871，四个变量形状一致|
|坐标数值|全部在合法经纬度范围；首网格经度约109.1468—130.7549、纬度17.5599—29.3370；数值合法不等于datum已确认|
|QCAll编码|实际仅0、1、2、3；属性明确0高、1中、2低、3无检索|
|接受像元|QC为0或1，原始有效范围[-0.05,5]内，非填充值且经纬度有效；依属性处理scale/offset|
|高/中质量有效比例|逐文件平均14.3820%；文件范围0—56.2266%；126幅无合格像元|
|通过QA的AOD实际范围|约-0.0500000007—4.99979353；产品允许的小负检索值不冒充负PM2.5浓度，也不无依据改写|
|时间/坐标定义元数据|7,231份中，时钟变量0、时钟属性0、明确全局datum属性0|

有效比例分母是整幅网格像元，不是云覆盖率、台湾陆地区域覆盖率或站点有效覆盖率。全年有365个文件名日期不等于已取得100个共同合格评价日期；未经可靠时钟定义，未生成正式全年三源/二源共同样本统计。AOD是550nm气溶胶光学厚度，不能当作PM2.5浓度或独立地面真值。

### 27.4 时间、坐标证据及客观剩余阻塞

本轮查证的第一手材料与结论保存在`output/taiwan_final_20261019/primary_evidence/`、`contract_evidence.json`、`local_source_metadata.json`及完整头审计。作者完整README确认台湾2017地面PM2.5和AHI AOD用途，但没有给出时间窗口、时区或datum；完整成员表没有作者处理代码。未找到外部处理仓库不意味着不存在，不能编造其实现。

|问题|已证实内容|仍不能证实的内容与影响|
|---|---|---|
|地面变量、单位|论文§3.1/§4.4为台湾EPA小时地面PM2.5，μg/m³|CSV是作者整理表，没有附带仪器QA；AQI不是仪器QA|
|地面DT|列可解析为小时记录；论文明确UTC与台湾当地时间关系|没有直接说明导出DT是否已转换、属于哪个时区，小时标签对应窗口的左右端点未说明；不能冻结正式时间匹配|
|AHI时间|论文说明半小时高/中质量AOD汇成小时；文件名有日期和时分|定制文件名的时区、原始扫描时刻来源及缺一幅时的小时规则未明确；文件内无time坐标|
|站点坐标|现行台湾官方AQX_P_07站点说明明确TWD97Lon/TWD97Lat|不能由现行字段或数值相似推断作者2017重命名Lon/Lat的转换历史、datum/epoch|
|空间距离计算|当前代码使用米制距离；Kriging/GWR/凸包为假设4326→3826，现有IDW为WGS84局部AEQD|两种投影并不完全相同，且输入datum未经证实；正式比较前须在来源确认后统一，不能宣称地理精度通过|

第一手链接：论文[DOI](https://doi.org/10.1029/2020EA001599)，[NOAA公开论文PDF](https://repository.library.noaa.gov/view/noaa/41813/noaa_41813_DS1.pdf)，[台湾官方站点字段说明](https://data.moenv.gov.tw/dataset/detail/AQX_P_07)，[政府开放资料目录](https://data.gov.tw/dataset/6075)。出版社页面403已保留，改读合法NOAA公开副本，未绕过限制。论文§4.4的UTC－当地时间示例只证实两种时钟的关系，不证明自定义CSV和文件名使用哪一种。不同市政监测产品的小时定义也没有移植到作者全国站表。

本轮没有凭PM2.5与AOD相关性、白昼分布或拟合RMSE反推时区，没有通过选择较好指标的时间偏移解决证据问题。当前正式放行条件为：整包完整性True；地面DT及窗口False；AHI时钟False；作者站表datum False。False表示尚无足够证据，不代表已证明数据本身错误。所缺条件是**作者导出/匹配说明或可核验的同源处理代码和原始坐标元数据**，不是新算法、GPU、CDS密钥或新的研究区域。

### 27.5 保留并独立复核的真实实验

本轮未重复训练、未重跑已完成32次算法图。原有160份输入现在也与整包对应成员逐字节核对一致；从已保存预测重算指标、回读32个NetCDF，数值、单位、坐标与CSV一致。完整性阻塞已解除，但时间/datum假设尚未被证实，以下结果依然为探索性WARN，不是正式论文复现或最终测试成绩。

UCI实验一复用原有人工遮蔽、划分和32行基线指标；从已保存预测重算核对一致，文件指纹未变化。没有为了本轮报告重复运行。

台湾固定种子11，50训练站、13验证站、14最终测试站；每个时刻允许用训练站同期观测预测留出验证站，不能称为未来预报。训练/拟合/残差使用训练站，查询不带验证/测试标签；本轮无最终测试评价，也不以测试站挑选模型。旧全量CSV的来源审计与测试标签参与拟合不是同一概念。

|任务/方法|共同验证N|RMSE|MAE|Bias|R²|原运行算法耗时合计/s|
|---|---:|---:|---:|---:|---:|---:|
|实验二 IDW|35|7.1892|5.4013|-1.0311|0.8656|5.309|
|实验二 Ordinary Kriging|35|8.8007|6.5688|-2.0461|0.7986|1.115|
|实验三 线性回归|52|21.0709|17.7260|3.8233|-0.1138|1.578|
|实验三 GWR|52|14.1188|12.3863|0.0749|0.4999|11.311|
|实验三 外部漂移UK|52|9.0081|7.5377|1.3680|0.7964|2.026|
|实验三 RF＋OOF残差IDW|52|15.8238|12.2048|3.2328|0.3719|6.232|

R²按1−SSE/SST计算，不是相关系数平方。耗时来自原运行记录，不能据此宣称普遍计算成本优势。实验二为4个预设季节时刻、9个实际可评价验证站，2方法8图；实验三为2017-01-02假设UTC的00—05时6时段、9个验证站，4方法24图。小样本、日期集中、时钟/坐标未证实，无法推出全年或跨区域优劣。

覆盖率必须同时报告分母：实验二52个潜在站时、51个地面有效、35个在训练站有效空间域，整体覆盖35/51＝68.63%；实验三78个潜在/地面有效站时、76个AOD匹配、52个在有效空间域，整体覆盖52/78＝66.67%。各方法只在同一预先合格队列进行主比较，合格队列内预测覆盖100%、执行失败0、非负投影计数0；不把被排除样本悄悄算成成功。`sample_qualification.csv`及`coverage_denominators.csv`保留排除原因和分母。

所有32份NetCDF是`sample`维的**留出站点预测集合**，不是覆盖全域的高分辨率网格。网格覆盖状态为`NOT_APPLICABLE_POINT_OUTPUT`，不能填写伪造的100%网格覆盖。域外/缺失样本保留资格标记，不补值、不无条件外推。

线性为AOD→PM2.5回归；GWR调用官方mgwr 2.2.1，固定自适应25邻居，非论文完整带宽搜索/十折复现；UK调用PyKrige，AOD进入specified drift；RF使用50树、深度6、最小叶3、种子11，按训练站5折交叉拟合获得OOF残差，再由训练站IDW残差校正。没有验证标签拟合、没有把AOD改名为浓度。不得用上述小样本排名证明一种路线在正式独立测试中获胜。

### 27.6 可复现故障、最小修复与回归

1. **质量门端口Bug，已复现并修复。** 原`quality`节点对`max(prediction,0)`检查后返回未修正原值，后续可能交付负值。新增确定性测试以-3输入复现失败，再令质量节点返回检查后的预测，并单列`raw_prediction`保留投影前值；指标和输出消费正确端口。修前源码备份SHA与旧实验记录一致，位于`output/taiwan_final_20261019/source_before_fixes/run_taiwan_preparation.py`。旧32次结果没有负值，故其数值不受修复影响，不为制造新结果重复运行。
2. **全量收集导入冲突，已修复。** 外部环境的`examples`常规包遮蔽本地同名命名空间，首次全量出现5项收集错误。新增本地`examples/__init__.py`明确包优先级，未修改算法、删除测试或屏蔽异常。
3. **验收布尔契约。** 正式放行须显式True，不能让字符串`UNVERIFIED`或`PASS`的Python真值触发放行。逐个缺证阻止及非布尔拒绝测试通过；这些合成夹具只证明程序逻辑，不证明数据来源。

|测试记录|实际结果|
|---|---|
|`full_regression_initial.xml`|5收集错误，原始失败保留|
|`quality_port_bug_reproduction.xml`|1失败，证明负值端口问题可复现|
|`targeted_quality_fixed.xml`|21通过、0失败|
|`targeted_final_archive.xml`|完整数据核验后再次21通过、0失败|
|`full_regression_final.xml`|1261项：1257通过、4跳过、0失败、0错误；260.236秒|

4项跳过是已有外部案例产物缺2项、可选ESMF缺1项、官方CMAQ profile/可选读取夹具缺1项；未新增跳过，不为消除跳过引入无关大型下载。警告和失败XML均保留。完整回归在核心修复稳定后执行；此后只增加独立下载/完整源数据审计辅助程序和报告，未再改动核心处理算法。最终补充定向测试，不重复消耗全量回归资源。

### 27.7 图、Objective及保护情况

沿用既有`OptGraph`真实父边、命名端口和`NumericalDAGExecutor.run_golem`，RF fit状态实际传入预测；训练/查询时间、站点角色、单位、覆盖域和双层质量门均参与执行。32次原真实图及数值Objective可回读，不是固定分数或名称占位。

当前多源基本配置仅线性/GWR/UK/RF残差4类，直接枚举即可。**不建议在此固定小空间扩大A/B/C；未运行正式搜索，不声称语义Mutation已提高预测精度。** 只有来源合同确认、充分共同样本、独立方案冻结，以及确有必要的非等价可搜索空间之后，才可能考虑正式优化比较。

冻结34候选清单SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`。未修改GOLEM核心、候选池、历史V1—V6实验、用户原始京津冀/气象/WRF数据。**本轮删除0文件、释放0字节**；第26章114份742,468,142字节的历史清理仅保留记录，本轮没有追加清理。成功/失败分段、来源、日志、旧结果和原manifest均保留。最终审计时项目所在磁盘约175.32GB可用；没有空间不足阻塞。

### 27.8 路径、复现命令与最终验收表

原始整包、台湾文件、成员清单及说明：`data/taiwan_2017_zenodo4420335/`，台湾文件仍位于既有`provisional_members/zhang_data/`以保持路径兼容，文件夹名称不是验收状态。完整性由新的校验与manifest决定；正式时空契约仍单独未通过。

实际实验：`output/taiwan_2017_preparation_20261019/golem_dag_smoke/`，含固定站点划分、逐样本预测、共同队列、汇总、NetCDF、质量JSON、图轨迹及数值Objective。新增验收证据：`output/taiwan_final_20261019/`，核心索引`delivery_verification.json`，另有来源证据、全源头/数组审计、网络失败和测试XML。没有创建新的分散Markdown交付报告。

新增/修改代码仅围绕本任务：`tools/resume_taiwan_ranges.py`、`extract_verified_taiwan.py`、`scan_taiwan_source_headers.py`、`capture_taiwan_primary_evidence.py`、`verify_taiwan_final_delivery.py`、`run_taiwan_preparation.py`、`examples/__init__.py`、`tests/test_taiwan_final_delivery.py`及既有`tests/test_taiwan_preparation.py`。成熟计算仍为现有Registry IDW、PyKrige、scikit-learn和官方mgwr；未自写第三方数学核心。

现有Windows Python3.12.14主环境`.venv`；PyKrige1.7.3、mgwr2.2.1；GOLEM桥接复用`.venv-golem/Lib/site-packages`中的现成官方包，没有新建大型环境或训练网络。以下在项目根目录执行：

```text
.venv/Scripts/python.exe tools/verify_taiwan_final_delivery.py
.venv/Scripts/python.exe -m pytest tests/test_taiwan_preparation.py tests/test_taiwan_final_delivery.py tests/test_numerical_dag.py -q -p no:cacheprovider --basetemp=output/pytest_taiwan_recheck_NEW --junitxml=output/taiwan_recheck_NEW.xml
```

第一条只核对现有文件/指标/契约，不拟合、不读取最终测试标签用于评价。完整提取/头审计程序为保护结果拒绝覆盖既有清单，不应对已完成目录机械重复执行；迁移时在全新项目数据副本运行。第26章算法复现命令仍只允许明确标注的探索性运行，不能绕过本章正式来源门。

|验收项|最终状态|依据/限制|
|---|---|---|
|官方完整数据获取及MD5|PASS|实算大小、官方MD5完全一致，再次独立复算通过|
|台湾必要成员提取及追踪|PASS|7233份文件、SHA256、完整成员表；美国大文件未提取|
|全年卫星实际可读/QA核验|PASS（功能核验）|7231实际数组；有效比例14.38%，不代表精度或足够共同样本|
|地面单位与变量定义|PASS|论文和数据列支持；无仪器QA字段|
|地面DT、窗口及AHI文件名时钟|未证实，BLOCK正式冻结|完整README/全部头信息没有充分定义|
|作者2017站表datum|未证实，BLOCK正式冻结|现行官方TWD97不能替代作者表转换来源|
|实验一|保留并核对通过|32行指标、预测与指纹复核，无重复训练|
|实验二/三工程运行|已真实执行并回读，WARN|8＋24真实图；35/52验证目标，仍有时空假设|
|正式修正后实验二/三|未运行|缺可靠时空合同，不伪造正式输入与成绩|
|最终测试站评价|未运行，封存|不利用其标签选择模型或调参|
|全量回归|PASS，4项明确跳过|1257通过、0失败、0错误|
|正式A/B/C|未运行/不放行|来源合同未满足，且当前4配置宜穷举|
|本轮数据删除|0|历史数据、证据、配置与结果全部保护|

**最终判定：下载、选择性提取、全年实际读取、已有结果核对、可复现程序修复及回归均已完成；正式科研数据准备仍为PARTIAL/BLOCKED，不以工程成功代替来源证实。剩余唯一类别阻塞为作者数据导出与地理参考的权威元数据缺失（具体三字段见27.4），并非需要继续换数据或扩算法。本轮停止新增尝试，保留完整数据和全部成果，不提出新的方向性选择，不宣称正式实验条件全部通过。**

## 28. 数据契约闭环与敏感性审计（2026-10-19）

### 28.1 验收结论

**不能正式冻结数据。时间假设会改变算法排序，且已经在相同地面记录的配对评价中观察到第一名反转；正式精度和算法创新对比继续暂停。** 坐标候选在当前可调用的PROJ操作下不可数值区分，不能解释为作者datum已经确认或不存在定位误差。本轮没有选择RMSE最低的时钟解释作为真实定义。

已实际完成有上限的一手检索、16种条件的敏感性运行、配对补充审计、文件回读及定向测试。无数据下载、文件删除、新算法、正式GOLEM搜索或全量回归。第27章完整校验与原32次运行继续复用，未改写历史结论或旧结果。

### 28.2 数据契约证据表

检索上限事先限定为3批、每批4个定向查询，实际12个查询；随后核对相关官方页面和论文定位，不继续无限检索。查询、链接、简短原文、字段作用和判断理由保存于`output/taiwan_contract_sensitivity_20261019/evidence.json`。完整原包、作者README及全部7231份头信息审计沿用第27章，不重新下载。未找到作者处理仓库不等于不存在。

|适用字段|一手来源及原文/定位|实际能证实什么|不能转移或推断的部分|
|---|---|---|---|
|原始AHI标准文件时间|[JMA HSD指南v1.3](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/fig/HS_D_users_guide_en_v13.pdf)，§4/Table4：`time zone is UTC`、`Observation start time`；Table6含观测起止MJD|原始`HS_...`文件名是UTC观测时间线，原始头部有扫描起止时间与观测时间信息|作者`results_Taiwan_...nc`不是HSD文件名，未保留可核对的原始父文件名或time变量；不能从标准规则证明作者定制文件名时区，也无法恢复逐像元实际扫描时刻|
|原始卫星导航|同一JMA指南§3：`based on WGS84`|支持将原始卫星导航按WGS84解释的条件方案|不能证明作者输出的变换历史，更不能证明地面站表datum|
|NOAA AHI算法背景|[NOAA AHI演示材料](https://www.star.nesdis.noaa.gov/jpss/documents/meetings/2015/SJASTM/Session5-Advanced_Applications.pdf)：`The GOES-R ABI algorithm was used.`|NOAA展示过带UTC说明的AHI检索与算法背景|不是本作者归档的定制命名说明；ABI、GASP、SST等其他产品的规则不直接套用|
|地面DT、AHI小时匹配|[作者论文NOAA公开副本](https://repository.library.noaa.gov/view/noaa/41813/noaa_41813_DS1.pdf)，§4.4：`UTC (9:00–10:00 a.m. local time)`；该节描述半小时AOD形成小时估计|台湾当地时间与UTC关系、论文的小时处理思路|没有定义CSV导出DT是否已转换、标签在小时起点还是终点、定制AHI命名与真实扫描时间的映射；仍未证实|
|小时均值含义|[台湾空气品质标准](https://oaout.moenv.gov.tw/Law/LawContent.aspx?id=FL015351)，第2条：`小時平均值：指一小時內各測值之算術平均值。`|说明小时均值是小时内算术平均|不能由现行法规反推出作者2017 DT端点或导出时区|
|站表Lon/Lat候选|[官方AQX_P_07字段说明](https://data.moenv.gov.tw/dataset/detail/AQX_P_07)：`TWD97Lon(經度（TWD97）)、TWD97Lat(緯度（TWD97）)`|TWD97是有出处的候选，不是随意添加坐标偏移|现行站表不是作者2017表，不能证明其重命名Lon/Lat的datum和转换历史|
|作者配套说明|[官方Zenodo](https://zenodo.org/records/4420335)，已校验完整包中的`zhang_data/readme.txt`：`PM2.5 surface measurements at Taiwan sites for 2017.`|确认作者数据用途，保留完整官方MD5证据|说明未补充时区、小时端点或datum；完整成员表未见处理代码，不能伪造字段依据|

因此三个关键字段仍为**未证实**，没有因本轮拟合表现或坐标数值相似而转为PASS。JMA新增证据提高了原始卫星规则的可追溯性，但缺少“原始产品→作者定制文件”的来源链，不能完成这一步契约闭环。

### 28.3 明确的假设与执行设计

- 地面时钟：UTC（g0）或台湾当地UTC+8（g8）；AHI文件名时钟：UTC（s0）或台湾当地UTC+8（s8）。这4种组合是条件审计，不是对真实性作四选一认定。
- 地面小时标签：起点表示`[DT,DT+1h)`；终点表示`[DT-1h,DT)`。每个标签转换成对应UTC条件下的区间起点，再与假设为名义扫描瞬时点的AHI文件名时间匹配；卫星按左闭右开小时聚合，仅使用实际存在且QA合格的扫描。不虚构扫描时长，不将文件名时刻宣称为已恢复的真实像元扫描时间。
- 站点坐标：WGS84（EPSG:4326，既有条件解释及原始卫星导航依据）或TWD97经纬度（EPSG:3824，官方字段依据）。无作者GCJ-02/TWD67依据，未为凑组合加入。卫星输出导航暂按WGS84条件解释，保留来源链缺失警告。
- 2×2×2×2＝16种条件。站点角色完全沿用种子11的50训练、13验证、14最终测试站；划分文件SHA256为`58d97ce058b86e9460e3bed68b5047fc1e67b6a86e2d1bf7088b1fac045ba915`。测试站先按编号排除，再解析浓度和匹配，未用于拟合、选择假设或评价。
- 参数、算法数学定义保持原样：Registry IDW、PyKrige OK/UK、线性回归、官方mgwr固定25邻居、RF50树/深度6/最小叶3/种子11及训练站5折OOF残差IDW。UK只使用该条件下的AOD漂移，未新增回归变量或算法。
- 每个窗口六方法使用同一合格训练子集及同一验证队列；训练站必须≥15、验证站≥3，并限制在训练站凸包内。IDW/OK不把AOD作为预测变量，但为了六方法公平比较也使用共同QA队列。固定角色不代表每种假设都有50个可用训练站，QA造成的实际子集差异逐窗口保留。

初审固定2017-01-15、04-15、07-15、10-15四个文件名日期及01/05两个小时。真实执行后发现不同时间解释多数没有同一地面标签交集，无法公平判断敏感性；原结果完整保留，没有挑选误差较好的日期。随后增加一个**唯一的固定配对补充设计**：同四个日期的00、01、08、09文件名小时，事先记录相邻小时和8小时边界理由及4个按字段指定的配对，不依据RMSE选择，不再继续扩展。所有16条件均重新做时间/空间匹配。`raw_DT＋station_id`作为跨条件配对键，配对真实浓度须逐值一致。

### 28.4 敏感性结果表

以下为配对补充设计下，各条件内部六方法共同合格队列的RMSE（μg/m³）。每行均实际检查了4326与3824两种坐标解释；二者输入数值完全一致，结果相同，所以合并展示，完整96行指标保存在CSV。**不同行的N和地面原标签可能不同，不可直接据表挑选“正确”时区。**

|地面/卫星时钟、地面标签|共同N/有效地面分母（潜在208）|AOD匹配验证N|IDW|OK|线性|GWR|UK|RF残差|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|g0/s0 起点|62/198|90|6.3194|7.1280|9.6995|7.5977|7.1372|7.9209|
|g0/s0 终点|62/197|90|6.7693|8.3450|10.2683|7.5703|7.9440|7.8809|
|g0/s8 起点|64/200|91|6.5949|6.2048|7.5633|6.4820|6.3866|6.9625|
|g0/s8 终点|64/199|92|6.6077|6.8660|7.7930|6.6349|7.0546|7.1074|
|g8/s0 起点|62/202|92|8.6445|8.7468|11.9307|9.2460|8.5838|8.7171|
|g8/s0 终点|62/203|93|9.1676|9.7302|11.3650|9.1876|8.9370|8.9381|
|g8/s8 起点|62/198|90|6.3194|7.1280|9.6995|7.5977|7.1372|7.9209|
|g8/s8 终点|62/197|90|6.7693|8.3450|10.2683|7.5703|7.9440|7.8809|

MAE、Bias、R²、每方法排名、各分母、覆盖率、失败率、耗时、投影修正数完整见`comparison_with_coverage.csv`，并由单独审计程序从逐样本预测重新计算核对。R²为1−SSE/SST，不是相关系数平方。

**在同一地面真值上的配对发现：**

1. 只改地面DT起点/终点，固定g0/s0/4326：共同29条。起点条件UK RMSE7.0956、GWR8.2228；终点条件UK7.0634、GWR6.9956，二者顺序反转，IDW均第一。不能据此判断哪个端点真实。
2. 只改AHI时钟，固定g0/起点/4326：共同28条。OK/UK从6.9895/7.0832变为6.8039/6.7088，二者先后变化，IDW仍第一。
3. 只改地面时钟，固定s0/起点/4326：共同27条且真实值逐条一致。g0条件IDW7.0309、OK7.0624，IDW第一；g8条件IDW6.8815、OK6.6707，OK第一。已经观察到首名反转，而非仅比较不同样本表格。
4. 只改站表4326/3824，固定g0/s0/起点：共同62条，全部六方法数值相同。实际操作为PROJ `TWD97 to WGS 84 (1)`，`proj=noop`，标称精度1米；最大坐标数值变化0。此结果是当前转换不可辨识，不是证明作者datum正确，更不是严格证明地面定位无误差。没有编造坐标扰动来强行制造差别。

只统计4326下的8个时间条件，28种两两配对中24种有≥2条共同标签，配对N为13—62；22种完整排序变化，11种第一名变化。16条件包括坐标重复的完整120对全部保留：104对有共同样本，其中88对排序变化；16对不足共同样本。**这些配对彼此高度相关，坐标条件还重复，不能当作独立重复实验或“变化概率”，也未进行显著性检验。** 小差异的名次变化不等于具有实际优势；以上预先指定例子只是证明该数据契约不确定性足以影响结论。

所有16条件的共同地面标签交集为0，故不存在一个覆盖全部条件的统一排名队列；不能把不同UTC/当地解释的日期标签直接配对。本章按共有原始地面标签逐对计算，不用伪造全条件共同样本。实际训练站子集受AOD QA影响，配对差异还包含训练覆盖变化，不宣称是纯时钟偏移的因果效应。

### 28.5 覆盖、失败、质量门和实际执行

配对补充每条件有16个请求窗口、13个潜在验证站，共208个潜在站时。地面有效197—203、AOD匹配90—93、空间/输入门后六方法共同62—64；相对于有效地面的输出覆盖约30.54%—32.16%，而合格队列内六方法预测覆盖均100%。没有只报告后一个数字。

每条件8个窗口因合格训练或验证不足而不执行，16条件累计128次窗口级`INSUFFICIENT_INPUT`，不是算法成功，也不是768次算法调用中的数值失败。实际可评价点仅9个验证站，主要来自4月和7月；1月与10月请求窗口未达门槛，因时区假设跨日还涉及4月14日、7月14日标签。**不能称为已通过四季或全年代表性验证。** 各窗口站数、地面缺失、AOD匹配、凸包合格数与拒绝原因均保留；没有放宽门槛或用填充值补足。

初审：192次实际算法图执行、192次完全相同输入的数值结果复用、3132行逐方法预测、192份实际NetCDF回读；补充：384次实际算法图执行、384次同输入复用、6000行逐方法预测、384份NetCDF回读。两阶段共576次实际执行及576份结果文件；重复坐标解释的复用不冒充新的实际执行或独立证据。缓存只接受完整相同数值输入，轨迹保留原执行出处。

实际调用中算法失败0；不合格窗口另外记录，不能把执行失败率0解释为全部原始数据覆盖。六方法统一执行非负投影并保留投影前值：补充表共有4条投影修正记录（包括相同坐标结果的重复引用），来自RF残差路线；原始值、计数和修正后输出均保存。所有新输出经过既有质量门，状态仍为WARN（来源契约未证实）；输出有限、非负、单位一致，NetCDF重读数值与预测CSV、站点坐标一致。

这里仍是`sample`维留站预测，不是连续区域网格；网格覆盖记录`NOT_APPLICABLE_POINT_OUTPUT`。缺失/域外目标体现在匹配与覆盖记录，不生成假值。不把物理检查通过当作独立预测精度证明。

### 28.6 发现的审计日志问题与定向测试

新增审计工具的第一次核对发现：空匹配表的缓存指纹没有包含请求窗口，多个空窗口会复用另一窗口的拒绝时间标签。确定性复现后，缓存键加入请求时间；未修改算法公式、GOLEM或原有质量门。

仅修复已生成日志的空窗口时间标签：初审24处、补充72处。原`executions.csv`保留，修正写入`executions_window_labels_corrected.csv`，逐行旧/新标签及原因写入`empty_window_label_repair.json`。只允许改动没有method、没有数值预测的`INSUFFICIENT_INPUT`记录，并与已保存`coverage.csv`窗口顺序、数量核对；预测、指标、NetCDF及真实算法执行时间未改，也未为此重跑算法。

最终定向测试**19通过、0失败**，证据`output/taiwan_sensitivity_cache_fixed_20261019.xml`；包括四种时间窗口转换、未知约定拒绝、不同空窗口不可复用同一拒绝标签、已有训练/查询隔离、RF交叉拟合、真实图执行、非负输出与正式放行证据门。未重跑上一轮已通过的全量回归。原32份输出SHA256和冻结34候选池SHA256再次核对未变；没有数据删除或重下载。

### 28.7 文件和复现

本轮只增加有针对性的审计程序及测试，不改现有算法数学定义：

- `tools/run_taiwan_contract_sensitivity.py`：调用现有实际DAG、质量门、六方法，生成条件匹配表及预测。
- `tools/audit_taiwan_contract_sensitivity.py`：逐样本重算、共同标签两两配对、独立NetCDF回读、旧结果/候选池保护核对。
- `tools/repair_taiwan_rejection_labels.py`：只修正本轮空窗口日志标签，保留原日志，不适用于原始数据。
- `tests/test_taiwan_contract_sensitivity.py`：确定性契约测试。

初审目录：`output/taiwan_contract_sensitivity_20261019/`；配对补充目录：`output/taiwan_contract_sensitivity_paired_20261019/`。各自包含`analysis_plan.json`、`station_split.csv`、每条件`matched_non_test.csv`、`predictions.csv`、原始和修正执行日志、`coverage.csv`、`comparison.csv`、`comparison_with_coverage.csv`、`paired_comparison.csv`、`pairwise_intersection_counts.csv`、`paired_rank_changes.csv`、`coordinate_operations.json`和`verification.json`。逐窗口目录含实际NetCDF、质量JSON、Objective和DAG依赖轨迹；补充目录另有按字段固定的`paired_analysis_plan.json`。第一手证据及12条查询记录在初审目录的`evidence.json`。未新增分散Markdown报告。

复现命令在项目根目录执行，须使用全新输出目录/项目副本以保护本轮结果（程序拒绝覆盖现有审计目录）：

```text
.venv/Scripts/python.exe tools/run_taiwan_contract_sensitivity.py
.venv/Scripts/python.exe tools/run_taiwan_contract_sensitivity.py --paired
.venv/Scripts/python.exe tools/audit_taiwan_contract_sensitivity.py
.venv/Scripts/python.exe tools/audit_taiwan_contract_sensitivity.py --paired
.venv/Scripts/python.exe -m pytest tests/test_taiwan_contract_sensitivity.py tests/test_taiwan_preparation.py tests/test_taiwan_final_delivery.py -q -p no:cacheprovider --basetemp=output/pytest_contract_NEW
```

当前已有结果无需重算，直接运行两个`audit_...`命令可重新核对。修复后的生成器不再需要空窗口日志修补步骤；旧原始错误日志和修补证据为历史保留。两阶段使用同一现有Python主环境及既有GOLEM桥接，无新依赖安装，无参数搜索，无最终测试标签评价。

### 28.8 对四个验收问题的明确回答

1. **能否正式冻结？不能。** 官方整包完整性通过，但地面DT时区/窗口、AHI定制时间及作者站表datum的来源链仍未闭环。本轮没有把假设选成事实，也未执行“正式冻结后全年基线”分支。
2. **假设是否改变排名？时间会。** 相同地面记录上的两两比较已出现排序和第一名变化。空间两候选在当前PROJ操作下相同，属于无法辨识，不能宣称已证明真实坐标不敏感。
3. **哪些可作正式依据？** 官方整包校验、源码/执行/回读、覆盖统计与数据契约敏感性发现可作为可复查的工程及局限证据；UCI原实验不受本轮台湾问题影响。台湾六方法的预测成绩只属于明确假设、小样本、相同验证站条件下的探索性结果，不能作为严格论文复现、正式全年排名或最终测试精度结论。
4. **能否开始算法创新对比？不能。** 暂停正式精度及GOLEM A/B/C；必须取得作者或可核验同源处理证据，明确DT和定制AHI时间映射、小时标签以及2017站表datum后，再固定合同与正式共同样本。当前只有少量可枚举路线也是独立限制，不通过增加无意义节点解决。

**本轮完成的是有实际运行和配对证据的数据可靠性审计，不是正式数据合同PASS。没有要求用户重新选方向，也没有因证据缺失停留在计划阶段；正式创新实验的暂停结论来自实际敏感性结果。**

## 29. 冻结工作契约、全年实验与小空间创新验收（2026-10-19）

### 29.1 实际完成与结论边界

本轮已按用户明确规定冻结项目工作约定，完成全年原始数据匹配及第二次完整复算、实验二和三的全年验证、验证集配置选择、选定配置的全年独立测试、当前四种多源图的穷举，以及五种固定种子的缓存数值目标顺序回放。没有重新选数据、下载、删文件、扩算法池、修改第三方GOLEM或重跑无关全量回归。

**工作契约可以冻结，不代表作者历史定义已被证实。** 本轮执行依据从第28章的“等待历史定义闭环”变为用户指定的可版本化工作规则；原作者DT、AHI定制文件名时区及站表datum仍标记未证实，没有以较好RMSE替它们背书。

**基线完整执行，但GOLEM搜索创新未被证明。** 当前本任务已有适配且已真实执行的多源配置为4种，穷举足够；没有人为制造新图，也没有用缓存顺序回放冒充A/B/C Mutation实验。此结论针对当前已验证搜索空间，不声称证明全34候选库的所有潜在组合只有4种。

### 29.2 冻结契约与已知/假设的区分

配置：`configs/taiwan_2017_working_v1.json`，版本`taiwan-2017-working-v1.0.0`。冻结副本：`output/taiwan_annual_working_v1/contract_frozen.json`。SHA256：`43e2e96221cd6e34adc06149331a97fd219bb386134f9e82ab283a48936ef9c4`，在本轮模型比较前保存；后续匹配、运行、选择、测试及输出均检查同一契约指纹。

|项目|本轮固定规则|证据性质|
|---|---|---|
|地面DT|无时区按Asia/Taipei解释；显式时区保留真实瞬时含义后转UTC，不重复减8小时；小时标签为[t,t+1h)|用户工作约定，非作者定义证实|
|地面变量|`stn_eng`字符串、`DT`时间戳、`PM2.5`数值；单位ug m-3；原始列、原始值、源文件和行号保留|单位由原论文支持；作者CSV本身无单位/仪器QA列|
|AHI时间|优先明确观测元数据；实际7231份均使用无可靠时区定义的定制文件名，因此依用户规定按Asia/Taipei作为名义时刻|全部标记`CUSTOM_FILENAME_WORKING_ASSUMPTION`，不是已恢复真实像元扫描时间|
|小时AOD|同小时内真实存在且QA通过的扫描先作27.5km站点邻域均值，再对可用扫描均值聚合；无扫描/无合格像元保留缺失，绝不跨小时借值|固定操作规则；不把少量瞬时样本均值称为完整小时连续观测|
|合成区间|若元数据提供区间，只接受能明确落入一个目标小时的区间；跨小时合成需另外说明时间模型，当前拒绝而不凭空拆分|实际文件没有可用合成区间；不宣称已实现任意合成产品时间分配|
|空间|原`SiteEngName/Lon/Lat`保留；经纬度采用EPSG:4326工作假设；所有距离计算含IDW统一EPSG:32651米制|作者datum未证实；不再混用旧IDW AEQD与其它方法3826距离|
|AOD质量|按存储表示处理有效范围、填充值；应用声明的scale/offset；QA只接受0/1；原范围允许小负AOD，不把它当负PM2.5|继承实际文件属性和既有规则，AOD单位为1而非浓度|
|地面质量|缺失、非有限、负值不进入拟合，不填目标；站点/区间重复拒绝；缺失行与缺失测值分别统计；无仪器QA即明确缺失|不伪造QA；未因423等较大正值而无依据裁剪原始浓度|
|有效域|训练站凸包内，训练≥15站、目标≥3站；无条件外推不支持|沿用前轮门槛，不为增加覆盖而放宽|

定制文件的名义时刻不能被本工作约定变成已证实的物理扫描时刻。原始站表基准及上游处理链也没有因此得到认证。本轮没有用相关性、日照分布或误差反向改时钟、改投影。

统一米制计算的附加审计：Magong、Kinmen、Matsu在EPSG:32651名义经度分区之外，保留投影范围WARN；没有据测试结果换投影或删站。77站两两平面距离相对同一WGS84工作坐标下椭球测地距离，最大相对差约0.1724%。这只是数值诊断，不是作者datum证明，也不是所有卫星像元误差上界。细节在`exports_coordinate_metadata_v1/manifest.json`。

### 29.3 全年匹配、缺失与实际复现

只复用已完整MD5验证的[Zenodo4420335](https://zenodo.org/records/4420335)台湾数据；论文[10.1029/2020EA001599](https://doi.org/10.1029/2020EA001599)，许可CC-BY-4.0。整包MD5仍为`d7329012a175b3ad52012d1f3d6dd50f`，本轮复用既有官方校验记录并核对大小，没有重下载或再复制13.4GB整包。

完整7231份AHI数组实际处理两遍；6张产品表的确定性CSV规范化SHA256逐项完全相同。记录见`reproducibility.json`；不是只检查目录或文件名。77站共用一个实际卫星网格，站点邻域计算统一米制；原始数据只读。

新数据位于`output/taiwan_annual_working_v1/data/`：

- `station_metadata.parquet`：原始坐标、工作坐标、米制坐标和固定角色。
- `ground_non_test.parquet`：训练/验证原始字段、UTC/当地小时、源行号、单位及缺失原因。
- `scans.parquet`、`source_scan_audit.parquet`：每份源文件、SHA、名义时刻来源、QA有效像元及逐站匹配/拒绝原因。
- `aod_hourly_all_sites.parquet`：包括测试站位置在内的无标签AOD协变量和实际有效扫描数量。
- `matched_non_test.parquet`：全年训练/验证左连接表，缺测保留NaN及`match_reason`，未对测试标签做匹配选择。
- `manifest.json`：契约指纹、产品表指纹和实际文件SHA；季度、月份、站点、角色及排除原因CSV另行保存。

全年地面673,152条原有记录，没有把不存在的记录补成观测。冻结划分及数据量如下；测试列的浓度质量计数仅在配置选择完成后生成，没有反馈选择过程。

|角色|固定站数|原始记录|有效地面PM2.5|有效PM2.5＋AOD匹配|
|---|---:|---:|---:|---:|
|训练|50|436656|417952|93791|
|验证|13|113880|110789|25667|
|最终测试|14|122616|119908|27506|

训练/验证共有119,458条有效匹配；AOD匹配占各自有效地面的22.44%/23.17%，并非全年24小时连续覆盖。地面有效范围本轮观察到1—423 μg/m³，正值极端保留；作者表无仪器QA，不能据此声称全部极值都已独立验证。

|月份|训练＋验证有效地面|有效匹配|覆盖率|
|---|---:|---:|---:|
|1|44416|9963|22.43%|
|2|40068|8662|21.62%|
|3|45003|9660|21.47%|
|4|43482|9709|22.33%|
|5|45028|8520|18.92%|
|6|43150|7658|17.75%|
|7|44116|11033|25.01%|
|8|44374|13140|29.61%|
|9|43473|13670|31.44%|
|10|45437|10877|23.94%|
|11|44109|7692|17.44%|
|12|46085|8874|19.26%|

季度覆盖率分别21.84%、19.66%、28.68%、20.23%。逐站详表和计数在`all_station_roles_and_counts.csv`，全角色汇总在`all_role_counts.csv`。源文件、QA、缺失目标、无同小时AOD、样本/凸包条件逐层保留，未把缺测置零。代码复现一致证明了计算稳定，不证明未知作者时区已经正确。

### 29.4 固定站点与防泄漏

仍使用前轮站点划分，SHA256为`58d97ce058b86e9460e3bed68b5047fc1e67b6a86e2d1bf7088b1fac045ba915`；未按成绩换站。

- 训练50站：Annan, Banqiao, Cailiao, Chiayi, Dali, Daliao, Dongshan, Fengshan, FugueiCape, Fuxing, Guanshan, Hualien, Hukou, Linkou, Linyuan, Lunbei, Magong, Meinong, Miaoli, Pingtung, Pingzhen, Puli, Puzi, Qianjin, Qianzhen, Qiaotou, Renwu, Sanchong, Sanyi, Shilin, Songshan, Taixi, Toufen, Tucheng, Wanhua, Wanli, Xianxi, Xiaogang, Xingang, Xinzhuang, Xitun, Xizhi, Yangming, Yilan, Yonghe, Zhongli, Zhongming, Zhongshan, Zhudong, Zhushan。
- 验证13站：Changhua, Datong, Douliu, Erlin, Guting, Hengchun, Keelung, Kinmen, Mailiao, Nantou, Nanzi, Shalu, Tamsui。
- 最终测试14站：Chaozhou, Dayuan, Fengyuan, Guanyin, Hsinchu, Longtan, Matsu, Shanhua, Tainan, Taitung, Taoyuan, Xindian, Xinying, Zuoying。

初始匹配与验证时，测试站在浓度数值解析和模型输入前排除，允许处理其无标签位置/AOD特征。每个时刻仅训练站同期观测参与拟合，查询表不含目标值；不是未来预测或跨年泛化实验。RF残差使用训练站5折OOF，不用验证/测试残差。

验证选择文件`selection_frozen.json`及`selection_freeze_receipt.json`于2026-10-19 10:10:57 UTC冻结，之后才执行最终测试。选择SHA为`298bc21471849485b86458125a9ee720c7e5d70c7d5d1804dec6bb1d1bf0bf4c`；测试命令检查此指纹及契约，修改选择或缺失凭据会拒绝解封。测试只执行选中的IDW/GWR，没有再比较其它测试方法或据测试改参数。

### 29.5 全年实验二、三的真实验证结果

每个实验内部主比较均为完全相同的验证站时目标和输入条件。实验二使用站点观测，实验三要求AOD共同有效；**两实验分母不同，不能直接以7.35对8.58判定融合有无增益。** 下表单位为μg/m³；耗时是并行环境中各次算法调用墙钟秒的累计，不是可跨方法求和的总墙钟，也不代表通用计算复杂度。

|任务/方法|共同N|RMSE|MAE|Bias|R²|累计调用秒|失败/调用|非负修正点|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|实验二 IDW|78522|7.3545|5.4452|0.3098|0.7843|151.46|0/8760|0|
|实验二 Ordinary Kriging|78522|8.6738|6.1933|-0.6244|0.7000|156.08|0/8760|0|
|实验三 线性回归|17163|10.9036|8.0322|-0.0730|0.5524|35.46|0/2327|1|
|实验三 GWR|17163|8.5760|6.3700|-0.0334|0.7231|59.15|0/2327|0|
|实验三 外部漂移UK|17163|8.6540|6.2777|0.0758|0.7180|50.84|0/2327|1|
|实验三 RF＋OOF残差IDW|17163|8.6924|6.4751|1.2615|0.7155|382.16|0/2327|55|

实验二请求8760小时，实际78522个合格目标、10个验证站、365个当地日期；占有效地面110789条的70.88%，其余不作无依据外推。实验三同样请求8760小时，AOD匹配25667条，进一步满足训练数/目标数/空间域的17163条来自10站、307日、2327小时；占有效地面的15.49%。6433小时未满足共同输入条件，明确记录为输入拒绝，不算算法成功或数值调用失败。所有实际执行方法在共同合格队列的预测覆盖100%，同时保留上述总体覆盖。

两任务合计26828次真实GOLEM图数值执行，四方法不是名称占位或固定分数。12个月分块、8个进程、每进程单计算线程；验证运行主进程墙钟133.20秒，不包含此前两遍原始匹配的时间。每次图的父边、端口、状态、质量和错误轨迹以JSONL保存，按月组织，未产生数万份零散报告。

配置和数学定义沿用现有实现：IDW逆距离平方、PyKrige线性变差/伪逆、GWR自适应bisquare固定最多25邻居、RF50树/深度6/最小叶3/种子11与训练站5折残差。未调整参数追求RMSE，未修改第三方核心。所有输出质量为`WORKING_CONTRACT_SOURCE_WARN`：数值、单位、CRS计算、非负门通过，但来源假设未获历史认证。GWR仅比UK验证RMSE低约0.078，不以小差异排序宣称统计显著或算法创新。

### 29.6 冻结方案后的独立测试

按验证集共同队列RMSE选定实验二IDW、实验三GWR，成本只作误差相同时的后续排序。最终14站的目标没有进入上述选择；独立评价后未重新选时区、投影、算法或参数。

|冻结测试配置|有效N|RMSE|MAE|Bias|R²|有效地面覆盖|失败/调用|
|---|---:|---:|---:|---:|---:|---:|---:|
|实验二 IDW|76728|6.6767|5.0073|-0.1594|0.8026|63.99%|0/8760|
|实验三 GWR|15888|8.3781|6.1839|0.6038|0.7055|13.25%|0/2439|

测试潜在122640站时、实际有效地面119908条；AOD匹配27506条。测试有合格预测的站点为9个，不冒充14站全覆盖。实验二有效日期365，实验三314；后者6321个请求小时未通过共同输入条件。GWR有17点按既定规则非负投影，原值保留；不是测试后新增约束。该表只有两个冻结配置的独立评价，不提供未运行的其它算法测试排名。

测试实际11199次图调用，主进程墙钟51.00秒；验证＋测试总38027次实际数值DAG调用，失败0。每站、每月与整体指标、覆盖分母及失败原因均在评价目录保存。全程仅空间留站，不把同站小时随机拆分称为空间泛化，也不宣称时间外推能力。

### 29.7 穷举与随机顺序基线：不伪造GOLEM创新

现有4条多源配置已全部以真实全年输入拟合、预测、执行质量门，其共同验证损失真实参与比较。`exhaustive_objectives.json`记录RMSE、MAE、Bias、失败、覆盖和实际调用成本；不存在模拟Objective或仅凭图结构合格判优。

固定穷举顺序linear→GWR→UK→RF的预算1—4，最佳验证RMSE依次10.9036、8.5760、8.5760、8.5760。另执行种子11、22、33、44、55的无放回随机顺序数值回放，每组预算均4；到第4次都得到同一枚举最优值。回放每次都重读实际验证NetCDF、重算误差并校验原始指标与文件SHA。

共24次数值回调，全部明确标记缓存命中、新拟合数0；此前26828次实际验证图调用没有被重复记成搜索新执行。这些回放只说明小有限空间的顺序/缓存行为，**不是GOLEM Mutation、多种子模型重训、或A/B/C实测**。曲线见`small_space_search/budget_curves.csv`，成本/种子/预算/缓存字段均保留。

- A普通GOLEM：`NOT_RUN_SMALL_SPACE`。
- B后验约束GOLEM：`NOT_RUN_SMALL_SPACE`。
- C语义引导Mutation：`NOT_RUN_SMALL_SPACE`。
- 当前四图穷举：已实际完成，全部合法、全部数值执行通过；不把该合法率视为C优于A/B。

按本轮“小空间先穷举、不人为凑节点”的明确例外，没有强行启动进化搜索。冻结34候选池和官方GOLEM0.5.0未变。**验证集上选到较低误差配置不等于证明语义引导搜索贡献；穷举和固定规则同样能获得该选择。** 因此本轮没有能够支持GOLEM创新优势的公平A/B/C证据，不得声称C更快、更准或更少非法图。

### 29.8 敏感性复用和来源局限

旧32次结果SHA复核未变；旧576份敏感性输出保留，未重跑全部替代时空假设。`reused_sensitivity.json`重用原有配对统计：同坐标条件下24组可配对时间方案中22组排序变化、11组第一名变化。主规则g8/s8/起点来自用户，而非这批分数的最优选择。

旧敏感性仅四日期小样本且使用3826/IDW AEQD，新主实验统一32651、全年；不能把旧数值直接当作32651下全年单因素对照。主规则全年成绩和过去的敏感性证据分开报告，未证明整个研究结论对所有合理定义稳健。作者原始时区/端点/基准仍未证实，AOD也是作者检索产品，不是独立地面浓度真值；本轮代码层防泄漏通过不等于已证明作者所有上游处理依赖。

主要结论仅限这个工作约定、台湾2017、固定站点划分和允许域。站时误差相关，不以庞大样本数冒充独立重复；未开展显著性检验，未宣称跨地区/跨年泛化、整域高分辨率真实精度或严格论文复现。

### 29.9 实际故障、修复与质量验收

1. 汇总器首次使用`groupby(...).observed`，撞上pandas的同名布尔属性，尚未选择或解封测试即失败。改为显式列索引，增加一致真值/冲突真值测试；原失败记录和空输出目录保留，全年模型没有重跑。
2. 最终小样本复现器误用按站排序的匹配表，而全年RF输入保留原始源行顺序；带相同种子的bootstrap面对不同排列并非同一输入。已以实际原顺序诊断，RF差异为0；修复核验器输入构建，六方法在原容差1e-10下全部重现。没有改RF、种子、数据契约、全年基线或测试结果。初次6次重放（5过/1失败）＋1次RF诊断＋修正后6次成功，共13次额外验证图调用，7次失败诊断成本没有隐藏。
3. 最终导出补齐经纬度显式单位和标准名称，另存`exports_coordinate_metadata_v1/`，8份文件数值与原输出完全相同，旧文件与其搜索SHA仍保留。不是重新拟合，不以元数据补齐宣称CF或目标模型原生认证。

自动定向与受影响回归：**25通过、0失败、0跳过、0错误**，见`output/taiwan_annual_summary_fix_tests.xml`。涵盖时区/跨日、不重复偏移、左闭右开、重复/缺失、QA/填充/比例、米制距离、测试解封指纹、训练/查询隔离、真实DAG及一致真值检查。未重跑已有全量回归。

此外已有实际而非夹具的验收：完整匹配重算两遍相同；六方法6个验证用例准确重放；8份全年验证/测试NetCDF经xarray回读并由h5py独立读核；8份补元数据导出再次独立读核。单位、小时bounds、坐标、有限值、非负及预测有效mask一致。输出是站点－时间`sample`集合，不是全域连续网格，网格覆盖状态`NOT_APPLICABLE_POINT_OUTPUT`；不在允许队列的缺测/域外点见匹配及覆盖表，不伪造网格值。

UCI原5类文件指纹与32行指标成果核对未变，没有重跑实验一。原始数据、既有日志、模型与测试、历史代码备份均保留。本轮删除0、下载0、依赖安装0、算法候选新增0。

### 29.10 路径、版本与复现

统一新产物根目录：`output/taiwan_annual_working_v1/`（本轮约367MB，含全年逐调用轨迹，不含重复13.4GB原包）。

- 契约：`configs/taiwan_2017_working_v1.json`及根目录`contract_frozen.json`。
- 新数据与统计：`data/`；全部77站角色/有效样本：`all_station_roles_and_counts.csv`、`all_role_counts.csv`。
- 逐月真实图结果：`runs/validation/`、`runs/test/`，每月预测、调用状态、覆盖Parquet和`traces.jsonl`。
- 主验证汇总：`evaluation/validation_attempt_1/`；主测试汇总：`evaluation/test/`。首次失败的`evaluation/validation/`保留，不误作有效输出。
- 每方法/每站/每月成绩：上述评价目录`comparison.csv`、`by_month_and_station.csv`；源I/O、失败原因与分母见两个`all_*parquet`。
- 建议读取的最终NetCDF：`exports_coordinate_metadata_v1/validation/`六份、`.../test/`两份；原未补坐标单位的数值版本仍作为计算证据保存，均无数据值改动。
- 选择和解封证据：`selection_frozen.json`、`selection_freeze_receipt.json`。
- 穷举与顺序回放：`exhaustive_objectives.json`、`small_space_search/`；不把缓存回放算作新算法执行。
- 总校验：`final_verification.json`、`validation_algorithm_reproducibility.json`、`reproducibility.json`；失败证据见`evaluation/audit_failure_01.json`、`replay_failure_02.json`。
- 代码版本：`code_snapshot/`17份必要源码/配置/测试副本及`source_and_environment.json`的SHA与环境版本，未替代或删除主源码。

现有环境Python3.12.14、NumPy2.5.3、pandas3.0.1、pyproj3.8.0、h5py3.16.0、xarray2026.7.0、scikit-learn1.9.1、PyKrige1.7.3、mgwr2.2.1、GOLEM0.5.0。沿用现有环境，GOLEM只作为图数据结构及既有DAG桥接，没有改优化器或下载新网络。

关键代码：`tools/taiwan_working_contract.py`、`taiwan_annual_adapter.py`、`prepare_taiwan_annual_v1.py`、`run_taiwan_annual_v1.py`、`audit_taiwan_annual_v1.py`、`enumerate_taiwan_objectives_v1.py`、`verify_taiwan_annual_v1.py`及`export_taiwan_coordinate_metadata_v1.py`。研究适配只调用已有算法，实现统一投影和版本化接口，没有重搭平台。

以下顺序用于干净项目副本/新数据版本；已存在匹配/评价目录拒绝覆盖，已完成月份自动复用。当前无需重新跑全年模型：

```text
.venv/Scripts/python.exe tools/prepare_taiwan_annual_v1.py
.venv/Scripts/python.exe tools/prepare_taiwan_annual_v1.py --verify
.venv/Scripts/python.exe tools/run_taiwan_annual_v1.py --workers 8
.venv/Scripts/python.exe tools/audit_taiwan_annual_v1.py
.venv/Scripts/python.exe tools/enumerate_taiwan_objectives_v1.py
.venv/Scripts/python.exe tools/run_taiwan_annual_v1.py --split test --workers 8
.venv/Scripts/python.exe tools/audit_taiwan_annual_v1.py --split test
.venv/Scripts/python.exe tools/verify_taiwan_annual_v1.py
.venv/Scripts/python.exe tools/export_taiwan_coordinate_metadata_v1.py
```

### 29.11 最终回答与停止状态

|问题|最终状态|
|---|---|
|数据契约是否冻结|**是，工作契约v1.0.0**；不等于作者历史定义认证|
|全年匹配是否可复现|**是**，7231份实际源数据完整重算两遍，6张表规范化指纹相同|
|基线是否完成|**是**，全年六方法验证，选定IDW/GWR分别完成独立测试，0实际执行失败，来源质量WARN保留|
|GOLEM创新是否有公平预算优势证据|**没有**；当前四图已实际穷举，A/B/C按小空间例外未运行；五种子仅缓存顺序回放，不是创新证明|
|能否提交正式科研结论|可提交**明确工作假设下的可复查基线/工程结果**；不能提交严格原论文复现、已证实历史时间/基准或GOLEM创新优越性结论|

已独立完成本轮可验证的数据、全年计算、冻结选择、独立评价、数值目标和复现工作。没有要求用户再选方向，也没有为了给创新结论一个YES而扩充图、调测试集或隐瞒来源假设。

## 30. 算法组件协作闭环与真实GOLEM数值搜索（2026-10-19）

**后续验收更正：30.1—30.8记录的是早期工程版本。未新增Registry ID不等于未扩展真实执行内核；其中IDW/RF执行模式的归属不符合本轮“34冻结候选”边界。所有“冻结库合规”推断撤回，不能用这些表证明严格池内搜索优势。真实预测、失败和版本证据均保留，不删除不利结果。最终验收和修复见30.9。**

### 30.1 本轮真正修改和执行了什么

本轮不是再做数据审计。新增`src/ecodata/algorithms/station_fusion_dag.py`，把已有台湾回归与残差实现拆成可执行模式，接入原`NodeAdapter`、`NumericalDAGExecutor`、`TypedGolemFactory`和官方`EvoGraphOptimizer`。没有重建平台、增加Registry算法ID或修改第三方GOLEM。

直接修复原`dag.py`的状态缓存指纹：过去任意对象按文字表示参与指纹，不同已训练模型可能有相同repr；现在按实际序列化状态计算，防止错用另一套拟合权重。新增定向测试验证两个repr相同、系数不同的模型指纹必须不同。

新增执行/汇总/验收入口：`tools/run_taiwan_composition_search.py`、`finish_taiwan_composition_search.py`、`verify_taiwan_composition_search.py`；新增`tests/test_station_fusion_dag.py`。数据契约、全年匹配表、已有算法数学定义及34项冻结池均未改动。

### 30.2 机器契约、组合和适用边界

`output/taiwan_composition_search_v1/contracts_34.json`保留全部34个冻结描述，并逐项列出本任务适用性、输入输出端口、执行模式、单位、时间、坐标、支持域、拟合边界、参数及错误条件。**本任务实际接入7个既有ID的站点处理模式，不是宣称34个组件都适用于台湾站点+AOD。**其余27项的原能力保留；需要栅格、土壤/垂直剖面、气象变量、配对历史模拟或权重的组件不能强塞进本站点任务。部分组件还没有本站点任务的命名端口适配，这属于适配边界，不代表该算法在所有任务中不可用。

实际组件ID：`fusion.residual`、`station.pykrige_ordinary`、`station.pykrige_universal`及MetPy的`linear/natural_neighbor/barnes/cressman`。`fusion.residual`新增的站点执行模式封装**已有台湾脚本中的**官方scikit-learn回归、IDW及残差处理；不是假称原有网格模式已经支持这些端口。IDW仍是已有实现，不新增第35个候选。

真实数据流由端口递归推导，不是预装完整路线模板：

```text
训练站 → 直接空间估计 ───────────────────→ 原始预测 → 双层质量门
训练站 → 回归fit → 拟合状态 → 背景预测 ──┐
               └→ 交叉拟合残差 → 空间估计 ┴→ 相加 → 原始预测 → 双层质量门
查询位置+AOD ─────────→ 对应预测/空间估计端口
```

训练状态、背景预测、残差训练表、残差预测、原始预测和验证后输出是不同类型；下游不能凭名称假装上游已经生成了数据。两个分支必须共享同一训练状态，其指纹还必须匹配当前训练输入。源叶节点由真实输入提供，不作为额外算法候选。

固定前提：PM2.5为`ug m-3`，AOD为无量纲辅助变量；UTC同一小时区间；统一EPSG:32651距离；仅训练站凸包内；查询不得携带目标标签；不得混入留出站拟合；全部目标预测有限才允许进入精度排名。负预测统一执行原有非负投影，保留投影前值及次数。单位、时间、坐标或状态不兼容时拒绝，不静默填字段。已有准备结果已经匹配时间/投影，因此未为了增加节点数重复添加时间聚合、重网格或单位换算。本轮也没有实现任意数据类型间的万能自动转换。

连续回归采用已有LinearRegression或原RF配置（50树、深度6、叶最少3、种子11）；残差使用训练站5折GroupKFold交叉拟合，不能用验证/测试标签。克里金使用已有线性变差函数和伪逆；MetPy调用官方`interpolate_to_points`。所有组件原始许可/来源仍见冻结描述；没有下载新模型或伪造权重。

### 30.3 图空间和反馈机制

从命名端口、实际生产者和拟合状态约束推导出**21种完整图**：7种直接空间估计，2种回归背景×7种残差空间估计。21种在统一真实验证子集上均成功执行，按预测保留10位小数检查得到21种不同输出行为；这只是该子集上的行为差异，不是普适非等价定理。

沿用原有有界类型化图工厂、官方single_add/change/drop/edge和共享的typed_tree_growth适配，所有组共用。同样的不完整质量输出图作为起点；允许缺少能力的中间图存活，但不能作为最终数值合格输出。缺口具体记录到缺少的节点输入端口及数据类型，包含fit/apply状态需要。正式运行每个种子均出现真实缺口补全轨迹。

正式目标返回真实RMSE、MAE、Bias、R²、覆盖、质量状态、逐节点轨迹及成本。不可执行/物理违例不能与精度折抵；不完整图只得到明确标记的`1000+缺口`可进化惩罚，**不是伪造RMSE**；拒绝值为1e6，最终只从质量合格的实际数值结果选图。失败及缓存按图/数据/代码指纹记录，避免同键重复计算。C根据端口能力缺口下降和节点软成本对同一官方提议族加权，GOLEM再用实际数值适应度选择后代。没有另写优化器，也没有给C额外完整图模板。

该实现是当前站点任务的类型化物理契约引导，不等于已实现任意环境任务的自动科学推理。对本任务之外的27项自动桥接、所有失效原因的自动修复，以及C各权重项的因果贡献，尚未验证。

### 30.4 冻结方案、预算与防泄漏

原工作契约v1.0.0及50训练/13验证/14测试站保持不变。搜索不读取测试目标。训练源保持原行顺序，避免RF bootstrap输入排序漂移；全部回归和OOF残差仅使用训练站，同一时刻允许使用训练站观测，故任务是空间留站估计，不是未来时间预测。

为限制计算成本，**搜索阶段使用全年12个月每月固定8个合格小时，共96小时、688条共同验证目标**；按可用性、原QA和几何支持域选取，不按任何模型误差挑日期。这是预先冻结的验证子集，不应写成优化器每次用了全年17163条验证目标。所有选中图另在全年固定验证队列核验，结果不再反向改选。

`protocol.json`在数值执行前冻结日期、种子、数据/代码指纹、规则和预算；初次定向测试发现初始图遗漏端口名沿用了另一任务的`field`而非`prediction`，已修正并保存执行前协议版本，未接触测试结果。

三组共用34项池、适用规则、图语法、起点、固定输入、真实RMSE目标、冷缓存、每种子最多20次计分回调、160次变异提议和120秒搜索上限。C一次选择可比较最多4个提议，所有提议均计费。先做各5次计分冒烟，再执行种子11/22/33/44/55。没有重跑旧32次、576次敏感性、全年匹配或无关全量测试。

**重要证据限制：14个测试站在第29章已经评价过，不能重新称为“从未使用的独立测试集”。**本轮先冻结15个搜索结果及指纹，再统一评价所选5个不同图，未用这些新测试结果重新选图或调参；结果仅作为既有留出站的复核，不是全新盲测确认。

### 30.5 A/B/C真实搜索结果

单位：RMSE为μg/m³。每行是该次搜索在同一688条验证目标上的最好合格图，不是测试集结果。

|种子|A普通GOLEM|B后验过滤|C语义引导|
|---|---:|---:|---:|
|11|8.880044|8.880044|7.758201|
|22|8.936484|8.936484|7.758201|
|33|8.593462|8.593462|7.758201|
|44|7.758201|7.758201|8.072764|
|55|7.758201|7.758201|7.758201|
|均值±样本标准差|8.385278±0.587029|8.385278±0.587029|7.821114±0.140677|

|平均每种子开销/探索|A|B|C|
|---|---:|---:|---:|
|新执行的完整图|3.8|3.8|8.6|
|实际变异提议|53|53|160|
|被拒绝提议|11.2|11.2|40.4|
|提议合法率|78.87%|78.87%|74.75%|
|总搜索墙钟秒|21.58|21.56|36.12|
|第一次合格数值回调序号|3.0|3.0|2.0|

C的验证均值比A/B低约6.73%，3个种子更好、1个更差、1个相同；但C花了约3倍提议、约1.67倍墙钟时间，**不是等计算量优越性**。C没有产生更少非法提议。A/B本次结果完全相同：共享的类型化生成/变异检查已经拦截这些冲突，B额外后验检查拒绝数为0。不能把此结果解释为所有任务上B都无用。

正式15次搜索共299次实际计分记录：81次新完整图执行、44次明确缓存命中、174次不完整图缺口反馈；官方Objective总回调300次，含1次预算后延迟回调，不计入已执行评价。C种子22在160次提议上限停止，只有19次计分，其余各20次；没有补跑凑数或隐藏开销。81次新完整图执行均在688条共同目标上输出齐全、数值质量通过（来源工作假设仍WARN），不是81个全球唯一结构；跨组同图按公平冷缓存重新计算。

21图真实穷举约70.77秒，其最优是现有IDW，RMSE7.758201。五种子随机顺序参照在20/21预算下4次命中最优，均值同为7.821114；该参照复用完全相同的穷举真实结果，**只是明确记账的缓存顺序参照，没有把它计作新执行或公平墙钟随机搜索**。C同样4次命中最优，因此目前没有优于该简单参照的证据，更不能宣称优于穷举。搜索空间虽已突破原四图，但仍只有21图，穷举完全可行。

原四种固定多源图仍保留并以相同688条标签比较：线性回归10.907007、GWR8.836090、UK9.004330、RF+残差IDW8.955798。它们来自第29章真实全年结果的同队列筛选，不重新拟合。审计严格核对四方法与新图目标的站点、UTC时刻和逐值真值一致。

### 30.6 冻结后全年验证与留出站复核

搜索完成后、读取本轮测试目标前，生成`search_selection_frozen.json`及SHA回执；15个选择对应5个不同图。统一完整验证队列17163条、10个有效验证站、307天；留出复核队列15888条、9个有效测试站、314天。原13/14个站的角色未改变，缺AOD/支持域不足者没有人为补齐。上述队列不是全部原始有效地面观测，不能用条件覆盖100%冒充全年全站覆盖100%。

|已冻结图|全年验证RMSE/MAE/Bias|既有留出站RMSE/MAE/Bias|验证/复核预测覆盖|状态|
|---|---|---|---|---|
|IDW|7.462786 / 5.568415 / +0.617234|7.154332 / 5.399625 / +0.595061|100% / 100%|数值质量通过、来源WARN|
|线性背景+OOF残差MetPy linear|8.556629 / 6.341477 / +1.879409|7.547695 / 5.717158 / +0.430909|100% / 100%|数值质量通过、来源WARN|
|MetPy natural_neighbor|7.822072 / 5.881002 / +1.609123|7.063894 / 5.415398 / +0.579452|100% / 100%|数值质量通过、来源WARN|
|MetPy barnes|不排名|不排名|99.7028% / 99.2825%|验证11小时、复核22小时BLOCK|
|线性背景+OOF残差Cressman|不排名|不排名|99.7028% / 99.2825%|验证11小时、复核22小时BLOCK|

Barnes/Cressman在少数空间构型下官方估计返回缺测。保持原参数及边界，整小时按统一规则拒绝，保留NaN、失败节点和分母；没有换成容易样本算RMSE，也没有在看过复核结果后添加回退图救分。不是解析错误或隐瞒失败；这说明96小时筛选不能保证全年覆盖。全部逐种子选择与对应全年结果见`seed_full_validation_and_reused_test.csv`，失败配置保留缺失精度，不用成功种子均值替代所有种子的结论。

IDW无负值投影；线性背景+残差linear在全年验证/复核分别投影16/38个负预测；Cressman在仍成功的小时分别3/24个，失败小时不冒充有效输出。其余所选图为0。原始和投影后预测均保存。投影是所有方法共用的既有规则，不证明模型原始输出天然物理正确。

5图×两角色×12月共120份输出，23830次真实小时DAG尝试，其中66次明确拒绝。加上搜索7776次、冒烟768次和穷举2016次，本轮研究调用共34390次小时DAG尝试；不能把节点缓存命中再计为新算法。全部120份NetCDF已回读，并由h5py独立核对pm25与逐样本Parquet一致；支持域外和失败处保留mask。输出是站点—时刻样本，不是整域高分辨率网格，网格覆盖标记`NOT_APPLICABLE_POINT_OUTPUT`，未编造全场RMSE。

### 30.7 故障修复、定向回归与真实性审计

1. 初始图端口名不匹配由确定性测试发现并修正，首次24通过/1失败记录保留；随后图生成验收通过，没有通过1001次随机重试寻找偶然成功。
2. 训练对象repr碰撞风险已用实际状态指纹修复并测试；不改变已有算法。
3. 缓存记录会携带原计算序号，初版汇总误将其作为本次回调序号。已按日志列表顺序修正预算曲线，另存旧曲线，保留`source_evaluation_index`。所有预测、搜索适应度、选择和原日志均未修改，不因此重跑数值实验。
4. 同队列核对首次撞上Parquet/pandas的UTC微秒/纳秒存储精度差异。只在审计副本显式统一为纳秒UTC后进行严格索引/真值比较，没有关闭断言、改变时区或覆盖数据。失败证据见`audit_time_precision_failure.json`；修复后完整核对通过。
5. Barnes/Cressman全年真实缺测被质量门拒绝，作为运行限制保留，不通过改参数掩盖。

最终必要定向与受影响回归：**43通过、0失败、0错误、0跳过**，见`output/composition_final_directed_tests.xml`。覆盖已有DAG、类型化生成、工作时间契约、台湾隔离/汇总和本轮状态指纹、共享状态、源标签拒绝、非有限值阻止、图恢复及缓存计费顺序。未跑无关全量回归。真实输出审计另见`acceptance_audit.json`：同一初始图、688条相同有序验证目标、逐值重算RMSE、全部选定图共同全年目标、选择指纹、源数据/契约/池指纹均核对通过。

本轮下载0、删除0、候选ID新增0；旧原始数据、全年匹配、UCI实验、32次历史结果、576次敏感性和历史失败证据保留。没有训练新网络或修改第三方GOLEM。实际使用当前已安装PyKrige/MetPy/scikit-learn/GOLEM环境，不另装大型模型。

### 30.8 最终验收结论与复现路径

|验收项|结论|
|---|---|
|组件→契约→生成→真实DAG→数值反馈闭环|**已真实执行**；7个既有ID的任务模式形成21图，不止原四配置|
|A/B/C公平候选/起点/数据/上限及真实数值搜索|**已执行**；15次运行，明确区分新执行、缓存、不完整图和预算后回调|
|完整图数值执行和质量门|96小时21图均通过；全年所选5图中3种完整、2种覆盖不通过，不谎报全PASS|
|冻结与防泄漏|新搜索不使用测试；先冻结再复核，原14测试站已暴露的限制明确保留|
|语义引导是否优于A/B|本次固定回调预算下验证均值较低、命中最优4/5对2/5；但提议和墙钟更高，1种子反而更差|
|是否已证明算法搜索创新|**没有**；小空间随机参照同样4/5、A/B区分度不足、缺少新盲测和机制消融，不能做显著性/泛化宣称|
|是否应直接扩大GOLEM实验|**不建议**；先保留本次真实闭环证据，21图优先穷举。不能靠添加无意义节点扩大空间|
|能否提交科研结论|可提交工作假设下的真实工程闭环、基线和探索性对照；不能提交普适语义搜索优越性、全新独立确认或作者时空定义已证实的结论|

统一产物：`output/taiwan_composition_search_v1/`，约361MiB（含必要逐节点轨迹、预测、120份月度输出和代码快照，不复制13.4GB原包）。

- `protocol.json`、`contracts_34.json`、`catalog.json`：执行前方案、34项任务契约、21图结构；`protocol_history/`保留执行前端口修正版本。
- `smoke/`、`search/{A,B,C}_{seed}/`：实际官方GOLEM记录、生成/连接/拒绝/缺口轨迹、计分、逐小时数值轨迹及预测。
- `seed_results.csv`、`budget_curves.csv`、`execution_accounting.csv`、`mode_means_std.csv`：五种子精度、预算变化、缓存及开销；旧预算曲线保留。
- `enumeration/`、`enumeration_results.csv`、`behavior_equivalence.json`：21图真实穷举和行为核验；`random_order_cached_reference.csv`明确标记缓存参照。
- `four_fixed_baseline_reference.csv`：四种旧配置同队列结果；原逐值预测继续位于第29章目录。
- `search_selection_frozen.json`及回执：最终选择，未根据复核结果改选。
- `final_frozen/`、`full_validation_and_reused_test.csv`、`seed_full_validation_and_reused_test.csv`：5图全年验证/既有留出复核、失败及覆盖；`final_checks.json`、`acceptance_audit.json`：实际核验。
- `code_snapshot/`及必要的`code_snapshot_revisions/`：所用源码/审计改动快照，具体SHA和环境版本见`acceptance_audit.json`。

复现入口如下。现有完成记录会复用，禁止把复用输出算成新实验；需要新实验版本时在干净副本/新输出根目录运行，并保留相同协议、数据和代码快照，不覆盖本轮证据。

```text
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage freeze
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage smoke --workers 3
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage search --workers 3
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage enumerate
.venv/Scripts/python.exe tools/finish_taiwan_composition_search.py --workers 4
.venv/Scripts/python.exe tools/verify_taiwan_composition_search.py
```

**本轮最终判断：真实自动组合和搜索执行闭环已建立；C有特定预算口径下的探索性优势，但没有证据证明其在等总计算成本、简单搜索参照及新独立数据上具有创新性优越性。**未为了得到YES修改数据契约、测试目标、第三方搜索源码或失败标准。

### 30.9 严格冻结池身份修正后的最终执行（2026-10-19）

**本节替代前面V1/V3的冻结池合规性和最终搜索数值结论，历史文件不删除。**前版确实执行了数值算法，但把IDW/RF藏在`fusion.residual`执行模式下，不符合34项冻结组件的身份边界。不能用“冻结JSON未改”代替“实际算法未越界”。本次删除这些任务绑定（不删除既有算法、基线或历史结果），从头运行严格池版本的预实验、五种子搜索、真实随机参照和穷举。没有把旧版较好的分数当作本次成功证据。

#### 30.9.1 真正完成的协作闭环

复用既有Registry、RequirementMapper、NumericalDAGExecutor、GOLEM官方算子及双层质量门，增加任务绑定，不另建优化器：

1. `registry.py`把任务端口适配器绑定到真实冻结ID；`pipeline.py:run_dag`通过统一管线调用既有DAG执行器。数据叶节点和质量节点是边界设施，不是增加的算法候选。
2. 34项的任务契约保存为`contracts_34.json`，记录适用性和排除原因。当前站点/AOD任务实际使用6个站点估计ID及`fusion.residual`，不是声称34个算法都能处理同一种数据。其他组件因为栅格/时间/训练权重等契约不符，不能随机混入。
3. 图工厂根据端口类型递归补齐父节点、命名绑定及fit→apply依赖，支持共享状态分支。正式语法产生18种图：6种直接空间估计，另有2种Kriging背景×6种残差空间估计。它们不是原四配置清单，也不是任意排列34个名称。
4. 状态缓存包含真实拟合内容指纹；背景和残差必须来自同一个拟合状态。发现过“子树替换生成两个不同fit状态”的实错，已经修复并增加确定性拒绝测试，不能依赖大量随机重试碰运气。
5. 真正调用PyKrige的OrdinaryKriging/UniversalKriging和MetPy的线性、自然邻域、Barnes、Cressman实现。`fusion.residual`只做已有的观测减背景、背景加残差运算，不伪装成RF、IDW或新网络。
6. 组合路径对训练站做5折整站交叉拟合。折内凸包外的OOF预测不外推，保留缺失和原因。残差支持域不足的目标只使用合法背景预测，并在NetCDF的`residual_usage`中标明；不填造残差观测。
7. 第一质量门在训练源进入拟合前触发；逐边检查单位、米制CRS、时间、契约版本、类型及状态来源。最终门检查有限值、非负投影和整个固定队列覆盖率；任一小时失败保留该小时所有目标，不从主比较分母删除。
8. 执行反馈含失败节点、原因、缺少能力、质量结果、RMSE/MAE/Bias/覆盖及耗时。相同失败指纹也缓存，避免把同一失败图反复计算；缓存命中与真实执行分开记账。

确定性验收：两条不同路线各用一次官方`add_separate_parent_node`从未完成图生成并实际执行，包括直接UK与fit/apply/OOF残差多分支图。另有测试明确验证连续补全缺口`3→2→1→0`，不是用2000次随机探针找偶然成功。`strict_generation_acceptance.json`保留真实父边、端口、后端类名、训练站与折内隔离轨迹。

**边界：**本次是站点预测及其空间估计任务，不是受训的任意尺度影像重构；NetCDF采用站点样本维度，不冒充全域高分辨率真值。图中UK的AOD是无量纲漂移协变量，不能和PM2.5浓度直接平均。普通Kriging/MetPy基线不使用AOD数值，仍在同一合格队列比较。

#### 30.9.2 固定数据、预算和执行条件

- 沿用`taiwan-2017-working-v1.0.0`、50训练/13验证/14原测试站；不重新匹配全年，不下载、不删除原始数据。
- 主搜索队列：预先按月份均匀抽取96个真实合格小时，共688个验证站小时；不是全年精度结论。所有18图使用相同目标、训练源、有效域和QA，预测查询不含留出标签。
- A默认Mutation；B默认Mutation＋后验契约检查；C能力缺口/物理语义引导提议。A/B/C共用端口语法、未完成初始图、硬约束、RMSE目标及缓存规则；第三方GOLEM核心未改。
- 种子11、22、33、44、55；每运行最多20次计分Objective、160次外层提议、120秒软时限。已开始的一次图计算允许完成，所以实际墙钟可超过120秒，不能据此宣称严格相同秒数。
- R为实际执行的随机完整合法语法采样，使用相同数据、目标、回调/提议/软时限和冷缓存；其生成完整图的策略不同于增量Mutation，明确作为简单搜索参照而非同一种Mutation消融。
- 提前收敛/种群不足等官方正常停止没有用重复图凑足20次。Objective中的不完整图是`1000+缺口`的拒绝反馈，物理/执行失败为统一拒绝；这些值不叫RMSE。主质量合格图按真实RMSE选择，不把软成本混入浓度误差。
- 12小时小预实验：A/B各20次计分、4次新图执行；C14次计分、6次新图执行。A/B另各有1次预算后的未计分回调，日志保留；不把它算作第21次有效评价。三组均真实生成新图、执行和反馈后才开展96小时对照。

#### 30.9.3 五种子真实验证结果

单位：RMSE为μg/m³。括号为“计分Objective/真实新图执行”。数据来自`seed_results.csv`；缓存与不完整图不算新图执行。

|种子|A默认|B后验|C语义引导|R真实随机参照|
|---|---:|---:|---:|---:|
|11|8.6137（15/2）|8.6137（15/2）|8.3391（8/4）|8.0728（10/6）|
|22|8.9365（8/2）|8.9365（8/2）|8.0728（8/5）|8.2443（7/6）|
|33|8.5694（8/4）|8.5694（8/4）|8.9365（8/3）|8.0728（5/4）|
|44|8.0728（8/2）|8.0728（8/2）|8.3391（5/4）|8.1665（8/7）|
|55|8.3391（8/3）|8.3391（8/3）|8.3391（6/5）|8.5694（5/4）|
|均值±种子标准差|8.5063±0.3227|8.5063±0.3227|8.4053±0.3186|8.2252±0.2054|

不同方法因正常停止和图成本不同，实际回调次数并不相等。另按每个种子四组共同可达到的计分前缀8/7/5/5/5次比较，均值A/B=8.5242、C=8.4053、R=8.2439。C相对A在2个种子较好、2个较差、1个相同；不能只引用均值。完整随预算变化见`budget_curves.csv`和`common_budget_comparison.csv`。

|累计开销（5种子）|A|B|C|R|
|---|---:|---:|---:|---:|
|计分Objective|47|47|35|35|
|真实新图执行|13|13|21|27|
|图结果缓存命中|0|0|4|3|
|外层提议|134|134|332|30|
|生成拒绝|27|27|98|0|
|总墙钟秒|465.8|466.3|590.6|650.3|

20次运行合计164次计分，74次真实新图执行，7次图缓存命中；其余为不完整图等非数值反馈。节点级缓存另记在逐图结果，不混入图级计数。C的98次拒绝包含它自行探索更多提议的代价，不能隐藏或把它说成更低非法率。A/B在当前类型化语法下轨迹和成绩相同，后验附加拒绝为0，因此此任务无法分辨B的额外价值。C确实更常补齐能力缺口，但这不等于精度或等计算成本优势。

`search_feedback_metrics.csv`进一步分开了中间图与真实失败：A/B各34次不完整图反馈，C10次；完整且质量通过的数值回调比例分别为13/47=27.7%、13/47=27.7%、25/35=71.4%（C含4次明确缓存）。首次得到合格完整图的回调位置，A/B五种子为8、2、3、2、3，C均为2。三组已实际执行的完整图在96小时队列上均通过，无数值失败；这不能解释成“不完整中间图都是非法物理图”。这里支持的是C较快补齐处理能力的工程观察，不是最终精度的显著性结论。各图允许使用输入训练站的同期真实观测，属于空间留站估计，不是提前预测未来小时。

按结构指纹跨种子去重，A/B分别实际触达9种合格图，C为13种，R为15种；四组并集18种。74次新执行指每次冷启动运行中真实新算的图，不是74个全局不同图，也没有把同图跨种子复算当成算法空间扩大。

#### 30.9.4 18图穷举及已有固定基线

18种图全部真实执行96小时、688个目标并完成NetCDF回读，预测向量保留到10位小数得到18个不同数值行为组。全部达到本任务覆盖要求，但状态仍为`WARN_WORKING_CONTRACT`，不把坐标/时间假设写成已证实。

|代表性图（真实执行）|RMSE|MAE|Bias|
|---|---:|---:|---:|
|MetPy自然邻域直接估计（穷举最优）|8.0728|6.0323|1.5382|
|MetPy Cressman直接估计|8.1665|6.1302|0.1307|
|MetPy线性直接估计|8.2443|6.1110|1.5145|
|OK背景＋OOF残差OK|8.3391|6.2286|0.2958|
|UK背景＋OOF残差OK|8.5694|6.2858|0.5996|
|UK背景＋OOF残差UK|8.6137|6.3591|0.5210|
|UK直接估计|9.0043|6.4892|-0.0251|
|OK直接估计|9.1118|6.5402|-0.5043|

其余10图的逐值结果、失败计数、投影修正次数、时间及覆盖见`enumeration_results.csv`，不选择性删除较差组合。部分残差图发生1次负值投影，原值保留。组合图约171～172个目标不具备可靠残差支持，采用明确标注的背景单独输出；不是把未经支持的残差外推到全域。

旧四配置在同96小时队列上的已完成真实结果为线性10.9070、GWR8.8361、UK9.0043、RF＋残差IDW8.9558。它们仅作为历史固定算法基线，不把未冻结的RF/IDW塞入搜索。各图排序只对该固定队列有效。穷举共享节点缓存，不能把其总耗时直接当作与20次冷启动搜索等成本的比较。

结论：当前18图空间足够证明自动组合与不同数值行为，却很小，直接穷举是合理参考；最优是简单空间方法，不是多源组合。C没有胜过R，不能据此宣称语义搜索优于简单搜索或多源融合必然更准。

#### 30.9.5 冻结后复用留出集核验（不是新的盲测）

所有20个A/B/C/R选中方案及参数先写入`strict_validation_selection_frozen.json`，再统一核验去重后的7图；指纹`265f9b020547ace7216f3764d2b8ef47c7ab4fd81bb0b0d6e08c3a54a42c76e2`。未使用该复核结果重选或调参。

事先固定使用相同96个时刻及原14测试站。实际有92个时刻、9个具备共同数据和空间支持的站点、598个目标符合冻结契约；其余4小时未补造。该分母与验证688不同，不把100%条件覆盖误写成原始测试站全年全覆盖。

|选中图（键前缀）|复核RMSE|MAE|Bias|有效覆盖|质量结论|
|---|---:|---:|---:|---:|---|
|自然邻域 b99d1e9a|7.2587|5.5099|0.4839|598/598|WARN：工作假设|
|线性 77d05f6c|7.4407|5.5998|0.4231|598/598|WARN：工作假设|
|OK＋残差OK c5dc82c9|7.5132|5.5366|0.6056|598/598|WARN：工作假设|
|UK＋残差OK 0da13eb0|7.8259|5.6005|0.3697|598/598|WARN：工作假设|
|UK＋残差UK 977dfada|7.8615|5.6432|0.4490|598/598|WARN：工作假设|
|Barnes a2367629|不排名|不排名|不排名|593/598|BLOCK：1小时5目标缺测|
|Cressman 41b0633b|不排名|不排名|不排名|593/598|BLOCK：1小时5目标缺测|

两种被BLOCK方法均在2017-06-07 01:00 UTC出现未覆盖目标。没有为了完成交付而调整搜索半径、改站、改时刻、填NaN或只评价其余593个目标。NetCDF仍可读，文件存在不代表质量通过。逐种子继承的最终结果见`seed_reused_holdout_metrics.csv`，不能删除失败种子后比较均值。

这些14个站此前已经在第29章/旧轮次读过真实值，所以本次只能称冻结后的“复用留出集复核”，不能作为全新独立确认。特别是A22/B22/C33选择Barnes、R44选择Cressman均未通过完整覆盖，不能以其他方法的较好分数替它们报PASS。

#### 30.9.6 测试、源码和复现证据

本次相关定向回归：**77通过、0失败、0错误、0跳过**；`output/composition_strict_final_tests.xml`保留记录。两条已有flox/NumPy时间精度弃用警告不影响通过；没有重复执行无关全量回归。覆盖真实状态指纹、逐边契约、OOF训练隔离、输入标签拒绝、质量阻止、三步Mutation补全、跨状态求和拒绝、失败缓存、Registry/Pipeline联动、冻结身份限制和旧流程兼容。

关键代码：

- `src/ecodata/algorithms/{registry,pipeline,dag,semantic_requirements}.py`：统一任务注册、需求映射、真实执行与缓存。
- `src/ecodata/algorithms/station_fusion_dag.py`：端口/状态/语义图工厂；其中旧模式保留为历史实现，当前严格任务不注册RF/IDW模式。
- `src/ecodata/algorithms/frozen_station_dag.py`：当前严格后端绑定、Kriging拟合、OOF残差及有效域掩码。
- `src/ecodata/fusion/residual.py`：复用残差算术；没有重写第三方核心算法。
- `tools/run_taiwan_composition_search.py`：实际官方GOLEM及Objective；`run_strict_random_reference.py`为冷启动随机参照。
- `tools/audit_strict_frozen_search.py`：生成与最终核验；`run_strict_frozen_holdout.py`为冻结后一次复用留出复核；`verify_strict_full_validation.py`分月核验全年验证覆盖，不用于改选。
- `tests/test_station_fusion_dag.py`及XML列出的受影响测试：实际断言，不以接口测试代替真实运行。

统一产物根目录：`output/taiwan_frozen_pool_search_final/`。包含`protocol.json`、34契约、18图、源码快照、实际内核绑定表、`pilot/`、15次`search/`、5次`random_reference/`、18图`enumeration/`、逐值Parquet/NetCDF、逐节点JSONL、缓存/预算/失败日志及上述CSV。辅助审计和复核脚本有各自源码指纹；执行核心9个文件与协议指纹一致。

冻结池SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`；工作契约仍为`43e2e96221cd6e34adc06149331a97fd219bb386134f9e82ab283a48936ef9c4`；站点划分仍为`58d97ce058b86e9460e3bed68b5047fc1e67b6a86e2d1bf7088b1fac045ba915`。不删除数据、不新增算法ID、不修改第三方GOLEM源码；历史失败和旧版不合规结果全部保留并说明。

复现须使用本节输出路径，不能使用前面30.8节的旧版默认路径覆盖历史记录：

```text
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage freeze --output output/taiwan_frozen_pool_search_final
.venv/Scripts/python.exe tools/audit_strict_frozen_search.py --phase generation
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage pilot --output output/taiwan_frozen_pool_search_final --workers 3
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage search --output output/taiwan_frozen_pool_search_final --workers 3
.venv/Scripts/python.exe tools/run_taiwan_composition_search.py --stage enumerate --output output/taiwan_frozen_pool_search_final
.venv/Scripts/python.exe tools/run_strict_random_reference.py --workers 2
.venv/Scripts/python.exe tools/audit_strict_frozen_search.py --phase final
.venv/Scripts/python.exe tools/run_strict_frozen_holdout.py --freeze-plan
.venv/Scripts/python.exe tools/run_strict_frozen_holdout.py --workers 3
.venv/Scripts/python.exe tools/verify_strict_full_validation.py --workers 3
```

已有完成文件会复用；复用不是一次新的实验。协议核心源码变化时必须使用新输出版本，不覆盖旧记录。未确认历史时间/坐标定义依然只是工作约定，576次旧敏感性审计只用于稳健性讨论，不能按RMSE反向选时区。

**最终判断：工程闭环及严格冻结池数值搜索已完成；C的优势没有被证实。**当前任务宜以18图穷举/简单基线为比较依据，不应为证明GOLEM优越而扩库或加无意义节点。全年覆盖核验见下一小节；部分方法质量不合格及旧测试暴露属于真实限制，不能通过降低验收门槛改成PASS。

#### 30.9.7 全年验证复核与本轮最终验收

已完成冻结7图的全部12个月复核，共84份月度运行，不用于改选方案或再次调参。共同目标是2327小时、307天、10个有效验证站、17163个站点小时；不是未经AOD/有效域筛选的110789条原始有效验证观测。条件队列覆盖100%不能替代对原始数据可用率的披露。

|冻结图|全年验证RMSE|MAE|Bias|覆盖率|失败小时|负值投影次数|
|---|---:|---:|---:|---:|---:|---:|
|自然邻域|7.8221|5.8810|1.6091|17163/17163|0|0|
|线性|7.9975|5.9732|1.5986|17163/17163|0|0|
|UK＋OOF残差OK|8.5045|6.2052|0.4358|17163/17163|0|1|
|UK＋OOF残差UK|8.5141|6.2116|0.3391|17163/17163|0|1|
|OK＋OOF残差OK|8.5680|6.1903|0.0981|17163/17163|0|0|
|Barnes|不排名|不排名|不排名|17112/17163（99.703%）|11|0|
|Cressman|不排名|不排名|不排名|17112/17163（99.703%）|11|0|

每个方法的相同17163个目标全部保留。两种不完整方法在当前MetPy默认邻域配置下产生非有限值，最终质量节点明确BLOCK，未删除51个失败目标后计算主排名，也未因看到留出结果而修改邻域参数。组合图的背景与残差调用、状态来源、支持域掩码和全年逐月结果均有轨迹。自然邻域的较低RMSE伴随正Bias，不能把单一指标低直接说成所有方面更优；组合图Bias较小但RMSE未占优。

本次全年复核发现：96小时最优/排名不能代替全年泛化，部分方法小样本合格但全年不合格。保留既定选择作为真实实验结果，不悄悄改用年度成绩更好的配置。当前结果仍属于冻结工作假设下的探索性空间估计，不能声称作者原始时区/坐标已获证实，也不能声称严格复现论文。

最终只读交叉核验`tools/accept_strict_search_delivery.py`已实际执行通过：

- 109份NetCDF（18份穷举＋7份复用留出＋84份全年月度）均回读；检查数值、观测、时间、站号、经纬度、单位、契约、米制投影、有效掩码及残差用途掩码。
- 109份中有16份文件的结果状态为BLOCK（14份月度＋2份留出），这16份**没有**被改成质量通过。文件完整性通过与方法质量失败分别记录。
- 各方法同一阶段的完整目标队列一致，重复站点/时间检查通过；冻结代码9文件、输入6文件、数据契约和34候选池SHA256未变。
- 最终选择SHA与全部留出回执一致；全年复核未读取测试标签，未用复核结果改选；原测试历史暴露仍明确标记。
- 运行版本：NumPy 2.5.3、pandas 3.0.1、SciPy 1.18.1、PyKrige 1.7.3、MetPy 1.7.1、scikit-learn 1.9.1；官方GOLEM沿用既有隔离路径，未修改第三方核心。
- 结果约438MiB（记录时，不含任何重复原始包）。`delivery_acceptance.json`保存最终证据和辅助脚本指纹；`postfreeze_full_validation_metrics.csv`和`postfreeze_full_validation_receipt.json`保存全年结果。无额外Markdown交付文档，无数据删除或重新下载。

|最终验收项|状态及准确含义|
|---|---|
|34组件任务契约、兼容关系和排除原因|完成；实际适用7个ID，不夸称其余27个可无条件参与此任务|
|自动生成有真实父边、命名端口、训练状态的DAG|完成；两路线真实生成、18种行为及确定性Mutation补全已验证|
|统一Registry→Pipeline→质量门→数值Objective|完成；真实执行、失败反馈、内容缓存及逐节点记录已验证|
|A/B/C五种子公平上限搜索与随机、穷举参照|完成；同一上限但不同实际停止时间，提供共同回调前缀和全部开销，不假装严格等秒数|
|数据与代码保护|完成；无下载、无删除、冻结契约/划分/池不变，旧失败与不合规结果保留|
|定向回归|77通过，0失败/错误/跳过；未重跑无关全量套件|
|所有选中方法的全年质量|部分通过：5/7完整，2/7被BLOCK；不得宣称全通过|
|冻结后留出评价|执行完成：5/7完整，2/7被BLOCK；为历史已暴露留出集，不是新盲测|
|C是否显示更快补齐能力|本小空间有工程证据：首次合格回调更早、触达图更多；不能等同科学精度收益|
|C是否胜过普通搜索及简单搜索|未证实；均值略优A/B但部分种子更差、R均值更好、A/B无区分度|
|是否已证明算法创新|没有；不作显著性、普适性、独立确认或所有环境模型支持的宣称|
|能否提交正式科研结论|可提交真实工程闭环及带限制的探索性对照；不足以提交“语义搜索稳定优越”的确认性结论|

```text
ENGINEERING_EXECUTION_LOOP = PASS
STRICT_FROZEN_POOL_CONFORMANCE = PASS
REAL_NUMERICAL_SEARCH = PASS
TARGETED_TESTS = 77 passed / 0 failed / 0 errors / 0 skipped
ANNUAL_SELECTED_GRAPH_QUALITY = 5/7 complete; 2/7 BLOCKED
REUSED_HOLDOUT_GRAPH_QUALITY = 5/7 complete; 2/7 BLOCKED
FRESH_BLIND_CONFIRMATION = NO
SEMANTIC_SEARCH_SUPERIORITY = NOT_PROVEN
```

所有本轮可独立完成的工程开发、真实执行、全年验证复核和证据核对已完成。不能靠继续调整已暴露测试集、无意义扩图或删失败样本把科研结论改成YES；因此在保存完整结果后结束本轮，不以“需要再写计划”代替已经实际执行的工作。

## 31. 用户城市小时Excel：单源时间恢复与真实搜索对照

### 31.1 范围、来源与不能越过的证据边界

本轮只读`D:/codex/数据/2021-2024全国空气城市小时数据/`，不下载、不更名、不移动、不覆盖原件，不重新执行台湾/AOD匹配。实际发现11个xlsx，不是预先硬编码数量。完整读取9,171,521条源记录、370个不同code；其中9,137,531条通过“时间可解析、实体存在、数值有限且非负、不冲突”的数值筛选。该筛选**不是仪器QA或原始发布来源认证**。

|文件地区简称|原始记录|原始空间标识数|
|---|---:|---:|
|安徽、福建、江西|952833|37|
|北京、天津、河北、山西、内蒙|953004|37|
|甘肃、青海、宁夏|712172|27|
|广东、广西|925814|35|
|海南、四川、贵州|872498|33|
|湖北、湖南|714285|27|
|辽宁、吉林、黑龙江|960984|37|
|山东、河南|977253|46|
|上海、江苏、浙江|781780|40|
|新疆|422287|16|
|云南、西藏、重庆、陕西|898611|35|

实际时间范围为2021-01-05 10:00至2024-04-16 00:00，**不是完整2021—2024四个自然年**。原始字段共16个：`code,name,province,longitude,latitude,o3,co,no2,so2,pm2_5,pm10,pollutions,aqi,level,quality,pubtime`。字段映射、逐文件工作表、类型、范围、缺失、重复和坐标情况保存于机器清单，原始字段不改。

已知限制：

- 文件是城市标识记录，没有可核实的监测站ID及城市统计支持域说明，不能把370个code直接说成370个独立地面仪器站。
- 表头未声明PM2.5物理单位，也没有可确认的时区、小时起止定义或坐标基准。保持无时区的原始`pubtime`标签，不转UTC、不擅自补EPSG。不同文件分别运行，不合并为一个物理浓度数据源。
- `quality/level`是AQI等级文本/等级，不作为仪器QA。零值不无依据地视为缺失；原始缺失保留。异常大数值标记并披露，不为了降低RMSE删除。
- 因此，本轮可验证的是**用户文件中已有数值的遮挡恢复**，误差单位为“Excel原数值单位，尚未确认”。不能将其自动改写为μg/m³的已认证浓度精度、空间降尺度或官方质量合格数据。

### 31.2 数据读取、时间轴与冻结方案

复用openpyxl读取工作簿元信息及独立核对，针对接近Excel行数上限的大文件，使用lxml按原有单元格格式流式提取，不把9百万行一次加载进内存。Excel日期起点取自工作簿epoch；不根据文件名猜时间。保留源文件SHA256、原始行号、整行内容指纹和字段映射。没有无条件对重复时间求平均：整行相同可去重；同实体/小时冲突则标为不可评价，其他实体的相同时间不算重复。

每个code单独建真实小时索引，原始空缺和轴上不存在的小时均为缺失，绝不串联不同地点。数值有效记录不少于48小时才进入时间任务；实体不足条件会记录，不补造。原始xlsx只读，所有派生文件集中在：

`output/city_pm25_reconstruction_20261019/`

其中`prepared/`和初次全量盘点复用；**最终实验结果以`strict_sealed_v2/`为准**。父目录早期实验结果已撤回，不能拿来补最终样本或成绩。

试跑文件按用户指定选择“空气城市小时数据-安徽、福建、江西.xlsx”。仅按完整性在月初起的30天窗口中选最大有效覆盖，平局取更早时间、实体编号稳定排序；没有查看RMSE后选区域。冻结窗口为2021-02-01 00:00至2021-03-02 23:00（720小时），10个实体：340100、340200、340300、340400、340500、340600、340700、340800、341000、341100。平局规则恰好选出安徽实体，**不宣称试跑覆盖了三个省全部空间分布**。其他实体随后在批量阶段照常执行。

前480小时（至2月20日23:00）用于验证选型，后240小时为冻结后测试。后10天的全部观测在验证算法、归一化和统计拟合中不可见，不仅仅隐藏当前seed的目标值。其他文件只使用冻结方案，不再次按省调参；外部实体重叠检查单独保存。

### 31.3 遮挡与实际算法

固定规则写在执行前的`experiment_config.json`，所有方法共用：

1. 随机遮挡：seed11/22/33/44/55，对每个实体原始有效数值抽取floor(20%×有效数)，不从自然缺测取真值。
2. 隔小时：依据完整时间轴位置而非Excel行号，分别隐藏偶/奇相位。保留另一个相位；不是2小时平均量分解。试跑锚点为冻结窗口起点，批量锚点为对应文件真实最早小时。
3. 连续缺失：6/12小时，按季度从真实完整且左右有观测的候选区间中以固定seed选一段；仅用时间和有效性选择，不看浓度高低。没有合格区间就不运行该子任务。
4. 峰值：取当前允许输入中的可见值90%分位；隐藏值不用于估计。保留自然缺失、人工遮挡、源行号、峰值标记及评价分母。

基础方法为已有UCI实验所用的pandas时间线性插值、前值填充、按小时中位数操作，以及本任务必要的SciPy最近邻基线。最近邻允许显式端点常值延伸；线性只在可见支持区间内插值；ffill只用过去；中位数与其他双侧方法属于回顾性恢复，不能混称预测未来。

四种最小基线只注册到当前任务的局部Registry，**未修改34项冻结池，也未冒充DualTimesField或高级模型**。DualTimesField仍保留，其明确单位、最多512输入、逐序列拟合等既有契约没有被绕过。分类、空间栅格、AOD等不适用组件没有强行放入本任务。

真正的处理图经原有`TypedGolemFactory→Registry任务端口→UnifiedAlgorithmPipeline.run_dag→NumericalDAGExecutor→质量门`执行。8种合法图来自4种直接基线，以及线性/ffill的缺失结果由最近邻/小时中位数条件回退的4种组合。回退只填主方法未覆盖处，不改变已有有效预测；不是为了增加节点再做无意义处理。

原始单位未证实时，不伪造μg/m³让质量门通过：图输入只按当前允许的未遮挡值尺度作明确的无量纲归一化，质量门检查单位1的数值、时间及非负性，输出按同一尺度还原，报告始终WARN原单位未证实。尺度不含遮挡真值。查询时间也进入缓存上下文，避免同一条序列不同目标误命中缓存。这种数值检查不等于物理浓度单位认证。

### 31.4 修复与历史结果保护

首个定向测试发现结果门要求显式单位：未把空单位偷偷设成已知PM单位，而是补齐上述无量纲归一化/还原记录及原单位未知警告；初次失败XML保留。另补目标时间缓存身份测试。

进一步发现早期验证仍能使用后10天的部分保留观测；跨mask seed时可能读到其他测试目标，尤其影响小时中位数统计。该版的试跑、搜索、冻结选择、已执行测试和部分批量结果全部保留，但**撤回正式比较与迁移资格**。`validation_context_withdrawal.json`和`pre_sealing_fix_source/`保存原因及代码；只停止了核对命令行后的本项目批量进程，没有删除原始数据。新建`strict_sealed_v2/`，复用完整提取数据，验证输入彻底截断到前20天后，重跑受影响实验。

修复后的新测试明确验证算法输入最大时间不超过验证截止时刻。相关定向回归39通过，涉及新任务及原DAG/图生成/时间处理，不重跑无关全量测试。

因早期撤回版曾计算后10天及部分外部文件，V2结果也**不重新包装为全新盲测**。修复依据是代码隔离问题而非测试RMSE；V2选择严格不读后10天，冻结后才复核。任何探索性结论必须保留这一历史暴露限制。

### 31.5 空间任务与研究结论边界

11份文件都有longitude/latitude字段，但缺少基准及城市值对应空间支持域的权威说明。按用户“坐标可靠才开展”条件，本轮空间留点任务明确为**不适用/未运行**；没有补城市坐标、假定WGS84或把城市聚合值说成逐站点观测。时间恢复结果不作为栅格降尺度证据。

本轮A/B/C/R使用同一个8图语法、未完成初始图、数据、质量要求和数值目标；seed11/22/33，每运行最多8次计分、50次外层提议、45秒软时限。随机20%及两个隔小时相位分别执行，全部为真实数值调用。A默认GOLEM变异、B后验契约检查、C既有能力缺口/简度引导、R真实随机图采样；无第三方GOLEM源码修改。小空间必须同时穷举，不能将结构合法或更早补齐端口说成精度创新。

各文件的批量误差、冻结方案迁移、搜索预算与验收结果在本章后续小节给出；不同文件单位未确认，不将其原数值RMSE强行混成一个全国物理浓度RMSE。

### 31.6 全部文件的实际时间恢复结果

共读取11份文件、9,171,521条记录、370个不同城市/区域code；365个实体满足不少于48个有效观测的时间任务条件。实际时间范围为2021-01-05 10:00至2024-04-16 00:00，并非完整四个自然年。五个不适用实体为612000杨凌示范区、131300雄安新区、369900赣江新区（各7个有效值），370684蓬莱市、210281瓦房店市（无有效值）；不补造数据。跨文件没有相同code；这不代表城市统计定义已经核实。

下表是线性插值在四个固定基线共同可评价队列上的RMSE，随机项汇总5个固定遮挡seed，两个相位分别保留。所有数字均为**对应Excel原数值单位，物理单位未确认**。每个文件独立计算，未挑选好看的省份，未删除极端值。

|文件地区|合格实体|随机20%|隔小时相位0|隔小时相位1|连续6小时|连续12小时|
|---|---:|---:|---:|---:|---:|---:|
|安徽、福建、江西|36|3.678|3.393|3.378|6.311|9.265|
|北京、天津、河北、山西、内蒙|36|8.953|8.189|8.037|10.081|12.308|
|甘肃、青海、宁夏|27|9.354|9.315|8.132|10.720|14.522|
|广东、广西|35|2.665|2.289|2.326|4.694|6.516|
|海南、四川、贵州|33|3.123|2.800|2.826|5.695|8.105|
|湖北、湖南|27|3.448|2.990|3.054|7.260|11.315|
|辽宁、吉林、黑龙江|36|6.189|5.225|5.642|12.127|17.828|
|山东、河南|45|10.813|12.958|12.838|8.116|12.824|
|上海、江苏、浙江|40|4.265|4.034|3.969|6.077|9.797|
|新疆|16|172.627|219.141|153.691|22.513|24.125|
|云南、西藏、重庆、陕西|34|4.274|3.720|3.773|7.305|10.318|

四个固定基线的共同样本比较中，线性插值在相位0、6小时、12小时任务均为11/11文件最优；随机缺失及相位1分别为10/11，另一个文件由小时中位数取得较低误差。不能据此宣称线性方法对任意污染过程最优。

极端值核对：新疆文件中659001石河子，2021-12-08 16:00，原Excel第377165行J列确为100061。已使用独立openpyxl读取核对，非本次流式解析产生；是否为真实污染浓度尚未证实。保留并警告，未裁剪或删除以降低RMSE。该值会严重影响平方误差和排名，因此新疆结果不能直接解释为重污染恢复能力。

共同队列之外仍保留所有应评价目标及失败分母，见`all_targets`。线性方法只内插、不外推，其各文件/任务最低原始目标覆盖率约99.9778%，而不是一律100%。共同队列100%覆盖是比较队列定义，不可掩盖边缘缺测。最近邻的端点延伸与仅过去的ffill信息条件不同，均在`information_mode`中明确区分。

试跑验证的线性RMSE：随机20%为4.306595（4,840目标）；两个相位为3.921750（2,390）和4.216078（2,390）；6小时6.403124（60），12小时11.521913（120）。后两项样本很少，不能据此形成强峰值结论。峰值RMSE、MAE、Bias、R²、运行时间、覆盖率及每次失败均保存在逐任务/逐文件表中，不只保存RMSE。

冻结后10天共同队列的固定基线结果如下；这些是本次隔离修复后的复核，仍保留31.4节的历史暴露限制。

|试跑后10天任务|目标数|线性RMSE|最近邻RMSE|前值RMSE|小时中位数RMSE|
|---|---:|---:|---:|---:|---:|
|随机20%（5个mask汇总）|2,340|3.181808|4.764138|5.598878|18.014675|
|隔小时相位0|1,200|2.852667|5.077811|5.077811|17.633089|
|隔小时相位1|1,190|2.889237|4.463578|4.463578|17.532437|

冻结的连续段位置未落入后10天，故该测试段的6/12小时评价为不适用，不填0、不重新挑选有利缺失段。连续任务的真实结果见验证与全文件批量表。

### 31.7 A/B/C/R实际搜索、穷举及公平性

完成3种时间任务×3种seed×4种机制，共36次搜索运行；实际计分回调270次，真实新图数值执行150次，缓存命中33次，外层提议1,047次，拒绝54次。计分回调还包括不完整/不可执行图的惩罚，**不能把270次回调说成270次真实预测执行**。累计搜索墙钟约32.813秒，仅指搜索记录，不含Excel解析、全文件批量预测或大清单校验时间。

每种机制共用最多8次计分、50次外层提议和45秒软时限。A/B/R各运行达到8次计分，C在seed11/22/33分别达到5/7/6次后触及50次提议上限；这是预算内实际结果，不以重复图或缓存凑满8次。未发生隐藏的优化器异常或1001次随机生成重试。所有运行上限由`comparison_integrity.json`验证。

下表为每个随机种子的最佳Validation RMSE；A、B数值完全相同仍如实报告。每行数据均来自真实执行结果，不是静态图评分。

|任务|seed|A普通GOLEM|B后验过滤|C语义引导|R随机搜索|
|---|---:|---:|---:|---:|---:|
|随机20%|11|4.306143|4.306143|7.005142|4.306143|
|随机20%|22|7.005142|7.005142|4.484296|4.306143|
|随机20%|33|7.005142|7.005142|7.005142|4.306143|
|相位0|11|3.923142|3.923142|6.575998|3.923142|
|相位0|22|6.575998|6.575998|4.117215|3.923142|
|相位0|33|6.575998|6.575998|6.575998|3.923142|
|相位1|11|4.215794|4.215794|7.817155|4.215794|
|相位1|22|7.817155|7.817155|4.305483|4.215794|
|相位1|33|7.817155|7.817155|7.817155|4.215794|

上述严格质量门队列与固定基线共同队列略有区别，例如回退方法会覆盖线性未覆盖的端点，不能把两张表小数差异当作算法内部精度收益。搜索各机制之间目标完全一致。

另以各seed中四组共有的5/7/6次计分前缀作敏感性比较，三seed平均结果：随机任务A/B 6.527858、C 6.164860、R 5.568807；相位0 A/B及C均5.756404、R 4.807427；相位1 A/B及C均6.646598、R 5.416248。**共同实际评价前缀下C仍未优于R**，不存在用C较少评价次数掩盖公平性问题的结论。

8种合法图×3任务已实际穷举24次；R在三个seed均找到了对应穷举最低误差。输出指纹还识别出部分结构在当前样本上的实际行为等价，不能按节点数夸大搜索空间。本任务只需补齐单一时间恢复能力，物理约束区分度弱，后验过滤并未带来不同结果。C还花费较多提议预算在小空间中重复尝试。本轮证据**不支持语义引导优于普通或随机搜索，也不能证明搜索创新成立**。该空间宜直接穷举/固定规则，不扩图凑创新，不扩大正式GOLEM实验。

### 31.8 冻结方案跨文件复核与失败记录

36次搜索各自选中的图在外部评价前冻结，共6种不同选中图。它们是真实的直接最近邻、直接ffill，以及线性/ffill加仅缺失处回退等组合。全部按原配置在11文件运行，没有逐文件拟合新参数或看结果再换方法。`frozen_transfer_by_search.csv`共792行，保留每个搜索选择、文件及两种评价队列。

剔除开发文件后，10个外部文件×3任务×3seed，每组90条共同队列汇总。A/B各80/90条达到完整数值覆盖，C/R各90/90；A/B的另外10条未被删除，其选中直接ffill图在边缘缺少历史支持，触发完整输出要求，失败及评价分母保留。不能只在A/B成功子集上排名。C/R的完整覆盖说明回退通路能运行，不意味着C具有更低误差。

其他省份可执行性已经真实验证；尚不能声称训练权重迁移或跨区域优势：本轮主要是无需权重的时间插值/回退规则，未使用新的深度模型；多个文件单位与统计口径尚未核实。冻结选择没有外部标签拟合，但31.4节早期暴露使这些结果仍属于探索性复核，而不是全新独立确认。

合并遮挡清单时发现测试阶段没有可选`peak_threshold`列，造成合并CSV列错位；修复为固定10列schema、缺失阈值明确留空。保存原失败清单SHA及修复前代码快照，未覆盖原始数据或源mask。修复只影响交付清单，不改变预测、遮挡位置或RMSE。现已逐行检查18,375,912条合并记录：验证9,840条、批量18,361,332条、测试4,740条，全部列数和阶段有效。重复seed/任务可引用同一原始观测，不能把这些行说成独立环境样本。

### 31.9 验收、代码与复现

最终定向测试为39项核心/兼容回归加1项清单schema测试，共**40通过、0失败、0错误、0跳过**。初次失败日志保留，不抹除历史。未运行无关全量回归。原始11份Excel的SHA256前后相同；独立读取器核对每份前32行及上述极值记录。没有下载数据、修改/移动/删除原始文件、改第三方GOLEM核心或34项冻结池。

两次无缓存冷执行确认同一验证输入及mask产生完全一致的预测，源输入最晚为2021-02-20 23:00，不含测试段；结果SHA保存在`pilot_reproducibility_probe.json`。环境为Python 3.12.14、pandas 3.0.1、NumPy 2.5.3、SciPy 1.18.1、PyArrow 25.0.1、openpyxl 3.1.5、GOLEM 0.5.0，完整版本与指纹见`reproducibility.json`。

|验收项|最终状态|实际证据/限制|
|---|---|---|
|全部Excel只读盘点|通过，11/11|dataset_manifest.csv、输入SHA、独立读取核对|
|单文件试跑后批量时间恢复|已执行，365实体|逐实体预测Parquet、基线指标、mask清单|
|随机/隔小时/6及12小时连续缺失|已执行，条件不足明确不适用|mask_applicability.csv、summary_by_task.csv|
|严格验证输入隔离|V2通过|截断输入测试、冻结记录；V1撤回，历史暴露披露|
|数值DAG及A/B/C/R|已执行36次运行|真实执行轨迹、失败/缓存分别计数|
|跨文件冻结方案|已执行|无外部重新选型；保留不完整覆盖与失败|
|空间留点|未运行/不适用|坐标基准、城市统计支持域证据不足|
|物理单位/来源认证|未通过|仅原数值恢复，不能声明物理浓度精度|
|语义引导搜索创新优势|不获支持|C未优于R，小空间穷举更直接|
|正式空间降尺度/科研创新结论|不具备条件|不把工程闭环或时间恢复当作空间精度、显著优势|

关键代码：`src/ecodata/algorithms/temporal_reconstruction_dag.py`、`tools/prepare_city_hourly_excel.py`、`tools/city_pm25_experiment.py`、`tools/run_city_temporal_search.py`、`tools/finish_city_pm25_experiment.py`、`tools/analyze_city_temporal_transfer.py`及`tests/test_city_pm25_temporal_experiment.py`。

最终交付根目录：`D:\codex\EcoDataEngine\output\city_pm25_reconstruction_20261019\strict_sealed_v2`。其中含用户要求的`dataset_manifest.csv`、`experiment_config.json`、`mask_manifest.csv`、`baseline_results.csv`、`search_runs.csv`、`summary_by_file.csv`、`summary_by_task.csv`、`reproducibility.json`；另有冻结图、完整搜索汇总、共同预算分析、跨文件复核、失败日志、质量记录及测试XML。各阶段预测文件支持Parquet回读。时间任务不伪造规则网格，不额外输出假NetCDF栅格。

以下为本轮已执行流程的复现命令；复现时应使用新的输出目录保护既有证据。准备阶段可直接复用已校验的`prepared/`，无需重新读取全部Excel：

```powershell
# 在 D:\codex\EcoDataEngine，首次提取才执行 prepare；已有完整prepared不要重复提取。
.venv/Scripts/python.exe tools/prepare_city_hourly_excel.py
$env:ECODATA_CITY_RUN='D:/codex/EcoDataEngine/output/city_pm25_reconstruction_20261019/strict_sealed_v2'
.venv/Scripts/python.exe tools/city_pm25_experiment.py pilot
.venv/Scripts/python.exe tools/run_city_temporal_search.py
.venv/Scripts/python.exe tools/finish_city_pm25_experiment.py pilot-test
.venv/Scripts/python.exe tools/city_pm25_experiment.py batch
.venv/Scripts/python.exe tools/finish_city_pm25_experiment.py summarize
.venv/Scripts/python.exe tools/finish_city_pm25_experiment.py audit
.venv/Scripts/python.exe tools/analyze_city_temporal_transfer.py
```

最终回答：11份文件均有合格时间实体；空间实验没有文件达到已核实坐标与统计支持域条件。时间线性插值在多数文件/任务表现最好，长缺失通常更难，极端原始数值会改变误差。自动组合和跨文件执行已经跑通，但语义引导没有胜过随机搜索；没有显著性或跨省精度优势证据。所有可独立完成的数值实验、必要修复、回读及定向测试已完成；剩余限制是原数据定义与任务搜索区分度，不通过修改测试规则、伪造元数据或增加无意义节点获得PASS。

## 32. 真实三源PM2.5融合：原生网格实际处理与正式证据拦截（2026-10-20）

### 32.1 本轮执行结论与范围

**整体任务部分完成，尚未达到用户要求的完整融合实验验收。** 实际修改了数据准备代码、执行真实整月网格处理、修复时间编码问题并运行定向测试；但没有伪造A—E、消融、搜索或测试站指标。原因不是算法跑通即可忽略的问题：最新本地`ground_metadata_status.json`明确写明`coordinate_datum=unresolved`、`formal_training_authorized=false`；现有GEOS层元数据也未提供与近地层对应的直接证据。

本轮指令明确禁止猜测坐标，因此没有沿用第19—20章的近似WGS84探索性开关。历史探索性数值全部保留，但不拿来冒充本轮正式结果。第31章时间任务的搜索结论与本轮不同数据、不同目标，不互相替代。

没有下载数据、训练新网络、安装依赖、修改第三方GOLEM核心或冻结34项候选池，没有删除任何历史数据。保留40训练站/10验证站/10测试站，逐站角色与第18章保存的审计清单完全一致。原始CSV、元数据及两套NetCDF运行前后SHA256一致。

### 32.2 实际来源与数据契约

|数据|本轮直接读取的证据|采用规则及限制|
|---|---|---|
|60站观测|`D:/codex/数据/京津冀_PM25融合测试_201908/01_地面监测站/ground_pm25_201908_60stations.csv`及同目录`ground_metadata_status.json`|已有来源链支持μg/m³、北京时间及标签前一小时均值；坐标基准仍未证实。只解析训练/验证浓度，测试站不解析目标值|
|GEOS-CF|请求目录`background_geos_cf/subset/geos_cf_pm25_201908_bth.nc`|保留`pm25_rh35_gcc`、`ug m-3`、RH35、UTC HH:30小时均值；不改名为地面观测定义|
|ERA5|请求目录`meteorology_era5/anonymous_public/era5_201908_bth.nc`|t2m/d2m=K，u10/v10=`m s**-1`，sp=Pa，tp=m；前五项GRIB为instant，tp为accum且注释为截至valid_time的一小时累计|

请求目录`D:/codex/数据/京津冀_PM25多源融合_201908`中的站点子目录主要为元数据，因此仍复用上述统一测试目录中的实际观测CSV，不重复复制或下载。坐标原数值范围依据原有元数据为114.4548—118.2185°、36.5776—40.3937°；这只能说明数值范围，不能证明基准。

地面时间变换：无时区标签依据已有支持证据按UTC+8解释，转UTC后减30分钟成为前一小时窗口中心。显式带时区的标签先保留其偏移含义，再转UTC，不能再减8小时。窗口边界写入NetCDF的`time_bounds`，缺少真实端点时拒绝生成特征。

GEOS层仍原样保留`lev=72`，其属性同时为`units=millibar`、`long_name=altitude`；原下载脚本使用`lev=slice(0,1)`。仅凭这些信息既不能断言是72hPa，也不能断言已验证近地层。RH35和地面定义的差别可由经验观测模型研究，但不等于物理调湿已解决；直接B基线即使以后可运行，也必须标记为不同定义的背景诊断。

### 32.3 已真实完成的数值处理

新增`src/ecodata/algorithms/pm25_alignment.py`作为现有DAG任务的边界数据准备工具，复用原有`DAGError`、`RawInputQualityGate`和`ResultQualityGate`；不是新算法库或另建融合平台。执行脚本为`tools/run_bth_fusion_evidence_gate.py`。

1. 核对60站角色、坐标稳定性及站点/小时重复，保留原始站点坐标，不指定或转换未经证实的EPSG。
2. 对地面标签、GEOS小时中心及ERA5真实两端整点求交集，得到735个窗口：2019-08-01 00:30至2019-08-31 14:30 UTC。月初超出当前背景覆盖的540条站点记录排除，不补齐。
3. 保持GEOS原生25×24网格、0.25°表示及原始1层，实际生成每个变量441,000个网格—时间值。ERA5与GEOS原有经纬度数组完全一致，使用严格相等检查，没有对站点作空间采样、没有放大网格或隐式重网格。
4. t2m/d2m/u10/v10/sp在两真实整点之间计算HH:30的线性瞬时协变量估计；明确`time: point`，**不叫小时均值**。
5. tp直接选择窗口终点的原始一小时累计，并标记`time: sum`；没有插值降水累计值、平均两个累计值或错误求差。GEOS自身保持`time: mean`。
6. 保留七种变量各自的有效mask，没有填造缺失值。当前735个窗口七种变量有限值比例均为100%，这是网格数值有限率，**不是站点匹配率、仪器QA通过率或融合精度**。
7. 真实写出并回读NetCDF，对数组、层、坐标、窗口、单位和mask逐项比较，并执行已有结果门。首次发现time/time_bounds独立编码可能违反CF一致性的警告，已统一秒级时间基准和日历，复测无该警告；原先两轮目录保留。

最终文件：`output/bth_pm25_fusion_20261020_final/native_grid_covariates_NOT_FUSION.nc`，23,764,440字节，SHA256为`ef0681047357e81fc50e814f5ab1cc7bece7ec62041a4cce2f984ac5ab5e7a7f`。文件属性和文件名均说明**它是对齐协变量，不是融合后的PM2.5预测**，不能交给老师称为本轮融合成果。

|站点角色|存储记录数|有限非负PM2.5|时间范围匹配记录|同时满足时间与数值有效|
|---|---:|---:|---:|---:|
|训练40站|29,746|29,235|29,386|28,878|
|验证10站|7,439|7,329|7,349|7,239|
|测试10站|仅核对站号/角色/时间/原坐标|未解析|不作目标值统计|未统计|

上述数量仍未通过正式空间匹配，不可直接当作最终融合队列。没有根据可预测性或误差筛选评价站点。测试目标未用于分布诊断、拟合、验证或选图；对源文件做字节级校验不等于使用测试标签拟合。

原生窗口数值范围：GEOS RH35 2.2008—872.0667 μg/m³；t2m 278.6286—310.9879K；d2m 270.9829—301.2966K；u10 −14.8250—8.8904m/s；v10 −17.9889—9.0426m/s；sp 82,161.293—101,573.852Pa；tp 0—0.0304573m。风分量有负值是方向信息，未套用浓度非负规则。

原始质量门t2m/d2m/sp通过；GEOS、u10/v10/tp为IQR离群提示WARN，未自动裁剪或删除。七种变量回读结果门没有BLOCK；其无真值提示保留。**这种检查通过不能消除站点CRS或GEOS层定义的阻塞。**

### 32.4 A—E、消融和搜索的最终状态

|请求方案|状态|具体阻塞|
|---|---|---|
|A 地面站IDW/GWR|BLOCKED / 未运行|站点坐标基准无可绑定的权威声明，不能猜测米制距离/投影|
|B GEOS留站背景评价|BLOCKED / 未运行|上述坐标问题；层定义待证实；RH35并非同定义地面目标|
|C GEOS经验偏差校正|BLOCKED / 未运行|不能可靠构建训练站—背景配对，不能凭历史近似坐标的结果作正式拟合|
|D GEOS＋ERA5＋站点|BLOCKED / 未运行|已有UK等接口保留，但站点空间契约未通过，不启动数值预测|
|E 自动组合|BLOCKED / 未运行|不能以结构合法或旧探索性缓存代替当前正式候选实际执行|
|移除GEOS的消融|BLOCKED / 未运行|仍需要真实站点—ERA5空间契约，不能因为少一种背景就绕过坐标问题|
|移除ERA5的消融|BLOCKED / 未运行|同C的站点—背景配对问题|
|普通GOLEM/随机/语义引导、3seed×8次上限|未启动|用户规定稳定融合前不得开展；实际新执行0、缓存复用0、最终测试0|

`method_status_and_metrics.csv`保存每项的原因，RMSE/MAE/Bias/R²/coverage/seconds均留空，而不是写0或拿历史不同样本成绩充数。本轮不能判断哪种源有效、提升百分比、固定流程和自动组合谁更优；三项结论分别为`NOT_EVALUATED`，**未评价不等于无效或被证伪**。

当前阻塞具体指向：能够明确绑定这份2019站点表的坐标基准说明；GEOS指定产品所保留层与真实垂直含义的可引用说明。这些必须是证据，不是改一下配置中的EPSG或把`lev=72`解释成方便运行的层。没有因此新增数据路线、下载模型或要求用户重新选择方向。

### 32.5 测试、保护与可复现交付

定向测试：`tests/test_pm25_alignment.py`、既有`tests/test_golem_pm25_preflight.py`及`tests/test_numerical_dag.py`，最终**26通过、0失败、0错误、0跳过**，耗时3.329秒。覆盖显式/无时区转换不重复偏移、真实HH:30窗口、tp累计终点选择、单位错误拒绝、缺真实端点拒绝、网格差异拒绝、重复时间拒绝、RH35不得改名、缺失mask传递、缺证据拒绝及NetCDF回读。测试fixture不计作真实融合效果。没有运行无关全量回归。

最终证据目录为`output/bth_pm25_fusion_20261020_final/`，包含实验配置及冻结池指纹、真实原生网格NetCDF、源文件与代码SHA、逐变量原始/结果质量JSON、站点角色未变清单、时间匹配表（不含测试标签）、训练验证数值检查、正式执行拒绝记录和A—E/消融状态表。测试XML为`output/bth_evidence_targeted_tests_20261020_final.xml`。完整真实准备执行耗时约2.483秒，非融合或搜索耗时。

来源文件SHA256仍为：地面CSV `f20503e6722e34d3b2b8a6f3995e0bce43c166151d092fc524fbe39d508e6ec1`；GEOS `a7465e77c2df499c71ed3173cba2ebd953753ecb2ba6e6d390c1386de287b43d`；ERA5 `edb509772818df16590553a2bb0dc7237de0f5de63c470809b97f25e58b0ebb1`。完整路径、大小、元数据指纹与代码指纹保存在`run_result.json`。所有历史成果保留，已有三源神经实验或第19章探索性指标没有重新解释成正式盲测；本轮测试站未解封不表示历史从未评价过。

复现命令（指定新输出路径，脚本拒绝覆盖既有证据）：

```powershell
# D:\codex\EcoDataEngine；只读取已存在的源文件，不联网下载。
.venv/Scripts/python.exe tools/run_bth_fusion_evidence_gate.py --output output/bth_pm25_fusion_new_run
.venv/Scripts/python.exe -m pytest tests/test_pm25_alignment.py tests/test_golem_pm25_preflight.py tests/test_numerical_dag.py -q -p no:cacheprovider --basetemp output/pytest_bth_new_run
```

最终验收：真实时间协变量准备、原生网格与mask、代码修复、正式证据拦截、定向测试和原始资料保护已完成。**真实空间融合对比、源贡献消融、自动组合优势和语义搜索创新均未完成，不满足本轮完整实验验收。** 这是明确的数据证据阻塞，不以合成数值、旧探索性结果或擅自假定坐标来获得PASS。

## 33. 第32章继续：真实多源PM2.5探索性融合、消融与算法搜索（2026-10-20）

### 33.1 本轮完成状态与工作约定

本轮依据用户新指令，不再将能够明确管理的实验假设当作停止条件。**已实际执行真实数据融合与对照，不只完成数据盘点。** 复用第32章生成并回读通过的735窗口协变量文件，不重新下载、准备整月数据，不修改任何原始资料。40训练站、10验证站、10历史测试站角色不变。

冻结约定版本为`bth_exploratory_v1`：原始经纬度暂按WGS84解释，距离统一转换至EPSG:32650米制坐标；这是用户授权的操作假设，**不是原站点表坐标基准已核实**。GEOS保留原始RH35标识和实际lev=72及其原始层元数据，不命名为已验证近地面浓度。ERA5瞬时变量和降水累计窗口沿用第32章真实对齐结果，不改变时间定义寻找有利误差。

任务属于**同期空间分析**：每个小时可使用当时40个训练站中的有效观测预测留出站，而不是预测未来。实际每小时有效训练站为27—40个。校正、漂移标准化、秩检查和残差拟合仅使用这些训练站，验证和测试标签不进入DAG查询端口。测试阶段也按冻结规则使用同期训练站，不使用测试标签拟合。这不证明跨时间、跨季节或跨地区泛化。

研究问题预先拆分为：数据源是否贡献信息、自动组合是否优于固定流程、语义引导是否改善搜索。三个问题分别验收，不用结构合法率改善替代精度证明。历史测试站已被评价过，所以本轮均标记为**探索性、回顾性测试，非全新盲测**。第32章的严格正式认证状态没有被改写成PASS。

### 33.2 真实执行路径与算法边界

新增任务适配`src/ecodata/algorithms/bth_exploratory_dag.py`，沿用原`AlgorithmRegistry.register_task_adapters`、类型化图工厂、`UnifiedAlgorithmPipeline.run_dag`、数值DAG执行器及双层质量门。没有重新建设平台，没有改第三方GOLEM源码。现有34项冻结池SHA256仍为`3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28`。

从命名端口和契约自动枚举出7条可执行DAG，调用5类已有冻结组件：`station.pykrige_ordinary`、`station.pykrige_universal`、`station.metpy_linear`、`fusion.quality_aware`的既有线性校正共享组件、`fusion.residual`。其余29项因任务数据不满足相应输入契约而不强行入图；完整34项任务模式记录在`algorithm_task_coverage.json`。未增加全局算法候选，未把RF或IDW冒充冻结组件。

|实验路线|真实实现及输入|数学/数据含义|
|---|---|---|
|A 地面基线|已有IDW权重函数；PyKrige OrdinaryKriging|仅用同期训练站浓度与位置，不用GEOS/ERA5解释变量|
|B GEOS诊断|原RH35场在留出位置的空间查询|跨定义诊断，不等同于同定义地面PM2.5预测，不纳入同语义候选选图|
|C GEOS经验校正|已有`fit_affine_source`，每小时训练站拟合斜率/截距|统计估计地面观测目标，不是已建立的物理调湿公式|
|C扩展|上述校正＋训练站5折交叉拟合残差＋OrdinaryKriging或MetPy线性残差插值|5折按站点隔离，先得到OOF残差，传递同一校正状态与owner指纹后相加|
|D1 去ERA5|PyKrige UniversalKriging，GEOS为specified drift|保留GEOS为特征，不直接把RH35与地面量加权平均|
|D2 去GEOS|同一UK，t2m/d2m/u10/v10/sp/tp为漂移|ERA5为辅助变量，不把温度/降水改名为浓度|
|D3 联合融合|同一UK，GEOS＋上述6个ERA5协变量|仅在训练站做均值/标准差及标签无关的秩检查，使用官方指定漂移接口|
|E 自动组合|同一7图空间中的普通GOLEM、随机搜索、语义引导|真实执行端口、fit/apply状态、残差分支及质量门，不是按算法名称打分|

7图为：普通克里金、三种协变量UK、单独GEOS校正、校正＋两种残差插值。IDW是用户要求的已有独立强基线，**不伪装为冻结池中的节点**。GWR本轮未运行，未把它写成完成；已有IDW和普通克里金满足本轮地面基线比较。质量感知共享线性校正函数被真实调用，但不因此声称通用动态多源可信度权重模型已验证。

查询集合由训练站在米制投影下的凸包确定，域外不外推。原生GEOS网格上的输出也只保留训练支持域，域外NaN并提供`valid_domain`。所有方法统一执行非负投影，逐样本保存投影前原值，不悄悄裁剪后只报告好成绩。

### 33.3 共同样本、覆盖和防泄漏

整月对照覆盖735个UTC窗口：2019-08-01 00:30至08-31 14:30，网格25×24、0.25°。原始数据自然缺失保留；不借值补月初越界窗口，不填造监测目标。

|队列|实际评价窗口|有效留出站|共同样本|空间域外排除|该时段有效目标总数|相对全部有效目标的覆盖|
|---|---:|---:|---:|---:|---:|---:|
|整月验证|735|7/原10站|5,012|2,227|7,239|69.24%|
|搜索/穷举/匹配敏感性|24|7/原10站|163|72|235|69.36%|
|整月回顾性测试|735|7/原10站|5,058|2,181|7,239|69.87%|

没有更换原来的10站名单；3个位置因无外推的几何支持规则未进入评价，不是看误差后筛掉。所有方法在共同支持队列的预测覆盖率均100%、数值失败率0；**不能把这个100%写成覆盖全部10站或整个研究区**。每种方法/阶段的站号、时刻和真值序列指纹完全相同，已自动断言。

整月原生网格实际支持77,693/441,000个网格—时间单元，约17.62%；其余保留mask和NaN，不能填满地图后宣称全域可用。24窗口的敏感性文件仅计算这些窗口，其他时刻不算已覆盖。网格加密、独立高分辨率全场真值和全场RMSE均未宣称。

先执行2个固定窗口小试，9种方法均成功后继续整月。搜索只使用预先按可用性/几何规则从735窗口等间隔选出的24窗口、163条验证样本；不按PM2.5高低或某模型误差选日期。**搜索误差与整月5,012条验证误差是两个队列，以下分别列出，不混用排名。** 固定流程和穷举在搜索队列上同样评价。

所有搜索结束后保存`selection_frozen.json`，才创建回顾性测试解封记录、读取本轮测试目标并计算冻结方案矩阵。各搜索种子的测试表现直接引用同一矩阵中对应图的真实预测，不重复训练或按测试结果换图；后续修复仅为清单编码/统计后处理，没有改变冻结模型、参数或数据规则。

### 33.4 整月真实对照结果

单位为已有证据支持的μg/m³；R²按整个共同队列的SSE/SST计算，不平均小时R²。以下为验证集实际结果：

|方法|RMSE|MAE|Bias|R²|运行秒数|
|---|---:|---:|---:|---:|---:|
|IDW|9.2684|5.9681|−0.4083|0.7617|1.107|
|普通克里金|10.0197|6.4563|−0.1609|0.7215|15.905|
|原GEOS RH35跨定义诊断|128.9123|93.0069|91.9829|−45.0986|0.700|
|GEOS经验校正|12.7069|8.1327|−0.6476|0.5521|17.306|
|仅GEOS的UK|9.8844|6.4061|−0.1753|0.7290|16.505|
|仅ERA5的UK|10.1119|6.4375|−0.3564|0.7164|16.347|
|GEOS＋ERA5联合UK|10.0581|6.4038|−0.3310|0.7194|16.901|
|GEOS校正＋OOF残差普通克里金|10.1324|6.5873|−0.2785|0.7152|21.828|
|GEOS校正＋OOF残差MetPy线性|9.5768|6.5206|−0.5585|0.7456|17.282|

冻结后回顾性测试的实际结果：

|方法|RMSE|MAE|Bias|R²|运行秒数|
|---|---:|---:|---:|---:|---:|
|IDW|7.1397|5.1674|0.8176|0.8494|0.856|
|普通克里金|8.6026|6.1823|0.8604|0.7813|16.068|
|原GEOS RH35跨定义诊断|127.1513|92.6109|92.0693|−46.7725|0.718|
|GEOS经验校正|10.3916|7.5995|0.2661|0.6809|16.786|
|仅GEOS的UK|8.4612|6.0544|0.7993|0.7885|16.239|
|仅ERA5的UK|8.0289|5.7554|1.0180|0.8095|16.235|
|GEOS＋ERA5联合UK|7.9717|5.6958|1.0224|0.8122|16.277|
|GEOS校正＋OOF残差普通克里金|8.6625|6.2752|0.6844|0.7783|21.593|
|GEOS校正＋OOF残差MetPy线性|7.3495|5.3931|0.9053|0.8404|17.726|

耗时包括对应方法在整个固定队列与原生支持网格的执行、预测文件和回读，不含所有数据加载/预处理；同机有不同阶段的运行负载，秒数作工程记录，不能据三次小搜索推断普遍硬件速度优势。

非负投影并非隐含修改：验证点上仅ERA5 UK及联合UK各4次、OOF残差普通克里金1次；回顾性测试点仅ERA5 UK 3次、联合UK 4次，其余0。点与网格的总投影次数另列，不能把网格负值次数写成留出站负值次数。所有原始预测在Parquet中保留。

### 33.5 数据源贡献、消融和稳健性

**融合本身：有相对弱基线的局部收益，但没有超过强基线。** 联合UK相对普通克里金的RMSE在验证集差0.0384（略差0.38%），在回顾性测试改善7.33%；相对IDW则验证差8.52%、测试差11.65%。因此不能把“加了多源”说成一定更准。

移除ERA5、只保留GEOS后：验证UK_geos为9.8844，比联合UK更好；测试8.4612，比联合7.9717差5.78%。ERA5的作用有明显位置/样本依赖，不是全队列一致改善。移除GEOS、只留ERA5后：联合UK相对仅ERA5的改善仅为验证0.53%、测试0.71%。这支持GEOS在该统计模型中可能提供少量补充信息，**不支持其贡献已经稳定证明**。

对31个UTC日期做500次配对日块自助法，只用于探索性不确定性描述，不把小时记录当独立实验重复。联合UK减仅ERA5的RMSE差，验证95%区间约[−0.160,0.072]，测试约[−0.154,0.066]，均跨0。联合UK减IDW则验证约[0.344,1.299]、测试[0.345,1.480]；在本月样本和约定下仍明显落后强基线。不据此宣称跨地区统计显著性。

GEOS校正＋OOF残差线性对比IDW，验证RMSE仍差3.33%、测试差2.94%。从大偏差背景到校正场/残差场的误差降低说明训练站约束有效，不代表证明GEOS原场与监测物理定义一致或lev72已核实。

逐站/逐日诊断已实际计算。联合UK验证平方误差中，1080A贡献约37.81%，2019-08-26贡献约13.82%；测试最高单站1078A约26.72%、最高单日08-26约15.84%，误差确有集中。进一步逐次去掉一个站点后，联合UK相对IDW的RMSE差仍全部为正：验证0.373—1.038、测试0.605—0.975；OOF残差线性相对IDW也全部为正：验证0.141—0.652、测试0.054—0.281。**落后IDW不是只由一个异常站点造成**。这些是诊断，不据此删除该站或重调方案。

### 33.6 空间匹配敏感性

主约定采用同一原生网格的双线性站点查询；敏感性仅改为最近邻，坐标约定、时间、站点划分、训练规则和163条共同验证目标全部保持不变。没有根据更低RMSE改用另一坐标基准或修改主结果。

|方法|双线性匹配RMSE|最近邻匹配RMSE|
|---|---:|---:|
|IDW|10.3345|10.3345|
|普通克里金|10.4220|10.4220|
|仅GEOS的UK|10.5405|10.6062|
|仅ERA5的UK|10.3993|10.4731|
|联合UK|10.5233|10.6638|
|GEOS经验校正|14.0551|13.9158|
|校正＋OOF残差普通克里金|11.0650|10.6729|
|校正＋OOF残差线性|10.7971|10.8829|
|GEOS RH35诊断|127.0650|131.1031|

部分中间排名发生变化，例如仅GEOS与联合UK、两种残差方法的相对次序；IDW仍是这两套匹配下最优基线。说明特征空间匹配方式对具体模型比较有影响，不能只公布最有利抽样方法。本轮没有声称已验证WGS84是真实基准；也没有将采样敏感性替代坐标基准不确定性审计。

### 33.7 真实GOLEM/随机/语义引导对照

复用共享`search_job`和官方`EvoGraphOptimizer`，只补可注入任务Evaluator/case_provider的适配接口及真实随机采样分支。A为五种官方Mutation，C使用既有能力缺口和节点软成本引导；C不多拿数据、模板、候选或更宽质量门。三者使用相同7图语法、相同未完成初始图、24窗口/163目标、RMSE目标、最多8次计分、120次外层提议和120秒上限，seed11/22/33。

每个完整新图都通过真实数值DAG处理；不是按静态图分数取代验证。未完成图统一惩罚，并消耗同样的计分预算。总计72次计分：33次真实新图执行、14次整图缓存命中、25次未完成图；另有2次超预算回调被统一延后，未计算数值、未计入8次上限。原始Objective回调共74次，不能把72次计分说成72次真实数值执行。中间节点缓存另见逐节点轨迹。

|机制|seed|计分|新图实际执行|整图缓存|提议/拒绝|最优验证RMSE|冻结方法|对应回顾性测试RMSE|
|---|---:|---:|---:|---:|---|---:|---|---:|
|普通GOLEM A|11|8|2|0|27/5|10.5405|UK_geos|8.4612|
|A|22|8|3|0|22/6|10.3993|UK_era5|8.0289|
|A|33|8|3|2|28/7|10.3993|UK_era5|8.0289|
|随机R|11|8|3|4|7/0|10.5233|UK_both|7.9717|
|R|22|8|5|2|7/0|10.4220|普通克里金|8.6026|
|R|33|8|5|2|7/0|10.3993|UK_era5|8.0289|
|语义C|11|8|4|1|64/6|10.4220|普通克里金|8.6026|
|C|22|8|4|3|84/15|10.3993|UK_era5|8.0289|
|C|33|8|4|0|60/9|10.5233|UK_both|7.9717|

三seed最佳验证RMSE均值±样本标准差：A为10.4464±0.0815，R/C均为10.4482±0.0660；对应回顾性测试均值A 8.1730、R/C均8.2011。R/C只是所选图的seed排列不同，没有均值精度优势。不能因C某个seed测试误差低就重新选择该seed。

按计分次数统计的完整可执行候选比例：A平均41.67%、C 66.67%、R 87.50%；这是**完整性比例而非全部结构合法率**，未完成图本身可以是合法中间图。成功进入数值执行的33次均无数值失败。C补齐能力缺口的轨迹更多，但平均提议69.33次，高于A 25.67次、R 7次；平均搜索墙钟C 3.574秒、A 2.885秒、R 2.454秒。原始拒绝及开销未隐藏。

7图在同一163条验证样本上全部真实穷举，预测指纹也显示7组不同数值行为，不靠改名凑图。池内穷举最佳为仅ERA5 UK（10.3993），池外独立强基线IDW仍更好（10.3345），固定流程最终冻结为IDW。搜索器没有创造出比同一空间穷举更好的新算法。此空间很小，**优先穷举/固定规则更直接，不支持继续扩大本任务正式GOLEM搜索**。

因此分别判定：多源相对部分基线有条件收益，但未胜IDW；自动组合未胜最强固定流程；语义引导提高部分候选完整性，但未显示相对普通/随机搜索的预测或成本优势。本实验没有证明算法创新成立，也没有因为负结果继续调测试集。

### 33.8 实现修复、自动测试与证据保护

首次小试发现GOLEM图对象不能直接与字符串基线名称比较，触发`str has no attribute root_nodes`；已改为明确的字符串类型判断，失败目录保留在`output/bth_exploratory_fusion_20261020/`。修复后新建`output/bth_exploratory_fusion_20261020_v2/`运行，没有覆盖初次失败。

共享搜索读取含中文路径的JSON时遭遇Windows默认GBK解码异常；已最小修改为显式UTF-8。修复发生在本轮搜索和测试前，旧搜索源文件及协议保存在`pre_utf8_fix/`；既有数值后端未变，已通过的小试、整月验证和穷举不重复计算。

修复协议在控制台传输时又损坏了两个中文路径的**显示标签**，不是数据内容。最终根据预修复原协议中唯一且相同的SHA256恢复路径索引，逐文件重新核验；保存`source_path_encoding_recovery.json`。未修改已冻结协议字节或选择记录，没有按模型结果更换源文件。

后处理发现24窗口队列的覆盖分母误包含了全月域外记录，已按实际评价时刻筛选域外清单，将163条共同样本的总有效目标分母纠正为235，而不是2390；原清单/原汇总保存在`pre_cohort_filter_fix/`。**预测、遮挡/目标、RMSE、图选择均未改变**，新增定向测试防止重现。

最终相关测试为**52通过、0失败、0错误、0跳过**：包含新任务全部7图真实内核执行、官方Mutation补全、标签隔离、当前小时状态归属、5折站点互斥、UTF-8协议与任务回调、SHA路径恢复拒绝数据变更、子集覆盖分母、交付网格时间窗口/mask注解回读，以及原DAG/时间对齐/站点融合兼容测试。Fixture仅用于接口断言；环境指标全部来自真实数据运行。没有运行无关全量回归。

原始站点CSV、GEOS、ERA5、元数据、已完成时间对齐文件和冻结池均核对指纹，未下载、修改或删除。正式坐标基准、GEOS近地层定义、仪器级QA、数据源是否完全独立仍未证实；没有用本轮误差证明这些历史事实。所有质量总状态保留WARN探索性说明，不把数值门通过改写为正式地理准确性通过。

### 33.9 交付、复现与结论边界

代码：`src/ecodata/algorithms/bth_exploratory_dag.py`、`tools/run_bth_exploratory_fusion.py`、`tools/analyze_bth_exploratory_fusion.py`；共享搜索必要适配位于`tools/run_taiwan_composition_search.py`，第三方GOLEM不变。新增测试为`tests/test_bth_exploratory_dag.py`。

最终目录：`D:/codex/EcoDataEngine/output/bth_exploratory_fusion_20261020_v2/`。主要文件：

- `protocol.json`、`fixed_station_split.csv`、`catalog.json`、`algorithm_task_coverage.json`：约定、输入/代码指纹、站点及7图契约。
- `smoke/`、`validation/`、`enumeration/`、`sensitivity_nearest/`、`retrospective_test/`：每种方法真实逐样本预测Parquet、数值结果JSON、依赖轨迹JSONL、实际网格NetCDF及几何排除清单。
- `all_method_metrics.csv`、`metrics_by_station.csv`、`metrics_by_utc_date.csv`、`cohort_coverage.csv`：完整指标及一致样本证据。
- `source_ablation_and_day_block_CI.csv`、`spatial_matching_sensitivity.csv`、`leave_one_station_out_error_diagnostics.csv`、`error_concentration.csv`：消融、匹配敏感性、误差集中与留一诊断。
- `search_runs.csv`、`search_comparison.csv`、`search/*/search_events.jsonl`、`selection_frozen.json`：72次计分、真实/缓存/失败分别计数、冻结选择。
- `grid_readback_and_coverage.csv`、`all_result_quality.json`、`reproducibility.json`及编码/覆盖修复记录：回读、mask、单位、环境和限制。测试XML位于`output/bth_exploratory_tests_final.xml`。

交付读取优先使用各方法目录的`grid_cf.nc`：在原冻结`grid.nc`数值不变的基础上另存明确UTC中心、前后30分钟边界、经纬度单位和`time_evaluated`标记；不覆盖原网格。原GEOS诊断变量明确命名为`pm25_rh35_diagnostic`，不伪装成融合估计。实际估计变量为`pm25_estimate`；WGS84仍在元数据标记为工作假设。新增元数据文件也已实际回读核对。

CPU环境沿用现有Python 3.12环境，PyKrige 1.7.3、MetPy 1.7.1、pandas 3.0.1、NumPy 2.5.3、xarray 2026.7.0；无需GPU或新模型权重。复现时应使用新输出目录；既有阶段有真实结果则复用，不重复下载：

```powershell
# D:\codex\EcoDataEngine；本轮操作约定已明确授权为探索性。
.venv/Scripts/python.exe tools/run_bth_exploratory_fusion.py --output output/bth_exploratory_new --stage smoke
.venv/Scripts/python.exe tools/run_bth_exploratory_fusion.py --output output/bth_exploratory_new --stage validation
.venv/Scripts/python.exe tools/run_bth_exploratory_fusion.py --output output/bth_exploratory_new --stage search
.venv/Scripts/python.exe tools/run_bth_exploratory_fusion.py --output output/bth_exploratory_new --stage sensitivity
.venv/Scripts/python.exe tools/run_bth_exploratory_fusion.py --output output/bth_exploratory_new --stage test
.venv/Scripts/python.exe tools/analyze_bth_exploratory_fusion.py --output output/bth_exploratory_new
```

`analyze_bth_exploratory_fusion.py`有历史编码档案时按唯一SHA恢复路径索引；全新复现无该修复时自动直接验证自己的UTF-8协议，不需要手工复制历史损坏路径。它只读取冻结预测做分析，不重新拟合模型或评价测试选型。

最终验收：真实融合、消融、敏感性、自动组合与3机制×3seed搜索均已执行；本轮冻结后的历史测试完成。**所得结论是有明确假设边界的负/混合结果：当前多源方案未胜IDW，自动组合没有额外精度收益，语义引导优势未获支持。** 不将同期留站分析等同未来预测，不将7个支持站点的成绩外推到全部10站/研究区，不将原生网格估计称为真实更高分辨率重构。52项工程测试通过也不是科研创新已证实。

## 34. 稀疏监测站条件下多源融合价值验证（2026-10-20）

### 34.1 本轮结论与实际范围

**已真实执行：原40训练站内，5、10、20个可用站分别10组固定组合，8种方法，共30组、240个方法案例。未使用原10验证站和10测试站的目标值，也未运行GOLEM。** 本轮是回顾性、假设条件下的空间交叉验证，不是全新盲测、未来预报或高分辨率真值验证。

结果：从20站减少到10站、5站，IDW误差逐渐增大，但联合特征模型仍未反超。GEOS与ERA5在所测试的RF模型内部具有互补信息，不等于多源模型已经超过同期地面站插值。**本次没有找到足以证明融合胜过IDW的稳定使用场景。** 不因负结果更换站点、参数、方法或评价时段。

### 34.2 数据复用、语义和防泄漏

- 地面数据：`D:\codex\数据\京津冀_PM25融合测试_201908\01_地面监测站\ground_pm25_201908_60stations.csv`；单位沿用已有核实与标准化记录的μg/m³，时间解释复用`ground_metadata_status.json`和现有`observation_centers()`。没有重新下载、匹配整月或修改原始CSV。
- 协变量：复用`output/bth_pm25_fusion_20261020_final/native_grid_covariates_NOT_FUSION.nc`，而非已有融合预测。共735个UTC小时中心，2019-08-01 00:30至2019-08-31 14:30。地面按原有北京时间标签、整点前一小时平均与GEOS中心对应；ERA5时间语义沿用已对齐成果，不重新按数组下标配对。
- 准备后的40站×735小时为29,400行，真实有限非负目标28,878行，缺失522行。有效目标范围0～152μg/m³，均值25.388。自然缺失不作为真值、不填目标。
- 输入读取器先判断原冻结`split`；非train行在解析时间、坐标和浓度前跳过。只保留原40站，且与历史`preparation_audit.json`的训练站ID集合逐一核对。文件SHA256校验可读取整个文件字节，但不将封存站目标解析为实验数据。
- 经纬度按WGS84作工作假设，距离统一EPSG:32650米制投影。未声称原坐标基准已经证实。GEOS保留RH35及原lev=72定义限制，不宣称近地面层或已完成湿度物理换算。
- **B项原始GEOS直接浓度误差：NOT_COMPARABLE_NOT_SCORED。** 由于RH35/层定义与地面观测尚未证明等价，本轮不输出直接浓度RMSE；它只作为可校准的候选预测特征。不能把未评价写成0，也不能沿用旧章跨定义诊断分数冒充本轮可比结果。
- 复用的6个ERA5特征为`t2m,d2m,u10,v10,sp,tp`；GEOS特征为`pm25_rh35_gcc`采样值。原始单位及属性保存在`source_variable_metadata.json`。温度、气压、降水等为协变量，不与PM2.5直接平均。
- 15个固定站时位置×7个特征共105项，独立用源文件相邻四格的双线性权重核对采样结果，全部相符。输入原始文件和数值代码SHA在分析结束再次验证不变。

### 34.3 站点组合与统一评价域

固定随机种子：`11,22,33,44,55,66,77,88,99,110`。仅根据坐标采用距离平方加权、无放回抽样；每组先随机选凸包顶点，再倾向选择距离现有点较远的位置。每种站数有10个不同组合，不看PM2.5数值、不按成绩重抽。每个seed中5站⊂10站⊂20站，可用站的整组时间序列一起使用。原固定40/10/10划分未改变；本轮的可用/留出仅是原40训练站内部的探索划分。

每小时只评价真实目标有限、且位于该小时有限可用站凸包内的留出点；不足3个非共线输入点或域外点不外推。**所有8种方法共用该域，即使RF可以在域外生成数值也不额外计入其主排名。** 所有排除项保存在`eligibility.parquet`，不把域内100%预测覆盖伪装为全部留出目标覆盖。

|可用站数|10组真实留出站时目标总数|支持域内评价站时总数|组平均空间支持覆盖率|每组支持留出站数范围|每组支持小时数范围|
|---|---:|---:|---:|---|---|
|5|252,657|98,426|38.96%（16.93%～69.56%）|6～25|726～735|
|10|216,507|149,170|68.89%（59.70%～82.13%）|18～25|735|
|20|144,331|113,587|78.69%（64.75%～84.66%）|13～17|735|

这些总数包含不同随机组合对相同观测的重复使用，**不是相互独立的观测数**。支持面积占40站凸包面积的组平均比例分别57.94%、86.59%、97.45%，说明坐标抽样未全部集中于一角；面积覆盖不等于站时目标覆盖。

另外固定`common_density`队列：每个seed均评价未被其20站选中的同一批位置，并要求被5站的当小时凸包支持。因此同seed在5、10、20站下的目标ID、时间、真实值完全相同，已核对内容SHA。每种站数均为62,661组内站时记录（10组相加，每组2,816～11,907），用于隔离“站数变化”与“评价位置变化”。该队列偏向可被5站覆盖的位置，不外推到全域。

### 34.4 真实调用方法与信息条件

没有新增预测算法、模型架构或注册项，仅新增本轮运行与分析脚本。复用：

|代码/方法|实际用途|数据条件|
|---|---|---|
|既有`idw_weights`，距离幂2|A：同期地面站IDW|每小时只用当时有限可用站观测|
|既有`fit_affine_source`|C：GEOS特征线性偏差校准|仅用选定站整月序列拟合斜率和截距|
|既有`EPATask.fit_rf`，RF_GEOS|非线性GEOS特征校准对照|仅选定站历史目标及GEOS特征|
|同一RF接口，RF_ERA5|D：气象协变量估计|仅选定站目标及6个ERA5特征|
|同一RF接口，RF_BOTH|E：联合特征估计|仅选定站目标及GEOS+ERA5特征|
|上述3个RF＋既有OOF平均站点残差IDW|残差校正敏感性|训练站分组交叉拟合，空间传播每站训练期平均OOF残差|

RF参数固定为既有50棵树、最大深度6、最小叶3、随机种子11、单线程，未针对当前成绩调参。每个训练状态有5折站点GroupKFold，再拟合完整可用站模型。30组×3特征方案=90个状态，含540次森林拟合；另有30次线性校准。交叉拟合的fit/hold站不重叠，任何当前留出站在任何日期的目标均未参与拟合。

选择RF整月拟合是为避免5站单小时高维外部漂移回归欠定；没有降低旧UK组件的输入门槛，也没有将不能执行的UK改名成RF。**本轮RF不是每小时即时融合：它使用所选站整个已有月份的历史信息，包括待评价日期这些输入站的观测；IDW只使用同期观测。** 此信息条件必须披露，因此结果只回答回顾性空间重建，不回答时间泛化。RF预测不输入当前留出站目标。

残差版本沿用原实现的“每站训练期平均OOF误差”，不是逐小时真实残差、不是独立时间预报残差；不将它冒充完整的同期观测引导修正。三种特征方案采用同一学习器与固定参数，能够做条件性的来源消融，但不能据此否定所有可能的融合模型。

### 34.5 真实主结果

下表为10组RMSE均值±组间标准差，单位μg/m³；每个站数内部8方法使用完全相同的域内评价样本。逐组全部指标见`method_metrics.csv`，含RMSE、MAE、Bias、R²、预测覆盖率、失败率、时间及非负投影次数。

|方法|5站|10站|20站|
|---|---:|---:|---:|
|IDW|10.494±0.876|9.553±0.736|8.044±0.239|
|GEOS线性校准|14.829±0.659|15.008±0.145|14.655±0.263|
|RF_GEOS|14.675±0.587|14.711±0.169|14.305±0.282|
|RF_ERA5|13.915±0.743|14.282±0.391|13.661±0.345|
|RF_BOTH|12.709±0.690|12.665±0.387|12.131±0.277|
|RF_GEOS＋OOF均值残差|14.689±0.575|14.631±0.266|14.158±0.308|
|RF_ERA5＋OOF均值残差|14.164±0.844|14.469±0.537|13.709±0.373|
|RF_BOTH＋OOF均值残差|12.879±0.637|12.843±0.542|12.147±0.333|

|站数|IDW：MAE / Bias / R²（组均值）|RF_BOTH：MAE / Bias / R²（组均值）|联合模型相对IDW的RMSE改善百分比|
|---|---|---|---|
|5|7.465 / -0.534 / 0.676|9.274 / +0.438 / 0.528|-21.68%|
|10|6.703 / -1.050 / 0.738|9.051 / -0.679 / 0.542|-33.07%|
|20|5.683 / -0.133 / 0.806|8.627 / -0.423 / 0.559|-50.90%|

改善定义为每组`100×(IDW误差−方法误差)/IDW误差`后取组均值，负值代表变差；不与“先平均RMSE再相除”混用。相应MAE改善为-24.89%、-35.50%、-51.93%。联合RF在三个站数下各0/10组胜过IDW，没有只报告最好种子。

**同一目标密度对照**：IDW的5/10/20站RMSE为10.049、9.154、8.205；RF_BOTH为12.410、12.242、12.101。该对照仍是IDW占优，联合模型随站点减少的相对劣势有所缩小，但没有转为正收益。

### 34.6 不确定性与来源贡献

每种对比进行1,000次配对重抽样：共同重抽UTC日块，并重抽站点组合；同一重抽样中所有组合共用日期抽样，方法始终配对。汇总每组百分比改善的均值，不将几十万相关小时记录当成独立重复。日期块仍可能存在跨日相关，站点组合也重叠，因此以下为**条件性的描述性区间**，不是跨区域独立重复的显著性证明。

|站数|RF_BOTH相对IDW的RMSE改善，95%条件区间|固定目标对照的改善，95%条件区间|
|---|---|---|
|5|-21.68% [-32.53%, -12.19%]|-24.29% [-36.92%, -13.83%]|
|10|-33.07% [-44.80%, -21.75%]|-34.14% [-48.71%, -21.58%]|
|20|-50.90% [-62.73%, -38.18%]|-48.02% [-63.14%, -34.41%]|

预先记录的稳定优势要求为区间下限>0、至少8/10组获益、共同队列预测完整。联合RF对IDW未满足。

来源消融使用同一RF参数及同一队列：

|站数|在GEOS特征上加入ERA5：RMSE改善 [区间]，胜组数|在ERA5上加入GEOS：RMSE改善 [区间]，胜组数|
|---|---|---|
|5|13.40% [9.41%,17.28%]，10/10|8.66% [-0.23%,16.16%]，10/10|
|10|13.93% [10.92%,17.10%]，10/10|11.32% [2.40%,18.33%]，10/10|
|20|15.20% [12.01%,18.44%]，10/10|11.17% [1.35%,18.78%]，10/10|

可支持：ERA5对GEOS单特征模型存在条件性增益；GEOS对ERA5模型在10、20站下也提供条件性增益，5站区间跨0应保留不确定性。**不能支持：这些增益足以战胜IDW，或者GEOS已经等价于真实地面PM2.5。** 静态平均OOF残差校正未形成联合模型的稳定提升，不因其存在就宣称已解决局地污染变化。

附加距离诊断完整保存在`metrics_by_distance.csv`，分0～20、20～50、≥50km。5站条件下≥50km的IDW与联合RF组均RMSE约11.240和12.087，仍未反超。20站≥50km出现8.277对7.999的局部均值，但仅389条组合内目标、涉及9组且胜负混合（5胜4负），不作为稳定适用场景证据，尤其不能只挑这一小子集报告优势。

### 34.7 覆盖、质量、时间与测试

- 所有30组均完成，记录的拟合/预测异常0；8方法在各自统一域内队列的预测覆盖均100%、失败率0。域外目标保留排除原因，不计为预测成功；全体留出目标实际可服务比例仍见34.3节。
- 原始预测另存`*_raw`列。三个残差版本在主队列中分别产生1,329、1,335、1,260次负值，采用全方法一致的非负投影并记录；IDW、线性校准和三个未加残差RF无负值投影。没有隐藏投影或按方法删除难样本。
- 共480份“方法×队列”的数值质量门记录通过有限值、单位和非负检查。**该质量门输入是站时样本数组，其原有CRS为空时PASS并不证明地理精度**；最终`delivery_quality.json`明确总体WARN_EXPLORATORY，坐标基准未证实、RH35/lev72定义未完全对齐。独立留出误差单独计算，不能与数值质量检查混淆。
- 所有30个逐样本Parquet已回读并核对预测值；seed11的5/10/20站另存3份站时NetCDF并全部回读核对，缺测/域外保留NaN。它们是留出站预测，不是连续格网或完整研究区输出，亦不宣称守恒、真实降尺度或模型原生输入验证。
- 每组完整处理平均墙钟约14.37、16.49、21.05秒；30组累计工作时间约519.09秒，以3个进程并行执行，**累计组时间不是批处理实际墙钟**。每组IDW数值块约0.216～0.220秒，线性校准约0.120秒；联合RF块约2.75、3.83、6.01秒。RF与对应残差版本复用同一拟合状态，计时包含共享拟合和成对预测/校正块，不可将这两行相加，也不能据此声称是两条独立运行的精密耗时对比。完整逐组成本保留在原记录。
- 小样本先跑seed11三种站数、各3个真实窗口，随后才运行全部30组。定向回归最终34 passed、0 failed、0 skipped，包含6个新测试及已有EPA任务、时间对齐、质量感知融合测试。未执行无关全量回归。首次小测试因已有`.pytest_cache`权限出现缓存警告，后续禁用pytest缓存并使用独立basetemp，未屏蔽测试异常。
- 新测试覆盖：封存行不解析、坐标选站可复现/嵌套、域外/不足点拒绝、失败预测保留分母、真实既有RF调用与组间OOF隔离、配对重抽样已知误差断言。合成数值仅作单元测试，不用于本章科研指标。

### 34.8 文件、复现与最终判断

唯一新实验根目录：`D:\codex\EcoDataEngine\output\bth_sparse_fusion_20261020`，约100MB量级，未删除或修改任何原始/历史数据。

- `config.json`：查看成绩前固定的变量、参数、语义限制、种子、方法及输入/代码SHA。
- `station_combinations.json`、`stations40.csv`：全部嵌套组合及原40训练站坐标。
- `train40_features.parquet`、`prepared_integrity.json`：735小时×40站的真实匹配协变量及缺失目标，来源为既有对齐数据。
- `run/n{5,10,20}_seed{11,...,110}/`：每组预测、资格分母、逐方法指标、三个RF状态、拟合隔离轨迹、质量门、失败记录和结果指纹；`smoke/`保留试跑。
- `method_metrics.csv`、`summary_by_station_count.csv`：完整480行队列指标及48行跨组均值/标准差，未筛选地区、种子或方法。
- `coverage_by_group.csv`、`relative_to_IDW.csv`：空间覆盖与全部逐组相对变化。
- `metrics_by_station.csv`、`metrics_by_UTC_day.csv`、`metrics_by_distance.csv`：逐站、逐日窗口、距离层诊断。
- `paired_uncertainty.csv`、`source_ablation.csv`：配对不确定性及来源消融。
- `method_availability.csv`：A～E及残差方法状态，包括未作物理浓度直接评价的GEOS原场。
- `source_variable_metadata.json`、`independent_feature_sampling_checks.json`、`delivery_quality.json`、`integrity_and_isolation.json`、`reproducibility.json`：源定义、独立采样核对、总体限制、隔离检查及输入/代码/产物指纹。
- `targeted_tests.xml`：真实34项定向测试记录。

新增代码仅`tools/run_bth_sparse_fusion.py`、`tools/analyze_bth_sparse_fusion.py`、`tests/test_bth_sparse_fusion.py`，复用既有算法实现；未修改注册表、第三方GOLEM或算法数学定义。代码版本以SHA记录，本目录没有可用Git仓库信息，不编造commit。

```powershell
# 项目 D:\codex\EcoDataEngine；复现使用新目录，原输出保持不动
.venv/Scripts/python.exe tools/run_bth_sparse_fusion.py --output output/bth_sparse_reproduce --stage smoke
.venv/Scripts/python.exe tools/run_bth_sparse_fusion.py --output output/bth_sparse_reproduce --stage run --workers 3
.venv/Scripts/python.exe tools/analyze_bth_sparse_fusion.py --output output/bth_sparse_reproduce
.venv/Scripts/python.exe -m pytest tests/test_bth_sparse_fusion.py tests/test_epa_task.py tests/test_pm25_alignment.py tests/test_quality_aware_fusion.py -q -p no:cacheprovider --basetemp=output/pytest_sparse_reproduce
```

最终回答：

1. **多源融合是否超过IDW？** 本轮固定的RF特征融合、线性校准及静态OOF均值残差方案没有；联合RF各站数均0/10组胜IDW。
2. **20→10→5站是否改变结论？** IDW误差上升、多源相对劣势缩小，但无反超；相同目标队列证实不是仅由评价位置变化造成。
3. **哪个来源有增益？** 在同一RF内ERA5对GEOS有条件性贡献，GEOS对ERA5在10/20站也有贡献；这与超过同期地面站IDW是两回事。
4. **何时简单方法足够？** 本研究已有同期观测、位于输入站凸包内的留站估计，IDW是当前更合理、更便宜的基线；不能把此结论外推到完全无同期观测、域外或未来时间。
5. **是否找到值得继续研究的场景？** 已证实协变量间存在互补信息，但未证实能提供超越IDW的稳定应用价值。较远距离有局部信号但样本不足，不据此启动GOLEM扩大搜索。后续若研究，应聚焦“协变量提供的时空变化为何不足以替代同期观测”和“静态平均残差不能表达瞬时局地变化”的证据问题，而非先增加算法数量。

验收：`SPARSE_TRAIN40_EXPERIMENT=EXECUTED`；`ORIGINAL_VALIDATION_TEST_USED=NO`；`NEW_DATA_OR_ALGORITHMS=NO`；`GOLEM_RUNS=0`；`STABLE_FUSION_ADVANTAGE_OVER_IDW=NOT_SUPPORTED`；`SCIENTIFIC_SCOPE=EXPLORATORY_CONDITIONAL_SPATIAL_CV`。没有因原坐标/层定义不确定而停止工程执行，也没有把工作假设升级为事实。

### 34.9 五站融合落后IDW的定点根因检查（2026-10-20）

**本次仅诊断，不改算法、不训练、不调参。** 读取上述10组五站实验的冻结模型、逐样本预测和原40训练站匹配表，核对已有结果指纹；没有访问原验证/测试站目标，没有重新执行30组实验。新增的仅为可复现分析脚本`tools/diagnose_bth_sparse_fusion.py`与2项诊断算术/依赖测试。

#### （1）已证实：当前方法没有在预测阶段显式接收同期站点观测

实际数据流如下，不是所有方案都获得了相同形式的同期信息：

|方法|预测阶段实际输入|站点目标如何参与|
|---|---|---|
|IDW|当小时有限可用站PM2.5、站点和目标坐标|直接插值当小时观测|
|联合RF|目标位置GEOS及6个ERA5特征、已拟合模型状态|选定站整月目标用于拟合模型，并非预测输入中的同期观测通道|
|联合RF＋均值残差|上述RF＋固定站点位置的训练期OOF平均残差|每站整月OOF误差先求均值，再空间插值；不是当前小时残差|

代码证据：`tools/run_bth_sparse_fusion.py:199`将`live.value`送入IDW；`:208`给RF的只有`target[features]`；`:210`校正使用存储的`state['residuals']`。该状态来自`src/ecodata/algorithms/epa_task.py:99-101`，其中`groupby('station_id').mean()`把OOF残差时间序列压缩为每站一个常数，再将时间统一赋为训练结束日期。

进行了**固定模型状态的预测依赖探针**：每个seed选取既有有效时刻序列中固定的首、中、末3个时刻，先重放并逐值核对历史原始预测，再仅在内存副本中将该时刻所有可用输入站观测加1μg/m³，不改变协变量、模型或静态残差。共30个站时窗口、每种方法414个留出目标比较：IDW预测全部加1；联合RF与静态残差版变化严格为0（浮点容差1e-10）。90条“窗口×方法”记录在`frozen_state_observation_probe.csv`。

这证明的是**给定已训练状态时，预测不实时响应站点观测**。不能写成“RF从未用过同期站点数据”：整月拟合确实包含这些输入站当小时的观测；若重新拟合模型，响应可能变化，但本次未重新拟合，也不测该效应。

静态校正的逐站时间标准差最大仅2.78e-15μg/m³，即数值舍入意义下为0；联合RF实际留出误差的逐站时间标准差中位数在各组约10.95～13.50μg/m³。每站常数修正显然不能跟踪这种逐小时误差变化。这是既有组件按其“历史平均偏差”契约运行，并非已发现它算错了OOF；**是当前路线未接通所需的同期观测修正功能，不能靠它的失败否定真正同期融合。**

原项目确实另有`EPATask.residual()`（`epa_task.py:128-143`）读取同期观测减去背景预测，但它要求应用日期晚于模型训练截止日期。本轮RF使用整月训练，不能直接绕过该保护调用它。本次没有删除时间保护、没有用训练内残差伪装独立残差，也没有开发替代算法。

#### （2）有数值支持：现有特征的局地空间信息较弱，但不是完全没有信息

对每组与原主评价完全相同的留出站时，分别比较时间平均趋势和减去当小时空间均值后的空间异常。下面为10组描述性相关系数的均值，不是10个独立地区的复现实验：

|特征/预测|各小时评价站均值的时间相关|同时刻空间异常相关|预测空间标准差/真实空间标准差|
|---|---:|---:|---:|
|原GEOS候选特征|0.746|0.116|不作跨定义浓度幅度比较|
|GEOS线性校准|0.746|0.116|0.439|
|联合RF|0.867|0.244|0.478|
|联合RF＋静态均值残差|0.867|0.234|0.521|
|IDW|0.947|0.472|0.747|

原GEOS与地面目标的总体候选特征相关约0.603，时间趋势存在信息；但站点之间的同时刻差异解释力弱。两个输入来源在复用的对齐网格上间距为0.25°；插值到站点不会自动产生局地污染细节。联合RF的空间起伏明显弱于观测，静态校正也没有补足这一点。RH35/lev72仍未确认等价于地面浓度，本节的原GEOS相关是预测特征诊断，不是浓度准确性认证，也不能把问题直接归因为湿度。

同时已有来源消融仍成立：5站下GEOS特征RF为14.675、ERA5特征RF为13.915、联合RF为12.709μg/m³。因此不能说“背景和天气完全无用”。更准确的说法是：**在当前固定学习器及特征表达下，二者组合有补充信息，但其局地/瞬时信息不足以取代同期站点插值。** 相关低不能证明任何模型都无法提取更多信息，也不能排除当前RF深度、特征表达和训练目标的限制。

#### （3）误差分解：大部分差距表现为当小时共同水平没跟准

对每个seed、每个小时的同一评价站集合，将预测误差精确拆为“该小时平均误差”和“减去该平均误差后的站间误差”，逐组验证MSE严格等于两项均方和：

|方法|总RMSE组均值|小时共同误差分量RMSE组均值|站间误差分量RMSE组均值|
|---|---:|---:|---:|
|IDW|10.494|5.405|8.961|
|联合RF|12.709|8.104|9.729|
|联合RF＋静态均值残差|12.879|8.163|9.890|

联合RF相对IDW的逐组MSE差平均51.137(μg/m³)²，其中共同小时误差增加36.698、站间误差增加14.440，前者占约71.8%。**这只是误差能量的代数分解，不能解释成71.8%的因果归因已被证明来自缺少同期观测。** 但它与输入依赖审计一致：缺少实时观测修正的路线连当前共同污染水平也没像IDW那样跟紧，不只是“小尺度纹理恢复不够”。

附加复用模型的诊断：所选5站的整月训练内RMSE，GEOS-RF约14.487、ERA5-RF约13.299、联合RF约11.352；留出为14.675、13.915、12.709。训练内误差本身仍较高，不是仅到留出站才突然崩溃；这支持继续检查现有特征/学习器的表达限制。训练内分数与留出分数队列不同，且偏乐观，**不是OOF评价或信息上限估计，不能凭该差距宣称排除了欠拟合、过拟合或其他原因**。

#### （4）结论、边界和最小后续方向

|判断|证据状态|
|---|---|
|IDW直接利用同期观测，当前RF没有独立的同期观测输入通路|已证实：代码、冻结状态重放及输入依赖探针|
|当前残差校正仅是静态站点均值，不能跟踪小时变化|已证实：5行残差状态及逐站时间标准差核对|
|原GEOS候选特征对局地空间差异信息较弱|有数值支持：相同评价样本空间异常相关0.116；受定义/模型/采样范围限制|
|多源数据完全没有价值，或真实同期融合必然不如IDW|尚无证据，不得由本轮外推|
|接通同期修正后能追回多少RMSE、能否超过IDW|未验证；本轮没有改路线或生成新的改进分数|

因此，**先修正问题表述：本轮验证了“协变量回归＋可选静态校正”不如同期IDW，并没有充分验证“背景＋气象＋同期站点动态融合”是否不如IDW。** 主要工程缺口是同期观测的显式校正通路未参与本轮预测；背景的局地空间信息弱也是现实限制。两者影响大小不能用现有结果作严格因果拆分。

如下一步开展单因素对照，应优先复用已存在的同期残差处理机制，并先满足其训练/应用时间隔离或合法OOF状态要求；固定当前5站组合、队列和参数，不添加网络、不改变测试集、不借调参追分。本次只指出这个最小缺口，**未实施该功能改变、未进行新精度比较**。

复现命令（分析目录已存在时脚本拒绝覆盖，保护本次诊断证据）：

```powershell
.venv/Scripts/python.exe tools/diagnose_bth_sparse_fusion.py --root output/bth_sparse_fusion_20261020
.venv/Scripts/python.exe -m pytest tests/test_bth_sparse_diagnosis.py -q -p no:cacheprovider --basetemp=output/pytest_sparse_diagnosis_new
```

诊断产物集中保存在`output/bth_sparse_fusion_20261020/diagnosis_20261020/`：`diagnosis.json`、`frozen_state_observation_probe.csv`、`static_correction_audit.csv`、`feature_correlation_by_seed.csv`及汇总、`error_decomposition_by_seed.csv`及汇总、`in_sample_vs_heldout.csv`、`diagnostic_tests.xml`。2项新增诊断测试真实通过，未重新运行既有34项或无关全量测试；原始数据、冻结模型、原预测与算法代码均保持不变。

## 35. LGHAP单源栅格降尺度与自动筛选/组合验证（2026-10-20）

### 35.1 实际完成及结论边界

本轮已真实取得并读取LGHAP AOD子集，完成数据规范化、官方LIIF本地训练、现有算法适配、候选能力筛选、组合生成、验证集穷举选择及冻结后的测试。不是仅提交候选清单。

**自动选择结果：三个尺度均为“双三次插值＋非负回聚合一致性QP”。** 在本轮6个测试区域—日期产品窗口上，相对固定双线性，聚合RMSE下降45.94%、43.32%、27.62%；相对验证集选出的最佳单算法双三次，下降21.77%、10.03%、7.99%。但若人工预先固定同一“双三次＋一致性”流程，输出与自动选择完全相同，**本轮没有证明自动化本身优于这个强固定流程**。收益来自选到了更适合当前退化算子的处理方法，而不是搜索额外创造了信息。

这是融合产品的已知参考场人工退化实验，不是独立地面AOD验证，也不是实际卫星从低分辨率生成新的可信1km观测。仅3个测试地区、2个测试日期，不宣称全国、多年或统计显著泛化。未启动GOLEM、未新增大型模型或更换数据集。

### 35.2 官方数据获取、许可与真实结构

官方来源：[Zenodo记录5652257](https://zenodo.org/records/5652257)，通过`https://zenodo.org/api/records/5652257`取得实际清单；[国内数据中心对应页](https://www.ncdc.ac.cn/portal/metadata/3fc0f146-3f29-4036-934a-16c78c8e569f)用于交叉核查说明，未使用其登录下载。背景出处为[LGHAP数据论文](https://essd.copernicus.org/articles/14/907/2022/)。许可CC BY 4.0，使用时须标明数据来源及作者；下载日期2026-10-20，逐成员UTC核验时刻保存在清单。

2019年官方归档`LGHAP.AOD.D001.Y2019.zip`大小1,223,813,866字节，官方MD5为`c4b13e2e7f64107f8ca824481a258b43`。本轮通过服务实际支持的HTTP206 Range读取ZIP中央目录和指定成员，每个请求核对Content-Range及长度，最多2次尝试。**没有下载整年包，因此没有宣称整包MD5通过**；提取成员由zipfile逐个核对中央目录CRC32，并记录SHA256。不是将成员SHA当作整包MD5。

|实际文件|字节数|用途|
|---|---:|---|
|LGHAP.AOD.D001.A20190101.nc|3,927,994|训练日期1|
|LGHAP.AOD.D001.A20190415.nc|3,891,522|训练日期2|
|LGHAP.AOD.D001.A20190715.nc|4,072,107|验证日期1|
|LGHAP.AOD.D001.A20190815.nc|3,962,747|验证日期2|
|LGHAP.AOD.D001.A20191015.nc|3,456,211|测试日期1|
|LGHAP.AOD.D001.A20191215.nc|3,836,394|测试日期2|

6个日文件合计23,146,975字节，约23.15MB，已设为只读。官方`Data user guide and demo codes.zip`348,990字节完整下载，MD5=`992c61392fbd67944aab29777ca0de31`核对通过。原始文件及官方记录、指南、成员清单、Range轨迹、CRC/SHA位于`data/lghap2019_official_subset/`，未下载全部21年数据。

实际NetCDF为`AOD(lon=6800,lat=4100)`，原始uint16，`scale_factor=0.01`、`add_offset=0`、`_FillValue=65535`。xarray解码后只转换一次，固定原始整数子块与手工解码逐值核对通过。统一为`aod(lat,lon)`，单位`1`（AOD无量纲），不改名成PM2.5。原文件经纬度属性不完整，WGS84根据该官方包的Python/R/IDL示例确定，并将依据保留；**未照抄示例GeoTIFF原点取最小坐标的做法**，本轮使用实际像元中心及推导边界。

指南明确D001为空间0.01°、时间日。这里“1km”为名义分辨率，经度方向米制尺度随纬度变，不能写成全国恒定1000m格网。日标签按文件名`Ayyyymmdd`保存，不擅自转成小时或给未说明的日界时区赋值。波长等本轮未获字段级证据的额外属性不猜测；所有方法使用同一产品，未开展跨产品光谱转换。

### 35.3 冻结划分、区域和退化算子

正式测试前固定以下位置与日期，每个位置提取96×96像元。没有根据方法得分改动窗口或删掉难样本：

|集合|日期|区域名与中心经纬度|
|---|---|---|
|Train|2019-01-01、04-15|京津冀(116.3,39.8)、四川(104.2,30.7)、河南(113.3,34.7)|
|Validation|2019-07-15、08-15|长三角(119.8,32.1)、关中(108.8,34.2)、湖南(112.7,28.2)|
|Test|2019-10-15、12-15|珠三角(113.3,23.3)、武汉(114.2,30.6)、山东(117.1,36.6)|

三集合区域互不重叠，不是相邻像元随机分组；日期也分离。共18个区域—日期参考窗口，均9,216个真实有限产品像元。空间梯度RMS范围：训练0.00881～0.03386，验证0.00548～0.02653，测试0.00668～0.02285（AOD/像元）。测试含较平滑和较复杂结构、高值AOD；没有把这些有限窗口称为覆盖全国全部空间形态。

目标仍为原96×96的0.01°网格；构造64×64、48×48、32×32粗输入，分别为1.5、2、3倍重构，粗分辨率名义为0.015°、0.02°、0.03°。所有格网外边界完全一致，原始浮点中心保留，边界由相邻中心中点推导。

退化算子A为球面面积重叠加权平均，纬向使用sin(latitude)面积权重，经向使用区间重叠；非整数倍率同样通过实际重叠处理，不是抽行或缩放PNG。以本实验中“细像元内常值”作为构造低分辨率观测的工作定义。**这不证明LGHAP原像元或原始传感器具有相同观测点扩散函数，也不将AOD当严格总量守恒变量。**

所有算法取得同一个粗输入c=A·y，预测时参数中只有粗观测、由坐标定义的A和目标坐标，没有细参考值y。原始细参考仅用于训练集训练或对应集合评价。主评价限于各粗网格中心包络内的共同支持像元，全部方法使用同一掩码；整图边缘的传统插值NaN通过明确的最近边界延拓处理，不把延拓当新增观测。回聚合检查采用整个输出和原始A。

1.5/2倍每窗口共同评价8,836像元；3倍受原float32坐标的实际边界关系影响，不同窗口为8,836或8,648，**同一窗口内各方法分母完全相同**。不得把像元/倍率/地区之间相关记录当成独立重复。`case_inputs/`保存全部54个集合—窗口—尺度的参考、粗场、实际评价掩码及稀疏A矩阵，不需根据描述猜测掩码。

### 35.4 候选评估与实际可执行算法

先进候选仅审查6类，优先用现有注册条目/官方源码。接口不满足时直接给出原因，而非无限安装或以插值冒充神经模型：

|先进候选|本轮判定与依据|
|---|---|
|[LIIF官方](https://github.com/yinboc/liif)|复用本地已核验官方commit `4c833f8f9d5d038df3ee2b50ebe649c30e628c0f`，BSD-3-Clause；使用已有CPU设备适配，真实完成AOD本地训练/推理。旧RGB及ERA5气温权重均不用|
|[Thera](https://github.com/prs-eth/thera)|现有条目仍是图像/GPU任务契约，无已验证AOD权重和当前任务适配；按现状排除，未声称官方方法绝对不能在CPU运行|
|[ClimateLearn](https://github.com/aditya-grover/climate-learn)|现有可执行权重仅支持特定ERA5气温K、6×6→24×24；变量、单位、网格、倍率均不匹配，不转用温度权重|
|[FNO官方后端](https://github.com/neuraloperator/neuraloperator)|现有EcoData接口需要站点约束和对应任务训练/验证，不是单份AOD二维输入即可用；排除当前接口，并非判定FNO数学上不能做AOD|
|UNO（同一已有后端）|同样缺该单源AOD接口/权重，未另行训练新网络或修改网络结构|
|[ZSSR](https://github.com/assafshocher/ZSSR)|现有条目仅研究候选，官方当前入口为PNG图像流程，学术非商业许可需遵守；没有当前数值栅格适配，不通过量化PNG伪装成AOD算法|

这一步选择LIIF做有限适配投入属于工程可行性决策，**不是人工预定最后的精度赢家，也不是已公平训练比较了全部6类模型**。最终自动优选只在真正可执行的候选中进行。

本轮4个实际估计器为：现有`spatial.nearest`、`spatial.bilinear`、`spatial.xarray_cubic`、官方LIIF本地AOD模型。前三项仍实际调用原Registry插件和xarray-regrid内核，新任务适配器只补充边缘规则及契约。附加步骤为统一非负投影，或非负回聚合一致性QP。

许可证注意：xarray-regrid官方为[Apache-2.0](https://github.com/xarray-contrib/xarray-regrid/blob/main/LICENSE)；历史`RegridPlugin`有BSD-3-Clause字段，不能将该旧字段当作上游许可证明。本轮未顺手改历史条目，另存`license_clarifications.json`明确纠正依据。CVXPY/OSQP沿用Apache-2.0依赖；LIIF官方许可与源码文件保留，不删作者信息。

LIIF训练：仅Train的6窗口、每窗口3种粗尺度，共18对，使用现有`liif_runtime.model_spec()`（官方EDSR骨干2残差块/16特征和官方MLP），不重写频谱层/隐式网络。单通道标准化后重复为官方3通道输入接口，不做RGB颜色映射。归一化均值0.5592533、标准差0.2234381仅从训练集合计算。Adam学习率0.001、L1、固定300步，每步1024个坐标，种子20261020，CPU 2线程。最后一步训练损失约0.05683（标准化单位）；权重相对相同种子初值的最大变化0.18848，确认不是随机权重冒充训练。实际训练约2.34秒，是轻量可运行原型，不是充分训练的LIIF上限。

主环境缺`tensorboardX`，直接复用已有`.venv-capability-v5`进程解决，没有新装大型依赖。官方模型唯一既有设备适配是`.cuda()`换为输入张量设备，源码SHA逐项核对。训练权重来源为**本地AOD训练，不是官方AOD预训练**；v3复用v2已完成的同SHA权重和同SHA训练窗口，没有重复训练。

### 35.5 自动筛选、组合和数值选择怎样执行

新增`src/ecodata/algorithms/raster_reconstruction_task.py`，使用原AlgorithmSpec、AlgorithmContext、AlgorithmRegistry、AlgorithmSelector.screen及UnifiedAlgorithmPipeline。没有再搭平台，没有改GOLEM源码：

1. 数据/任务描述：连续AOD、无量纲、WGS84、规则中心坐标、缺失状态、目标网格及倍率；`task_descriptors.csv`另存粗场均值、标准差和梯度。当前这些数值统计用于描述，**不是已经学成的逐窗口智能路由模型**。
2. 调用原selector的compatibility筛选，记录每个注册候选的拒绝原因；研究-only/错误单位/缺站点/缺权重/不匹配气温任务不能执行。完整库存筛查不等于深入评估无限多个先进候选。
3. `reconstruction_plans()`根据可执行任务插件类别、输入语义和实际输出契约生成4估计器×2约束共8种计划。约束先在声明的目标输出结构上检查，再由真实上游输出执行验证，不给不兼容节点伪造字段。
4. 单算法方案为估计器＋基础非负检查；可选组合为估计器＋QP。QP实际调用CVXPY/OSQP求`min ||z-z0||², subject to A z=c, z>=0`，只规定与本次粗观测生成算子一致，不宣称AOD总质量守恒。求解失败或误差>1e-5阻止该方案。
5. 先对一个验证窗口的3尺度跑24次冒烟，全部通过；随后验证集6窗口×3尺度×8计划=144次真实执行。原Pipeline的输入质量门、逐步执行和输出质量门均参与，保存每步后端/端口对应数据和输出。
6. 新增最小多样本任务选择器`select_raster_validation()`：只接收stage=validation，要求完整覆盖、相同每窗口评价分母，每尺度按6窗口等权MSE最小选择；并列比较实测成本、再按ID。不是手工从结果里挑图，也没有将原单样本`AlgorithmSelector.select()`伪称为多样本接口。
7. 同时冻结最佳单算法、最佳组合、全池最佳，写入`selected_plans.json`。三尺度全池均选双三次＋一致性，单算法均选双三次。文件时间已核验早于所有正式测试输出。
8. 测试仅按冻结规则执行；为透明比较，预先声明的8方案均各评价一次，但不根据测试再换赢家。每种方案预测回读、指标和质量记录均保留。

仅8图，穷举已经便宜且给出参考上限，未另跑普通GOLEM、语义引导或随机搜索，不人为添加无意义节点。测试后的8图汇总也显示已冻结方案是本轮聚合RMSE最低者；这是事后核对，不是拿测试重新选型。

### 35.6 正式测试结果：全部方案

以下RMSE为6个测试区域—日期等权MSE开平方，单位均为无量纲AOD。符号“＋一致性”均含非负约束；单算法也统一进行了非负检查。共同支持域上覆盖率均100%、负值0、执行失败0。

|实际方法|1.5倍|2倍|3倍|
|---|---:|---:|---:|
|最近邻|0.007662|0.008070|0.011928|
|双线性（预固定基线）|0.004739|0.006321|0.009836|
|双三次（自动最佳单算法）|0.003275|0.003982|0.007738|
|本地训练官方LIIF|0.017270|0.020121|0.026415|
|最近邻＋一致性|0.005717|0.008070|0.011928|
|双线性＋一致性|0.002898|0.004600|0.008323|
|双三次＋一致性（自动全池选择）|0.002562|0.003583|0.007120|
|LIIF＋一致性|0.003150|0.006779|0.012541|

不能因LIIF这次较差而宣称其方法普遍不如插值：本地训练仅6窗口300步，官方模型的能力上限未验证。也不能把LIIF＋QP的改进全部归功于网络，它明显受益于已知退化算子的观测一致性。

|尺度|固定双线性MAE|自动单算法MAE|自动组合MAE|自动组合Bias|自动组合空间相关|相对固定双线性RMSE改善|相对最佳单算法改善|
|---|---:|---:|---:|---:|---:|---:|---:|
|1.5|0.003439|0.002529|0.001994|-2.31e-6|0.998791|45.94%|21.77%|
|2|0.004502|0.003047|0.002789|-3.97e-7|0.997941|43.32%|10.03%|
|3|0.006669|0.005361|0.004938|+1.61e-5|0.993816|27.62%|7.99%|

改善定义为聚合RMSE的相对差；`paired_test_improvements.csv`还保存逐窗口差异，不能与“逐窗口百分比先求均值”混用。每个尺度的6个测试窗口中，组合相对双线性及最佳单算法均6/6降低RMSE；只有3地区×2日期，不作独立大样本显著性宣称。

不同地区/日期的总体表现如下，为该窗口三尺度等权MSE开平方，不隐藏高值或较复杂窗口：

|测试窗口|固定双线性RMSE|自动组合RMSE|
|---|---:|---:|
|珠三角2019-10-15|0.005112|0.003951|
|珠三角2019-12-15|0.009637|0.005865|
|山东2019-10-15|0.003560|0.002789|
|山东2019-12-15|0.004220|0.003039|
|武汉2019-10-15|0.005828|0.003688|
|武汉2019-12-15|0.011613|0.007717|

逐窗口、逐尺度、逐方案完整RMSE/MAE/Bias/R²均保存在`test/metrics.csv`；没有只用易样本计算排名。

### 35.7 结构、峰值和回聚合检查

梯度误差使用两个方向的一阶数值梯度差的合成RMSE，单位AOD/目标像元：

|尺度|固定双线性梯度RMSE|最佳单算法双三次|自动组合|
|---|---:|---:|---:|
|1.5|0.004168|0.003100|0.002369|
|2|0.005428|0.003839|0.003511|
|3|0.007955|0.006767|0.006485|

高值阈值来自训练产品像元90分位0.84，测试前冻结。测试仅2个窗口含该阈值以上像元，1.5/2倍共13,978个、3倍13,861个共同评价高值像元，不能把无高值的4窗口指标填为0。固定双线性高值RMSE为0.006142/0.008301/0.012992，自动组合为0.002873/0.004369/0.009804。

原先固定的单点argmax峰位误差（像元）也完整保留：固定双线性为3.536/1.672/4.603，自动组合为1.672/1.838/0.402。**2倍峰位指标没有改善，不能宣称所有指标都更好。** 数据量化间隔0.01、参考峰值有最多14个并列像元，单点argmax对并列值和浮点细扰动敏感。测试冻结后另做半量化间隔0.005的峰值集合最近距离诊断，单独标为事后敏感性分析，原指标不删除、不用于重新选型；结果在`peak_tie_sensitivity.csv`。

所有一致性组合的整个输出回聚合RMSE约1e-16，且最大绝对误差通过1e-5检查。双线性约0.00291/0.00382/0.00465、双三次约0.00162/0.00167/0.00282。对最近邻的整数倍率，其本身已经近似符合块平均一致性，附加QP几乎没有精度收益，系统没有因此选它或强制所有任务组合。该一致性只验证已知A和c，**不是独立精度证明，也不是AOD总量守恒认证**。

### 35.8 成本、失败修复与定向测试

- 正式验证144次、测试144次均完成；冒烟24次完成，全部NetCDF逐值回读通过，另有8次固定验证窗口的资源专用重放，不读取测试标签重选。
- 冒烟、验证、测试各方案块实测时间累加约2.15、9.56、9.79秒，含质量门、输出和回读，不含下载、整个进程启动等；不能冒充整项任务墙钟。局部训练2.34秒。原型数据准备、日志和资源检查另有实际开销。
- 热状态下单窗口双三次约0.039～0.040秒，自动组合约0.080～0.094秒，推理成本约增加一倍；固定双线性约0.040秒，1.5倍平均0.095秒含该阶段首调用开销，不作精密微基准排名。验证集选型本身多花约9.56秒，已知该任务后直接固定被选流程可避免重复选型。
- 每方案用独立进程在同一个验证窗口2倍任务做资源核对：双三次约0.281秒、QP组合约0.327秒，进程峰值工作集约400MB和424MB（十进制），LIIF组合约467MB。包含导入、状态及求解器常驻内存，不称为纯内核增量内存；CPU时间和前后RSS详见`resource_profiles.csv`。未使用GPU。
- 初次主环境缺`tensorboardX`，改用已有V5进程解决；JSON配置重读出现tuple/list类型比较错误，规范化JSON类型后解决，旧目录留存。最终测试前已使用v3冻结配置；没有通过改尺度或参数追求更好成绩。
- 新测试在独立环境第一次缺src路径而收集失败，补充测试路径后通过。元数据导出遇到AlgorithmSpec无限尺度上限不能写严格JSON，用显式`unbounded`字符串表达，未改变数值实验。官方LIIF的torch.meshgrid未来参数警告和已有flox弃用警告保留，非数值失败。完整记录见`failure_and_fix_history.json`。
- 8项新定向测试通过；与原算法框架一起执行15 passed、0 failed、0 skipped。覆盖整数/非整数重叠算子、常量场、QP回聚合/非负、缺必要输入拒绝、测试记录禁止用于选择、共同掩码不一致拒绝、组合无收益时选单算法、缺AOD权重拒绝。未重跑无关全量回归。

### 35.9 代码、数据产物和复现

本轮新增文件：

- `src/ecodata/algorithms/raster_reconstruction_task.py`：现有插件适配、声明观测投影、官方LIIF权重接口、兼容组合与多样本验证选择。
- `tools/fetch_lghap_subset.py`：官方目录、有限Range读取、ZIP成员CRC/文件SHA核查。
- `tools/run_lghap_reconstruction.py`：固定划分/退化/训练/冒烟/验证/测试入口。
- `tools/analyze_lghap_reconstruction.py`：冻结方案汇总、资源核对、峰位并列诊断与证据核查。
- `tests/test_raster_reconstruction_task.py`：本任务定向测试。

**本轮没有修改原GOLEM、原算法库数值实现或历史PM2.5数据/结果。** 通过任务工厂`registry_for()`把新适配器注册到已有Registry，再走原Pipeline和双层质量门；不是另写优化器。当前是可导入/命令行执行的AOD任务适配，**未宣称已经接入中文网页上传界面的全局自动路由，也未宣称通用于任意环境变量**。

正式实验目录：`D:\codex\EcoDataEngine\output\lghap_reconstruction_20261020_v3`，约35.8MB。v1配置失败和v2训练/试跑证据保留，v3复用同指纹权重；原始日文件单独留在`data/lghap2019_official_subset/`。

关键产物：

- `config.json`、`dataset_manifest.csv`、`task_descriptors.csv`：冻结条件、18窗口指纹、粗场和任务特征；`source_decode_check.json`说明未重复缩放。
- `training_contract.json`、`training_log.csv`、`liif_aod_local.pt`：官方源码commit、本地300步训练、归一化/高值阈值与权重指纹。
- `advanced_candidate_review.json`、`executed_capability_catalog.json`、`license_clarifications.json`：最多6类先进候选的实际状态、可执行契约和许可界限。
- `smoke/`、`validation/`、`test/`：每次真实预测NetCDF、逐步骤质量JSON、筛选原因CSV、失败JSON及逐样本窗口指标，不以缓存或模拟值冒充执行。
- `case_inputs/`：54个集合/窗口/尺度的粗场、参考、同方法共同掩码及实际稀疏退化算子，完整追溯输入与输出。
- `selected_plans.json`、`selection_evidence.json`、`policy_comparison.csv`、`summary_all_plans.csv`：选择理由及全部固定方案/自动方案对比；`paired_test_improvements.csv`保存全部窗口差异。
- `peak_tie_sensitivity.csv`、`resource_profiles.csv`、`failure_and_fix_history.json`：非选择性诊断、资源和失败修复证据。
- `task_tests.xml`、`related_tests.xml`、`final_acceptance.json`、`reproducibility.json`、`artifact_fingerprints.json`：真实测试、验收、环境/数据/配置/模型/结果指纹。

实际环境：Python 3.12现有V5进程，NumPy2.5.3、xarray2026.7.0、PyTorch2.14.1 CPU、CVXPY1.9.3、OSQP1.1.3、xarray-regrid0.4.2。V5复用既有依赖位置，未声称重建了一套完全隔离、不共享包的环境。

复现用新输出目录，不覆盖正式结果；已有原始日文件直接CRC核查复用，不重复下载：

```powershell
# D:\codex\EcoDataEngine
.venv/Scripts/python.exe tools/fetch_lghap_subset.py --output data/lghap2019_official_subset --dates 20190101 20190415 20190715 20190815 20191015 20191215
.venv-capability-v5/Scripts/python.exe tools/run_lghap_reconstruction.py --output output/lghap_reproduce --stage train
.venv-capability-v5/Scripts/python.exe tools/run_lghap_reconstruction.py --output output/lghap_reproduce --stage smoke
.venv-capability-v5/Scripts/python.exe tools/run_lghap_reconstruction.py --output output/lghap_reproduce --stage validation
.venv-capability-v5/Scripts/python.exe tools/run_lghap_reconstruction.py --output output/lghap_reproduce --stage test
.venv-capability-v5/Scripts/python.exe tools/analyze_lghap_reconstruction.py --output output/lghap_reproduce --resources
.venv/Scripts/python.exe -m pytest tests/test_algorithm_framework.py tests/test_raster_reconstruction_task.py -q -p no:cacheprovider --basetemp=output/pytest_lghap_reproduce
```

### 35.10 最终验收与研究问题回答

|研究问题|实际结论|
|---|---|
|能否自动筛出合法且效果良好的方法？|本任务已验证：契约筛选、真实执行、仅验证选择并冻结，全池8图穷举，拒绝不匹配气温/RGB/需站点接口|
|组合是否优于最佳合理单算法？|在本次参考产品与声明退化条件下，双三次＋一致性优于最佳单算法双三次；3尺度×6测试窗口均降低RMSE|
|自动筛选是否优于固定方案？|优于预先固定双线性；不优于人工固定同一双三次＋一致性。自动化没有获得额外输入信息，也未证明方法学创新独立成立|
|不同尺度/区域是否稳定？|本次3尺度选择一致、6测试窗口RMSE获益；样本小且相关，不能推广到全国多年。峰位单点指标并非全尺度更好|
|收益是否值得成本？|若追求该退化模型下RMSE与回聚合一致性，约增加一倍单窗口成本可获得约8%～22%相对最佳单算法的RMSE改善；若任务已知，固定选中方案更省选型开销。不存在无条件效率优势|
|是否需要进化搜索？|仅8合法流程，穷举已完成，当前没有运行GOLEM的必要，未人为扩空间|

最终状态：`DATA_SUBSET=VERIFIED_MEMBERS`；`OFFICIAL_LIIF_LOCAL_TRAINING=EXECUTED`；`AUTOMATIC_SCREENING=EXECUTED`；`AUTOMATIC_COMPOSITION=EXECUTED`；`VALIDATION_SELECTION=FROZEN_BEFORE_TEST`；`REAL_TEST_COMPARISON=EXECUTED`；`GLOBAL_UI_ROUTING=NOT_IMPLEMENTED`；`INDEPENDENT_OBSERVATIONAL_ACCURACY=NOT_VALIDATED`；`UNIVERSAL_AUTOMATION_INNOVATION=NOT_PROVEN`。本轮到此结束，不继续扩数据或算法。

## 36. 三类合成栅格任务的输入驱动自动筛选与组合验证（2026-10-20）

### 36.1 本轮完成范围与证据边界

本轮已实际修改项目代码、增加正式中文上传页面、完成小规模试跑及全部预定样本的数值实验。不是再次运行PM2.5，也没有下载数据、训练大型模型、修改GOLEM或扩充冻结34项搜索池。

核心区别：第35章依赖高分辨率验证参考选型，本轮新增的主模式在处理未知上传文件时**不接收高分辨率参考**，只使用粗网格特征、结构化任务要求和开发集冻结的路由模型。自由文本只记录，不把任务名称或A/B/C标签转成指定算法。

所有精度均相对于本轮**合成参考场**，并非独立环境监测精度。已证明的是受控分布内的输入驱动路由和组合效果；没有证明任意环境变量、真实卫星场、全新分布或通用算法搜索创新。

输出根目录：`output/three_task_routing_20261020/`。180个上传文件来自90个独立参考场，每个参考场生成两个尺度。主实验真实执行1,440个候选流程，1,440份NetCDF均回读，质量门均PASS，全部使用相同的10,000像元评价分母，无执行失败、无缩小队列排名。开发试跑12例随后作为开发缓存复用，未冒充12次新探索。

### 36.2 数据、划分及预先固定规则

|项目|固定实现与实际数量|
|---|---|
|目标/输入网格|100×100；50×50（2倍）及40×40（2.5倍）|
|坐标与空间支持|明确声明EPSG:4326；110～111°E、30～31°N；统一外边界，像元中心坐标；静态合成场，无真实观测日期|
|单位/语义|无量纲`1`、连续非负`field`、像元均值；不是PM2.5或AOD观测|
|退化算子|经向球面面积权重×纬向重叠权重的稀疏矩阵，粗像元权重和为1；不是简单抽点|
|分块场A|随机分块和幅值；边界为0.1度整数倍，同时与两种粗网格对齐|
|平滑场B|正背景、随机梯度及1/2/4阶低中频变化|
|峰值场C|平滑背景、不同宽度的高斯峰；一半额外含单细像元尖峰|
|每类划分|12开发场、6验证场、12测试场；两尺度分别形成24/12/24次上传，共60次/类|
|独立性|开发/验证/测试种子隔离；同一参考场两尺度不作为两个独立统计重复|
|随机种子|基准20261021；类别、划分及索引固定偏移，详见生成代码与私有评价清单|
|评价掩码|所有参考像元均有限，固定全场100×100；不仅评价各方法容易预测的内部像元|
|冻结|`protocol.json`在算法实验前生成；开发模型与固定流程在验证/测试前冻结；测试前另存`pretest_freeze.json`|

生成器、面积算子及回读均已执行检查。窄峰还保存了“不同细场、相同粗输入”的直接证据：在同一粗像元交集内移动尖峰，两种粗输入差异均小于1e-10。由此不能唯一确定尖峰细尺度位置；不会要求算法从同一输入恢复两个不同细场。

主评分预先规定为：

`RMSE/s + 0.3×梯度RMSE/s + w边界×边界带RMSE/s + w峰×峰幅误差/s + 0.5×w峰×峰位误差/对角线 + w一致性×粗尺度RMSE/s`，其中`s=max(std(实际粗输入),1e-8)`。

任务要求是用户可填写的偏好，不是算法标签：边界任务`w边界=1`，峰值任务`w峰=1`，连续场观测一致性优先级为1，其余一致性优先级0.3。三类均保留RMSE等独立指标，不用复合分数替代原始误差。非负是硬约束；一致性在通用请求中是明确的偏好，QP流程自身必须满足最大误差1e-5。这里的一致性不是未经论证的环境总量守恒。

### 36.3 正式平台入口与职责隔离

新增现有Streamlit侧栏入口“单源栅格重构”，而非另建平台：

`上传NetCDF → recognise_file → 物理元数据白名单 → RawInputQualityGate → 粗输入特征 → Registry契约筛选/组合 → 冻结路由 → UnifiedAlgorithmPipeline → ResultQualityGate → NetCDF及决策记录下载`。

批量实验使用真实Streamlit `UploadedFile`对象，调用与页面完全相同的`process_raster_upload`处理入口，不直接调用独立插值脚本来冒充平台流程。另通过AppTest实际运行三类文件的页面解析、按钮、自动路由、计算和下载生成。AppTest的上传传输由测试夹具提供，**没有声称完成浏览器网络上传的人工UI测试**。

选择与评价隔离措施：

- 平台接口没有reference、task_id、family或expected_winner参数；只接收上传文件、结构化任务要求、冻结模型。
- 上传元数据仅白名单传递CRS、单位、数据性质等物理信息；原文件名只作审计，不进入特征。
- 特征为粗场均值、标准差、幅差、梯度均值/95分位、平坦像元比例、边缘集中度、曲率、局部极值比例、峰/中位数差、唯一值比例和倍率，共12项。
- 执行边界启用Python文件访问审计，禁止读取`sealed_reference`；主实验180次平台调用无参考目录访问尝试。针对性测试确证访问会被拒绝。
- 参考场只由外层独立评价器在路由和执行返回后打开；执行器质量门不接收参考，因此其`accuracy_assessed=False`符合预期。参考精度在独立评价器计算。
- 标签注入、文件改名及自由文本改成“预期赢家”均不改变特征或选型；相同冻结模型的预测逐值可复现。

现有适配器仅作必要的变量参数化，保留旧AOD默认参数；本地LIIF权重仍限定无量纲AOD，不进入本轮合成场候选。第35章旧源文件另存于本轮`source_before/`，旧实验和输出未覆盖。

### 36.4 候选与自动组合机制

采用4个真实数值估计器，各接强制非负投影或非负＋粗观测一致性QP，共**8个流程**。所谓“单算法”指一个空间估计器加必要非负检查，不含一致性QP。

|估计器|真实实现/作用|限制|
|---|---|---|
|最近邻|复用已注册空间插件|保留块状结构，但不能恢复平滑细节|
|双线性|复用已注册空间插件|平滑且成本低，可能削弱边界与峰值|
|双三次|复用已注册xarray/SciPy插件|局部高阶重构，可能过冲；不能凭空辨识窄峰|
|PCHIP|新增唯一形状保护候选，直接调用SciPy官方`PchipInterpolator`，固定先纬向再经向|依据观测割线构造分段三次，保护已观测形状；不是隐藏细峰恢复器，轴序固定|
|一致性QP|复用CVXPY/OSQP实现：最小化与初始重构的平方距离，约束`z≥0, Az=c`|只对声明的退化算子一致，不声称恢复额外信息或普适守恒|

PCHIP原理与接口来源：[SciPy官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)，BSD-3-Clause。没有重写第三方核心算法。

组合通过现有AlgorithmSelector筛选和Registry生成，先检查变量、单位、二维坐标、倍率所对应的目标网格及算子形状，再在真实中间输出上复核。非兼容候选保留拒绝原因。任务级Registry不改冻结GOLEM候选池。

主路由采用最小可用的开发集元回归：StandardScaler仅在72个开发上传样本拟合，KNN回归（k=5、距离加权）预测8流程×6误差分量，再按当前用户偏好求分数。只在预测分数差≤0.002时优先声明成本较低的流程。没有A→最近邻、B→PCHIP、C→双三次的分支规则，也没有将验证或测试最优标签输入主路由。

模型SHA256：`a3602ed14075fbaa75c6a5666c04d99dae60ce4b1c805bf43a8f5b291f2ae851`。

另保存“验证真值辅助选型”：仅用验证样本按需求权重选固定流程，单独评价，不进入主路由。事后oracle仅由独立评价器使用测试真值逐样本选取评分最低流程，不是可部署方案，也不是RMSE单指标下界。

### 36.5 全部候选的独立测试结果

下表为每类12个独立测试场×两尺度，统一全像元队列的平均RMSE（无量纲）；全部流程覆盖率100%、失败率0%、负值0。

|固定流程|A分块场|B平滑场|C峰值场|
|---|---:|---:|---:|
|最近邻＋非负|约9.05e-14|0.006204|0.032877|
|最近邻＋QP|约机器精度|0.005747|0.031560|
|双线性＋非负|0.065522|0.001892|0.023205|
|双线性＋QP|0.047000|0.001125|0.019667|
|双三次＋非负|0.064085|0.001358|0.020054|
|双三次＋QP|0.058760|0.001070|0.018844|
|PCHIP＋非负|0.044969|0.001108|0.020614|
|PCHIP＋QP|0.032223|0.000710|0.019353|
|自动单算法|约9.05e-14|0.001108|0.020960|
|**自动筛选/组合**|**约9.05e-14**|**0.000710**|**0.019226**|
|开发集冻结的全局固定流程：双线性＋QP|0.047000|0.001125|0.019667|

自动方案独立指标：

|任务|MAE|Bias|梯度/结构及补充结论|
|---|---:|---:|---|
|A|1.51e-14|1.30e-15|边界带RMSE约2.09e-13，边界距离0像元；没有不必要平滑|
|B|0.000337|2.33e-10|平均RMSE比最佳固定单算法PCHIP降低约36.0%；组合在两尺度均有收益|
|C|0.001173|4.79e-09|总体平均RMSE并未超过最强固定双三次＋QP；不能只与较弱基线比较|

所有逐例RMSE、MAE、Bias、梯度、边界、高值、峰位、一致性、覆盖率、失败及输出引用均在机器可读CSV/JSON；均值、标准差、最小值、最大值及尺度分组在`test_summary.csv`，没有只挑最好种子。

### 36.6 自动选择是否真正随需求和输入变化

|任务/尺度|自动选择次数|解释与验收|
|---|---|---|
|A，2倍及2.5倍|最近邻＋非负，24/24|从输入的平坦度、边缘和开发损失预测中选到简单方案；与复杂QP重构同为机器精度，不为零收益额外组合|
|B，2倍及2.5倍|PCHIP＋QP，24/24|预测的平滑重构误差和一致性偏好支持组合；不是按名称查表|
|C，2倍|双三次＋QP，12/12|实际预测高值及梯度损失，选择高阶＋一致性|
|C，2.5倍|双三次＋QP 5/12；双线性＋QP 7/12|同一任务描述也出现不同决策；其中一些选型不如最强固定方案，应保留而非按测试结果修正规则|

非整数尺度同样按面积重叠退化，不用整数像元复制假装2.5倍任务。自动方案2倍/2.5倍平均RMSE分别为：A约机器精度；B 0.000548/0.000871；C 0.018208/0.020244。

### 36.7 配对不确定性、最强固定流程及净收益

统计单位是参考场，不是像元或同场两个尺度。每类12场，两尺度先在场内合并；整体比较对三类分层重抽，保持每类12场。固定种子、2,000次场级配对bootstrap，报告逐项95%百分位区间；未做多重比较校正，不将这些区间包装成全面显著性证明。

差值均为“自动－对照”，负值更好：

|比较|指标平均差|95%配对区间|解释|
|---|---:|---|---|
|B自动组合－最佳单算法PCHIP|RMSE -0.000399|[-0.000431,-0.000367]|平滑场组合有可重复净收益|
|C自动－最佳固定组合双三次＋QP|RMSE +0.000382|[-0.000039,+0.000850]|没有证据称自动方案优于该强对照|
|C自动－同一最强固定组合|任务分数 +0.000400|[-0.044227,+0.051991]|近似持平且不确定，不能宣称峰值选型创新成功|
|整体自动－开发集冻结固定流程|RMSE -0.015952|[-0.016907,-0.014980]|在该预定任务混合上优于固定双线性＋QP|
|整体自动－测试事后最强全局固定流程：最近邻＋QP|RMSE -0.005790|[-0.007178,-0.004413]|没有只挑弱基线；此强对照由评价器事后计算，不参与部署选择|
|整体自动－同一最强全局固定流程|任务分数 -0.116779|[-0.159902,-0.078594]|评分约降低5.12%，RMSE约降低46.56%；仅限本轮合成任务分布|
|整体自动－验证真值辅助选型|RMSE +0.000127|[-0.000010,+0.000278]|主路由没有超过具有验证参考的辅助模式|
|整体自动－事后评分oracle|任务分数 +0.010275|[+0.000787,+0.024113]|仍有选型遗憾；oracle仅是评价参考|

整体自动平均RMSE 0.006645、任务分数2.165632；事后最强全局固定流程为0.012436、2.282411。平均评分受任务偏好及窄峰幅差影响，必须结合各任务的原始误差阅读；不能把一种任务混合上的收益外推到任意用户数据。

### 36.8 峰值信息限制与失败案例

|C测试子组|独立场数/上传数|自动RMSE|峰幅绝对误差|峰位误差（目标像元）|
|---|---|---:|---:|---:|
|可分辨宽峰|6/12|0.002361|0.012674|0.166667|
|额外含不可充分辨识窄峰|6/12|0.036091|2.531140|32.012277|

窄峰明显没有被准确恢复；粗输入只保留面积平均信息，不包含唯一细尺度定位依据。不能把宽峰的成功解释为能恢复任意尖峰。宽峰组也不能证明自动路由优于最强固定组合，子组仅6个独立场。

PCHIP保护的是已观测局部形状，不是保证最大恢复幅值；在C上双三次＋QP更强。在A上给平滑估计器增加QP会产生局部振荡/过冲，约束通过也不等于参考精度好。选择器在A避开此类组合，但在C仍会选错部分流程；未查看测试误差后重训或改阈值。

特别说明：A的最大值通常占一整块，用`argmax`任选一个像元会因浮点并列而显示非零“峰位误差”，这在A不具定位意义，**不用于A验收**。A以实际要求的边界指标验收。原始诊断列保留，不用无意义峰位数字声称重建失败。oracle按任务分数选择，故其RMSE也不一定低于按该分数路由的另一方案。

### 36.9 质量、时间与资源成本

- 1,440/1,440候选输出质量门PASS，全部回读且有限、非负，预测覆盖100%。所有QP流程回聚合通过1e-5最大绝对误差限制；未QP流程实际一致性误差保留。非负投影共修正256个负像元（所有划分合计），没有静默忽略。
- 主实验全部候选流程累计计时约39.56秒，不含导入、数据生成和文件读写；测试单流程均时：PCHIP 1.52ms、最近邻9.18ms、双线性9.45ms、双三次8.62ms，接QP后约41.82/45.75/48.12/53.03ms。顺序执行和预热条件可能影响短时测量，不能据毫秒差作稳定硬件排序。
- 为补充资源证据，另以开发样本运行48个流程（每类一个场×两尺度×8），不使用测试或参考值，不更改路由。5ms采样进程RSS，平均采样峰值约459～483MB；QP平均增量约17～20MB。含Python/依赖和分配器缓存，不是独占算法显存或精确瞬时峰值。
- 再以同6个开发上传测实际自动部署路径，只执行选中流程：冷启动约3.06秒，后5例约0.107～0.148秒/文件，已含解析、特征、Registry筛选、路由和计算。全候选基准耗时不能冒充自动部署耗时。
- 平滑场为约36% RMSE改善和一致性付出约40ms QP核成本；在本CPU小网格上可接受，但相对PCHIP核成本约27倍，另有约0.1秒入口/路由开销。A选择简单方法确实减少无益QP；对已知固定任务，直接固定最佳流程仍更省路由成本。未证明平台在所有情形下同时更快、更准。

### 36.10 定向测试、实现状态及历史保护

已完成相关定向回归27项全通过（新增上传/隔离/生成/页面12项＋原栅格8项＋现有算法框架7项）；随后增加3项冻结路由复现/改名不变测试，3/3通过，共30个不同相关测试通过，无失败、无跳过。另保留三类页面专项AppTest的3/3记录（与27项中的页面测试重合，不重复计数）。

覆盖：两尺度面积和分块对齐、实际8流程执行、非负和一致性、窄峰不可辨识证据、缺单位拒绝、隐藏目录访问阻止、属性/文件名标签不入特征、正式侧栏入口、页面按钮与下载、同一输入/模型逐值复现、旧AOD适配与通用Registry/Pipeline回归。只有flox/NumPy timedelta弃用警告；首次测试另有原项目pytest缓存目录权限警告，后续禁用缓存provider并使用独立basetemp，未屏蔽失败或质量异常。

未运行无关全量回归、GOLEM搜索或旧PM2.5实验；本任务仅8个可穷举流程，不需要人为引入进化搜索。正式页面目前限定正方形、完整二维经纬度连续栅格，边长4～200；无单位、缺失像元、非兼容CRS等不能在本路径静默修补。真实遥感复杂投影/缺测尚未验证。

|功能|状态|
|---|---|
|正式NetCDF上传解析、Registry筛选、组合、执行、双层质量门及下载|已实现并端到端验证|
|不读取高分辨率参考的主路由|开发集冻结；本轮独立合成测试验证|
|验证参考辅助模式|单独实现、冻结并评价；不是未知用户输入的自主路由|
|各流程穷举及事后oracle|已执行；仅作独立评价，不进入路由|
|简单任务避免无益复杂组合|本轮A已验证，不能泛化为所有简单任务|
|平滑场组合优于最佳单算法|本轮B已验证|
|峰值场自动路由超过最强固定组合|未证明；C结果近似持平且有选错|
|从同一粗场唯一恢复未观测窄峰|不可辨识，本轮未实现也不作不合理要求|
|真实环境精度、跨生成器/分布外泛化、通用科研创新|尚未验证|

历史数据、实验和配置未删除。共享适配器修改前快照保存在本轮`source_before/`，必要代码当前版本与SHA256在`reproducibility.json`；执行版本另存`source_executed/`。不将新功能状态反写成第35章当时已具备的功能。

### 36.11 代码、产物与复现

关键代码：

- `src/ecodata/interface/raster_upload.py`：正式共享上传处理、输入特征、能力匹配、开发路由及真实执行。
- `src/ecodata/interface/raster_reconstruction_page.py`、`src/ecodata/interface/app.py`：中文入口及结果下载。
- `src/ecodata/algorithms/raster_reconstruction_task.py`：复用空间算法适配、面积算子及一致性QP；保持旧AOD默认接口。
- `tools/generate_three_task_rasters.py`：生成器；私有参考不传给平台。
- `tools/run_three_task_routing.py`：实际UploadedFile客户端、参考隔离、开发拟合、冻结、全样本执行和独立评价。
- `tools/summarize_three_task_routing.py`、`tools/profile_three_task_upload.py`：仅读已执行结果分析及有界开发资源测量。
- `tests/test_raster_upload_routing.py`：本轮定向测试。

产物均在`output/three_task_routing_20261020/`：

|文件/目录|作用|
|---|---|
|`protocol.json`、`pretest_freeze.json`、`frozen_selection.json`|实验预设、测试前冻结及开发集最佳固定流程|
|`uploads/`、`dataset_manifest.csv`|普通用户粗网格输入、任务请求、来源与SHA256|
|`sealed_reference/`|生成器/独立评价器专用参考、私有类别及不可辨识证据；平台不得访问|
|`router.joblib`|仅开发样本训练的冻结路由模型|
|`runs/*/trace_and_quality.json`|完整输入摘要、特征、候选及排除理由、决策、8流程逐步质量和成本|
|`runs/*/candidate_*.nc`、`metrics.json`|1,440份可回读输出、SHA256及独立评价|
|`all_metrics.csv`、`comparison_with_validation_assisted.csv`|各例、固定/自动/oracle及单列验证辅助模式|
|`test_summary.csv`、`subtype_summary.csv`、`selection_counts.csv`|两尺度、各任务、峰值子组的分布与实际选择|
|`full_paired_intervals.csv`|场级配对差异及95%区间；明确事后对照|
|`pipeline_times.csv`、`resource_profile.csv`、`deployment_profile.csv`|执行时间、开发资源和真实单选部署成本|
|`quality_readback_index.csv`、`execution_log.csv`、`reproducibility.json`、`final_audit.json`|质量、回读、缓存/新执行、代码指纹及最终审核|
|`final_acceptance.json`|独立复核180个输入和1,440个输出SHA256、预测试模型/协议未改变，以及30项不同相关测试全通过|
|`targeted_tests.xml`、`repro_tests.xml`、`ui_tests.xml`|真实pytest结果；重合测试不重复计数|

本轮目录约172MB量级（含全部输入、参考、输出与证据；最终目录大小随源码快照和测试记录略增）。

在新目录复现（生成器拒绝覆盖已有protocol，保护旧实验）：

```powershell
.venv/Scripts/python.exe tools/generate_three_task_rasters.py --output output/three_task_reproduce
.venv/Scripts/python.exe tools/run_three_task_routing.py --output output/three_task_reproduce
.venv/Scripts/python.exe tools/summarize_three_task_routing.py --output output/three_task_reproduce
.venv/Scripts/python.exe tools/profile_three_task_upload.py --output output/three_task_reproduce
.venv/Scripts/python.exe tools/profile_three_task_upload.py --output output/three_task_reproduce --deploy-only
.venv/Scripts/python.exe -m pytest tests/test_raster_upload_routing.py tests/test_raster_reconstruction_task.py tests/test_algorithm_framework.py -q -p no:cacheprovider --basetemp=output/pytest_three_task_reproduce
```

测试中的页面/冻结模型集成案例默认核查本轮实际交付目录；新目录实验可由共享上传接口调用其`router.joblib`，页面也可填写该模型路径。环境沿用项目`.venv`，使用CPU、NumPy/xarray/SciPy、scikit-learn、CVXPY/OSQP、Streamlit、h5netcdf、psutil；不新增大型运行环境。

研究流程采用证据分层、预设划分和配对统计约束；流程辅助工具来源：[Scientific Agent Skills（2026）](https://arxiv.org/abs/2609.00065)，不将其作为数值算法或精度证据。

### 36.12 最终六项回答

1. **能否只凭真实可用输入与任务需求选型？** 可以；正式入口不接收参考，输入特征＋开发冻结模型在本轮独立合成测试已运行。分布外可信度尚未证明。
2. **简单任务能否避免复杂处理？** 本轮对齐分块场24/24选择最近邻＋必要非负检查，无收益QP被避开；不等于对所有边界场都能完美判断。
3. **组合是否优于最佳单算法？** 平滑场成立，PCHIP＋QP平均RMSE降低约36%；峰值场组合也有改善，但自动方案没有超过最强固定组合。组合不是普遍必需。
4. **峰值需求识别与改善是否可靠？** 识别了高值/位置偏好并选择高阶或平滑估计＋一致性；宽峰可恢复，窄峰失败明确保留。没有证明峰值场自动路由创新优于合理强对照。
5. **自动筛选是否比固定流程更有价值？** 在本轮预设三类混合中优于任何一个全局固定流程；若已知具体任务并固定其最佳流程，自动路由不一定更准或更省时。整体收益不能掩盖C的局部不足。
6. **哪些得到支持？** 输入隔离、合法执行、A避免无益组合、B组合收益及本轮任务混合上的整体收益有真实数值证据。真实环境应用、未知分布泛化、窄峰唯一恢复和普适算法创新仍未证明。本轮完成，不再扩展算法或任务。

## 37. 北京PM2.5混合缺失恢复：处理图搜索与真实数值评价（2026-10-20）

### 37.1 最终执行范围及有效版本

本轮已完成实际代码适配、北京12站数据核查、训练状态、三类遮挡、2015年计时、50次正式搜索、949种规范处理图的真实数值穷举参考、冻结后2017年评价、定向回归、配对分析及文件完整性验收。没有下载数据、扩充通用34项池、修改第三方GOLEM或删除历史结果。

唯一实验根目录：`output/beijing_mixed_gap_20261020/`。

**主结果仅使用`strict_grammar_v2/`。** 首版50次搜索及其2017年输出全部保留，但撤回主比较资格：图空间审计发现49次完整执行、31个不同结构键含“同一方法给自己回退”或“完全相同的分流分支”，它们不在949种规范语法内。原因是原生成器过滤了这些结构，Mutation后的公共检查器却没有同步拒绝。

修复位于项目自身`MixedGapFactory.inspect`，A/B/C/C-gap/R全部共用；没有改变算法内核、阈值域、数据、H₀、目标函数、搜索种子或预算。修复后重跑全部50次搜索，所有完整数值执行图均属于相同的949图规范语法。首版证据见`withdrawal_v1.json`、`source_v1/`及原子目录；不能用首版较好的种子填补V2。

V2在全部搜索完成后，依据数值函数AST、训练状态SHA256和2016年公开输入SHA256一致性，复用首版**合规949图**的实际穷举计算。复用发生在搜索结束后，未将穷举成绩透露给搜索器，也未把复用当成949次新执行。

数据及2017年曾被历史实验和首版使用，本轮只能称**冻结后探索性复核**，不是全新盲测。

### 37.2 数据来源、单位、时间及使用权限

复用`D:/codex/数据/公开数据三层预审_20261018/01_UCI_Beijing/`的来源清单与官方CSV。[UCI 501](https://archive.ics.uci.edu/dataset/501/beijing)，DOI `10.24432/C5RK5G`，CC BY 4.0。官方ZIP 7,959,991字节，原清单SHA256为`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`；本轮逐一复核12份实际CSV的SHA256，原件只读。

实际12站：Aotizhongxin、Changping、Dingling、Dongsi、Guanyuan、Gucheng、Huairou、Nongzhanguan、Shunyi、Tiantan、Wanliu、Wanshouxigong。每站35,064小时，2013-03-01至2017-02-28；合计420,768条，PM2.5有效412,029、原始缺失8,739。唯一递增小时轴，无站内重复时间；没有负值或Inf需剔除。最小值2、最大值999，未按误差或高浓度删除极端记录。

- 单位由已保存官方页面的原文`PM2.5: PM2.5 concentration (ug/m^3)`支持，统一元数据写为`ug m-3`，数值不缩放。
- 时间使用原始无时区小时标签。不擅自转换UTC，不宣称已经核实小时起始/结束或平均窗口；该不确定性不被伪装成已证实的产品定义。
- 官方NA按自然缺失处理；没有仪器QA字段，不把有限且非负的数值筛选叫作仪器认证。
- 不读取经纬度做空间估计，不使用其他站同期观测、不使用其他污染物或气象协变量。
- 原文件整体审计曾读取包含2017年的CSV用于时间轴及质量统计；这些原值没有传给搜索器。V2复用首版已存在的2017年输入文件，不能把文件存在或历史读过的事实抹去；逐方案2017年复核只在V2选图再次冻结后执行。

|用途|固定区间|实际实现|
|---|---|---|
|训练状态|2013-03—2014-12|每站小时中位数和日历KNN，仅本期观测拟合|
|开发/机制及预算校准|2015|H₀封存之后运行，未用2016年成绩设预算|
|搜索与验证|2016|所有方法同一公开遮挡输入与私有评价队列|
|冻结后复核|2017-01—02|搜索方案及参数冻结后统一评价；历史暴露明确保留|

小时中位数按站、0—23小时拟合，最低样本数30，不足用同站训练总体中位数。实际各小时最低训练数634～656，回退阈值已实现但真实训练数据未触发低样本分支。各站总训练有效数15,482～15,990。日历KNN复用原UCI方法的24邻居、距离加权、小时与年内日序正余弦特征（分母366），没有新增参数优化。训练高值阈值为各站训练PM2.5的90分位数，约166～198μg/m³，冻结后不重算。

统计模型总拟合约0.169秒，训练状态所有组均可复用，成本单列；没有从2015/2016/2017年的可见上下文重新拟合小时统计或KNN。

### 37.3 遮挡、资格队列和目标函数

mask种子101/202/303，独立于搜索种子11/22/…/110。所有场景分别运行，不叠加人工mask：

1. 随机：从原本有效小时抽取floor(20%)，不从自然缺失取评价真值。相邻遮挡和自然缺失合并后的实际可见缺口长度逐点记录，不把相邻点说成孤立单点。
2. 内部段：1/3/6/12/24/48/72小时，每站每mask每年度期望20块，四季各5块；要求整段及紧邻两侧均有观测。区间仅按可用性抽样，双侧24小时保留区不重叠，实际选段间距至少48小时。2016年每个长度均接受720块，即12站×3mask×20块。
3. 边界：固定168小时窗口，季度起始月1/4/7/10；三个mask分别从该月1日起偏移0/14/28天，窗口互不重叠。分别隐藏开头/末尾6/24/72小时，目标整段和可见侧紧邻小时必须原本有效。没有合格观测则保留短缺，不调日期凑效果。

2016年每个边界方向/长度原期望144窗口，实际接受：末尾6/24/72小时分别139/114/83，开头分别136/123/90。2017年只有1—2月，没有虚构其余季度样本。

|年份|评价目标记录（含不同mask/场景重复）|不同站点－小时原始目标|模型输入case数|
|---|---:|---:|---:|
|2015|200,967|86,863|见该年summary.json|
|2016|201,189|86,496|973|
|2017|46,476|15,148|491|

重复mask、重复场景及同一缺口中的小时都不是独立环境样本。自然缺失保留、可被算法恢复，但不进入准确率分母。资格、mask及分母在任何算法执行前固定。

主目标是**分层平均RMSE**，不是全部目标点的单次RMSE：

`J = mean_站点 mean_mask { [RMSE随机 + mean_内部长度(RMSE) + mean_边界方向和长度(RMSE)] / 3 }`。

同一站/mask/长度内的多个合格缺口先合并计算该组RMSE；边界同一方向/长度的合格固定窗口同理。仅依据可用性提前冻结的非空分层参与其内部平均，不按方法删分层。2016年502个、2017年491个有效分层，权重和均为1。另报逐点RMSE、MAE、Bias、高值误差，不能在主结果不利时换主指标。

所有目标必须覆盖、有限、非负，原始未遮挡值逐值保持。统一非负投影保留原始预测、修正次数与幅度；本次真实方法均未产生需要修正的负预测，单元测试另实际验证了负值投影。完整图质量失败保留全部分母及原始部分预测，主排名记不合格，不在各自容易的成功子集排名。

### 37.4 实际接入与信息隔离

复用路径：`任务局部Registry → TypedGolemFactory → 真实GOLEM图 → UnifiedAlgorithmPipeline.run_dag → NumericalDAGExecutor → ResultQualityGate → 独立评价器`。没有另写优化器，没有模拟误差。

本轮新增最小任务适配器`src/ecodata/algorithms/mixed_gap_dag.py`：

- 5个实际方法：时间线性、最近邻、前填、训练小时中位数、训练日历KNN。
- 通用控制：按**当前输入NaN连续段**识别长度、按阈值分流、主方法无输出位置才用备份、合并及原始有效值保护。
- 明确训练状态端口；训练中位数不复用旧城市任务“从当前可见序列估计”的口径。
- 不可变、内容指纹化的任务数据，避免每条边反复复制整年输入；每次独立搜索有全新节点/内核/图缓存，组间不共享搜索预测结果。
- 历史内插可用缺口两侧观测，边界只能用窗口内真实可见侧。前填仅过去，其余双侧/历史统计不称为未来预测。

算法节点不接收truth、人工mask类别、H₀或测试误差；输入只有每case站点/时间/受遮挡序列及合法训练状态。批量计算逐case建立独立Series，不把其他站或同站其他mask副本的可见值当作供体。改变另一个mask副本的观测不能改变本副本输出，该隔离已有测试。

单列质量查询端口只给缺失位置索引，不含目标值；用于验证覆盖和输出，不作为分流依据。评价器在DAG返回后才用独立目标表计算误差。每条边及缓存均有数据/状态内容指纹。

为保存真实失败预测，仅对公共`DAGError`增加可选`details`并在既有包装中传递；质量BLOCK仍然抛出，未屏蔽异常。所有对照均调用相同质量门。

本轮5方法是明确版本的**任务局部候选池**，不是原通用34池。训练态方法通过已注册命名端口运行，不把旧通用`TemporalBaseline.run`宣称为新的训练状态接口。具体输入/输出端口、模式及限制在`candidate_contracts.json`。

### 37.5 H₀、公共图空间及真实触发

H₀在查看开发方法误差前封存，未替换为H₁：

`可见缺口≤6h：线性，缺输出则训练小时中位数；>6h：训练小时中位数 → 合并原有效值 → 质量门`。

公共语法允许单方法、条件回退、一个长度分流及每分支至多一个回退；阈值统一1/3/6/12/24/48小时。方法分配公平开放，禁止循环、递归控制、同方法自回退及完全相同分流分支。所有搜索从缺少重构能力的质量节点初始图出发；GOLEM按相同公共工厂扩充种群，不能加载H₀专用图初始化。

共949个规范结构，全部已有真实2016年执行证据：441个质量通过、508个因目标覆盖不全BLOCK。通过者按1e-8量化预测指纹得到408组，组内33个代表比较均确认最大容差一致；不同量化桶不被自动宣称在所有容差、所有数据上不等价。空间不是旧任务的8图，但也不是不可穷举的大空间。

代表性自动发现图：C、seed11，验证J=33.555104：

`可见缺口≤24h：线性，缺输出则最近邻；>24h：仍先线性，缺输出才用训练小时中位数 → 合并保护 → 质量门`。

它与H₀的差异不是仅改节点数：阈值6改为24；长段不再无条件丢弃两侧观测、直接用历史中位数；短段边界回退也由中位数改为最近邻。训练状态仍只来自2013—2014。

在同一2016年201,189个目标上追加两次有界控制归因执行，逐值重现已保存结果：

|图|有效线性供给目标|有效最近邻回退目标|有效训练中位数供给目标|
|---|---:|---:|---:|
|H₀|68,924|0|132,265|
|C seed11图|181,371|7,362|12,456|

H₀短分支70,598个目标，其中1,674点实际回退；C图短分支102,316点，其中7,362点实际回退，长分支98,873点，其中12,456点实际回退。控制节点确实参与输出，不是名称占位。完整归因在`representative_controls.json`。

当前静态DAG会先计算两分支/备份，再按条件选择输出，并非惰性计算；核缓存减少重复。逐节点原始计数包含全部输入缺失位置，不能冒充最终生效目标数，因此另作以上“有效目标”归因。额外2次控制审计不计入搜索预算。

### 37.6 A/B/C/C-gap/R、预算及缓存的公平口径

|机制|实际做法|
|---|---|
|A|GOLEM官方默认变异算子之任务适配：add/change/drop/edge及已有类型化tree-growth钩子；共用语法和最终质量门|
|B|A＋后验时间支持/覆盖契约检查，未完成图允许继续演化|
|C|共用官方优化器，候选提议权重`exp[2×缺口下降－4×可见支持风险]`；不使用预测误差给候选贴预期赢家|
|C-gap|提议概率只保留缺口下降，去掉上述软支持风险项；仍保留与B/C相同的后验覆盖门。不是“删除系统全部语义”的消融|
|R|从同一规范语法随机生成完整图；不是同Mutation消融；重复图用独立缓存，不充当新评价|

风险仅来自实际可见缺口长度、左右支持及方法能力，代表时间支持语义，不足以证明复杂环境物理语义的通用价值。对949图的独立审计显示：本队列后验覆盖判断与真实质量通过情况一致，未发现误拒绝通过图或放过失败图；这没有使用参考浓度误差。

2015年实测预算后冻结：10个种子11/22/33/44/55/66/77/88/99/110；每运行最多50个**唯一计分图**、500次外层变异/随机提议、1,000个原始回调安全上限、600秒搜索软时限；没有按组提高预算。穷举资源软限900秒，开发估计约339秒，故允许执行/复用全空间。

不完整图反馈为`1e6+缺口数`，质量失败为统一失败值；其`objective=null`，都不是μg/m³ RMSE。只有完整且质量通过图以真实J排名。缓存回调与超预算回调不新增计分。每个新完整图均实际运行全部固定验证队列。

V2合计：2,351个唯一计分图，其中1,281个不完整反馈、1,070次真实完整图数值执行；真实执行732通过、338质量失败。原始回调3,158次＝2,351计分＋760图缓存命中＋47超限延后、不执行回调。没有拿缓存命中凑满50次。

外层提议合计12,944，另记录3,487次提议拒绝、437次后验拒绝；这些计数含不同阶段检查，不能简单相加成不同失败图数。GOLEM扩充种群及嵌套子树构造另有完整工厂轨迹：A/B/C/C-gap/R分别142/166/502/491/519次类型化构造。这些不是额外数值Objective，不能与外层Mutation次数混为一个分母；初始化/嵌套生成开销已进入墙钟，详见`factory_generation_accounting.csv`。

搜索计时从固定队列及执行器装载之后开始，所有组一致；训练拟合、装载、完整进程资源另列，不把搜索计时冒充原始文件解析到交付的总时间。软时限允许已启动图结束，本轮无600秒超时。A计分7～50、B均50、C42～50、C-gap12～50、R均50；提前结束保留，未补种子或重复图。

原始JSON记录全部直接计分/缓存事件；47个延后回调有明确计数，但原代码未保存其独立时间戳。`raw_callbacks.csv`将它们明确标记为“按延后计数展开、无独立时刻”，没有伪造时间或图执行。

### 37.7 全部固定方法与H₀的结果

以下J与逐点RMSE单位均μg/m³，二者定义不同。破折号代表完整队列未通过，**不是0误差**。

|方法|2016验证J|2016逐点RMSE|验证覆盖率|2017复核J|2017逐点RMSE|复核覆盖率|
|---|---:|---:|---:|---:|---:|---:|
|线性|—|—|90.1496%|—|—|85.8314%|
|最近邻（合格强单算法）|37.537314|56.903298|100%|56.261989|87.510504|100%|
|前填|—|—|94.8993%|—|—|92.8544%|
|训练小时中位数|71.934626|76.261992|100%|116.224738|124.072641|100%|
|训练日历KNN|99.135866|105.467120|100%|157.465323|165.021370|100%|
|H₀|43.538006|62.692939|100%|75.065455|110.360329|100%|

H₀在本队列没有优于强单算法：验证J高约15.99%，复核高约33.42%。复核MAE/Bias为64.863/−31.540，最近邻为47.518/+2.076；训练高值阈值以上RMSE分别234.012和145.117。不能只用小缺口上的成功说H₀整体合理。

2016年具体场景的逐点RMSE：

|内部长度|线性/H₀短段结果|H₀实际结果|最近邻|
|---|---:|---:|---:|
|1h|11.983|11.983|20.415|
|3h|15.975|15.975|20.895|
|6h|22.750|22.750|27.582|
|12h|32.834|79.102|39.652|
|24h|46.955|74.806|54.358|
|48h|55.442|75.423|62.625|
|72h|71.517|77.212|79.430|

H₀的小内部段确实有效，但从12小时起无条件转历史中位数丢弃了有价值的同期两侧信息；并非所有长段都输给最近邻（72小时略好），不能选择性省略。边界6小时H₀开头/末尾RMSE约64.492/65.745，最近邻30.554/28.829，历史中位数回退尤其不利；末尾72小时中位数又可优于前值/最近邻。完整方向、长度、站点、mask、高值、失败分母均保存于`group_metrics.csv`。

线性与前填虽在部分内部段有效，但按本实验统一完整覆盖要求不能进入总体合格排名；没有偷偷给单算法增加回退，也没有删除边界目标让它们合格。

### 37.8 全部搜索种子的真实结果

每行同一搜索种子，主指标为实际分层J；所有50次运行均找到质量合格图，优化器异常0。表中失败候选的成本仍留在各运行记录。

|seed|A验证J|B验证J|C验证J|C-gap验证J|R验证J|
|---|---:|---:|---:|---:|---:|
|11|37.53731|33.55510|33.55510|33.55510|36.91985|
|22|69.15025|40.55506|36.49647|38.64418|33.68858|
|33|42.03319|35.38917|37.25325|37.55918|33.55510|
|44|33.55510|36.09301|35.91327|36.19256|33.68343|
|55|36.86413|39.34792|34.78993|38.64418|34.73413|
|66|40.49422|34.57151|33.68343|33.68343|33.68858|
|77|33.68343|33.68343|36.87232|41.66009|33.74571|
|88|38.64418|34.73413|33.74571|33.74571|35.24017|
|99|41.79170|35.90515|33.75386|39.00814|34.17884|
|110|70.75422|35.90515|38.51726|35.91327|36.09703|

相同选图冻结后的2017年复核J：

|seed|A|B|C|C-gap|R|
|---|---:|---:|---:|---:|---:|
|11|56.2620|52.1145|52.1145|52.1145|61.5918|
|22|103.7005|68.5439|55.2119|66.7559|51.8672|
|33|71.2980|53.8310|56.9061|57.8863|52.1145|
|44|52.1145|54.3071|54.4639|54.9551|51.7086|
|55|55.6928|67.2987|54.4623|66.7559|60.6029|
|66|60.0454|51.9664|51.7086|51.7086|51.8672|
|77|51.7086|51.7086|56.6020|72.9927|52.0144|
|88|66.7559|60.6029|52.0144|52.0144|53.7765|
|99|73.8341|53.8073|52.1910|67.0755|51.6936|
|110|115.6268|53.8073|67.1618|54.4639|54.4516|

50个搜索选择＋6个固定基线引用41个不同图，在2017年每个不同图只统一执行一次。全部搜索所选图通过复核质量门；失败的两个固定单算法仍保留，不伪造为41个都合格。

|机制|验证J均值±种子SD|复核J均值±种子SD|复核逐点RMSE均值|搜索墙钟均值/秒|真实数值执行均值|其中合格图均值|
|---|---:|---:|---:|---:|---:|---:|
|A|44.4508±13.7608|70.7038±22.0768|95.7026|8.884|11.4|5.3|
|B|35.9740±2.2934|56.7988±6.3839|88.7225|9.742|11.9|11.9|
|C|35.4581±1.7926|55.2837±4.5934|86.2231|10.713|16.2|16.2|
|C-gap|36.8606±2.7196|59.6723±7.9187|88.0757|10.247|17.5|17.5|
|R|34.5531±1.1799|54.1688±3.7767|86.7994|7.324|50.0|22.3|

这里种子SD只描述搜索随机稳定性，不是独立环境重复的标准误。按事先规定J，R优于C；C的逐点RMSE略低于R，不能因此换主指标宣称C获胜。

### 37.9 搜索质量、等误差成本及穷举参考

- A/R新完整图质量通过比例分别约44.98%/44.60%（各运行比例均值）；B/C/C-gap均100%。后验覆盖检查确实避免无效数值执行，但B本身已达到同样效果，不能全部归功C的Mutation。
- 首次合格图所需计分均值A/B/C/C-gap/R为5.5/2.1/2.3/2.6/1.9次；墙钟均值2.188/1.508/1.214/1.315/1.644秒。C在首次合格墙钟上较快，但计分次数不是最少，也没有总体最快。
- 达到强单算法验证J≤37.537314的成功种子数分别4/8/9/5/10。只在成功者计算的到达成本单独标记，不能将其解释成全部种子的平均优势。
- 按每个种子五组共同可达的计分前缀重新比较，验证J均值A44.4508、B35.9928、C36.1588、C-gap36.8606、R34.7478。共同前缀下C还略差于B，额外优势并不稳定。真实执行数/墙钟共同前缀也另表保存。
- 949规范图池内最优验证J为33.475119；各搜索器找到的最好J为33.555104，均未取得该精确池内最优。穷举最优未替换任何搜索选择，也未被额外用于挑2017年赢家。
- 完整穷举首版合规计算约103.62秒，使用同一枚举运行内的节点缓存；V2核验并复用约16.34秒，不是重新计算949图的耗时。它与每次独立冷缓存搜索8～11秒不是相同缓存口径，不能直接除时间声称某机制更快。
- 这一固定任务有949种规范结构、441种质量通过图，完整穷举可承受。因此已有非平凡组合空间，但没有证据认为此处必须使用GOLEM而非随机搜索或直接枚举。

搜索累计计时约469.10秒，另有装载、开发、训练、完整性分析与文件输出。2秒采样的整个严格实验进程记录199次，进程寿命峰值约1.074GB（约1.00GiB）；这不是每张图的独占内存。报告目录含撤回首版约2.29GB，严格版约0.85GB，最终少量源码快照/审核会略增；未删除数据，D盘仍约160.5GB可用。

### 37.10 不确定性、季节与结论边界

按站点及共同7天时间块作1,000次配对双向重抽样；连续缺口全段分配到其起始时间块，不拆小时；不同mask及场景共享时间块权重，不当成独立重复。重抽样中的空分层不填0，按共同出现的有效分层重归一化。固定重抽样种子20261020。

下表为冻结后复核J之差，负值表示前者更好，区间为逐项探索性95%百分位区间，未作多比较校正：

|比较|平均差|95%区间|允许的解释|
|---|---:|---|---|
|H₀－最近邻|+18.8035|[-1.6455,+26.9772]|本队列点估计更差，不能将其泛化为所有环境过程都差|
|C－A|−15.4202|[-20.5812,−8.5200]|本次设置有改进，但包含后验门及提议机制的共同影响|
|C－B|−1.5151|[-1.7305,+0.3673]|不能证明比后验过滤稳定更优|
|C－R|+1.1148|[+0.7640,+1.6507]|按预设主指标，随机搜索更好|
|C－C-gap|−4.3886|[-5.7798,−0.2418]|本次软支持风险项有条件性收益；仍保留硬覆盖门，非全部语义贡献|
|R－最近邻|−2.0932|[-5.2309,−1.2102]|真实搜索出的组合可比强单算法改善，但不等于GOLEM改进获胜|

北京站点相关、2017年仅冬季两个月、历史数据多次暴露及人工缺失选择条件，均限制外推。测试J高于全年验证不能单独归因搜索失效，季节/目标分布和可用窗口亦不同。原生自然缺失没有真值，本实验只能验证人工遮挡的原有效观测恢复，不能证明自然缺失MCAR或实际未观测浓度正确。

### 37.11 定向测试、文件验收及复现

相关测试共46个不同测试通过，失败0、跳过0。旧相关DAG/时间/图生成回归33项＋本任务13项，不把多次XML中的重合测试重复相加。覆盖实际缺口边界、训练状态、跨mask隔离、条件回退、原值保护、负值投影、失败预测/分母、缓存身份、超预算回调不执行、H₀不参与搜索、共同语法及冗余Mutation拒绝。没有运行无关全量回归。

原始12个CSV再次校验未变；协议、训练状态及选图冻结SHA256未变；所有完整搜索执行均属于同一规范语法。冷缓存H₀重复执行两次，逐值一致；另两次有效控制归因也复现原预测，不冒充新搜索执行。NPZ预测写出后回读，Parquet目标队列回读分析；质量失败与文件完整性通过分别记录。

三个实际搜索曲线分别按计分次数、真实数值执行数和墙钟呈现10个种子的原始轨迹，不画未出现的RMSE、不把失败值当浓度误差、不外推提前停止后的新增评价。PDF/SVG/PNG均已生成并逐图检查，最终PDF最小字8pt、碰撞检查均无失败；单面板无需多面板对齐。300dpi PNG是预览而非投稿TIFF，原始种子轨迹不是均值曲线，故未伪加不确定性带。

环境：Python3.12.14、NumPy2.5.3、pandas3.0.1、SciPy1.18.1、scikit-learn1.9.1、joblib1.6.0、xarray2026.7.0。沿用现有GOLEM代码环境，实际`thegolem`0.5.0，[官方仓库](https://github.com/aimclub/GOLEM)，BSD-3-Clause，没有修改第三方核心。

**环境警告：** 现有执行方式沿用项目主环境并加载`.venv-golem`代码；GOLEM0.5.0在Python3.12的声明约束是NumPy<2、SciPy<1.13，而本机继承的数值栈更新。真实运行和定向测试已通过，但不能声称是干净、完全符合上游依赖锁的安装，跨机器复现须核对`environment.json`，不能假定普通pip解析自动得到同一组合。

关键新增/修改路径：

- `src/ecodata/algorithms/mixed_gap_dag.py`：任务局部实际方法端口、通用控制、共享严格语法及可见支持风险。
- `src/ecodata/algorithms/dag.py`：仅必要的失败细节传递，保留质量BLOCK。
- `tools/prepare_beijing_mixed_gap.py`：来源审计、训练状态、预设mask及独立目标。
- `tools/run_beijing_mixed_gap.py`：官方GOLEM执行、统一Objective、冷缓存、穷举参考、冻结及统一复核。
- `tools/analyze_beijing_mixed_gap.py`、`plot_beijing_mixed_gap.py`、`audit_beijing_gap_controls.py`、`verify_beijing_mixed_gap.py`：配对分析、真实曲线、有效控制归因与最终核验。
- `tests/test_beijing_mixed_gap.py`：新增13项风险/机制测试。

严格版主要产物：

|产物|用途|
|---|---|
|`protocol_frozen.json`、`H0_sealed.json`、`H0_graph_sealed.json`、`revision_evidence.json`|预验证规则、人工参考及共享语法修复证据|
|`data_contract_and_masks.json`|对已冻结代码/mask的执行后说明，不冒充新写的预注册协议，不改冻结协议|
|`source_audit.csv`、`training_states.joblib`、`training_statistics.csv`|真实来源SHA、训练状态与高值阈值|
|`data/2015,2016,2017/`|公开受遮挡输入、独立目标、mask、资格短缺和分层权重|
|`candidate_contracts.json`|真实命名端口、方法与控制边界|
|`search/*/summary.json, factory_trace.json, executions/*/`|全部搜索、提议、节点依赖、实际预测、失败及缓存|
|`enumeration_summary.json`|949图完整参考及实际原始执行路径；复用明确标记|
|`selected_graphs_frozen.json`、`frozen_test/`、`test_results.json`|冻结清单、41种统一复核输出与映射|
|`search_runs.csv`、`raw_callbacks.csv`、`factory_generation_accounting.csv`|逐种子精度、预算、延后回调及工厂开销|
|`fixed_and_frozen_metrics.csv`、`group_metrics.csv`|完整评价队列、各站点/场景/长度/高值及失败分母|
|`best_so_far.csv`、`common_budget_prefix.csv`、`common_execution_time_prefix.csv`、`equal_error_cost.csv`|三种成本坐标、共同预算及到达质量阈值分析|
|`behavior_audit.json`、`control_trigger_counts.csv`、`representative_controls.json`|结构/数值行为、原始控制与有效目标归因|
|`paired_environment_intervals.csv`、`paired_seed_differences.csv`|环境相关结构重抽样与单列搜索种子稳定性|
|`actual_cold_reproducibility.json`、`final_acceptance.json`、`tests_final.xml`|复现、源数据/配置/预测哈希核验及测试|
|`search_curve_*.pdf,svg,png`、`figure_review.json`|实际曲线及图示QA|

公共源码修改前快照在根目录`source_before/`，撤回首版源码在`source_v1/`；严格执行源码、配置及输出指纹随交付保留。首个准备阶段的只读数组及datetime整数单位断言故障已最小修复，终端错误与修复说明在`implementation_fixes.json`。未屏蔽失败、未用其他数据替代、未删除旧结果。

在项目目录复查当前交付（不会重跑搜索）：

```powershell
$env:ECODATA_BEIJING_MIXED_OUTPUT='D:/codex/EcoDataEngine/output/beijing_mixed_gap_20261020/strict_grammar_v2'
.venv/Scripts/python.exe tools/verify_beijing_mixed_gap.py
.venv/Scripts/python.exe tools/analyze_beijing_mixed_gap.py
.venv/Scripts/python.exe -m pytest tests/test_beijing_mixed_gap.py -q -p no:cacheprovider --basetemp=output/pytest_beijing_mixed_review
```

完整复现须使用新输出目录，先准备同样原始数据/训练状态/mask，再校准并运行`run_beijing_mixed_gap.py`；默认输出路径用于保留首版，不应误当作严格版。已有`summary.json`会作为同目录已完成运行复用，复现不同版本前必须核对协议/输入指纹并选新目录，不能混装旧结果。所有方法均已跑完，本轮不继续扩大预算。

流程辅助工具已注明：[Kassis等，Scientific Agent Skills，2026](https://arxiv.org/abs/2609.00065)，本轮核对最新v2记录；它不是PM2.5算法或精度证据。

### 37.12 最终验收判断

1. **人工参考是否成立？** H₀可执行且保护观测，但整体未超过合格强单算法；其“大于6小时直接用历史中位数”的假设被本队列结果否定，不能用小段收益掩盖。
2. **GOLEM能否独立找到有效图？** 可以，50次真实数值搜索全部得到合格图，发现与H₀不同的阈值/回退/长段主方法；不是仅判断图结构。
3. **组合是否有价值？** 有条件成立：池内最优及部分搜索图优于最近邻，但人工组合并非当然更好；最优搜索不是必须GOLEM。
4. **C是否更有效？** 比A更稳定、比C-gap有条件改善，但相对B优势不稳；R在预设主指标和总墙钟上更好。不能宣称语义引导全面获胜。
5. **C是否更省成本？** 能减少无效数值执行，首次合格墙钟较短，但提议/检查成本高，平均总时间反而长于R；数值调用效率不等于总计算效率。
6. **科研贡献能支持到哪一步？** 支持真实处理图协作、信息隔离、共同质量门、明确预算和可解释的时域支持引导；不支持单凭本次单源时间任务证明跨尺度、多源或普适物理语义创新。此固定池可穷举，不扩大搜索空间凑优势。

最终状态：`REAL_NUMERICAL_DAG=PASS`；`TRAIN_AND_TARGET_ISOLATION=PASS_FOR_THIS_IMPLEMENTATION`；`STRICT_SHARED_GRAMMAR=PASS`；`SEARCH_RUNS=50/50_EXECUTED`；`ENUMERATION=949/949_REAL_REFERENCE`；`FROZEN_FOLLOWUP=EXECUTED_EXPLORATORY`；`H0_BEATS_STRONG_SINGLE=NO_ON_THIS_COHORT`；`C_BEATS_A=SUPPORTED_CONDITIONALLY`；`C_BEATS_B=NOT_ESTABLISHED`；`C_BEATS_RANDOM=NO`；`UNIVERSAL_PHYSICAL_SEMANTIC_ADVANTAGE=NOT_PROVEN`。


## 38. 平台统一语义决策链贯通与北京混合缺失恢复重跑（2026-10-20）

### 38.1 本轮范围与证据边界

本轮修改了项目自身的公共模块并实际执行实验，不是重新调研或新增算法。第37章旧C检验的是任务局部“端口缺口＋去重时间支持条件风险”，**不是完整平台特征—能力—算法机制**。`strict_grammar_v2`、旧数据、全部结果和失败记录均保留。本轮有效新版本：`output/beijing_mixed_gap_20261020/unified_platform_v4/`。

“统一机制”在本章指当前时间恢复任务的公共提取器、规则映射、Registry契约、实际兼容性、输出路径评估和真实GOLEM反馈贯通；不宣称所有历史入口已经改造，也不宣称具备通用精度预测模型。

### 38.2 先读代码得到的调用链审计

|环节/入口|本轮前实际状态|本轮后状态与源码|
|---|---|---|
|数据识别、语义与质量|已有且部分入口调用；元数据有消费者，UNKNOWN保留|复用 `semantic_requirements.py:SemanticFeatureExtractor.extract`；增加同类 `extract_series`，读取可见数组、时间轴、缺失段及两侧支持|
|能力规则|`RequirementMapper`已有，但没有缺失恢复规则；北京未调用|在同一 `semantic_rules.json` 最小扩展恢复、完整覆盖、单侧支持、仅过去要求、观测支持软目标、平均时间语义|
|Registry能力与适用性|`capability_catalog`已有；旧北京只用真实任务端口，没有公共需求匹配|`router.py:AlgorithmSelector.decide`复用目录及真实插件 `compatibility`；科学方法仍为原有5种任务内核，未扩充冻结34项库|
|早期GOLEM|`GolemCapabilitySearch`调用 `run_golem_search.py:metrics`，全图能力并集＋结构成本评分|保留兼容，不用于本章数值实验；没有把“调用旧类”当作贯通证据|
|北京数值入口|已有真实DAG/质量门/J；C使用局部风险表|`Evaluator.decision`只转换公开数据/任务；调用与生产相同的 `AlgorithmSelector.decide`；新搜索不调用旧 `conditions/semantic_risk`|
|输出路径、状态|类型化父边和状态依赖已实现；能力语义未接通|`SemanticDecision.assess_graph`消费 `NodeAdapter.decision_contract`，逐输出端口递归、条件分支选择、缺失回退；不使用全图能力并集|
|生成与变异|已有共同类型化语法、官方Mutation、有限尝试|保留 `TypedGolemFactory`、`typed_tree_growth`；C在真实变异候选之间调用公共 `semantic_proposal_weights`；概率实际用于抽样|
|真实执行与反馈|已调用真实算法，返回独立数值J|保留 `UnifiedAlgorithmPipeline.run_dag`→`NumericalDAGExecutor`→现有科学内核→质量门→独立Evaluator；J不被静态语义分数替代|
|生产栅格入口|已有12项栅格特征、开发集训练路由、执行质量门，但绕过公共需求映射|`process_raster_upload`同样调用 `AlgorithmSelector.decide`，实际筛选首步；约束尾节点仍检查其真正接收的预测网格|
|台湾历史数值入口|给factory赋semantics/requirements，但guided主要消费端口缺口与节点数|只审计、保留，不冒称已接入本轮新引导；需后续按对应空间契约迁移|

逐函数当前行号、旧/新调用关系、字段消费者、规则消费者见 `call_chain_audit.json`，原源码快照在 `unified_platform_v3/source_before/`，本次实际执行源码在新目录 `source_executed/`。

**哪些特征实际起作用：** 连续/分类语义进入能力规则和实际兼容性；单位/非负性进入未知项与质量约束；单侧支持触发边界恢复需求；缺失长度只用于已声明控制节点的路由谓词，不映射算法赢家；可见缺失位置的频数进入软支持偏好。有限值总数、采样间隔、源数据空间/时间未知项主要用于审计，本任务没有时间聚合、坐标转换或新尺度要求，不能宣称这些已影响当前概率。原栅格12项形态特征仍属于其开发训练模型，不能直接迁移为北京精度预测。

其他领域的软目标（地形、类别比例、来源可靠度、模型网格偏好等）在本轮时间引导中没有对应消费者，完整清单写入审计JSON；Registry的声明成本也只作记录，不进入本轮概率，不宣称实现了软成本引导。不因属性已声明就声称实现。仅过去要求已验证匹配拒绝未来读取方法，但本次实际任务是双侧恢复，不把此检查冒充对所有旧入口的因果执行改造。

### 38.3 公共决策怎样真正改变搜索

```text
公开受遮挡序列（没有恢复真值/场景标签）
 -> extract_series：原生时间、UNKNOWN、可见缺失段/邻侧支持/位置频数
 -> RequirementMapper：恢复能力、完整覆盖、边界支持、非负验证
 -> capability_catalog + 真实 compatibility + 声明的命名端口语义
 -> assess_graph：目标输出祖先路径的缺口、绑定状态、支持证据
 -> semantic_proposal_weights：候选权重/归一化概率/实际选择
 -> 官方GOLEM处理图 -> Pipeline/DAG -> 真实预测 -> 质量门 -> 独立J反馈
```

三层判断分开：①错误类型、未来读取禁令等硬不适用；②输出路径未满足的能力/尚缺的端口；③合法方案间的支持证据软偏好。完整覆盖是所有组最终质量门的共同要求；B/C/C-gap可提前拒绝已确定不满足覆盖的完整图。不完整图继续进化，UNKNOWN不自动判非法。

方法的双侧、任一侧、左侧或历史状态支持，由 `TemporalBaseline`实际内核契约登记；任务适配器只声明输入/输出和组合运算含义。状态必须连接到当前预测节点的state端口，不能借用另一分支的历史状态。条件分支按实际缺失长度挑选支持，回退仅在主预测不可用的位置生效。观测保护和非负投影仍由真实质量节点执行。

软偏好仅表示“当前保留观测的支持证据”，**不表示算法精度**：比如边界最近邻是明确的端点延拓，历史中位数是历史估计，均可合法执行。软权重采用全部可见缺失输入位置频数（含自然缺失），不把一次罕见条件与大量位置等权；硬覆盖则检查条件是否存在。它不使用私有目标队列的场景权重。数值J仍严格使用原冻结的站点/掩码/三大场景/子场景分层权重，二者不是同一个统计目标。

沿用系数2与4，无2016/2017调优。C-gap关闭软项，只保留公共能力缺口引导；C完整开启。实际引导批次中非均匀概率计数：

|mode|sum|count|
|---|---|---|
|C|833|1345|
|C-gap|714|1417|

逐批候选、每个输出端口缺口、权重、概率、操作、图指纹及最终数值反馈在 `search/*/decision.json`、`summary.json`、`factory_trace.json`、`executions/`。没有向策略传入H₀、2017成绩、文件名、人工预期赢家或旧穷举排名。

### 38.4 机制验收、发现的问题及处理

受控测试验证：单侧/双侧改变相应能力需求与路径覆盖；past_only改变实际匹配；开关软偏好改变概率而非合法性；无关描述/文件名不改变选择依据；状态在错误分支不能抵消当前state缺失；条件分支不是能力并集；未知契约保留UNKNOWN；微秒/纳秒存储不改变实际小时特征；软支持采用位置频数。生产真实上传与北京实验调用同一决策API，且已有栅格真实页面/下载测试通过。界面仍只支持其规定栅格，不宣称新增北京上传界面。

2015年两条真实路线“最近邻”和“线性＋最近邻回退”执行、质量检查、预测NPZ回读通过且输出不同。相关初始机制回归74项通过；最终受影响回归见下节。

保留的异常记录：旧测试曾依赖旧源码字符串，已改为检查公共调用/开关并增加行为测试，没有删除失败案例。5次及20次评价冒烟中A种子11未找到合格图，但实际执行/质量反馈正常；前20图与历史A逐图逐状态一致，旧A第25次才成功。这是有限预算未命中，不是生成器Bug；失败JSON保留，未增加好种子或改规则。

第一新版本v3运行时发现时间数组同时使用`datetime64[us]`（685例）和`datetime64[ns]`（288例），提取器直接比较底层整数导致采样间隔错误记UNKNOWN。停止该版本，保留7份完整搜索摘要和在途记录；公共提取器统一`.as_unit('ns')`后测得真实1小时。插值内核未改，但不混用错误日志；新v4重新冻结和执行。没有删除v3，没有据结果调规则。资源监控首次命令转义失败，改为复用原监控脚本的明确命令行参数；结束时发现监控器也匹配了自己的`--entry`参数，已改为匹配实际脚本位置，并按执行进程固定PID排除58条其他PID采样，原始行全部保留。XML合并首条命令转义失败后改用无嵌套转义的同等命令，未改变任何测试结果。以上不是算法调优，不隐藏失败。

### 38.5 冻结协议、公平预算及数据隔离

原12站CSV、训练状态、2015/2016/2017公开输入、掩码、私有评价队列及权重逐文件SHA一致；没有重新匹配或下载。2013—2014拟合；2015仅用于既有校准/机制验收；2016为真实搜索队列201189目标；2017为冻结后46476目标的历史暴露探索性复核，不是新盲测。人工参考H₀及主指标J未变，所有方法同队列、同质量门，不在成功子集排名。

同一949种规范图、同一未完成初始图、10种子11/22/33/44/55/66/77/88/99/110。每运行最多50次独特Objective计分、500外层提议、1000原始回调、600秒软墙钟，原提前停止保持；不完整反馈也占计分，不用缓存补次数。所有组独立冷执行器/内核缓存，未声称清空操作系统文件缓存；共同训练状态的原记录拟合成本0.169256秒，本轮直接复用，不冒称重新拟合。

|组|共同类型化语法/安全/真实质量门|后验过滤|能力缺口提议偏好|观测支持软偏好|
|---|---|---|---|---|
|A|是|否|否，官方Mutation|否|
|B|是|是|否，官方Mutation|否|
|C|是|是|是，公共决策|是，公共决策|
|C-gap|是|是|是，公共决策|关闭|
|R|是|否，最终门相同|否，均匀共同语法完整图|否|

旧34项冻结组件库未改；任务实际科学运算仍是既有5种时间方法及原控制语法。其余组件没有合法时间任务端口，不能任意塞进图；不能把“平台注册34项”说成本实验执行34种算法。候选契约记录也明确：某些历史单Series插件入口（例如calendar KNN直接run）并未新增实现，本章证据针对已注册、实际调用的命名端口任务内核。

所有正式搜索完成后、读取新冻结2017结果前，先确认每个完整执行图均属于949图规范空间；再真实执行949图穷举。没有使用旧穷举成绩引导新搜索。选图/参数/代码指纹再次冻结，统一执行2017复核。所有PASS必须质量通过且预测文件回读。

### 38.6 固定流程与强对照

单位：μg/m³；J是原分层RMSE加权主目标，不等同所有点直接RMSE。失败方案保留完整分母、覆盖率和原始部分预测，不用局部误差排名。

|method|validation|val_J|val_RMSE|val_MAE|val_Bias|val_coverage|followup|followup_J|followup_RMSE|followup_coverage|
|---|---|---|---|---|---|---|---|---|---|---|
|linear|FAIL|未评价|未评价|未评价|未评价|0.9015|FAIL|未评价|未评价|0.8583|
|nearest|PASS|37.5373|56.9033|29.9231|0.6139|1.0000|PASS|56.2620|87.5105|1.0000|
|ffill|FAIL|未评价|未评价|未评价|未评价|0.9490|FAIL|未评价|未评价|0.9285|
|hour_median|PASS|71.9346|76.2620|51.8390|-12.0585|1.0000|PASS|116.2247|124.0726|1.0000|
|calendar_knn|PASS|99.1359|105.4671|75.7265|12.6595|1.0000|PASS|157.4653|165.0214|1.0000|
|H0|PASS|43.5380|62.6929|36.6883|-8.1750|1.0000|PASS|75.0655|110.3603|1.0000|
|linear_nearest|PASS|33.4751|51.3065|27.6041|0.5953|1.0000|PASS|51.1507|82.6210|1.0000|

新增的解释性强对照“linear_nearest”只是原池已有的线性＋最近邻回退合法图，不新增算法、不替代H₀。不能只胜过偏弱H₀就声称自动组合有效。

### 38.7 全部搜索结果与成本

|mode|n|val_J|val_SD|followup_J|followup_SD|followup_RMSE|scored|real|valid|proposal_legal|real_quality_pass|search_seconds|setup_seconds|total_seconds|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|A|10|44.4508|13.7608|70.7038|22.0768|95.7026|45.7000|11.4000|5.3000|0.7645|0.4498|12.6330|0.3476|12.9806|
|B|10|35.9740|2.2934|56.7988|6.3839|88.7225|50.0000|11.9000|11.9000|0.7466|1.0000|14.9196|0.2899|15.2096|
|C|10|36.2167|2.2055|58.1378|6.5368|89.1797|42.6000|16.9000|16.9000|0.7235|1.0000|14.7737|0.2900|15.0638|
|C-gap|10|36.0530|2.1095|57.7746|6.5638|88.9314|45.0000|19.5000|19.5000|0.7051|1.0000|16.9016|0.2891|17.1907|
|R|10|34.5531|1.1799|54.1688|3.7767|86.7994|50.0000|50.0000|22.3000|1.0000|0.4460|7.3941|0.2865|7.6806|

proposal_legal是外层提议通过共同结构/端口检查的比例；real_quality_pass是完整真实执行通过最终质量门的比例，不把二者或重复的后验检查次数混作同一分母。

实际计数：{"scored": 2333, "real_executions": 1097, "quality_failures": 338, "incomplete_feedback": 1236, "callbacks": 3160, "graph_cache_hits": 786, "deferred_callbacks": 41, "proposals": 12425, "proposal_rejections": 3294, "posterior_rejections": 280}。计分、完整真实执行、质量失败、不完整反馈、缓存、拒绝与延后回调分开记录。初始化/子树产生的真实工厂构建总次数1808，不冒充额外Objective评价；每组计数详见CSV。

部分运行未达到50次计分：A种子99为7次；C种子22/33/110分别31/37/8次，C-gap种子22/33/44/110分别31/37/40/42次。保留统一的提议/回调上限和GOLEM原提前停止、无法补足下一代的正常退出记录，没有补次数或补种子；不把这些运行称为“空间已穷尽”。`common_budget_prefix.csv`和`common_execution_time_prefix.csv`另提供各配对种子实际共同达到的计分、执行和时间前缀对照。存在重复提议与局部停滞，属于当前机制的实际效率限制，不隐藏、不用重启凑更好成绩。

搜索总时间666.22秒，输入装载/特征提取/规则映射/匹配等setup总计15.03秒，逐运行平台总计681.25秒。公共决策服务内部时间不含外层特征遍历，不能只报告那一个很小数；setup已包含该部分。逐运行total计时到摘要组装，最终摘要/工厂日志写出等还有额外开销，进程生命周期资源采样保留，不能解释为完整端到端耗时的精确下界之外无成本。

进程OS生命周期峰值工作集4.230GiB；从进程创建到最后一次正确PID采样的墙钟795.77秒（含搜索、穷举和复核；采样停止有约2秒粒度，不作精确完成时长）。资源监控晚于最初若干运行启动，原始采样范围与说明见 `resource_summary.json`，不冒充全系统资源测量。保存完整嵌套轨迹会增加内存，不能只报算法内核成本。best-so-far同时保存计分次数、真实执行次数、搜索墙钟和加上setup的平台墙钟；未外推提前停止后的成绩。

逐种子完整结果（无删除失败或差种子）：

|mode|seed|scored|real_executions|unique_valid|best_validation|test_objective|seconds|platform_total_seconds|
|---|---|---|---|---|---|---|---|---|
|A|11|50|13|1|37.5373|56.2620|13.7323|14.4954|
|A|22|50|11|3|69.1502|103.7005|10.3347|10.6343|
|A|33|50|14|6|42.0332|71.2980|10.8497|11.1464|
|A|44|50|18|10|33.5551|52.1145|14.0393|14.3378|
|A|55|50|10|8|36.8641|55.6928|12.5501|12.8652|
|A|66|50|10|3|40.4942|60.0454|12.7129|13.0128|
|A|77|50|14|10|33.6834|51.7086|15.1152|15.4066|
|A|88|50|10|7|38.6442|66.7559|16.2533|16.5598|
|A|99|7|4|1|41.7917|73.8341|2.7375|3.0354|
|A|110|50|10|4|70.7542|115.6268|18.0046|18.3122|
|B|11|50|13|13|33.5551|52.1145|8.8753|9.1591|
|B|22|50|12|12|40.5551|68.5439|10.0700|10.3571|
|B|33|50|17|17|35.3892|53.8310|13.3561|13.6394|
|B|44|50|13|13|36.0930|54.3071|13.8402|14.1320|
|B|55|50|7|7|39.3479|67.2987|15.3026|15.5992|
|B|66|50|10|10|34.5715|51.9664|14.6681|14.9616|
|B|77|50|11|11|33.6834|51.7086|18.1086|18.4017|
|B|88|50|13|13|34.7341|60.6029|17.1311|17.4185|
|B|99|50|13|13|35.9051|53.8073|19.4024|19.6940|
|B|110|50|10|10|35.9051|53.8073|18.4421|18.7332|
|C|11|50|22|22|33.5551|52.1145|12.6752|12.9590|
|C|22|31|15|15|35.3892|53.8310|10.9189|11.2076|
|C|33|37|14|14|37.5592|57.8863|13.2122|13.4990|
|C|44|50|16|16|37.4652|58.2674|15.6826|15.9726|
|C|55|50|21|21|34.7899|54.4623|16.2123|16.5032|
|C|66|50|25|25|33.6834|51.7086|18.5415|18.8390|
|C|77|50|12|12|38.4536|66.8559|17.4997|17.7850|
|C|88|50|23|23|33.7457|52.0144|19.3955|19.6863|
|C|99|50|15|15|39.0081|67.0755|18.7619|19.0544|
|C|110|8|6|6|38.5173|67.1618|4.8376|5.1317|
|C-gap|11|50|20|20|33.5551|52.1145|11.6793|11.9696|
|C-gap|22|31|15|15|35.3892|53.8310|11.3545|11.6490|
|C-gap|33|37|14|14|37.5592|57.8863|13.7095|13.9914|
|C-gap|44|40|18|18|36.1926|54.9551|14.3117|14.6010|
|C-gap|55|50|19|19|34.7899|54.4623|15.0480|15.3400|
|C-gap|66|50|25|25|33.6834|51.7086|18.3846|18.6728|
|C-gap|77|50|22|22|38.4536|66.8559|19.8904|20.1771|
|C-gap|88|50|18|18|33.7457|52.0144|17.2604|17.5443|
|C-gap|99|50|25|25|38.6442|66.7559|22.9613|23.2534|
|C-gap|110|42|19|19|38.5173|67.1618|24.4166|24.7088|
|R|11|50|50|18|36.9198|61.5918|7.3736|7.6593|
|R|22|50|50|17|33.6886|51.8672|7.1774|7.4629|
|R|33|50|50|22|33.5551|52.1145|7.2162|7.5079|
|R|44|50|50|29|33.6834|51.7086|7.3454|7.6323|
|R|55|50|50|19|34.7341|60.6029|7.5877|7.8847|
|R|66|50|50|23|33.6886|51.8672|7.4956|7.7792|
|R|77|50|50|17|33.7457|52.0144|7.4843|7.7690|
|R|88|50|50|25|35.2402|53.7765|7.5204|7.8024|
|R|99|50|50|29|34.1788|51.6936|7.5142|7.7966|
|R|110|50|50|24|36.0970|54.4516|7.2259|7.5118|

### 38.8 穷举、配对不确定性与创新判断

949图全部真实执行，其中441通过、508质量失败；最优验证J=33.475119。完整输出的支持后验与实际成功/失败逐图交叉核验一致。此空间可穷举，不据此声称GOLEM必要或最优。

解释性强固定流程linear_nearest达到同一穷举最优，说明当前候选池内无需昂贵搜索就可取得该上限。C相对A的改善不能全部归因于完整语义引导，因为单独B已有很大改善；新C的能力/支持证据可以提高有效提议，却没有输入→精度预测模型，也不能保证把预算集中到最低误差图。软支持与原场景加权J不是同一个目标，这是可能的限制而非经本实验单独证实的因果解释；C与C-gap的对照用于检验其实际净收益，不按测试成绩重新定义偏好。

2017复核配对差异：负值表示左侧方案J更低。沿用按站点和7天时间块的双向配对重抽样1000次；同一连续缺口按起点周整体处理，遮挡复本不是独立环境样本；不同搜索种子只表示算法随机性，不当成不同环境。以下为探索性点式95%区间，无多重比较校正，不作确认性显著结论。

|a|b|objective_difference|ci95_low|ci95_high|
|---|---|---|---|---|
|H0|nearest|18.8035|-1.6455|26.9772|
|C|A|-12.5661|-17.4109|-8.7568|
|C|B|1.3390|-1.3447|2.8714|
|C|R|3.9689|-0.2169|5.3119|
|C|C-gap|0.3632|-0.4912|0.6839|
|R|nearest|-2.0932|-5.2309|-1.2102|
|linear_nearest|nearest|-5.1113|-7.1909|-3.9449|
|C|linear_nearest|6.9871|0.8225|9.4249|
|R|linear_nearest|3.0181|0.7829|4.4180|

C相对A：平均J差-12.5661，95%区间[-17.4109, -8.7568]；本探索性队列上的配对J更低。
C相对B：平均J差+1.3390，95%区间[-1.3447, +2.8714]；区间跨零，稳定优势未确立。
C相对C-gap：平均J差+0.3632，95%区间[-0.4912, +0.6839]；区间跨零，稳定优势未确立。
C相对R：平均J差+3.9689，95%区间[-0.2169, +5.3119]；区间跨零，稳定优势未确立。
C相对linear_nearest：平均J差+6.9871，95%区间[+0.8225, +9.4249]；本探索性队列上的配对J更高，未获得优势。

上述结论只针对当前队列和点式区间，不能因结构合法率上升就推论精度提升。若强固定linear_nearest不劣于搜索，应优先考虑简单流程；不为追求优势追加预算、调参或扩大候选池。

### 38.9 新旧C的可比范围

|mode|old_local_val_J|new_val_J|old_local_followup_J|new_followup_J|
|---|---|---|---|---|
|A|44.4508|44.4508|70.7038|70.7038|
|B|35.9740|35.9740|56.7988|56.7988|
|C|35.4581|36.2167|55.2837|58.1378|
|C-gap|36.8606|36.0530|59.6723|57.7746|
|R|34.5531|34.5531|54.1688|54.1688|

相同数据、掩码、J和949共同语法允许作上述历史数值比较，但新C不等于旧局部C：前者由共享规则/能力/兼容性/端口状态驱动，支持频数用于软偏好；后者是任务局部端口缺口和去重支持风险。新C-gap本身也使用公共路径能力缺口，不机械当作旧gap-only实现。历史时间不在同一时段采集、旧版未完整列setup成本，不能用新旧墙钟推断因果加速。

### 38.10 定向回归、文件验收与复现

最终相关测试100项，失败0、跳过0；没有重复无关全量回归。核验2102份实际预测文件SHA/回读、949规范空间、50运行预算、每次回调账目、原12CSV、复用掩码和训练状态、旧协议/冻结/结果哈希以及新源码/冻结指纹；冷缓存强固定路线两次真实执行逐值一致。环境沿用第37章，不安装新依赖，第三方GOLEM核心不变；旧GOLEM声明依赖与当前主数值栈的版本警告仍保留。

关键新增/修改：公共 `semantic_requirements.py`、`semantic_rules.json`、`router.py`、`golem_search.py`、`dag.py:NodeAdapter.decision_contract`；原 `temporal_reconstruction_dag.py`补实际方法支持声明，原 `mixed_gap_dag.py`只绑定运算含义；原生产 `raster_upload.py`和原北京runner消费同一核心。`run_beijing_unified_decision.py`只做复用、验收、冻结和编排，不另写选择策略。独立分析/验收工具、监控的必要参数扩展也保留源码指纹。

主要机器证据：`protocol_frozen.json`、`reused_artifacts.json`、`revision_evidence.json`、`mechanism_acceptance/`、`tests_mechanism.xml`、`tests_final.xml`、`candidate_contracts.json`、`call_chain_audit.json`、`guidance_consumption.csv`、`search/`、`factory_generation_accounting.csv`、`enumeration_summary.json`、`pre_followup_grammar_acceptance.json`、`selected_graphs_frozen.json`、`frozen_test/`、`fixed_validation_results.json`、`test_results.json`、`search_runs.csv`、`best_so_far.csv`、`paired_environment_intervals.csv`、`historical_comparison.csv`、`final_acceptance.json`。预测为实际队列NPZ及元数据/图JSON，与原任务一致，不将时间恢复冒称地理NetCDF降尺度。

复查（不会重跑搜索）：

```powershell
$env:ECODATA_BEIJING_MIXED_OUTPUT='D:/codex/EcoDataEngine/output/beijing_mixed_gap_20261020/unified_platform_v4'
.venv/Scripts/python.exe tools/verify_beijing_unified_decision.py
.venv/Scripts/python.exe tools/analyze_beijing_mixed_gap.py
```

完整重现应在独立新目录复用相同准备产物，按 `run_beijing_unified_decision.py prepare -> 定向pytest -> accept -> freeze -> run` 顺序执行；源码指纹改变则不能混入已冻结版本。研究流程技能仅帮助证据/防泄漏和配对分析，非算法或精度证据；使用的[Kassis等，Scientific Agent Skills，2026，当前v2](https://arxiv.org/abs/2609.00065)已核对，不把技能使用当作研究创新。

旧章节的源码哈希校验应使用其各自归档源码复现；当前公共源码已扩展，不能直接用当前源码绕过旧版哈希保护。报告生成控制台遇到旧GBK不能输出下标字符时，显式设置 `PYTHONIOENCODING=utf-8`，不改变报告数值。

### 38.11 最终验收口径

当前时间恢复路径的**平台调用链已实际贯通**，机制确实消费输入支持特征、能力规则、真实契约和端口状态并改变概率；生产栅格与实验共享核心API，旧精度路由未强行迁移。正式冻结后的50次搜索、949图参考及统一2017探索性复核均已执行。公共机制不具备仅凭输入数值就预测所有算法精度的能力，其他旧入口和声明型软目标仍有明确覆盖边界。

工程链路成功与科研精度优势分开：最终数值结论以上述固定强对照、五组配对结果及不确定性为准，无优势也保留。单一北京站点时间恢复、小而可穷举的候选空间、既有年份暴露、未知原始时区/小时统计窗口均限制外推，不能宣称跨尺度、多源或普适语义搜索已被证明。

## 39. 必需能力最终验收与缺口时间语义边界修复（2026-10-20）

### 39.1 核实与最小修改

核查第38章及实际消费者后确认两个边界问题。仅修改项目自身的契约、验收与必要调用连接，没有改变科学算法内核、冻结算法池、原始输入、历史协议或实验结果。本轮未执行搜索、数值穷举、全年匹配或整批实验，也未运行无关全量回归。

|修改位置|修复内容|
|---|---|
|`src/ecodata/algorithms/router.py:SemanticDecision.assess_graph/require_graph`|必需能力按具体输出路径及可见支持条件分别记为满足、确定缺失、未知；完整图确定缺必需能力时拒绝，不完整图保持可演化；最终验收遇UNKNOWN明确返回无法验收，不当作满足。|
|同上，分流/回退传播|按实际条件选择短/长分支；回退按主预测支持条件选择主/备路径，主路径未知时仅保留能证明的共同事实。不从其他分支借能力，也不因未生效分支缺能力而拒绝。软支持证据未知或软目标未实现，不变成硬性拒绝。|
|`src/ecodata/algorithms/pipeline.py:run_dag`|接收成对的公共decision/factory，数值执行前调用统一最终验收；拒绝与UNKNOWN均不进入数值计算。旧调用未提供任务需求时明确记`NOT_REQUESTED`，不冒称能力验收PASS，保留既有API兼容。|
|`tools/run_beijing_mixed_gap.py:Evaluator.execute`|无条件把公共decision/factory交给Pipeline，与A/B/C/R开关无关；未在北京入口另写能力判断策略。后验过滤仍只属于原实验定义的提前过滤。保存任务验收状态、失败阶段及是否进入数值执行；提前验收拒绝不再计为真实数值执行或质量失败，另记`task_acceptance_failures`。|
|`src/ecodata/algorithms/semantic_requirements.py`|公共时间轴检查统一微秒/纳秒存储，拒绝重复、非递增、缺失时间标签及未声明的数值时间；保留缺失记录数、名义网格时长及其单位，无法确定的时长保留UNKNOWN。|
|`src/ecodata/algorithms/mixed_gap_dag.py`|现有缺口分流明确声明阈值单位hours、要求规则3600秒间隔；实际执行复用相同时间检查，再按原缺失条数分流。非1小时或不规则输入明确拒绝，没有改写降采样/重建算法。|

能力矩阵是逐条件证明，不是全图能力并集。已知不适用仍可拒绝，但暂时未补齐的能力不会使所有中间图被淘汰。公共最终验收与实验后验开关已分离：即使A/R不启用提前过滤，北京实际执行入口也不能绕过最终任务验收。其他未接入decision的历史DAG入口只得到原数值/质量检查及`NOT_REQUESTED`，本轮没有声称全部旧入口都已完成任务能力验收。

### 39.2 当前时间支持范围

通用提取器可以识别规则30分钟和不规则时间轴，**但现有缺口长度分流仅支持规则1小时标签**；不把这一限制扩展为所有时间插值算法均不支持其他频率。分流在公共决策及实际执行两处都检查同一前提，不能只修改日志字段后照旧按“小时”处理半小时数据。

`gap_record_count`和兼容旧字段`length`的单位均为records；`duration_seconds`在规则时间轴下等于缺失网格槽数×标签间隔，不规则时为UNKNOWN。它不是由数据证实的观测平均窗口，也不推断时间戳代表小时起点/终点或任何时区。规则1小时数据中每条恰等于1小时，原阈值及`<=`边界不变。`datetime64[us]`与`[ns]`先统一到纳秒再比较间隔。

### 39.3 实际定向测试与错误记录

最终执行79项相关测试，失败0、跳过0，其中新增边界用例19项。覆盖：完整图缺必需能力、未完成图继续演化、UNKNOWN契约/匹配/时间前提、错误分支借能力、合法有效分支、未生效分支、软目标与未知软证据、统一最终验收及调用参数、规则1小时/30分钟/不规则、微秒/纳秒一致性、6/7小时阈值、重复/倒序/缺失时间标签。另复核已有统一决策、北京DAG、数值DAG、状态依赖与类型化生成测试；没有启动真实搜索循环。

初轮新测试曾因pandas返回只读时间数组而在构造不规则测试输入时失败，已改为显式复制，未改生产数据或屏蔽测试；首轮失败XML保留。有限复核脚本初次构造不完整图时在删边后再次读取`root_node`，被遗留索引节点变为多根列表；改为删边前保存原根、重建可达图。该故障发生在数值复核前，没有修改第三方GOLEM源码。

### 39.4 北京有限代表检查，不代替完整复验

仅复用第38章的2015公开受遮挡输入、既有训练状态、保存的图及2015引导控制，不读取私有目标队列，不计算新RMSE，不解封2017。修改前源码快照、文件指纹与检查结果集中在`output/decision_boundary_fixes_20261020/`。

|代表图|修改前→后：能力缺口|修改前→后：硬拒绝|新增明确最终状态|
|---|---|---|---|
|最近邻|0→0|否→否|PASS|
|线性＋最近邻回退|0→0|否→否|PASS|
|H₀|0→0|否→否|PASS|
|完整但覆盖失败的线性图|2→2|是→是|REJECT|
|历史小时中位数|0→0|否→否|PASS|
|缺field端口的中间图|5→5|否→否|INCOMPLETE；不能当最终结果|

六个代表的共同判断字段均一致。对2015存档“最近邻/历史中位数”二候选控制，重算软项开/关两种情况下的权重及归一化概率，与修改前及存档记录逐项完全一致。

实际数值复核只取公开输入中按原始顺序及缺失支持选择的3个样本（索引0、8、1），共2166个待填位置。最近邻、线性＋最近邻回退、H₀各进行修改前/后一次，共6次小规模DAG执行；原始预测与非负投影逐值一致，H₀对应的最终分流输出也一致，预测文件写出/回读通过。没有重新生成全年预测或人工遮挡，不把自然缺失当作真值评价。

另用一个明确标为单元契约标记的“必需但未提供能力”复现原缺陷：修改前完整图`missing_capabilities`非空却`hard_reject=False`，修改后明确REJECT；它不是新增科学算法或新实验指标。科学内核函数AST与修改前一致；受保护的旧协议、冻结选图、结果、2015输入及训练状态哈希未变。

### 39.5 结论限定、产物与复查

本次有限范围没有发现北京代表案例的决策、候选概率或恢复值发生实质变化，**不等于证明全部949图、全部提议或全年实验完全不变**。当前尚无依据要求立即重跑所有旧数值结果；第38章原负结果仍保留，不重新包装为修复版优势。

但新的最终验收会把部分失败提前归类为“任务要求拒绝/无法验收”，而不是进入算法后才失败；未来版本的真实执行次数、质量失败次数、运行成本及可能的预算轨迹需要按新口径记录。若要以修复版全流程作正式机制或成本结论，应另行冻结新协议并复核受影响对照，不能把旧源码版本的完整验收或计数直接移植到本版本。本轮按要求不启动该重跑。

关键证据：`source_before/`、`summary.json`、`representatives.json`、`weights.json`、`executions.json`、`bounded_predictions.npz`、`counterexample.json`、`first_tests.xml`、`tests.xml`（包含测试夹具首轮失败）、`tests_fixed.xml`、`tests_final.xml`。工具为`tools/check_decision_boundaries.py`，新增测试为`tests/test_decision_boundaries.py`；该工具本身没有搜索、穷举、训练或完整误差评价入口。

定向测试实际命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_decision_boundaries.py tests/test_unified_decision.py tests/test_beijing_mixed_gap.py tests/test_numerical_dag.py tests/test_station_fusion_dag.py tests/test_golem_typed_generation.py -q -p no:cacheprovider --basetemp=output/pytest_boundary_final --junitxml=output/decision_boundary_fixes_20261020/tests_final.xml
```

该命令是测试复现说明；已存在的本轮检查证据无需反复覆盖生成。修改前快照用于解释本次差异，旧正式实验继续由其归档源码和原协议复现。本章遵循科研证据审计边界，未用有限测试或代码检查声明整个系统无错误。

## 40. GOLEM真实数值搜索公共编排与页面集成（2026-10-21）

### 40.1 核实的缺口及迁移

核对第38、39章和实际调用者后确认：统一特征、规则、能力匹配及真实DAG执行已经存在，但创建优化器、变异包装、预算、图缓存、回调记账和选图仍在`tools/run_beijing_mixed_gap.py:search`。`GolemCapabilitySearch`调用的是旧结构评分后端，不是本次真实数值编排；没有把它改名冒充新流程。

本次沿用既有模块，新增最小公共编排`src/ecodata/algorithms/numerical_search.py`及现有小时任务适配`temporal_search.py`。前者直接调用官方`EvoGraphOptimizer/Objective/ObjectiveEvaluate`和既有类型化工厂，不复制优化器、不导入tools。后者承接原Evaluator的公开数据适配、统一决策调用、真实执行和独立数值评价，不另建算法偏好规则。原runner的`search`现在仅装载冻结输入、建立原分组评价器并提交；`search_request`是研究客户端，页面`process_temporal_request`调用相同的`submit_temporal_search`。

实际调用链：

```text
原有主页面 → 时间恢复图搜索 → CSV解析/结构化需求
研究客户端 → 已解析公开输入/冻结需求
                 ↓ 同一个 submit_temporal_search
只读识别 inspect_dataset + RawInputQualityGate + 时间/端口检查
→ SemanticFeatureExtractor.extract_series → AlgorithmSelector.decide/RequirementMapper
→ Registry能力/compatibility + 实际端口及历史状态契约
→ TypedGolemFactory未完成初始图 → 官方GOLEM
→ semantic_proposal_weights → 实际 random.choices 抽样
→ 完整图 require_graph → Pipeline.run_dag/NumericalDAGExecutor/质量门
→ 独立评价器真实误差 → 官方Objective → 预算停止/选图/产品导出
```

公共编排统一支持A、B、C、C-gap、R策略开关。A无软/缺口提议偏好，B增加共享后验提前过滤，C消费公共能力及软支持概率，C-gap仅消费能力缺口，R调用同一工厂随机生成完整图；所有完整图均执行相同最终验收。本轮只实际运行C的有界集成验收，没有重新比较这些策略的研究优劣。H₀不进入初始化、提议或奖励。

### 40.2 输入、评价与安全边界

首期范围为**规则1小时时间标签、非负PM2.5、明确单位`ug m-3`**，不是通用多源/空间自动化。时间窗口和时区未证实的元数据保留UNKNOWN；标签恢复不猜测小时平均起止。CSV需要`station,time,目标变量列`，各实体分别处理；重复、非递增、NaT、半小时及不规则间隔拒绝。仍沿用第39章记录数/小时阈值契约，不修改重建内核。

页面要求明确目标列、单位、双侧或仅过去访问、覆盖要求、支持偏好、质量确认及预算。历史状态须由用户显式提供；只读JSON可提供各实体24小时中位数和`training_end`，不能上传可执行pickle。KNN只接受已有合法受训状态，本页不加载任意模型二进制。缺实体、截止日期不早于输入、缺所需训练字段的候选会排除，并显示理由；不静默调用北京旧权重，也不自动在当前输入拟合历史模型。训练截止元数据是调用者提供的隔离声明，不是平台独立证明了该状态的全部训练来源。

三种评价情形严格分开：

|条件|公共服务行为|可作的表述|
|---|---|---|
|显式合法validation参考或独立误差评价器|参考仅留在评价对象；节点、特征和候选权重不接收标签。页面参考CSV为`station,time,truth`，对应位置必须已在输入中缺失；不自动造遮挡。|本候选和预算下验证误差选图，不承诺全局最优。|
|无真值，显式选择覆盖指标|`CoverageEvaluation`返回有限覆盖指标；质量门仍为硬条件，不输出RMSE/精度最优。|覆盖/可执行性选图；当前完整覆盖硬门下指标可能无法区分多个合法图。该模式有接口和定向测试，未另跑一轮真实无真值搜索。|
|无参考且未声明可计算指标|明确BLOCKED/说明所缺条件，保存失败记录。|不运行精度搜索，不回退结构分数冒充精度。|

完整确定缺能力拒绝；UNKNOWN不能冒充PASS；中间图允许继续演化；条件分流和回退按有效路径验收。最终门不随研究策略关闭。查询位置与目标数值分开，所有自然缺失仅作为待处理输入；未被合法参考覆盖的位置不用于误差计算。

### 40.3 搜索、产品及发现的真实故障

服务统一管理独立冷图缓存、节点缓存计数、提议/回调/唯一评价/真实执行/不完整反馈/最终验收拒绝/质量失败/延后回调，保留实际图、权重和抽样索引、轨迹及选图。目录已有请求或结果时明确拒绝重启，避免覆盖；页面只在按钮提交时执行，重绘和下载读取session_state结果。单进程互斥锁防止并发请求串扰全局随机状态，不新建调度系统。墙钟上限在不可中断科学内核返回后检查，不承诺强制杀死正在执行的内核。

选中图保存为`selected_graph.json`；候选预测为NPZ并回读；从实际DAG中间输出生成`processed.csv`，包含case、实体、原始时间、原值、处理值和缺失掩码。未预测成功的非评价位置仍保留缺失并报告产品覆盖率，不补造数据。完整产品与误差评价分母分别记录。请求、输入/状态指纹、源码指纹、决策和结果均可下载。服务计时包含识别/决策及最终产品导出，搜索阶段另计；CSV解析在实际页面单列计时，下面的脚本CSV验收没有另行采集解析耗时。

两处真实验收故障均保留了失败证据：

1. **提议上限边界**：第一次CSV验收提议30次后，原`limited()`连已生成待评价候选也拒绝，只有不完整反馈。修复为停止新提议，但允许其余评价/回调/时间预算内的已生成候选被评价；没有加预算、改种子或重写优化器。
2. **JSON历史状态存储类型**：JSON中位数列表虽形状正确，但旧预测器需要数组索引。公共任务适配层只读转换为numpy数组，原输入不修改、算法内核不改、没有重训。原失败`list indices must be integers or slices, not Index`及修复前后运行均保留。

### 40.4 真实有界验收结果

所有案例和预算在执行前写入协议，仅用2015公开受遮挡输入及原2013—2014训练状态。没有读取2017数值结果调策略，没有启动任务一、全量北京对照或949图数值穷举；类型化工厂内部推导有限端口语法不等于执行数值穷举。

二入口一致性验收固定案例索引`[0,8,1]`、种子11、C、最多5个唯一候选评价/50个外层提议/100回调/60秒。原合法评价位置1734个，不把2166个全部缺失位置都当作已知真值。

|入口|唯一评价|不完整反馈|真实执行|外层提议|后验拒绝|图缓存命中|质量失败|最佳验证RMSE|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|页面处理函数|5|2|3|30|1|0|0|16.410845|
|研究客户端|5|2|3|30|1|0|0|16.410845|

两入口各有6次原始回调、1次预算延后。去除计时后，语义决策、候选域、每条已评价图、权重/概率、所选索引和最终图完全一致。官方优化器从未完成图出发，得到并真实执行三条不同分支/回退结构，实际误差进入官方fitness；没有预置完整图替代搜索。选中产品17688行、有限覆盖100%、CSV和NPZ回读通过。页面侧决策/准备约0.531秒、搜索约0.666秒；研究侧约0.009/0.586秒，首次初始化差异不能解释为策略性能优势。该成对验收源码留存于`source_pair_acceptance/`；后续只对触及的预算边界、JSON适配和入口包装进行定向复测，没有重复全部对照。

另走**真实CSV解析**的最新交付验收：固定原案例8、168小时、6个原有合法参考点；种子11、4评价/30提议/80回调/30秒。`csv_upload_state_fixed`最终4评价=2不完整+2真实执行，6回调=4评价+2延后，质量失败0，缓存命中0，最佳验证RMSE=5.930149；服务端完整耗时约1.041秒，168行产品覆盖100%、回读PASS。两个不同案例集合的RMSE不可相互排名；六个参考点只能验证接口和真实评价，不支持精度研究结论。

`bounded/`是首次成功的二入口试跑；`bounded_final/`保留成对验收及CSV故障/修复：`csv_upload`为预算边界失败，`csv_upload_fixed`、`csv_upload_delivery`保留JSON适配修复前的失败候选，`csv_upload_state_fixed`为最终交付。没有删除负结果或换预算补救。

### 40.5 测试、界面和交付边界

本轮累计通过**94个不同定向测试**，不把重复复测计为新增覆盖：主体92项通过；后续受影响56项通过（含新JSON状态用例），主页面导航另1项通过。其余未受最后格式适配影响的已通过测试没有再次运行。初轮两项失败来自旧测试仍检查/替换runner内已迁出的实现，已迁到公共消费者并保留预算断言及首轮失败XML。覆盖错误时间、未知契约、缺必需能力、无评价条件、误用test、错配状态、真值隔离、缓存/拒绝/执行分账、提议顶格后的待评价批次、页面重绘不重启、真实产物ZIP及回读。没有无关全量回归。

Streamlit AppTest验证了结果展示、决策展示、真实NPZ/CSV下载包及刷新不重启；另从现有`app.py`导航进入新页面并验证上传控件。**人工浏览器上传点击验收：未运行**，不把AppTest等同于人工浏览器验收。通过原`run_dev.bat`启动后，侧栏选择“时间恢复图搜索”；现有“单源栅格重构”仍是原学习路由，未改名为GOLEM。空间、多源、模型专用任务尚未迁入此提交服务；旧结构搜索入口及其他历史数值入口也没有被宣称全部迁移。

主要文件：`src/ecodata/algorithms/numerical_search.py`、`temporal_search.py`；`src/ecodata/interface/temporal_search_page.py`、`app.py`；北京runner现在是公共服务客户端；`tests/test_platform_numerical_search.py`和受影响的原北京测试；有界工具`tools/accept_platform_numerical_search.py`。证据集中于`output/platform_search_service_20261021/`，包括修改前/验收版本/交付版本快照、协议、输入与状态指纹、决策、实际图、NPZ、CSV、失败日志及XML。原第38、39章保护的协议、冻结图、结果、公开输入和训练状态哈希仍一致；没有修改第三方GOLEM源码或原始数据。

复查已有证据不需重跑搜索。若确需重现有界验收，应为工具指定新的独立输出目录，使用同一固定协议；已有目录的重启会明确拒绝。定向测试复现为`.venv/Scripts/python.exe -m pytest tests/test_platform_numerical_search.py tests/test_beijing_mixed_gap.py tests/test_unified_decision.py tests/test_decision_boundaries.py -q -p no:cacheprovider --basetemp=output/pytest_service_recheck`；本轮实际XML见`tests_delivery.xml`，主导航见`tests_navigation.xml`。

最终回答：**在上述明确的小时PM2.5任务和合法评价条件内，已由平台公共模块贯通识别、能力映射、匹配、官方GOLEM搜索、完整图验收、真实数值执行和反馈，并有真实端到端记录；尚不是任意任务通用自动化。**本轮不检验C胜过其他策略，不更新第38章精度优劣结论，也不把工程集成成功当作科研创新成功。

## 41. 空间、时间、多源统一公共搜索集成与有界验收（2026-10-21）

### 41.1 结论与执行范围

本轮将第40章已有真实数值搜索编排扩展为 `PlatformSearch/v2`，**空间、时间、多源三类均已有上传字节到公共决策、官方GOLEM、真实DAG、质量门、独立评价及NetCDF回读的执行证据**。主页面、工具包JSON客户端、原时间恢复客户端共用公共服务，不要求用户选算法或完整图模板。

这不是任意文件、任意变量都能自动处理的承诺。默认公共注册表80项中，58项原状态为available、22项仍非available；本轮当前验收协议实际执行14个注册ID，其余逐项区分公共绑定、条件资源、研究目录和模型专用层。没有把80个ID都标为本轮运行成功。没有启动任务一、北京全量对照、949图数值穷举、正式A/B/C研究、模型训练或新数据下载；没有删除数据。

证据根目录：`D:/codex/EcoDataEngine/output/platform_integration_20261021/`。当前主验收采用空间 `runs/spatial_*_semantic_final`、时间和多源 `runs/{temporal,multisource}_*_delivery`；早期试跑、失败及修复前结果全部保留。最终成对核对见 `entry_parity_final.json`，不能用早期未收紧统计语义的候选数量代替最终候选域。

### 41.2 原有模块、最小连接及实际调用链

原来已有：69项默认Registry、上传识别、SemanticFeatureExtractor、RequirementMapper、能力目录、AlgorithmSelector.decide、三态路径能力验收、类型化GOLEM工厂、官方优化器、共享数值搜索循环、Pipeline/NumericalDAGExecutor、双层质量门，以及独立的空间/时间/站点融合科学实现。第40章服务仍主要接受小时恢复任务，空间和站点实验尚未由统一上传协议进入。

本轮新增的两个公共模块不是新算法：

- `src/ecodata/algorithms/platform_search.py`：字节输入、角色和元数据声明、任务分派、现有Registry插件的命名端口绑定、独立评价协议、真实产品导出，以及公共JSON工具包客户端。
- `src/ecodata/algorithms/station_search.py`：将已有严格站点/AOD DAG接到同一服务。数据准备、训练状态、协变量和查询位置均经真实父边传递；科学运算仍调用旧PyKrige/MetPy、交叉拟合残差及残差加法。

实际链路为：

`search_page.process_search_upload / platform_search.submit_file_request / 旧submit_temporal_search客户端`
→ `platform_search.submit_search`
→ `recognise_file + RawInputQualityGate`
→ `SemanticFeatureExtractor + RequirementMapper`
→ `AlgorithmSelector.decide + Registry.compatibility`
→ 对应类型化任务工厂与端口、状态契约
→ **同一个** `numerical_search.run_numerical_search`
→ 官方 `EvoGraphOptimizer / Mutation`（R为同一公共语法的随机完整图参照）
→ `SemanticDecision.require_graph`
→ `UnifiedAlgorithmPipeline.run_dag → NumericalDAGExecutor`
→ 现有科学算法、结果质量门
→ 只在独立评价侧读取参考值 / 计算已声明工程指标
→ 真实Objective反馈、冻结选图、NetCDF写出并回读。

公共服务不导入研究runner，也未复制GOLEM源码。既有隔离环境的科学计算桥接脚本仍由原插件调用，不是用实验runner充当生产后端。北京runner经第40章的公共客户端间接进入本服务，历史归档不改。旧 `GolemCapabilitySearch` 仍是结构评分后端；旧栅格学习路由仍是另一种模式，没有改名为数值GOLEM。其他历史空间/多源研究runner保留原协议，**未声称所有历史runner已逐一迁移**；新的三类页面/工具包请求必须走上述公共入口。

对既有公共模块的修改限于：兼容性检查异常转为逐候选可追溯排除；增加通用compute/fit端口证明，状态和能力仅从实际输入路径继承；增加缺失的空间、时间、地形、土层、协变量、特征、统计和极值能力规则；补齐现有包装算法的能力声明；时间旧API改成兼容客户端。没有改插值、克里金、神经算子、偏差订正或第三方优化器数学内核。

### 41.3 全库逐项接入矩阵及条件

统计口径：默认Registry 69项；复用原栅格上传注册6项及旧时间恢复5项，公共默认共80个ID；用户显式提供AOD权重时，另注册**已有**的 `raster_task.liif_aod`，不将其算成默认第81个可用模型。算法别名和包装器不是独立科学方法，不能把注册数解释为搜索空间的有效行为数。

完整机器可读逐项矩阵为 `algorithm_library_matrix.json / .csv`。每行包含真实ID、状态、层、科学作用、源码文件及SHA256、来源/许可、输入类型和语义、尺度、输入/输出契约、所需观测/辅助/训练/权重、真实compatibility/run函数、公共绑定、任务族、能力消费者、状态依赖、各实际请求的适用性/排除原因、当前执行证据及历史声明证据。单位和时空支持没有统一伪造默认值，具体条件以该行输入契约、前提及源码compatibility为准；缺声明的部分保留条件性，不推断“支持所有单位/CRS/倍率”。`model_adapter_layer.json`另存原模型Profile，未混入通用候选池。

下表“实”只指当前有效验收目录真实执行并回读；“条件”指公共执行绑定已存在，仍须满足该算法实际条件，**不表示本轮逐一真实运行**；“目录/阻塞”不进入可执行搜索。“门”是复用公共质量边界，而非自由预测节点。

|实际ID|公共路径/作用|本轮状态与必要边界|
|---|---|---|
|baseline.temporal_calendar_knn|小时恢复状态端口|条件：显式匹配的合法历史状态，不静默复用北京状态|
|baseline.temporal_ffill|公共时间插件run|条件：已移除目标值、时间顺序及信息访问限制|
|baseline.temporal_hour_median|小时恢复状态端口|条件：显式历史中位数状态及时间隔离|
|baseline.temporal_linear|公共时间插件run/旧恢复DAG|条件：真实保留观测、适用时间支持|
|baseline.temporal_nearest|公共时间插件run/旧恢复DAG|条件：时间支持及合法外推契约|
|bias.xsdba_dqm|公共偏差订正run|条件：train配对、同单位、应用时间不重叠、允许拟合|
|bias.xsdba_eqm|公共偏差订正run|条件：同上，明确量化参数与样本量|
|bias.xsdba_extreme|公共极值订正run|条件：显式extreme_value_correction任务、阈值单位、规则序列和足够尾部数据|
|bias.xsdba_qdm|公共偏差订正run|条件：train配对及独立应用时段|
|downscale.climatelearn_t2m|公共学习型空间run|条件：显式同意已有匹配权重、既有区域air_temperature/K、6×6→24×24，非任意变量|
|downscale.granite_wxc|研究目录|阻塞：checkpoint_missing，无可执行公共后端|
|downscale.qp|公共空间约束重构run|条件：目标网格、合法训练观测及现有QP参数|
|downscale.topopyscale|研究目录|阻塞：python_version_blocked|
|features.galileo_nano|公共feature_encoding run|条件：官方完整输入组、显式权重同意；输出embedding而非PM2.5/LAI预测|
|fusion.idw|公共站点到网格run|条件：训练站、单位、明确CRS与目标网格|
|fusion.precipitation_fusion|研究目录|阻塞：checkpoint_missing|
|fusion.primer|研究目录|阻塞：license_unverified及实际运行条件未满足|
|fusion.quality_aware|公共同变量融合run|条件：同物理定义、来源对齐、train拟合/内部validation定权、外部评价须隔离|
|fusion.residual|公共run及站点状态DAG|实：训练状态→交叉拟合残差→空间估计→与同一状态背景相加|
|fusion.uncertainty_weighted|公共同变量融合run|条件：可核实的同变量定义及明确正不确定性，不凭空排名来源|
|neural.fno|公共学习型run|条件：显式允许训练、原观测/内部验证契约、参数与资源齐全；本轮未训练|
|neural.multiscale_uno|公共学习型run|条件：同上，非任意变量和任意倍率保证|
|neural.observation_guided_uno|公共学习型run|条件：同上，真实观测及辅助输入|
|raster_task.coarse_consistency|公共上游产品→QP后处理|条件：真实观测算子和原粗场父端口；不是AOD总量守恒定律|
|raster_task.nonnegative|公共上游产品→非负后处理|条件：真实上游网格；另有合成边界数值执行测试，不计真实数据算法数|
|raster_task.pchip|公共空间run|实：连续二维栅格及明确目标坐标|
|raster_task.spatial.bilinear|原双线性包装run|实：连续场；边界延拓规则仍按原契约记录|
|raster_task.spatial.nearest|原最近邻包装run|实：同上，不将包装器重复计成新数学方法|
|raster_task.spatial.xarray_cubic|原三次包装run|条件：有限二维连续场及匹配网格|
|research.climatediffuse|研究目录|阻塞：原权重/许可/数据契约证据，未伪造可用|
|research.consistency_climate|研究目录|research_only，未接数值后端|
|research.cru|研究目录|阻塞：所需训练/不规则序列契约未取得运行证据|
|research.dsfno|研究目录|research_only，原流体权重不能冒充环境变量模型|
|research.fords_full|研究目录|阻塞：完整流程依赖/权重，不等同于已有局部ForDs组件|
|research.geofar|研究目录|阻塞：任务训练数据/权重|
|research.phydl_nwp|研究目录|阻塞：匹配气象训练输入及模型|
|research.qmapp_extrap|研究目录|阻塞：独立匹配CDF与应用契约|
|research.wassdiff|研究目录|阻塞：降水任务权重及相应条件|
|scale.semantic_stack|公共feature_stack run|条件：变量分组与物理语义；输出特征集，不冒充融合预测|
|soil.mpspline2|公共soil_profile run|条件：cm深度、真实剖面、R运行时及原缺失处理规则|
|spatial.bilinear|公共空间run|实：连续规则网格|
|spatial.conservative|公共空间run|实：既有cell_mean/flux契约；不默认支持每像元总量|
|spatial.cubic|旧研究目录|research_only/CatalogOnly，不能因名称近似就当xarray_cubic|
|spatial.nearest|公共空间run|实：合适语义；粗分类聚合不能被最近邻抽样替代|
|spatial.xarray_cubic|公共空间run|实：连续场与适用边界|
|spatial.xarray_fractions|公共类别组成run|条件：明确category_fractions输出，不冒充整数类别|
|spatial.xarray_max|公共spatial_statistics run|条件：明确请求max统计，不混入普通重采样|
|spatial.xarray_mean|公共空间/统计run|条件：既有聚合网格及mean定义，当前正式验收未选中|
|spatial.xarray_median|公共spatial_statistics run|条件：明确median统计；修复前试跑保留但不计当前真实验收|
|spatial.xarray_min|公共spatial_statistics run|条件：明确min统计；合成边界测试核验输出cell_min元数据|
|spatial.xarray_most_common|公共分类聚合run|条件：真实整数类别、聚合目标；本轮无真实分类文件端到端验收|
|spatial.xesmf_bilinear|条件后端|阻塞：ESMF/ESMPy依赖缺失|
|spatial.xesmf_conservative|条件后端|阻塞：依赖及显式像元边界条件|
|spatial.xesmf_nearest|条件后端|阻塞：ESMF/ESMPy依赖缺失|
|station.metpy_barnes|公共站点run/残差端口|条件：投影、覆盖及原Barnes尺度参数|
|station.metpy_cressman|公共站点run/残差端口|实：合法残差分支，有限支持域|
|station.metpy_linear|公共站点run/残差端口|条件：有效站点几何和覆盖域|
|station.metpy_natural_neighbor|公共站点run/残差端口|实：实际训练残差空间估计|
|station.pykrige_ordinary|公共站点run/fit→apply|实：本轮用于残差空间估计及既有状态路径|
|station.pykrige_ordinary3d|公共三维站点run|条件：实际三维坐标及原PyKrige契约，本轮未验证|
|station.pykrige_regression|公共协变量站点run|条件：真实辅助字段和合法拟合输入，本轮未验证该插件模式|
|station.pykrige_universal|公共站点run/外部漂移DAG|实：AOD实际进入指定漂移数组，不是只在日志出现|
|station.pykrige_universal3d|公共三维站点run|条件：三维坐标、漂移变量与合法目标|
|super_resolution.diin|研究目录|research_only；无可执行后端不伪造|
|super_resolution.liif|通用研究目录|research_only；不同于已有任务限定v5.liif/AOD本地插件|
|super_resolution.sentinel5p|研究目录|license_unverified及传感器契约未落实|
|super_resolution.thera|研究目录|gpu_required，尚无本轮环境变量执行证明|
|super_resolution.zssr|研究目录|research_only，需要适当训练条件|
|temporal.dualtimesfield|公共不规则时间run|条件：显式允许训练、真实时间/缺失掩码、原512样本上限|
|temporal.semantic_resample|公共时间run|实：真实累计降水与瞬时气温分别执行，不把累计量当瞬时值|
|temporal.xclim_statistics|公共时间统计run|条件：原运算、单位、频率和统计定义，不自动变成恢复算法|
|terrain.fords_pressure|公共terrain_pressure_adjustment run|条件：气压Pa、高程m、温度K及相应参考辅助字段|
|terrain.fords_temperature|公共terrain_downscaling run|条件：真实DEM/气象网格及原3×3等前提，本轮未新增运行证明|
|v5.emiproc_spatial|独立model_preprocessing任务|条件：显式模型/opt-in、投影面积网格和kg总量；非通用随机节点|
|v5.emiproc_temporal|独立model_preprocessing任务|条件：明确日总量、24个UTC时段、归一化时间比例；未新增原生验证|
|v5.liif|公共学习型空间run|条件：同意已验证本地ERA5/K权重、原区域和1.5—4倍支持，非LAI/NPP|
|v5.pyresample_bilinear|公共curvilinear_regridding run|条件：二维经纬度、目标网格、正radius_m，非守恒重网格|
|v5.pyresample_nearest|公共curvilinear_regridding run|实：既有真实USGS DEM二维经纬度输入→规则网格|
|validation.result|公共ResultQualityGate边界|门：实际公共执行复用同一质量门，不作为预测器或精度证据|
|vertical.metpy|公共vertical_interpolation run|条件：真实垂直坐标/单位及原插值支持，本轮未新增运行证明|

另有条件项 `raster_task.liif_aod`：只接受显式AOD本地权重路径、维度/单位/区域及1.5—3倍契约；没有静默加载，也没有新训练。各学习算法均需显式资源许可/同意，提供AOD权重不能因此授权其他模型。没有新增第三方算法包、下载权重或改变原34项历史协议。

### 41.4 语义、硬约束、软目标与图生成的实际消费者

结构化请求支持多个角色输入、目标变量/单位/性质、网格、时间频率与区间语义、训练/辅助/目标网格资源、权重同意、评价协议、策略和预算。原始科学字段不从文件名猜；文件名仅保留追溯，匹配依据是读取后的字段和明确声明。

1. **硬性输入安全**：单位冲突、CRS冲突、未知必要变量语义、错误时间区间、原质量门BLOCK先拒绝；用户额外提交但无消费者的硬约束明确返回“No implemented hard-constraint consumer”，不静默忽略。
2. **必需能力**：完整图逐输出端口/祖先路径验收；缺能力拒绝，UNKNOWN返回无法验收，未完成图允许演化。最终验收先于数值执行且不受A/B/C/R策略关闭。插件缺能力说明不能凭算法名称推断其有效。
3. **任务专用物理契约**：规则网格单调性由原重网格compatibility验证；曲线网格由V5实际二维坐标/目标/radius契约验证；累计/平均区间由原时间语义算子及真实bounds验证；土层、垂直、地形、偏差配对由各现有插件验证。硬要求名称不是自动执行一切物理定律的通用DSL，只有实际消费者已存在的声明才能获得相应证明。
4. **统计与输出表示**：min/max/median须明确请求空间统计，不能混入普通连续场重采样；输出元数据标明cell_min等含义。类别组成须明确请求category_fractions，不冒充整数类别。每像元总量若缺面积/积分分配契约，通用空间入口明确拒绝，不能借“conservative”名字放行。排放总量留在显式模型预处理层。
5. **状态和分支**：通用compute节点仅从明确主输入继承能力；fit状态通过实际父边传递。站点背景、残差与相加沿用原拟合指纹检查和站点交叉拟合，不从其他分支借状态/能力。训练、查询和AOD分别是真数据源叶节点，不是新增算法。
6. **实际提议权重**：公共 `semantic_proposal_weights` 的能力缺口下降进入 `random.choices(weights=...)`；保存归一化概率、候选图指纹及chosen_index。既有retained_observation_support软项仅在恢复场景有证据时消费。粗尺度一致性偏好、覆盖/外推风险、来源追溯、模型网格偏好、未实现的成本偏好等**不声称全部影响了提议概率**；`unsupported_soft_objectives`明确列出未消费项。追溯信息虽被保存，不等于已成为优化偏好。数值精度始终来自实际独立评价，不由静态语义预测。

不是所有数据都开放全部算法。默认禁止未授权训练、未确认权重、错误物理来源加权。多份同变量栅格需显式一致physical_definition；AOD等协变量走auxiliary或station_covariates角色，不可改名为浓度参加同变量平均。特殊站点任务目前保留原台湾EPSG:4326工作约定、EPSG:32651距离、至少15个训练站及单小时契约，不能直接宣称适用于任意地区/任意频率。

### 41.5 真实有界端到端验收矩阵

提前冻结的主协议：种子11、最多8次唯一评价/80次外层提议/150回调/30秒；输入与来源SHA见 `protocol.json`。从已有真实文件取小窗口，未按误差挑日期。数据上传解析、特征、匹配、实际图、权重抽样、边契约、质量、参考侧评价及文件回读均保存。不完整反馈计入唯一评价但不冒充数值执行；缓存和延后回调另计。

|案例|实际输入与约束|数值执行/未完成|结果与范围|
|---|---|---:|---|
|连续栅格空间|已有LGHAP真实AOD训练tile，16×16窗口按既定取样得8×8上传，目标6×6且在覆盖域内；单位1、cell_mean、明确CRS|7 / 1|8次唯一评价、12回调、78提议；7个合法图、5种数值行为；覆盖100%、质量通过、NetCDF回读通过。评价是覆盖，不是独立精度。|
|累计量时间|既有真实ERA5 tp单位置24小时，原accumulation_comment支持1小时结束标签；保留真实[t−1h,t] bounds，目标6h累计|1 / 1|只有semantic_resample一条有效路线，4个6小时间隔；不把累计量当瞬时插值，不声称搜索优越。|
|多源多分支|已有台湾非test匹配表，按时间先后及覆盖/凸包规则固定首个合格小时；46训练站、9验证位置；观测、查询、真实AOD三份上传|4 / 3|8回调、80提议、25生成拒绝、8后验拒绝、1图缓存；直接UK及UK状态/交叉拟合残差/OK或MetPy/残差相加实际执行；9点固定分母、回读通过。|
|二维经纬度空间|已有真实USGS DEM二维经纬度文件及显式规则目标网格，radius_m=20000|1 / 1|公共入口自动识别曲线网格任务，真实调用pyresample_nearest，回读通过；未声称守恒或更高真实精度。|
|瞬时量时间对照|同一已有ERA5文件的t2m；原GRIB_stepType=instant、GRIB_units=K；显式瞬时定义，目标6h|1 / 1|实际执行插值/采样时间路径；需求不含temporal_accumulation，与tp累计规则不同。把真实tp错误声明为instantaneous被明确阻止。|
|分类对照|合成小网格，仅用于契约边界测试|非真实数据验收|实际改变required能力和合法候选；分类聚合、组成与连续插值分开。本轮没有新增真实分类文件端到端结果，不标为真实分类PASS。|

空间最终7个执行图采用conservative、nearest、PCHIP、双线性包装、xarray_cubic、原bilinear及最近邻包装，别名数值去重后5种行为。没有将本轮早期误放行的median试跑算成当前普通重采样成功。

多源评价只有9个验证点，直接UK所得最佳RMSE=**7.747969**、MAE=**6.328378**、Bias=**0.550705** μg/m³。其他真实路线RMSE为7.793771、8.432839、8.351744。它们用于证明真实数值反馈及非等价分支可执行，不足以支持精度创新或普遍优劣结论。使用原冻结非test数据，未读取封存测试站目标；时间及站点基准仍是原有明确工作约定，不改写为作者定义已证实。

三类成对入口的决策、候选域、权重/概率/抽样索引、评价队列、已评价图、最终图及每个成功产品数组完全一致，计时字段不参与相等比较。成对使用独立冷图缓存，训练状态和节点缓存另记。

|主案例（工具包一侧）|准备秒|搜索秒|服务总秒|图缓存|质量失败|结论|
|---|---:|---:|---:|---:|---:|---|
|空间最终协议|0.625|2.523|3.156|2|0|仅覆盖指标，合法图间无观测到的目标差异|
|时间累计|0.081|0.613|0.701|0|0|单路线、无优化区分度|
|多源|0.105|1.706|1.835|1|0|真实RMSE反馈；不是科研比较|

服务准备成本包含实际字节解析、质量检查、语义/兼容性与图绑定，不藏在搜索耗时之外。文件读取/浏览器上传及ZIP下载本身不包含在上述服务计时内；各次初始化与冷暖库差异不解释为策略优势。墙钟边界在正在运行的科学内核返回后检查，不承诺硬杀死内核。

### 41.6 五种策略及无真值评价条件

A/B/C/C-gap/R仍由同一个 `numerical_search` 实现并显式配置。A关闭语义提议偏好；B加后验契约过滤；C加共享能力缺口/可用软证据引导；C-gap关闭可独立定义的软引导；R在同一有限任务语法中随机生成完整图。所有组都受相同最终能力、执行安全和质量门约束。

另做一次**配置连线检查**，不是研究对照：同一真实空间输入、种子11、各最多3次唯一评价/20次提议/80回调/10秒，冻结于 `strategy_wiring_final_protocol.json`，不追加好种子或扩大预算。

|配置|实际唯一评价|真实数值执行|提议|图缓存|是否选出合格图|
|---|---:|---:|---:|---:|---|
|A|1|0|20|0|否：预算内仍为未完成图|
|B|1|0|20|0|否：预算内仍为未完成图|
|C|2|1|20|0|是|
|C-gap|2|1|20|0|是|
|R|3|3|5|2|是|

五组候选域一致，异常字段均为空，预算停止有记录。A/B此小预算负结果保留；不能写成所有策略均完成数值执行，也不能据此证明C优于A/B/R。初始图不完整，未用人工赢家初始化。旧日志里“小空间不能补足种群”与预算停止保留，不是吞掉异常后伪造完整搜索。没有再次触发1001次随机图失败循环。

评价协议明确区分：有合法validation参考时，只在评价对象中计算真实固定队列RMSE/MAE/Bias；无真值且用户显式声明coverage或coarse_consistency时，执行真实工程指标，不输出“精度最优”；无适用评价资源则BLOCKED/告知尚不能排序，不回退结构分数，也不自动制造遮挡。参考不能声明为final test；预测缺失不得偷偷删掉困难参考点后排名。所有图相同指标时返回NO_OBSERVED_OBJECTIVE_DIFFERENCE。数值质量PASS不等于仪器、地理定位或环境精度已证实。

### 41.7 页面、工具包、测试及失败修复

主侧栏新增“统一算法图搜索”，原转换、模型准备、学习路由和小时恢复兼容页保留。页面支持多文件及角色、目标变量/单位/性质、网格/时间结构化要求、训练/权重显式许可、独立参考和工程指标、预算、失败与排除、产品/图/决策下载。只有按钮提交会启动服务；session_state保留结果，重绘/下载不重复搜索。语义、规则和提议策略不在页面另写。模型原生输入仍使用原模型模块，不将普通NetCDF当成模型原生文件。

**实际浏览器验收已执行**：本地 `http://127.0.0.1:8502/`，真实浏览器文件选择器上传 `inputs/spatial.nc`、填写科学目标、提交C模式小预算、看到成功及停止状态、点击下载ZIP。真实浏览器运行目录 `output/platform_search_requests/fe4ec7ed797c413d930cf63adc243113/`，5次唯一评价=1未完成+4真实执行。ZIP保存并独立打开，4个NetCDF实际可读、有限值/单位/网格检查通过，CRC检查通过；SHA256为 `d92d04be822bf3eab0a5a64afc07d5f5839a975c7be8b3f1a63307127b39be92`。`browser_acceptance.json`、`browser_download.zip`、`browser_result.png`和后续紧凑显示截图`browser_result_final.png`为证据。通过浏览器Rerun保留同一结果及计数，没有再次提交。此浏览器案例在统计语义收紧前运行，产品为最近邻；后续只是同一产品重绘，不冒称已经在浏览器重跑了新候选域。时间/多源做了真实成对页面处理函数验收，没有把它们描述为另两次浏览器上传。AppTest另行测试新页面展示、下载按钮及重绘不重启，与浏览器验收分开。

实际修复及证据：

- 个别算法compatibility对不相关结构抛异常，原来可能中断整库筛选；现在记录到该候选，其他合法算法继续，原调用链保留。
- 既有包装算法/约束和部分聚合缺公共能力声明，补声明及真实产品→约束端口；算法数学不变。
- 初次站点预检错误地额外要求AOD非负，与原已冻结筛选产品允许的小负检索值冲突；删除这条新增且无依据的阈值，保留有限值/覆盖规则，PM2.5非负约束不放松。失败在 `multisource_toolkit/failure.json`，修复及后续运行保留。
- 曲线网格科学任务曾被附加普通规则网格要求，后处理探测也不应对二维地理坐标直接interp；现已将相应适配限定在真正的规则网格，真实DEM公共搜索通过，未重写pyresample。
- 空间min/max/median及类别比例不能与普通数值场/类别码输出混称，收紧输入契约、科学目标和表示，补输出统计元数据，重新执行受影响的空间成对案例；时间和多源未因这个修复重跑。
- 数值行为指纹不再包含算法名称等说明属性，避免别名制造“新行为”；用户新增但未实现的硬约束、未知/错误资源均明确拒绝。
- 一次新增测试用例将工厂返回的Node误当Graph，测试包装修正为OptGraph；失败XML保留，没有据此屏蔽业务异常。

定向测试：主受影响批次 `tests_delivery.xml` 为**141通过、0失败、0跳过**；后续边界批次48通过，统计/规则受影响批次 `tests_semantic_final.xml` **102通过、0失败、0跳过**，最终统计元数据用例1通过；最后 `tests_release.xml` **20通过、0失败、0跳过**，覆盖重复位置的任务/硬约束声明冲突与非负布尔类型检查。去重后累计**146个不同用例通过，尚未修复的失败0**；不把重复复测叠加计数。`tests_post.xml`的旧失败保留。仅运行相关模块/库契约测试，没有无关全量回归。现有flox的NumPy时间单位DeprecationWarning保留，未屏蔽。

覆盖单位/CRS/UNKNOWN、缺评价、缺训练状态/权重/辅助、小时/半小时/不规则边界、累计区间、类别/统计差异、完整图确定缺能力/未知契约、跨分支状态所有者、隐藏参考隔离、同固定队列、缓存/拒绝/执行/预算分账、实际算法及文件回读、页面重绘。半小时累计聚合通过的是原语义算子边界测试，**没有把第39章只支持规则小时的缺口分流扩展成半小时支持**。

### 41.8 复现、历史保护和最终边界

主要文件：`platform_search.py`、`station_search.py`、`interface/search_page.py`；小范围修改 `router.py`、`semantic_requirements.py`、`semantic_rules.json`、`capabilities.py`、`raster_reconstruction_task.py`、`open_source_spatial.py`、`temporal_search.py`、`interface/app.py`及旧时间页范围说明。公共优化循环复用 `numerical_search.py`。相关测试 `tests/test_platform_integration.py`；有界验收/盘点/回读工具为 `tools/accept_platform_integration.py`、`audit_platform_library.py`、`verify_platform_integration.py`，不是生产搜索后端。

修改前5个关键公共文件在 `source_before/`；旧时间服务、能力和栅格实现的原交付/历史快照亦保留。其他文件不能凭本轮交付快照反称有逐字节修改前取证。实际阶段源码分别归档 `source_delivery/`、`source_delivery_final/`、`source_delivery_semantic_final/`，最终发布版本为`source_release/`及对应清单；每次请求另记所用核心源码SHA，旧源码/实验未覆盖。当前工作目录没有Git仓库元数据，因此使用实际源码SHA而不编造commit。三个复用来源文件SHA复核一致，见 `protected_inputs.json`；旧第40章manifest保留并记录读取指纹。没有删除或修改用户原始Excel、京津冀、气象、WRF、UCI、台湾包或旧实验。

环境：Windows11、Python3.12.14，原`.venv`及既有GOLEM0.5.0隔离运行时，实际包版本见 `environment.json`。未安装新依赖。常规启动仍用现有主页面；本轮浏览器额外使用8502本地端口验证，不暴露外网服务。

工具包现在可以直接调用Python API `ecodata.algorithms.platform_search.submit_search`，或提交 `toolkit_request.json` 这种结构化文件，不进入实验runner：

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m ecodata.algorithms.platform_search output/platform_integration_20261021/toolkit_request.json output/platform_search_requests/另一个未使用的请求目录
```

它只读取请求中的输入路径/用途、科学任务、评价资源和预算，并调用同一个submit_search；已有输出目录拒绝覆盖。`tools/verify_platform_integration.py parity --suffix delivery --spatial-suffix semantic_final --output-name 另一个核对文件.json`只核对已保存真实产物，不启动搜索。重新跑有界验收时必须显式给新的suffix，不覆盖现有结果；不应把本节复现命令自动升级成正式研究实验。

上述工具包命令已实际运行到新目录 `toolkit_cli_release/`：种子11、原固定3评价/20提议小协议，2次唯一评价=1未完成+1真实算法执行，质量失败0，真实产品回读通过并选出图。它也核实了最终发布前新增声明一致性检查未阻断既有合法请求；没有因这一前置检查重跑未受影响的三类成对实验。

最终验收：

|事项|结论|
|---|---|
|空间、时间、多源同公共编排且有真实数值/回读|已通过指定小案例|
|页面处理函数/工具包共用、同种子决策/权重/结果一致|三类均通过|
|官方GOLEM从未完成图构图并执行不同候选|空间及多源通过；累计时间只有一条路线|
|真实浏览器上传、提交、下载|空间案例通过；其他类型未另做浏览器点击|
|全库逐项来源/契约/公共路径核查|80项及1个参数化旧插件已记录；只有14个ID计当前真实执行|
|任意任务、全部58项available的真实数值重验|未完成且未声称；条件/历史证据在矩阵逐项保留|
|真实分类文件端到端、全部地形/学习/土层/模型原生验证|本轮未新增，不能以边界单测替代|
|全部软目标、通用精度预测路由|未实现；未消费目标明示，不转成非法硬门|
|五种策略配置|均接公共服务；A/B本次极小预算未取得完整图，未造成功|
|科研优越性、正式独立确认、任务一正式实验|未运行，不作结论|

最终回答：**在现有明确契约和合法评价资源内，平台已可从统一入口自动组织空间、时间、多源的语义匹配、改进GOLEM、真实执行、评价反馈及产品输出，并有分别的端到端证据；并非所有库条目、数据类型和模型原生输入均已完成数值验收。**达到的是有边界的工程集成，不是算法精度创新证明。本轮到此结束，不自动启动正式研究实验。

## 第42章 数据问题驱动自动构图与GOLEM定向变异（2026-10-21）

### 42.1 范围、历史与实际改变

本轮是有界工程验收，不是算法优越性研究。保留第38—41章、旧C、原始数据和全部失败记录；未下载数据、未跑北京全量对照/949图穷举、未训练收益模型。新输出根目录：`output/problem_directed_20261021/`。修改前源码及SHA在`source_before/`、`before_manifest.json`；执行源码及指纹在`source_executed/`、`verification.json`。

复用而非重建：`SemanticFeatureExtractor/RequirementMapper`、JSON规则、Registry与compatibility、`SemanticDecision.require_graph`、`TypedGolemFactory`反向端口语法、官方GOLEM优化器、`Pipeline.run_dag`与真实DAG执行器、双层质量门、公共`submit_search`和原页面处理函数。新增明确策略名`C-directed`；A/B/C/C-gap/R旧策略保留，旧C仍是先产生随机提议再赋权，未被改称为本轮机制。公共页面与工具包默认进入新模式；历史时间研究客户端可继续显式使用旧C。

原缺口：公共选择记录尚不足以构成“问题→可执行图编辑”；旧C的三个随机提议未保证涉及缺失端口。现在新模式先通过问题与能力需求定位实际缺失端口，再从同一注册契约的生产者反向补齐依赖；之后才调用已有共享语义权重并实际抽样。未新建数学算法或独立优化器。

### 42.2 公共五步链路及边界

1. `semantic_requirements.py`：解析后记录真实输入有限值数、缺失数/比例、负值数；保留原时间支持特征。训练偏差诊断只接受显式train配对、同单位、同队列且与应用期无重叠；不满足返回UNKNOWN。没有可靠参考不能判断来源准确性。
2. `semantic_rules.json/RequirementMapper`：命中规则生成结构化问题，含稳定ID、变量/角色、时空范围、实际测量/事实、任务要求、来源、CONFIRMED/UNKNOWN/WORKING_ASSUMPTION、必需/可选、所需能力及目标输出端口。新增粗观测一致性必需/可选规则、合法训练偏差可选规则和可见时间缺失规则。能力仍由已有目录和契约匹配，不按文件名、任务A/B/C或预期赢家选算法。
3. `golem_generation.py`：从目标输出kind反向调用现有`recipes`，递归找到生产节点和其输入/状态。候选需通过实际父边绑定、深度/节点/循环/端口检查及公共能力验收。预筛时的能力集合只用于缩小提议，不能作为最终路径证明。
4. `numerical_search.py`：新模式产生connect（补缺失生产者与其依赖）、insert（包含原上游计算的处理节点）、replace（同输出类型合法替代）、simplify（去可选步骤/合法旁路）。重新构建可达图，保留实际共享父节点和状态；不把同类型但不同训练状态合并。候选按计算指纹去重。每次记录目标端口、问题ID、理由、增删算法、连接、父子图和缺口变化。并非每个任务都适用全部操作。
5. 官方GOLEM消费新操作；完整图仍先`require_graph`最终验收，再Pipeline真实计算、质量门、独立评价和NetCDF回读。UNKNOWN不能冒充PASS；未完成图可继续演化。新策略不能关闭共同安全条件。

配置上界：每次默认最多4个候选、6次候选构造尝试、每kind最多64个反向配方，沿用最大深度/节点限制；25%概率保留官方默认变异探索（允许区间0.1—0.9），不是100%定向。保存了归一化权重、实际抽样结果及官方优化日志。旧策略的默认参数不改。小候选空间的新模式种群上限取`min(定向候选上限,合法根配方数)`，防止官方初始化为了填满种群而耗尽提议预算。

必要公共适配：对明确像元平均/通量语义、完整足迹的二维规则网格，利用已有面积权重建立真实粗观测算子与原粗网格观测端口。约束节点在规划阶段检查显式端口/资源契约，运行时检查真实上游产品；不再用会在边缘产生临时NaN的规划插值探针错误排除约束，也没有填造预测。`raster_reconstruction_task.py`仅补后处理资源声明，不改QP/非负数学内核。

### 42.3 损失、反馈与未来学习记录

有独立验证参考时，L为固定验证队列实际RMSE；参考只留在独立评价器。无真值时本轮空间L为`RMS(A*x-y_coarse)`，时间L为负的实际有限覆盖率，均明确`accuracy_assessed=false`。AOD观测一致性不是未经论证的质量守恒。

操作收益记录为`r=L(parent)-L(child)`，仅父子不同、两者均真实新执行、质量PASS且同一评价协议/输入队列时可计算。缺口惩罚、不完整图、失败和缓存本身不成为收益标签。多步骤依赖补齐、默认变异或多个提议都可能产生同子图时，标为复合/无法单独归因；不能把整图差异说成单节点因果贡献。记录位置：各运行的`operation_transitions.json`，含状态特征、父子图、操作、评价指纹、收益、归因和可否作为单操作标签。没有训练Q或其他收益预测模型。

以工具包v4运行计：空间8条可计算整图差异、0条可单独归因标签；多源6条整图差异、1条当前可归因标签；时间0条。重复客户端不是新增独立科研样本。**已具备采集格式，不具备足量、独立且有覆盖的效果学习训练数据。**

### 42.4 提前固定的真实小案例与结果

客户端：`tools/accept_problem_directed.py`仅提供真实文件、任务、独立评价与预算，不维护策略。种子11，共用最多8次唯一候选评价、64次提议、150次回调、30秒墙钟上限，未根据误差换种子。v1—v3失败保留；v4为修复后最终证据，不能删掉失败假装一次通过。

|案例|真实输入及语义|新完整图真实执行|图缓存命中|最佳实际目标|搜索秒 / 定向生成秒|验收边界|
|---|---|---:|---:|---:|---|---|
|空间|既有LGHAP BTH 2019-01-01 16×16产品，以已有面积算子生成8×8输入，再查询原16×16足迹|5|8|粗观测一致性RMS 9.71445e-17|2.199 / 0.083|非负、端口、质量、产品回读通过；不是独立实测精度|
|时间|既有真实ERA5降水24小时单点，明确1h累计区间，聚合6h|1|3|负有限覆盖率 -1|3.006 / 0.248|仅1条合法聚合路径，无优化区分度|
|多源|复用既有台湾单小时46训练站、9验证点PM2.5与配套AOD，query不含目标标签|5|5|固定9点验证RMSE 7.747969 ug m-3|2.172 / 0.122|真实多分支fit→state→background/残差→sum；探索性工作坐标/时间约定保留|

输入、预算和评价协议在各`protocol.json`，真实图/节点轨迹/预测与NetCDF在`runs/*_bounded_v4/`。`summary.json`分别记录callbacks、incomplete_feedback、proposal_rejections、posterior_rejections、real_executions、quality_failures、graph_cache_hits等，未用缓存凑新执行数。三类最终运行质量失败均0，但存在有据可查的生成拒绝：多源21次构造拒绝、1次后验拒绝；时间11次默认提议拒绝、48次无适用定向候选，均受64次总提议限制，不解释为额外有效探索。

多源实际图包括真实PyKrige拟合、训练状态、背景、训练残差、残差插值及加和父边，不是算法名称占位。空间实际搜索选择过connect/replace/simplify及默认变异；insert的合法上游连接在定向行为测试验收，不能声称本次随机轨迹已经选择并数值执行了insert。双客户端三类各自实际执行，不能将两次相同结果当独立重复实验证据。

成本：每次公共响应另记解析、识别/匹配/绑定、决策、定向生成、执行/质量、独立评价和输出回读耗时，见`preparation_costs`及逐行成本；进程导入启动不算搜索耗时。匹配/特征缓存均按内容与规则/资源键管理，执行质量检查不被缓存跳过；本轮冷进程匹配均未命中，不声称已经证明缓存收益。

### 42.5 页面、工具包与浏览器证据

三类固定种子客户端实际调用同一`submit_search`：页面处理函数与工具包在去除耗时字段后，语义决策、候选域、逐次操作/权重/概率、选中图和数值产品一致，独立回读对比通过。证据：`verification.json.client_pairs`；校验程序`tools/verify_problem_directed.py`只读结果，不触发搜索。

真实浏览器从主页面上传`spatial_area_mean.nc`，声明单位1、像元平均、目标网格、非负和必需粗观测一致性；没有上传参考，明确按覆盖工程指标搜索。5评价/50提议/30秒预算，真实执行3图，质量失败0，选中多步骤约束图，NetCDF回读PASS；覆盖目标无区分度，不称精度最优。请求目录：`output/platform_search_requests/eafde9acb0f94ce69ee64ab5bac3ebd4/`。页面显示、真实ZIP下载均完成；下载包`C:/Users/LX/Downloads/platform_search (1).zip`已检查CRC及包含真实`product.nc`，SHA和成员列表保存在`verification.json.browser`。截图`output/problem_directed_20261021/browser_result.png`。时间/多源本轮只做页面处理函数客户端，未另做浏览器逐项点击。

### 42.6 故障、定向测试与未实现内容

首轮空间成功；时间/多源因候选空间小于请求种群、且定向根候选截断，填充种群重复提议提前耗尽预算，没有数值执行。保留原失败summary/日志；修复为新策略按根配方与每次候选上限设置种群，未改官方GOLEM。时间仍有大量无可编辑动作，已经分账，不伪造更多路线。另修复验收工具自身两个断言问题：不完整输出既可通过connect也可经合法输出包装替代补齐；Windows读取JSON显式UTF-8。保留初始测试失败XML。

新增与受影响定向测试共69个不同测试最终通过：初轮原有49项通过（`tests_final.xml`另有新增2项失败，未删证据），新增修正/扩展后5项通过（`tests_extra.xml`），公共数值服务15项通过（`tests_service.xml`）。覆盖实际端口补齐、insert/replace/simplify、语义改变候选、特征缓存按值失效、训练偏差隔离、小空间真实官方GOLEM执行、收益空值/归因条件；原第39章完整/未知/分支验收及时间单位测试继续通过。已有flox弃用及Windows pytest缓存权限警告保留，不当作算法错误，也不声称全系统无错误。没有跑无关全量回归。

限制：当前问题规则能驱动声明能力与端口依赖，不能预测哪种算法RMSE最低；平滑度/峰值等未建立证据的软偏好没有被包装为精度模型。全库条件能力仍按第41章矩阵保留，未全部重验。训练偏差规则的诊断与能力提议完成边界单测，本轮没有额外真实偏差校正数据实验。四类编辑不是四种任务都必然适用；当前仅少量可归因标签。旧C与新策略没有开展公平优越性比较，不能宣称新策略更准或更快。台湾坐标/时间仍为工作约定，结果不是新独立盲测。

复现：`.venv/Scripts/python.exe tools/accept_problem_directed.py spatial|temporal|multisource --client toolkit|page`（现有同名输出拒绝覆盖；重跑须显式新目录）；只读核验`.venv/Scripts/python.exe tools/verify_problem_directed.py`。定向测试见对应XML及`tests/test_problem_directed.py`。

本轮结论：**已将问题清单、共享能力规则、反向端口构图、定向操作、实际概率抽样和真实执行反馈接在公共平台内，三类均有真实有界证据，页面与工具包共用；完成的是机制工程验收，不是算法优势证明。尚未实现/启动操作收益学习，当前数据量不足以可靠训练；不自动追加大型实验。**

## 第43章 第42章局部绑定、动作归因及枚举顺序修复验收（2026-10-21）

### 43.1 结论与版本

本轮功能验收结论：**下表列明的验收条件全部PASS，无FAIL或外部阻塞；结论限于已声明契约和固定小案例，不代表精度优势或任意数据均支持。**新增显式策略`C-local`，公共入口默认为此策略；原`C-directed`保留第42章全局问题策略，A/B/C/C-gap/R保留原有搜索策略。没有修改第三方GOLEM、训练效果模型、下载数据或运行大规模对照/949图穷举。

公共契约版本为`platform_search_v4_local_problem_actions`。最终同源码验收目录：`output/local_problem_20261021/runs/*_v5/`，不是第42章目录。修改前快照在新目录的`source_before/`，执行源码与SHA在`source_executed/`和`verification.json.source_fingerprints`。v1—v4搜索目录、早期失败XML和第42章全部记录保留。只读核验记录可以重新生成，不将中间版本混称最终验收。

### 43.2 可直接审查的缺口—修复—证据表

|原缺口或验收项|修改位置|行为测试|真实案例证据|结果|剩余限制|
|---|---|---|---|---|---|
|问题统一挂总输出、来源不明|`router.py:bind_problem_ports`；`semantic_requirements.py`；`semantic_rules.json`|单来源网格变化、两分支不同问题ID、内容失效且B不变|`aligned/mismatch_*_v5/local_port_contracts.json`、`decision.json`|PASS|输出级非负/融合要求仍合理绑定共同输出；来源级问题绑定自己的对齐端口，不强行把所有输出规则局部化|
|全局wanted遍历、所有动作挂相同问题ID|`golem_generation.py:problem_candidates`|局部编辑只绑定所选问题、正确融合输入端口、无关子图保持、强制局部能力验收|`real_local_edit_candidates.json`、mismatch搜索提议及图|PASS|从未完成根构建全部依赖时属于复合构建，明确记录新增/共享来源；不伪装为仅修改一节点|
|配方先按登记顺序截断|`recipes/_fair_recipes`，仅新策略启用|反转生产者次序，未截断规范集合一致；上限1、种子0—9覆盖全部3个合法重网格生产者|`factory_trace.json`中的`local_recipe_enumeration`和截断记录|PASS|只有有界覆盖性质，不声称枚举所有参数、状态或大空间组合|
|替换后旧孤立节点被当第二输出|`problem_candidates`保留原输出根并重建可达闭包|完整图source0父边替换，source1完整程序保持|真实mismatch最终轨迹多次`fusion.uncertainty_weighted.source0`替换|PASS|只允许真实契约兼容连接；非法或UNKNOWN仍拒绝交付|
|插入/简化被边界包装及截断掩盖|保留实际科学上游程序的分组排序、组内受控轮转；科学节点计数分类|可选约束insert/simplify；简化用明确目标、空数据问题ID，不冒称修复缺陷|空间搜索实际操作与增删边记录|PASS|某次随机轨迹未选中的操作只按候选层验收，不补种子追成功|
|深拷贝后Bundle指纹不稳定、源叶未进入外部缓存键|`dag.py:Artifact.fingerprint`；`platform_search.py`显式`factory.SOURCES`|复用同执行器改变A，仅A失效、B缓存命中；改变中间输出影响下游|`real_intermediate_intervention.json`、执行节点与边指纹|PASS|估计器仍使用原有内容序列化；未承诺跨版本复用旧缓存|
|相同任务的适用性变化是否生效|已有compatibility，新增多源支持域一致性前置检查|单源同任务仅改像元mean→point，保守方法退出；混合point/area多源无观测转换契约时拒绝|真实网格变化使A出现3个适用的Registry重网格绑定|PASS|不同统计支持域没有被静默当同一变量平均|
|页面/工具包及当前浏览器|公共`submit_search`，未新建页面逻辑|同请求、固定种子，文件改名/上传逆序后决策、提议、图及产品一致|`verification.json.client_pairs/browser`、`browser_current.png`、实际下载包|PASS|浏览器本轮验收空间上传；多输入与台湾多源做页面处理函数成对验收，不冒称全部类型逐个浏览器点击|
|新增及受影响回归|`tests/test_local_problem.py`及原公共/状态/DAG测试|109项通过、0失败、0跳过|`tests_accepted.xml`|PASS|仅定向回归，不表述为整个平台无错误或全量研究复现|

### 43.3 哪个数据变化，导致哪个分支变化

固定真实LGHAP BTH 2019-01-01小窗口，目标始终为同一个8×8网格。对照中A、B都为该8×8表示；受控变化只把A换为同一产品的原16×16表示，B的数组、单位、支持域、角色及目标网格不变。两种分辨率都来自**同一真实产品**，不是两个独立观测来源；这里只检验平台工程行为。等权常量`uncertainty=1`是明确的工程先验，不是实测传感器误差。

- A稳定身份为`role:primary:aod`；B为`role:background:aod`。单角色可用角色+变量定位；同角色多输入必须明确声明不同`source_id`，缺失/冲突则拒绝，不能靠文件名或当前排列猜身份。数据内容指纹与身份分开，A内容变化不把它改名为另一个来源。
- A网格与目标不一致，产生`source_grid_mismatch`，局部必需能力为`continuous_spatial_resampling`，绑定A的`...aligned_role:primary:aod`端口，不再统一挂`task_output.field`。事实包含源网格语义/范围、输入有效统计、规则证据及源内容指纹；目标坐标和用途保存在同请求协议中。
- 原来直接进入融合的A输入，现在必须由A源叶→已有Registry重网格内核→A对齐端口→融合`source0`供给。真实可用绑定为nearest/bilinear/conservative；不是新算法。适配器只做角色、目标网格和科学端口绑定，调用原compatibility与run。
- B已在目标网格，继续由B源叶供给`source1`，其资源记录和已有子图规范签名不变。A重建后共同下游融合当然需要重算；这是正常依赖影响，不是B的数据、方法或子图被改写。
- 初始图只有未满足的输出节点时，局部问题通过源身份和待满足kind反向定位，补齐必要闭包；动作明确标记为复合/共享构建。已有完整图上，局部替换只改融合的A输入父边；不能从另一来源借能力。真实轨迹已观察到此种source0替换。

所有动作保存真正的单问题ID或明确简化目标、来源、目标端口、所需能力、父子指纹、增删节点/边、规范保护子图、源到输出路径及共享/复合标记。最终`assess_graph/require_graph`另验局部必需端口能力：即使全局融合能力满足，A局部能力确定缺失仍REJECT，未知仍UNKNOWN，不能交付。未完成图仍可演化。原条件分流/回退的最终能力验收与状态所有权检查继续复用，不改数学定义。

### 43.4 公平有界枚举与策略边界

先按局部kind和可提供能力剪枝，生产者按规范身份与固定种子的哈希顺序排列，再惰性轮流展开每个生产者；子依赖、参数乘积均有界，不先物化巨大笛卡尔空间。已有科学上游程序的保留程度用于局部编辑排序，不是预测精度偏好；只在同等保留程度内轮转，避免轮转把合法插入/简化挪出小预算。问题流和同级候选流也有受控轮转。候选仍按规范图签名去重，重复、拒绝计入提议；记录未展开流和`complete_space_claim=false`。

固定上限：每次4候选、6次构造尝试，反向每kind最多64配方，最大深度9、节点24；默认25%官方变异探索保留。`C-local`明确采用官方GOLEM串行种群1，使一次只暴露一个合法局部子图时不因填种群耗尽预算；这是**独立声明的工程策略配置**，本轮不拿它与其他组作公平效果优劣结论。旧C-directed仍走旧生成器；旧A/B/C/C-gap/R没有偷偷切为局部策略。源叶外部指纹与共同最终验收等安全修复对公共执行有效，所以新协议/缓存不能冒充旧历史协议的逐比特复现。

### 43.5 最终版本真实有界结果与连接证据

全部从未完成图进入官方GOLEM，未手工预置最终图。固定种子11、最多5次唯一候选评价、48次提议、100次回调、20秒搜索墙钟。没有按误差换样本、种子或预算。候选与数值评价分离；空间和累计时间仅优化明确覆盖指标，不能称精度最优。

|最终v5案例（工具包）|真实新图执行|图缓存命中|真实目标|搜索秒|结论|
|---|---:|---:|---:|---:|---|
|双输入已对齐|1|3|负有限覆盖率 -1|2.261|合法直接融合；无需强加重网格|
|仅A网格不匹配|3|2|负有限覆盖率 -1|9.286|A重网格→融合→质量门及回读通过；B保持|
|单源真实空间|4|0|负有限覆盖率 -1|2.984|公共构图、真实执行、约束和回读通过，不作精度声明|
|真实ERA5累计时间聚合|1|3|负有限覆盖率 -1|2.955|明确累计区间按6h聚合，仅一条合法路径，无选择区分度|
|真实台湾PM2.5+AOD|4|0|9点验证RMSE 7.793694 ug m-3|6.717|真实多分支拟合/状态/背景/残差与质量门通过；原坐标/时间工作约定保留|

每种任务的实际失败、未完成、缓存、回调、生成耗时及解析/匹配等成本均在各`summary.json`、`factory_trace.json`和逐图`result.json`，没有把缓存或不完整惩罚当真实新执行。当前小样本结果不是算法优势研究；台湾本轮最佳值也不能拿来宣称优于旧版本或另组预算。

真实连接干预使用**GOLEM已选出的mismatch图**，不是手工最终路线。先执行原图，再仅在重网格中间产物输出端加0.2，原始输入文件不动；已有等权融合的最终AOD输出增加0.1（实测最大0.10000000000000009，数值容差1e-10内），B源节点缓存命中。两个结果均经过公共Pipeline、最终能力验收、质量门和NetCDF回读。它证明下游消费了该实际中间产物，不能绕回原始A输入；这是明确的功能干预，不是环境效果数据、算法参数调优或新数学内核。证据：`real_intermediate_intervention.json`及其baseline/altered节点边轨迹。`real_local_edit_candidates.json`另记录同一真实完整图上A局部编辑和B保护签名。

同请求成对通过：aligned、mismatch、台湾多源三组，页面处理函数故意改名并逆序上传；去除计时后语义决策、候选域、逐次操作/概率、选图、数值产品相同，见`verification.json.client_pairs`。单源空间、时间另有公共入口真实运行，不把只有一条路线说成选择优化。

当前最终版浏览器（不是第42章旧截图）：本地8506主页面上传本轮真实`mismatch_primary.nc`、声明单位/网格/非负、选择无真值覆盖指标、提交搜索、展示成功并下载ZIP。请求目录`output/platform_search_requests/ad239d60a59640109ad3207eb2d8e72d/`；5评价/50提议/30秒上限，实际新执行4图、图缓存1次、质量失败0、30次提议（含20次重复）。`C-local`模式、产品回读PASS、覆盖目标无区分度均已核对。下载`C:/Users/LX/Downloads/platform_search (2).zip`经过CRC及包内真实NetCDF独立读取，SHA/清单证据在`verification.json.browser`，截图`output/local_problem_20261021/browser_current.png`。

### 43.6 测试、失败保护及适用边界

最终`tests_accepted.xml`：**109通过、0失败、0跳过**。其中新增14项局部问题行为测试，覆盖：A单来源网格变化；同一单源科学任务仅改mean/point支持导致保守法退出；混合支持域多源无转换契约时拒绝；上下游真实数值干预；无关描述/输入排序不影响；登记顺序和截断可达性；两分支不同问题的精确归因；输入内容变更及B复用；受控预算；旧策略隔离；insert/simplify；局部必需能力缺失与UNKNOWN不可交付。原条件路径、状态、小时边界及公共入口定向回归保留。只有flox已有弃用警告，未屏蔽失败或删除测试。

真实发现并修复的故障都保留在新目录早期XML和版本化搜索中：边替换导致旧孤立节点；Bundle深拷贝不稳定指纹；普通源叶缺外部数据缓存键；小空间填种群导致零执行；轮转挪走保留上游的合法插入/简化；早期测试工具漏传执行键。不是用提高随机尝试次数解决，也没有删掉失败记录。源码变化后重跑受影响的有界案例，最终证据统一为v5。

当前支持限制：本轮新增的来源级自动对齐针对明确单位、同一物理变量和统计支持、规则二维经纬网格及明确目标网格。只转接已有重网格内核；非恒定辅助/不确定性字段没有传播模型则明确拒绝，不静默丢弃或插值后冒充正确误差。表格站点/AOD使用原有真实适配器。未实施任意异构支持域观测算子、所有可能数据问题的精度预测、操作收益学习、普适最优组合或大规模公平研究。条件分流的样本有效性仍以原公共能力验收为准，不能把结构路径展示当作所有场景的实际科学贡献证明。

所有最终v5输入SHA重新核对；中间干预前后直接使用的小型输入文件SHA一致。原始用户数据未写入/删除，旧实验目录未覆盖。记录这些被核对的文件，不把有限哈希检查夸大为对整盘所有文件的审计。

复现入口：`.venv/Scripts/python.exe tools/accept_local_problem.py mismatch --client toolkit --version 新目录版本`，也可选aligned/spatial/temporal/multisource；`--client page`调用同一页面处理函数，已有目录拒绝覆盖。核验当前结果运行`.venv/Scripts/python.exe tools/verify_local_problem.py`，不触发搜索，只做读回、有限中间产物干预和证据核对。完整测试命令及执行环境对应XML和源码归档。

最终逐项结论：来源/端口绑定PASS；局部问题驱动动作PASS；问题ID与修改范围PASS；真实依赖和中间产物消费PASS；单条件变化PASS；无关变化及无关分支保持PASS；登记顺序/有界截断PASS；公共入口与当前浏览器PASS；新增及受影响定向回归PASS。**本轮功能验收完成；算法优势、效果预测与收益学习未实施、未证明。**

## 第44章 两类能力匹配与两层机制验证（2026-10-21）

### 44.1 范围与逐项进度

本轮完成公共必需/可选能力接口、受限随机森林收益评分、声明/识别两层的有界实际执行。**没有完成可信的跨区域收益评分验证，也没有证明评分改善最终选图。**不覆盖第38—43章结论；不新增科学算法、不下载数据、不启动正式A/B/C或大型穷举。证据统一位于 `output/capability_matching_20261021/`。

|现有内容|核实状态|本轮复用或最小修改|行为证据／边界|
|---|---|---|---|
|变量类别、单位、时空统计语义|已实现且有行为证据|复用提取器、规则及compatibility|连续/分类、瞬时/平均/累计、冲突拒绝；科学含义仍需正常元数据|
|网格关系、来源身份与端口绑定|已实现且有行为证据|复用C-local，不另写匹配器|仅A网格变化产生A局部问题；B资源及子图保持|
|缺失、单/双侧支持与时间轴|已实现且有行为证据|复用extract_series、小时控制契约|声明/识别得到相同需求；30分钟和不规则输入不能进入小时分流|
|12项栅格形态/尺度特征|已实现且有行为证据|把原raster_upload公式移到公共SemanticFeatureExtractor，旧函数转为调用它|旧路由回归通过；没有输入隐藏场类别、文件名或参考误差|
|必需能力最终路径验收|已实现且有行为证据|复用require_graph；支持显式required_capabilities|完整缺失拒绝、UNKNOWN不交付、中间图可演化；可选收益不能豁免硬要求|
|可选一致性、合法训练偏差规则|已实现且有行为证据|复用既有规则；增加公共可选证据账本|诊断有依据不等于应用收益已知；没有参考时不判来源准确性|
|实施上下文相关的收益估计|原来确实缺失，本轮已实现并实际运行|新增capability_scoring公共服务，RF轻量训练/版本装载|仅本章AOD一致性开发范围；未推断所有能力具有固定收益|
|可信的跨区域评分泛化|当前证据不足，不能可靠推断|严格保留弃权范围，不放宽门槛|四川12条留出全部分布外，部署评分覆盖0/12|
|其他能力收益模型|确实缺失|保持None及证据不足|未造空模型、默认高分或0收益标签|
|旧KNN栅格路由|已有实现，本轮未重验其全部模型结论|保留，不改称能力评分模型|仅共用原形态特征和旧接口回归；未重训|
|曲线网格、神经模型、模型专用等全库状态|已有实现但本轮未逐项再验|沿用第41章条件矩阵|本章不以有限案例宣称全库重验通过|

机器可读逐项表：`feature_capability_audit.json`。前后源码在`source_before/`、`source_executed/`，最终版本与SHA见`release.json`。核心文件修改前已备份；raster_upload的早期函数依据初读记录还原，晚捕获版本另存，不冒称该份为逐字节原始快照。历史数据、旧报告章节、失败与协议均保留。

### 44.2 公共接口、两个消费者及信息边界

`search_page.process_search_upload / submit_file_request / 研究客户端` → 同一`platform_search.submit_search` → 解析/质量 → 公共特征/声明核对 → `RequirementMapper/AlgorithmSelector.decide` → C-local局部能力及端口 → 官方GOLEM → `Pipeline.run_dag` → 最终能力门/真实运算/质量门 → 独立评价 → 反馈及产品。

1. **必需类：** 物理语义、实际网格/时间关系及明确要求推导；保留规则、源身份、端口和路径。`requirements.required_capabilities`只接受目录中真实能力ID；显式必需不能因收益低而省略。最终验收不受研究策略开关绕过。未完成仍可演化，不能用全图能力并集借用其他分支。
2. **可选类：** `optional_capability_matches`返回能力、问题、候选/资源条件及证据状态；无估计为`None`，不是0。模型结果逐实施上下文记录于`implementation_scores/optional_scoring`，不赋一个脱离算法和上游的“能力固定收益”。显式可选请求可用`requirements.optional_capabilities`，但不会因此升级硬要求。
3. **声明接口：** `feature_declarations`每项含`source_id, feature, value, unit, status, confirmed, evidence`。允许变量语义、网格关系、缺失及支持条件等白名单事实，不允许算法或完整图答案。可由文件核实的事实冲突直接拒绝；未知不被声明偷偷覆盖为已证实。元数据补充继续通过既有Reader声明协议。确认后的值实际送入同一个Mapper，既有源角色/端口仍负责局部事实。
4. **评分模型公共管理：** `CapabilityScoringService.train`将合法配对、分组与协议固化到`output/capability_models/<model_id>/`；只接受平台模型ID，装载核对模型SHA，不让客户端上传任意pickle或另注入规则。页面现有需求JSON可提供`optional_scoring={model_id,enabled,acknowledge_limited_development}`；受限开发模型必须显式确认，默认不启用。
5. **实际概率消费：** 仅显式启用的C-local使用模型；其他旧策略保持。相关可选节点候选以 `0.9*normalize(w*exp(clip(r_pred,-1,1)))+0.1/n` 保留探索；不含该可选节点的配对基准乘数为1，这不是人为0收益标签。若未知、分布外、实现不符或能力已成为必需，回退原权重。固定参数已写开发协议，不看留出结果追调。最终fitness仍是独立实际RMSE/工程指标，不是预测收益。

字段/任务/单位/损失族、具体前驱算法、可选实现、支持域和尺度均限定评分上下文。当前服务只评分单源二维AOD开发配方；其他时间/偏差/多源能力不会冒用这个模型。小时恢复适配器也消费相同的声明核对服务，缺合法历史状态继续排除历史算法。

### 44.3 历史证据审查与冻结的有界补采

只读审查第42/43章52份操作记录，共550条转换，以及旧LGHAP验证CSV和三类合成场指标CSV，逐文件数量及排除理由在`historical_evidence_audit.json`。这不是全项目所有历史实验的穷尽审查。

旧转换有完整图差异/单操作标志，但未满足本轮每侧参数、信息条件和明确单阶段科学步骤签名的全部证据门；不能把原`L(parent)-L(child)`直接当归一化标签。CSV名称相似也不能证明相同目标队列和参数；合成场多目标及重复策略记录不能混成真实AOD精度收益。因此本次训练不直接接收这些行，保留原证据，未改写旧结论。

在产生新损失前冻结`protocol.json`：只用已有`tiles/train/`的BTH、Henan、Sichuan三个区域、2019-01-01/04-15六个参考场，各取固定左上24×24窗口；面积聚合到12×12及8×8。既有nearest/bilinear/cubic适配的上游不变，只替换一个尾阶段：非负处理 → 既有非负+粗观测一致性QP。QP包含非负，新增的是一致性能力，不改内核。参考是LGHAP融合产品，不是独立仪器真值。

72次Pipeline实际调用得到36条合格配对、0失败；每个产品写出后回读。保存输入、参考、队列、参数、信息条件、单位、步骤、源SHA和质量结果。`paired_evidence`拒绝失败/UNKNOWN/惩罚、不可比信息、复合变化及近零基准；相同内容即使改客户端/分组名仍去重。标签为 `r=(RMSE_before-RMSE_after)/max(RMSE_before,1e-8)`，epsilon单位为无量纲AOD单位1；近零单列，不造收益。工程覆盖负损失不能进入该误差模型。

本次全部补采都是明确的开发资料，不再保留独立测试身份；没有读取未来封存参考。重复日期、倍率、算法实现及两客户端不算独立样本。

### 44.4 RF真实模型、留出诊断与不能通过的条件

模型`aod_consistency_dev_v1`使用已有scikit-learn：64树、深度4、叶最少2、单CPU、种子17。输入是实际粗网格12项特征+任务/损失族+前驱与尾实现上下文，不接收目标数组、隐藏类别或文件名。BTH/Henan为24条训练；Sichuan为12条留出（**仅1个区域、2个参考场，绝非12个独立环境重复**）。没有随机打散同源像元或重复遮挡。

|原始RF留出诊断（相对收益单位）|结果|
|---|---:|
|RF预测MAE|0.0230494|
|无数据特征的常量基线MAE|0.0505936|
|无数据特征的实施全局排名MAE|0.0256716|
|RF能力实施排序的平均选择遗憾|0|
|全局实施排名的平均选择遗憾|0|
|常量按固定首项选择的平均遗憾|0.1314463|
|公共评分器在留出上的适用覆盖|**0/12**|

**这些MAE只是无范围门的原始RF诊断，不能作为公共部署评分通过证据。**预先固定的12维训练范围门在四川全部弃权，因此适用范围内留出MAE为未评价/None，不能填0、不能称RF可信。模型已有真实训练、原始留出计算和排序记录，但**C项可信评分验证证据不足**。不修改范围门来获得通过，不用树间分歧充当置信区间；只有一个留出区域，未给虚假的独立重复显著性或可靠95%区间。

模型及训练来源见`output/capability_models/aod_consistency_dev_v1/{report.json,evidence.json,model.joblib}`；部署弃权复核、加入/不加入诊断和结论见本轮`model_diagnostics.json`。其他能力缺少独立的实施配对，保持证据不足，不自动扩采。

### 44.5 两层实际矩阵与局部行为

两层相同实际数据、种子11、候选池及质量门：第一层显式确认可核实特征后送公共Mapper；第二层去掉特征答案，由提取器产生事实。科学单位和区间仍保留正常元数据，系统不凭数值猜单位。

|固定案例|两层实际执行/一致性|行为结论与证据边界|
|---|---|---|
|真实LGHAP A/B均对齐|各1个新完整图，产品一致|无需强加重网格；覆盖工程指标-1，不是精度|
|仅真实来源A网格改变|各3个新完整图，产品一致|A局部`continuous_spatial_resampling`及对齐父边出现，B资源完全一致；多分支实际执行|
|真实ERA5累计降水聚合|各1个新图，产品一致|区间累计按6h处理；保留原起止证据，不冒充瞬时|
|真实ERA5瞬时气温处理|各1个新图，产品一致|与累计要求不同，但复用同一既有时序插件；不凭相同图ID断言数学行为相同|
|明确合成分类栅格|各1个新图，产品一致|类别保持，双线性不适用；仅功能边界，不是环境精度|
|明确合成区间平均序列|修复后各1个新图，产品一致|按时间权重平均，不求累计和；仅功能边界|
|真实开发AOD+评分，声明/识别|各4次实际尝试，其中3通过、1边缘覆盖失败；选图一致|评分有范围内开发输入，可检验实际概率消费，不是独立模型泛化评价|
|同请求页面处理函数/工具包|选图、非计时决策、概率和产品一致|同一公共服务；ZIP真实产品CRC/回读通过，本轮未作浏览器点击验收|

独立比对在`verification.json`，包括8组两层/客户端核对、真实父边图和产品。评分开启/关闭共享同一候选域，最大提议概率差为0.0236971；实际`random.choices`消费权重并保存chosen_index，不只增加日志。固定种子恰好选择相同最终图是有效负结果，不追加好种子。

最终小案例：双线性输出RMSE 0.00598664，双线性→QP为0.00525743；两者均真实计算。开/关评分最终都为0.00525743，**不能证明评分增加选型收益**。这个参考场也属于评分开发资料；其数值只作处理链验证，绝不是未见测试上的提升证据。

单/双侧缺失、合法训练偏差/证据不足、UNKNOWN、缺训练/权重/评价资源、声明冲突、无关名/排列、状态所有权、上游真实产物传递及预算缓存另有定向行为测试。偏差案例只验证训练诊断及候选证据状态，未声称完成真实偏差收益模型。原始文件未修改，隐藏参考只在评价器，评分使用特征白名单，不接收参考值。

### 44.6 发现的故障、修复及测试

实际平均值案例首次失败：共享规则使用`interval_mean`，旧内核仅认`mean`。在已有`plugins.py:TemporalSemanticPlugin`边界将同义语义规范化，执行后保留用户原语义；compatibility提前调用原校验。没有重写时间重采样或猜窗口。新测试验证3小时平均为[1,4]、源数据不变、UNKNOWN拒绝；初次失败目录保留，最终为`layers/interval_mean_*_v2/`。

补跑受影响时间/转换测试时，3个旧UI测试找不到按钮：主页面已默认统一搜索，测试却假定默认数据转换。只给这3个测试补显式导航，不改生产UI、不屏蔽断言；初次失败XML保留。平均值和相关测试76项复测全部通过。

本轮分批共**164个不同定向用例通过**（含原有受影响回归，不是164项全新测试）：`tests_affected.xml`113通过；`tests_final.xml`59通过；`tests_mean_fixed.xml`76通过；`tests_score_final.xml`26通过。批次有重叠，不能相加当独立数量。最后26项针对最终评分接口复核。保存旧`tests_mean.xml`的3个失败，不以最后一份覆盖失败证据。已有pytest缓存权限和依赖弃用警告保留，未运行无关全量回归。

### 44.7 成本、路径与复现

新配对采集3.617秒，模型训练0.069秒；历史证据脚本审查约0.099秒。最终工具包评分案例搜索约0.662秒，其中评分推理约0.019秒，总公共服务约0.764秒；仅代表此机器小输入，不是效率优势。`costs.json`逐运行分开列准备、匹配/绑定、生成、评分、真实计算/质量、独立评价及输出；还包括v1—v3重复集成验收及失败的额外成本（全部验收搜索合计约21.14秒，服务约24.35秒，不含测试及进程启动）。没有用少执行几图代替总成本优势。

核心修改：`semantic_requirements.py`、`router.py`、`capability_scoring.py`、`numerical_search.py`、`platform_search.py`、`temporal_search.py`、`plugins.py`；页面与旧栅格特征函数只作公共调用及说明。无tools导入公共服务、无新的局部赢家映射。

开发配对及产品：`pairs.json`、`paired_products/`；协议`protocol.json`、`layer_protocol*.json`；最终评分搜索`runs/*_v4/`；两层案例`layers/`；归档模型`output/capability_models/aod_consistency_dev_v1/`；源码及环境SHA、状态和测试去重索引见`release.json`。已存在模型/运行目录拒绝覆盖；`accept_capability_matching.py --version 新版本`核对训练证据指纹后复用模型，不重复训练。只读核验：`.venv/Scripts/python.exe tools/verify_capability_matching.py`。实验客户端仅供给固定数据/协议/独立评价器，没有匹配策略。

证据与分组统计规范参考了本地academic-paper-writing/statistical-analysis技能；软件工作流文献为Kassis、Agarwal、He、Patel、Brueckner (2026), [Scientific Agent Skills](https://arxiv.org/abs/2609.00065)（已核对官方条目）。该引用不构成本平台实验效果证据。

### 44.8 最终独立验收

|项目|状态|准确含义|
|---|---|---|
|A 必需能力匹配|PASS|源/端口/有效路径及最终要求真实验收，不因可选收益跳过|
|B 可选能力评分接口|PASS|真实证据门、None/不适用、版本及上下文明确；非强制|
|C 真实证据评分模型与可信留出验证|**证据不足**|真实RF和原始留出已执行，但部署门内留出0/12、仅1留出区域，可信模型未完成|
|D 声明特征驱动组合|PASS（上述有界支持案例）|确认值实际进入共享Mapper；冲突/未知/不支持均显式处理，不输入赢家|
|E 实际识别驱动组合|PASS（上述有界支持案例）|两层需求、候选、局部编辑及产品核对通过；合成分类/平均只算功能测试|
|F 公共入口端到端|PASS（处理函数/工具包/真实GOLEM/产品下载包）|没有冒称本轮做了浏览器人工验收或所有算法逐一重验|
|G 信息隔离与成本记录|PASS（列明范围）|参考分离、组隔离、失败/缓存/版本与成本保留；未称总成本更优|

**本轮工程机制已完成列明的有界验收；可信效果评分及自动选择优越性仍未证明。**不能用A/B/F通过替代C；不把开发场上的QP改善归因于评分策略。由于现有独立开发覆盖有限，本轮停止，不自动扩算法库、补大型采集或启动正式优越性研究。


## 第45章 三张画像表与模型需求驱动的短组合选择（2026-10-22）

本轮在公共模块新增显式Profile模式，复用模型需求、算法契约、端口配方、真实执行器与质量门；不调用进化优化器，历史GOLEM模式保留。没有安装新依赖、下载数据或运行正式遮挡/优越性研究。详细字段、评分、API、来源及限制见docs/profile_matching.md。

### 45.1 三表与公共入口

数据画像复用DataSemantics、实际缺失检查和已有12项栅格特征；当前梯度仅为像元索引描述，不是物理距离梯度。含缺失/非二维场不填造形态特征。实际科学节点输出重新提取阶段语义，写入端口metadata.stage_portrait和轨迹，真实父产物仍交给下游compatibility。算法画像来自当前AlgorithmSpec，不给无证据副作用和经验成本填分。

模型画像复用RequirementStore中CALPUFF、WASP、传统SWAT2012、WRF、CAMx、CMAQ、WRF-Chem，EPICC排除。单位、物理变量、类型冲突拒绝；只将明确1h/1D时间契约自动传为目标频率，自由文本不猜。用户选择已记录角色和字段，未审字段拒绝。敏感性为None，剩余角色列出，full_model_ready=false；不把标准化准备模板冒充全模型原生契约。

页面和工具包共用submit_search，mode=Profile。默认最多64个有界配方、三个科学节点，明确保存截断，不宣称完整穷举或所有算法任意串联。复用OptGraph仅作为既有图容器，不使用变异/交叉。初始按声明成本安排尝试，不称精度最优。有合法参考/工程指标时实际执行，按明确权重、尺度、阈值和近似持平容差比较实际损失；缺必要指标报错，工程覆盖率不能冒用RMSE。无评价资源只推荐，执行0次，不隐式捏造评分。

### 45.2 排序实现与证据边界

PlanRankingService复用项目现有官方scikit-learn DictVectorizer和浅层RandomForestRegressor，以任务组内平均并列排名学习候选顺序。输入为可见特征、模型字段/目标及有序组合/结构，不含参考、实际损失或文件名。只接受显式development、来源分组分离、完整跑完该有界池且产品SHA匹配的记录；同内容不能跨组，重复客户端不增样本。至少两个训练组和一个留出组只是工程门，不是样本充分性保证。模型SHA/运行库版本检查，未见上下文/有序组合或超范围返回None；整组预测不完整则回退规则顺序。旧AOD单阶段收益服务/模型未修改，不能用整图收益冒充单节点归因。目前没有发布可信通用排序模型，训练测试仅为功能夹具。

借鉴来源：scikit-learn官方GitHub及BSD-3-Clause实现；Ranking Trick论文的排名目标；AutoFolio只参考数据组织，未复制旧框架。没有将SwiftTS、AMLA或新的复杂框架硬接入。

### 45.3 实际验收

证据目录output/profile_matching_20261022/。初轮15项功能测试通过；受影响tests_accepted.xml为121项通过；最后tests_profile_release.xml为18项通过，去重122项，不相加计数。包括七模型字段契约实际数值执行/回读、输入冲突、目标权重消费、排序实际fit/load/predict、模型/产品指纹、超范围弃权、重复来源隔离、无参考零执行及Streamlit交互。合成夹具不算环境精度证据。

复用已有ERA5派生粗气温，在WRF-Chem气温字段契约下，最终real_toolkit_release和real_page_release各实际执行14候选，选图、模型配置和数值产品一致，实际写出后回读；原始来源指纹未变。指标为coverage，accuracy_assessed=false，不能称精度提升或WRF-Chem实读。real_verification.json记录来源，release.json和source_executed记录最终源码。页面验收是处理函数与Streamlit AppTest，不冒称外部浏览器点击。

### 45.4 未完成范围

专用hourly_recovery及站点/协变量状态端口保留原入口，Profile明确拒绝，未假接。七模型完整原生输入/程序验收、模型敏感性学习、所有含缺失场的形态画像、任意算法串联、可信泛化及搜索/组合优势均未完成。依赖弃用和pytest缓存权限警告保留。当前完成的是有界字段准备机制及排序工程通路，不以测试数量替代科学结论；本轮到此，不自动追加大型实验。

## 第46章 三表规则实际消费、组合验收与逐指标学习（2026-10-22）

本章修正第45章的范围：第45章主要打通字段契约、真实执行和排序服务，不等于三表风险匹配已经完成。本轮在平台Profile公共入口加入实际规则消费者，不改历史GOLEM、科学数值内核和旧AOD收益模型。

### 46.1 实际评分逻辑

新增portrait_rules.py，直接读取数据表matching_facts、算法表matching_properties和模型表matching_requirements。模型配置仍为七个已审核的准备契约；新增用户明确声明的priorities/hard/evidence，不从模型名称猜敏感性。

当前有依据的维度为range_preservation、continuity、no_mixing（复制输入原值）、nonnegative、coarse_consistency（明确观测算子一致性，不是物理质量守恒）。实现类白名单审核最近邻/双线性及既有包装、非负投影和粗观测QP，其他算法性质保留未知。二维有限规则网格内部凸组合/连续性性质是数学依据；像素相邻差/值域和负值比例从真实可见输入计算，不当作物理梯度、噪声强度或实测精度。

权重归一化；每维贡献=权重×数据激活强度×(+1有保证，-1无保证)。未知保留正负区间，点分None；零激活标INACTIVE。按下界保守排序，再按声明成本和节点数处理同分，区间不是统计置信区间。每项保留模型声明、数据字段、算法实现/数学依据和实际父连接。硬需求不能用分数抵消，未知保证也在执行前拒绝；产品回读后再次检查值域/非负，QP自身检查残差。无合格候选明确NO_FEASIBLE_CANDIDATES。

组合沿实际父连接传播性质，不相加节点分数，不默认正协同。运行后用同次合格且已执行的单算法最小损失减去组合损失，记录正/零/负收益；没有可比单算法时为None。最终仍用实际task_loss；在预先声明的持平容差内才用适配分、成本和节点数破同分。无评价资源只推荐、执行0次。

### 46.2 学习和三表填充

平台公共API train_profile_matching复用PlanRankingService和既有scikit-learn。完整分组开发记录学习objective/RMSE/MAE/abs_bias中合法可用的各项指标，再按当前需求权重和尺度重新计算预计损失。删除将偏好哈希当类别的旧路径；参考、实际评价标签和文件名不进入输入特征。算法能力、声明成本、传递性质和数据描述进入学习特征。旧相对排名预测只保留诊断兼容，不再作为新模式排序依据。

产品SHA、逐候选result.json指标和按记录偏好重算的task_loss交叉核对；至少两训练组与一留出组，来源内容不能跨组，重复客户端不增样本。训练产生不可覆盖的模型、report和portrait_evidence，加载核验模型/证据指纹、运行库与画像版本；未见上下文/有序图或数值超范围弃权，整组预测不完整则退回三表规则。按指标报告留出支持数和误差，不将功能夹具当环境效益。

只用训练组填充数据画像的处理案例、单算法输出值域/有效率/负值比例和带缓存口径的端到端成本、组合实测收益及模型字段准备案例；留出组不回填，整图收益不归因到单节点。加载指定版本模型后这些观测进入当前三表记录和学习排序；不自动覆盖数学保证、用户需求和未知模拟器敏感性。旧v1画像模型需重训，原模型/历史不覆盖。

### 46.3 端到端证据

证据目录output/portrait_rules_20261022/。最终公共入口、画像/学习、平台集成、P1、P2时空融合及旧能力评分定向回归105项通过（release_tests.xml）；学习负例改用复制的破坏样本，另有learning_release.xml对应验收，去重仍为105项，不累加。

- 同一合成输入，只改模型需求continuity/no_mixing，首个真实执行算法分别为双线性/最近邻，写出产品不同。
- 同一需求nonnegative，输入由非负变为含负值，选图从单节点变为重建→非负投影，实际连接、产品非负和分数激活均被检查。
- 硬连续/值域需求过滤不满足或未知的候选；撤回算法表中的双线性性质证据会改变公共入口过滤结果。域外不能借用域内保证；互相冲突的硬需求不交付。粗观测需求实际生成QP组合并执行，域外裸插值失败记录保留，边界包装后合格路径通过。
- 无关展示元数据变化不改变决策；未知点分为None；同性质组合不重复奖励；单算法更好或持平的组合结果保留；无参考执行为0。
- 真实fit/load/predict和三表填充、留出隔离、偏好重加权、OOD弃权、标签与模型指纹负例通过。仅功能验证，无可信通用生产排序模型发布。

tools/verify_portrait_rules.py复用既有ERA5粗气温文件，real_release目录记录不可覆盖请求、三表、候选评分、实际图和产品。连续需求选raster_task.spatial.bilinear，复制原值需求选raster_task.spatial.nearest；同一气温场两个产品最大差异约2.618575 K。页面处理函数与工具包结果一致，原始输入SHA未变，数值产品指纹及读取检查通过。采用coverage工程指标，不能由该差异断言谁更准确；页面处理函数验收不冒称外部浏览器上传点击。

### 46.4 仍有边界

规则不是所有算法、所有特征的完整性能知识库；噪声/动力/峰值保持和未经评审的算法性质仍未知。可以追加有来源的训练观测并版本化学习，但不会自动生成未经验证的科学规则。中间画像写入真实stage_portrait/stage_matching_facts；当前图预先有界生成，不宣称执行中动态重规划。任意算法串联、专用站点/小时状态端口、七模型原生运行验收、跨区域可信泛化和组合科学优势仍不在本次证明范围。

## 第47章 插值画像、科学池冻结与无参考学习推荐接线验收

本轮只解决正式空间实验前的公共接线问题，未运行计划中的ERA5小规模效果实验，未下载数据。小型合成端到端训练/执行属于功能验收，不是新科学效果证据。输出位于output/profile_reference_free_20261022/，最终检查数量和源码指纹以release.json为准；中间失败记录保留。

### 47.1 画像实际参与评分

portrait_rules升级为v2_interpolators：PCHIP复用已有ShapePreservingRaster/SciPy，在有限、递增规则轴的二维内部传播连续性、值域及源场非负时的非负性质，不宣称各向同性二维C1或细峰恢复。三次插值复用XarrayAggregationPlugin→xarray-regrid→xarray.interp(cubic)，至少四个样本/轴且位于有限二维内部时给连续性质；值域不提供保证、非负未知。噪声、峰值和数值精度的未知项不填造。

端到端检查两者的性质进入实际候选分数，并执行写出/回读；硬值域需求保留PCHIP而排除无保证的三次插值。外域/非法轴/不足采样不沿用内部证明。画像传播读取_port_bindings的命名field/source0，而不是父节点列表的位置，避免带QP算子/粗观测资源时读错上游。

### 47.2 公共入口冻结和去重

Profile的profile_options.candidate_pool.algorithms在生成前筛选实际任务适配器，复用TypedGolemFactory语法，不手写八张实验流程。指定四种插值包装及coarse_consistency、max_steps=2，合法资源下生成四单方法及四QP组合；硬粗观测一致性再筛为四图。缺必要适配器拒绝，达到生成cap也拒绝，不静默截断冻结池。

原始nearest/bilinear/xarray_cubic与对应raster包装规范为一个科学内核和明确边界规则，requested/effective/aliases、实现/规范版本、池指纹、语法/合法图数量写入candidate_pool.json。域外原生算法和包装不本来等价，这里是明确采用包装边界规则；PCHIP保留坐标裁剪规则。测试证明别名重复/请求顺序及随机种子不改变八图集合，不通过数值碰巧相同来去重。

### 47.3 评价类型与预测解耦

画像schema升级portraits_short_plans_v3_reference_free。public_vector不再输入evaluation_kind、metric、参考指纹或参考数组；改用实际源/目标网格、目标频率、语义及冻结池上下文，未放宽未见上下文/数值范围的弃权门。旧schema模型保留并明确要求重训，不能拿旧权重假装新接线已工作。

RMSE、MAE、abs_bias等具体指标预测可在reference_error训练后用于无参考recommendation_only。objective头单独保留标签语义，reference_error/RMSE不能与declared_task_metric/coverage混训，也不能在无参考时充当同一个objective。目标头不支持或池/范围未见时明确回退；保留支持数、所需指标和回退原因。覆盖率不能满足实测RMSE偏好。

无参考可显式设置preferences.weights.RMSE，返回排序、预计指标和分数，但best=None、real_executions=0、actual_accuracy=NOT_ASSESSED，精度阈值未核实。页面处理函数和工具包均走同一submit_search，不在客户端补规则或偷塞参考。

### 47.4 验收与边界

tests/test_profile_reference_free.py使用非恒定合成粗场和对应分块参考，三组真实数值开发记录经完整图执行、质量检查及回读后训练小型功能模型。随后页面/文件客户端不提供参考，并将RegistryEvaluator.execute和DatasetReference.__call__设为禁止调用；仍对八个候选得到有差异的RMSE预测和相同学习排序，执行数为0。修改只用于后验评价的参考值不改变预执行排序，参考评价发生在执行后；该检查不把改变后的参考误差当预测能力证据。

另验收硬需求、非法/不可用ID、截断、别名顺序、跨种子集合、objective语义隔离/混训拒绝、改池/改目标上下文和超范围弃权。公共自动构建的粗观测算子与冻结退化算子逐项相等，QP命名field端口指向真实重建节点。已有P1/P2、平台集成、三表及旧能力评分回归纳入最终检查；不是全项目所有测试或正式实验。

模型与画像学习仍是有限开发功能，不能据此声称跨区域泛化、组合收益或精度优势。当前验证的是指定合法候选池和支持域中的无参考推荐，不是七模型原生运行、任意串联或动态重规划。后续正式实验仍需按待审方案冻结数据、指标、独立评价与预算，不能直接重用开发参考作最终盲测。


## 第48章 小规模空间重建实验：组合局部收益与评分选优负结果（2026-10-23）

本轮已按批准方案通过平台公共入口完成实验，不再仅为框架验收。详见docs/spatial_reconstruction_pilot_results_20261023.md；冻结与结果目录output/spatial_pilot_20261023_v2/。原始数据及生产算法、评分器、支持域门未改动。

### 48.1 数据隔离、训练与执行

复用既有ERA5京津冀t2m 24×24参考，固定2019年8月1—14日每日00/12时，2倍及3倍退化重建，共48任务。1—4日16任务/128图训练，5日4任务/32图内部留出，7—9日12任务/96图验证，11—14日16任务/128图审计；6、10日间隔。四个插值内核和已有coarse_consistency组件由公共冻结池生成四单图、四QP组合，全部384次完整图真实执行，v2无质量失败。节点缓存命中保留，不冒称每次内核均重新计算。

新学习模型era5_spatial_pilot_20261023_v2由train_profile_matching训练，沿用RF64/depth4/minleaf2/seed17，128条拟合记录、32条独立日期留出记录，未调参；画像学习证据仅含拟合记录。保留固定双线性（硬一致场景加QP）、三表规则、学习评分、10种子均匀随机及成本排序。所有推荐/执行使用submit_search公共入口；288份无参考推荐real_executions=0、best=None。每任务top1在参考评价前确定，全部审计选择统一封存后才开启审计评价，各策略共享完整真实产品矩阵。

### 48.2 审计结果

主指标为固定共同内部区域的面积加权RMSE，单位K，逐任务后等权平均。无硬一致性要求M1：固定0.513075、随机0.507466、成本/规则均0.669274、学习含回退0.651850。学习比固定差27.05%、比随机差28.45%，四个审计日期的日均比较均未胜过固定和随机。

硬粗网格一致性M2：固定双线性→QP 0.447732、随机0.494273、成本/规则均0.432148、学习含回退0.431774。学习比固定低3.56%，但规则与成本选择完全一致；学习只比规则低0.0865%，不能归因为稳定学习优势。每场景审计学习仅1/16任务启用、15/16回退；验证为3/12启用。只读越界诊断发现多个数据形态特征超出开发min/max支持域，未放宽门槛。完整目标包含粗中心外边界，连续性质规则不能借用内部保证，软评分UNKNOWN且区间重叠，退回成本/并列次序；硬约束确实筛为四QP图。

组合与评分分开解释：双线性加QP平均RMSE 0.513075→0.447732，降低12.74%，16胜0负；PCHIP 0.457578→0.432148，15胜1负；三次0.444520→0.420080，14胜2负；最近邻持平。全域/边界、梯度、值域及粗残差均保留，不保证所有指标改善。池内事后最好组合相对最好单方法14胜2负、平均低0.024446 K，是后验诊断，不是自动推荐成绩。

### 48.3 回读、撤回证据与限制

独立tools/audit_spatial_pilot.py核对384个产品SHA、坐标、单位、有限性并重算RMSE，最大差2.22×10^-15 K；288份无参考推荐、审计封存先于评价、训练证据日期与冻结源码/原始输入指纹检查通过。指标/完整队列/审计禁用评价的3项定向测试通过，不把此前118项回归算作本轮重跑。

首版output/spatial_pilot_20261023_v1/保留120图评价失败：参考携带无关lsm辅助坐标。v2只在实验副本移除非维度辅助坐标，未放宽公共评价器；首版不训练、不混入最终成绩，脚本/撤回原因保留。

因此：本轮支持特定退化重建任务中部分组合的数值收益，没有证明三表评分或学习排序的稳定选优优势。仅一个已用区域/变量、人工退化、4审计日期组，不是新独立观测、跨域泛化或模型运行效果；Profile为有界生成排序，不声称GOLEM进化搜索创新。本轮未根据审计成绩修改门槛、重训或追加实验。


## 第49章 扩展空间重建与可靠选优迭代（2026-10-23）

详见docs/spatial_next_results_20261023.md；证据目录output/spatial_next_20261023/。七基础方法及七QP组合全部走公共入口。训练仅2013、开发验证2014年3—6月、最终审计7—12月；支持域未放宽。新增逐项弃权诊断、已有任务内排序头公共接入及明确标识的开发固定回退。

M1最终24任务：学习覆盖87.5%；推荐RMSE=1.204391 K，强固定=1.204807，规则=1.989462，随机=1.629858。

M2最终24任务：学习覆盖87.5%；推荐RMSE=1.204391 K，强固定=1.204807，规则=1.350801，随机=1.441899。

独立核对7252个产品，重算336张审计图，定向测试96项最终通过。全部迭代负例、诊断、组合持平/负收益和日期差异保留。六审计日期组仅作有界探索证据，不能宣称任意区域泛化、下游模式改善或普遍算法创新已证实。
相对强固定平均仅低约0.0345%，24任务1胜1负22平，六日期描述性区间跨零；这部分优势尚未得到可靠证明。组合本身另有正证据：三次→QP平均1.298212→1.204807 K，24/24改善。新发布模型默认采用已选策略，页面和公共入口均直接读取默认值；不需实验客户端另外硬编码排序。

## 第50章 背景场与GP残差协作算子接入（2026-10-23）

检索官方GitHub后复用已安装scikit-learn GaussianProcessRegressor，新增fusion.gp_residual，默认注册表及公共submit_search/Profile冻结候选池可用。详见docs/gp_residual_integration_20261023.md。

同一时刻、同变量和单位的二维背景与训练站形成残差，固定正信号/噪声标准差及米制相关长度，零均值GP预测残差后加回背景。参数必须明确提供，关闭优化器，不读取评价参考；拒绝非法角色、时间/变量/单位冲突及背景覆盖外采样。保留背景、修正、条件残差标准差和输出时间坐标，不把条件方差当总误差或自动融合权重。

公共Profile无参考推荐0次执行，以及有独立评价资源时的真实执行/产品回读通过。新增10项、平台/画像相关38项、既有融合7项，共55项不同定向/回归测试通过；时间坐标修正后新算法10项重跑通过，不重复计数。证据为output/gp_residual_tests_release.xml、output/gp_residual_fusion_regression.xml和output/gp_residual_tests_time.xml。中间测试夹具与旧注册数量断言失败保留，现注册数量对应增加一项。

未改三表评分政策、学习支持门、旧冻结实验池或生产模型；新增算法没有无依据精度/守恒等保证。此次为协作算子接入，不是内部四阶段均可独立搜索替换，不是专用station_covariates Profile全面贯通，也未运行新的真实数据效果实验。后续可用于已校准地面PM2.5背景加同期站点修正，仍需保留纯站点和无修正对照，不能声称已胜过IDW。

## 第51章 公共多阶段算法协作接通与数值验收（2026-10-23）

详见docs/collaboration_integration_20261023.md。此次补连接机制：各来源可独立校准/跳过后融合、各来源重网格后融合、重网格→可选校准→融合、融合→可替换残差估计→背景相加、融合→非负投影。残差候选实际调用GP、普通克里金、MetPy线性，线性要求完整空间支持；校准包装复用已有fit_affine_source。公共bind_registry、Profile、三表、质量门和页面处理函数共用这些节点。

来源端口分阶段避免循环，残差/背景在构图时共享同一父图且执行时核对指纹。冻结池保留结构直通但排除未请求科学节点，任务内空间别名不误换，所有权筛选不能隐藏枚举截断，适配器实现纳入指纹。默认3科学节点不变，明确请求可设到6，验收五节点图显式设5；实际分支算法逐个计数。

80项去重测试通过；独立审计32份组合产品SHA、有限性、单位及适用的背景+残差数值算式全部通过。证据：output/collaboration_release2.xml、output/collaboration_guard.xml、output/collaboration_clients.xml、output/collaboration_independent_audit.json。修复联合执行暴露的可变端口类型闭包和约束后标量时间丢失，失败记录保留。页面是处理函数实际执行验收，不冒称浏览器上传。

这是合成数据调用真实内核的运行可用性验收，不是环境效益实验、任意算法串联、专用station_covariates Profile全贯通或已证明画像选优。没有新装依赖、重训生产学习器或放宽支持域。新增组合需按新池冻结后进行真实数据独立评价；GP条件方差和仿射缩放输入标准差均不自动当总误差。

## 第52章 可版本化属性注册与三表评分参数校准贯通（2026-10-23）

详见docs/registered_joint_scoring_20261023.md。公共Profile现采用joint_portrait_v1：真实语义与端口/硬要求筛选后，按三表适配区间下界、声明成本和节点数安排推荐，交付首个通过质量门与明确阈值的候选。recommended_key在执行前确定；事后实测最优另列hindsight_best，不替换评分推荐。旧学习指标/排名仅诊断，affects_selection=false，旧task_rank和development_fixed不再替代联合评分；偏好变化时旧任务内排名诊断也不跨目标解释。旧结果和模型不覆盖，本轮未重跑其正式效果比较。

新增attribute_registry.py及默认portrait_attributes_v1.json，支持有来源的属性注册、软项停用、父版本SHA、创建时间、追加修订理由和不可覆盖快照。硬消费者与软启停分离，初始硬词汇不可删除/改成无效提供器；未知属性保留None及区间。当前消费者为既有保证、明确声明成本和尚无测量器的未知项，不允许任意配置代码，也不声称新物理测量可凭配置自动获得。

scoring_calibration.py及公共calibrate_profile_scoring复用完整、真实执行并回读的分组开发矩阵，核验产品SHA、执行指标、任务损失、候选覆盖和来源隔离。只搜索有支持、已请求项的有界软参数，按训练组选择遗憾校准，留出组只评价；不执行新图、不训练替代选择器、不改硬要求或用户评价偏好。参数版本绑定注册表、池、目标网格、模型角色/字段与需求，异常指纹/上下文拒绝。输出未知数量、激活方差和任务内贡献区分度诊断，不自动删属性。

端到端实际证明：属性启停/版本回溯改变算法及数值产品；新增有消费者属性参与决策、未知项不伪填；同一输入改变权重会切换单算法与非负投影组合；关闭软项仍保留硬过滤；校准版本经页面处理函数和文件入口真实执行/写出/回读；故意逆向的旧学习排序不能覆盖三表。128项去重定向/回归通过，含P1/P2和既有协作。独立不调用原评分器的回放核验27份决策记录、46份产品，全部通过；最终证据为output/registered_joint_release_acceptance.json及registered_joint_independent_audit.json。

本轮仅合成输入调用真实数值内核的功能与边界验收，没有下载数据、安装新框架、运行正式环境效果实验或将合成参数发布为生产默认。源码和说明明确保留：规则适配不等于精度收益，最低分组数不是统计充分性，未实现自动发现物理属性/自动生成测量器或在线自改参数。中间失败与旧断言纠正保留；依赖弃用警告未隐瞒。


## 第53章 多源重建与三表实测评分探索性实验（2026-10-23）

详见docs/多源重建与三表评分实验_20261023.md。复用原始train40站、GEOS RH35/lev72及ERA5协变量，按日期分开发16任务、留出8任务、最终测试24任务；每任务固定遮站队列及11张公共Profile合法图。528份真实产品与1584次训练站三折图评价完成，独立重读SHA/误差/隔离通过，130项回归通过。原始20个保留站未引入，自然缺失不计真值，坐标WGS84仍为工作假设；这是已研究档案的探索性重新隔离，不是新盲测、未来预测或CMAQ原生运行。

三种特征配置复用既有RF内核，并与已有普通克里金/GP残差、背景叠加和非负投影组合。station_probe作为可选注册消费者，将可用站三折的RMSE/MAE/绝对Bias归一为-e/(s+e)，与原声明成本按相同三表公式加权；开发矩阵只校准软乘数，硬规则不变。推荐先于独立误差评价；事后最优独列。没有新增黑盒排序器。它需要训练观测和真实试算，不是零样本画像预测；额外成本明确计数，未知科学属性不伪填。未改全平台默认参数。

最终平均任务RMSE：最强固定普通克里金8.746970；开发所选固定组合9.579966；初始评分8.805814；校准评分8.785213；随机期望9.200855。校准比随机均值好4.52%，比最强固定差0.44%，比初始仅好0.23%；6日期分组区间均不足以证明稳定优势。GEOS特征RF加克里金残差相对自身均值改善5.99%，联合特征RF加克里金改善0.56%，气象RF加GP反而变差6.47%。组合有局部均值收益，但未胜最强固定方案；功能贯通不等于创新效果获证。评分与负误差秩相关均值仅0.0814，4/24次命中池内事后最优，仍需改进实际辨别力。

首版因普通克里金误检查未使用协变量的满秩，导致冻结池上下文变化而拒绝测试，未产生测试成绩；修复接口后新版本重跑开发/留出/测试，旧失败和产品保留。未根据最终测试调整规则。最终证据目录output/multisource_joint_20261023_v2，含冻结协议、环境、源码、完整矩阵、校准选择、独立结果及负结果。


## 第54章 三画像状态评分、响应记忆与遗传寻优接入（2026-10-24）

详见docs/三画像状态评分与遗传寻优_实现验收_20261024.md。显式portrait_policy将版本化模型场景、注册状态属性、算法状态转移和软参数校准接入公共Profile入口，页面/文件共用；旧默认入口保留。评分为按场景权重与软乘数加权的输入至终态风险下降，未知响应保留区间，参数校准与在线评分使用一致区间。逐节点更新状态；分组开发记录学习实际指标变化，不另以黑盒排名替代三表，最终测试拒绝训练，同内容改名不构成新独立组。

七模型模板保留未知科学敏感性；本地文档/源码线索需人工审查，不能自动证明敏感性。显式多变量求和关系可作硬检查，但不代表全面动力一致性。复用官方GOLEM多目标进化，记录父子图、有效/无效变异及Pareto集合；当前128张上限的冻结图域可精确核查保守评分最优，非真实效果全局最优，也不声称评分计算加速。

最终相关52项回归通过（regression_release.xml）；此前较广135项通过（regression_final.xml）。acceptance_v3含6次合成场景及1次真实多源入口验收，35份实际数值产品，真实多源11图全部执行。独立审计回算风险/区间/推荐顺序并回读产品SHA，7份决策、35份产品通过，11次实际改变图的有效原生变异。已验证数据和场景偏好变化影响推荐，未知项不伪填，推荐不读取参考真值。响应库8条整链及8条算法响应仅为功能样本。

证据目录output/joint_state_20261024，含独立审计、最新产物、源码发布指纹；旧中间证据保留。未安装新依赖、未重跑正式效果实验、未验证原生WRF/CMAQ运行，未将合成场景设为生产默认。第53章未胜最强固定方案的负结果不变；本章证明实现生效，不证明科学优越性。


## 第55章 逐源角色、阶段顺序、时段与协调约束的组合决策（2026-10-24）

本章取代第54章中以GOLEM为新机制载体的路线：新入口使用三画像初筛后独立单轴/多轴变异，不调用GOLEM优化器。旧入口只作兼容。详见docs/组合决策机制_实现与验收_20261024.md；公共submit_search的Profile模式传入composition即可，空配置可按实际画像/兼容性建立候选域，页面处理函数、文件和实验同入口。

方案含字段、时段、逐源处理链、主辅角色/权重、汇合位置和后处理。已真实实现先后依赖及可选顺序、早/晚汇合、各源不同算法、重叠时段处理、上下文检查、时间插值与累计区间边界核对。观测和协变量通过专门资源端口进入已有算法，最终测试标签不得进入拟合。多字段联合选择，显式求和关系验收8种分配中通过2种、拒绝6种。未知动力约束不伪造，未实现硬消费者拒绝；阈值必须有任务依据，不能据此宣称所有物理协调已自动保证。

平台注册89项、67项available（含适配器，不等于独立科学方法）。新增一个有真实执行端口的主辅加权均值，复用现有数值内核，不把人为权重冒充不确定度。真实档案两个相关站点重建产品、4种重建方法的冻结域为12×12×2=288种结构；全部执行后240种通过，48种因最终非负约束失败，产生32种逐元素精确值不同的通过产品。自动扩展域为14×14×8×14=21952种结构；公共CLI实际搜索20次、18通过2失败，全域可用总数尚未穷举，不填造数值。

最终157项回归通过，含12项新定向测试及原P1/P2、协作和公共入口检查。独立审计7次平台决策、538份产品SHA/数值/评分/推荐顺序，以及88条真实改变基因型的提议通过。变异阶段与附加穷举审计严格分开：时段功能案例内部评分提高0.222222，真实固定域变异提高0，均记录；不作环境精度优越性声明。中间元数据缺失、同分冗长方案和兼容性错误修复及旧失败保留。没有安装依赖、重跑正式效果实验或原生WRF/CMAQ验证，第53章负结果不变。

最终证据output/composition_20261024/acceptance_v5、independent_audit.json、regression_verified.xml及源码发布指纹。可直接使用examples/composition_real_288.json或composition_auto.json；旧档案的WGS84工作假设、相关源性质及科学证据限制全部保留。已有旧响应库不会自动套用到新图语法，未知画像/敏感性仍不瞎填。


## 第56章 全量三画像初筛、原生变异接入与互补时空融合验收（2026-10-25）

本章修正第55章初筛时随机抽16种的逻辑：当前为全部合法候选先三画像评分，再进入变异及真实执行；超过完整筛选上限明确拒绝，不随机截断。复用逐源、阶段、时段及协调公共入口，新增显式粗观测一致的线性残差投影融合。新请求接入官方ALNS7/AlphaUCB，并真实比较pymoo MixedVariableGA；不使用GOLEM优化器。开发同分依事前工程规则暂选ALNS，不构成算法优越性或全局最优证明。

NCEP真实温度受控降采样形成12×12/48小时与6×6/6小时互补源，恢复隐藏细场，56结构全部评分及真实执行穷举参考；非独立传感器或WRF验证。原轮评分推荐RMSE1.507611K落后开发最强固定1.211714K，完整保留。之后仅原四组开发真实执行数据校准算法画像误差二阶矩，继续三画像风险差联合评分，未知范围不填造，不另训替代排序器。冻结新四组2014日期及5种子后测试：校准推荐1.189751K，原评分1.361006K，粗源1.881679K，细源5.076219K；相对原评分降低12.58%、相对粗源降低36.77%。开发最强固定也为1.189751K，池内事后最优1.175612K；ALNS、basic、随机扩展及仅初筛相同。因此仅支持组合收益及本任务评分修正有效，未证明优于最强固定或变异额外收益。随机扩展共享画像初始方案，不冒充完全无评分随机推荐。

最终143项不同定向及兼容测试通过（145次执行）。原轮独立审计170次决策2192份产品；新轮96次决策1040份产品及224份开发产品二阶矩独立核验通过。文件、页面处理函数及实际CLI无参考真值选图/数值指纹相同；未宣称浏览器逐控件或原生下游模型运行验收。不可变画像已发布data目录，直接可运行配置与源码指纹在output/complementary_20261024/delivery。完整结论和边界见docs/互补分辨率融合_最终交付_20261025.md；冻结协议、全部负结果与审计均保留。第53章及更早冻结池证据限制不变。


## 第57章 完整分时逐源组合与真实验证误差反馈实验（2026-10-25）

已按用户确认路线在公共平台接入SearchValidation专用反馈：三画像先完整筛选，再以真实验证RMSE接受变异，最终测试不反馈。原生ALNS/AlphaUCB继续复用，新增现有SciPy空间中位数适配器，未知画像不填造。每时段细源时间处理1×粗源空间方法/滤波先后6×主辅或投影3×后处理2=36，两时段独立共1296结构，固定同策略36；不是整库全部可能组合。四个开发及四个验证案例全域真实执行，另有四组后续日期测试。

四验证案例×五种子、各64次唯一评价：ALNS命中19/20，普通变异7/20，纯随机0/20，仅初筛0/20；未命中保留。独立测试平均RMSE：ALNS 1.188895K、普通变异 1.206315K、纯随机 1.292468K、仅初筛 1.265064K、开发期最强固定 1.186976K。本轮ALNS比纯随机更容易在预算内命中验证全局最优。 不把命中验证最优等同测试最优或普适显著优势。

全局参照明确为冻结1296种合法图的最低真实验证RMSE。预算变异不保证全覆盖；完整覆盖配置先变异后穷举剩余，可认证有限域验证最优，不能归功于变异或宣称真实未知误差全局最优。148项不同测试通过，独立审计88决策、14248产品及最终测试产品通过。文件、页面处理及CLI同结果；图/顺序实际数值差异及分时局部性有产品证据。

数据仍为同一真实NCEP场的受控互补分辨率构造（12×12/48小时与6×6/6小时），非独立传感器、非原生高分辨率、未运行WRF。没有新训黑箱模型。详见docs/分时逐源组合_实验交付_20261025.md及output/scheduled_fusion_20261025全部冻结配置、源码和负结果。第56章和更早负结果保持不变。

补充完整覆盖验收：实际CLI先运行ALNS再补齐剩余1196图，总1296图均通过，独立重算确认验证最优1.4077641152217724K并返回有限域验证最优认证；此项不计入20次效率样本，原失败记录保留。验证组2230的分时滤波降低验证误差却劣于后续测试固定方案，负结果详见scheduled_policy_case_diagnostic.json。最终分时边界诊断、NetCDF属性序列化与CLI模块类身份边界均修复，原科学源码/结果及修复差异分别封存，72个不同推荐/测试图重放差不超过1.14e-13K。


## 第58章 三路并行评分与独立时空对照（2026-10-25）

新增公共composition.parallel_scoring：数据适用性、模型输入适配、组合协同三路并行，以输入中可见锚点遮挡的真实局部执行补齐响应，不训练替代排序器，未知维持区间。先全域评分后变异，探测72图成本单列。

8开发案例校准权重，8组四区域/双时期验证—测试案例；开发/验证1296全域真实执行。全机制测试RMSE 0.602081K，开发最强固定0.596796K；验证命中19/24。随机32图0.714661K，随机104图0.651946K。逐路消融及所有负结果保留。

分数不再全部相同；是否选优依照真实测试/消融结果判断，不以链路验收替代效果。71项不同测试、独立数值与评分审计、实际文件/页面处理器/CLI通过。数据仍为同源受控分辨率构造，非独立传感器、非WRF响应；不承诺测试全局最优。详见三路并行评分_多区域实验交付_20261025.md及output/parallel_portrait_20261025。


## 第59章 共同强基准与变异预算（2026-10-26）

共同强基准模式已接公共评分器：开发冻结最强单算法/固定组合，可见遮挡选择统一参照；协同不再线性加分，只作收益判断和同主评分下比较。

共同强基准＋ALNS的32/128/256预算验证最优命中分别为15/20/20/20；256预算初筛/后续命中为5/15。

原协同加权＋ALNS的32/128/256预算验证最优命中分别为16/20/20/20；256预算初筛/后续命中为0/20。

随机初始化、无三路软评分ALNS的32/128/256预算验证最优命中分别为7/20/20/20；256预算初筛/后续命中为0/20。

纯随机的32/128/256预算验证最优命中分别为0/0/0/20；256预算初筛/后续命中为0/0。

全程正式搜索无穷举补齐；4组新配对×5种子，测试隔离。32项测试及独立产品/评分审计通过。资源调整、未完成轨迹、重放与负结果保留，不使用穷举结果冒充变异优势。详见共同强基准与变异预算_实验交付_20261026.md及output/common_reference_20261026。
128次预算后三个变异组均20/20，测试均追平开发最强固定0.828483K。新共同基准组5次初筛、15次变异；没有证明新公式优于旧公式，预算增加的收益不能归于新公式。4个新验证案例最优仍为同一固定图；不作普适收敛或个性化组合优越性宣称。


## 第60章 SWAT+真实模型基线核验（2026-10-26）

本轮按用户要求仅核实并跑通真实模型基线，不做融合、不率定、不调整评分。原有SWAT+适配器主要为62.0.0输入准备，不能代替完整流域原生运行；原有NASA POWER天气与WASP/USGS径流未强行跨流域配对。

已固定公开SWATdata提交2cd16e3f2453063362e32350041d57c1646e780f，下载配套SWAT+59.3 Windows程序、LREW gauge J完整文本工程、同流域气象和出口观测，以及GIS。131个陆地HRU、3段河道、3个塘库；GIS134单元含3个PND水体。原样模拟2000—2012年成功，原有3年预热不变。随后只打开channel日/月/年输出，全部其他输入逐个哈希保持不变；复跑河道及流域年度水量输出逐字节一致。

出口根据路由确定为channel1，非按效果挑选；原生flo_out单位ha-m，换算为m³/s后与2003—2012年3653天同流域日均观测对齐。RMSE0.390432m³/s，MAE0.250520m³/s，NSE0.627325，PBIAS（模拟减观测）+106.891160%。存在明显总水量高估，不宣称已率定或融合获益。模拟期降水含11处-99标记，保留原工程处理，内部补值未逐日追踪；RDS未附质量旗标。

官方62.0.0与附带旧格式OSU参考工程的失败原样保留，退出2并发生温度数组越界；未完成全部根因定位。59.3配套工程运行成功不能认证平台62写出器。本轮尚缺62版工程贯通、原始观测质量复审、基线率定证据及同流域同期互补双源强迫；没有将原生模型调用包装为平台全链路完成。

详见docs/SWATplus真实模型基线核验_20261026.md；复现tools/run_swatplus_lrew_baseline.py；来源、输入哈希、原始失败/成功日志、逐日比较和独立回读均在output/swatplus_baseline_20261026。现已完成原生模型—出口径流—同流域观测—可重复基线，本阶段停止。


## 第61章 SWAT+径流高估原因审计（2026-10-26）

针对第60章106.89%高估，未调参、未改气象或物理路由、未跑融合，仅在副本打开分量输出并交叉核对同日期Rev59.3历史源码。日河道输出与前轮逐字节相同，单位ha-m转m³/s正确；gauge J出口路由和现有资料内的日期配对未发现错误，原始观测质量旗标仍缺。

主要问题已定位为rout_unit.con三条RU的同一tot水量分别按0.7送河道、0.3送塘库、1.0又送含水层，合计2.0。源码tot为地表/侧向/排水，rhg才为底部补给；非HRU接收端不自动归一化或改变水文类型。含水层将tot当补给又出流至河道，重复计水。2003—2012陆地产水6423.84万m³，含水层补给6423.63万m³，含水层出流6450.02万m³；模拟出口12927.31万m³，观测6248.36万m³，差6678.95万m³。地下水出流量级为偏差96.57%，非修复收益保证。陆地perc每年为0，进一步支持补给接错的判断。

11处缺测由源码中的天气发生器处理，依据原生降水输出与站点面积权重反推累计生成约4.5094mm，仅总降水0.0412%，不能解释普遍接近翻倍的偏差。含水层参数按本版字段还会得到49950mm初始储量，语义可疑；评价期储量下降约347.58万m³但深层渗漏约321.18万m³，持续误接补给仍为主体。没有证据将此单归为未率定。

第60章结论收紧为“原生模型和评价链路可执行”，物理有效基线验收未通过，不能直接开展正式融合比较。未修复或以调参掩盖此问题；先完成水量路由和版本/初值语义修复再验收。另纠正上游93个顶层文件含1份历史channel_yr.txt，不全是输入；此前评价使用全新生成日输出，未用历史年输出。仅更新复现辅助启动器以排除该缓存输出，不改平台公共模块。完整核查docs/SWATplus径流高估原因核查_20261026.md；证据output/swatplus_baseline_20261026/bias_audit；只读重算tools/audit_swatplus_baseline_bias.py。旧运行、负结果均保留。


## 第62章 Robit单源降水重建与SWAT+格式输出（2026-10-26）

按用户授权复用队友Robit1990—2013逐日气象，仅对降水做内部遮挡恢复，其他变量原样输出。1990—1995拟合历史月均值和尺度，1996—2001冻结强固定方案，2002—2007搜索验证，2008—2013隔离测试。三个时段/缺口案例，三种子不视为独立流域。

复用公共platform_search/Composition、三画像状态评分及ALNS+AlphaUCB，新增薄日降水适配和可见日探测范围，未另训替代排序器。候选为4单方法加36个短/长方法组合，共40结构；三画像分数确有3或4种不同区间。单点坐标不是空间网格，不宣称多源融合或空间精细化。纯长度案例有数值等价结构，不夸大40种科学行为；开发期混合缺口定向验收两节点分别真实填11与42天，原值保护与导出回读通过。

独立测试RMSE(mm/day)：固定线性10.087601、最近邻11.484134、前填12.812776、历史月均值7.747872；开发最强固定实际为月均值，仅评分也为7.747872；三画像+ALNS、无评分ALNS、随机8及随机40均8.457340，较强固定差9.16%。无评分或组合优越性证明，未以测试结果重调权重。验证更偏好前填的案例泛化较差，原负结果保留。评分采用可见遮挡RMSE和明确的输入总量偏差代理；SWAT敏感性未知，非原生模型效果评分。

每次评分+ALNS额外40探测再执行8验证图，平均0.907s；随机8平均0.587s且效果相同，无净效率优势。随机40成本对照不是严格等墙钟。独立回读开发/验证699图、480探测记录、最终21个测试图产品；另重放480次探测核对分数，不计为搜索优势。41项测试通过，失败接入记录保留。

现有SWATPlusAdapter导出五类天气并逐列回读。delivery提供2008—2013完整2192天的原始、固定、仅评分、评分变异和随机五套天气数据及索引，不带队友工程控制和WGN，不运行SWAT，不评价径流。详细docs/Robit单源降水重建_交付_20261026.md；全部冻结请求、轨迹、产品和负结果output/robit_single_source_20261026。本阶段链路验收完成，止于天气格式输出。


## 第63章 自主组合与统一真实误差判优（2026-10-27）

按本轮用户确认，新增公共Autonomous模式，页面、文件客户端、API共用；不使用旧Robit四方法40图手写池或画像软分，扫描95个注册项（73个available，含包装/校验项），依据实际父节点兼容性逐层生成连接，不设人工最大步数。参数有契约按域展开，未知仅试默认并标明覆盖未知。保留旧入口及全部旧结果。当前新自动评价仅支持单站逐日恢复，不宣称空间/多源自动评价已完成。

统一目标为各日历窗三组固定可见遮挡RMSE均值，再等权跨窗平均。2002—2006供搜索、2007供冻结方案确认；基线也是按同一目标自动选出的单算法。组合须每组确认不差且平均严格更好才替换；这不是显著性或泛化保证。1990—1995统计状态复用。历史年份已使用，本轮为机制诊断，非新盲测；没有原生SWAT评价。

最终公平轮转运行：852提议，772不兼容，80候选真实执行（22完整评分、49仅中间状态、9失败）；36个含原输入的独立工作状态。实际内核调用搜索1074、确认6、部署1，共1081；21.09秒。提出双父节点但真实单站数据未形成可执行双父产物，执行图含一/两个算法节点；合成定向测试另验收双父节点真实执行。最终仍为单算法daily_month_mean(max_gap_days=14)，J=6.666195、确认RMSE=6.174676mm/day，无组合收益。选择后旧参考回读232点RMSE=9.727667，仅探索诊断，不与上一轮不同年份遮挡的测试汇总直接比较。

80候选资源上限触发，NOT_CERTIFIED_CURRENT_BEST_ONLY；91个注册项端口数或参数域声明不完整，不能给全库合法组合总数或全局认证。状态去重对契约不完整插件不构成全结构等价证明；第三方内核尚无进程级硬超时。失败包括1次结果质量门及8次有效观测不足，原日志保留。首版顺序调度及修正公平轮转说明保留，未改数值目标或质量门追优势。

独立重算336项面板误差及窗口聚合一致，80图无环且73项均有尝试；产物原值保留、有限非负，SWAT天气文件精度回读通过。45项相关回归通过，另5项自主入口/页面检查通过，错误路径日志保留。详细docs/自主组合与统一判优_交付_20261027.md；冻结协议、两版运行、独立审计、测试及源码指纹output/autonomous_decision_20261027。只交天气，不修改队友file.cio/WGN/拓扑，不运行SWAT。

补充认证边界测试：参数域声明存在但其取值生成器无法执行时，禁止发出全局最优认证；tests_certificate.xml 单项通过。


## 第64章 开放时空互补融合实际效果（2026-10-27）

复用NCEP温度及原面积平均、时间重采样、空间重建和细节投影融合；新增公共AutonomousField，页面/API/文件同入口。95基础注册项/73可用，自动逐帧适配后158项/136可用，适配器不算新科学方法。无手写完整池或人工最大步骤，开放多父节点、逐源处理与先后连接；未知参数域保留默认范围未知。不是GOLEM或遗传变异实验，不将收益归因于三画像软评分。

8×8低时间频率源每48h一帧，4×4高时间频率源每6h一帧；每案例17时刻，隐藏中间可见精细锚点自验证选图，冻结后用全部三锚点输出，最终仅评价14隐藏精细帧。开发2、验证2、测试4时段，空间区分；测试四时段来自同一留出区域。数据是同一NCEP场的控制退化、原生2.5度，并非独立传感器或新外部数据；不验证原生WRF/SWAT。

测试主指标为等权帧/案例/种子平均空间RMSE(K)：自主1.0373，开发期最强固定1.2986，最好已发现精细单源2.5884、粗单源3.5181，同180候选随机1.9296，补充时间配对随机1.9296。自主较开发固定平均降20.12%，赢3/4案例；较时间配对随机降46.24%。固定对照从全部247个开发已发现配方逐一跨开发案例重放选择，不限制自主候选池。

保留负结果：第四案例自主平均比固定差0.1181K；三个种子全案例平均0.9235/1.0905/1.0980K。对固定的四时段配对改善描述性95%重采样区间[-0.0105,0.5331]K，跨零，不支持稳定普遍优越性。全部测试与补充对照产物先冻结哈希再评分；未据测试改权重或替换种子。

共48次正式运行，9767候选实际执行、29826内核调用、478失败；失败与早期V1—V5诊断全部保留。测试自主平均25.05秒/752.3内核，原随机18.90秒/348.3，时间配对随机26.23秒/629.9；并发及其他负载限制计时，不声称效率最优。另有开发固定对照64.03秒/508适配器调用；诊断审计不计为搜索优势。资源上限不等于合法空间，未闭合，不能给穷举最优或全局认证。

23项最终相关回归通过；独立重放36次最终选图的搜索评分及部署产物均一致。交付前仅补强终端物理语义拒绝门，额外验收48次旧选中图仍合法、损失一致，保留冻结版本/增量diff/当前版本；未重跑全部搜索，不把旧耗时称为补门后新计时。详见docs/开放时空互补融合_实验协议_20261027.md及docs/开放时空互补融合_实验交付_20261027.md；证据output/open_complementary_20261027。结论限于本控制基准的融合和输入内选图收益；无全局最优、变异创新、独立传感器或下游模型优势证明。


## 第65章 Robit与ERA5第二来源降水融合（2026-10-28交付）

使用队友Robit日天气与新下载ERA5降水，1990—2013共8766日。站点11.6N/37.36E，ERA5最近0.25度格点11.5N/37.25E约距16.32km，作为站点代理而非实测。ERA5不直接同化本地区雨量计日降水，但Robit生产谱系未核实，不能认证绝对独立。当地UTC+3为主日界、UTC作预先声明敏感性，不按测试成绩换日界。请求/原始JSON/CSV/哈希均保留，没有上传站点值。

原站点完整，本轮为遮挡恢复，非提高完整原观测精度。1990—1995拟合，1996—2001开发选固定，2002—2007验证选图，2008—2013最终测试。随机单日/连续3日/连续14日每年约20%遮挡；各情境算法输入同时移除所有开发、验证、测试隐藏值。标签不参加融合；可见原值精确保留。48份预测冻结后计测试。六年份为分析块，种子与遮挡类型不是独立扩增样本。

复用公共AutonomousField/DAG/时间恢复/融合/xsdba，新增点序列缺失回退、绑定训练状态校正与非负适配。基础100注册项、78可用，含包装项，不是100种降水科学方法。开放实际接口连接、不手写完整图池；本轮不是GOLEM/遗传变异或三画像软评分消融。27次正式运行每次180候选，三种子11/29/47；4860候选真实执行，其中4719完整输出、141中间状态；48900不兼容提议保留，正式执行失败0。固定对照从开发发现591配方、582种三情境均完整者选择，不是全库穷举。

测试平均年度RMSE(mm/day)：站点最强单源8.4112、ERA5原值8.4878、先验校正ERA5 8.3502、开发最强共同固定8.1255、自动8.1286、随机8.1429。自动较站点降3.36%，分单日/3日/14日为0%/5.84%/3.70%；对共同固定反而差0.0384%，较随机好0.1757%。年度配对改善的描述性95%区间对站点[-0.0276,0.7296]、对固定[-0.3416,0.4515]、对随机[-0.0132,0.0478]，均跨零。负结果保留；未证明稳定来源收益、搜索优势、极端降水全面改善或全局最优。

验证自动平均37.56秒、随机19.96秒，虽均180次执行但xsdba训练调整调用54/27次，不能称真实计算成本相同或自动更高效。开发固定重放另125秒/1776适配器调用；27次正式搜索含135次xsdba训练调整。V1—V5诊断不混入V6效果，点序列/区间起止/局部NaN问题修复后正式重跑。测试后只有天气日期索引与审计路径修补，不改科学结果，冻结源码及补丁说明保留。

80项回归通过，独立重放27选图评分、核查48预测哈希和信息隔离通过。交付15套2008—2013共2192日的天气试验包（三情境×单源/ERA5/固定/自动seed11/随机seed11）；五类气象列回读通过，非降水原值保留。只交天气，不覆盖队友file.cio/WGN/流域拓扑，未运行原生SWAT。详细报告docs/Robit_ERA5降水融合_实验交付_20261027.md；证据output/robit_era5_fusion_20261027，数据data/robit_era5_1990_2013。


## 第66章 已有ALNS变异接入公共平台与Robit补跑（2026-10-28）

按用户要求纠正上一轮未使用变异的路线：新增明确PortraitMutation公共模式，页面、API、文件客户端同入口；先三画像评分/契约筛选，再使用已有ALNS 7.0.0库、AlphaUCB选择与HillClimbing接受机制，适配实际开放DAG的算法替换、参数、节点增删、源/分支重连与顺序交换，不改用GOLEM或开放枚举。复用原Composition状态风险减分公式及Executor，不更换算法数值内核。未知保持区间；空邻域不获同分奖励，已非负的恒等裁剪跳过，同分探索奖励为0。旧有限轴入口保留原行为。

唯一匹配的注册模型/角色/字段/任务策略及属性版本可自动加载；无标签的逐日点序列从真实可见值构造隔离遮挡，不向节点传标签，不进行额外确认集选优。多策略歧义和不支持的自动评价明确报错；未认证任意数据/模型全自动或SWAT敏感性。数据/需求改变画像分支优先级、真实父子变异、同入口与信息隔离均有测试和公共入口实跑证据。

复用第65章输入、训练状态和遮挡，哈希不变；1990—1995拟合、1996—2001开发、2002—2007选图、2008—2013计测试。三遮挡×种子11/29/47×ALNS/同邻域随机/仅初始化，共27次。上限180图评估请求/40000提议/120秒；初始化计入，完整缓存不记新增计算。保留旧单源、开发固定、开放搜索及随机，新增27预测加旧48共75份，先冻结后统一计分。测试上一轮已看过，本次是同数据机制补跑，不是新盲测；未按测试改机制或换种子。

测试等权年度/情境/种子RMSE(mm/day)：仅初始化8.3742、ALNS 8.3133、同邻域随机8.0958、开发最强固定8.1255、旧开放搜索8.1286、旧开放随机8.1429、站点单源8.4112。ALNS对初始化改善0.73%，对固定差2.31%，对同邻域随机差2.69%；9次有4次降低验证目标，全部验证不劣于初始，但没有测试优势。对初始化/固定/随机年度配对改善描述性95%区间为[-0.0597,0.2420]/[-0.3914,0.0225]/[-0.5060,-0.0433]，六年且历史已探索，仅作诊断。

必须保留的负结果：9次初筛各32个兼容算法候选全部同一画像区间，序列算法响应缺证据，不能说调用了评分公式就有效选优。1237条ALNS父子图评估覆盖重连342、算法317、参数201、插入200、删除147、交换30；选中图两张膨胀至37/40算法或校验节点，不是37/40种科学方法；仅2/9最终图同时使用两源算法分支。其余按效果保留单源，不强迫融合。

ALNS共1543图请求、2640新增适配器调用，部署另99次，平均93.37秒；同邻域随机1616/2448/另72次/45.19秒；仅初始化306/288/另9次/20.07秒。ALNS子图965次有新增调用、251次全缓存、21次无新增调用前拒绝；随机1197/90/23。ALNS可评分1386、中间42、失败115，随机失败45，均保留。成功轨迹可见xsdba调用至少39/32/27，失败上游记录不全，不能称精确分类总量；总适配器调用和墙钟包含失败消耗。无效率优势、穷举补齐或全局最优认证。

57个不重复相关测试通过；最终自验证入口实跑、独立重放27选图并按年重算、75预测哈希和输入隔离核查通过。233文件源码冻结后未改科学代码。新增9套三情境×ALNS/随机/初始化seed11天气包，每套2192日，五类天气回读通过；其他种子全量预测保留。不跑SWAT、不覆盖完整观测/file.cio/WGN/流域拓扑。详见docs/Robit_ERA5变异补跑_协议_20261028.md、docs/Robit_ERA5变异补跑_交付_20261028.md及output/robit_era5_mutation_20261028。结论为自动变异链路接通，研究中的效果选优目标尚未得到支持。


## 第67章 多源响应评分、科学去重与联合变异验收（2026-10-28）

复用平台PortraitMutation公共入口，修复同误差增肥接受、完整科学状态去重，新增多源遮挡响应评分和真实联合破坏—修复。保留来源/校正/单位/区间/拟合状态，未知不填；独立遮挡缓存从根重执行。ALNS与随机统一最好父图及180新增注册算法调用总账，探测/失败/部署计入。50项定向回归与旧Robit三场景真实链路验收通过。

新数据为三处西澳GHCN雨量站＋ERA5降水；2014—2017拟合、2018—2019开发、2020—2021验证、2022—2024测试。3/14日遮挡、3种子、4搜索组共72次，另保留开发固定/单源/旧转移；108份预测先冻结后测试。主年度RMSE：完整3.523148、随机3.451785、去响应3.494671、去联合3.485021、共同固定4.111305、逐案例固定4.128171 mm/day。完整对随机改善-2.067%，对共同固定14.306%，对逐案例固定14.656%；负数为退步，全部保留。

完整成本和逐种子负结果见详细报告。模型准备画像权重仍为声明先验、总量偏差为输入代理，不是SWAT实测响应；仅3站，不能推断普遍优越或认证全局最优。108预测独立重放、信息隔离、父图/接受/总账/哈希审计通过；30份仅降水输出每份1096日回读通过，不伪造其他天气、不运行原生SWAT。详见docs/三画像响应与联合变异_实验交付_20261028.md及output/unseen_portrait_response_20261028。


## 第68章 小流域七源生态重建与自主随机变异（2026-10-28）

完成黄山附近HydroBASINS出口4120774820完整上游1056.6km²的2013—2017七类数据下载，740份卫星/高程原生窗口、5年CLCD和4份气候空间子集。公共接入核对0.1mm/0.1℃转换、云/饱和/LAI/NPP QC、坡度先细算与类别面积比例。FVC为Landsat训练端元代理，不是地面真实覆盖度。

公共入口新增通用栅格缺口/先验协变量预测适配与明确应用任务契约，复用既有SciPy、feature_forest与随机变异/图去重/预算执行。主搜索没有手写完整候选池或固定特征组，节点、特征子集、参数、边和最终图由平台自己选择。栅格不适用点序列探测时保持UNKNOWN；不冒用WRF/SWAT敏感性。39项回归、6次开发和18次正式搜索通过。

2013—2014拟合、2015开发、2016选图、2017测试；3个互不重叠8×8块、3种子。33份主预测和9份验证选出的最佳已发现单方法预测均先封存再测试。FVC RMSE：随机0.06796952、不变异0.06890233、开发固定0.07625406、全协变量森林0.07831975；随机对不变异/固定改善1.353822%/10.864400%，对最佳已发现单方法改善1.353818%（负数为退步）。不把末位数值噪声、选到单算法或网格对齐称为组合创新。

全部候选真实执行，完整失败/负结果、成本和软墙钟超限保留；独立重放、源码/输入哈希、训练目标隔离和单位审计通过。单流域单测试年，不支持普遍优势或全局最优；未运行生态/水文原生模型。详见docs/小流域七源生态重建_实验交付_20261028.md及output/basin_ecology_2013_2017。

补充证据边界（本轮七变量）：辅助变量森林实际执行阶段/次数为{'three_portrait_initialization': 36}，9个随机最终图均未保留森林；不能把本次FVC处理组合改善叫作七变量联合融合收益，也未充分探索全部跨源分支。胜出图仍有重复插值操作，未逐节点证明贡献；保留原结果，不按测试再修图。


## 69. 半日期隐藏时空重建（2026-10-28）

# 半日期隐藏高分辨率重建：实验交付

平台公共入口自行生成候选、真实执行并随机变异。2016选图冻结后，2018部署接口不接收测试标签。

## 数据与实验

同一流域三个48×48细像元区域；Landsat8约30m与独立MOD09Q1约250m/8天，统一计算NDVI。目标是缺失日期的细网格恢复，未降到1km。0.0003度细网格实际约29m×33m，不是精确30m正方形。
2013—2014仅用于区域QA；2015开发；2016验证选图；2018测试。全年影像中每日期至少30%像元QA有效，交错与连续中段各隐藏floor(n/2)个符合条件日期，两端锚点保留。云覆盖像元不作真值；所有方案必须在同一固定有效参考支持评价。原季节影像/70%清晰门限因2016年仅两个日期而在效果实验前修订，保留旧记录。
参考是QA清晰Landsat派生NDVI，不是地面观测；MODIS光谱及复合期内观测日差异保留，不宣称精确守恒。

2017年首轮诊断结果完整保留：随机变异0.06229850、不变异0.06110817，变异较差；粗源插值对照因遗漏data_kind连续类型标记被拒绝，属于接入错误。补齐已知物理类型标记后重新冻结；2017不再充当新盲测，新增2018年作为最终留出年。没有根据2018误差调权重或选路线。

| 方法 | 隐藏日期等权RMSE | MAE | 梯度误差 |
|---|---:|---:|---:|
| random | 0.10329973 | 0.07694368 | 0.04031150 |
| none | 0.10524780 | 0.07865283 | 0.04083842 |
| development_fixed | 0.10075424 | 0.07483207 | 0.04156857 |
| fine_time_linear | 0.12180452 | 0.09368888 | 0.04269629 |
| fine_time_nearest | 0.12346716 | 0.09213987 | 0.04700010 |
| coarse_only | 0.11513650 | 0.08190790 | 0.05574634 |
| anchor_residual | 未完整完成 | — | — |
| anchor_residual_physical | 0.10581251 | 0.07854427 | 0.04294656 |

## 结果边界

| 案例 | 开发期可见/隐藏 | 验证期可见/隐藏 | 测试期可见/隐藏 |
|---|---:|---:|---:|
| patch0_alternating | 5/5 | 4/4 | 3/3 |
| patch0_contiguous | 5/5 | 4/4 | 3/3 |
| patch1_alternating | 5/4 | 4/4 | 4/4 |
| patch1_contiguous | 5/4 | 4/4 | 4/4 |
| patch2_alternating | 4/4 | 6/5 | 3/3 |
| patch2_contiguous | 4/4 | 6/5 | 3/3 |

- 随机变异相对 none 的平均RMSE改善：1.8509%；负数表示更差。
- 随机变异相对 development_fixed 的平均RMSE改善：-2.5264%；负数表示更差。
- 随机变异相对 fine_time_linear 的平均RMSE改善：15.1922%；负数表示更差。
- alternating：random=0.10144238, none=0.10272370, development_fixed=0.09238559, fine_time_linear=0.12257466, fine_time_nearest=0.12118999, coarse_only=0.09850643, anchor_residual_physical=0.10086313
- contiguous：random=0.10515708, none=0.10777190, development_fixed=0.10912289, fine_time_linear=0.12103439, fine_time_nearest=0.12574433, coarse_only=0.13176658, anchor_residual_physical=0.11076190
- random：每次平均实际调用179.22，平均服务时间13.25秒；缓存不重复计调用。
- none：每次平均实际调用27.50，平均服务时间0.95秒；缓存不重复计调用。

随机变异最终图使用两个源分支：17/18；不能仅凭下载了两源就声称发生了融合。

全部逐种子结果、失败、真实调用账本、初始化及变异轨迹保留。开发最强固定来自发现图的跨案例真实重放，不是理论全空间最优。追加高变化时段分层只用开发阈值，不重新选图。
单流域、三个空间相依小区域、一个测试年；重复种子不增加独立流域样本量。不能由此声称普遍最优、恢复未知真实细节或改善SWAT+/WRF。

## 实现与验收

复用公共搜索、画像、状态去重、随机变异、DAG执行及质量门。补充公共栅格时间插值与可见锚点残差节点（不是STARFM），补充固定有限参考支持及无标签固定图部署。没有新增候选整图池；未知栅格响应仍标UNKNOWN，没有用另一套学习排序代替画像。
首版开发的float32/float64误差计算差异已修正：统一双精度误差计算，使用按参考规模计算的舍入同分尺度，再比较图复杂度与成本；精确网格中心插值不再在相同完整网格上重复执行。首版结果全部留在diagnostic_float32目录，不能当作正式优势。正式数据、种子与预算保持不变后重跑。
最终轮在开发年验收所有固定对照后才运行正式验证。原始残差控制仍保留；另外预先冻结残差加NDVI物理范围投影控制。范围投影在无违例的新输入上合法返回原值，搜索仍跳过已知恒等投影，不把这种部署边界当作算法失败。
同样修复栅格补缺/时间插值的跨年幂等边界：新输入已经完整时可原样继续执行，搜索仍不添加这类无效节点。旧部署失败在2018参考未评分时发现并归档，修复后重跑开发与验证。
独立审计：输入哈希及阶段隔离通过；所有封存产品用公共接口重新执行并逐值核对；正式运行初始化成对核对，失败不剔除。详见independent_audit.json。

## 最终验收补充（2026-10-29）

最终留出年为2018。随机变异RMSE=0.10329973，不变异初始化选优=0.10524780；相对改善1.8509%（负数为更差）。
预先冻结固定对照中，测试表现最低的是development_fixed，RMSE=0.10075424；随机变异相对改善-2.5264%。此比较不反馈选图。
随机变异在6个区域×遮挡案例中优于不变异：4/6。只有三个空间相依区域，不能把种子或像元当成大量独立案例。
定向检查47项通过。扩展回归36项通过、2项失败；两项都是旧栅格上传测试写死8个输出，而当前和本轮修改前的冻结源码均返回14个，已独立复现，未改无关测试凑全绿。
主实验71份已封存产品独立重放，36次正式轨迹核查；粗源扰动补充重放18份。物理越界失败单列保留，没有丢弃失败后平均。
峰值、空间纹理和时间变化误差见image_diagnostics.csv；可见观测保护、固定QA支持覆盖率另列。空间差分是纹理代理，不是动力平衡指标。
不变异对照也会对初始化候选真实执行并按验证误差选择；本轮不能单独归因三画像软评分的优势。栅格响应仍存在UNKNOWN，未声称已学会预测所有数据上的算法效果。
页面浏览器交互未在本轮验收；这里验证的是平台公共文件/API入口及共用数值执行链。
2017负结果、float32诊断与未评分的部署边界诊断完整留在本地diagnostic目录；交付包包含其摘要，正式轮完整图、产品、轨迹和输入均在包内。源卫星大文件本地保留，包内包含所需标准化数据与原始来源/哈希记录。
全部开发、验证、失败、部署和复核成本保留；旧中断重放的成本未完整落盘，不能声称诊断成本为零，也不把额外重放算作变异效率优势。

## 70. 互补算法公共接入与协作验收（2026-10-29）

本轮按“增加真正能够配合的算法”补充五个公共节点：bias.xsdba_scaling、spatial.denoise_tv、spatial.denoise_bilateral、spatial.wavelet_lowpass、fusion.wavelet_detail_transfer。分别采用现有xsdba 0.5.0、scikit-image 0.26.0和PyWavelets 1.10.0真实内核，新增可选依赖及上游许可证记录。TV与双边为同一角色的可替代方法；不同角色的偏差校正、噪声处理、低频背景和细节融合可与已有时间/空间重建节点协作，并非五个节点必须全部使用。

默认算法库及空间、多源公共入口已接入；复用现有三画像、随机变异、科学状态去重、DAG执行和质量门，没有另建整图候选池或人工加分。新增算法未知效果响应仍标UNKNOWN。真实校正需要历史训练配对，且历史/应用前处理状态一致；未预设只有一种处理顺序。修复xsdba栅格输出维度顺序，并修复双边滤波后端绝对数值范围导致的单位一致性偏差。初版失败日志和产品保留。

最终定向与共用模块回归60项通过。测试验证公共图生成机制确能连接新增方法、公共随机变异入口可以执行、生成图能不带验证真值部署。另用2013—2014历史训练和2015真实开发输入完成16×16×10的连接验收，应用阶段11次注册调用，五个新增节点全部真实执行，可见观测保持。保存配置独立重放并核对产品及台账，未使用2018留出真值。

固定验收图仅证明接口可协作，未输入为优化器指定路线；自动生成测试证明可达性，不证明搜索优势。未跑正式效果对照，scientific_advantage=NOT_EVALUATED；不能据此声称已优于第69章最强固定方案。页面点击交互未单独验收。第69章冻结源码、数据、产品及结论保留，本轮库扩充不改写其历史实验结果。

详细交付：docs/互补算法接入与协作验收_20261029.md；数值、调用台账、画像、独立核对及测试证据：output/complementary_library_20261029/。

## 71. 五类互补算法公共接入及顺序验收（2026-10-29）

接入temporal.whittaker、reconstruction.dineof、fusion.guided_filter、spatial.arosics_coreg、fusion.starfm五个公共节点。实际后端分别为VAM Whittaker 2.0.6官方编译内核、sinkr 0.8.0固定提交原始R函数、OpenCV contrib 5.0.0.93、AROSICS 1.13.2及固定提交starfm4py原始predict内核。源代码、发行包、哈希与许可证保留；DINEOF并未冒称使用审计过但未执行的Fortran发行版。

复用默认算法库、公共文件/API入口、既有三画像、GraphMutation随机变异、科学状态去重、真实执行和质量门。没有另建整图模板池、替换评分器或人工奖励新算法。未知收益等画像响应继续UNKNOWN。

验收已覆盖Whittaker前后两种STARFM顺序、两个父节点的引导/目标和参考/移动角色、逐源不同处理、同日可见锚点及边缘补缺。配准须显式允许，公共constraints.forbid_spatial_shift可否决；DINEOF不恢复整期完全无观测的场，Whittaker需规则时间轴，STARFM目前仅接0—1地表反射率及明确传感器不确定度。没有把NDVI或降水改名冒充反射率。投影栅格补缺不再误按经纬度处理。当前公共栅格变异/部署仍沿用既有time,lat,lon工作坐标契约，未宣称任意x/y文件无需转换均可直接搜索。

AROSICS/GDAL独立Python 3.11环境已安装执行；Whittaker的NumPy1 ABI在专用子进程包路径中加载。平台主NumPy 2.5.3未降级。ABI诊断和初始失败记录保留，错误副本移除。

回归及边界验收合计75项不重复检查通过。完整公共注册表上的变异器实际生成含五类节点的图，后续补边后经公共无标签入口完成7次注册调用，并由保存输入、配置和图独立重放逐值一致。测试人员选取生成编辑以覆盖各节点，只证明可达/可执行，不代表生产优化器被强迫使用全部五类，也不证明搜索优势或全局最优。

复用2015真实开发NDVI，完成DINEOF、时间插值、Whittaker与逐源处理后的引导融合及范围投影，输出46×16×16，7次注册调用；独立重放一致。未读取2018留出真值。AROSICS与STARFM另经受控影像和自动生成全图验收。未跑正式精度对照，scientific_advantage=NOT_EVALUATED；第69章固定优于变异的历史结论不因库扩充而改写。页面点击流程本轮未单独验收。

详细说明：docs/五类互补算法接入验收_20261029.md。完整证据：output/complementary_five_20261029/，包括画像、测试XML、真实开发产品与台账、自动生成图、独立核对及运行环境记录。

## 72. 算法库更新后同数据随机搜索复测（2026-10-29）

复用第69章相同流域、三个48×48区域、交错/连续中段隐藏两种方式及原时间划分。当前源码共用下，以原冻结库108个注册ID与完整更新库118个注册ID对照，只有库成员差异；注册ID含不可用状态，不能当作实际可用科学方法数量。新库不设人工整图池。用户明确要求平台自行决策并使用随机搜索；主实验实际optimizer.backend=random，evolution_used=True，未使用ALNS或人工替换选图。固定及不变异仅作对照。

两库各随机/不变异、三个相同种子11/29/47，共72次搜索完成。统一数值调用保护上限1800，为原实验十倍；提议400000、画像探测600、服务保护3600秒。本轮优先数值效果，不宣称效率优势。2016选图，所有2018产品封存后统一评价；2018已在此前查看，因此明确属于回顾复测而非新盲测。并行调整前的中断运行另归档，不丢失诊断记录，也不计入正式均值。

结果不能证明更新后具有稳定整体优势。旧库随机变异18/18完成，隐藏日期等权RMSE=0.10441861；新库随机变异仅16/18完成，不删除失败后计算总体均值。种子齐全的5个案例中，新库相对旧库2个改善、3个变差，另1案例不完整。新旧库不变异RMSE均为0.10524780。原开发期最强固定组合仍为0.10075424，不能声称新库随机搜索整体优于该方案。

两次新库随机部署失败均在patch2_contiguous。验证输入细源6期、每像元最低3个观测；测试输入3期、每像元最低2个观测，选出的DINEOF节点还固定max_modes=5，不满足新输入适用条件。现有执行器正确拒绝，但冻结图部署尚无“拒绝后自动改选合法图”的闭环。本轮未人工更换算法、放宽DINEOF条件或依测试误差重搜。另一失败为原始锚点残差固定对照越界，独立保留。

新增方法真实进入搜索：引导滤波、小波细节融合、DINEOF、双边/TV去噪、小波低频及Whittaker均有真实调用。STARFM不适用于当前NDVI契约；AROSICS要求至少64×64且显式允许配准，当前48×48不满足；均值偏差校正缺历史训练配对，未为入场编造条件。更新后随机选图16/18重复同类操作，单图同一操作最多28节点；重复平滑不能被解释成28种互补能力。局部改善不能单独归因多能力协作或三画像评分。

独立检查72次候选库、统一额度、随机后端及图注入边界；105份已部署产品复算主指标并核对可见观测保护，源码与原输入哈希通过。完整协议、源码、图、逐图数值执行、调用账本、失败和指标位于output/library_update_retest_20261029/。详细交付：docs/算法库更新同数据复测_20261029.md。未修改上传关系识别、任务自动推导或生产选图逻辑；本轮仅开展库更新效果检验。


## 73. 当前输入自主重选与部署闭环修复（2026-10-29）

第72章为2016选图后冻结迁移到2018的回顾对照，其两次DINEOF部署失败不能作为每批输入自主处理已完成的证据。本轮在公共入口新增current_input策略：当前可见数据重新遮挡评价、既有三画像筛选和random随机变异，随后在完整当前输入真实部署；不接受预设整图，不调用ALNS。页面正常无外部标签模式与文件工具共用该入口，数据/目标变化使旧页面结果失效。

目标时间、网格、完整性检查前移到候选数值评价，避免只覆盖已有日期的低误差图成为最终结果。部署重新核对适用性、范围及物理质量，保护观测；失败按内部验证原排序自动改选，全部失败明确报告NO_EXECUTABLE_CURRENT_GRAPH。调用与预留部署时间共用总账本，不放宽科学算法条件。严格冻结迁移接口仍保留用于指定图部署及对照，并明确拒绝current_input策略。

搜索/独立确认遮挡互斥并均从搜索输入及同源副本移除；自然缺失不作真值，确认误差在原值保护前计算，不参与选图。原6期验证输入及原失败的3期测试输入种子11、29三组全部重新搜索并真实部署通过，产品分别11×48×48、6×48×48、6×48×48，可见观测保持、全部有限、决策不读取外部真值。独立保存输入/图重放产品逐值一致，确认RMSE独立复算一致。

86项回归及2项追加边界验收合计88项通过。页面薄客户端调用已测试，实际页面点击未单独验收。本轮针对已声明来源与目标的连续栅格重建，不宣称任意上传关系识别已补齐；不足观测或资源/质量不满足仍会明确失败。未跑正式精度对照，scientific_advantage=NOT_EVALUATED，不改写第72章负结果或宣称全局最优。详细交付：docs/当前输入自主重选与部署验收_20261029.md；完整轨迹、失败诊断、XML、独立审计与冻结源码：output/current_input_repair_20261029/。


## 74. 修复后同数据自主随机变异效果复测（2026-10-29）

复用第69/72章黄山小流域Landsat约30m和MODIS约250m/8天NDVI：三个48×48区域、交错/连续中段隐藏约半数日期，6案例×随机/不变异×种子11/29/47，共36次自动选择；另保留六种原固定对照的36次部署。公共文件入口调用当前输入自主选择，随机组实际backend=random，未使用ALNS、预设整图池或人工替换胜出方法。每个计分结果来自真实执行，缓存与科学状态等价复用不记新增数值调用。生产算法、评分权重及数值内核本轮未改。

当前可见部分划分互斥搜索/确认遮挡，均从选择根输入移除，确认不参与选图；隐藏日期测试真值未传入决策。全部预测封存后统一读取测试参考。原2013—2014质量支持和2015开发固定方案保留；新版按2018当前可见数据重选，旧版2016选图后迁移，因此前后结果不是单一代码因素的因果比较，且2018已被查看，属于回顾复测。

原资源上限为1800注册调用、400000提议、600探测、3600秒。长搜索暴露收尾问题：3次在部署前耗尽服务时间，并非DINEOF新输入适用性失败。测试误差未打开时冻结统一追加部署协议：所有36次自动运行均按公共GraphMutation.rank和deploy_ranked恢复原实测排序，每次追加10000调用/3600秒，只部署、不新增搜索/候选/训练/权重/参数，不人工指定图；固定部署已有同等额度。全部追加成本计入，3项原预算失败单独保留，不能将补充后的完整率冒充原预算内完整率，资源收尾缺陷本轮未修生产代码。

统一补充部署后，随机18/18、不变异18/18均有真实产品；固定35/36通过，原始锚点残差越界1项保留。隐藏日期等权、再按案例/种子等权RMSE：随机0.10680521，不变异0.09950147，原2015开发最强固定0.10075424。随机相对不变异误差高7.3403%，相对固定高6.0057%，两组比较均仅2/6案例更低。不变异平均比固定低1.2434%，仅2/6案例改善；不支持普遍优势。随机31712次总注册调用，不变异594次，已包含原尝试和追加部署，不支持效率优势。

旧同扩充库随机仅16/18完整，不补填其总体均值；完整匹配5案例中新版2改善、3变差，另外1案例旧版不完整。负结果及全部种子保留。既有栅格DAGResponse仍缺完整实测三路响应、保持UNKNOWN；本轮不能解释成三画像收益机制全部获证或全局最优。目标是NDVI重建，不代表SWAT+/WRF获益。

独立核对36次决策、71份产品的主指标/固定支持/原观测保护、原输入和源码/历史对照哈希。6案例种子11保存图用相同单线程数值环境重放逐值一致；初次未统一线程的1.9984e-15差异诊断保留，未改变容差或评分。决策和补充部署均不读取外部参考。详细交付：docs/修复后同数据自主重选效果复测_20261029.md；冻结协议、统一部署修订、原失败、全部轨迹、封存产品、指标与审计：output/current_input_effect_retest_20261029/。


## 75. 公共入口提前停止、重复候选优化与超时自动收尾（2026-10-29）

保留平台GraphMutation随机变异及契约允许的完整注册节点，新增候选检查复用、不可变拓扑缓存、共享子图重建和惰性随机笛卡尔积，避免反复检查或先展开海量组合。同分排序改为数值舍入桶、节点数和稳定图ID，实测耗时不再决定选图；公共服务统一数值线程。没有换GOLEM/ALNS/贝叶斯优化或新增手写图池，没有改科学插值/融合内核。

只用2015开发输入：12条真实搜索轨迹及每案例4种可见遮挡敏感度，校准发布visible_stagnation_2015_v1。相对实质改善阈值2.9180%，连续86次新数值评估无累计实质改善停止；连续5次迭代无新数值评估亦停止。37种候选等待策略中12种满足全部开发轨迹的实测敏感度容差，选最省评估者；仅是NDVI开发校准的工程尺度，不是置信界或全局最优保证。测试真值未用于校准。

公共服务初始化后先在完整当前输入上保存真实合格checkpoint，再继续随机搜索；终止后自行执行最终候选，按原平台验证排序复核，最终失败则自动返回原实际合格checkpoint。搜索与部署额度独立但合并记账，部署包含初始保存及最终执行；不再需要实验客户端补跑。页面与API共用入口、默认设置和显式种子，页面数据/目标/预算/种子/策略变化使旧结果失效。通用应用契约可直接复用，不强制无关下游模型字段。

昨日6案例×3种子的随机/不变异共36次主运行，加18次随机重复，54/54均由一次公共服务调用自行输出PASS。随机18次均按收益停滞停止；服务耗时中位数23.63秒、最长329.92秒；总注册调用4720、总服务计时1071.58秒。相对第74章累计计时下降95.28%、调用下降85.12%，属于并发/策略均改变的整体回顾结果，不做单组件速度因果归因。原3个超时种子本次约20、19、12秒自动完成。

96项定向/回归检查通过，包括搜索时间耗尽仍自动真实部署、最终执行失败回退合格checkpoint。18对重复的图、停止原因和新评估次数一致；15对数值逐值相同，3对最大差异2.220e-15，均在运行前记录的float64舍入界内，坐标精确一致。逐位检查诊断与QA修订保留；未修改输出数值、质量门或测试评分。Streamlit真实表单提交与API严格数值一致（文件上传传输用原始字节替身，后台执行未mock），并验证初始BLAS线程数8的普通入口结果一致；未做浏览器文件对话框点击。

全部主预测和重复先封存，再读取隐藏日期真值。71份主产品独立复算指标和观测保护，决策阶段外部真值读取零，原输入/冻结源码核对通过。RMSE：随机0.10875856、不变异0.09950147、原开发固定0.10075424；随机精度仍不占优，负结果保留。原始锚点残差1项越界仍明确失败，固定35/36通过，不删除或放宽门槛。2018已被查看，仍为回顾性验收；不代表下游模型获益、普遍精度优势或全局最优。

详细交付：docs/公共入口提前停止与自动收尾验收_20261029.md；开发校准、源码前后、96项测试、页面实测、54次公共服务轨迹/逐图真实调用/产品、复现、封存指标和审计：output/autonomous_search_closure_20261029/。借鉴OptunaHub停滞停止与Makarova等2022自动终止思想，未照搬其高斯过程理论保证。


## 76. 新区域独立时空融合与纯平台验收（2026-10-29）

独立下载爱荷华州41.74N、91.82W农田窗口2021/2022年Landsat8 C2 L2 T1约30m与MOD09A1.061 Terra500m/8日反射率及QA，356个原生窗口全部取得，保留URL、元数据和哈希。未复用昨天黄山观测，不声称该矩形已核实为流域。按官方QA及NDVI定义转换，仅做原尺度附近重投影，粗输入未预先升至高分辨率或融合。2021年15个合格日期用于开发链路检查；2022年7个日期分别隐藏3个，交替与连续缺失共享原始数据，不当独立区域。平台内部可见遮挡选图，外部隐藏标签仅在所有输出封存后评分；未训练新模型或调当前评分/停止规则。

公共入口保留随机变异、默认预算及既有2015-NDVI停止条件，未改生产源码、候选池、算法参数或深度。开发10次、正式18次、页面/API2次，共30次全部自动完成并通过完整目标/范围/原值保护检查；没有脚本补部署、外部真值或人工参考图进入主决策。8对随机与不变异的起始图、初始误差、自验证与确认数组逐项一致。页面为真实Streamlit控件/提交和真实后台，上传传输用原字节夹具；页面/API及正式种子11结果一致，未验收原生文件对话框。

预封存人工参考是粗时序背景加可见高分空间残差，再经既有范围约束。正式平均RMSE：随机0.11656917，不变异0.12507065，人工参考0.10499146，高分单源时间插值0.12983925，粗源时空插值0.12881846。随机相对不变异误差降低6.80%、相对高分单源降低10.22%，但比人工参考高11.03%；6次配对中仅3次优于不变异。人工组合相对高分单源降低19.14%，说明本批有组合价值，不能声称搜索已稳定选优。交替缺失随机均值0.07544939，连续缺失0.15768895，后者劣于不变异0.14470235。全部负结果保留，无结果后重调或换种子。

6张随机最终图均实际包含两个源，算法节点3—9个，含可见锚点、时间/空间恢复、导向滤波、小波及TV等真实注册算子；连接/顺序/参数由平台生成，参考图只作对照。随机正式单次约72—277秒，均因验证收益不足停止，未穷举、不宣称全局最优或下游SWAT+/WRF收益。详细：docs/新区域独立融合平台验收_20261029.md；协议、冻结版本、原始出处、30次调用、逐候选轨迹、配对审计、全部产品及独立评分：output/iowa_independent_fusion_20261029/；新原生数据：data/iowa_independent_fusion_20261029/。


## 77. 公共入口整日期验证与缓存短组合漏评修复（2026-10-29启动，10-30完成）

修改公共current_input/field_mutation及版本/页面缓存标识。目标日期缺失时自动用整日期或观测日期块作自验证及独立确认；同源副本同步遮挡，少于4有效可见日期明确阻止伪验证，不再点遮挡冒充。缓存已执行祖先共用终点契约/物理门/真实误差独立参选，零新增管线调用，单列前缀评价与耗时，不计新停滞执行。没有硬塞锚点人工路线，也没更换随机机制、算法内核、评分权重或既有停止预算。

103项独立测试通过，包括真实公共入口栅格补缺/TV长图的短前缀独立评价；开发实际又有7/19条缓存短图评价。2021开发10次通过后，以新2023年Landsat/MODIS数据作测试：10高分日期隐藏5，粗500m8日，共184个新原生窗口，3种子、分散/连续掩码；全体30次调用通过。8对初始图/误差/自验证及确认一致；所有成功已缓存祖先均有终点评价。共609条前缀评价，6个正式随机结果中0个来自缓存短图。全部输出封存后才评分，无外部真值参与决策、无实验脚本补部署，页面/API复现通过。

2023平均RMSE：随机0.12618816，不变异0.14081900，人工参考0.11872362，高分单源0.14087004，粗源0.12675143。随机相对不变异收益+10.39%，相对人工参考-6.29%；正数=降误差、负数=变差。逐种子及负结果全部保留，无结果后调参。2023与2022年份不同，不能据均值直接声称旧11%差距已消除。连续缺5日期但内部只支持遮2日期，support_limited已披露；原2015停滞容差未针对新验证重新标定，三画像栅格实测收益仍UNKNOWN。机制验收不等于稳定选优、普遍泛化或全局最优。

详细：docs/整日期验证与短组合修复验收_20261030.md；修改前源、测试/失败保留、冻结协议、2021/2023公共入口轨迹、产品、审计、独立误差与成本：output/task_validation_prefix_fix_20261029/；新原始数据：data/iowa_fusion_2023/。


## 第78章 可靠选优公共入口与新区域实验（2026-10-30）

# 可靠选优公共入口实施与实验验收

2026-10-30。全部数值结果由封存产品回读，未根据测试结果修改规则。

## 实施与机制验收

- 公共入口已接通整日期留一多折：各折独立输入/执行器；同源别名同步遮挡，独立粗源保留；所有必需折失败即不可选，禁止成功子集评分。
- 稀疏输入每折保留其余可见帧；不再同时额外剔除确认帧。留一只是内部交叉验证，不声称独立确认。
- 另从公开注册的2020/2021开发数据按传感器、物理定义、可见/隐藏索引及最小日历距离构造整图历史模拟。历史主评分与当前局部诊断分列，不任意混合。
- 三画像实测损失：数据确定任务、验证支持与归一化；应用明确RMSE/MAE/绝对Bias目标；算法/完整图提供真实逐折响应。未知模型敏感性不造数；原始算法库、随机变异、执行质量门保留。
- 画像响应与主评价共用执行结果；共同参照增益只作诊断，不能重复给精度加分。开发冻结参照不注入搜索候选，实际不能运行则记UNKNOWN。
- 全局近等价集合按有效节点数及确定性成本选简图；支持最终合法删节点实测。正容差须有开发依据，本轮留组校准未显示收益，发布delta=0，未人为降低门槛。因此不能宣称本轮证明了正容差简约机制的效果。
- 页面与API用相同服务；政策/历史注册内容进入页面请求指纹。真实Streamlit控件提交与API同种子图和产品一致，上传字节注入是唯一UI测试夹具，后端未模拟。
- 106项相关测试全部通过。初次校准夹具误把相同真值文件标成两个独立组，被防泄漏检查正确拒绝；改成不同年份、增加可区分候选后通过。旧失败XML保留。

## 数据与隔离

新增524个Landsat8/MOD09A1原生窗口；沿用QA、物理缩放及NDVI定义。精细约30m、低频；粗约500m、8日。输入时保留不同网格与时间，不在实验端预设融合路线。

Iowa2020（13合格日期）及2021（15）用于开发；Iowa2024（8日期、保留4隐藏4）用于同图库诊断；Kansas2024（7日期、保留4隐藏3）为未见区域终检。分散与连续掩码共享原始观测，三种子不是独立地区，属于小规模探索。

开发结束后冻结源码、政策、病例注册及全部输入SHA。公共调用阻止读取外部真值；候选产品全部封存后才计算隐藏值误差。没有手工补部署、人工路线注入或替换失败种子。固定方案仅为对照。

## 同一35图的选图检验

35图来自337个既有开发结构的节点数分层与固定随机抽样，另含原封存对照；未按本轮测试效果挑图。它只是诊断图库，不限制生产搜索空间。

新规则平均遗憾 0.00508174，旧规则 0.00597703，下降 14.98%；最大遗憾均为 0.01016347。平均RMSE 0.10254128 对 0.10343658，改善 0.87%。只是一个地区年份两种缺口的小幅证据，不能称全局最优或稳定泛化。

## 未见Kansas完整对照

36次自主搜索（6组×2掩码×3种子），8次固定对照均输出合格产品。最终44个科学实验产品封存后统一计分。

|组|平均日期RMSE↓|平均科学节点|真实算子调用总数|
|---|---:|---:|---:|
|development_strong_fixed|0.13031477|固定图|10|
|fold_only|0.05789672|5.5|2936|
|full|0.05996978|2.6666666666666665|3973|
|legacy_single|0.06398925|4.666666666666667|1980|
|no_parsimony|0.06738482|5.833333333333333|4729|
|no_portraits|0.05672608|3.0|3722|
|none|0.09823164|1.0|774|
|predeclared_fixed_coarse_only|0.09017601|固定图|4|
|predeclared_fixed_fine_time_linear|0.17287436|固定图|2|
|predeclared_fixed_human_anchor_range|0.06014156|固定图|4|

full相对旧规则平均改善6.28%；配对胜出5/6。相对不变异改善38.95%。负值表示变差。

## 解释与证据限制

- full=新整图实测与随机变异；legacy_single=旧单块验证；fold_only=局部多折、不用历史病例；no_portraits=关闭画像软目标与实测锦标赛、仍保留相同历史支持及质量门；no_parsimony=关闭简约；none=相同新评分初始化后不变异。不同消融同时改变可解释的提议/排序，不能把所有差异都归于单一权重。
- 当前应用仅明确NDVI重建RMSE，三画像归一化本身不改变同任务RMSE排序，收益主要须来自验证支持和实测探索；没有证明下游WRF/SWAT实际收益。
- 历史日期为最近可得匹配，非严格同分布；Iowa诊断连续任务2020匹配最大偏差44天，2021为8天。报告保留间隔/锚点、空间尺度及QA支持，不能宣称解决了所有真实长缺口。
- 开发共同强固定图在某些当前折不满足输入支撑，因此收益参照标UNKNOWN，未造分或偷偷换参照。
- 本轮未校准出正简约容差；旧2015停滞停止政策仍保留，不能当新多折任务的统计保证。
- 没有独立历史确认窗口；最终Kansas是独立于开发选择的新区域检验，样本量不支持普遍可靠声明。若终检负结果，不回头用同一测试调参。

## 可复核产物

EXPERIMENT_PLAN.md、graph_bank_manifest.json、source_manifest.json、published_policy_manifest.json、tests_delivery.xml、page_acceptance/acceptance.json、final_prescoring_seal.json、final_metrics.json、final_summary.json、paired_comparison.json，以及每次公共调用的request、fold输入/标签、attempts、完整图、response_scoring、total_cost和product。

机制通过与效果通过分别判断；效果表不支持的创新不宣称完成。



本轮最终判定：机制验收通过，稳定选优创新未验收。新机制比旧规则平均误差下降6.28%，但算子调用3973对1980，成本约翻倍；一个配对退化10.40%。相对本次测试最强预设固定（人工参照）仅改善0.29%，不能当显著优势；关闭画像软引导的消融误差更低（完整机制比其高5.72%），局部多折消融也更好。不能据此证明历史画像转移或实测锦标赛提升选择质量。简约功能支持有依据的正容差与实测剪枝，但本轮delta=0，节点下降不能归因于学习到的正容差。所有数据已经用于本轮报告，后续修改不能把它们继续称作未见测试。


## 第79章 锦标赛停止与历史适用性修复（2026-10-30）

# 锦标赛停止配合与历史适用性修复验收

2026-10-30。复用平台公共入口、随机变异、算法库、三画像整图评分、执行器、缓存、质量门与账本，没有新增科学处理算法，没有实验端指定生产路线或人工补部署。

## 实际改动

1. 随机变异的停滞单位改为有实际新计算的变异回合。一次最多3张锦标赛图只贡献一个停滞单位；缓存/空提议不贡献数值回合，原无新评价停止仍保留。独立数值评价数另记。锦标赛全部真实调用仍收费。
2. 无软引导也消耗相同策略抽签随机数，减少一个无关的随机流差异；后续搜索路径仍会因候选数量与选择不同而不同，不能称完全相同提议序列。
3. 账本增加搜索、各折、探测、失败、剪枝、checkpoint和部署共用的真实调用硬上限。默认总数1800，而不是搜索1800再另加部署1800；缓存不计新调用。预算耗尽后只允许返回先前真实保存的合格checkpoint，不能另跑补救。
4. 历史病例适用性增加可见信息门：日期距离≤半个目标中位间隔、间隔畸变≤一个目标中位间隔；空间步长比0.8—1.25；可见精细锚点及独立粗源中央80%值域交叠；历史有限覆盖不少于当前一半；粗源有日期支撑且季节变化不反向。没有足够支持为UNKNOWN，明确不符为INAPPLICABLE，不进入主历史评分；没有合适历史则回到当前局部多折，并报告长缺口支持未知。

这些阈值是预先冻结的工程支持门，不是通过测试成绩学习的“收益预测器”，也不证明相似历史一定可迁移。三画像依旧只有明确的重建目标，不臆造下游模型偏好。

## 机制与数据验收

120项相关测试通过；最终元数据修正后又复核26项。覆盖1/3次探测不提前耗尽回合耐心、真实总成本包含失败与部署、日期/网格/分布/季节不匹配拒绝、隐藏历史精细真值改变不影响适用性、真实公共多折执行及同种子复现。

开发只用已见2020/2021，4次调用均合格。相同开发任务中，full与无软引导各到86个数值变异回合时才按停滞收尾，前者154次候选数值评价，后者86次，未再因3张一组而提前三倍耗尽耐心；额外计算完整记账。另一个开发任务按无新数值评价规则收尾，不能声称所有搜索都运行同样多回合。

新数据固定Nebraska2024（7个QA合格精细日期，保留4隐藏3）与Minnesota2024（11日期，保留6隐藏5），各分散/连续整日期掩码。376个原生公开窗口全部获取成功；复用原QA、缩放及NDVI计算，精细约30m、粗约500m/8日，输入时仍保持不同网格与时间。

正式36次自主调用（full/none/no_portraits，各4任务×3种子）及16次冻结固定对照，共52次。页面实际表单和API重复另外验收；仅上传运输为夹具，后端未模拟。每张计分候选完成全部必需折，所有最终产品封存后才读隐藏参考。

源码、政策、历史资源和输入SHA冻结；正式执行期间禁止读外部真值；所有失败、预算停止和负结果保留。固定/不变异组不人为花满额度，实际成本照列。固定图仅部署其预先选定处理，不进行搜索；不能以它与搜索的调用量比较宣称搜索更省。

## 独立新区域结果

RMSE为每日期误差后按任务/种子等权汇总；两掩码共享原始观测，种子不是新地区，不做普遍显著性声明。

|组|平均RMSE↓|平均节点数|真实调用总数|
|---|---:|---:|---:|
|development_strong_fixed|0.13682165|固定图|20|
|full|0.17229242|4.583333333333333|12170|
|no_portraits|0.14961853|6.416666666666667|7681|
|none|0.17864943|1.0|948|
|predeclared_fixed_coarse_only|0.17555993|固定图|8|
|predeclared_fixed_fine_time_linear|0.16221498|固定图|4|
|predeclared_fixed_human_anchor_range|0.12589225|固定图|8|

full相对none平均误差改善3.56%，配对胜出7/12；最差配对改善-102.42%（负数表示退化）。

full相对no_portraits平均误差改善-15.15%，配对胜出3/12；最差配对改善-102.42%（负数表示退化）。

逐任务均值：

|任务|full|none|no_portraits|
|---|---:|---:|---:|
|new_minnesota2024_alternating|0.10973083|0.11066846|0.11051854|
|new_minnesota2024_contiguous|0.26885120|0.28531728|0.23009808|
|new_nebraska2024_alternating|0.12357692|0.10706081|0.09966188|
|new_nebraska2024_contiguous|0.18701073|0.21155117|0.15819563|

## 历史病例在新任务上的自动判定

- new_nebraska2024_alternating：UNKNOWN；development_iowa2020: INAPPLICABLE (calendar_assignment, sampling_intervals)；development_iowa2021: INAPPLICABLE (calendar_assignment)。
- new_nebraska2024_contiguous：UNKNOWN；development_iowa2020: INAPPLICABLE (calendar_assignment, sampling_intervals)；development_iowa2021: INAPPLICABLE (calendar_assignment)。
- new_minnesota2024_alternating：UNKNOWN；development_iowa2020: INAPPLICABLE (calendar_assignment, sampling_intervals)；development_iowa2021: INAPPLICABLE (calendar_assignment, sampling_intervals)。
- new_minnesota2024_contiguous：UNKNOWN；development_iowa2020: INAPPLICABLE (calendar_assignment, sampling_intervals)；development_iowa2021: INAPPLICABLE (calendar_assignment, sampling_intervals)。

正式自主调用最大真实成本1800/1800，checkpoint收尾2次。费用包含不成功的算子及额外探测，不把它们排除后再说公平。

## 结论边界

- 机制修复是否正确，与full是否胜过所有固定/消融是两件事；消融更好也原样报告。
- 没有旧版在这两处新数据上的同预算独立对照，不能将与上轮不同地区的均值差当成本次修复因果收益。
- 不适用历史被拒绝是支持门生效，不等于已解决跨域迁移；回到局部多折时，仍不能宣称验证代表所有长缺口。
- 早停耐心86及相对收益约2.918%的旧数值保留为工程设置；本轮修复计数单位并严格计费，没有宣称这两个参数已被新任务统计率定。
- 本轮没有校准出新的正简约容差，没有新模型训练或添加新的融合算法。科学收益以实测结果为限，不保证全局最优。
- 若预算只能返回早期checkpoint，它仍是合格且真实执行过的结果，但可能不同于最后验证最优图；须按实际交付产品评分。

## 文件

EXPERIMENT_PLAN.md、source_manifest.json、input_manifest.json、mechanism_acceptance.json、tests_verified.xml、tests_metadata_verified.xml、all_outputs_sealed_before_scoring.json、final_metrics.json、final_summary.json、comparison.json及search_runs/fixed_runs/page_acceptance。每次调用保留折输入、真实图、数值响应、失败、完整成本与产品。


最终验收判定：机制通过，软引导效果验收未通过。完整机制RMSE为0.17229242，不变异0.17864943，无软引导0.14961853，最好的预设固定对照0.12589225。完整机制比不变异平均改善3.56%，但比无软引导高15.15%，比该固定对照高36.86%。实际算子调用12170对7681，完整机制多58.44%。所有12配对保留：相对不变异赢7/12，相对无软引导赢3/12；预算检查点两例按实际交付产品计分。测试结果出来后未修改规则，未将默认路线偷偷改成消融组。后续可以定位局部验证与真实长缺口失配、锦标赛的额外计算收益及最终部署费用，但不能再把这两处数据当未见测试。


## 第80章 任务匹配验证公共入口与新地区实测（2026-10-30）

# 任务匹配验证公共入口与新地区实测验收

2026-10-30。用户没有给定实际用途的数值精度门槛，本轮报告实测效果；不编造达标阈值，不宣称全局最优。

结论：机制及统一入口通过验收，36次自主搜索都输出合格产品；固定对照10次成功、2次失败。完整组只有连续缺期优于本轮固定对照最低误差，另外三类均未超过固定；随机变异的新增收益很小，三画像软引导未显示稳定优势。应将本轮交付理解为验证机制和实测证据交付，不能写成算法效果全面领先。

## 已实现的机制

平台公共入口保留随机变异和既有三画像逻辑，新增任务验证编排：

1. 时间缺口按真实日期、隐藏跨度、左右观测间隔和季节选择连续块；不能拿单日期留一冒充已匹配长缺口。匹配容差不超过一个目标采样间隔，且不超过跨度的25%，这是明确的工程支持范围，尚非统计率定阈值。
2. 空间、时空缺口从输入缺失的连通块提取几何，在有原始观测的位置迁移同形遮挡；时空块同时检查实际时间偏移。每种类型取最大连通块作为代表，最多3个互不重叠的验证遮挡；不宣称所有较小形状、距离边界和土地类型都已独立验证。
3. 多源隔离：同步遮掉同源精确副本，保留真正独立源。声明为同源派生或网格变换后的文件，若不能从遮挡后的原始数据重放，明确返回证据不足，不对已平滑结果只遮同位置就算隔离。先前拟合资源必须声明独立训练时间。未声明的隐蔽血缘不能自动保证识别。
4. 尺度细化复用现有球面面积重叠算子，构造降低分辨率的代理验证；交付后做原观测尺度的面积回投一致性。二者都不等于未知细尺度真值。降尺度、回投各自计入实际成本。
5. 匹配不到的长缺口仅返回局部诊断及证据不足，不放宽历史病例适用性。证据状态、应用精度状态和停止原因分开；合格文件、预算停止或误差停滞均不被写成实际用途精度达标。

核心代码为src/ecodata/algorithms/task_validation.py，调用在field_mutation.py，隔离复用current_input.py，页面search_page.py显示验证选择和证据范围；页面与实验调用platform_search.py同一入口。算法数值内核、历史政策、停止参数、随机变异后端未为本轮成绩调整。

## 机制证据

- tests_final.xml：101项回归通过；tests_insufficient.xml补1项低支撑边界；交付回归再补匿名血缘边界和2项失败日志。不同测试合计105项，重复运行不相加，最终定向以tests_delivery.xml为准。
- 测试覆盖真实公共图执行、各必需折完整执行、别名同步遮挡、独立源不变、派生泄露阻断、长缺口证据不足、尺度代理和回投。
- 四类新输入的mechanism_acceptance.json记录了实际自动选择的协议；时间任务还自动识别并加入原始MODIS中的自然空间缺口。
- 页面验收使用真实Streamlit表单及提交事件，仅注入上传字节传输。新时空块数据的页面与直接API，同种子处理图、停止原因和数值产品完全相同；未模拟搜索后端，未手工部署补跑。
- 固定图失败成本日志问题在读取测试误差前发现并记录计划；48次原实验封存后才补日志。另补“声明派生关系却缺少主源ID”时的证据不足门，防止匿名血缘绕过检查；本轮输入均有源ID，不进入新增边界。两项日志测试最初因测试默认GBK读取UTF8失败，显式UTF8后通过。source_executed保留实际实验源码，delivery_manifest.json列出交付时platform_search.py、task_validation.py的日志与缺元数据边界修正；未改数值内核、评分、当前有ID任务的验证构造或重跑选图。

## 数据与实验边界

先固定威斯康星州44.50N、89.50W及2025年，再因MODIS目录只覆盖到6月、同期仅一幅完整Landsat，不足以做预定时空试验，改用同地区未见2024年。保留2025年124个原始窗口及不足记录；更换依据是数据覆盖，发生在任何算法效果评价之前。

2024年176个原始数据窗口下载完整、无失败，每个已保存文件与接收记录SHA256一致。复用既有官方产品QA、红/NIR比例因子和NDVI计算：Landsat8 C2L2 T1约30m，MOD09A1.061约500m、8日合成。共同季节内选取所有局地24×24完全有效的10幅细影像，粗影像27期9×8；这是按QA选可遮挡真值，不按算法成绩筛选。MODIS时间戳只代表合成区间起点，不假称与Landsat精确同时。

本轮应用契约只要求NDVI物理范围和有效覆盖；未运行SWAT+、WRF等下游模型，不能据此宣布下游模拟精度改善。未接入自动重放任意外部派生流水线，也未证明所有独立源共同失效模式均被模拟；这些情况不能从“同源隔离测试通过”推导为已解决。

空间测试遮一幅4×4块（16个真值）；时空测试遮两期4×4（32个）；时间测试遮连续3期（216个有效真值，隐藏跨度16日、锚点间隔32日）。原始自然缺失始终不当真值。尺度测试将30m观测做明确面积平均得到12×12输入，再重建24×24，内部另在6×6→12×12评分；外部5760个原生细像元只在封存后评价。这是嵌套退化代理，不是已证明真实500m→30m的超分能力；其低分辨率观测允许含细观测的面积均值，不能说它与细真值统计独立。

学习和停止政策继续冻结在既有开发证据上，2024年不参与校准。对当前可见值构造内部验证折，外部测试标签不传平台，36次搜索和12次固定调用结束并封存产品后才算误差。完整、不变异、无软引导各3种子11/29/47；固定对照在选图前冻结，仅供比较，不作为搜索候选模板。各搜索同一1800实际算子调用、600探测调用及3600秒上限，保留平台提前停止及自动检查点收尾。

对照含义：full保留三画像软引导和随机变异；none保留同样初始化而不继续变异；no_portraits保留随机变异、相同验证与硬约束，关闭画像软引导；固定组由平台真实执行预先写明的图，不搜索。

这四种构造来自同一个地区年份，互相不独立；种子重复也不是独立地理样本。空间、时空遮挡规模较小，是本轮小规模验收，不能证明对半幅云覆盖普遍有效。没有同一输入的旧版配对试验，不能把本轮与上轮不同地区的均值差说成修复带来的因果提升。

## 封存后的实际误差

各日期仅对原本有限且被隐藏的真值计算RMSE，再对有真值日期等权平均；没有真值的日期不作为零误差或NaN混入均值。尺度代理单列，不与缺失恢复混成一个总分。

|任务|完整机制RMSE|不变异RMSE|无软引导RMSE|固定对照最低RMSE（事后）|
|---|---:|---:|---:|---:|
|单期空间块|0.026872|0.026872|0.024443|0.018878|
|连续两期空间块|0.020093|0.020093|0.019565|0.017264|
|连续3期缺失|0.081703|0.081736|0.078798|0.086016|
|尺度重建代理|0.041228|0.041616|0.041453|0.040268|

- 单期空间块：完整机制相对不变异的误差改善+0.00%，相对无软引导-9.94%，相对固定最低-42.35%。负数表示更差。
- 连续两期空间块：完整机制相对不变异的误差改善+0.00%，相对无软引导-2.70%，相对固定最低-16.38%。负数表示更差。
- 连续3期缺失：完整机制相对不变异的误差改善+0.04%，相对无软引导-3.69%，相对固定最低+5.01%。负数表示更差。
- 尺度重建代理：完整机制相对不变异的误差改善+0.93%，相对无软引导+0.54%，相对固定最低-2.38%。负数表示更差。

完成并可数值评价的产品：46/48。未产出2例，所有失败和负结果保留；预算检查点收尾6例，均按实际交付产品评分。完整机制与固定组的最低值是事后比较；“固定最低”不能冒充部署前已经知道的最优算法。

两条固定负结果分别是连续缺期先空间补缺（整期无可用格点）、未裁剪的三次尺度插值（越过NDVI物理范围）。不调换顺序、不追加裁剪来替换原结果。旧固定失败路径未落盘的两份原始成本凭证明确为未知，不按0计；日志修复由独立测试验证，不倒填原实验时间或费用。旧客户端在失败回执缺字段时触发的总封存断言，已用仅读取48份原回执的封存步骤处理，没有新增科学计算。

实际选中路线也已记录：空间与时空块的完整组都交付temporal.raster_linear；时间缺口组主要交付temporal.whittaker；尺度组从最近邻出发，部分种子叠加TV平滑/三次重网格。多源输入不意味着强制融合，平台确实可以选择单源路线，但这种自主选择并未在本轮各任务普遍胜过固定融合。连续缺期对固定最低约5%的改善主要已存在于初始化，不能将它全部记作变异贡献；变异相对不变异只有约0.04%，尺度代理约0.93%，两种块恢复为0。

## 真实计算成本

|任务|完整机制平均调用|不变异平均调用|无软引导平均调用|
|---|---:|---:|---:|
|单期空间块|722.7|124.0|789.3|
|连续两期空间块|722.7|124.0|719.0|
|连续3期缺失|1591.0|186.0|1504.3|
|尺度重建代理|321.3|46.0|87.3|

|任务|完整机制平均分钟|不变异平均分钟|无软引导平均分钟|
|---|---:|---:|---:|
|单期空间块|2.48|0.32|3.48|
|连续两期空间块|12.39|0.25|4.32|
|连续3期缺失|13.26|0.77|13.97|
|尺度重建代理|2.62|0.15|0.40|

以上按每次实际请求平均，包含内部验证各折、三画像额外数值探测、失败调用、初始检查点、部署和尺度审计，缓存遍历不重复算调用。对照共享上限而非强制耗满。固定算法仅执行预定图，单独保留实际成本。数据下载与共同预处理、页面/API重复验收是工程验收开销，另存原始记录，不算成搜索效果优势。

墙钟时间是在4个进程并行条件下记录，并非严格单请求独占机器的部署延迟。实际适配调用次数不等于等FLOP；不能只比较次数就宣称效率占优。部分请求的提议/去重/评分等编排开销明显，已按真实总耗时保留，不能称运行效率已经全面解决。

## 最终结论的界限

任务验证的自动切换、多源隔离、尺度代理及页面统一入口已完成机制验收。算法选得是否更好必须看上表各任务，不将机制通过写成效果全面胜出；不因负结果切换默认路线，不用测试误差回改权重或掩膜。本轮没有用户精度门槛，停止于预先冻结的小规模实测与证据交付，不宣称实际用途精度已达标。

新增编排的明确适用范围是time、lat、lon连续栅格。纯静态2D、未来预测的单侧回溯、多源共同故障的任意同步重放、未知传感器PSF和未声明血缘，均不能据本轮验收推断已经解决。没有匹配参考时的代理或局部证据，会与精度保证区分。

## 可复核文件

output/task_matched_validation_20261030包含EXPERIMENT_PLAN.json、prepared.json、fixed_controls_frozen.json、source_manifest.json、input_manifest.json、mechanism_acceptance.json、tests_final.xml、tests_lastcheck.xml、pre_scoring_seal.json、final_metrics.json、final_summary.json、scientific_acceptance.json、page_acceptance及全部search_runs/fixed_runs。各搜索保留实际折输入、参考隔离、图、数值响应、失败、总成本和交付产品。


## 第81章 验证代表性、独立画像筛选与执行预留复核（2026-10-30）


结论：公共入口的机制修改和端到端复现通过；可靠选优的效果验收未通过，不能宣称问题已经全部解决。48/48次平台请求产出合格产品，36/36次搜索交付最终验证选中的图，0次检查点替代；但半幅空间恢复有严重泛化失败。尺度代理只有小幅收益。负结果全部保留，不用2024年测试标签回改本轮代码、画像、权重或候选池。

## 1. 公共入口实际改动与复用

生产版本：platform_search_v10_representative_prior_reserved_deployment。复用原有platform_search、GraphMutation、Executor、SearchAccount、task_validation、算法注册表/能力描述、portrait_rules.algorithm_properties、模型state_policy和既有可靠选图/停止模块。未换回GOLEM或ALNS，默认仍是平台随机变异。未在实验脚本中生成、选择、挽救生产处理图。

验证：空间/时空块在原有形状和日期跨度匹配上，增加上下左右边界接触类型、以缺口宽度为单位的距离类别和可见外环的中位数、离散度、相邻变化匹配；时段候选增加可见两端分布与独立源是否可用的匹配。描述不读取被遮住的值。只有可支持的候选才构造内部验证折。无匹配样本时保留证据不足，不能把普通遮点当成长缺口证明。尺度仍复用面积退化及回投影审计，明确只属于代理验证。

资源：候选执行前检查“完整各折预计调用+最终图重执行调用+1次诊断”是否装得进共享总额度，部署期间可使用预留。墙钟预留取至少5%服务上限或既有算子中位时间估计的2倍、最多一半上限，是工程保险量，非统计保证。失败、各折、初始检查点、探测、最终部署和尺度审计均入账；缓存不重复记调用。正在执行的单个内核仍可能越过软时间边界，不保证任意环境绝不超时。

筛选：新增candidate_portraits.py只计算独立于验证误差的可见画像先验。取消原来“随机取最多3个、真实验证后择优”的画像分支。现在先在已合法提议中按先验分配一个执行机会，再真实执行和评价；保留约一半随机探索，不排除未知响应的算法。根邻域仍扫描可调用注册库，初始化按同一先验排序，没有人工指定路线或固定最大图深度。

## 2. 三画像评分究竟是什么

先验P(g)=Σ_j w_j×a_j(x)×r_j(g,x)。w_j来自已有模型/应用问题的重要性归一化，并复用有依据的既有校准乘子；a_j是当前可见缺失比例、网格不匹配或已定义问题风险。算法响应依据真实注册能力和已审阅的传递性质，不从名字猜性能。

缺失恢复方向响应使用support/(1+variation×distance)：support为该方向具有可见支持的缺失比例；variation为可见相邻绝对变化/可见标准差，时间变化按实际日期间隔修正；distance为空间或时间最近可见支持的平均距离。需要网格转换时只看空间方向。已注册投影/非负操作可提供对应操作属性，同类动作只计一次，不因重复堆节点获得额外先验收益。此公式是可见平滑性与支持距离的工程启发，不是学习得到的误差预测、严格误差界或最终物理保证。观测保护后的最终状态仍须实际质量检查。

未知项保留[0,w_j×a_j]区间及依据；先按下界、未知项数、节点数排序，同分随机。每条优先级记录data、algorithms、model_terms、score_interval和validation_loss_used=false。未知的融合收益、峰值保持、动力敏感性未被填成经验分数。

最终选图仍依据各必须验证折的真实数值结果、质量门及既有简约选图规则；先验负责探索机会，不能压过真实反例。旧response_scoring中由验证误差计算的响应仍作为实测诊断保留，不再冒充本次独立先验。没有用另一个学习排序器替代三画像，也没有在最终误差上再叠加同一份误差变换以假装独立证据。

## 3. 机制验收证据

115项去重后的相关测试最新结果通过，见tests_verified.xml；包含边界匹配、隐藏值不进入邻域描述、共享预留、空提议须覆盖变异类别才允许相关提前停止、公共入口真实搜索和最终选图交付测试。先前失败记录未删除：新测试最初把公开Framewise算法ID写成原始ID，修正测试名称后重验；140次极小预算的诚实返回状态也曾被测试误要求为PASS，保留原记录并按实际状态验收。

公共入口反事实检查：保持数据和种子不变，将缺失/非负问题权重100:1交换为1:100，补缺与非负候选的优先次序确实交换；可见时空变化改变也会改变方向优先级。这里只证明已接入并影响候选决策，不证明假设模型需求合理或预测收益正确。

Streamlit AppTest走真实页面控件和提交事件；仅上传字节传输用夹具。相同输入、seed11和600总调用上限，页面/API选中相同图、相同停止原因、593次实际调用、逐值相同产品，无人工部署。默认公共预算仍1800，页面预算现已覆盖搜索与最终执行的共享总额度。

2020/2021开发输入先作资源可行性检查：591和584次调用，均交付最后选图，未读取开发或最终外部标签计算误差，未重新训练或调整权重。600是本轮小规模试验资源上限，非已优化预算；已有停止政策继续冻结，不假称为新任务已重新充分校准。

## 4. 新数据及冻结协议

新地区Illinois2024，40.3N、89.1W，与前轮Wisconsin不同。152个官方原始数据窗口下载完整，0失败，文件SHA256与接收记录一致。复用既有QA和红/NIR比例因子：Landsat8 C2L2 T1约30m，MOD09A1.061约500m、8日合成，NDVI无量纲、范围[-1,1]。MODIS日期只是合成起点，不声称与Landsat精确同日观测。

按固定QA规则保留所有局地24×24完整细影像，共5期（6月14/30日、9月2/18日、10月20日），粗影像27期9×8。半幅空间缺口遮9月2日中间12×24；时空缺口遮9月2/18日同一半幅，按“有两端观测的众数相邻时间间隔、并列取最后”在评价前选定，未按成绩挑时段。空间验证3折、时空1折。长时间任务隐藏粗影像中间连续3期；当前输入还有自然云缺口，任务支持明确为LIMITED_DIAGNOSTIC_ONLY，6折局部诊断并未给32日锚点长缺口提供充分匹配证据。尺度从细影像面积平均到12×12，再重建24×24，内部用6×6→12×12代理，不能等同真实500m→30m超分辨率。

完整(full)、不变异(none)、无软引导(no_portraits)各3种子11/29/47；36搜索请求+12冻结固定请求。各搜索同一数据、质量门、注册候选、种子和600总实际调用/200探测/3600秒上限；无软引导仍随机变异，不变异只有相同初始化。各同种子组初始合法候选集合审计一致，全部根邻域已评估；38/45/51等数量仅是当次根邻域，不是所有可生成组合数。没有为了穷举把图库压成几个模板，也没有做完整穷举或全局最优认证。

固定图在当前测试评价前冻结，通过平台同一真实执行器运行，仅作比较，不塞入搜索初始化。空间固定为线性、三次、可见锚点残差；时间为线性、最近邻、Whittaker λ=1（依据既有开发/历史实测补入已知强基线）；尺度为最近邻、双线性、三次；各自带事先声明的NDVI范围投影。尺度固定三项没有覆盖PCHIP，需结合不变异组所选PCHIP一起解读，不能称胜过库中所有固定算法。

生产源码、输入、政策、控制、协议先冻结；48次全部完成并封存产品后才计算外部测试指标。决策期间审计阻止读取外部标签、原始完整细真值和固定图库。封存文件哈希读取不参与数值选图。自然缺失不当真值，原始已知值逐值保护。未做人工续跑或手动补产品。

## 5. 封存后真实效果

按有隐藏真值日期的像元RMSE，再对日期等权平均；不同任务单列。下表三个搜索列为3种子均值，固定最低是事后乐观比较值，未用于选图。

|任务|完整机制|不变异|无软引导|冻结固定对照最低（事后）|
|---|---:|---:|---:|---:|
|半幅空间缺口|0.174885|0.061454|0.206441|0.021662|
|连续两期半幅缺口|0.024779|0.023632|0.024973|0.023632|
|连续3期时间缺口|0.056640|0.060899|0.057364|0.048020|
|尺度重建代理|0.017681|0.017718|0.017815|0.017909|

- 半幅空间缺口：相对不变异误差改善-184.58%，相对无软引导+15.29%，相对冻结固定最低-707.33%。负数为变差。
- 连续两期半幅缺口：相对不变异误差改善-4.85%，相对无软引导+0.78%，相对冻结固定最低-4.85%。负数为变差。
- 连续3期时间缺口：相对不变异误差改善+6.99%，相对无软引导+1.26%，相对冻结固定最低-17.95%。负数为变差。
- 尺度重建代理：相对不变异误差改善+0.21%，相对无软引导+0.75%，相对冻结固定最低+1.27%。负数为变差。

完整组相对无软引导的四项均值均较低，但同一地区年份、四个相关构造、三个搜索种子不足以证明稳定普适优势。完整组只在尺度代理胜过本轮冻结固定最低；相对不变异所选PCHIP只有约0.21%改善，不能夸大为强组合创新。长缺口较不变异改善约6.99%，仍不如Whittaker固定对照。空间块完整组3种子RMSE为0.06936/0.13335/0.32194，波动和负结果非常明显。

独立回读最差空间产品并直接用float64数组重算RMSE=0.3219412050，与原指标一致。最后选中的键和实际执行键一致，不是错误文件、预算退回或评价日期混用。空间组内部验证误差下降而测试明显变差，已确认有排序泛化失败；有些固定候选已真实评价，却按当前内部验证输给更差的复杂路线。不能把它简单归结为算法没搜到。

边界类别和外环相近仍不能保证缺口隐含结构、精确位置及季节相同：例如空间验证位置含y=1/2，真实缺口y=6，当前宽度距离分箱过粗，三项外环描述也不是已校准适用性检验。时空仅1个匹配折。独立先验目前只有较粗的能力/平滑性/支持距离偏好，未知融合收益未解决；这限制了选优能力。零开发支持容差下仍出现7–9节点多次平滑组合，不同数值状态无法仅靠去重消除。以上是保留轨迹后的问题定位，不依据测试成绩修改本轮规则。

## 6. 真实成本

|任务|完整平均调用/秒|不变异平均调用/秒|无软引导平均调用/秒|
|---|---:|---:|---:|
|半幅空间缺口|593.7 / 37.5|121.0 / 7.9|588.7 / 31.4|
|连续两期半幅缺口|452.3 / 167.4|43.0 / 2.1|594.0 / 93.3|
|连续3期时间缺口|589.7 / 40.6|265.0 / 11.2|582.3 / 32.2|
|尺度重建代理|166.7 / 16.4|46.0 / 2.2|147.0 / 18.8|

调用包含全部实际适配器调用及失败、验证折、探测、检查点、最终执行与尺度诊断，缓存不重复计数。墙钟为共享账本服务时间，在4进程并发环境测得，非独占机器的精确部署延迟；最长完整请求约328秒。共同下载/预处理及页面双跑另存，不算搜索收益，调用次数不等于FLOP。三画像没有在本轮证明全面计算成本优势。

## 7. 哪些已经完成，哪些没有

已完成：新机制进入公共入口；独立可见三画像先验实际改变候选优先级；随机探索保留；各折完整图真实执行；共享额度预留最终部署；页面/API同条件复现；新地区四类对照、测试隔离、48个产品及全部负结果封存。

尚未达标：可靠地为大块/长缺口选出更好组合；充分校准验证情境的代表性；对未知组合收益、峰值/偏差/动力响应的可靠画像；抑制所有非恒等冗余平滑；多地区稳定效果及效率优势。未运行SWAT+/WRF，当前应用画像只是NDVI范围与覆盖要求，不能声称验证了真实下游模型敏感性。尚无用户精度目标，不宣布用途精度达标或全局最优。

本轮停止于冻结协议下的完整验收和负结果交付，没有为追优势循环修改已看过的测试。后续若修正上述代表性/复杂度问题，Illinois2024只能作为诊断/开发案例，必须另留未见数据，不再把本轮当新测试。

## 8. 可复核文件

output/representative_prior_reserve_20261030：before、source_executed、source_manifest.json、input_manifest.json、protocol_manifest.json、EXPERIMENT_PLAN.json、prepared.json、fixed_controls_frozen.json、tests_verified.xml及所有原始测试XML、mechanism_acceptance.json、development_budget_acceptance.json、page_acceptance/acceptance.json、pre_scoring_seal.json、delivery_audit.json、final_metrics.json、final_summary.json、scientific_acceptance.json、negative_diagnosis.json、全部search_runs和fixed_runs。每个搜索保留图、初筛先验、变异、各折隔离输入和内部标签、失败及总成本；运行脚本仅提交请求，事后评价脚本不被平台导入。


## 第82章 六类验证策略与保守选图回路验收（2026-10-30）


本轮目标是第一版能正确、可解释地完成处理，已知严重选错不再被“验证通过”掩盖；不把超过所有固定算法作为交付条件。复用公共入口、现有六类验证、三画像先验、随机变异、真实Executor、质量门和资源预留，没有改数值算法、另建系统或把人工路线塞入生产搜索。

## 修正的具体问题

|策略|本轮处理与边界|
|---|---|
|零散遮点|按当前缺失比例和受影响时段构造；最少两点供搜索/确认分离，没有缺失时的20%仅属诊断。修复部分参考声明传递，保留原值保护和同源遮挡。|
|长时间缺口|保留持续长度、真实日期间隔和两端支持检查；补齐独立源有效支持，季节比较改为中点对中点，年末按循环日历比较。边界外推无证据时仍明确不足。|
|空间块|四边距离精确匹配，同一网格不再把y=6移成y=1/2；增加日历窗口与来源支持匹配。外环没有可测相邻变化时返回未知，不填0伪装平滑。|
|时空块|同时满足空间位置、连续期数、时间跨度、日历和来源条件；所有连通组件均审计。现有折数上限用尽或较小形态无法验证，会标注不足，不再只证明最大块就把整体标成匹配。|
|多源隔离|继续重用同源别名同步遮挡、已声明派生血缘/拟合期检查，叠加到点、块、时间和尺度任务；独立背景不能偷换成测试真值。同源变换无法重放时明确拒绝，未知血缘不能保证自动识别。|
|尺度验证|保留面积退化和回投影，新增与缺口恢复验证组合；不会因选择尺度任务就提前忽略时间/空间缺失。代理证据仍标为代理，不冒充真实细尺度精度。|

同一输入可以同时采用多种策略。新增公共入口测试实际执行“尺度变换+空间缺口+同源别名隔离”的完整图，全部必须折均真实执行，目标尺寸与观测保护仍检查。

## 选图保护与运行回路

输入检查→任务识别→组合验证策略及证据审计→初始化合格候选→冻结保守短图→有限随机变异→通过证据门才能替换→原始输入最终执行→质量检查/收尾。

基线从平台自己已经真实执行合格的初始图中选，优先已有审阅证据支持凸值传递或逐点投影的短图；这是避免无依据外推/过冲的已知操作属性，不是精度预测。没有这种候选会记录baseline_transfer_known=false，不伪造保证。基线保留可获得的真实数值比较，不再因稳定编号丢弃实测证据；验证不足时仅作暂定选择，不宣称该比较代表真实缺口。完整与无软引导共享相同保守门，三画像只调整完整组的候选机会。

后续图只有在任务证据不受限、至少两折、每一折不劣且均值改善超过已有开发/舍入容差时，才能替换这份合格基线。没有用测试成绩新调容差，没有删除随机变异或限制成固定模板。若保护门已确定当前证据不能采用新图，只做一次随机候选诊断后按明确原因停止，不再白跑到86轮停滞；有充分匹配折时仍允许继续变异。真实执行一致改善的合成多折实例也通过，证明保护门不是把所有新图一律封死。

这是保守采用门，不是未知目标误差保证。三画像仍用于候选机会，不假称已经可靠预测未知融合收益。资源预留、提前停止、失败账本、最终质量检查均保留；生产正常入口仍使用随机变异。需要不变异的消融通过同一公共submit_search/PortraitMutation后端显式声明，不修改正常用户入口的随机策略。

## 工程验收与证据完整性

127项去重后最新定向/回归结果通过，tests_verified.xml及tests_provenance.json可逐项追溯原始XML。早期暴露的部分参考传递问题已修；新增真实多折测试最初构造的时段只有1个匹配窗口，随后使用确有两个窗口的合成时段，保持至少两折和实际数值改善断言，未放宽门槛。初始失败记录保留。

独立补充测试发现季节起点/中点不对称后，停止并保存首轮为interrupted_trial1，未算效果、不混入最终批次。旧版源码快照的依赖路径加载失败、一次不变异客户端走了只允许随机的入口，均未进入科学管线；日志/拒绝回执保留，修正运行时路径和对照提交方式后重新提交。没有用脚本补造生产结果或手工部署图。另有四项旧快照图库请求因搬迁后R/Whittaker依赖路径不完整，恢复原有固定运行时后统一重跑；缺依赖结果保留在old_environment_incomplete，不用于新旧方法比较，资源映射和校验见runtime_restoration.json。

最终28项有效请求全部完成：相同图库新旧各4项共8项，正常开放搜索20项（完整组4任务×3种子；不变异及无软引导各4任务×seed11）。不是28种候选算法，搜索内部全部数值候选另外计账。20项当前开放搜索均交付选定图，0次预算检查点替代，观测逐值保护、质量门和共享调用上限通过。页面AppTest真实控件提交与公共API同条件同种子得到相同图、调用数、停止原因和逐值相同产品，仅上传传输使用夹具。

相同图库在评价前冻结为上轮3个固定对照及3个完整组输出图，旧版与本版分别走公共evaluate_graph_bank。每张冻结图的原始输入产品RMSE新旧相同，说明没有暗换数值内核来制造收益。生产开放搜索不读取这个图库。最终v11r2批次统一源码和固定依赖，28项请求及产品全部封存后才读取回归真值；未在最终批次内修改生产算法、评分或权重。决策期间始终阻止读取外部真值，未人工续跑生产搜索或按测试回调参数。

## 已知案例回归结果

Illinois2024已被上一轮评价过，本轮明确只作为故障诊断/回归数据，不是未见最终测试。RMSE按有隐藏真值日期等权平均，自然缺失不当真值。完整组为3种子11/29/47均值，其余新对照只有seed11，不能据此作统计显著性推断。固定控制保留于共同图库；其中尺度固定三项未包含PCHIP，因此图库最低也不等于全库最优。

|已见回归任务|上版完整组均值|本版完整组均值|不变异seed11|无软引导seed11|
|---|---:|---:|---:|---:|
|半幅空间缺口|0.174885|0.021662|0.021662|0.021662|
|连续两期半幅缺口|0.024779|0.023632|0.023632|0.023632|
|连续3期时间缺口|0.056640|0.060899|0.060899|0.060899|
|尺度重建代理|0.017681|0.017718|0.017718|0.017718|

|相同候选图库|旧选图RMSE|新选图RMSE|图库最低RMSE|
|---|---:|---:|---:|
|半幅空间缺口|0.133352|0.021662|0.021662|
|连续两期半幅缺口|0.023780|0.023632|0.023632|
|连续3期时间缺口|0.056314|0.060899|0.048020|
|尺度重建代理|0.017686|0.019115|0.017634|

按1e-10数值核对容差、相对本轮小型图库最低值，严格未选中最低的任务数从4/4变为2/4。这是描述性回归数，不是泛化成功率；微小差距和大幅退化应结合原始RMSE理解，不能仅凭计数夸大。

## 未解决与交付范围

本轮修改和实际验收针对time/lat/lon连续规则栅格，不代表已逐项重验所有站点序列、非结构网格和下游模型。日历窗口和可见邻域匹配仍属保守工程规则，不是训练得到的适用性置信界。较少影像、单折、多形态折数不足、尺度代理等任务可能继续输出基线，不能宣称评分已获得全面优势。中间版共同图库和完整回归曾显示长缺口/尺度退步，未作为交付版；随后只改通用基线采用规则和无意义搜索停止条件，旧负结果保留，未按案例指定算法或调整数值内核。当前流程不保证全局最优、未知缺口误差或真实SWAT+/WRF改善。未声明的隐藏数据依赖、未知传感器PSF及所有时空外推情形也不在本轮保证内。

本轮保留随机变异，但证据不足时即使内部验证更好也不采用。最终四项完整组均返回与不变异相同的基线，不能宣称变异或三画像额外提高了精度；本轮作用是阻止不可靠替换。长缺口均值0.060899仍比上版0.056640高约7.52%，尺度0.017718比上版0.017681高约0.21%，负结果保留，不宣称全任务改善。最终完整请求客户端墙钟约15.8–47.0秒，属于并发环境下的实测，非通用时延保证。耗时按metrics.json逐请求保留。中间版曾因证据不足仍白跑约3分钟，已作为completed_trial2完整保留；最终版按不能采用新图的条件及时收尾，不按调用数推断时间优势。页面与API复现成立，不保证不同机器墙钟截止时仍有完全相同搜索轨迹。

第一版验收按实际已测输入的链路、隔离、保守采用、资源和错误收尾判断；不宣称之后任何数据都绝无错误，也不为追求效果继续在已见数据上循环修评分。

## 文件

output/validation_safety_20261030内保存before源码快照、before_manifest.json、source_executed、source_manifest.json、data_manifest.json、PLAN.json、graph_bank.json、tests_verified.xml及全部原始XML、tests_provenance.json、page_acceptance、seal.json、metrics.json、comparison.json、acceptance.json、28项runs、client_logs及interrupted_trial1、completed_trial2及其中的拒绝提交/不完整旧环境记录。


## 第83章 DEV整期影像缺失的多源融合小检查（2026-10-30）


结论：平台处理链路能运行，多源融合算子也确实可调用并使用两源数值；但本次三种情景没有观察到实际融合增益。平台最终均采用细源自身的时间恢复，没有把多个文件强行合成。相对本轮固定融合参考，这个选择与外部隐藏真值的比较方向一致；不能据此宣称平台总能选最优、所有融合组合无用，或本项目已证明融合创新。

## 为什么选这个任务

现有Honcut土壤pH文件只有4个深度层，DEM是静态高程，二者变量、结构和验证真值不对应，不适合硬拼成这轮空间融合测试。按用户授权改选有现成独立传感器和细尺度真值的DEV Iowa2021窗口：Landsat8约30m细空间观测与Terra MOD09A1.061约500m、8日合成观测，目标是恢复整幅缺失的细影像。整幅全缺不能在该时刻做局部空间补洞，保留细源空间信息并借助粗源时间变化具有明确工程动机；是否真的提高效果必须实测。

复用既有官方产品QA和比例因子处理结果，验证开发资源文件与已发布注册表SHA256一致。取固定中央24×24细网格、与粗源共有时段内全部局地QA完整日期，共12期；粗源26期9×8。两者同为NDVI，单位1、相同物理定义，模型/应用契约只要求覆盖和[-1,1]范围，不冒充SWAT+/WRF敏感性。细网格约25×33m来自既有最近邻地理坐标转换，不声称新增了原生空间细节。粗源保留原始QA缺失，不在脚本中先补好。

细源原始参考包含2021年4—10月数据；共同窗口实际目标为4月17日至10月1日。日期和裁剪按元数据/验证可行性选定，未按恢复误差挑区域。MODIS时间戳只是8日合成起点，不声称与Landsat精确同时观测。该DEV区域年份已被项目使用过，本轮是工程诊断，不是独立未见测试或泛化证明。

## 实验与隔离

主实验两种掩膜：隐藏2021-08-30一整期（576个真值像元），或隐藏08-23/08-30连续两期（1152像元）。主目标网格保持细源24×24，不做简单粗化来凑效果。平台自动识别整期时间缺口：前者2折匹配验证，可继续随机变异；后者只有1折，按现有保护门保留基线、一次候选诊断后停止额外采用。

冻结完整多源组3种子11/29/47、细源单独自动组seed11、多源不变异seed11、多源无软引导seed11；每种掩膜另有细源线性、细源Whittakerλ1、粗源单独、可见锚点残差融合+范围投影4个固定对照。合计20请求。所有自动组只传数据和目标，固定图不传搜索。预算统一300实际算子调用/100探测/15000提议/180秒上限，只是小规模诊断，不代表充分搜索或全局最优。

20请求完成后，因短缺口下粗源没有实测收益，另做一次预先声明的探索性压力补充：按日历选内部前后锚点间隔最大的日期2021-07-06，隐藏整期，细源锚点06-04/08-07相隔64日。重复完整/单源/不变异/无软引导seed11及3个已知适用固定对照，共7请求；不再重复已知不适用的Whittaker，不调算法、评分、权重或参数，结果无论正负到此停止。这是看过首批结果后的探索补充，不能当成事前确认性实验。

共27个处理请求：16个平台正常/消融请求全部交付，9个适用固定对照交付，2个固定Whittaker因输入日期不规则被契约检查拒绝，成本0次实际算子调用但保留检查耗时和错误回执。没有更换失败图冒充成功。当前生产代码保持v11r2不变，与上一轮源码清单逐文件校验，不是本轮又改代码重训或扩大预算。

每条进入数值评价的图真实执行，产品和请求封存后才读取本阶段外部隐藏细影像。原始自然缺失不当真值；已知细源像元逐值保护；内部验证只能使用当前可见细源及独立MODIS。当前运行阻止直接读取外部标签和原始完整2021细源文件，平台历史验证也排除当前目标年份。已有开发权重未重训，仍不能说这些DEV影像在项目历史上从未被使用。

## 实测结果

RMSE无量纲，仅对隐藏的原本有限像元计算，并按隐藏日期等权平均。前两行完整组是3种子均值，64日补充只有seed11；三个情景不独立，不作显著性或泛化声明。

|DEV情景|平台完整组RMSE|细源线性RMSE|粗源单独RMSE|固定双源融合RMSE|
|---|---:|---:|---:|---:|
|缺一整期，锚点相隔16日|0.024114|0.024115|0.099757|0.039757|
|缺连续两期，锚点相隔25日|0.031450|0.031450|0.101418|0.050341|
|探索性：锚点相隔64日|0.093937|0.093937|0.137237|0.113911|

固定融合相对细源线性，前两种情景误差分别增加约64.87%和60.07%；64日补充也未胜过细源线性。没有仅凭固定融合优于较弱的粗源单独方案，就把它算成融合收益。

主实验完整组的结果与单源自动、不变异、无软引导基本相当。单期完整组三种子RMSE约0.02411437—0.02411466，双期均0.03144992。没有证据表明本轮三画像软引导或随机变异带来了有意义的精度提升。主实验完整组约17—67秒，64日补充约17秒，均在预定上限内；并发环境墙钟不等于独占机器部署时延。

## 确认融合真的接通，而非仅有图名

读取每个最终图的可达源节点：主实验完整组6次以及64日补充最终都只依赖source0细源。不是因为把source1写在graphs.json里就称完成融合。搜索日志确实出现了双源结构候选并做了数值评价，但有些结构可能退化为单源/零权重；不能把执行记录数量当作有效独立融合方案数。

另对冻结的固定融合参考做选择后的输入扰动，仅验证数值依赖，不反馈选图、不替换正常产品、不使用外部真值：改变可见细源0.01后，隐藏输出约变化0.01；改变靠近隐藏日期的粗源观测0.02后，单期隐藏输出最大变化约0.01469、双期约0.01340。两源确实影响该融合参考的输出。4次诊断请求、8次实际算子调用单独计账。这个证明适用于固定参考，不冒充平台自动采用了它，也不证明其精度更好。

融合参考内部真实把粗网格插值到细网格并传递可见细源残差，不是只做网格对齐后评分。平台最终选中的单源图没有执行该跨源尺度转换，不能说最终推荐证明了500m→30m融合优势或真实超分辨率能力。

## 本轮发现与限制

这份数据有时空互补的工程动机，但在三个实测窗口中，现有固定融合参考没有贡献优于细源时间插值的有效信息，因此不适合作为融合优势的正面案例。可能涉及空间代表性、传感器响应、合成时刻及时间变化差异；本轮未做足以区分这些原因的因果诊断，不能武断归因为某一项，也没有证明所有融合方法都不行。

单期少数选图仍出现同网格重复三次插值。相对直接线性方案，seed11像元最大改动约9.79e-5，RMSE差约2e-7；这类小差异不算真实融合或有意义的组合创新，作为简约选图/数值同一性问题记录，未在本轮为了结果再改阈值或算法。两期和64日补充按证据门正常保守收尾，未让额外搜索强行替换基线。

本轮可确认数据输入、自动验证分流、候选执行、质量检查和产品输出可运行，适用融合算子能真实使用两源。不能确认多源融合增益、评分/变异稳定优势、下游模型改善或用途精度达标。没有用户精度阈值，不能把PASS解释成恢复误差必然满足业务要求。实际观测到的全部负结果及两个不适用固定对照均保留；不继续寻找更有利日期凑正结果。

## 复核位置

output/dev_fullframe_fusion_20261030：PLAN.json、freeze.json、fixed_controls.json、cases、20项runs及logs、seal.json、metrics.json、comparison.json、acceptance.json、candidate_source_audit.json、SOURCE_AUDIT_PLAN.json、source_response_audit.json、source_probes，以及long_gap_extension下的探索计划、冻结清单、7项runs、seal和metrics。生产源码和本轮数据/协议均有哈希，原始产品不修改。


# 84. Humaya新数据平台能力检查（2026-10-30）


本轮实际使用用户D:/codex/数据下最新Humaya流域包。30项预先声明的处理请求和1项同输入同种子重复请求均已真实执行；全部产品重新打开，检查坐标、完整性、非负、单位及原始有效观测保护。具体状态见表。本轮只测试降水预处理，未运行原生SWAT+，没有模型径流改善结论。

## 数据与隔离

气象压缩包MD5重新验证为30afa5e86e13a760f79ecf5fe4a786fa，与交付清单一致。6个站每日序列均为8036条，1980-01-01至2001-12-31；无负值或非有限值，逐值核对与包内6站CSV一致。站点真实经纬度和海拔保留，不把6站拼成虚构规则网格。降水按现有日降水适配器使用mm/day与accumulation，逐日区间边界显式保存。此处是日量文件已有表达，不是新增分辨率降尺度。

1980—1997年仅拟合每站历史月均值和总体标准差，复用已有月统计计算方式，未训练另一套排序器。输入1998—2001年；每年固定隐藏1月15—17日、7月15—28日，各17天，共68天。各年设置相同缺口长度，使搜索年份也有3/14天缺口。位置只按日历冻结，不按误差、降雨量挑选。外部评价只使用2001年每站17个隐藏有限值，共102个点；可见2001年观测仍可作为历史恢复的上下文，不是未来预测实验。

全部请求完成并封存产品哈希后才读取外部标签；每个工作进程禁止打开external_labels.nc。其他年份被隐藏值未参与评分、拟合或外部测试统计。没有把自然未知缺失当真值。各站和天数相互相关，本轮不作统计显著性或跨流域泛化声明。

Daymet对照使用包内MFBC脚本的原始输入Datos_Daymet_6_puntos_exactos_con_las_estaciones.csv，不使用站点订正后的输出。R脚本先读取该文件再计算订正因子，已核查。Daymet是站点参与构建的网格产品，不能当作独立观测真值；其原生像元空间支持保留cell_mean，站点保留point。单像元提取坐标相同不等于空间支持相同。本轮沿用交付文件8036行的日索引，未重新验证原生Daymet下载到导出的闰日处理，因此这个分支只作接口/约束诊断，不作独立融合精度证明。

## 平台入口与对照

正常组只提交数据、缺口恢复目标、明确单位和SWAT+降水准备契约，调用process_current_input；不提供手工图、不指定胜出算法。保留随机变异。无变异组走公共submit_search的none消融；固定线性、历史月均值用公共execute_frozen_graph，固定图只用于对照，不进入平台搜索。每站另测原始Daymet背景输入。所有自动组同种子11、同预算300实际算子调用（含探测/失败/部署）、100探测、15000提议、180秒上限，和此前小检查一致，未按隐藏结果加预算。固定图无需搜索但真实执行，成本单列。

复用已注册robit-era5-multisource-swatplus-v1的完整、非负日降水准备要求；名字沿用已有版本，不代表学习到了Humaya的SWAT水文敏感性。未修改生产源码、算法内核、评分权重、候选库或停止条件。原有开发停止参数在此任务上calibration_scope_match为false，故不能把其停止阈值解释为已针对Humaya校准的收益界。

## 实测结果

RMSE和MAE为各站外部隐藏点误差，各站等权平均，单位mm/day。成本包含实际调用及失败调用；秒数为客户端各请求耗时之和，非并行批处理墙钟时间。只有固定线性、固定月均值两种固定参照，不能称为整个算法库的最强固定方案。

|方案|成功请求|平均站点RMSE|平均站点MAE|实际调用总数|请求耗时合计秒|
|---|---:|---:|---:|---:|---:|
|平台＋原始Daymet|6/6|9.717624|6.992092|961|76.92|
|固定线性|6/6|9.614369|6.986536|6|28.98|
|固定历史月均值|6/6|7.322441|5.051511|6|31.68|
|平台不变异|6/6|9.717624|6.992092|156|43.29|
|平台随机变异|6/6|9.717624|6.992092|374|53.20|

|站|随机变异|不变异|固定线性|固定月均值|加入Daymet|
|---|---:|---:|---:|---:|---:|
|1|18.760072|18.760072|18.760072|12.928393|18.760072|
|2|5.372649|5.372649|5.372649|4.842039|5.372649|
|3|10.457196|10.457196|9.837661|7.224407|10.457196|
|4|8.073428|8.073428|8.073428|5.637151|8.073428|
|5|5.293476|5.293476|5.293476|3.664425|5.293476|
|6|10.348926|10.348926|10.348926|9.638228|10.348926|

最终选图（选图早于读取外部真值）：

- 站1／daymet_support：reconstruction.daily_linear, source；来源['source0']；新增变异评价43。
- 站1／random：reconstruction.daily_linear, source；来源['source0']；新增变异评价5。
- 站2／daymet_support：reconstruction.daily_linear, source；来源['source0']；新增变异评价52。
- 站2／random：reconstruction.daily_linear, source；来源['source0']；新增变异评价5。
- 站3／daymet_support：reconstruction.daily_ffill, source；来源['source0']；新增变异评价55。
- 站3／random：reconstruction.daily_ffill, source；来源['source0']；新增变异评价10。
- 站4／daymet_support：reconstruction.daily_linear, source；来源['source0']；新增变异评价73。
- 站4／random：reconstruction.daily_linear, source；来源['source0']；新增变异评价16。
- 站5／daymet_support：reconstruction.daily_linear, source；来源['source0']；新增变异评价82。
- 站5／random：reconstruction.daily_linear, source；来源['source0']；新增变异评价19。
- 站6／daymet_support：reconstruction.daily_linear, source；来源['source0']；新增变异评价41。
- 站6／random：reconstruction.daily_linear, source；来源['source0']；新增变异评价14。

完整误差、Bias、雨日误差、调用数和状态保留于metrics.json；不只展示胜出方案。相同输入、同种子重复请求的产品使用xarray逐值核对相等，验证结果见acceptance.json。单个重复不能证明所有平台任务均完全确定。

## 明确发现的限制

1. 日序列分支仍绕过已补充的六类栅格验证。field_mutation.submit以lat×lon是否大于1分流，1×1进入visible_validation/RecoveryAssessment。它取1998—2000年、seed1701内部遮挡，不执行独立确认；新栅格多折、代表性证据门没有在这个分支等价生效。不能把前轮栅格验收泛化到日降水。
2. 本次内部遮挡集中在2、3、5月，没有7月雨季。站1内部126天只有2个正降水日，无法代表测试中的雨季14日缺口。这不是外部测试值泄漏，而是验证样本代表性不足。长度相同不足以保证降雨季节/事件特征相同。
3. 协议称mean_calendar_window_RMSE，评价声明也写RMSE，但实际调用autonomous_field.measured且未设置objective_scope。默认是逐帧空间RMSE的均值；对于单站单像元等于MAE，不是整段RMSE。这是目标标注与实际计算不一致，不能把搜索目标值直接当作整段RMSE。外部表格的RMSE独立按sqrt(mean(error²))重算。
4. Daymet与站点空间支持不一致，简单回退/加权融合不能直接合法串联。相关候选错误保留在attempts.jsonl；最终如果只含source0，就只是单源恢复，不称真实融合成功。平台尚未在本轮自动补齐有依据的点—像元观测支持转换，不能为了融合把cell_mean改写成point。
5. 输出PASS只是结构/数值/物理检查通过。平台自身task_acceptance仍报EVIDENCE_INSUFFICIENT，未有用户精度阈值，也未执行原生SWAT+。本轮可确认链路和产品可用性检查运行，不能确认日降水可靠选优机制完整、用途精度达标或变异/三画像稳定优势。

本轮到此冻结；不利用2001年结果修改规则再把同批数据称为新测试。按本轮测试范围保留发现，未修改生产代码。

## 复核文件

output/humaya_platform_20261030包含PLAN.json、data_checks.json、source_manifest.json、cases、31项runs、logs、seal_before_labels.json、metrics.json、comparison.json、validation_audit.json、acceptance.json及可复核准备/提交/评价脚本。每个请求的公共入口请求、候选轨迹、产品和成本独立留存。


# 85. 日降水任务匹配验证与统一RMSE修复验收（2026-10-30）


最终生产版本：platform_search_v12r1_shared_daily_validation_rmse。已在平台公共模块修复日序列漏分支、RMSE口径、不同入口时间参数与初始化选图四处问题。129项机制/回归测试通过；追加的14项日序列复核也通过（属于129项中的同一组，不相加冒充143项）。最终30项对照请求和1项同种子重复请求全部交付，所有产品回读合格。对6个正常请求的18个内部验证折，用公共冻结执行器重放最终图，再用独立NumPy公式重算误差，全部与平台评分一致。结果是有限范围内的机制验收，不是全平台无错误的保证。

## 改了哪些公共代码

- field_mutation.submit：取消按lat×lon把日序列分到旧visible_validation的自动决策分支，单站与栅格统一进入task_validation.plan。实验、页面与文件公共入口均验收。旧visible_validation辅助函数仍为兼容保留，不再用于默认自动决策。
- task_validation._temporal：复用原来的缺口时长、月日距离、可见锚点和背景可用性匹配，允许真实1×1站点上下文；单站额外要求遮挡日期数等于实际缺口长度。将各实际缺口的匹配块装入3个共享折，同一折内保留所有块锚点，避免遮挡相邻块后改变验证缺口长度；同一个块可以支持多个相似目标，但不重复计分，也不把折称为独立重复样本。只枚举时长容差内的右锚点，避免多年份日数据的无效平方枚举。
- current_input.raster_validation_folds：完整单站序列使用日期折，不再套用至少两个空间像元的遮点逻辑。单点、长缺口和边界缺口都经过同一任务规划器；无匹配证据时保留LIMITED_DIAGNOSTIC_ONLY，不能冒充长缺口已验证。
- autonomous_field.measured：默认统一RMSE=sqrt(mean(error²))；frame_RMSE仅作诊断。明确声明的calendar_year_RMSE兼容目标仍单独标注。field_mutation与dag_response移除对RMSE的逐帧均值覆盖，实测反馈、共同参考和最终评价使用同一真实指标。多折默认取逐折RMSE等权均值；归一化只用可见输入尺度，协议明确记录。
- field_mutation公共入口统一生成目标时间参数，避免自动入口与none入口实际可调用算子不同。最短合格初始图之间按实测损失排序；原有稳定传递白名单仅记录知识状态，不再排除误差更低、已通过硬检查的短图。后续变异替换仍须至少2个匹配折、各折不退化、平均改善，未放宽质量或替换门槛。
- platform_search.VERSION更新，旧请求缓存身份与新版区分。总计6个生产文件变更，未另建执行系统、未增加新科学算法、未针对本批测试值改权重或阈值。完整差异见production_patch.diff。

## 验收与首轮撤回

新增14项日序列测试包含：冬季3日与雨季14日同时匹配；所有块锚点实际保留；完整/单点/长缺口/边界分流；同源副本遮挡、派生源与重叠训练状态拦截；RMSE与MAE排序相反的数值反例；随机/none/页面/toolkit共同入口；短图基线不能因为白名单忽略更低RMSE。最后一个测试也验证随机与none的初始图相同。

首次运行pytest默认临时目录遇到既有Windows权限问题，改用新的隔离临时目录运行，没有改变测试断言。首轮v12实验的输出完整，但审计发现none入口缺时间参数；另用合成反例复现初始化跳过低误差短图。失败测试control_repro.log、baseline_repro.log及全部首轮结果保留于output/daily_validation_fix_20261030，并以WITHDRAWAL.json撤回其公平对照用途。不是把这轮结果删掉后只留有利数值。两处修复依据是入口同条件与评分一致性，不是依据2001年误差选择规则。

首轮逐折回放的诊断客户端曾漏传服务扩展后的时间参数，第13个回放失败。保留失败日志和回执；改为复用平台实际request.json后18折回放通过。最终验收从一开始就复用服务扩展请求，另做18折回放。诊断回放不参与选图，不混入搜索优势。

## 数据与对照冻结

复用原Humaya6站1980—2001年日降水。所有原始准备文件与上一轮逐字节一致；不修改数据、遮挡日期、训练统计或随机种子。1980—1997年仅拟合月均值与标准差；1998—2001年为平台当前输入，每年隐藏1月15—17日和7月15—28日；外部评价只读取2001年隐藏的每站17点，共102点。平台可利用当前可见的2001年数据产生内部验证，这是历史重建的当前输入自监督，不是整个2001年都不接触的未来预测测试。

外部标签文件不传平台，每个工作进程禁止读取external_labels.nc；全部30项请求完成、产品封存后才评价外部隐藏值。内部每折标签只交评价器，原始缺口不当真值，同源副本同步遮挡，派生输入和拟合资源沿用公共来源隔离检查。真实2001年结果在上一轮已被查看，因此本轮只能称同批修复回归；没有依据这些值调参，也不能称新独立盲测。

随机、none和Daymet背景组每站各1项，同种子11，同预算300实际调用、100探测、15000提议、180秒；每条进入评价的候选真实执行。保留固定线性与固定历史月均值。另重复站1的正常随机请求一次。所有正常/none对的初始图和3折输入SHA256已核对相同，不只是预算相同。未穷举、未声明全局最优、未手工把月均值或组合图塞进搜索；固定图仅供对照。

## 实测结果

单位mm/day；各站隐藏点RMSE后各站等权平均。MAE单列，不与RMSE混用。固定参照只有线性和历史月均值两种，不称整个库最强固定方案。调用数包含探测、失败调用和部署；耗时为客户端单项相加，受缓存和机器状态影响，不据此与旧轮宣称算法提速。

|方案|成功请求|平均站点RMSE|平均站点MAE|实际调用总数|请求耗时合计秒|
|---|---:|---:|---:|---:|---:|
|平台＋原始Daymet|6/6|7.259294|4.741194|1370|25.56|
|固定线性|6/6|9.614369|6.986536|6|7.00|
|固定历史月均值|6/6|7.322441|5.051511|6|7.08|
|同条件不变异|6/6|7.322441|5.051511|264|13.87|
|平台随机变异|6/6|7.259294|4.741194|738|18.73|

|站|随机变异|不变异|固定线性|固定月均值|加入Daymet|
|---|---:|---:|---:|---:|---:|
|1|12.917602|12.928393|18.760072|12.928393|12.917602|
|2|4.756041|4.842039|5.372649|4.842039|4.756041|
|3|7.181113|7.224407|9.837661|7.224407|7.181113|
|4|5.459345|5.637151|8.073428|5.637151|5.459345|
|5|3.599586|3.664425|5.293476|3.664425|3.599586|
|6|9.642079|9.638228|10.348926|9.638228|9.642079|

旧版自动RMSE=9.717624，最终自动=7.259294，同批下降25.30%。相对本轮固定月均值变化为0.86%下降，相对公平不变异对照为0.86%下降；负数表示变差。逐站差异和所有负结果保留，不能从几个相关站和一个主随机种子推出稳定优势。此轮未做无软引导消融，不能单独归因于三画像，也不是独立证明算法创新。

最终随机图的实际拓扑顺序（未提前提供给平台）：

- 站1：reconstruction.daily_nearest {'max_gap_days': 3} → reconstruction.daily_month_mean {'max_gap_days': 14}；来源['source0']。
- 站2：reconstruction.daily_nearest {'max_gap_days': 3} → reconstruction.daily_month_mean {'max_gap_days': 14}；来源['source0']。
- 站3：reconstruction.daily_ffill {'max_gap_days': 3} → reconstruction.daily_month_mean {}；来源['source0']。
- 站4：reconstruction.daily_nearest {'max_gap_days': 3} → reconstruction.daily_month_mean {'max_gap_days': 14}；来源['source0']。
- 站5：reconstruction.daily_nearest {'max_gap_days': 3} → reconstruction.daily_month_mean {'max_gap_days': 14}；来源['source0']。
- 站6：reconstruction.daily_nearest {'max_gap_days': 3} → reconstruction.daily_month_mean {'max_gap_days': 14}；来源['source0']。

所有正常组与原始Daymet背景组最终来源列表均可查selected_sequences/metrics；只有source0时是站点自身恢复，不称多源融合成功。三表使用现有注册SWAT+降水准备契约，真实模拟器敏感性仍UNKNOWN。未训练新的排序器，未用另一套排序替换三表。

## 成本与保留证据

最终正常对照成本见上表；最终18折重放另需36次实际调用，只记审计，不作为搜索结果或增益。

|其他计算（不计入最终搜索对照）|请求数|实际调用|失败请求|
|---|---:|---:|---:|
|撤回的v12首轮31请求|31|2347|0|
|首轮诊断客户端缺时间参数的回放|13|24|1|
|首轮使用完整请求的18折回放|18|33|0|
|最终同种子重复请求|1|96|0|

## 尚未解决或尚不能证明

- 匹配时长、季节和可见锚点仍不能知道缺口内部是否有未观测暴雨。证据只支持内部可比条件，不能保证未知事件恢复准确。没有独立确认窗口或用户精度阈值，平台不会宣称用途精度达标。
- Daymet是站点参与构建的产品，并非独立卫星真值；point站点与cell_mean网格的支持转换本轮未补齐。硬约束仍拒绝不相容的直接混合，没有为了融合改写空间语义；本轮不证明真实跨源融合收益。
- 变异是否稳定优于不变异、三画像软引导是否提供独立收益，仍需未使用数据和多种子验证。即使本轮某个均值变好，也不能替代这些证据。前一轮负结果仍有效，未撤回为错误结果；本轮仅纠正机制与对照入口。
- 未运行原生SWAT+，未做径流率定或下游模型精度验证；也没有增加原生空间分辨率。本轮结果是降水恢复产品及平台公共处理链路验收。

结论范围：本次日降水漏分支、指标覆盖和入口不一致已在公共代码修复，并完成机制及实际数据回归；不能据此承诺平台所有任务永不出错或总能选到最优。

## 可复核位置

最终output/daily_validation_v12r1_20261030：PLAN、输入及源码哈希、生产diff、31项runs、日志、全部候选轨迹和产品、seal_before_labels、metrics、comparison、effects、selected_sequences、acceptance、metric_replay_audit及18折重放、side_costs。机制证据与撤回首轮位于output/daily_validation_fix_20261030。原用户数据文件夹与前轮结果未覆盖。


# 86. 站点与Daymet网格融合适配验收（2026-10-31）


结论：这份Humaya数据的站点—Daymet对应、历史映射、组合选择、真实执行及产品输出已经接入平台公共入口。84项正式请求和1项复现均成功，18个正常图验证折的独立数值重放、6站各2项输入扰动均通过。正常点网组18项请求全部实际使用站点和Daymet，不是只读了两个文件。相对单源，本批平均RMSE下降34.96%；但未胜过最好的固定月比例方案，随机变异几乎无额外收益，5号站融合反而更差。只证明限定输入条件下功能可运行及本批条件性效果，不证明独立传感器融合优势、全局最优或所有数据类型自动打通。

生产版本：platform_search_v13_station_grid_prior_fusion。6个生产文件变更，其中2个新文件；复用现有组件，未另建系统或替换搜索后端。源码清单及241文件执行快照见source_manifest.json/source_executed，精确差异见production_patch.diff，修改前文件已按上一版SHA256恢复核验留存。

## 具体补齐的环节

1. **原始Daymet降水CSV读取**：公共read_structure/recognise_file识别官方单像元文件头，自动提取查询经纬度、变量、mm/day、year/yday、产品引用与来源身份。依据官方365日历转换到明确日标签；闰年2月29日保留，未提供的12月31日保留NaN，不补0、不挪动后续日期。保留原始文件头和校验和。
2. **对应关系**：从实际point与grid_estimate/cell_mean支持、查询位置、CRS、同物理量/单位、日区间、历史来源身份与几何确定可用配对，不按文件次序或站号指定算法。官方已提取像元只绑定其明确查询位置，不把查询坐标假装成原生像元中心。规则二维格网复用xarray nearest/linear取样，禁止空间外推；真实案例用官方提取像元，二维格网取样在合成定向测试中验证。背景多余日期按站点日期精确选取，不插值日累计量。
3. **网格到站点估计**：新增StationGridFusion适配器，调用现有fit_affine_source、PriorSeriesCalibration与SeriesMissingFallback。它用独立历史站点/网格配对拟合点估计，再用当前Daymet估计站点缺口；已有站点值优先保留。原始背景仍为网格估计，只有经过明确预测模型的输出才声明为点估计，不能冒充新增观测。
4. **平台选组合**：3种可真实运行的方法注册到platform_registry，与已有单源恢复、前后处理共同供默认随机变异入口选择。正常请求只提交数据、目标和授权拟合，未传手工图。适配器图上为两输入节点，内部四阶段明确记录；可与现有日恢复节点串联。固定图只用于对照。默认仍保留用户要求的随机变异。
5. **信息隔离与约束**：公共make_context保留完整历史资源及元数据，重复历史角色不再静默覆盖；状态缓存身份包括历史资源。新适配要求历史结束早于应用期、点/网格身份与几何一致、训练与应用日区间可核验。未知单位、未知支持、不匹配位置、重叠训练、缺少历史配对均拒绝。普通同支持的xsdba校正增加已知point/grid支持冲突检查，不能替代必要的支持估计。保留原有质量门、验证和部署资源保护。

## 数学含义与原始语义

模型均仅从历史有限配对估计。仿射：点估计=max(0,a×网格值+b)，a/b用既有最小二乘求取；整体比例：r=历史站点雨量和/历史网格雨量和，点估计=r×当前网格值；月比例按月份分别计算r。分母为零、月资料不足等不可识别情形拒绝，不填假系数。最后复用where：有当前主序列值就保留，否则用点估计补缺。先做的时间恢复结果也属于估计，并不因为进入主分支就变成真实观测。

这是历史配对的经验点预测，不是从面积均值精确反演点观测；未宣称保持区域雨量总量或获得真实细空间分辨率。没有可核验的不确定度时不虚构逆方差权重，代表性误差和参数预测不确定性标UNKNOWN。

原先实验人工给Daymet标过cell_mean；官方说明的是1km格网插值估计，没有据此建立精确面积均值定义。本次原生读取保守标为grid_estimate，不改成point，也不假定精确面积观测算子。适配器同时支持用户有依据声明的真实cell_mean输入。此处纠正元数据过强假设，不是为绕过相容检查而改标签。[Daymet原生提取说明](https://daymet.ornl.gov/single-pixel-tool-guide)、[产品说明](https://daac.ornl.gov/DAYMET/guides/Daymet_Daily_V4.html)。

## 先机制验收，再冻结正式运行

214项不同测试通过，包括29项新增点网检查；重复执行的测试按名称去重，不累计成更多独立案例。覆盖3种校正×提取像元/真实规则格网、观测保留、原始网格不变、实时两源扰动、错误位置/身份/单位/区间、历史重叠和测试资源拒绝、零分母、未知像元几何拒绝；公共随机、none、页面包装、原始CSV和文件顺序变化均验收。额外网格文件的公开入口测试确认平台选中正确位置/来源，没有用第一个文件代替匹配。

早期新增读取器测试出现NumPy将1×1数组转Python float的断言写法错误；改为明确item()后重跑通过，失败日志保留。没有为效果改科学阈值或筛掉失败种子。最终完整回归与新增测试通过后才生成PLAN、冻结输入和源码、运行正式84请求；其间及评价后生产代码均未修改。

## 数据、配对与隔离

原气象ZIP的MD5复核为30afa5e86e13a760f79ecf5fe4a786fa。6站原始降水、原有月均值状态、1998—2001年遮挡输入与2001年外部标签均与上一轮一致。每年固定遮挡1月15—17日与7月15—28日，外部误差仅计算2001年每站17个原本有限值，共102点；不将已保留观测计入误差分母。

为核查像元来源和日历，补取同6个位置、1980—2001年官方单像元降水CSV。6条序列在全部8030个原生日上与交付的未订正Daymet逐值一致，未更换成更有利的产品；官方原本缺少的6个闰年末日保留为未知。未使用Qmap/MFBC全记录订正输出。原始CSV直接作为平台背景输入，实验脚本未提前采样、校正、补缺或指定胜出路线。

每站1980—1997年6570个有限历史站点/像元配对进入公共适配器拟合；5个该期间不存在的Daymet闰年末日不当配对。应用为1998—2001年；当前站点可见值不能重训映射，测试值不拟合。18年历史月均值状态沿用现有文件，所有组可用；该既有状态不是本轮新增拟合，也没有假称其预处理免费等于新适配器实时拟合。

每个处理进程禁止打开external_labels.nc和原始完整气象ZIP；84项请求及复现全部完成并封存产品后，评价器才打开隐藏标签。实际禁读次数为0。内部验证使用任务匹配3折，参考只从当前可见值生成，真实缺口不当真值。18对随机/none同种子请求的初始图与全部验证输入SHA256一致；六站seed11的单源/点网验证主序列折也一致。独立NumPy公式重算18折RMSE/MAE，与平台及归一化目标一致；回放只核验，不参与选图。

2001年站点结果此前已被查看，本轮是同批扩展回归，不是新盲测。更重要的是Daymet为站点驱动的格网估计，上游是否包含这6站的记录未核实。因此“平台没有读取隐藏标签”不等于“Daymet上游与标签统计独立”。这些效果不能当作真实站点完全停测或独立传感器融合优势的无泄漏证明。

## 正式对照与实测

随机点网、none、单源各6站×3种子（11/29/47），另外5条固定路线各6站×1，共84项；另重复站1、seed11一次。所有正常/消融请求同300次公共节点调用、100次探测上限、15000次提议和180秒上限。每条进入评价的候选真实执行。固定路线真实执行，未接入搜索初始化，也未给正常组额外补跑。每站历史角色、目标模型及物理质量门一致；单源只移除背景及背景历史。

下表为每站每种子RMSE先计算、再站点等权平均；单位mm/day。不同组请求数不同，调用总数不能直接冒充单次效率。一个新适配节点包含取样、拟合、应用与保留观测补缺四阶段，算一次登记节点调用；适配器调用次数另列，其中拟合实际执行且耗时已计入请求。客户端计时包括公共处理，不包含Python启动、下载或初次数据整理；不据此与旧轮宣称提速。

|方案|成功请求|平均RMSE|平均MAE|公共节点调用总数|含拟合适配器调用|平均处理秒/请求|
|---|---:|---:|---:|---:|---:|---:|
|平台点网融合＋随机变异|18/18|4.721133|3.047986|2354|720|27.26|
|同条件自动筛选、不变异|18/18|4.721183|3.052858|1224|360|19.47|
|平台单源随机变异|18/18|7.259294|4.741194|3192|0|21.18|
|固定仿射映射＋补缺|6/6|5.026358|3.311936|6|6|5.96|
|固定整体比例映射＋补缺|6/6|5.015836|3.230912|6|6|5.94|
|固定月比例映射＋补缺|6/6|4.696456|2.987402|6|6|5.87|
|固定单源线性|6/6|9.614369|6.986536|6|0|6.35|
|固定单源历史月均值|6/6|7.322441|5.051511|6|0|5.89|

|站|自动点网随机|点网不变异|单源随机|固定月比例点网|
|---|---:|---:|---:|---:|
|1|7.977732|7.978036|12.917602|7.968252|
|2|3.627075|3.627075|4.756041|3.627075|
|3|4.339980|4.339980|7.181113|4.201405|
|4|3.053685|3.053685|5.459345|3.053685|
|5|4.027398|4.027398|3.599586|4.027398|
|6|5.300925|5.300925|9.642079|5.300925|

自动点网相对单源平均RMSE降低34.96%。本批5/6站比单源好，5号站反而变差：单源约3.600，融合约4.027。不能把平均改善说成每个站都获益。

5条固定路线中，事后最低全站平均为月比例映射4.696456，自动4.721133反而高0.525%。这一最佳固定只作事后比较，不用于替换平台推荐。3号站已实际评估月比例模型：验证RMSE约3.605，高于整体比例的3.396，所以选择整体比例；测试排序相反（月比例约4.201、整体比例4.340）。不是该候选没接通，也不能根据测试值回头换图。5号站验证融合月比例约5.416，优于单源月均值约7.170，但外部这段缺口上单源更好；验证不能保证未知事件排序。

变异相对none的RMSE平均下降仅0.001071%，实际差约0.000051mm/day，几乎没有额外收益；18请求节点调用从1224增加到2354，成本更高。这不是变异或三画像创新优势证据。三种子的平均结果一致，只说明本批有限预算的计算稳定；不代表穷举或全局最优。负结果、完整固定对照和每种子轨迹全部保留。

六站seed11的实际图顺序：

- 站1：reconstruction.daily_nearest {'max_gap_days': 3} → fusion.station_grid_affine {'sampling': 'extracted'}；真实依赖['source0', 'source1']。
- 站2：fusion.station_grid_monthly_ratio {'sampling': 'extracted'}；真实依赖['source0', 'source1']。
- 站3：fusion.station_grid_ratio {'sampling': 'extracted'}；真实依赖['source0', 'source1']。
- 站4：fusion.station_grid_monthly_ratio {'sampling': 'extracted'}；真实依赖['source0', 'source1']。
- 站5：fusion.station_grid_monthly_ratio {'sampling': 'extracted'}；真实依赖['source0', 'source1']。
- 站6：fusion.station_grid_monthly_ratio {'sampling': 'extracted'}；真实依赖['source0', 'source1']。

只有站1出现先恢复短缺口再做仿射映射补余缺；其他站根据验证选择不同映射。本轮没有强迫平台复现该组合，也没有把固定月比例注入推荐。新适配器是在物理/数据接口需要的位置组合现有数值组件，不包装成新的科学插值算法创新。

## 确实使用两源的检查

全部18个正常点网请求均包含两个来源。对每站seed11选定图做选择后的两项冻结执行诊断：当前缺口日期的Daymet增加0.1，缺口输出最大变化约0.093—0.119；另填入一个合成可见站点值，产品准确保留它。两项都不使用真实隐藏值，历史拟合系数均不变。这个机制是“保留当前主序列、用历史配对模型估计的网格值补剩余缺口”，不宣称每个缺口都同时利用当前邻站观测。

18折回放和12项两源诊断共30项额外公共请求、35次登记节点调用，单独计账，不记为搜索发现、搜索精度优势或额外评价预算。同种子复现另有136次调用。最终验证最优未交付或检查点回退的请求列表为[]；未用实验脚本替平台完成失败任务。84项正式请求的登记调用总数为6800。

## 使用条件与尚未完成

- 本轮支持单站目标与有可靠元数据的官方Daymet降水提取序列；EPSG:4326规则网格取样有机制测试。原生Lambert全域网格、曲线格网、其他变量和多站联合空间估计没有在本轮真实数据中端到端验收，不能外推为所有点网数据都支持。没有把358个不规则提取位置改造成虚构矩形网格。
- 拟合需要明确授权及严格独立的历史站点/背景配对。没有历史资料或未知语义时拒绝该适配，保留单源选择；不从任意未标注文件自动猜训练期、单位或来源。不支持把整个原始气象ZIP直接上传并自动完成所有准备；本轮原始Daymet CSV可直接读，站点和历史资源使用已有规范NetCDF接口。
- 当前每个请求为一处站点，六站逐一走同一个平台入口。不是一个请求完成整个流域的全部气象场融合，也不是站点反推全域格网或真实空间超分辨率。
- 没有点估计预测区间或代表性误差传播，未核实Daymet上游站点独立性。不能称严格独立精度验证。5号站退化和固定方案稍优仍是有效负结果；本轮到此冻结，不追着测试误差调参。
- 无用户用途精度阈值，模型敏感性仍UNKNOWN。复用SWAT+降水准备契约，不代表真实SWAT+已运行；未验收下游格式合并、径流率定或径流改善。实际产物是恢复降水NetCDF。

页面现有“允许使用本次明确提供的训练资源拟合”控制已复用。cases/各站/input.json给出相同公共入口所需文件和目标：primary.nc为目标站点，daymet_raw.csv为背景，station_prior.nc为training_reference，grid_prior.nc为historical_simulation，state.nc为已有月统计状态。用户指定目标和合法数据角色，不指定算法。页面函数与API已验收，未另做浏览器点击验收。

## 复核位置

output/station_grid_fusion_20261031：下载及日历核对清单、PLAN、输入/源码哈希、生产diff、执行源码快照、机制与回归XML、85项runs及日志、封存清单、metrics/comparison/effects、fair_control_audit、single_source_mask_audit、selection_diagnosis、runtime_audit、18折metric_replays与12项source_probes/product_audit。每条请求的产品、拟合区间/系数、数值轨迹和成本保留。原用户文件夹和先前负结果未覆盖。


# 87. 通用空间关系融合与Humaya实际交付（20261031批次）

本轮完成的是平台公共模块增量及区域降水实际交付；没有合并队友转换器、极简改版页面或运行原生SWAT+。不把此前单站恢复实验当作区域融合证据。

## 公共模块和自动决策

当前生产版本platform_search_v14r1_spatial_relationship_mosaic。process_current_input / submit_file_request / 页面process_search_upload共用同一分发，按数据结构、角色和目标选择既有栅格重建或新增空间关系适配。新适配支持规范站点表、EPSG:4326规则网格和配对空间位置集合上的时序连续变量；四类基础关系已有真实数值机制测试：多站点→空间估计、站点＋空间背景、同变量多空间来源、不同单位协变量→拟合辅助估计。温度和降水入口均实际执行，不只为Humaya写特例。

新增2个源模块，修改7个既有生产模块。NumericalDAGExecutor、质量门、SearchAccount、搜索停止器、模型字段硬契约、原IDW距离公式、残差加减及最小二乘均复用。field_mutation原随机循环原样提取到composition_optimizers.random_graph_search，由旧引擎和新任务引擎共用。没有换成GOLEM、ALNS或另一套学习排序。

新任务提供9种带类型端口的组件（含背景、保留观测、协变量三种源端口；数值组件为IDW、协变量仿射响应、仿射/比例校正、残差IDW、凸组合）。观测矩阵是处理图的显式输入，不作为隐藏评分旁路。可以连接不同组件，不能把不同物理变量直接相加。未适配到这些新端口的其他库算法，不因本轮而自动获得空间位置集合能力。

模型已知单位、变量及非负硬要求参与准入；未知敏感性、预测区间和新端口软收益仍UNKNOWN，不套用其他任务历史权重。新空间端口最终按实际留位置RMSE选图；三画像软收益、跨任务训练排序或完整下游模型偏好优化不属于本轮已证明功能。共同误差数值精度内选短图，随机变异保留。

## 来源识别和语义保护

同产品的不同位置资料，依据已声明、可追溯的mosaic_group、产品与处理层级及来源证据识别为拼接，不当作独立来源。重叠有限值必须相同，冲突拒绝而不是平均；未知来源不猜。该元数据由原始资料核对后整理进入文件，并非学出来的隐含关系。用户目标是区域估计，未指定赢家算法。

本次复用作者6站完整逐日降水、原始358位置Daymet及此前核验过与原包六站位置未校正序列一致的官方Daymet查询文件。Qmap、MFBC校正结果未作为输入。两份Daymet视图同产品，平台拼接后363个不同位置、1处相同重叠；随后选择背景与站点观测的处理图。

原始358位置不是43×49完整矩形网格，未为接入补造矩形栅格。输出保留原358目标位置、原ID/名称/经纬度和1980—2001原日期标签；不提高真实观测分辨率，不声称358个观测站或精确面积平均。日累计使用明确日区间，禁止线性插值累计值；日界/时区未额外核实，不伪造UTC含义。

Daymet导出中的-99仅出现在6个闰年12月31日，处理为未知，与官方原始查询的缺日一致，不补零。最终图在这些日期通过站点IDW估计，覆盖2148个输出值。

## 实际执行和验收

116项机制与回归测试通过，0失败/错误/跳过。包括四类输入、站点与网格真值隔离、源值扰动、协变量单位不混入目标、输入语义拒绝、同产品拼接/冲突、原单站与网格功能及超时收尾。页面函数和文件客户端同输入同seed结果一致；未做浏览器点击或队友部署验收。

搜索部署复用资源账本。此次还修正了账本的已有问题：配置了独立部署额度后，搜索超时不再耗掉该额度；总调用量仍受限。旧超时测试改为在有执行记录的提议后模拟超时，允许同轮证据门先停止；source-only合格交付按实际源读取轨迹检查，不虚造数值调用。新空间引擎另有真实新增调用后超时仍执行产品的测试。

正式6项请求全部PASS：自动随机变异、同输入不变异、仅站点单源、固定IDW、固定背景＋残差IDW、同种子重复。另有固定图库同口径回放、三类输入扰动、完整计时复现和实际交付ZIP解压复现，均另记账，不当作搜索优势。

自动搜索在10000次提议处结束，78张图实际评价，54合格、24因固定留站评价位置输出不完整等被拒绝；失败图、错误原因、图和成本保留。不是所有候选均合法，也不是穷举或全局最优证明。所选完整图重新部署，最终结果与同种子复现及压缩包内脚本复现一致。

验证按位置整段留出，当前折只有其他位置的观测进入校正；前80%日期选图，后20%冻结后确认。共38568个搜索留站点日、9648个后段确认点日；自然缺测不当真值。由于是此前使用过的同批历史资料，以及Daymet上游可能包含这些站点，不能称新盲测或严格独立产品验证。

|方案|搜索留站RMSE(mm/day)|后段留站确认RMSE|
|---|---:|---:|
|自动随机变异融合|4.425025|4.021167|
|同输入不变异|6.582041|5.123515|
|仅站点单源自动处理|6.582041|5.123515|
|固定IDW(power=2)|6.635910|未单独评价|
|固定背景＋残差IDW(power=2)|4.549675|未单独评价|

本次确认RMSE相对单源低21.5%，但没有对固定方案的确认段评价或多个独立种子稳定性证据，不能据此宣布普遍优于最强固定、全局最优或创新已经证明。没有因结果调参、追加赢家初始化或用确认段切换产品。

所选实际处理图：同产品拼接后的背景源、当前日期保留站点矩阵 → 残差IDW(power=1) → 公共非负质量约束。站点、六站位置网格片段、358位置网格片段分别+0.25的冻结图输入扰动，最大输出变化分别0.2367659569、0.1293714848、0.25，未重新选图，证明三个输入部分真实参与。

原未拼接试版的所有输入/源快照/输出保留在pre_mosaic_trial，最终交付已替代：原版错误地把同产品视图当竞争来源，160个位置会不必要地采用仅站点估计；最终只有2148个点日因原始Daymet确实缺日才回退。它的指标不冒充最终产品，修复依据是覆盖与来源关系，不是测试成绩。

## 输出质量、成本和未完成

最终8036×358=2876888个降水估计，0缺值、0负值。160个位置在站点凸包之外，虽然有Daymet背景，但其站点校正属于外推，NC与位置质量表已标注；无独立精度认证。191948个初步负估计按统一非负约束置0（约6.67%），未删日期，验证也应用同一约束；不证明质量/水量守恒。未知误差区间不造数。

主随机请求380次登记节点调用，不变异39次，单源236次；固定完整图2/3次。调用粒度是批量时空矩阵节点，不是单日或单点；图内缓存不计新增调用。解析、质量检查与同产品拼接在账本建立前发生，不能仅将account.service_seconds说成总耗时。独立端到端复现已另量测，包含读入、公共解析、拼接、选图、输出和回执，约4.81秒，不含Python启动/导入和压缩；并行请求与其他设备时长不同。固定图库回放36调用、三项扰动共9调用，计时/导出复现另记，未算作变异收益。

现有验收范围是规范表格/NetCDF中的区域尺度时序连续场。新端口未验收任意投影、原生曲线格网、任意陌生ZIP自动解包识别、类别/空间总量分配或区间均值；守恒面积转换需原有相应适配，不能用点采样伪装守恒。瞬时协变量可在已知日期之间线性取样，不外推。没有重写全部算法端口，也未接通三画像新端口软收益学习。

没有运行SWAT+、修改/率定模型参数、证明径流改善、合并队友原生转换器或完成极简GUI。队友应读取NC或逐日CSV＋位置清单再转换，核对工程版本、气象点映射和其他天气变量；本包不充当完整模型工程。没有用户用途阈值，不宣称实际用途精度达标。

## 交付与复核

数据ZIP包含NC、CSV、358位置表、背景可用性和外推标记、决策/错误/图/成本、116项XML、来源及输入、portable config/reproduce.py和转换接口.json。ZIP逐成员SHA256、CRC及压缩包内NC回读通过；已实际解压该ZIP运行所附脚本，数值及科学属性与主结果一致。CSV保留10位小数，最大舍入误差5.00009e-11 mm/day。

另有9生产文件＋2测试文件的增量合并补丁与production.patch、基础/修改后SHA256、合并说明。它依赖既有EcoDataEngine公共模块，不是可盲覆盖任意版本的独立软件；不同基础版本须逐项合并并重启既有服务。平台源快照243个源码/配置文件在正式运行前冻结，交付时再次核对未变。

所有最终证据位于output/spatial_fusion_v14_20261031；原数据与旧实验未覆盖。唯一总报告仅追加本章，不改此前结论。


# 88. Humaya三画像软引导配对消融（20261101）

结论：软评分已实际参与候选选择；五组种子均未改善最终精度。保留为显式试验开关，平台默认仍关闭，不据此更改既有融合交付结果。

## 预先冻结的对照

复用8036天、6站和358个目标位置及同产品Daymet拼接。种子11、29、47、71、101，每个种子各跑开启/关闭两臂，共10次公共入口submit_file_request。两臂使用相同源数据、候选组件、初始候选、候选生成器和预算（objective_limit=900、total_call_limit=1100、proposal_limit=10000、wall_seconds=300），同样保留前80%搜索、后20%冻结后确认。完整位置留出，遮挡标签不进入对应折画像。确认结果不更新评分和搜索。

为使候选优先选择可比较，两臂每轮均生成4个提议；关闭时取第一个随机候选，开启时约半数轮使用三画像锦标赛，其余保留随机。它们仅为待评提议，只有进入评价的候选才登记数值成绩。所有登记候选均真实执行，失败保留。该池宽与旧轮单提议不同，因此主要结论来自本轮配对对照，不把与旧轮预算消耗差异算成软评分收益。软引导在初始化和变异候选选择中生效，最终选图依据相同真实验证RMSE与数值等价时的简约规则。

## 实现与证据边界

复用candidate_portraits.py既有公式，抽出score_profile公共函数；spatial_portraits.py仅作配对位置适配。以保留站点的地理近邻变化、空间支撑与查询距离构建数据画像；以真实组件契约判断空间作用能力，重复动作不重复奖励；模型采用现有robit-era5-multisource-swatplus-v1与robit-daily-state-v1的等权数据准备偏好，作为显式试验迁移，不能解释为已校准SWAT水文敏感性。expected_error始终UNKNOWN。各项分数记录在portraits.json和candidate_results.json中。

不引入人工预期路径，不把验证误差改名当软评分，不训练新排序模型，不在看到确认结果后调权。请求开关为spatial_soft_guidance，候选池参数为spatial_candidate_pool；开启必须附已注册模型画像策略。缺少模型策略拒绝运行。

## 实测

| 指标 | 关闭软引导 | 开启软引导 |
|---|---:|---:|
| 搜索RMSE（5种子均相同，mm/day） | 4.4250250612 | 4.4250250612 |
| 确认RMSE（5种子均相同，mm/day） | 4.0211667997 | 4.0211667997 |
| 平均登记算法调用（含部署/确认） | 359.6 | 342.8 |
| 平均完整公共请求秒数 | 3.180 | 3.379 |

开启后平均登记调用减少4.67%，完整请求时间增加约6.25%，计时包含解析、画像、打分、搜索、输出和回执，排除进程启动/导入与外部归档。该小样本计时不构成显著速度结论。画像和锦标赛虽然不新增科学候选探测调用，但确有计算开销，已包含在完整时间内。

五组最终都选中Daymet背景+站点残差IDW(power=1)。所有配对输出逐值相同，8036×358输出均有限且非负。候选实际评价数关闭平均73.6，开启平均67.8，失败分别22.8和19.2，全部保留。

软评分不是空开关：五种子严格因分值更高而改变提议选择分别262、265、271、282、249次，其中产生新登记评价分别5、10、8、8、7次；另有按未知项数量、简约和稳定ID打破同分的变化，不能全算成分数收益。已评价候选的软分主要只有0和0.2424216478两个档。它能区分“有空间处理动作”与“没有已知空间适配响应”，无法区分IDW、残差订正及其不同参数的真实精度。保留现状的合理结论是：工作了，但本任务上仅有有限搜索优先级信息，没有最终精度优势，也没有总耗时优势。

## 验收及局限

40项定向/回归测试通过，0失败/错误/跳过，覆盖共享公式、权重作用、对应折留出标签与确认日期隔离、软评分实际改变执行候选、缺少策略拒绝，以及原空间融合和图变异功能。source_executed、protocol.json记录本轮源码与输入哈希，运行后复核不变；output_audit.json记录10次输出复核。首版一次测试因直接构造对象未赋budget失败，已修正构造器并重跑完整40项通过；失败日志保留在output/soft_spatial_audit_20261101.log，不覆盖。

本轮无新开发训练或权重校准；旧批次及确认数据此前已查看，属于已知数据的机制对照，不能作为新盲测。Daymet上游站点使用仍未核实，不能称严格独立产品精度证据；未执行SWAT+及径流收益评价。候选未穷举，不认证全局最优。不声称三画像软评分的研究创新已被证明。原结果ZIP和补丁ZIP保留原版，没有把本轮可选适配偷偷写入旧交付包。

证据：protocol.json、results.json、decision_audit.json、output_audit.json、tests.xml、run.log及各seed_arm文件夹。


# 89. FVC关闭软评分与随机变异复测（20261101）

结论：三画像软评分在当前PortraitMutation公共入口默认关闭，保留硬约束、随机变异、真实验证误差选优。原评分功能保留为显式选择，不删除。FVC本轮随机变异相对本轮不变异改善2.5689%，相对原开发期冻结固定路线在当前平台的重放改善0.6007%；相对旧轮随机结果退步0.7648%，不能宣称此次关闭评分带来了总体提升。

## 数据与实验边界

用户已确认是黄山附近小流域FVC（植被覆盖度），不是土壤pH样例。复用output/basin_ecology_2013_2017原输入：2013—2014训练、2015开发、2016验证选图、2017测试；三个原8×8空间缺口。FVC来源是Landsat端元代理，不是地面真实覆盖度。本轮为已知测试数据的回溯复测，没有新盲测，也没有重新拟合任何三画像排序器。原气候、地形、土地利用等协变量通过原接口提供，不能仅因文件入库就声称所有来源都对胜出图有贡献。

原配置、先验训练资源、区域和标签不变；平台使用当前完整可适用注册库与科学接口自动生成节点/连接及变异，研究者不提供预期赢家。使用相同旧预算：objective_limit180、proposal_limit40000、wall_seconds120、probe_call_limit60。三种子11/29/47，random与none各9次；2015冻结的development_fixed、nearest、linear各3次，共27次。固定路线只进入对照请求，不传给随机搜索。每个数值候选均真实执行，失败记录不删除；预算停止不表示全局最优。

选图走submit_file_request公共入口，作为受控基准显式提供2016 search_validation。2017标签仅在全部27份预测与SHA256封存后读取评价，所有搜索请求均无test_labels路径。该轮不属于无外部验证标签的自动验证规划或浏览器点击验收。可见的2017缺口外输入允许用于历史恢复，缺口内2017真值保持隔离。

## 结果（各区域先对种子取均值，再对三个区域等权平均）

| 方案 | 测试RMSE（FVC，0—1） | 平均登记算法调用 |
|---|---:|---:|
| random | 0.06848937 | 180.0 |
| none | 0.07029520 | 45.0 |
| development_fixed | 0.06890324 | 3.0 |
| nearest | 0.21324611 | 1.0 |
| linear | 0.23450632 | 1.0 |

random三个区域RMSE分别0.07178917、0.07421291、0.05946603，none分别0.07329274、0.07663316、0.06095971；区域均值均有改善，逐种子为7/9优于none，2/9退步。种子及区域并不相互独立，不能当作九个独立流域样本。random九次平均完整请求约18.48秒，none约1.49秒；变异需要额外计算，不声称速度优势。

旧轮random均值0.06796952，本轮0.06848937，略差0.76%。当前代码与算法库已经更新，不能将旧新差异单独归因于关闭评分。原固定图在当前数值内核重放的结果也与旧轮不同，固定对照比较统一使用本轮重放结果；不按2017测试再选一个新固定赢家。本轮没有额外跑当前版本“开启评分”的匹配臂，因此不能用此轮证明关闭评分本身更优。

## 公共入口与验收

field_mutation.submit及platform_search.submit_search的PortraitMutation入口默认soft_guidance=False。软评分关闭时matching明确记录DISABLED，不调用三画像状态评分；响应探测关闭，初始化不按软分排序，随机驱动不切换画像引导，最终按实测验证误差与已有简约规则选图。单位、变量、输入输出兼容、物理范围、可见值保护和质量门保留。可选的历史Profile/GOLEM模式保留，未将其称作本次变异路径。

27次均PASS，所有输出有限、范围0—1，原可见值逐值保护。9个随机胜出图用公共execute_frozen_graph和平台封存的规范化请求重放，无验证/测试标签且不重新选图；最大差异1.6653345369e-15，小于原选图的浮点同分精度。重放另花52次登记调用，不算搜索收益。首次审计误用未映射栅格目标日期的原始薄客户端请求而拒绝；换用平台封存规范请求后，逐值完全相等断言检测到上述浮点末位差异，最终按既有objective_tie_resolution核对通过。原审计失败日志完整保留，不把它们算成候选搜索失败。

70项定向/回归测试通过，0失败/错误/跳过。新增用例禁止关闭后的score_state调用并验证真实候选仍可执行。源码与输入哈希运行前冻结、结束及独立重放后核对未变。原数据、结果和ZIP不覆盖。

## 可交付结论

当前无软评分的随机变异链路可以运行并产生合格输出，本轮相对不变异有小幅精度收益，维持默认关闭软评分是合理的工程选择。没有证明全局最优、跨流域普遍优势、生态模型或SWAT+收益，也不能将回溯复测写成新的独立科研证据。

证据目录：protocol.json、source_executed、run_receipts.json、predictions_frozen.json、test_metrics.json、effect_summary.json、independent_audit.json、tests.xml、run.log、audit_final.log，以及runs与replays_prepared。原始失败审计保留audit.log、audit_normalized.log。


# 90. FVC自动多折验证复跑及固定对照更正（20261101）

结论：自动多折计算、隔离和公共执行链已实际验收；当前数据缺口的代表性仍不足，平台明确EVIDENCE_INSUFFICIENT。随机变异在第一轮后被既有保护规则停止，最终保留2013—2014历史FVC均值先验。随机变异与不变异测试RMSE均0.06890324，原2015开发冻结完整固定图为0.07625406。不能由此宣称变异取得精度收益、完整融合胜出或验证已经覆盖全部大缺口。

## 1. 先验收机制，再执行对照

未修改任何生产源码、评分权重、候选库、保护门或停止规则。复用原三个区域和先验训练数据。选择输入裁到2015—2016，辅助资源保留2013—2014训练标签；2015之后training_target全NaN。2017整年不进入选图输入，旧validation_labels.nc不提供给服务；未设置search_validation或evaluation。

按平台现有task_validation.plan自动生成遮挡，不由实验脚本指定验证坐标或标签。三个区域共审核9个自动折，有限参考值与遮挡前可见值一致，算法输入对应位置全部NaN；参考日期均不晚于2016。mechanism_gate用公共evaluate_graph_bank真实执行一个最近邻补缺图，账本显示search与两个search_fold各有一次注册数值调用，三个fold_responses齐全。修改第二折的评价数值会改变公共aggregate结果，证明不是只评价首折。

三折主要对应可找到参考的自然空间缺口。输入还存在大型自然缺测与跨两年的持续缺测块，自动协议逐项记录INSUFFICIENT；保留LIMITED_DIAGNOSTIC_ONLY，未放宽成TASK_MATCHED_INTERNAL。此处“多折运行成功”与“缺口代表性充分”分别验收。

## 2. 公共入口与隔离

随机9次：submit_file_request(decision_policy=current_input) -> process_current_input -> 统一服务 -> task_validation.plan。不变异9次作为显式对照，通过公共submit_file_request(mode=PortraitMutation、mutation_control=none)，validation保持None，进入同一自动规划分支。固定3次用公共evaluate_graph_bank，唯一图为2015开发期已封存development_fixed，仅出现在固定对照中。

虽然自动请求与对照请求的模式元数据不同，每个区域各组实际验证标签与遮挡输入逐值/哈希一致，均3折；没有给随机路径输入固定图。所有组统一使用当前平台标准预算1800次调用、1800次总调用、400000次提议、3600秒与600探测上限；软评分关闭后不使用额外画像探测。预算是上限，实际提前结束。

每次选图后封存图及SHA256，再用平台公共execute_frozen_graph部署原2015—2017完整可见输入，正确映射raster_target_times；部署不重新选图，不接收验证/测试标签。全部21张图和产品封存到predictions_frozen.json后，才读取2017的test_labels.nc评分。此轮未执行浏览器点击验收；页面与文件端共用的公共数值服务已经实际调用。该数据的测试年此前已见，属于回溯复测，不能写成新盲测。

## 3. 结果

| 组 | 公共请求数 | 测试RMSE | 选择与执行成本 |
|---|---:|---:|---|
| 随机变异 | 9 | 0.06890324469 | 共1224次登记调用，平均136次 |
| 不变异 | 9 | 0.06890324469 | 共1197次登记调用，平均133次 |
| 原完整固定图 | 3 | 0.07625406150 | 共54次登记调用，后续留年部署另9次 |

三个区域先分别对种子取均值，再对区域等权平均。随机与不变异每个区域的结果分别同为0.06897822、0.07677121、0.06096030。当前产品较完整固定路线低约9.64%，但胜出的是历史先验，并非变异后的组合；不能归因于变异创新或多源融合收益。

九次随机均实际评价一个新变异候选（三折），随后因INSUFFICIENT_MATCHED_VALIDATION_FOR_PROMOTION停止。所有九次最终图是s1，即原先验源；source-only留年执行不产生注册科学算法调用，不能虚增为融合计算。变异后候选及失败记录全部保留。代码没有用测试误差替换结果。

21次请求和冻结图部署均PASS；输出有限、范围0—1，原可见值逐值保护。选图阶段每个最终候选都有三个真实fold_responses。现有支持保护没有被取消；全部task_acceptance仍为EVIDENCE_INSUFFICIENT，不称已达到未给定的用途精度门槛。

## 4. 对上一轮固定对照的更正

上一轮fvc_nosoft_retest_20261101使用mutation_control=fixed时，通用执行器将已执行前缀也纳入候选，最终best未必是声明固定图：block0选中另一前缀，block1和block2选中s1。故此前报告的“固定RMSE0.06890324”和“随机比固定改善0.60%”不是完整固定路线对照，现明确撤回这一比较。旧日志、输出和报告保留，另附control_comparison_correction.json。

本轮使用已有evaluate_graph_bank限定固定图集合，三个区域selected_bank_id均为development_fixed，best.key均与原封存顶层图md99b2f16267174ef964b70ae3eddb57318bd112fc0257381f3951554b71fa100严格相同。因此本轮固定结果0.07625406才对应原完整路线，也与原旧实验固定成绩相符。没有把固定图塞入自动初始化，也未为这次对照修改生产实现。

遗留边界：旧mutation_control=fixed仍可能返回其前缀，本轮没有修复；当前规范对照使用已有graph-bank入口。graph-bank结果的默认stop_reason仍显示RESOURCE_BUDGET_EXHAUSTED，实际仅18/1800次调用每个区域，没有耗尽预算，这是原有元数据边界，不能据此分析成本或归因结果。

## 5. 验收与保留证据

83项现有自动验证、前缀、多折、可靠选优和随机变异定向/回归测试通过，0失败/错误/跳过。independent_audit.json核对21次请求的外部标签未提供、软评分关闭、三个真实折、对应遮挡、组间同折输入、第二折能影响总分、2017只作最终评价、固定图身份与最终物理输出。源码与输入哈希运行前冻结，结束后及独立审计再次验证不变。

mechanism_gate的数值调用成本单列于其total_cost.json，不能算成搜索优势；各选择、初始化、失败、检查点/部署和冻结图部署账本都保留。未做新模型率定、SWAT+/生态模型模拟、全局最优认证或新的学习评分。存在完整输出不等于真实缺口误差已获保证。

证据：mechanism_acceptance.json、mechanism_gate_receipt.json、protocol.json、source_executed、run_receipts.json、predictions_frozen.json、test_metrics.json、effect_summary.json、independent_audit.json、tests.xml、run.log、runs/*/validation_folds及held_year_deploy。运行脚本仅组织输入、调用公共接口、封存和评分，不手工选图或补换失败种子。


# 91. 有效验证下持续随机变异控制修复与FVC验收（20261101）

结论：已在平台公共服务修复控制逻辑，不再以验证代表性有限禁止搜索或锁住先验。最终148项回归通过；21次正式公共请求全部PASS。九个随机运行实际评价202—331个变异候选，接受9—28次改进，全部替换初始图，未退回s1先验。测试RMSE随机0.06747838、不变异0.07006986、原完整固定0.07625406，分别改善3.6984%和11.5085%。区域1反而退步，完整负结果保留。

## 改动及范围

复用原随机变异、图执行、自动验证规划、质量门和多折实测目标，仅修改四个生产文件：reliable_selection.py、field_mutation.py、composition_optimizers.py、platform_search.py。源码差异见production.patch；before和source_executed保留修改前/正式执行快照。公共版本platform_search_v15_measured_validation_random。

1. 初始方案使用已有全局等价容差排序：先选完整、合格的实测验证目标最优方案，仅在容差内比较复杂度。移除“先挑最少节点、再冻结”的独立限制。
2. 后续候选必须完成全部必需折并通过既有质量门，与初始方案使用相同fold_ids和loss_signature；按原多折平均实测损失排序。缺折、失败、口径不一致、被科学状态去重支配的候选不能替换合格结果。允许平均更好但单折较差的候选，不新增另一套评分。
3. 删除因为support_limited或仅一折而在首轮强停的规则。完全没有有效隔离验证时仍拒绝精度搜索；有限代表性仍在task_acceptance明确EVIDENCE_INSUFFICIENT，不伪造用途达标或全局最优。
4. 复用执行前的调用预算检查can_evaluate，在提议选择时排除无法完成整图/全部折及最终执行预留的候选；所有操作族自上次新调用以来都未带来新调用、且已有提议被预算阻止时按资源预算收尾。最低一轮调用也不足时不继续发起提议。预算/质量/图空间没有为追优势放宽；这个停止判断只针对已提议邻域，不代表穷尽全部合法图。
5. 未显式指定且超出任务范围的NDVI收益停滞阈值不套用到FVC；当前FVC保持原数值/提议/时间资源上限，显式停止配置继续支持。图库对照完成的stop_reason更正为DIAGNOSTIC_BANK_COMPLETE。

三画像软评分仍关闭：公共请求soft_guidance=False、response_scoring.enabled=False、portraits_enabled=False，matching明确DISABLED，候选先验不参与初始化或变异，未训练其他排序模型。保留单位、语义、物理范围、可见观测保护及所有质量检查。指标中沿用portrait_response的记账字段记录真实误差分解，不是额外画像软分。

## 先机制验收，再正式复测

最终版公共入口定向用例在support_limited=True下持续260轮，实际新变异评价80个，验证目标从0.04443533降到0.02468079。另一真实两折用例将最近邻图0.19477389替换为线性图约2.21e-16；即使代表性标记有限，实测更好仍可替换。零有效折拒绝、缺折/口径不一致/失败/支配候选拒绝、同分简约、预算不足完整一轮即停止，以及所有操作族预算阻塞后的收尾均有定向测试。

正式FVC首组作为不读取测试标签的机制检查，并计入21次正式结果而不重复挑种子：220个实际变异候选，目标0.22265833降到0.14813325，确认可以持续搜索及替换后再启动其余组。首组未向搜索器提供任何预期好图。选择后已经执行并封存的2017产品尚未评分，直到所有21份图与输出封存完成才统一读取测试标签。

机制验收中保留两次测试评分前的资源收尾诊断，位于diagnostic_budget_floor与diagnostic_unaffordable_proposals，均有WITHDRAWAL.json。前者发现剩余额度不足完整一轮时仍会提议，后者发现理论上能跑小图、实际提供的组合却都超预算时会空转；修正后正式对照统一在同一源码快照重跑。诊断未用于选图、调参数或选择测试赢家；其账本未全部正常收尾，至少原始attempts中记录的调用已保留，不声称诊断成本为零。

## 数据、入口与隔离

同一黄山流域三个原8×8目标块，2013—2014标签只供既有算法拟合，选图仅使用2015—2016输入；没有外部search_validation/evaluation。2017整年不进入选图输入，辅助training_target在训练期以后均NaN。自动验证仍由task_validation.plan识别自然缺口并生成3折，不手写验证位置或预期算法路线。

随机9次走submit_file_request(decision_policy=current_input)；不变异9次为公共无变异对照，validation保持None进入同一自动规划；固定3次走公共evaluate_graph_bank，唯一图为原2015开发冻结development_fixed。各区域组间3折实际参考及遮挡输入哈希一致。固定图不进入随机搜索。

预算与上轮自动实验一致：1800次调用/总调用、400000次提议、3600秒及600探测上限；软评分关闭后不做画像探测。实际新增或失败调用、初始化、检查点和最终执行均记入服务账本。冻结图后统一使用公共execute_frozen_graph处理原完整可见输入，部署不重新选择，不接收验证或测试标签。全部21份图和输出先封存，然后评分2017目标缺口。

FVC参考是Landsat派生端元代理，非地面真值。该测试年此前已查看，本轮是回溯复测，不能写成新盲测。数据的自然大型/持续缺口代表性仍有限；多折执行正确不能保证未知大缺口误差。

## 测试结果

| 方法 | 次数 | 测试RMSE（FVC 0—1） | 平均服务登记调用 |
|---|---:|---:|---:|
| random | 9 | 0.06747838 | 1793.67 |
| none | 9 | 0.07006986 | 135.00 |
| fixed | 3 | 0.07625406 | 18.00 |

三个区域先对种子取均值，再对区域等权平均；重复种子不增加独立流域样本数，不提供伪独立像元显著性。

| 区域 | 随机 | 不变异 | 随机相对改善 |
|---|---:|---:|---:|
| block0 | 0.06493490 | 0.07329274 | 11.4034% |
| block1 | 0.07732592 | 0.07623678 | -1.4286% |
| block2 | 0.06017433 | 0.06068007 | 0.8334% |

九个种子配对中5次优于不变异、4次退步；区域1三个种子均退步，区域2种子47退步。没有剔除或补换种子。平均精度收益与每个案例都改善是两种结论；不能说泛化问题完全解决。

随机九次初始化验证目标均被后续更好图替换，最终选图均不是先验s1。独立审计验证最终图确为可部署合格候选中的全局等价容差内最优实测结果；这个“最优”仅限已评价图，并非整个可行图空间的全局最优。胜出图的各节点独立贡献未逐项消融，本轮不能据此认证每个节点都必要。

变异平均约1793.67次服务调用，不变异135次，固定18次；变异需要更多计算，不声称速度/成本优势。冻结图跨年部署另89次调用。独立审计重放27个随机胜出图的验证折，另210次调用，单列audit_cost.json，不计入搜索收益。机制/失败诊断和测试也有额外成本，完整原始日志均保留。

## 验收及未证明事项

148项最终发布回归全部通过，0失败/错误/跳过；早期一项历史测试仍假定默认开启软评分，现改为显式soft_guidance=True以测试其可选功能，原失败XML保留。未删除失败测试或放松断言来迎合FVC结果。

21次正式请求和跨年固定图部署均PASS；输出有限且0—1，可见值逐值保护。独立审计核对软评分关闭、初始实测最优、后续持续搜索及全池正确选图、组间同折、遮挡隔离、完整固定图身份、2017不参与选择、最终输出及源码/输入哈希。27个胜出图折的重放误差与原日志在既有浮点精度内一致。生产源码与输入在正式运行和审计间均未变化。

保留task_acceptance=EVIDENCE_INSUFFICIENT。没有新增真实模型精度门槛、运行SWAT+/生态模型或证明通用优势。没有进行新的下载、更新算法库、模型率定或恢复三画像引导。正常随机变异已经实现并通过真实轨迹验收；进一步提升验证代表性仍需要合适数据和独立效果证据。

此前第90章更正的旧fixed分支前缀选图边界仍保留，本轮固定对照继续使用已存在、锁定完整图身份的graph-bank公共入口，没有把完整固定图与其前缀混比。页面薄客户端共用本公共服务及新版本指纹；本轮验收为文件/API流程，未另做浏览器点击或重启队友环境。

证据目录：mechanism_acceptance.json、real_data_mechanism_receipt.json、protocol.json、source_executed、before、production.patch、run_receipts.json、predictions_frozen.json、effect_summary.json、test_metrics.json、independent_audit.json、audit_cost.json、release_tests.xml、run.log、audit.log及runs/*的原始候选、错误、折、轨迹、图与部署产品。


## 2026-11-03 GitHub最新版更新与旧GOLEM退役

- 更新来源：私有仓库1775828095-ux/EcoDataEngine的codex/latest-platform-20261103分支，提交5448623959fef7167eba002dde1830c26cf0e7ab。此时main尚未合并该最新版。
- 已将最新版及本轮必要修复同步到D:/codex/EcoDataEngine；Git协作目录D:/codex/EcoDataEngine-GitHub保持对应修改。此次未推送、未合并远端分支。
- 删除旧GOLEM优化器实现、独立启动脚本、requirements-golem.txt及pyproject的thegolem可选依赖，移除页面历史GOLEM选项及包公共导出。默认公共服务使用PortraitMutation；旧策略请求明确拒绝，不静默替代。
- 保留当前随机变异、自动识别、融合、模型转换、科学算法及质量门。混合时间算法注册不再为取得算法而实例化旧GOLEM工厂。历史任务定义保留少量拒绝式兼容入口，不再加载或执行GOLEM。旧历史实验不再是受支持的在线入口；其原始源码可由Git提交或备份恢复。
- 新分支漏带workbench.css，已复用原平台对应样式，修复主页面启动失败。
- 定向验证：test_golem_retirement、test_current_input、test_minimal_workbench、test_workbench_folder_input共29项不同测试均通过。首次协作目录运行27通过、2失败，失败为协作目录缺少本地R/sinkr依赖，未改变算法和断言；同步原目录后两项及3项退役测试复测5通过。包含阻断golem导入后的算法注册、旧策略明确拒绝、主页面AppTest、真实小规模随机变异和文件夹输入检查；未执行全量回归或正式研究实验，未作浏览器人工验收。
- 原数据、output、external、.venv及历史结果保留，未清理旧环境。源码修改前备份D:/codex/platform_update_20261103/before_source.zip；更新来源和逐文件SHA256见update_plan.json、sync_manifest.json；测试日志tests.txt、tests_local.txt，早期失败记录保留在测试临时产物中。
- 本次仅确认上述工程更新与定向行为，不据此声称算法精度或全部模型原生执行均已验证。


## 2026-11-03 CAMx EPA首日ZIP平台内部执行

通过主工作台相同公共入口prepare_uploads，对CAMx_EPA首日工程与气象融合数据_20261102.zip执行两次。仅安装注册仓库内existing_air转换插件（--no-deps --no-build-isolation），未安装CAMx或第三方运行工具，未执行包内脚本。ZIP为5,170,342,324字节，56个文件，展开11,919,178,839字节；触发平台10 GiB安全限制。转换器需求检查还发现CAMx配置D:/EcoDataEngine/cases/CAMx740_20261020/official/native_complx_emission_probe/config.json不存在；现有插件只绑定7.40 NO/PEC排放更新，而输入是7.20气象及完整工程。未放宽限制或伪造替代配置。

状态：公共入口实际调用完成，自动融合/格式转化未执行，CAMx原生模拟未执行。源文件未改动。运行证据output/camx_epa_platform_20261103/run_01.json、run_02.json、attempt_01与attempt_02的archive_preflight.json及scientific_inventory.json、summary.json、plugin_install.txt。前置检查只读取目录和有限科学头，不等同完整SHA256复核。该阻塞不是改一个路径即可修复，需要已有平台具备这套7.20工程的合法适配绑定及受控大文件处理；本轮未开发新科学转换算法，也未把7.40配置套用7.20。


### 2026-11-03 取消平台10 GiB文件门槛
按用户要求移除本地文件、目录、ZIP成员及总展开体积的人工大小限制，保留路径越界、符号链接、重复路径等安全检查。文件夹入口定向测试6通过。原CAMx ZIP重新通过公共入口：archive_preflight为CONTINUE，展开字节11919178839；后续仍因已有CAMx转换配置缺失阻止：缺少转换配置：D:\EcoDataEngine\cases\CAMx740_20261020\official\native_complx_emission_probe\config.json。未尝试将其他版本工程配置套入本案例。未修改科学算法，未重跑数值实验。证据output/camx_epa_platform_20261103/attempt_03_no_size_limit。此修改取消平台10 GiB业务门槛，不表示内存/磁盘或Streamlit HTTP上传层无限；大工程可用现有本机路径入口。


## 2026-11-03 通用工程解析与上传依赖自动绑定

本轮范围是把“解析上传工程→绑定其实际文件”从特定机器/特定案例配置中抽出；不是宣称所有环境模型已通用转换或完成融合。新增公共模块src/ecodata/models/project_binding.py，自动准备服务automatic_preparation.prepare_uploads在既有专项bundle处理器未接管时调用。主页面和工具包仍共用此服务；不在页面或实验中另维护工程选择策略。保留既有WRF/CMAQ/CALPUFF专项bundle入口及历史转换器，普通数据继续原流程。

### 可复用开源实现

直接安装并固定f90nml 1.5.0（Apache-2.0），源码https://github.com/marshallward/f90nml；相关JOSS软件论文https://doi.org/10.21105/joss.01474。其负责Fortran语法、数组、引号、注释等解析，不重写解析器。PseudoNetCDF https://github.com/barronh/pseudonetcdf （LGPL-3.0）仍是平台已有CAMx格式工具；本轮没有复制第三方源码或新增科学写出内核。来源及用途见output/project_binding_20261103/sources.json。

### 机制与边界

- 通用惰性文件资源层支持ZIP成员与本机目录，不将GB级工程全部读入内存；对控制文件及实际依赖流式SHA256核验，保留原始参数与引用。
- 根据模型格式级文件角色绑定，不写死研究区、版本号、工程目录或队友配置。CAMx覆盖控制文件中已声明的化学参数、光解、初边界、排放、地表和气象输入，区分空路径、输出以及restart=false的无效重启分支。WRF覆盖明确max_dom下的初始与主边界依赖；WRF重启尚不支持。其他模型保留原驱动，不声称新增通用适配完成。
- 相对路径严格以控制文件目录解析；旧绝对路径只能凭上传集合内至少两级路径后缀的唯一匹配迁移，并记录匹配依据。不打开上传资料之外的主机路径，不按单一文件名猜测。歧义、多个独立控制文件、重复成员、符号链接、越界、缺文件均拒绝。脚本里的内嵌namelist不当独立控制文件，更不执行脚本。
- bound_resource按参数及数组偏移提供真实依赖流，使用前重新核验内容，防止绑定后换文件。参数数组保留f90nml原始start_index元信息；bindings.indexes是解析器列表的零基偏移，不是原生Fortran下标。
- 输出ENGINEERING_BOUND及下载绑定记录，native_ready=false。记录包不含原始大文件，也不冒充可直接运行的模型工程。原生格式、时空网格/单位/版本科学一致性、融合写回及模型执行须另行真实验证。未静默替换旧工程、伪造产品或跳过为PASS。

### 实际验收

通过平台公共入口处理原CAMx EPA首日ZIP：12项文件引用绑定，逐内容SHA256核验，耗时131.98秒。再用最终代码处理原目录，绑定参数/内容指纹与ZIP完全一致；依赖总计11,877,477,180字节。通过bound_resource实际读取Met2D，网格396×246、25个时次、1层，t2单位K，首值292.35498046875；这是数据访问证据，不是气象精度或融合证据。目录绑定、重验该端口并读取数组共20.43秒。

新增及受影响定向测试共24项不同检查通过（tests_accepted.txt的23项，tests_ui_final.txt的1项）。包括两套不同目录案例、CAMx与WRF格式、绝对路径迁移、歧义/缺失/越界、内容变化拒绝、真正读取绑定父资源、ZIP/目录一致、原有准备与融合回归、主页面提交/状态/下载AppTest。AppTest不是人工浏览器验收，本轮未进行人工浏览器操作。保留早期脚本内嵌配置误识别及UI测试缺导入的失败记录，并补针对性测试复测；未运行无关全量回归。

### 交付与未完成

已完成：公共工程解析、上传资源绑定、内容核验及消费接口，CAMx不再为建立工程绑定而读取固定7.40配置。尚未完成：本例7.20气象融合后的原生写回/完整科学契约验收、CAMx原生模拟；CMAQ shell等其他格式的通用绑定不在本轮新增支持范围，不宣称任意模型任意工程即插即用。工程级泛化已与模型格式驱动分离，今后相同已支持格式的其他工程无需逐个写死路径。

证据根目录output/project_binding_20261103：summary.json、sources.json、real_zip与real_directory_final的project_binding.json/execution.json/绑定记录ZIP、测试日志、执行源码executed_source.zip。修改前快照位于同目录automatic_preparation.py、workbench_page.py、pyproject.toml。源码与本地Git协作目录同步，未自动推送GitHub。
复现：.venv\Scripts\python.exe -m pytest tests/test_project_binding.py tests/test_minimal_workbench.py tests/test_workbench_folder_input.py -q -o addopts=''。真实调用为prepare_uploads(local_folder_uploads(工程目录, include_all=True), 'camx', 新输出目录)，ZIP则使用LocalUpload(文件名, Path(文件路径))。


## 2026-11-03 CAMx气温融合与原生写回有界执行（按用户要求停止扩展）

新增公共格式适配src/ecodata/models/native_meteorology.py，project_binding在CAMx工程带有明确NOAA解码气温及站点坐标字段时调用；缺观测时仍返回绑定记录。科学运算复用platform_search.process_current_input、spatial_fusion_task及其现有残差/IDW/校正/随机变异、DAG和质量门，不新增融合内核。几何复用已有CAMx球体LCC网格函数，网格参数从输入头读取，不写死地点/日期/工程路径。主页面增加真实写回与未运行模型的区分信息。

实际执行：复用本轮真实工程绑定，调用公共气象适配及公共数值服务成功；real_run_03耗时31.12秒。此前real_run_01的NumPy整数日期兼容错误、real_run_02的已有_grid返回值数量适配错误已修复，失败记录保留。未反复调参。NOAA气温质量1/5的即时报告转换为K，仅匹配真实原生时刻，不做逐小时填造。8个时刻、站点逐时凸包交集内195个原生格点参与写回，共改变t2的1560个值；其余时空位置、风、三维场和全部其他变量/属性/时间标记逐项回读一致。

公共服务实际评价25张不同图，单源15、多源10，均PASS；固定留站选择队列27个参考值上，最佳单源RMSE=2.0528626218 K、MAE=1.4585345070 K，多源选中图RMSE=1.2116471249 K、MAE=1.0052865191 K。此为小样本选型结果，不能称独立精度收益：五站不足外层独立留点，post_selection为NO_SEPARATE_LOCATION_TEST；源上游同化独立性未知，站点坐标基准采用已披露操作约定。

native_project目录已按通用绑定复制全部11.88GB依赖，副本逐内容哈希核验；CAMx.in依赖改为该副本内相对路径，科学参数保留、输出路径沙箱化。更新的met2d仍为NETCDF3_CLASSIC，原生字段t2/形状/dtype/属性/其余数据均回读检查通过。原始工程未写入。下载EcoData-meteorology-update.zip为更新气象及控制/证据，不含其他大型输入；完整工程在本机native_project。native_writeback_verified=true，native_schema_verified=false，native_ready=false，CAMx模拟未运行；不把格式写回等同全物理平衡或模型原生读取。风的观测高度/旋转契约未解决，保留原值，不称全部气象变量已融合。

用户要求快速结束后未继续追加开发/回归。新增适配模块尚未完成独立定向单元回归、最终页面人工验收和最终版本公共prepare_uploads全链再次执行；本次真实数值处理、原生写回、完整依赖复制及回读有执行记录，不能扩大为整个系统验收。现有project_binding入口接线已修改但尚无本轮最终入口独立验收。不自动启动CAMx或大实验。

产物：output/native_meteorology_20261103/real_run_03/native_project（完整依赖工程）、EcoData-meteorology-update.zip、execution.json、native_validation.json、meteorology/protocol.json、observation_quality.json及search目录全部图/评分/成本/轨迹；此前失败在real_run_01、real_run_02。修改前快照在本轮output根目录。已同步本地Git协作源码，未推送。


### 2026-11-03 GitHub同步
按用户授权，将本地GOLEM退役、移除10GiB业务门槛、公共工程绑定、原生气温融合写回及相关页面/测试/报告提交至codex/latest-platform-20261103分支。原始数据、运行环境、12GB工程副本及output运行产物不纳入Git提交。native_meteorology已真实执行及原生写回回读，但独立定向回归尚未完成，CAMx原生运行未执行；本次同步不改变这些验收边界。


### 2026-11-03 页面上传200MB遗漏修复
此前取消10GiB后台门槛未取消Streamlit默认200MB页面门槛，已承认并修复。.streamlit/config.toml显式maxUploadSize=102400 MB（100 GiB）；Streamlit该选项要求数值，因此不称浏览器无限大小。现有本机文件夹入口不设平台体积门槛。桌面快捷方式原本已指向最新源码；为避免旧服务缓存，保留旧进程不终止，在8502端口启动新服务，并将桌面启动器的健康检查、端口参数和打开网址改到8502。AppTest实际上传控件proto.max_upload_size_mb=102400，页面无异常，8502健康检查200。自动打开浏览器操作被工具策略拒绝，用户可手动访问http://localhost:8502。证据output/desktop_upload_fix_20261103/verification.json，配置及启动器修改前已备份。未启动融合或模型实验。


### 2026-11-03 WRF大小常量残留引用修复
用户页面报MAX_FILE_BYTES导入失败；不是旧目录未覆盖，而是此前取消公共常量时遗漏integrations/existing_air/existing_air_converters/wrf_bundle_pipeline.py中的导入及ZIP/嵌套tar体积检查。已一并移除，不恢复体积门槛；路径、链接和重复检查保留。源码src及integrations检索无剩余常量引用。新增2项回归加文件夹入口6项，共8项通过，记录output/desktop_upload_fix_20261103/tests_wrf_fix.txt。原插件源码已备份，修复同步本地Git协作目录。未运行数据融合或模型模拟。


### 2026-11-03 WRF旧大小常量引用修复
真实页面错误ImportError: MAX_FILE_BYTES定位到integrations/existing_air/existing_air_converters/wrf_bundle_pipeline.py仍导入已移除的MAX_FILE_BYTES/MAX_TOTAL_BYTES。移除其旧导入以及ZIP/嵌套TAR体积拒绝分支，未恢复大小限制，保留路径和链接安全检查。新增三项定向测试实际通过：WRF ZIP分派不再导入失败、非ZIP输入正常交还公共路径、路径越界仍拒绝。src及integrations已无上述常量引用。未运行数据实验。证据output/upload_limit_import_fix_20261103/tests.txt，修改前文件同目录备份。


### 2026-11-03 新WRF 20080205数据公共入口试跑
按用户要求仅使用现有平台，以WRF_20080205_完整工程与多源观测补缺数据_20261103.zip调用prepare_uploads，耗时5.68秒。旧MAX_FILE_BYTES导入错误未再出现；当前WRF专项入口识别原生工程后因没有SST资料拒绝，明确错误：已识别WRF原生工程，但没有识别到本入口可自动处理的SST资料。不会把工程或嵌套资料当作站点文本；请核对待更新变量及支持格式。。这不是证明数据缺失或格式错误，而是当前专项入口仅支持SST用途，尚未把该包站点气象补缺任务分派到适用流程。未融合、未转换、未运行WRF；按用户快速停止要求未继续修代码或重跑。日志output/wrf_20080205_platform_try_20261103/execution.json，原数据未改。


## 2026-11-03 数据任务先行的公共模型入口分派

### 原因与最小公共修复

原WRF入口先进入SST专项处理器，并可能把“没有SST”当成整个WRF请求不可处理；模型配置也可能在实际任务匹配前被读取。现在automatic_preparation.prepare_uploads先调用公共task_dispatch.dispatch_preparation，以实际CF变量身份、单位、时间声明与观测缺失事实匹配任务，再调用既有数值公共服务。科学数据任务与下游模型格式写出分层。请求可选task仅声明type/quantity，不指定算法；页面仍无需选算法。WRF SST专项入口保留其现有sst_update契约，不替换新的独立SST检查逻辑。

新增格式级任务适配src/ecodata/models/task_dispatch.py：按CF标准名/physical_variable与单位匹配，使用平台scientific_schema、UnitEngine、cf-xarray及xarray/HDF按需读取；不从文件名、案例地区或模型编号决定任务。已声明的解码观测表覆盖气温、气压、相对湿度、露点字段；匹配同物理量背景后自然缺失可进入恢复。无匹配/完整观测不制造遮挡，未知单位和无时区静态日期明确拒绝。HDF头不连续时复用h5netcdf按需元数据读取，避免将“64KiB没有完整头”误当文件无效。数据别名按内容指纹去重，不当独立来源。

地理共址报告从训练队列排除（不平均，不按目标值择优）；原始QC非零值不参与拟合；自然缺测输出仅新增estimate和状态，保留原始值、QC及其来源。只使用背景明确时刻，未默认为±1小时平均/借值；后台使用当前公共空间数值任务自己的分块验证、DAG执行、质量门和随机变异，未写另一套算法偏好/赢家/搜索循环。空间坐标按披露的操作约定解释，近地面代表性与上游同化依赖未宣称已证实。

### GitHub可复用成果调查

HydroMT：https://github.com/Deltares/hydromt ，官方架构https://deltares.github.io/hydromt/v1.4.1/dev/architecture/architecture.html ，采用其数据目录/模型组件职责分离思路；Intake：https://github.com/intake/intake ，参考数据读取与处理链分离；WRF-Python：https://github.com/NCAR/wrf-python ，核对原生变量/诊断量读取边界，不为已经规范化的数组重新安装Fortran环境。实际运行继续复用平台已有CF、Pint单位、文件目录及数值引擎，没有复制第三方算法或把架构参考写成库已集成。NCAR QC说明https://www2.mmm.ucar.edu/wrf/users/wrfda/Tutorials/2017_July/docs/WRFDA_obs.pdf 作为解码QC 0/缺失-88的依据；其他非零不自行宣称有效。机器可读来源sources.json。

### 新WRF真实执行

原始WRF_20080205_完整工程与多源观测补缺数据_20261103.zip通过prepare_uploads入口实跑，10.17秒完成TASK_PROCESSED。识别air_temperature/K与明确analysis_time_utc=2008-02-05T12:00:00Z，使用638条非共址的可用报告；同一时刻546条共址报告不当独立训练位置，55条非零QC不进入拟合。40条原有自然缺失均在背景支持域内，输出40个估计值；原值和QC未改，自然缺失无真值，未虚构恢复误差。

共16张图实际评价，单源12、多源4；444个固定验证值上，最佳单源RMSE=3.8935689136 K、MAE=2.8537200584 K，最佳多源RMSE=3.9826277065 K、MAE=3.1125059712 K。因此本次选择原背景来源spatial.source，没有强制融合。194个未参与选型的分块位置上，最终方案RMSE=4.6897656327 K、MAE=3.2343199582 K、Bias=1.8531546524 K。这里只是内部空间分块复核，上游背景可能同化过相关观测，不能称全新独立科研确认；40个自然缺失的真实恢复精度仍未知。

### 页面与定向验收

最终23项定向测试通过：不同物理量驱动（气温/气压）、无关文件名/顺序、内容别名、无自然缺失不造遮挡、缺时间和单位冲突、WRF/CAMx共用分派消费者、SST处理器不垄断其他任务、大小常量回归、原主页面与文件夹入口等。测试不靠固定伪造RMSE证明实跑。主页面AppTest加载同批真实platform_inputs，选择WRF、提交、成功展示并提供“下载任务处理结果”，18.05秒，无异常；这是页面处理函数验收，不冒称人工浏览器操作。

首次页面验收发现公共入口/页面文件已恢复成此前内容，接线未保留；保存当时快照后，仅重新合入dispatch_preparation调用和TASK_PROCESSED展示，保留当前SST新契约，复测通过，并新增生产入口行为测试防止“只测适配模块而未测接线”。初始HDF头探测未匹配的失败也保留，不用它认定资料无效。

### 完成范围及限制

已完成：任务识别和分派不再限定WRF=SST；同一公共任务机制按声明数据与实际问题调用已有能力，WRF/CAMx等目标模型共享，未按工程名称加接口。SST旧路线仍可适用，是否调用由其契约决定。气温/气压匹配变化有机制测试；新WRF气温缺失恢复有完整真实数值证据。

尚未完成：任意原始格式、任意任务的自动理解；多背景用途歧义自动决断；三维平衡初始场生成与WRFDA同化；本次任务结果的原生WRF状态写回及wrf.exe运行。TASK_PROCESSED、native_ready=false，不称模型原生转换完成。本次没有更换测试数据、调参争取融合获胜、扩算法库或启动研究对照。

证据根目录output/task_dispatch_20261103：real_zip_01的execution.json、task_dispatch.json、task-1/contract.json、recovered_observations.csv、search下实际图/评分/成本/轨迹、EcoData-task-products.zip；ui_real_final/verification.json、tests_final.txt、sources.json、summary.json。修改前源文件快照及final_integration_snapshot保留。代码与本地Git协作目录同步，未擅自推送。复现入口prepare_uploads([LocalUpload(文件名,Path(ZIP路径))], 'wrf', 新目录)；测试命令.venv\Scripts\python.exe -m pytest tests/test_data_task_dispatch.py tests/test_upload_limit_import_fix.py tests/test_minimal_workbench.py tests/test_workbench_folder_input.py -q -o addopts=''。


## 2026-11-03 WRF融合落后背景的定点诊断（不调参、不重搜）

复用real_zip_01保存的16张候选图和固定分块划分，仅回放原背景、已选中的最佳融合（affine）、原残差IDW基线三条固定图；6折固定支持指纹完全一致，三条总体RMSE与原记录误差小于1e-10。未改算法、参数、数据掩码或选图规则，未重新搜索，不读取外层测试标签来调整方法。保存逐站预测、逐折诊断及真实DAG边/执行轨迹。

### 已定位的主要原因

1. 验证场景与真实补缺场景明显不同。444个分块验证位置距离最近保留训练站的中位数289.05 km、90分位978.33 km；40个实际自然缺失位置对最终可用638条观测的对应距离中位数16.82 km、90分位66.40 km。当前评分主要衡量跨远距离地理块迁移，对真实缺失附近仍有观测的场景代表性不足。因此“当前分块验证融合落后”不能直接推出“站点对这些自然缺失没有价值”。这些自然缺失没有真值，不能反向宣称局部融合一定能改善。

2. 站点实际进入计算，不存在本次图未读取站点的证据。affine每折从保留站点y和对应背景拟合slope/intercept，残差图也有真实observations父边，完整回放可重复；无未识别退回identity的日期。背景RMSE/MAE/Bias为3.89357/2.85372/+1.27649 K，affine为3.98263/3.11251/+0.46697 K，残差IDW为4.72363/3.40269/+0.28996 K。校正减小整体偏差，但放大部分区域误差，不能将偏差减少当成整体精度提升。

3. 地区偏差不一致，简单校正难以跨块传播。例如第2折（零基fold2）训练观测减背景的均值-1.77648 K，留出地区却为+1.44084 K；残差修正方向错误，该折RMSE由3.38462增至6.65383 K。第0折训练偏差-0.28491 K、留出-5.65395 K，最近训练站中位451.34 km。残差修正与真实所需修正的总体相关为-0.35690；其MSE变化分解E[c²]-2E[c*(truth-bg)]=4.64222-(-2.51062)=+7.15285 K²。说明当前残差传播在此验证设定下确实不可靠，而非单纯搜索器没有挑到一条已评价的更优路线。

4. 当前数据利用和搜索覆盖有限，不能归结为数据本身信息不足。只用精确12 UTC背景的一帧；3143条报告中1239条同一时刻，638条最终有效非共址条件观测。546条共址报告未当独立训练位置，55条非零QC不用于拟合；其他时刻未强行混合。HGT、地形高差、其他气象变量没有作为本任务辅助输入，当前背景在站点处为现有paired-point最近格点采样。站点减网格高程差与背景误差相关约0.299（仅关联，不证明因果）。16张图只有4张多源，5张与已有图预测相同，实际11种验证行为，40次提议后停止。因此这不是全算法库或所有合理融合的失败证明。

### 判断与后续边界

主要可证实问题是评价场景错位，以及当前简单全局/远距离残差处理的局限；并无“站点未参与计算”的证据，也不足以认定站点本身没有信息。现有平台在这套验证目标下选择背景有依据，但不能据此替代对真实近邻补缺场景的判断。后续应先在开发范围预先固定与自然缺失的可用站点距离/覆盖相符的遮挡验证，并保留原地理分块压力测试；这是一项待执行的协议修正，不能按结果追选有利划分。不得通过增加搜索次数、换种子或调权重追求提升。本轮未修改这些协议或开展新效果实验。

诊断证据output/wrf_fusion_diagnosis_20261103：summary.json、frozen_validation_predictions.csv、fold_diagnostics.csv、actual_gap_support.csv、execution_traces.json、diagnose.py、run.log、source_fingerprints.json。以上诊断针对当前验证结果；自然缺失精度、长期泛化及WRF原生模拟收益仍未证明。


## 2026-11-03 其余四模型入口限定审计（只读及有界分派探针）

范围SWAT+、CAMx、CMAQ、CALPUFF。未改算法、搜索、模型入口代码，不运行原生模型、不重做融合实验。只读检查当前源码及已安装entry points，并用明确标为合成的微型合法NetCDF做分派探针。源码当前指纹及全部探针在output/model_entry_audit_20261103，不把测试输入当真实环境实验。

|模型|当前实况|案例/任务限定|判断|
|---|---|---|---|
|SWAT+|源码在库，但本机ecodata.model_converters未注册swat_plus|local_swat_converter.convert及contracts绑定Hubbard W2/HQ单站、站点坐标、固定日期和原工程SHA256，只收日降水/最高最低温；其他变量源代码明确拒绝|仍有，而且未注册是额外接入缺口。不能说页面可选就已接通原生转换|
|CAMx|已安装existing_air_converters|旧通用回退实际是CAMx7.40 NO/PEC排放专项，读取固定config；本机该路径缺失|部分缓解，未完全解决。公共工程绑定及t2原生写回已实现，但其余任务仍可能掉回旧固定入口|
|CMAQ|已安装existing_air_converters|cmaq_reviewed.prepare按整个ZIP大小/哈希认领，固定LAI、2018-07-01、25×1×105×100网格和特定卫星产品；普通回退仍固定NO配置。cmaq原生csh预检查可在公共分派前直接拒绝|限定最明显，尚未通用工程/任务分派|
|CALPUFF|已安装existing_air_converters|有按实际控制内容和引用解析的CALMET/CALPUFF5.8+SURF2.0路线，非固定ZIP；限IPROG0/4等已支持契约，MM4风融合还限特定空间/高度条件；其他输入回退固定受体配置，路径本机缺失|比固定案例绑定好，但仍只覆盖部分任务/格式，非全任务通用|

### 实际验证及公共缺口

1. 用相同微型NetCDF逐模型调用converter_requirements及prepare_uploads。CAMx/CMAQ/CALPUFF普通回退均因各自固定配置路径缺失报错；SWAT+未安装插件，落到默认模型需求，无匹配字段。该探针只说明回退分支行为，不表示这些数据已齐备整个模型输入或本应原生成功。
2. 四模型分别提供相同“不支持的声明任务”探针，均到公共task_dispatch并返回共享的不支持原因，4/4通过；证明普通数据的公共分派不是WRF独有。但不等于四种原生格式都接通。
3. CMAQ额外提供一个合成的未审查csh控制包，公共archive_preflight先拒绝，task_dispatch.json尚不存在。证实模型原生脚本限制仍可能提前挡住数据级任务分派。这既不是应执行上传脚本的理由，也不能用放宽路径/科学安全校验来解决；应区分数据任务和原生模型交付的验收阶段。
4. 公共preparation_table当前仍用converter.requirements替换模型需求，而这些requirements常依赖固定机器config；automatic_preparation中上传工程又须匹配pinned engineering SHA，否则拒绝。新的数据级公共任务在识别成功时可先返回，不代表后续原生适配已摆脱旧分支。
5. 合理限制与工程耦合要区分：原生格式、模型版本、单位/网格、状态依赖必须检查，不能简单删掉。固定某流域、日期、站点、完整ZIP哈希及他人绝对路径则不应作为模型所有任务的公共门槛。历史专项路线可保留，但未匹配时应交由任务能力分派或给出具体缺口，不能冒充全部模型能力。

建议优先处理CMAQ的全包匹配及分派前拦截、SWAT+的注册与单流域契约；随后补CAMx/CALPUFF其他任务的合法资源契约。修复应继续复用公共任务和工程绑定机制，不为每个案例新增入口。本轮仅完成审计，不宣称上述剩余问题已修复。

证据：live_entry_probes.json、shared_dispatch_and_swat_contract.json、cmaq_preflight_probe/execution.json及archive_preflight.json、entry_matrix.csv/json、source_fingerprints.json。入口级合成探针4种回退+4种公共分派+1种CMAQ前置拦截；未执行科学数值效果测试，未新增大量测试或启动原生模型。


## 2026-11-03 四模型入口泛化接入与有界验收

### 原问题与本次修改

承接前节审计，范围为SWAT+、CAMx、CMAQ、CALPUFF。原普通回退依赖固定开发机config；SWAT+未注册且默认Hubbard工程；CMAQ整包哈希专项在数据分派前可能拦截其他合法任务。本轮未更改融合、评分、变异或数学内核，未运行原生求解器或开展效果调参。

公共链路为页面/工具包 → automatic_preparation.prepare_uploads → 安全预检查 → component_service.prepare_components（明确原生组件任务）/现有task_dispatch（真实数据处理任务）→ 原生格式严格检查与既有工程驱动/已有一般数据准备。安全路径检查共同执行；未审查原生脚本不再阻挡无需执行该脚本的数据任务，原生工程交付仍严格检查，上传程序从不执行。

新增公共ecodata.model-task.v1契约，声明目标模型、科学任务、输入资源和所需单位/时间/网格等configuration；引用只在本次上传资料中解析，可带内容指纹。该契约不接受算法赢家、可执行程序或开发机配置路径。普通用户页面布局及模型选项不变，可在ZIP/本地资料中带任务JSON；工具包也可通过task参数传入同一契约。资料不足或用途有歧义仍明确拒绝，不根据文件名猜科学含义。完整、无歧义且元数据齐备的单源SWAT+逐日天气有自动识别路径；本轮真实NASA数据采用显式合法科学配置，并非证明任意天气文件都能免声明识别。

实际复用ecodata.round1.convert及既有23项写出任务，非新增23个算法：SWAT+天气、土壤、土地利用、WGN、HRU、含水层、连接、空间土地利用、管理日历；CAMx排放、初始、边界及字段更新；CMAQ排放、边界、官方剖面及字段更新；CALPUFF地面气象、降水、探空、地理、受体、点源。模型版本/机制/物种、单位和支持域等仍由原写出器检查，未放宽科学契约。entry_matrix.csv逐项区分真实执行、合成执行及仅绑定未重执行，不能把23项全部说成本轮真实验收。

四模型requirements不再装载旧机器固定case，SWAT+插件已安装注册，setup_handoff.bat增加内部两转换包安装以保留接入。旧专项实现保留，CMAQ/CALPUFF已认领的历史工程路线仍保留；未匹配输入不再默认调用不相关NO/受体/Hubbard工程。缺原生用途契约时保留已准备数据并返回明确缺项，不能宣称完整转换成功。

原生NetCDF字段更新复用既有native_netcdf_update，不重写CAMx/CMAQ内核：目标来自本次上传，检查目标SHA及产品native_reference_sha256、维度顺序、形状和单位；只修改声明变量/掩码，保持其他数据和元数据。没有原生目标或来源绑定证据时拒绝，不能按相同形状猜同网格/时间。

### GitHub成果及证据边界

查阅HydroMT的模型组件与驱动分层（https://github.com/Deltares/hydromt），沿用“公共数据与任务机制＋模型原生组件”的已有平台结构；未声称安装或移植HydroMT模型插件。独立科学格式读取继续复用已安装PseudoNetCDF（https://github.com/barronh/pseudonetcdf，LGPL-3.0），原生写出复用项目已有成熟实现。SWAT+天气索引/站点文件与官方Editor说明核对（https://swat-model.github.io/swatplus-editor-documentation/edit-inputs/weather/）。没有复制新的第三方核心代码，也没有把GitHub项目存在等同于本平台已实现全部能力。

### 真实运行与泛化行为验收

|模型|实际输入及任务|本轮结果|明确范围|
|---|---|---|---|
|SWAT+|NASA POWER 365日天气；官方13土壤/52层记录|两条公共入口真实导出；土壤独立shlex/Decimal回读与官方内容一致|不是固定Hubbard站点/日期；天气、土壤组件，不是完整流域工程或SWAT+运行|
|CAMx|已有EPA首日真实Met2D及此前实际处理产品|25×1×246×396的t2原生写回，实际改变1560值；其他变量/元数据/原文件保持，约7.87秒|复用旧真实融合产物验收写回；本轮没有重新证明精度提高|
|CMAQ|真实EDGAR区域排放；官方BCON静态剖面值|两条公共入口真实导出及回读；EDGAR面积/摩尔换算由独立数值断言检查|官方静态剖面不是动态真实化学场；不等于CMAQ原生运行|
|CALPUFF|NOAA实际地面气象；NOAA USCRN实际降水|两条公共入口真实导出；降水独立读回并与原始NOAA记录逐值/时间核对|气象组件，非完整CALMET/CALPUFF工作流运行|

另有CAMx排放LATLON及初始场LCC两项明确标注的合成格式验收，不能算真实排放验证。网格位置受控改变会进入相应原生输出，不再由旧固定区域覆盖；既有内核定向测试还覆盖其他时间、单位、网格和非法条件。泛化含义是同一任务/格式契约可接受不同合法资料与参数，不是所有未知格式、全部版本和任意用途无需元数据即可运行。

四模型均通过当前主页面AppTest从本地目录提交、调用公共prepare_uploads、生成真实原生组件包及下载按钮；与直接工具包调用比较所有科学数组和元数据一致。CMAQ I/O API的CDATE/CTIME/WDATE/WTIME仅是生成/写入时钟，单独排除后科学内容一致。曾出现的哈希差异及测试比较数组字典错误均保留失败日志，修正的是验证方法，没有更改科学产品追求通过。AppTest不等于人工浏览器操作，本轮未做人工浏览器验收。

最终定向测试147通过、0失败、0跳过（9.00秒）；没有跑无关全量回归。涉及公共组件入口、数据级任务分派、工程绑定、CAMx排放、CALMET降水、SWAT+土壤、CMAQ排放。包括任务/模型冲突、缺输入、越界、源指纹、mode冲突、程序注入拒绝、原生写回缺目标绑定、未审查脚本的数据/原生阶段区别及四模型页面/工具包一致性。初次pytest因旧.pytest_tmp目录权限错误未进入相关用例，随后使用新验收目录，不删除原目录；失败记录保留。

### 状态与尚未证明部分

四个模型公共任务入口：PASS（已列契约范围）；移除旧案例作为默认门槛：PASS；真实组件与输出回读：PASS（上述7项真实任务）；合成格式变化：PASS（2项）；页面处理函数/工具包共用：PASS；全部23任务本轮逐一真实重跑：未运行；人工浏览器验收：未验收；任意完整工程自动解析与绑定：未全面支持；原生模型读取及求解运行：未运行；融合精度提高：本轮未评价。

尤其不能把原生组件包当成完整模型工程：SWAT+多工程根/多站点关联仍需明确目标；CMAQ任意csh/MPI工程不执行未知脚本；CALPUFF既有驱动仍受已支持版本、SURF/IPROG等条件约束；CAMx/CMAQ字段写回不认证跨变量平衡。普通资料缺乏原生任务或目标工程时会明确失败，不回退到他人的固定案例。本轮没有消除这些科学/格式边界，也没有宣称四个模型所有任务都已经通用。

证据目录output/four_model_integration_20261103：before修改前快照、executed_source执行源码、source_fingerprints.json、entry_matrix.csv/json、summary.json、real_acceptance各案例execution.json/handoff/原生产品包、camx_real_update/execution.json及产品、verified_tests.txt与全部失败日志。源码及报告同步本地EcoDataEngine-GitHub工作区，未擅自推送远端。历史数据和实验结果未删除或覆盖。

复现定向测试：.venv\Scripts\python.exe -m pytest tests/test_public_model_components.py tests/test_data_task_dispatch.py tests/test_project_binding.py tests/test_camx_emissions.py tests/test_calmet_precipitation.py tests/test_swat_plus_soil.py tests/test_cmaq_gridded_emissions.py -o addopts='' -q --basetemp=output/新测试目录。实际请求来自本次上传，不在工具包或页面分别维护case选择策略。

2026-11-04收尾核验：补充CAMx/CMAQ公共掩码写回测试，发现写出器要求传入完整合法目标数组，已在公共适配层按明确掩码与原生原值组合，不改底层写出内核。独立回读同时验证掩码内采用真实产品、掩码外及其他变量保持。合成默认CDF5在本机原生库无法r+更新，失败日志保留，改用实际CAMx同类NETCDF3_CLASSIC验证已支持格式；不能宣称全部NetCDF编码都可更新。最终release_tests.txt为149通过、0失败、0跳过，9.30秒，取代上面的147项中间统计。原23项任务目录未扩大。


## 2026-11-04 GitHub工程解析复用核查与公共依赖盘点补充

### 搜索结论及本轮边界

用户要求查找现有方法复用，目标为完整工程的通用解读。当前找到的成熟成果均有模型/语法边界，不能据此承诺任何完整工程均可自动理解。工程完整是必要条件，不能代替语法、版本及运行时条件。此项“任意完整工程均适用”未达到，不写成PASS。

核对上游源码并保存原文、许可及版本证据：

|上游|实际发现|复用方式与限制|
|---|---|---|
|swat-model/swatplus-editor，6ce225dc8ac58989d49c9ebaf8b7b3846e4b2b09，src/api/fileio/config.py|File_cio.read明确raise NotImplementedError；write支持多种配置/NetCDF天气|不能把官方Editor当成现成任意工程读取器。沿用平台既有weather_dependencies，并依官方readcio_read的按位置读入规则补充CIO直接引用，支持已声明rev.60/61/62的已知布局|
|swat-model/swatplus/src/readcio_read.f90|各组按位置读取，标签文本不作为语义判据；有天气路径/输出路径|公共盘点不再依赖climate等固定标签字符串来定位CIO各组；未知版本/字段数量不符保留UNKNOWN。未复制其数学内核|
|Pavinberg/calpy，5c0571907d3ae34544d89b0949a8b9b6a7b87273，MIT|load_calpuff要求22组，部分组特殊处理；类含默认网格、日期、运行器与绘图库|不导入整个包，避免套用默认项目和额外执行依赖；继续复用平台已有control_values及dependency_plan，后者负责激活输入而非另建第二张规则表|
|USEPA/CMAQ，树版本13c473c8e834ca6ee5fcff468e16b015362e83be|官方脚本依赖source、变量、条件、日期循环和命令替换，非静态文件目录|只读识别I/O API输入声明及可直接绑定的字面路径，运行时变量/激活条件保留UNKNOWN，未伪造静态求值结果|
|marshallward/f90nml|已安装使用的Fortran namelist读取器|现有CAMx/WRF绑定继续复用；它不能解释SWAT+表或C-shell脚本，不将其宣传为通吃解析器|

来源：https://github.com/swat-model/swatplus-editor/blob/6ce225dc8ac58989d49c9ebaf8b7b3846e4b2b09/src/api/fileio/config.py ；https://github.com/swat-model/swatplus/blob/main/src/readcio_read.f90 ；https://github.com/Pavinberg/calpy/tree/5c0571907d3ae34544d89b0949a8b9b6a7b87273 ；https://github.com/USEPA/CMAQ/tree/13c473c8e834ca6ee5fcff468e16b015362e83be/CCTM/scripts 。本轮未新增外部大型依赖、未声称calpy已作为新生产后端集成。

### 实际补充的公共能力

新增models/project_inventory.py复用ProjectFiles安全资源解析、内容SHA、SWAT既有天气依赖检查、CALPUFF既有控制读取与激活输入闭包。多个SWAT工程根分别盘点，不替用户选出“最佳”工程，不混合它们的站点和输入；索引到观测文件的真实父子引用保留。缺文件不因目录看似完整而放过。CMAQ脚本从不执行，不读取宿主环境变量来猜本次输入；引用有行号，分支激活未知单独记录。

接到公共prepare_uploads的只读工程兜底，并给未通过原生控制预检查但可以安全盘点的工程返回ENGINEERING_INSPECTED、native_ready=false。安全路径检查仍在最先；未知脚本不被执行。既有组件导出、数据任务、已支持原生处理优先级保留；CAMx/WRF现有工程绑定不改。本轮没有把盘点包包装成融合成果或模型可运行包。页面增加原状态展示分支和下载文案，未增加控件。

### 真实现有工程与行为验证

1. Humaya真实原始SWAT+整包，经公共prepare_uploads读取四个工程根：共1348条直接/天气索引引用全部绑定，各工程仍有“深层数据库/版本未原生验证”提示。返回ENGINEERING_INSPECTED，15.14秒；不读取或复制整个压缩包到内存，不选一个方案覆盖其他方案。没有运行模型或改原工程。
2. 官方CALMET5.8.4与CALPUFF5.8.4两个原ZIP一起上传，公共入口识别两个控制工程，复用dependency_plan绑定9条已有激活输入引用；其余原生兼容性未认证，返回ENGINEERING_INSPECTED，约0.084秒。
3. 本地官方CMAQ首日运行脚本作为真实语法小样本（本项仅脚本，非完整数据工程验收），识别55条输入声明，当前均需运行时变量展开，0条可确认为已激活绑定；返回ENGINEERING_INSPECTED，约0.048秒。不能据此说完整CMAQ工程已自动绑定。本轮没有扫描全部GB级CMAQ数组，也没有运行csh。

51项定向测试通过、0失败、0跳过，8.16秒。包含按位置读取与自定义组标签、索引子依赖、缺文件、来源内容变化、多工程分离、未知版本、文件名/顺序无关、引用越界、CMAQ引用中的引号/#处理、未知变量与无脚本执行、页面警告与下载；同时回归已有工程绑定和四模型组件入口。合成边界测试不作为真实环境精度证据。页面处理函数AppTest已验，人工浏览器未验。未重跑研究实验或无关全量测试。

### 明确剩余项

“通用工程解析”的公共资源与依赖表示得到补充，已覆盖的格式不再绑定某流域、日期或ZIP名称；但全部模型深层跨表语义、未知版本、任意C-shell程序的实际路径及所有完整工程仍未解决。CMAQ要获得可证明的实际闭包，需要可信运行环境导出的明确输入清单，或覆盖相应脚本语义的求值器；不能执行不受信任上传脚本来凑成功，也不能把大模型猜测当确定绑定。SWAT+完整深层引用与所有版本、CALPUFF未支持驱动仍需各原生格式能力。当前有据可查的结论是部分泛化解析/绑定与明确未知诊断，非“任意完整工程都已适用”。

证据：output/project_generalisation_20261104内before、上游源码/许可/API树及upstream_fetch.json、source_fingerprints.json、executed_source、real_cases与public_real的project_inventory.json/产品包/summary、tests_final.txt、summary.json。原数据/旧结果未改，修改同步本地Git协作目录，未推送。复现：.venv\Scripts\python.exe -m pytest tests/test_project_inventory.py tests/test_project_binding.py tests/test_public_model_components.py -o addopts='' -q --basetemp=output/新目录。


## 2026-11-04 统一工程上下文、自动绑定与原生写回验收（engineering_writeback_v2）

### 实际实现与复用边界

在既有EngineeringProject、ProjectFiles、原生格式写出器、bind_project与prepare_uploads上补工程产物挂接，不建立新算法、模型内核或优化器。公共component_service调用新增engineering/native.py：使用源资源SHA256、实际变量、网格/时间支持签名、原生头和单位绑定，而不是ZIP名、固定流域或开发机路径。上下文中的资源、变量、端口映射和控制引用参与写回前验证；写出复制工程，保留非目标成员及原始文件，产品ZIP逐成员SHA及CRC读回。

新增WRF/CAMx/CMAQ的update_project操作；实际从原生资源提取并合法处理、保留溯源的产品可自动触发。旧v1签名按原算法校验，当前v2使用规范数组序列化；未知版本拒绝。错误模型族、单位、网格、时间、原生范围/类别、错控制端口、来源变化均拒绝；掩码外/非有限位置保留原值，质量门分别记录，不以格式通过证明环境准确性。WRF显式域尺寸和起始日期与实际目标核对；未声明默认值UNKNOWN。没有控制文件的单资源写回不声称完整工程。

SWAT+复用现有attach_weather/attach_inputs：按真实file.cio工程根、站名、站表引用、原生站点元数据挂接；官方站表LON与LONG别名统一，不改科学算法。CALPUFF复用控制dependency_plan及read_surface，通过实际激活SRFDAT、站序、时区和时间轴挂接，声明CALMET的METDAT下游依赖NOT_REGENERATED。当前桥接不是所有CALPUFF驱动。移除的旧大小限制限于本轮天气/表附件路径，不能宣称所有历史UI/扫描器都无限制。

GitHub一手复用核查沿用上一节已有取证：swat-model/swatplus-editor的simulation.py及license、USEPA/CMAQ目录、Pavinberg/calpy接口和许可（output/project_generalisation_20261104/upstream_fetch.json）。没有将上游目录存在当成可运行证据，也没有复制第三方模型数学内核。

### 真实执行结果及独立回读

|真实案例|结果|实际证据与限制|
|---|---|---|
|WRF20230331初始场d01/d02|PROJECT_UPDATED|真实T2做单位往返再写回，最终约31.29/30.07秒；两个域属于同一工程，不计独立工程数。控制端口、显式网格/日期重新核对通过；字节相同是预期往返，不是融合提高。更早冷启动耗时另留日志，不宣称总成本优势。|
|CAMx EPA真实Met2D|PROJECT_UPDATED|复用此前实际融合产品；原生t2实际1560值变化，其他内容保持；最终约6.22秒。未重新做融合精度实验。|
|CMAQ真实MCIP METCRO2D|PROJECT_UPDATED|TEMP2单位往返及读回，约1.66秒；这是资源级更新，未解决任意csh动态CCTM闭包。|
|Humaya SWAT+ Gauge|PROJECT_UPDATED|真实8036日PCP/TMP，6项新增/修改，约1.72秒；独立读回日期、值差0。天气往返不是新预测收益。|
|官方CALMET/CALPUFF5.8.4|PROJECT_UPDATED|168小时5站SRFDAT，约1.48秒；独立读回值、缺失掩码、站序、时区一致。未生成新的METDAT，未运行求解器。|
|Humaya Daymet固定首站|BLOCKED|原PCP六处-99（闰年day366），另有不完整2002年末行；保留原始失败，不换站、不填造，未完成工程写回。|

新包状态及现有工程状态分开记录。上面PROJECT_UPDATED均native_ready=false、complete_model_input=false、native_model_executed=false；CMAQ、CAMx资源级产物不能算整个模拟工程已通过。最终新增原生头/控制约束通过115项定向回归，并对真实已有输入重新检查头、签名、WRF实际控制绑定；没有为每次保护性检查反复重跑全部大型数组。

### 用户新增“轻量案例包_工程与观测.zip”最终验收

原包495311011字节，SHA256=2e3559f76fedadf442c9df26881404718704fbfa1e3597d582743933c27187d1；解出6个直接成员到新验收目录，逐文件SHA、内层清单和原包前后相同证据保留。包内程序未执行。公共prepare_uploads实际结果：

|上传范围|实际结果|原因及阶段|
|---|---|---|
|外层总包|BLOCKED|当前公共资源层不递归展开工程ZIP。首次误入旧WRF本机默认配置路径；本轮修复为明确要求分别上传内层工程ZIP与观测，修复后0.54秒明确拒绝，旧失败日志保留。不是自动处理通过。|
|WRF内层ZIP＋NOAA观测|ENGINEERING_BOUND|57.78秒，实际控制和资源绑定；tasks为空、fusion NOT_EXECUTED，未融合写回。|
|CALPUFF内层工程ZIP|CONVERSION_FAILED|53.56秒，真实SURF产物和检查已产生，但原有原生流水线依赖的本机CALMET/CALPUFF程序路径不存在；不能宣称完整运行。|
|Robit_demo_4.0.zip＋NASA POWER|BLOCKED|有QGIS/SQLite、地理资料和天气，未提供当前路径可直接挂接的file.cio。不能从GIS工程生成TxtInOut的现有缺口未解决；未把未知文件强行当融合输入。|
|CAMx目录EPA臭氧ZIP|STOPPED/未完成|其中只有观测CSV，无CAMx工程；公共尝试180秒墙钟到限停止，日志和停止记录保留，没有结果或精度成功声明。|

因此这个新增总包没有完整跑通；它揭示包装、GIS工程导出、旧原生运行时依赖及缺CAMx工程四类边界，不能用之前格式往返通过替代。本轮不绕过缺失依赖、不执行上传未知脚本、不挑换数据凑通过。

### 测试、保护与交付

最终定向回归115通过、0失败、117条警告，13.34秒；没有跑无关全量或研究搜索。覆盖公共入口、工程绑定/库存、实际父子资源、状态/来源变化、错误模型/单位/支持域、掩码、原生范围、SWAT站表及天气、当前页面处理和下载。旧测试调用已移除页面函数的失败被保留，迁到当前页面公共入口，不跳过原有科学断言；独立验证器曾假定.tem，改为读取真实站表.tmp引用，未改变正确产品。

仅新产物/复制工程被写入；用户原始数据、历史结果不删除、不覆盖。证据集中output/engineering_writeback_20261104：before修改前源码、executed_source执行源码、source_fingerprints.json、acceptance_matrix.csv、summary.json、real_final/text_final、final_context_acceptance、独立数值读回JSON、tests_user_pack.txt及user_light_case完整清单/失败/停止记录。当前AppTest不等于人工浏览器，人工浏览器未验收。本轮未进行精度提高评价、任意完整工程认证或模型求解。

复现：.venv\Scripts\python.exe -m pytest tests/test_engineering_writeback.py tests/test_public_model_components.py tests/test_project_inventory.py tests/test_project_binding.py tests/test_swat_weather_project.py tests/test_swat_inputs_project.py tests/test_data_task_dispatch.py -o addopts='' -q --basetemp=output/新的定向测试目录。

结论：已实现并真实验证已列原生资源/天气契约的统一上下文—绑定—写回；任意完整工程全链路仍未完成。具体未完成项如上，不能称全面泛化。修改同步本地Git协作工作区，未推送远端；没有删除任何新增案例数据。


## 2026-11-04 三项工程集成缺口修复与轻量案例包最终验收（engineering_three_fixes_v1）

本轮范围为WRF绑定后处理衔接、CALPUFF转换/求解分离、SWAT+编辑工程导出。未新增科学算法，未改融合内核、评分、变异或选择策略，未重跑历史研究对照，未启动模型求解。主页面控件和布局保持。所有公开处理仍从prepare_uploads进入。

### GitHub复用与实际调用

- SWAT+直接复用官方swat-model/swatplus-editor的SetupProject、Swat2012WeatherImport/WeatherImport、WgnImport及WriteFiles。固定v4.0.0，commit `3606ac4c3b54585072a557382b8e3b60b44dd354`；目标文本契约rev62.0.0。官方源码/许可在external/swatplus-editor；上游文件SHA、网址和环境版本在upstream_sources.json。隔离环境.venv-swat-editor使用peewee3.18.3；复建命令为powershell -File integrations/swat_editor/setup.ps1。不在上传时联网安装或执行上传脚本。
- WRF核查NCAR/wrf-python及官方WRF工具资料，未为此另装/改写诊断框架。复用现有xarray/Scipy配对坐标插值（公共spatial_fusion_task.sample）、既有fuse_temperature与原生字段写出器。xarray-regrid仅支持其已有规则网格契约，未将其冒充原生WRF曲线网格直接重网格。
- CALPUFF核查Pavinberg/calpy等已有控制/运行管理，继续使用平台已有CALMET SURF导出、独立读取器、数据处理及依赖规划；不新建另一套转换算法，不把第三方运行脚本当自动上传后端。相关一手链接： https://github.com/swat-model/swatplus-editor 、https://github.com/Pavinberg/calpy 、https://github.com/pydata/xarray 、https://docs.xarray.dev/en/stable/user-guide/interpolation.html 。

### 具体修复与公共调用链

|原问题|修改位置|现在实际行为|验收|
|---|---|---|---|
|WRF识别后直接ENGINEERING_BOUND，不组织处理|automatic_preparation.py、project_binding.py、新wrf_meteorology.py；native_meteorology.py提取原有公共温度调用|原生控制绑定优先于旧专项补充包入口；从真实初始场读取变量/网格/时间，识别实际观测及OISST，按适用条件调用已有处理，再将真实产品送公共component_service更新工程|真实SST更新通过；合成三站T2真实数值入口及写回通过；真实包单站T2明确不适用|
|CALPUFF把本机求解器路径当转换硬门槛|integrations/existing_air/.../calpuff_native_bundle.py、conversion_bridge.py|普通转换不检查/启动CALMET或CALPUFF；直接调用已安装SURF写出器并回读、复制实际控制/输入。仅显式run_models阶段保留求解器要求；桥接遵守转换器native_ready=false|同一真实包转换通过，未要求旧开发机程序路径|
|SWAT+只有TxtInOut挂接，不能读GIS/Editor数据库|新swat_editor_project.py与独立worker、公共automatic_preparation.py|识别数据库schema并解析实际资源引用；在副本调用官方GIS初始化、天气及WGN导入、文本导出；原始工程不改。先校验天气文件名/外键/月表/路径，再调用上游代码；生成后检查CIO及天气依赖|原包生成file.cio及WGN；68条直接引用全部绑定|

任务绑定不是按Robit、ZIP名、站点名或固定开发机路径选择。SWAT+WGN通过字段及月份外键识别同工程下唯一的统计站/月表；显式配置的WGN数据库按实际上传资源绑定并校验月份契约。CSV列顺序只做可追溯规范化，不改系数。未知版本、缺资源、多个工程根/配对歧义、路径越界均明确拒绝。Editor副本中的输出/天气/数据库路径导出后还原相对路径，便于移动；安装环境和可信上游代码独立于科学数据。

### 真实轻量包验收（不是科研效果实验）

继续使用用户原495311011字节轻量包及此前已解出的内层资源，未下载替代实验数据，原6份输入SHA均未变化。总包递归拆包仍未实现；本轮分别向同一个公共入口上传其内层工程及对应资料，不把此步骤写成总包直接上传成功。

1. **WRF：PROJECT_UPDATED。** 输入同一WRF工程ZIP及NOAA Memphis观测。公共入口实际读取OISST原始有效范围/填值、K单位、原生日期、海陆/冰及可用湖泊掩码，将同日海温分析采样到原生初始网格，通过既有字段写出器改1640个SST值。约124.79秒；独立比较所有其他变量、维度、类型和属性保持，下载包CRC通过；最大SST变化约6.82953K，仅表示数值改动。d02没有有效海温交集，保持原值。

   时间约定明确是**回顾性日分析初始化**：在OISST实际UTC覆盖日内用于初始SST，不将每日12点标签伪装为整日真实瞬时扫描，不用于证明实时预报。多日期派生子集携带单日覆盖元数据时不猜改，改用包内覆盖区间明确的原始逐日分析；同日重叠冲突拒绝。LANDMASK/SEAICE、明确LAKEMASK（存在时）及OISST无效覆盖处保留原值，未推断缺失的湖泊边界。

   **这不是单站气温融合成功。** NOAA TMP按官方Global Hourly字段及QA解码（与ISD-Lite不同），实际同期只有1站；现有留站评价至少3个同期位置，因此T2保持不变并记录NOT_APPLICABLE，不把单站偏差推广到整个WRF域。T2与SST绝不混为同一变量；气温的高度/代表性等仍有适用边界。SST更新、自动工程衔接通过，不等于精度或动态平衡提高。

2. **CALPUFF：CONVERTED，native_ready=false。** 同一真实工程包先经过既有风分量处理，再把本次产品导出为SURF输入，约52.75秒，168小时×5站。最终版转换函数另复用已生成真实处理产品回读验证，未重复搜索。实际8字段数值、站序、时区和缺失掩码与本次产品一致；没有运行求解器，也不依赖旧D:/EcoDataEngine程序目录。

   输出仅为已处理原生输入及工程依赖。新的CALMET METDAT未生成；若将来要求完整运行CALPUFF，需单独运行CALMET并验证结果。这是运行阶段的合理依赖，不再阻止普通融合/格式转换。未宣称模型浓度运行或融合精度提高。

3. **SWAT+：PROJECT_UPDATED。** 同一Robit工程ZIP及NASA POWER文件经公共入口实际调用官方Editor。导出69份原生文本文件（含file.cio、天气及weather-wgn.cli），68条直接引用全部绑定、无缺失引用或天气关联错误，约5.84秒；未运行SWAT+。

   **纠正本轮中间判断：** 起初看到项目库WGEN_User/WGEN_User_mon、配套库wgn/wgn_mon均0行，且未生成weather-wgn.cli，误以为原包缺WGN资料。随后完整核查发现原包Wgn目录已有两份真实CSV；问题实际是未配置/未导入，不是原始资料缺失。已补公共字段识别及官方two_file导入，天气站实际链接bdrwgn。独立比较原CSV的12个月×14项统计与原生输出，最大差约0.000004986，符合官方五位小数输出；没有用零、其他流域或新造统计替代。早期失败、部分产物及初判记录全部保留，最终结论以real_swat_wgn和independent_readback_final.json为准。

   原NASA POWER辅助文件保留，但本轮未将其与原天气强行融合；其高程/代表性及融合任务不在这个编辑工程导出修复中解决。官方GIS初始化使用官方默认参数，记录official_gis_defaults_used，不声称还原作者未保存的标定参数。当前核验Editor4.0.0/QSWAT+4.x的已列数据库/天气契约；GWFlow等尚缺完整绑定条件时明确拒绝，不宣称所有版本自动可用。

### 功能回归、页面和信息边界

最终tests_signedoff.txt：**141通过、0失败、135 warnings，17.00秒**。包含端口写回、原始输入保护、格式/单位/时间/范围错误、WGN真实月份支持及空表拒绝、CSV外键/无关分支/改名、工程路径越界、未知版本、模型运行与转换分离、页面/工具包一致性等。未执行无关全量回归，未降低物理门槛。

真实Robit额外通过当前主页面AppTest提交并产生可下载产品，与直接工具包调用比较全部69份原生文件，排除官方第一行生成时间注释后内容一致，实际下载控件可见。小型WRF同一请求经页面和工具包的原生输出及计划一致。AppTest不是人工浏览器验收，本轮没有声称做了人工浏览器上传。

合成三站T2另做有界链路检查（不是环境精度证据）：通过公共输入→共享处理→真实数值搜索→产品→原生写回。固定种子11，15提议上限、20秒，公共记账实际32次算法调用（搜索30、部署2），9个候选记录，0.1秒左右服务计时；不报告合成RMSE为科学成果。首次测试误设10次总调用，小于服务既有96次部署预留，未能执行；根据明确资源契约校正到144次上限（至多48次搜索空间调用余量），未改优化器或按误差追调，失败及校正协议保留。另有一次测试输出目录复用错误日志保留。

本轮修复了时间字符数组S19/S1解析与写回目录创建两个实际接口错误。旧基础页面测试默认“未装转换器”，在当前已安装转换器环境原本也失败；已保存旧入口复现CONVERSION_FAILED证据，在这些无转换器测试显式隔离发现接口，原有内容、CRC、页面及异常断言未删除。

### 交付、保护及复现

证据集中output/three_integration_fixes_20261104：before修改前快照、executed_source及source_fingerprints.json、upstream_sources.json、各真实receipt/产品/质量记录、独立读回、page_real_swat.json、acceptance_matrix.csv、summary.json、所有失败日志及合成协议。已验证原始输入未变，未删除数据、未覆盖历史研究结果；没有新增算法、训练模型或启动正式研究对照。

桌面当前主运行目录仍为D:/codex/EcoDataEngine，服务按现有launch_platform.ps1重启至127.0.0.1:8502，health=200；修改同步本地EcoDataEngine-GitHub工作区并核对内容，未推送远端。源码和环境安装成本未完整单独计时；上面的秒数是各处理入口墙钟，不声称总成本优势。

复现定向测试：.venv\Scripts\python.exe -m pytest tests/test_three_engineering_fixes.py tests/test_engineering_writeback.py tests/test_project_binding.py tests/test_project_inventory.py tests/test_public_model_components.py tests/test_data_task_dispatch.py tests/test_swat_weather_project.py tests/test_swat_inputs_project.py tests/test_minimal_workbench.py tests/test_workbench_folder_input.py -o addopts='' -q --basetemp=output/新的验收目录。

最终判定：三个指定工程集成方向已实现并有上述边界内的真实证据；WRF真实单站T2融合不适用、原生模型求解未运行、精度增益未评价、外层总包递归仍未实现。这些限制不因工程功能通过而改称完成。


## 2026-11-04 五模型通用性只读复核：专项通过不等于整体通用

本次仅核对当前公共入口及实际调用，不运行新实验、不修改科学代码。验收口径是：在已声明科学/格式契约内，普通用户更换同类工程与数据后，不必增加任务专用入口，即可由平台识别任务、组织适用处理并写回；不是要求无视模型版本或未知科学含义。

按整个模型的“数据识别→任务推导→融合组织→原生写回”口径，当前五模型均不能标记整体通用已完成（0/5）；已支持格式/任务的部分通用能力仍成立，不能反过来说所有工作都没做。

|模型|确有的通用部分|尚未贯通或限定点|整体判断|
|---|---|---|---|
|SWAT+|按真实数据库/字段识别的官方GIS与天气/WGN导出；多类组件写出及部分工程挂接|普通上传自动任务以完整天气识别及编辑工程导出为主；其他组件通常依赖明确任务声明，未统一自动推导所有合法处理/融合任务|部分建立；工程导出层较完整|
|WRF|原生初始/边界绑定、来源和支持域校验、公共字段写回|自动处理明确分支仍为T2气温与OISST/SST，且不覆盖全部工程模式；不能从SST路径成功推断新变量自动适配|部分建立；仍有专项路径|
|CAMx|namelist输入引用绑定、资源复制、多个组件与通用字段更新|自动气象路径只消费唯一met2d_grid内t2及现有观测契约；其他变量/目标仍需相应任务绑定|部分建立；未统一所有处理任务|
|CMAQ|排放/边界等组件及带溯源的原生NetCDF字段更新|C-shell的运行时变量、条件、循环与source仍UNKNOWN；缺完整工程实际激活依赖解析及后续自动任务组织|部分基础可用；完整工程链路未建立|
|CALPUFF|支持格式内的控制依赖解析、SURF等组件、转换与运行分离|完整自动处理受5.8格式、IFORMS/IRTYPE/IPROG等已审查条件约束；当前融合组织偏地面气象/风分量专项，未形成所有受支持任务的统一推导|部分建立；格式/任务有边界|

判断依据为automatic_preparation.prepare_uploads的调用次序、component_service.CATALOG及自动识别分支、project_binding.bind_project、wrf_meteorology.prepare_wrf_meteorology、native_meteorology.prepare_meteorology、project_inventory._cmaq及CALPUFF原生driver的实际适用条件。组件需要格式适配/科学声明本身并不是缺陷；这里未完成的是普通入口的任务推导与处理编排尚不能完整覆盖已有可执行组件，不能用“存在组件目录”替代贯通证据。

上节三项工程修复及141项定向测试仍然有效，范围是已列任务和格式的行为正确性；不再将其解释为五个模型整体泛化完成。本次没有新测试，因此不增加通过数量，也未改写旧实验结论。


## 2026-11-04 五模型公共工程输入端口机制与有界验收

### 验收结论与边界

**五模型共用输入端口处理机制：5/5已接入并有真实原生格式往返证据。任意完整工程、任意任务或任意版本的全部自动理解：未验收通过。** 两者不能合并为“五模型全功能通用PASS”。本轮没有运行五种模型求解器，没有证明新多源融合精度改善，也没有将同一原生场的派生副本伪装成独立来源。

上一节指出的专项入口缺口确实存在。本轮新增公共scientific-input-port编排，而非再为T2/SST或某个包名写分支。现有专业组件/历史路线保持兼容；本轮公共自动端口覆盖范围如下：WRF/CAMx/CMAQ为已登记、时空支持明确的固定水平原生数值场；SWAT+为原生逐日天气各站各变量；CALPUFF为现有SURF编解码及活动SRFDAT输入。其他组件原有显式科学任务仍可调用，但不把它们登记为本轮已自动推导。

### GitHub复用与实际调用链

已核查[Basic Model Interface](https://github.com/csdms/bmi-python)、[HydroMT](https://github.com/Deltares/hydromt)、[PseudoNetCDF](https://github.com/barronh/pseudonetcdf)。前两者提供具名变量/网格/单位与模型组件组织的架构参考；未安装新框架或复制其模型内核。实际直接复用当前xarray、Pint/pint-xarray、既有原生编解码器和写出器，以及上轮已核验的[SWAT+ Editor](https://github.com/swat-model/swatplus-editor)4.0.0，commit 3606ac4c3b54585072a557382b8e3b60b44dd354。版本及用途见upstream_sources.json、environment.json。

页面和工具包继续共同调用 `ecodata.algorithms.automatic_preparation.prepare_uploads` → `models.engineering.preparation.prepare_engineering_inputs`。`engineering.ports`仅提供各格式实际输入资源、变量、单位、时间、坐标/站点及原生写回绑定；公共模块据物理变量与支持域匹配来源，经既有UnitEngine与空间采样，再消费AlgorithmSelector.decide的实际eligible结果、既有recommend/execute_with_fallback/Pipeline及双层质量门，最后调用原生写出器。单来源只作明确的对齐/换算，标SINGLE_SOURCE_ALIGNED，不冒称融合或精度选优。多来源必须有现有算法真实要求的资源；本轮测试的逆方差融合需要实际标准差，不默认等权，不能安全传播协方差时拒绝标准差重采样。

修改位置：`src/ecodata/models/engineering/ports.py`、`engineering/preparation.py`，公共入口的前置衔接；`native_source_inspection.check_native_values`由旧组件与新公共写回共用；UnitEngine补CALMET格式明确单位和无量纲别名；页面只在本路线成功后替换“接口预留”旧提示，未增加控件或更改布局。没有改科学融合内核、评分、搜索器或第三方源码。

科学声明协议 `ecodata.engineering-semantics.v1`允许确认资源角色、变量物理意义、时间基准/统计区间、支持高度及方向等，不允许指定算法、赢家、运行命令、单位覆盖或填造数据。实际文件已经给出的事实不允许被声明覆盖。未知的CMAQ脚本激活资源需要明确的可移植输入角色；这不是“任意C-shell已被自动解析”。普通带source_id及科学变量元数据的CF来源也可识别；纯数据上传仍回到兼容的既有入口，不静默套用其他工程。

安全要求：时间基准和统计含义一致；SWAT日区间另核86400秒，不仅匹配日期；原生时间缺失/重复/非递增拒绝；移动网格、未支持垂直/边界支持拒绝；源CRS/站点身份明确、无覆盖不交付空更新；QA规则未绑定不静默忽略；原生范围/类别/位掩码守卫统一消费；风分量方向未知不猜地理风，WRF Q2保留混合比定义，不当作比湿。输出工程副本逐个非目标成员SHA256复核，源文件不修改，ZIP CRC及原生产品回读。

### 五种真实格式验收

|模型|实际变量|处理数值位置数|总耗时|结果|
|---|---|---:|---:|---|
|wrf|PSFC;T2|19602|36.05s|PASS：输入端口处理/原生回读|
|camx|sfctemperature;t2|4870800|12.19s|PASS：输入端口处理/原生回读|
|cmaq|PRSFC;TEMP2|525000|2.58s|PASS：输入端口处理/原生回读|
|swat_plus|precipitation;relative_humidity|17532|1.13s|PASS：输入端口处理/原生回读|
|calpuff|wind_speed;temperature|1680|4.74s|PASS：输入端口处理/原生回读|

证据位于 `output/unified_five_model_20261104/real_signedoff_*` 的receipt.json、native_sources.json、public/engineering_inputs下的实际product.nc、源/结果质量门、plan.json、decision.json及工程ZIP。真实输入复用已有WRF20230331原生初始场、CAMx官方met2d、CMAQ20180701 MCIP、已由官方Editor导出的Robit工程及原CALMET/CALPUFF示例。

**上述新验收的来源是这些原生真实数值的可追溯抽取，必要时作K↔degC或Pa↔hPa往返；它们不是独立观测，没有虚构新的测量或不确定性。** 因而证明的是多变量公共适配/单位/支持/写回，不是五个真实多源融合实验。独立脚本又比较了产品与原生写回字段、全部非目标成员SHA、CRC和源数据清单。SWAT天气/ CALMET保持其原生日期/站号含义；模式网格经纬度按现有适配器的操作约定处理，不由数值回读宣称测绘精度已独立证实。

### 行为测试与页面

tests_signedoff.txt：182通过、0失败、159条第三方deprecation warnings，26.14秒；强化真实融合输出值断言后再执行tests_values.txt，结果据该实际日志记录，不重复无关全量回归。涵盖五模型同一公共服务、原生写回、改名/排列/无关文件保持、新压力变量和单位换算、单位/时间统计/时间基准/物理定义冲突、区间长度、声明冲突、无不确定性拒绝、不安全标准差重采样拒绝、类别/范围守卫、语义eligible确实阻止执行、已存在的工程绑定/组件/界面回归。逆方差内核用明确标注的合成数据，独立断言输出为283K或4mm/day，不以算法名称出现代替数值执行。

五个模型的新公共路线均经Streamlit AppTest与工具包比较原生输出及实际选择算法，下载控件可见。**签收版另有真实浏览器验收**：通过文件选择器上传Robit原生工程包和两个实际来源文件，点击处理，看到“公共工程输入端口已处理”，实际下载包重开CRC正常，原生目标字段逐字节等于工具包结果。也核验本地路径输入。浏览器只覆盖这个真实SWAT案例，不能称五个模型均人工浏览器验收；AppTest和真实浏览器分开记录。详见browser_acceptance_signedoff.json、browser_upload_download_signedoff.zip；桌面8502已从实际主目录重启，健康检查200。

### 未通过的更强标准

|要求|状态|具体缺口|
|---|---|---|
|五模型共用实际输入端口处理与原生更新|PASS（上述输入契约范围）|不依赖包名、固定位置或指定赢家|
|五模型全部任务由普通上传完全自动推导|未完成|SWAT非天气表、复杂垂直/边界及部分格式仍走已有显式科学任务，未统一自动推导|
|CMAQ任意完整脚本工程的激活依赖闭环|未完成|动态source、环境、循环和条件不安全执行；明确资源声明可更新真实MCIP，但不证明CCTM完整激活输入|
|所有版本/网格/变量无条件适用|不支持此宣称|未知科学定义、方向、QA、区间、格式必须有真实契约，不能靠数值或文件名猜|
|新公共机制在五个真实多源案例提升精度|未评价|新真实验收是原生数据往返；多源内核行为有合成证据，旧真实融合保留原适用边界|
|五模型求解器原生运行|未运行|本轮验收输入准备和写回，不替代各模型运行及科学验证|

因此本轮提供了可复用的跨模型公共底座和五种真实格式证据，**仍不得用“共同函数存在”或“5份文件成功”把用户要求的任意完整工程整体通用标成全部完成**。未知项和失败没有降级为PASS。第一轮坐标发现漏扫、原生单位尾空格、单位别名和错误变量声明等失败记录保留；已修复有因问题。处理先前CALPUFF导出综合包时发现含多套不同控制方案，按歧义拒绝；最终改用原始完整明确工程验收，未把拒绝记录删去或称综合包也通过。

### 复现与历史保护

复现命令：`.venv\Scripts\python.exe -m pytest tests/test_unified_engineering_ports.py -q -o addopts=''`；真实验收脚本是run_real_ports_signedoff.py，需新建输出目录后运行，不能覆盖已有签收产物。其源数据、声明和输出路径均在脚本及机器清单中固定。独立核验见verify_delivery_signedoff.py。protocol.json写明支持/拒绝范围，source_fingerprints_signedoff.json与executed_source_signedoff保存实际生产源码；最终增强测试快照另存delivery_source。没有新增大数据下载、删除资料、改变旧实验、运行GOLEM研究、启动任意上传脚本或运行求解器。

SHARED_INPUT_PORT_MECHANISM = 5/5（明确契约范围）  
ARBITRARY_COMPLETE_PROJECT_GENERALITY = NOT_VERIFIED  
NEW_REAL_INDEPENDENT_FUSION_ACCURACY = NOT_EVALUATED  
NATIVE_MODEL_EXECUTION = NOT_RUN


## 2026-11-04 五套原始输入公共入口复测（不是字段往返验收）

用户要求测试之前五个数据。本次输入版本和完整路径固定于 `output/five_original_inputs_retest_20261104/cases.json`：WRF采用轻量包中的完整WRF ZIP与NOAA Memphis CSV；CALPUFF采用轻量包原工程ZIP；SWAT+采用原Robit ZIP与NASA POWER CSV；轻量包没有CAMx/CMAQ完整工程，因此CAMx采用既有5,170,342,324字节原工程与气象融合ZIP，CMAQ采用既有firstday_delivery完整交付目录（76文件、10,417,050,933字节）。没有以EPA观测ZIP冒充CAMx工程。

全部从当前 `automatic_preparation.prepare_uploads` 正常入口执行，未制作字段往返来源，未补匹配清单/算法/人工处理图，未改平台源码、质量门、策略、预算实现或原始资料。每个输入逐文件运行前后SHA256一致；使用新目录保留全部结果及失败。仅设有界测试墙钟上限，未新增数据大小限制；五项均在上限内返回，未发生超时。返回码0只表示测试runner记账完成，下面的FAILED按实际平台异常统计，不把它们当成功。

|模型|公共入口结果|实际产品独立回读|入口及回读耗时|原件SHA不变|
|---|---|---|---:|---|
|wrf|PROJECT_UPDATED|PASS|152.50s|是|
|calpuff|CONVERTED|PASS|52.41s|是|
|swat_plus|PROJECT_UPDATED|PASS|6.42s|是|
|camx|FAILED|NOT_RUN|64.20s|是|
|cmaq|FAILED|NOT_RUN|0.64s|是|

**WRF：** 实际更新d01的1640个SST值，最大数值改动6.82953K；逐变量核验其他变量、维度、类型、属性保持，陆地/海冰位置未修改，ZIP CRC通过。同期气温只有1站，T2保持不变，不称气温融合成功，不将SST数值变化解释为精度收益。

**CALPUFF：** 原始工程实际经已有平台处理与转换，SURF的168小时×5站×8字段逐项等于本次surface_product.nc，独立编解码回读及CRC通过；未运行CALMET/CALPUFF，也未生成新的METDAT。没有绕过运行程序路径检查来假装模型求解成功，本次本来只要求输入准备/转换。

**SWAT+：** 官方Editor实际导出69份原生文件，68条直接引用绑定；WGN的12×14月统计与原包CSV最大差4.986e-6（官方输出舍入），file.cio及WGN关联存在，CRC通过。NASA POWER保留，但本次成功是工程导出，不是NASA POWER融合完成，也未运行SWAT+。

**CAMx：FAILED，观测入口重复。** 公共读入器合并了原始全年NOAA、当日子集和decoded表；质检接受65416条记录后发现218组站点/时间重复，共631条重复记录，其中5组温度不同。按实际namelist的UTC工程窗口2021-12-22 00:00至2021-12-23 00:00检查，窗口内195组、585条重复记录的温度均一致。当前重复守卫在工程时窗筛选前触发，未区分原始/子集/解析表的来源重叠，故未进入数值融合和原生写回。不能简单说观测全部冲突，也不能无条件平均或把全过程重复都自动去掉。诊断调用原公共解析器，仅观察duplicated的原返回值，没有改变它或给算法传入筛选后的数据。详见duplicate_accepted_observations.csv及block_diagnosis.json。

**CMAQ：FAILED，原目录中的源码归档触发安全检查。** CMAQ官方源码ZIP有95个符号链接，典型为同仓库机制文件的相对引用。目录收集将源码归档也作为上传输入，统一archive_preflight在工程处理前按符号链接拒绝；没有跟随/解压这些链接。这不证明MCIP气象文件损坏，也不能把本次失败泛化为“气象融合算法不支持CMAQ”：本次尚未走到那里；排除该前置问题后动态C-shell工程闭包仍需另验。应由公共输入角色分层区分源码/参考归档与运行输入，而不是为了这个包放宽全部链接安全检查。

**结论：五套原输入已全部真实测试；3套完成上述有限处理和产品回读，2套未通过。不能宣称五模型原工程完整自动链路均通过。** 也不能将3套产物成功解释为三套均完成融合或精度提高。所有模型求解器本次均未运行；本轮只测试和定位，没有偷偷修复/裁剪输入后把结果改成PASS。未重复无关全量回归。

可复现入口：本目录run_retest.py（须使用新的输出目录，避免覆盖此次结果）；协议、源代码指纹、逐输入/输出清单、receipt.json、original_unchanged.json、独立回读、完整异常及summary.csv均保留。报告是唯一交付文档，机器证据独立保存。


## 2026-11-04 观测身份归并与软件参考隔离：CAMx/CMAQ公共修复及原件重测

### 修复内容

本轮只修两处公共入口组织问题，不改算法、融合内核、随机变异、评分、物理约束或原始数据。没有按CAMx/CMAQ包名、日期或路径写特殊放行规则。

1. **观测身份与来源归并。** 复用 `models/source_catalog.py` 增加reconcile_records公共操作；实际身份是提供方/变量通道、站点、规范化UTC时刻，温度、QA、报告类型、坐标与高程必须全部相同才归并。每条保留记录携带稳定record_id及所有实际source_member/source_sha256/source_row；相同数值但报告或QA不同仍拒绝，不同提供方不因数值相近被合并。未知身份/科学字段不当作已确认值。没有平均、挑选报告优先级或改变测量数值。

2. **先按任务支持核验，再检查当前任务内冲突。** `native_meteorology`的公共解析与最终核验分开；CAMx、WRF均在读取实际原生TFLAG/Times后，用同一_scope_observations进行最终处理。ISO时间转换UTC后比较，datetime64微秒/纳秒结果一致；精确匹配，不向整点取整或借跨时刻值。没有声明目标时间支持时，仍核查全部记录。时窗外数据保留在原件，记录排除数量；当前支持内有冲突仍阻止。此修改调整的是核验顺序，不改既有fuse_temperature的时间匹配或科学计算。

3. **输入角色公共隔离。** 正常页面/工具包/实验仍调用 `automatic_preparation.prepare_uploads`；该公共边界先调用source_catalog.partition_uploads，再进入原有_prepare_uploads。只有本地下载清单中的固定GitHub提交、文件实际SHA256/字节数、仓库提交目录及源码/README/许可结构一致，且未发现支持的运行数据容器时，ZIP才作为software_reference保留为不执行的参考资料。归档原文件名不是判据，清单中的旧文件名也不作匹配依据。不是所有GitHub ZIP都忽略；缺证据、校验不符、混有科学容器（含改名的NetCDF/HDF/TIFF）、运行控制等均保留原安全检查。参考包不解压到磁盘、不跟随符号链接、不导入/执行，原件不移动、不删除；分类理由和SHA存入input_roles.json并附入下载包。实际未重新联网下载或验证上游URL，记录的是本地来源声明与内容核验。

修改文件：`src/ecodata/models/source_catalog.py`、`native_meteorology.py`、`wrf_meteorology.py`、`src/ecodata/algorithms/automatic_preparation.py`；新增 `tests/test_input_roles_observation_identity.py`。archive_preflight的活动输入符号链接安全守卫没有放宽，模型接口及页面布局没有改变。

### 行为测试

初始复现测试7失败、1通过，失败记录保留。最终tests_signedoff.txt为**136通过、0失败、126条依赖deprecation warnings，98.44秒**，为新增及受影响的定向回归，非全量实验。新增文件11项（含参数化）：原始/子集/解析别名归并、UTC等价表达、输入顺序、冲突及QA/报告差异、提供方身份隔离、未知身份拒绝、微秒/纳秒及重复目标时刻、无清单/错误校验/混合数据与改名容器不得隔离、公共入口真实消费分类等。相关WRF、工程绑定、组件、五模型公共端口、页面与工具包测试仍通过。

### 完全相同原件的真实重测

没有手动删减原输入、没有新增匹配数据/字段往返产品或人工指定算法。继续使用前次cases.json中的CAMx 5.17GB完整ZIP，以及CMAQ firstday_delivery原目录76文件/10.42GB。逐输入执行前后SHA256均一致；原始失败及本轮所有日志保留在新目录 `output/input_roles_observation_fix_20261104`。

|对象|修复前|修复后|实际验收与限制|
|---|---|---|---|
|CAMx原完整包|合格观测站点时间重复，FAILED|CONVERTED|真实数值处理、当前产品原生写回及下载包回读通过；未运行CAMx求解器|
|CMAQ原完整目录|源码包95个符号链接被当运行输入，FAILED|ENGINEERING_INSPECTED|源码拦截已消除，正常完成工程引用盘点并生成带角色记录的下载包；尚未融合或生成新的完整原生工程|

CAMx：原生气象文件实际有25个整点TFLAG。65416条QA合格记录中，65303条不属于这些精确时刻；当前支持内113条归并为41个科学观测，72条完全同值/同QA/同报告记录保留为溯源别名。先前按全天窗口诊断的195组重复包含非整点报告，因此不是同一个分母；本轮没有把非整点观测移到整点。9个至少3站同期覆盖时刻实际处理195个共同有效域格点，写回**1755个t2值**。

独立验证从本次search/product.nc反查原生时刻与格点，逐项构造预期写回值，确认全部输出对应当前产品，其他变量、属性、时间与未处理区域保持。完整工程各依赖的SHA256复核通过，下载ZIP中更新气象文件与当前工程副本一致。平台实际选中MULTISOURCE，但post_selection记录为NO_SEPARATE_LOCATION_TEST、上游观测独立性未证实，所以**不宣称精度改进**。实际公共入口加回读约494.50秒，含原件前后指纹及启动的总计约509.17秒；耗时包括读取/复制与核验约12GB依赖，不把它和修复前提前失败的耗时当性能对照。

CMAQ：76个原上传资源中，校验匹配的18,837,775字节源码快照独立列为software_reference，75个资源继续进入原流程；95个源码符号链接未跟随。源代码原ZIP仍存在，SHA一致。输出清单包含相同的输入角色记录，实际识别3份C-shell控制声明并保留UNKNOWN。**剩余的是既有动态C-shell条件、循环、source和日期展开未闭合；本轮没有把工程盘点标成融合或CCTM成功。** 实际公共入口加回读约50.31秒，总计约82.66秒。

完整原件执行时源码快照为executed_source_real_v1；最终另加强了“无扩展名科学容器也不能伪装源码”的拒绝守卫，补测全部136项。对两套原件再次执行最终分类，与真实重测的分类逐项相同；观测与数值执行代码SHA没有变化。没有重复CAMx的8分钟处理以制造新运行次数。分类等价证据为classification_equivalence.json；最终源码为delivery_source。独立产品核验为independent_verification.json，计数及原件清单在对应case目录；summary.csv、protocol_signedoff.json及源码指纹均保留。

本轮两项指定入口故障均有修复及原件证据；CMAQ完整融合链路仍未通过，不用此次修复替代它的后续验收。未运行无关全量回归、正式GOLEM科研对照或原生模型求解器。


## 2026-11-04 CMAQ动态C-shell输入绑定、参考场处理与原生写回闭环

本节接续上轮ENGINEERING_INSPECTED限制；不改写历史失败结论。复用原76份、10,417,050,933字节输入，未下载、删除或改变原件。源码快照继续隔离为software_reference，其余75份经同一公共入口处理。本次不是CCTM求解器运行，也不是精度提升研究。

### 公共实现与范围

页面、工具包仍调用`automatic_preparation.prepare_uploads` → `project_binding.prepare_project_binding` → 公共CMAQ声明绑定 → `native_reference`的产品格式/空间时间适配 → 原`platform_search.process_current_input`、Pipeline和质量门 → `component_service`/既有原生字段writer → `engineering.native.finish_project`。没有新增页面、人工指定赢家、重写融合内核或修改GOLEM。

`models/control_flow.py`仅解释有界声明：变量、三值条件、有限while/foreach、受支持的日期/整数表达式及上传域内source；不执行上传shell、命令替换、mpirun或模型。未知分支不同值保留UNKNOWN；循环/分析步数有上限；未知条件中的模型调用不能冒充确定执行帧。`models/cmaq_binding.py`按实际I/O API端口及日期绑定，不按工程名、日期、尺寸、包SHA或开发机路径分派。仅未知机器根可用实际唯一的至少两段路径后缀重定位；内部未知、歧义不能猜选。

本原件实际选择有数据覆盖的一日运行声明，绑定37个端口；两日声明因下一日核心MET/GRID缺失被拒绝，证据完整保留。`config_cmaq.csh`及编译环境未提供，source保持UNKNOWN，另有1项未解决输入声明。因此只认证本次MCIP输入准备与写回，不认证完整运行闭包或任意C-shell语言。

`models/native_reference.py`按真实MCD15A3H产品元数据复用已有MODIS QA、尺度解码、有效域/无外推及公共空间处理；不是针对本包写死的入口，也未冒称支持所有卫星产品。原生LCC参数和MCIP声明地球半径与GRID经纬度交叉核对，最大差2.5915676e-05度；这验证文件内部几何一致性，不是仪器地理精度。128,659/160,801像元通过QA；部分QA覆盖明确保留，不填造像元。四日合成区间实际覆盖全部25个原生时刻，采用“静态植被参考增量＋保留MCIP逐小时异常”操作约定，绝不当成逐小时卫星观测；植被类型/覆盖率科学一致性未得到独立验证。运行起点还须与CTM_STDATE/CTM_STTIME和TFLAG一致。

### 真实结果与选择边界

|验收|结果|实际证据|
|---|---|---|
|原上传目录自动输入绑定|PASS|retest_v2/cmaq/public/project_binding.json；37条真实资源边与SHA|
|公共数值处理及多源候选|PASS|search/candidate_results.json、graphs.json和逐节点真实轨迹|
|原生MCIP写回及质量门|PASS|writeback/components/native_updated.nc、raw_quality.json、result_quality.json|
|完整工程副本与全部成员回读|PASS|receipt.json、output_members.json；ZIP CRC及成员SHA一致|
|独立I/O API读取及数值核验|PASS|independent_verification.json；PseudoNetCDF ioapi|
|原始输入保护|PASS|original_unchanged.json全部76份一致|
|CCTM原生求解|未运行|运行环境/source/部分依赖未闭合，native_ready=false|
|环境精度增益|未评价|没有独立LAI真值；不能用自留出误差宣称提升|

公共选择在186个相同自留出参考位置评价了61个单源和29个多源候选，均真实执行通过。最佳单源IDW RMSE=0.64842415，最佳多源RMSE=0.65732591，单位为LAI无量纲。这是卫星主源重建代理指标，**不是独立环境精度**。平台保留其真实选择SINGLE_SOURCE，未为本轮任务强制选择多源赢家；最终应用将选中参考场与原MCIP背景的逐小时异常组合。多源运算链已实际跑通，但本例不能宣称多源融合优于单源。完整图、所有失败/成本及选择依据均保留，原有策略与预算未改。

原生输出形状25×1×105×100，实际更新184格×25时刻=4,600个LAI值，范围0–8，最大绝对变化2.25390577；其余41个变量、全部原生全局/字段属性及TFLAG保持。未处理网格保持；产品与当前处理场数值一致；逐小时异常在2e-6容差内保持。独立读取成功后PseudoNetCDF/netCDF4清理阶段出现Not a valid ID析构警告，日志保留；不影响已完成的数组读取与断言，但不把它隐去。工程ZIP约2.591GB，SHA256=86acf9ee62650b383f3d516fdd7daab82a7ef28285dfe885205fea66a75999b0。

### 发现、修复与定向测试

首轮真实融合成功但写回被严格支持域校验拒绝：netCDF4属性序列化将大端float32的VGLVLS数组误解释为主机字节。已在公共`engineering/native.extract_native_field`中仅规范属性数组字节序、保持值及支持域指纹，不关闭验收；新增往返回归通过。首轮failure.txt和全部产物保留，修复后才完成第二轮原件验收。

126项不同的新增/受影响定向用例通过；tests_03为52通过，tests_04为73通过，最终tests_final为19通过（含重复复测，增加1项未知条件调用用例，不重复计数）。覆盖安全条件/循环/source/未知日期、歧义配置、跨年日期、数据覆盖选控、QA/解码、属性字节序、公共绑定、原生状态、页面同服务及下载。较早扩展检查中有2项旧`test_cmaq_external_project`因缺固定历史run.csh而失败、2项跳过，保留tests_02，不算本轮通过，也没有伪造那份历史依赖。未运行无关全量回归。

最终解析器又加强未知条件调用守卫；通过final_binding_equivalence.json确认原件全部绑定、参数与已执行版本逐项相同，未重复数值处理。源码前后快照、最终源码及SHA保存在本轮目录。公共入口含工程复制和回读耗时575.71秒，加原件前后指纹总计596.52秒；数值公共服务约169.29秒，其余主要为10GB输入流式读写/压缩/校验。不同阶段和失败成本均保留，不宣称性能优势。

### 文件与复现

全部证据根目录：`output/cmaq_dynamic_binding_20261104`。成功原件验收位于`retest_v2/cmaq`；下载产品为其`public/native_reference/writeback/EcoData-updated-project.zip`。只使用原始工程与配套实际卫星参考，没有新增模拟数据或手工指定图。核验命令：`.venv\Scripts\python.exe output/cmaq_dynamic_binding_20261104/verify_real.py`（先已有成功receipt）；定向测试见tests_03/tests_04/tests_final及对应源码。最终源码在executed_source_final，签署协议在protocol_signedoff.json，运行环境为现有.venv。

本机协作checkout仅同步本轮文件并核对前版本，未远程推送。实际桌面8502进程已按原启动器重启，健康200，runtime_before/after保留；本轮没有另做浏览器交互验收。原生文件写回与独立格式读取已完成，运行CCTM及证明科学精度改善仍未完成。

官方依据（只用于模型脚本与运行输入角色，不替代本包实际核验）：[USEPA运行说明](https://github.com/USEPA/CMAQ/blob/main/DOCS/Users_Guide/CMAQ_UG_ch05_running_a_simulation.md)、[官方运行控制示例](https://github.com/USEPA/CMAQ/blob/main/CCTM/scripts/run_cctm_Bench_2018_12NE3_CB6R5.csh)。未复制第三方模型或shell内核。


## 2026-11-04 公共无真值诊断：SALib有界输入偏移敏感性接入

按用户“选择一个合理好用的，直接接进去”授权，复用SALib 1.5.2 Morris方法和既有冻结图执行器，接入源共同偏移压力诊断。不新增融合算法、不改变异/fitness/验证/选图、原产品或模型writer，不把稳定性当精度。无新页面控件，无大型实验、下载数据、重做原生工程或覆盖历史结果。

公共消费者：`platform_search.process_current_input`完成原处理并冻结方案后 → `stability.attach/assess` → 原`execute_frozen_graph` → 原解析、Pipeline与质量检查 → 输出`stability/stability.json`及逐次真实产品/执行成本。原公共结果增加`post_selection_stability`，summary更新；`models/fusion_assessment.assess_task`将它原样纳入既有交接/下载评价记录，accuracy_improvement仍为NOT_ESTABLISHED。CMAQ等原生适配器接收的公共结果同样包含本诊断，不需要模型单独规则。页面当前处理与工具包共用上述服务；低层`submit_search`和直接冻结部署不自动触发此诊断，防递归和历史实验预算污染，可明确调用公共assess，不声称迁移全部历史入口。

### 冻结协议及边界

`source_offset_morris_v1`：仅对目标连续变量、已有单位、1—3个非恒定输入源，按可见数据标准差的1%为偏移尺度，固定2条Morris轨迹、4级、seed11，最多8次部署。不是估计出的仪器误差，不承诺已收敛的全局敏感度。首轮发现负偏移会让背景原有零值变负，3/6次被原质量机制拒绝，记录为INCOMPLETE并完整保留；修正是**事先将因子采样域限制在显式nonnegative契约内**，不截断结果、不放松质量门、不按效果追调扰动幅度。未声明非负则不猜测非负要求。

每次使用同一冻结图及原训练/状态契约。原始字节、坐标、时间、缺失位置保持；参考、观测、历史状态和目标资源不参与扰动，维持原执行侧用途。不同变量辅助场、未知单位、恒定源、分类输入、超过3源等给出排除/未评价原因，不静默截断、不伪造0风险。源顺序按角色/稳定身份/内容指纹排序，因子有原内容SHA；图/基线产品SHA均记录。

报告每次整场响应RMS、最大变化、负值数量、产品与成本；SALib的mu_star/sigma仅描述**平均输出**的响应及变化，不是测量误差、95%置信区间、精度收益或来源权重。平均响应可能掩盖局部抵消，故同时保留整场差分；零响应只能限定于该偏移方向，不能自动判某来源无用。没有主观稳定性通过阈值，DIAGNOSTIC_COMPLETE只表示规定检查做完。失败为INCOMPLETE/NOT_EVALUATED，不影响原合格产品，但诊断未完成状态保留。60秒仅在部署之间检查，不强杀在途内核；一次部署仍受原执行器预算控制。独立记账，不并入搜索评价数，缓存命中不冒充重算。

### 当前真实案例与行为结果

全部复用上节CMAQ已经生成的真实primary/background/target及历史图，不改原件、不重新做完整工程。

|案例|实际动作|结果与范围|
|---|---|---|
|历史已选单源图|6次冻结图真实部署|DIAGNOSTIC_COMPLETE，零失败；卫星平均输出mu_star约0.01692879、背景为0，与图中无背景有效路径一致|
|历史最佳多源图|1次基线部署＋6次真实扰动部署|DIAGNOSTIC_COMPLETE，零失败；卫星mu_star约0.01269659、背景约0.00290514，证明检查能检测实际双源数值影响，不仅查节点名称|
|公共当前输入入口|预设小预算objective12、总调用200、提议4、搜索墙钟15秒，实际只评价2个候选、0次提议；之后6次诊断部署|处理PASS，诊断DIAGNOSTIC_COMPLETE；未用它声称验证了变异或搜索优势|

单源诊断约11.60秒，多源诊断约8.43秒，公共入口附加诊断约11.72秒，均记录真实cost事件；额外部署由1—3源数量决定，成本不隐藏。这些结果说明来源影响检查可用，**没有证明多源更准确或改进旧排名**，原90候选比较及原生工程产物不变。

### 测试、依赖、部署与复现

34个不同定向用例通过：tests_02为33通过（新增6＋原空间回归27），最终新增下载记录消费者测试后tests_signedoff为7通过（含重复复测，不合计40独立用例）。覆盖真实SALib消费、无参考读取、源指纹及原字节不变、非负域、执行失败不报通过、无图/分类/无变化时未知、预算停止、公共入口附加而不改产品、交接记录不升级精度结论，以及原空间契约/真实DAG/信息隔离回归。测试合成边界不作环境精度证据。真实数值证据在`real_cmaq_v2`、`real_multisource`、`real_public_bound`，第一轮失败在`real_cmaq`，没有删除。

安装仅增加SALib 1.5.2、multiprocess 0.70.19、dill 0.4.1，未升级原科学依赖。已写入pyproject的既有verification extra及requirements-verification.txt，缺依赖时返回明确NOT_EVALUATED，不影响原处理。pip check仍有环境既存python-dateutil与aiobotocore/botocore、Pillow与Streamlit版本约束冲突；均为本轮未改动的原版本，记录pip_check.txt，不冒称环境依赖全绿。

本轮目录`output/robustness_integration_20261104`含前快照、最终执行源码、指纹、依赖安装/检查、配置、测试与全部失败记录。复现诊断测试：`.venv\Scripts\python.exe -m pytest tests/test_stability_diagnostic.py -q -o addopts=''`。实际多源重算脚本multi_check.py使用独立新输出目录运行，已存在的验收目录保持不可覆盖。最终元数据添加response_unit/排除原因及下载消费者后，针对性复测通过；真实多源验收使用最终源码。

本机协作checkout仅同步本轮文件并前版本核对，未推远程。实际桌面8502按原启动器重启，健康200，runtime记录保留。本轮没有新增浏览器交互验收，界面布局及控件未变。此版本支持数值源偏移诊断，不是任意模型/所有误差形式的自动科学认证。

方法来源：[SALib源码及方法](https://github.com/SALib/SALib)、[官方使用说明](https://salib.readthedocs.io/en/main/)。复用库函数，没有复制其敏感性算法。


## 2026-11-04 真实融合与模型工程轻量案例只读选取及核验：WRF五站气温

本轮仅找数据、下载必要新来源、解压和只读核验；不改平台代码，不新增算法、不调参，不执行融合搜索、原生写回或模型求解器。唯一报告仍为本文件。证据根目录为 `D:/codex/数据/WRF_20230331_多站气温融合轻量核验_20261104`，供页面输入的路径是其 `platform_inputs` 文件夹；使用说明为该目录的TXT，机器清单为evidence中的JSON/CSV，没有另建Markdown研究报告。

### 现状与候选选择

实际通过Git读取当前origin/main：`e28070d23919702afc5759052216f7a5006eb962`。核对automatic_preparation、workbench_page、project_binding、native_meteorology、wrf_meteorology及engineering/preparation六个直接相关生产文件，运行目录字节均与main一致；保留版本和逐文件指纹。当前页面模型是SWAT+、WRF、CMAQ、CAMx、CALPUFF，共用部分工程输入端口，但不能据此宣称任意工程/变量自动适用。已阅读本报告最新公共端口、原件复测、CAMx重复观测/CMAQ软件参考隔离、CMAQ动态绑定和稳定性诊断章节，并回看历史单站WRF不可适用、旧机器固定路径、ubESTARFM原包混有源码/缺TIFF科学元数据等失败边界。不能再用旧GOLEM或旧固定路径概述当前能力。

深入比较不超过三套：①本地已有NCAR 2023-03-31 WRF工程＋新同期NOAA站点（选中）；②Humaya真实SWAT+作者包（已有609,589,832字节模型归档及气候/建模资料，CC-BY-4.0，规模与多年份整理成本更高）；③官方1990-01-09 CALMET/CALPUFF四小时工程（真实气象、假设排放，已有原生运行记录，但新历史站点来源/格式协调不如本轮WRF方便）。候选比较仅保存在candidate_comparison.json；后两者未重复下载，也未以假设排放案例充作独立环境精度证据。

### 最终数据、来源、许可与体积

复用 `WRF_20230331_官方小案例_20261102/severe_outbreak_2023_output.tar.gz`（389,679,208字节，SHA256 04ba6923cae0565e858e1255b99298031d3f590efd0aaefb05a8e6c65bcd0055）。该官方归档确实同时包含namelist.input、wrfinput_d01、wrfinput_d02、wrfbdy_d01及参考wrfout；本轮只解出前四份真实运行输入，不把wrfout当输入或观测真值。它们与先前本地case_native逐字节一致。复用全部81份WRF v4.5运行资源，每份都与既有来源清单SHA256一致；没有重新下载全球地理数据、FNL历史或完整模拟输出。

新的NCEI Global Hourly接口请求只取2023-03-31 00:00至2023-04-01 00:00 UTC的小范围站点资料，两份原始CSV合计122,318字节。请求的Little Rock旧ID无返回，记录为缺站，不伪造第三站；实际新获得Nashville、Atlanta、Knoxville、Louisville四站。第五站Memphis原始65,503字节CSV从已有轻量案例包原样复用，核对原ZIP成员并另向NOAA仅查询本次实际使用的00 UTC瞬时记录，STATION/DATE/TMP/REPORT_TYPE/坐标/高程均一致；2,771字节核验响应不作为第六个来源。没有重下完整已有观测。原始响应保留所有QA、报告类型及额外变量，未预插值或制造缺口。

平台输入文件夹合计126,408,684字节。其中工程ZIP为126,220,863字节，SHA256 8e5815b0689760d6fc6ad5cf529cbbac989469b8f72fb3f1820a539a0c14a8ac；其85个文件展开为219,776,576字节（四份原生输入＋81份运行资源），不需要解压大量历史资料。完整核验目录另保留展开工程、输入ZIP及证据，逻辑占用约349 MB，不能将输入ZIP体积冒充整个核验目录占用。大小计划、原归档目录、请求时间/状态、接收字节数及SHA256均保留；本地SHA256是追溯指纹，不冒称上游签名或官方发布校验值。新下载的原始CSV与本轮解出的原生工程已设只读；复用外部原件运行前后SHA不变。

WRF 4.5自带LICENSE实际为UCAR/NCAR不主张专有权、视为public domain，要求保留声明并免责；没有套用其他版本的MIT等许可。NOAA NODD允许公开使用、请求署名、不暗示背书，不得将修改资料称为未改原件。NCAR教学数据公开可获取，但未找到单独数据CC许可，不把代码许可扩展为所有数据的商业再分发保证。官方许可、教程和格式文档保存在official_documents。ISD文档现行广告URL实际返回HTML S3页面，不能按文件名当PDF；另一地址完整下载与Range方案各出现截断，保留失败并停止重试，最终复用本地已存在的完整1,187,390字节、137页官方格式PDF，重新核对PDF魔数和正文。未接受残件。

### 真实数组与工程完整性

用f90nml读取真实namelist：两域，30 km/10 km，2023-03-31 00 UTC至次日00 UTC、24小时、6小时边界间隔。实际读取两个wrfinput和wrfbdy中的全部数值变量，检查维度、有限值和掩码；T2单位K、原生定义TEMP at 2 M，初始时刻准确。初始场含地形、土地利用和海陆等原生变量。wrfbdy的Times最后是18 UTC，但四个区间结束标签分别为06、12、18、次日00，连续覆盖整段24小时；未误判缺最后六小时。

复用当前公共`bind_project`只读绑定：目录与ZIP均BOUND，同样的两个initial_state和一个boundary_state、资源SHA相同。独立重新打开交付ZIP做CRC与逐成员SHA比对，并实际回读T2数组；98份交付数据/文档清单及85个工程ZIP成员复核通过。没有用目录存在、网页或绑定状态代替数值读取。工程数据层是从已生成初边界运行/写回所需的现成工程；不是从头重做WPS/geogrid的完整工具链。

求解程序另记：交付不含匹配WRF 4.5的wrf.exe；本机wsl --list返回尚未安装WSL，项目external中未找到可确认匹配的WRF程序。本轮未装环境或运行求解器，不能称Windows直接原生模拟已就绪。换到已有WRF 4.5及NetCDF/MPI依赖的Linux环境，可按保存的官方README/说明使用原始工程；完整求解验收仍须SUCCESS COMPLETE WRF和实际结束时刻/两域输出核对。

### 为什么是真融合需求，如何匹配

两类信息是FNL驱动的空间连续WRF初始背景，与同域同刻真实地面气温站点，不是把多个变量或文件数量叫融合。背景提供区域结构，观测提供局地点位约束，可测试气温场的站点—背景修正/补充及其原生输入写回。仅作诊断的最近格点背景与站点温差为约-6.68至-4.69 K；这是信息不一致的信号，含时间/测高/地形/格点代表性影响，不作为融合RMSE或收益结论。

原样保留NOAA不规则点位及小时/次小时/天气报频率，WRF仍为Lambert曲线经纬网格、30 km/10 km，没有提前对齐。需要按官方TMP缩放因子10及摄氏度定义解码，再转K；时间按UTC和瞬时天气报取准确初始化时刻，不按日期平均、不向整点舍入。空间按真实XLAT/XLONG、投影和站点坐标映射。当前公共解码器接受FM-12/FM-15/FM-16及QA 1/5；本包218条原始记录中209条通过该QA，9条未通过；再限定原生时刻后恰有5个合法FM-12站点。五站凸包内有d01的154个、d02的1,365个原生格点中心，其余网格应保持原值。该计数是支持域检查，不是已执行的融合效果。

### 能验证、不能验证和平台具体缺项

本轮确认当前main已有的原始Global Hourly解码、精确时刻匹配、至少三站/空间凸包前置条件以及真实工程只读绑定可接受本案例。现有T2分支可调用公共实际候选评价/随机变异及公共写回机制；本轮故意没有运行这些环节，不能把历史合成测试或别的真实SST成功算成本包融合/写回通过。

五站是待融合输入，不是独立参考；也无法逐站排除已被上游FNL同化。没有独立精度真值，官方wrfout不是观测。因此可用于下一步检验融合数值执行、支持域保护、原生写回、物理范围和稳定性，不能由本轮证明精度提升或预报收益。只有一个初始化时刻、五站，不构成长期泛化实验。源ISD文档说明有符号地理经纬度；精确站点大地基准及仪器测高未逐站认证，平台按EPSG:4326地理坐标运行约定处理，应保留此限制，不能宣称测量级配准已证实。点/网格地形与代表性、T2更新后的三维动力一致性亦未认证；风向旋转、复杂垂直/边界任务、移动网格等不能推为已支持。

代码没有修改；实际生产src/integrations相关类型文件前后全部指纹一致。未执行平台搜索、变异、融合、写回或模拟；没有为该工程写专用补丁。证据包括case_verification.json、platform_version_audit.json、native_array_checks.json、platform_readonly_binding.json、observation_quality_and_support.json、file_manifest.csv、memphis_provenance_recheck.json及independent_reopen.txt。可用现有.venv运行evidence/read_only_check.py复核，首次整理脚本不要覆盖已封存只读资料。

官方来源：[NCAR案例与原生输入说明](https://www2.mmm.ucar.edu/wrf/wrf_tutorial/tutorial_met_data.html)、[WRF v4.5运行资源](https://github.com/wrf-model/WRF/tree/v4.5/run)、[NOAA接口说明](https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation)、[NOAA开放使用说明](https://registry.opendata.aws/noaa-isd/)。具体API查询串、旧资料来源链和下载异常均在机器证据中保留。


### 2026-11-04 本案例队友交接整理

原资料133文件完整复制至 `D:\codex\交付\WRF五站融合案例_队友交接_20261104`，保留原始数组、展开工程、平台输入ZIP与CSV、官方说明及全部失败/核验记录；新增队友操作与限制TXT、可搬迁的Python标准库校验脚本和逐文件SHA256清单。说明区分平台自动处理与Linux WRF原生模拟，明确不含平台安装包/求解器、未运行本例融合/模拟、独立性/测高/地形/动力一致性限制，并提示历史证据脚本中的本机绝对路径不能要求队友照搬。

外发ZIP为 `D:\codex\交付\WRF五站融合案例_队友交接_20261104.zip`，253,624,308字节；展开348,600,683字节。SHA256 `ab5e7f448483b8d1d167320f473dbd9c1e3233b7f8cc8b709ebecbdc4124e8ad`，CRC及136项清单成员逐字节校验全部通过，原工程85成员与展开文件一致。已有源码和科学数据未改动，未发送消息或上传外部服务。旁附zip.sha256及打包核验JSON，未新建Markdown报告。


## 2026-11-04 多案例轻量资料扩充与只读验收（两套新流域＋Humaya扩展）

按用户继续扩充案例并集中交给队友的要求，本轮只寻找、取得、读取和核验资料，不改平台代码、不新增算法、不调参、不执行融合搜索/变异或模型求解器。Git获取当前main仍为e28070d23919702afc5759052216f7a5006eb962。最终交付文件夹 `D:\codex\交付\EcoData_多案例工程与融合数据_20261104`，ZIP `D:\codex\交付\EcoData_多案例工程与融合数据_20261104.zip`，32,460,714字节，展开96,610,778字节。SHA256 `b51e67199aaebce27e6301d1ad747b58844880090e2f45edb46467d18af368b2`。压缩包CRC、1,087项文件清单SHA256、三个工程ZIP与展开成员一致性及新来源真实数组回读通过。362个实际平台src/integrations文件指纹检查未变化。

### 资料范围，不虚增案例数量

|资料|原生工程/来源|实际核验|当前平台边界|
|---|---|---|---|
|Mandena，马达加斯加|新作者工程，Zenodo21633869，CC BY4.0；rev61.0.1格式、1987—2022年，82份活动平铺输入；另保留备用SWIFT资料|file.cio直接引用/天气站/WGN基础检查无错误，天气语法及日期序列可读；原PCP确有2,191个-99|当前weather_ports拒绝缺测标记，BLOCKED；原SWAT基线有WGN机制，不能把-99填0后称成功|
|Cangandala，安哥拉|新作者工程，Zenodo17736146，CC BY4.0；rev61.0.1格式、2011—2022年，119份平铺输入|基础工程引用/完整天气数值语法与日期序列检查通过；公共端口实际读出48个|未执行融合；原包名ERA5L与PCP标题MSWEP有来源标签冲突，未认证原强迫来源，不当独立实测真值|
|Humaya，墨西哥|复用已有Zenodo18820320的Gauge基线，CC BY4.0；rev60.5.7格式、1980—2001年，91份平铺输入|基础引用/完整天气数值语法与日期序列通过；实际读出18个端口|是已有工程的新来源/新测试扩展，不能称第三个新下载流域；未执行融合或模拟|

以上是三套真实资料/工程测试任务，其中两个是本轮新取得的地理区域，一个为复用扩展，全部属于SWAT+。不声称这轮新增三个独立新流域、覆盖五模型，或三套已全部无条件一键融合。基础输入引用检查仍不认证所有深层数据库引用、物理参数及原生求解可行性。

### 新数据和真实融合需求

每例实际下载并保留NASA POWER/MERRA-2偏差订正日降水原网格（约0.5°纬度×0.625°经度、连续7日）及NOAA CMORPH v1.0偏差订正卫星日降水原文件（0.25°×0.25°、共同3日）。共同期分别为Mandena 2021-02-07—09、Cangandala 2015-01-03—05、Humaya 2001-07-03—05，均处于各实际工程时段。新来源和对应区域实际数组均非空、原降水有限值非负，空间/时间/单位从NetCDF与官方接口确认，不靠文件名。CMORPH只取必要日，每日全球原文件约0.4MB；POWER仅取小区域，没有下载大段全球历史。没有提前对齐网格、拼成伪高分辨率数据或指定算法赢家。

需求是检验不同机制分析资料之间的降水互补，形成指定区域/时段的新驱动；Mandena有作者明确讨论的原强迫缺测/天气发生器问题，Humaya有实际站点/山区空间代表性问题，Cangandala可比较完整原强迫与两种新分析来源。只是可检验的融合需求，不预设多源一定比单源好。必要转换包括真实坐标映射、CMORPH 0—360与POWER正负经度等价整理，以及日降水率/日累计量和目标模型日支持的明确约定。

两个新产品均声明UTC日。POWER请求显式指定time-standard=UTC。CMORPH原文件标P1D，但time_bounds实际写当日00:00:00—23:59:59，端点差86399秒；初次审计按半开86400秒假设中止，最终保留原始边界、实际差值和P1D声明，明确要求包含终点的发布时间约定需被显式规范，未修改原数据或放宽平台验收。原模型天气端口按model_calendar_day组织；原站点观测日基准未逐站确认，不将新产品UTC定义倒填给原观测。新UTC驱动的目标日约定和原站点比较必须分别处理。

没有提供有依据的误差标准差阵列，不能编造sigma或默认逆方差/等权。POWER和CMORPH均可能共享部分订正雨量信息，作者旧强迫亦可能共享来源，独立参考尚未确立。不能把输入、模拟输出或自留出代理误差当作独立精度证据；本轮不证明融合精度提升、变异优势或模型预报收益。

### 取得方式、失败与限制

原作者ZIP体积较大：MarMenor260MB、Mandena源包336MB、Cangandala ERA5L包492MB、Ngango ERA5L包545MB。先读官方记录大小/许可，再读取HTTP Range原目录，选择必要原生输入，仅下载对应压缩字节块，逐成员检查原CRC、解压字节数和SHA256，避免下载GIS与大量旧结果。最终选用的Mandena数据下载约0.67MB、Cangandala约1.96MB；Humaya原600MB归档完全复用，未重复下载。未完整下载的大作者ZIP不宣称通过其官方整包MD5。原始压缩块、目录、选取计划和下载时间/状态保留。

以下未通过候选不计入三套正式资料，原件/证据集中放在90_未通过候选：官方Ames参考工程有缺失/空文件和站点/WGN关联问题；Osu有版本/覆盖问题；MarMenor基础引用虽通过，逐行天气读取发现CA21_PCP.pcp中如1.158.200.891的异常数值；Ngango原包两个工程表示分别触发严格字段或日期序列问题。没有替作者补文件、猜小数点或改日期来凑通过。这些负例可用于质量守卫测试，不用于宣称真实融合收益。

CHIRPS服务请求曾返回与请求期不符的实际日期（已保存的诊断响应为2020-01-01，而请求为2019-09-10至2019-09-16），被拒绝，未重命名伪装目标时段；其他日期请求和替代服务失败也保留。Daymet主服务401，镜像格式请求400，格式校正后请求超时，停止继续获取，没有虚构Daymet网格。POWER一度框422按官方最小二度框规则修正，只扩大请求范围、不改变原分辨率或日期。候选/服务失败与限定状态统一存JSON，不新建分散Markdown报告。

### 平台接入、运行说明与交付

已读取当前main公共端口实际实现：原始外部产品的source_id/角色、物理字段、时间语义、空间支持并非均能由这些原文件自动组织；UTC日与model_calendar_day也不能无声明匹配。当前三套只能认证上述真实资料、基础引用及有限只读端口状态，不能说这些原始产品均已在页面完成自动融合。Mandena另有-99/WGN的明确读取阻塞。本轮没有创建假不确定性、硬编码规则、专用补丁或用旧固定流程冒充接通，新增源码仅为数据目录内的取得/核验脚本。

根目录TXT说明逐例来源、用途、输入方法、原始分辨率、日界/源身份/独立性限制、SWAT版本和多年运行成本；一次只提交某个案例的platform_inputs，不提交整个混合包。程序环境不在数据包中；使用队友已有平台和匹配SWAT求解器，不默认不同大版本完全兼容，不在原件目录启动多年大量输出。核验脚本“01_校验交接包.py”可换盘运行；--arrays会用既有xarray/numpy重新读真实NetCDF，默认只需Python3.11标准库。它明确显示Mandena端口BLOCKED，不将完整性PASS升级为平台融合PASS。没有启动任何原生模型。

本交付保存三个正式任务、五份不计数的候选表示、官方来源/许可、原始压缩块、下载与全部核验记录。外发ZIP旁有SHA256及交付核验JSON。最终压缩包重新打开核对全量成员后才交付，没有执行外部发送消息或上传。后续应该用这些真实案例检查并修复公共机制，不以这次仅取得数据宣称平台效果已经验证。

来源：[Mandena作者记录](https://zenodo.org/records/21633869)、[Cangandala/Ngango作者记录](https://zenodo.org/records/17736146)、[Humaya作者记录](https://zenodo.org/records/18820320)、[NASA POWER接口与时间定义](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)、[NOAA CMORPH产品](https://www.ncei.noaa.gov/products/climate-data-records/precipitation-cmorph)。实际下载URL及许可均在包内机器清单中，不以网页说明替代真实数组核验。


## 2026-11-04 三套SWAT+原资料输入缺陷复现、公共入口修复及保留性复验

本轮按用户明确范围，先复现再修改，只处理外部降水产品识别、原生缺测/WGN兼容读取、阶段状态与逐来源报告。运行目录D:\codex\EcoDataEngine，源码目录D:\codex\EcoDataEngine-GitHub，修复前本地和远端main均为e28070d23919702afc5759052216f7a5006eb962。修复未提交或推送。原交接ZIP SHA256仍为b51e67199aaebce27e6301d1ad747b58844880090e2f45edb46467d18af368b2。

### 复现与分类

原包全新解压至output/swat_source_repair_20261104/original_inputs。逐套只提交platform_inputs的原工程ZIP与4份NetCDF，再用同一原工程ZIP单独对照。六次均执行实际app.py的Streamlit AppTest页面处理事件，无处理器模拟；保留页面元素/状态、日志、实际output/workbench任务目录和下载原包。这是实际页面公共处理函数与真实文件的运行记录，不是浏览器视觉截图，也不是模型求解运行。

三项均复现：完整输入source_fields返回0，三套全部ENGINEERING_INSPECTED；12个原始NetCDF数组可完整读取。Mandena独立weather_ports报负值或缺测标记，另两例48/18端口成功。六次页面status均显示“数据处理完成”，下载仅project_inventory.json及handoff.json。这三项分别属于产品接入缺陷、原生缺测契约兼容缺陷、状态及可追溯报告缺陷，不能归咎于资料损坏或证明作者模型无效。

### 修改范围及科学边界

新增engineering/precipitation_products.py只依据文件内提供方、产品标识和变量语义识别POWER/CMORPH，记录实际坐标、时间边界、单位、缺测、原始元数据与哈希；不依赖文件名制造身份，不修改原始数组/属性。CMORPH多日期分片共享产品source_id，来源身份不代表误差独立性。识别产品仍需通过科学准入：没有确认的日基准、率/累计契约、QA规则等显式列为blocker，不能只添字段就当完成转换。preparation.source_fields及prepare_engineering_inputs使用此识别和准入记录，不再因无自定义source_id而默默忽略原产品。

swat_weather_project._records增加仅供原生读取的allow_native_missing选项；默认转换产品校验仍严格拒绝缺测标记。原生<=-97按SWAT缺测标记保留，weather_ports中仅以NaN掩码表示，保留日期、站点引用、原始记录与作者WGN，不补零、不删日期、不执行发生器。部分天气写回只改实际提供有限新值的记录，未处理缺测保持原标记；拒绝处理结果引入新的缺测哨兵或负降水。原生WGN仅复用既有weather_dependencies检查具名引用和12月统计结构，不等于作者原生运行有效性已证实。

project_inventory与workbench_page使用明确的盘点状态，新增source_admission.json和processing_status.txt，页面/下载逐来源解释识别结果、未使用原因和需确认条件，分别记录工程盘点、融合执行、原生写回、模型运行四个事实。没有新增配置选项、改界面风格、改融合内核、评分、搜索、算法选型或科学参数。

### 同一批未修改原资料的复验结果

|案例|完整输入修复前/后|原天气读取修复后|最终真实阶段|
|---|---|---|---|
|Mandena|0→4份外部资源，2产品家族|48端口；pcp_local.pcp原2,191处缺测保留，WGN结构检查无错误|ENGINEERING_INSPECTED；待科学信息确认，未融合/写回/模拟|
|Cangandala|0→4份外部资源，2产品家族|48端口|ENGINEERING_INSPECTED；待科学信息确认，未融合/写回/模拟|
|Humaya|0→4份外部资源，2产品家族|18端口|ENGINEERING_INSPECTED；待科学信息确认，未融合/写回/模拟|

另三次仅工程ZIP对照均为0外部来源、ENGINEERING_INSPECTED，页面明确只有盘点。Mandena不同站点引用同一原PCP文件，不能把48端口中的重复缺测掩码合计当作物理文件缺测数。所有六次最终页面无异常、有可读下载；最终下载四项为project_inventory.json、handoff.json、source_admission.json、processing_status.txt。旧下载与中间调试失败保留，不隐去负记录；after_verified与pytest_final_3.log为最终验收证据。

仍阻塞的科学条件：POWER文件本身没有UTC/LST声明，虽然包中其他下载凭据记有UTC，但本轮实际提交的platform_inputs不含该凭据，平台不能越界读取再倒填声明。工程model_calendar_day与新来源UTC日对应未知；CMORPH原mm/day为率、时间边界包含23:59:59，未擅自延长边界或把单位改名当积分。Humaya原0–360经度与负经度工程站点仍需保持网格支持建立对应，未执行无契约重采样。没有真实误差标准差/独立参考资料，不能编造权重或宣称精度提高。资料识别已修复不等于科学契约全部齐全；本轮没有强行融合。

额外问题只记录未扩修：Cangandala原包ERA5L与PCP标题MSWEP的来源标注冲突；同产品日期分片不具误差独立性；工程全部深层依赖、版本兼容和原生运行仍未验收。用户排除的文件头探测“截断”日志原样保留。核验脚本曾需使用文件流避开本机netCDF4直接打开中文路径限制，此为独立读取方法选择，未修改平台文件路径处理。

### 验证与原件保护

158项定向/回归测试通过，覆盖产品改名不影响身份、文件名不能伪造身份、单位/时间边界/QA冲突、缺测读取与温度逻辑、部分天气写回、未处理日期保留、五模型公共端口、工程盘点与页面流程。三个真实工程另用实际write_ports做无数值更新的保留性检查，82/119/91个原成员全部逐字节不变，ZIP CRC回读通过；这三个native_preservation包只证明写回器保留性，不是融合结果，不能将其中writer阶段标签当作真实融合证据。实际六次用户流程没有工程写回。

原交接ZIP及解压出的1,088个文件前后SHA256均一致；与main对比，非目标src/integrations受跟踪文件无差异。7个目标源码/测试文件同步至源码目录和当前运行目录；没有另写实验局部规则。完整证据位于D:\codex\EcoDataEngine\output\swat_source_repair_20261104：before、after_verified、verification_summary.json、integrity_verified.json、code_sync.json、原下载ZIP、页面状态、错误日志及复验说明.txt。唯一总报告仅追加本节，未创建新Markdown报告。

依据：[SWAT+官方天气缺测说明](https://swat-model.github.io/swatplus-editor-documentation/edit-inputs/weather/)、[SWAT+天气发生器说明](https://swatplus.gitbook.io/io-docs/introduction-1/climate)、[NASA POWER时间定义](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)、[NOAA CMORPH产品](https://www.ncei.noaa.gov/products/climate-data-records/precipitation-cmorph)。同时读取既存官方SWAT源码快照commit6f515368525e7d84089ae0f9d33f4c9115de5679中cli_precip_control.f90，确认缺测分支；没有运行上传脚本或模型求解器。

部署复核：已重启本机正式8502平台，当前PID16408，健康检查HTTP200；7个目标文件两目录哈希一致。重启后又用同批原资料完成六次真实页面处理流程，after_deployed/summary.json记录均无异常、均明确ENGINEERING_INSPECTED；此阶段仍未融合、写回或运行求解器。


## 2026-11-04 通用降水适配、缺项反馈与多模型真实资料验收

按用户继续补齐通用适配并扩大真实资料验收的要求，本轮修改平台公共入口，复用已安装且有来源依据的xclim 0.59.1、xarray、cf-xarray/Pint与h5netcdf，不写死Mandena/Cangandala/Humaya、日期、路径或算法赢家。GitHub main基线为e28070d23919702afc5759052216f7a5006eb962；本轮14个目标源码/配置/测试文件已同步运行目录和GitHub工作副本，未提交或推送。页面仍只有数据输入、模型选择、自动处理和下载，没有添加科学参数配置面板。

### 本轮真正完成的适配

- 依据官方日产品定义和实际边界自动做降水率×有效区间积分，派生降水量单位为mm；修正SWAT原生逐日降水端口和写出报告的mm/day误标。常规连续序列实际调用xclim.rate2amount，单记录或显式区间用现有Pint量纲计算，不依靠文件名或只改单位标签。旧单站日天气接口仍接受其既有mm/day日平均率输入，经实际24h积分写出mm；真实NASA POWER全年资料已逐值与原JSON对照。
- CMORPH的原始P1D和00:00:00—23:59:59边界作为原证据保留，派生半开区间明确为24h，未重写原数据。POWER已核查的日产品时间步长可以确定积分长度，不能因此推断UTC/LST日界。
- 用xarray对经度做等价重标记/排序，同时携带数组和单元边界；同产品分片通过join=exact拼接日期，不默认扩网格、不平均重复日期、不把不同日期视作独立来源。不确定性辅助字段不再被误当成第二降水字段，也不制造sigma。
- 对原生虚拟天气站可提取原网格单元驱动估计，保留原单元支持；这是坐标对应，不是新点观测、降尺度或精细化。真正执行时仅在各来源与工程共同日期上运行内核，再按原时间索引散回；共同期外的原记录和-99保持，非目标变量/文件保留。
- 原下载凭据必须匹配文件SHA256、官方接口、参数和日期，才用于核实POWER的UTC/LST选项。仅消费本次上传的凭据，不越界读取原包其他目录；原凭据不是新造科学声明。来源日与模型日不一致时拒绝平移日累计标签，两个LST缩写也不足以证明同一日界，避免把地方太阳时误当当地标准时。
- 准入继续调用现有AlgorithmSelector及真实融合内核，未修改评分、随机变异、候选赢家或融合权重。缺项集中进入required_information.json、页面及下载包；已完成的适配产物也可下载，不能把适配记成融合/写回完成。

### 三套未修改原资料复验

各自原platform_inputs（一个工程ZIP和四个原始NetCDF）及同一工程ZIP单独对照共六次实际app.py页面处理流程，全部无异常并保留状态、任务目录和原下载包。此处使用无处理函数模拟的Streamlit AppTest，不冒充浏览器视觉验收。12个原始数组实际生成6组适配产物；独立逐数组核对数值、缺测掩码、日期和原网格保留通过。原交接包解压的1,088文件哈希未变。

Mandena、Cangandala、Humaya均仍为ENGINEERING_INSPECTED，三套融合次数和工程写回次数均为0。已完成的是产品适配与准确缺项反馈，不能写成三套融合通过。Mandena原2,191处PCP缺测照常保留，WGN没有运行。

再分别附上原包sources_original中既有POWER_receipt.json做三次公共入口复验，平台自动核实UTC，三个案例的剩余问题均收敛为：

1. MODEL_DAY_UNCONFIRMED：原工程日降水统计的日界未知。用户已表示不清楚；原始天气记录和已交付作者文本没有找到可靠日界依据，不能按经度、文件日期或新产品UTC定义倒填原工程。
2. FUSION_INPUTS_INCOMPLETE：现有融合候选需要的同支持真实标准差或已绑定的独立训练/观测资源不齐。三份原资料没有这类误差资源；没有从两个来源的差值编造误差或默认等权。不同日期/版本产品的身份本身不证明误差独立性。

只提交原五文件时，POWER凭据不在输入中，会另外提示SOURCE_DAY_UNCONFIRMED；这是上传范围内缺凭据，不能要求平台从未上传的旁边目录读取。原包已有凭据可随工作副本一并提交，原NetCDF不需更改。当前三套暂不能真实融合并写回，科学信息缺失不被伪装成软件已完成。

### 扩展真实资料：五种模型路径均实际处理并独立回读

环境修复后，重新用既有原资料通过相同公共入口运行，未重复下载数据或人为指定算法赢家。全部原输入前后哈希一致，最终下载ZIP CRC通过。独立检查实际原生数组与文件，不能只凭成功提示验收：

|路径|实际结果|独立核验边界|
|---|---|---|
|WRF官方轻量案例|SST写回1,640个值|原NetCDF变量/属性保留，非SST变量、陆地与海冰约束读回；不是站点气温融合通过|
|CALPUFF官方案例|168小时×5站×8字段原生SURF输出|原生SURF解析数值与实际处理产物一致；未运行CALMET/CALPUFF|
|Robit SWAT+|69个原生工程文件、68项引用绑定|原WGN月统计与导出文件在预设舍入范围内一致；这是工程导出，不能证明额外POWER CSV已融合|
|CAMx EPA案例|t2写回1,755个值|原生格式、时间、其他变量及属性保留；没有把输出变化当精度提升|
|CMAQ官方案例|LAI写回4,600个值|与实际处理产物一致，其他变量/属性保留，另用SciPy NetCDF3读取TFLAG/LAI验证|

五套通过的是上述已有工程处理/原生格式路径，不等于五套均完成独立精度验证、所有工程深层依赖齐全或原生求解器运行。没有启动WRF/CCTM/CAMx/SWAT+/CALMET/CALPUFF模拟，没有宣称融合精度提高、全局最优或三画像/变异新优势。

### 测试、环境和正式实例

188项定向与回归测试全部通过，包括已齐全契约下两个来源真实执行融合内核并部分写回、中文输出路径、日界与LST冲突、未知凭据/误差、辅助标准差、单位/区间、经度/边界、缺测保护、五模型公共端口和页面下载。一个旧UI测试依赖已删除的“首轮专项转换”侧栏，已迁移到当前极简页面，保留真实writer执行、下载字节与错误清除验证，没有恢复旧选项或跳过测试。

发现平台虚拟环境继承的python-dateutil 3.0与botocore/aiobotocore要求冲突，Pillow13与Streamlit要求冲突；只在D:\codex\EcoDataEngine\.venv中安装兼容python-dateutil2.9.0.post0、Pillow12.3.0，并在pyproject.toml写明兼容范围。系统包没有卸载或修改，pip check最终无冲突。正式8502实例已重启并HTTP200，运行源码与工作副本目标文件一致。依赖修复后再次完成五套真实处理、独立回读及页面复验。

完整证据集中于D:\codex\EcoDataEngine\output\generic_precip_adaptation_20261104：delivery.json、验收与运行说明.txt、source_sync.json、pytest_signedoff.log、pip_check_final.log、deployment_health.json、raw_signedoff、raw_adaptation_signedoff_verified.json、original_receipt_release.json、extra_cases_release和independent_native_release_verified.json。中间失败及旧日志保留；原先排除的文件头“截断”日志未修。独立PseudoNetCDF检查曾在清理阶段发出非致命警告，最终改用现有SciPy作独立NC3读取，没有修改第三方源码。

复用依据：[xclim GitHub](https://github.com/Ouranosinc/xclim)、[xarray GitHub](https://github.com/pydata/xarray)、[NASA POWER时间定义](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)、[NOAA CMORPH定义](https://www.ncei.noaa.gov/products/climate-data-records/precipitation-cmorph)、[SWAT+逐日降水格式](https://docs.swat.tamu.edu/62.0.1/input-reference/pcp/)。xclim固定版本源文件和许可已取得；xarray指定标签两次获取失败已保存，实际复用已有安装与核查的官方API，不伪称下载到该标签源码。唯一总报告仅追加本节，并同步同名工作副本。


## 2026-11-05 工程端口接回公共随机变异并清理固定插件绕行

本节修正上一轮的范围判断：此前新增工程端口虽然完成了产品适配，却走AlgorithmSelector固定适用性筛选和直接执行，没有接入公共process_current_input。因此把当前融合失败笼统归为缺误差/校准资料是不准确的；该结论不能继续作为平台完整选优链路已接通的证据。旧记录保留，本节给出实际修复及验证范围。

### 通用调用链与清理范围

现在为：prepare_uploads → 工程/产品语义与支持核验 → search_bridge → platform_search.process_current_input → 已有任务匹配验证/随机变异/真实候选执行/选图 → 端口质量门 → 原生写回。移除工程入口fusion_prerequisites前置固定插件门槛、FUSION_INPUTS_INCOMPLETE笼统提示，以及process_port中的直接recommend/execute_with_fallback路径。库内AlgorithmSelector、融合内核、评分、变异、验证策略和其他已经正常的入口保留，不是全局删除这些公共模块，也没有给三个流域写专用分支。

新增工程搜索桥接只转换端口形状/单位/真实坐标/逐日区间，不设置候选组合或赢家。实际输入来源按稳定身份排序形成公共接口所需的primary；验证对象是该声明来源的可见值代理，不宣称独立真值。三个及以上实际地理位置进入已有空间留出路径；单位置序列进入已有时间验证路径，样本不足返回VALIDATION_EVIDENCE_INSUFFICIENT。站点ID不能代替经纬度，两处不规则位置也不伪装成规则栅格。无模型软偏好/物理上界时，扩展现有application_contract使hard-only模式可明确保留未知，不编注册属性、评分权重或范围。

同物理量、单位、时间定义、支持类型、来源集合的工程端口按真实坐标联合搜索，防止把多站任务拆成缺乏空间验证样本的单点。相同位置别名不制造额外站点，搜索后映射回各原生资源和时间索引。各端口复用一个shared_search_id及独立成本记录，不能按重复端口报告重计搜索次数。共同期外记录及原生缺测标记保留。只有全部所需组成功后才交付工程更新；搜索证据不足会带诊断下载，不回退到不相关解析分支。页面对未完成任务改为“尚未生成融合交付”，避免将有候选尝试的任务误说成完全未执行。

### 真实来源与工程复验

使用原三套资料和原包POWER凭据，在原文件哈希不变的前提下，原工程均明确仅剩MODEL_DAY_UNCONFIRMED，不再要求先提供误差表或校准资料才能启动通用搜索。原工程日界仍未确认，因此原工程写回次数为0；没有倒填UTC、补零、删日期或冒充完成。

为独立核验公共数值搜索，对三个地区已有UTC来源在原POWER网格中心做源数据级预览，复用实际派生降水量及原单元提取，没有伪造原工程日界、指定算法或构造独立真值：Mandena执行12个候选/36次提议，Cangandala执行12个候选/29次提议，Humaya执行12个候选/36次提议。三者evolution_used均为true，均保存选中图及最终数值执行轨迹（分别2/3/2个部署节点）。这是三个真实来源任务的变异/验证/数值执行证据，不是原SWAT工程写回，不证明融合精度更高或全局最优。原始1,088个文件哈希核对未变。

另以明确标注的合成完整契约工程，真实通过公共API与当前极简页面执行联合搜索、候选数值管线和9份天气文件写回，验证原生文件绑定、非目标内容保护及共享搜索成本。它是端到端机制证据，不作为真实降水精度证据。单位置100日时间序列也实际进入公共时间验证与随机变异并交付产物；两日样本则明确反馈验证证据不足，没有用静态加权绕行。

### 回归与部署

138项相关回归全部通过，覆盖工程桥接、多端口分组、无sigma真实搜索、验证不足、公共页面/工具入口一致、原生写回保护、单位/日期/坐标冲突、五种模型端口、原降水适配和原应用契约。原固定加权路径的测试断言已改为公共搜索契约或真实证据不足预期；九文件页面测试保留了实际writer执行和下载回读。未改库内加权算法实现，没有删除质量守卫来使测试通过。文件头“截断”日志仍不在本轮修复范围。

运行目录与GitHub工作副本的9个目标源码/测试文件已同步，其他src/integrations文件无变动；正式8502实例重启后HTTP200。没有Git提交或推送，没有模型求解器运行。证据集中在D:\codex\EcoDataEngine\output\engineering_search_reconnect_20261105：修复验收说明.txt、delivery.json、pytest_signedoff.log、real_checks.json、real各案例的公共search请求/验证协议/候选结果/变异轨迹/最终执行、code_sync.json和deployment.json。唯一总报告只追加并同步同名工作副本。


## 2026-11-05 三套新增资料只读审计与真实公共入口复现

本轮按用户要求先检查，不修改平台源码、原始文件、科学声明、算法、评分或参数，不执行上传脚本或原生模型。核实运行目录D:\codex\EcoDataEngine、8502现有实例及工作副本HEAD e28070d23919702afc5759052216f7a5006eb962；HEAD不代表当前无未提交改动。复现调用实际prepare_uploads，并用无处理函数模拟的Streamlit AppTest运行实际app.py页面。没有把AppTest称为浏览器视觉检查。

三份原ZIP逐一完整CRC通过；外包SHA256及解出的45个文件在检查前后相同。两份SWAT作者包MD5与随包保存的发布者校验记录一致，WRFDA完整gzip流CRC通过。实际读入8份新增NetCDF全部数组、4份WRFDA背景/边界NetCDF全部数值数组，检查13个内部符号链接目标均存在。SWAT两份内层ZIP另做完整CRC与file.cio直接引用检查；并未执行求解器，也没有因此认证全部数据库深层引用或作者模拟结果。

|资料|原包直接提交实际结果|另用原字节内层工程和原来源复验|结论|
|---|---|---|---|
|01_WRFDA_真实多源同化资料.zip|页面与API均报“嵌套资料不是可识别的GRIB气象场：be/be.dat”|读到WRFDA的be、ob、rc及3DVAR/4DVAR配置，四份背景/边界数据正常|当前WRF转换器把tar.gz走成仅GRIB强迫分支；WRFDA同化任务尚不在当前已实现流程内，不能将合法be.dat称为损坏|
|02_Tordera_既有工程与多源降水.zip|内层TxtInOut.zip、空间图层ZIP及说明/审计/凭据JSON被列为用途不明，未找到内层工程|SWAT+ rev61.0.2.61工程被识别后，因“相对湿度必须使用0至1比例”异常终止；仅工程提交则返回带同一错误的工程盘点|既有通用接收缺口，也有作者原湿度数据与标准格式冲突；两者不能混为同一个原因|
|03_Saguling_完整工程与多源降水.zip|内层轻量工程ZIP、完整7z与说明/审计/凭据被列为用途不明|使用随包原字节轻量工程ZIP、4原始NetCDF及原凭据，识别SWAT+ rev60.5.7、产品及168个天气变量端口，返回ENGINEERING_INSPECTED，唯一必需确认项为MODEL_DAY_UNCONFIRMED|接收问题确实存在；解开后新增产品适配已有作用，但原降水日界缺信息仍不能安全跨源写回；168是变量端口含重复引用，不是168站|

Tordera实际有86个天气文件、291576个<=-97缺测值；保留掩码，未补零、删日或运行WGN。15个湿度文件累计84656条值超过1，最大100，示例84.66667、99。官方*.hmd规范要求fraction而非百分数（https://swatplus.gitbook.io/io-docs/introduction-1/climate/hmd.cli-and-humidity-data-files）；本轮不能按数值范围擅自除100。现有weather_ports先验证所有气象文件，因此不相关湿度也会中断降水任务；带来源时直接抛异常、仅工程时返回诊断，反馈不一致。这是可复用改进点，但没有本轮修改。Tordera的工程日界也未从现有说明得到确认，公共流程尚被更早的湿度错误挡住。

Saguling实际31个天气文件无<=-97缺测，主索引直接引用齐全。两案例的站点范围与新增格点存在空间覆盖和日期交集：Tordera CMORPH2020-01-20至22/POWER01-18至24，Saguling CMORPH2018-02-03至05/POWER02-01至07；实际区域数组有有限非负值。原始不同网格没有改动。新增来源UTC日定义不等于原SWAT工程统计日已知，不能凭同名日期代替区间确认。WRFDA官方教程明确be.dat为背景误差、ob为观测、rc为背景/边界，另需WRFDA程序及表文件（https://www2.mmm.ucar.edu/wrf/users/wrfda/OnlineTutorial/3dvar/testcase.html）。本轮未接入或运行WRFDA。

三套原包均未进入本次随机变异选图、融合交付或原生写回；没有新的算法精度结论。先前“已验收范围内可运行”不能扩展为任意工程ZIP全自动可处理。后续建议只补通用受限嵌套工程识别、工程根/重复版本消歧、凭据与说明分流、任务类型识别及逐资源质量反馈；保留未知日界和单位质量门，不硬改数据或另写案例专用算法。

证据集中于D:\codex\EcoDataEngine\output\new_three_data_audit_20261105：findings.json、raw_integrity.json、inner_archive_checks.json、netcdf_arrays.json、wrfda_arrays.json、inner_integrity_weather.json、02_humidity_audit.json、geotemporal_overlap.json、outer_results.json、page_results.json、inner_results.json，以及每次真实任务目录、原下载包和日志。检查脚本reproduce.py仅位于output审计目录。本节是检查结论，未作修复完成承诺。


## 2026-11-05 GitHub复用与三套新增原包接收修复（尚未达到全部融合验收）

用户要求寻找GitHub可复用成果并让三个数据全部测试成功。此处必须区分“原包可识别、正确报告阻塞”与“真实融合写回成功”：本轮完成通用接收修复与真实复验，三套仍未完成融合写回，因此不能标作用户要求已全部完成。原始输入、日界、湿度值、模型参数、随机变异、验证、评分与算法赢家均未人工改动。

### 实际复用与修改

核查并保存py7zr（https://github.com/miurahr/py7zr，LGPL-2.1）、WRF/WRFDA官方源码（https://github.com/wrf-model/WRF）和SWAT+官方源码（https://github.com/swat-model/swatplus）。实际安装py7zr1.1.4，在公共入口直接调用其SevenZipFile/WriterFactory流式读取原7z；ZIP/TAR复用Python标准库。上游仓库commit、许可文件、查阅代码及SHA256记录于upstream_reuse.json与upstream目录。WRF/SWAT+官方代码本轮只用于核验格式与依赖，不能说已接入并运行其求解器。

新增archive_inputs.py，由prepare_uploads统一调用：受限展开嵌套ZIP/TAR/7z；保留虚拟相对目录、原字节和全部来源路径；同路径同SHA资源合并，冲突不选版本。TAR内部链接仅在包内核实目标后变成字节副本，不创建文件系统链接；WRFDA时间戳的冒号只允许在只读虚拟POSIX成员名中，默认工程路径限制保留。保留既有纯GRIB依赖包；发现真正工程控制文件后，工程内部引用的附件ZIP保持原样，不作为包装层拆掉。限定5层、10GiB展开字节与成员数量，所有读取及下载均不执行上传脚本。没有给三个地区写专用分支。

新增wrfda_inputs.py识别实际wrfvar1/wrfvar18配置及背景/观测/误差资源，返回明确的WRFDA_BACKEND_NOT_CONNECTED及诊断下载，不再把be.dat误当损坏GRIB。这是任务识别修复，不是WRFDA同化实现。当前PATH没有da_wrfvar.exe，WSL --status返回50，没有配置可调用的同化后端。官方后端需要观测算子及运行依赖（https://github.com/wrf-model/WRF/blob/master/doc/README.DA），不能用原普通WRF/SST转换替代。

weather_ports新增按真实外部来源物理量判断错误相关性：非本次匹配变量的原天气错误单独报告并保留原文件；若错误涉及目标变量仍拒绝。Tordera原湿度84、99、100没有自动除100或被认证合法。降水端口不再因无关湿度先抛异常，因而能够实际识别和适配新增降水；仅提交原工程时仍如实保留天气质量错误。未修改科学取值或削弱目标变量质量门。

### 三套原资料与当前状态

|原包|修复前|修复后实际范围|剩余阻塞|
|---|---|---|---|
|WRFDA|报be/be.dat不是GRIB|展开原TAR、读取配置并追溯背景/观测/误差资源，返回ENGINEERING_INSPECTED及下载|WRFDA_BACKEND_NOT_CONNECTED；没有同化或原生写回|
|Tordera|内层工程/凭据被判用途不明；拆开后因湿度抛异常|原外包自动找到实际SWAT+工程，识别4份原降水产品并真实适配；湿度及原缺测另列|MODEL_DAY_UNCONFIRMED；原工程降水统计日界仍无可靠依据，不能写回|
|Saguling|内层ZIP、7z及凭据被判用途不明|原外包自动识别工程、4份原降水产品并真实适配；作者完整7z与轻量副本的同路径同字节资料归并，工程附件保留|MODEL_DAY_UNCONFIRMED；不能写回|

两套作者发布元数据、Tordera所关联的软件/工作流均实际获取并保存，仍未找到可确认原逐日降水统计日界的依据。UTC新产品不能倒推作者原观测/订正序列的日界。Tordera存在多个观测提供者，不能统一猜为UTC。没有让用户重复确认其已表示不知道的信息，也没有制造科学声明绕过质量门。WRFDA官方代码存在不等于当前机器已有可运行后端。以上是剩余客观条件，不能靠GitHub通用解压工具补成真实融合成功。

### 验证及部署

168项相关测试通过，覆盖嵌套包、7z真实解码、重复/冲突、路径穿越、TAR内部链接、POSIX时间戳、工程附件保留、纯GRIB依赖保留、非目标湿度隔离/目标湿度拒绝，以及既有天气、公共搜索桥接、页面和工程写回回归。pip check无冲突。实际app.py无处理函数模拟的AppTest对三份未修改原包执行页面流程；页面无读取异常、可下载诊断，但状态均为ENGINEERING_INSPECTED，不算融合验收通过。页面不是浏览器视觉验收。

另外用旧WRF/SST真实案例经同一公共入口复验，实际PROJECT_UPDATED、输出ZIP CRC通过、原输入哈希不变，耗时约167秒；它是既有路径回归，不能替代新WRFDA成功。原三份外包SHA256保持不变；Tordera原内层361个文件与Saguling轻量副本307个文件，包括GIS与工程附件，在规范接收副本中逐文件哈希相同。中间附件核验失败与旧日志保留，修正后重新复验，没有删除失败证据。

8个本轮目标源码/依赖/测试文件同步到GitHub工作副本并核对SHA，覆盖前的工作副本目标文件另行备份。没有提交或推送。运行平台8502已重启，健康检查HTTP200；界面、选项及算法/评分/变异机制未改。唯一总报告只追加本节并同步同名副本。

证据：D:\codex\EcoDataEngine\output\three_case_integration_20261105中的upstream_reuse.json、upstream、pytest_release.log、pip_check.log、page_signedoff.json及页面真实任务目录、saguling_signedoff、raw_and_project_integrity.json、wrfda_backend_environment.json、regression_real/wrf/receipt.json、code_sync.json和deployment.json。当前不生成“全部通过”交付包，保留未达成的验收项。
## 2026-11-04 同仓库最新版同步与定向复验

本次仅同步和必要验证，没有新增算法、修改融合/选型逻辑，也没有重新运行大型实验。用户已确认来源和共同仓库均为 https://github.com/1775828095-ux/EcoDataEngine ，使用独立同步分支和PR，不新建仓库、不覆盖主分支。

- 远程main实查为 `e28070d23919702afc5759052216f7a5006eb962`（Integrate shared engineering binding, native writeback and stability diagnostics）。原本地分支为 `codex/latest-platform-20261103`，提交 `ea2b0f51ecd38ded4c29c94fc6b8c959c7ed35f8`。祖先检查通过，main已直接包含本地全部提交，新增一个上游提交，无分叉、无实质合并冲突。
- 同步分支为 `codex/sync-main-20261104`，保留origin原地址和原分支。开始时无已跟踪文件的未提交修改；已有未跟踪测试目录原样保留并补充忽略规则，未纳入提交。
- 源码、配置、依赖、测试、启动脚本与本唯一总报告均随完整Git历史取得。此次额外改动只涉及测试临时目录忽略、协作目录说明、总报告索引和本节记录，不修改业务模块。

### 功能核对及支持边界

|核对项|实际代码依据|本轮验证及边界|
|---|---|---|
|公共工程解析、引用绑定|`src/ecodata/models/project_inventory.py`、`project_binding.py`、`control_flow.py`、`cmaq_binding.py`|依赖解析、动态C-shell绑定及异常拒绝定向测试；不是任意控制脚本执行器|
|五模型输入准备及原生写回|`src/ecodata/models/component_service.py`、`engineering/ports.py`、`engineering/preparation.py`、`engineering/native.py`及`integrations/`|覆盖WRF、CALPUFF、SWAT+、CAMx、CMAQ已支持格式、科学端口与写回契约；未知格式、缺变量或不匹配工程仍可拒绝|
|公共多源处理、质量与评价记录|`src/ecodata/algorithms/automatic_preparation.py`、`platform_search.py`、`src/ecodata/models/fusion_assessment.py`、`task_dispatch.py`|同一公共服务的测试包含实际数值消费、身份隔离、元数据拒绝及下载记录；不等于新增真实精度实验|
|SALib有界输入扰动|`src/ecodata/algorithms/stability.py`；`pyproject.toml`的verification extra、`requirements-verification.txt`|7项诊断测试通过；仅固定工程压力方向的响应检查，不证明融合精度提高或任意误差下稳定|

### 本机依赖与定向检查

读取适用AGENTS及接入说明后，使用已有Python 3.12验证环境，显式将本同步副本的src和两转换包放在PYTHONPATH最前，防止误测D盘旧源码。本机该环境缺SALib及f90nml；只在本项目被忽略的 `outputs/sync-verification-deps/` 安装SALib 1.5.2、multiprocess 0.70.19、dill 0.4.1、f90nml 1.5.0，未升级已有科学依赖，未改动D盘环境。实际版本：pytest 9.1.1、NumPy 2.5.3、SciPy 1.18.1、xarray 2026.7.0、netCDF4 1.7.4、Streamlit 1.64.0。该环境的pip check返回No broken requirements found；这不代表可选能力全部安装。

12份受影响测试共171项：在英文临时目录完成的组合运行164通过、7个页面测试失败；失败日志指向中文工程路径下NetCDF/HDF5创建输出失败。随后以 `git archive HEAD` 将同一e28070d源码原样导出到独立英文临时目录，仅重测这7个页面用例，全部通过。因此171个不同用例均取得通过记录，但**中文项目目录下7项页面路径兼容问题仍未修复**，不能称当前中文路径部署已完全验收。未为此修改NetCDF写入逻辑或测试断言。第一轮沙盒/中文路径受阻和缺f90nml失败均保留，不计为通过。发现缺依赖后复验，未运行全量回归。

定向测试文件：`test_project_inventory.py`、`test_public_model_components.py`、`test_engineering_writeback.py`、`test_unified_engineering_ports.py`、`test_three_engineering_fixes.py`、`test_cmaq_dynamic_binding.py`、`test_data_task_dispatch.py`、`test_input_roles_observation_identity.py`、`test_workbench_local_path.py`、`test_workbench_folder_input.py`、`test_minimal_workbench.py`、`test_stability_diagnostic.py`（均位于tests）。本机日志在被忽略的 `outputs/sync-engineering-final.txt`、`outputs/sync-targeted-verified.txt`、`outputs/sync-stability-tests.txt`、`outputs/sync-stability-final.txt`、`outputs/sync-pages-ascii.txt`。原样英文源码副本为本机临时核验资源，不作为额外平台交付或替代Git源码。

### 启动、资源及未验收事项

`启动平台.bat` / `start_handoff.bat` 调用同目录 `launch_platform.ps1`；该脚本以PSScriptRoot定位本项目的 `.venv/Scripts/python.exe` 和 `src/ecodata/interface/app.py`，不照搬D盘部署目录。本同步副本未创建正式.venv、未重启现有网站；若部署请在英文项目路径按setup脚本安装，并核对转换包安装及依赖，再使用该项目自己的启动器。测试时复用旧环境不等于完成本副本环境安装。

旧案例转换绑定 `integrations/existing_air/existing_air_converters/bindings.json` 和 `integrations/local_swat/local_swat_converter/binding.json` 仍含原开发机路径及指纹，属于有界旧案例配置，不能冒充另一台机器可用的通用资源。此次未擅自替换这些科学资源或捏造新路径。公共上传工程路线与这些旧固定案例配置的限制应分别核查。

仓库保留原有SWAT二进制及其运行库，不新增上传本机环境、密钥、原始大数据、输出或缓存；GitHub源码并不包含所有历史原始工程、观测、实验结果及WRF/CMAQ/CAMx/CALPUFF原生程序。此次没有启动上述原生模型做新积分，没有重新核验历史真实案例的全部物理结果，也未做融合精度验收。历史报告中的本机output路径不意味着对应文件已经随Git传给队友。


## 2026-11-05 WRFDA常规3DVAR真实后端接入与两套原始资料验收

先将此前全部平台更新同步GitHub：main和codex/latest-platform-20261103均确认a4cd41f4b839946bdd31c40e6f34e55596cd18d6，保留远端056a795说明更新与本地实现记录；未发布原数据、环境或密钥。随后按用户确认路线接入官方WRFDA实际求解。本节取代上一节“WRFDA仅能识别、后端未接入”的现状描述；旧失败证据保留。

### 后端与平台公共入口

复用官方WRF v4.7.1，commit f52c197ed39d12e087d02c50f412d90d418f6186，GNU Fortran13.3 serial编译实际da_wrfvar.exe。程序SHA256为1d06f56f3e52adc2a31f119f52df155f9a7879559103e64f5acaa810eb39a67b。源码checkout的已跟踪科学文件未手工修改，官方构建生成的Registry/Fortran/头文件另存source_build_status.txt，不把生成文件误称科学补丁。

本机Windows不能直接运行官方Linux WRFDA，采用独立本地QEMU11.1/WHPX＋Ubuntu24.04最小镜像。未启用Windows可选功能、安装系统服务、申请管理员权限或重启宿主机。QEMU发布SHA512与Ubuntu发布SHA256实际验证；镜像、独立可写overlay、密钥、启动介质保存在源码仓库外D:\codex\.ecodata-wrfda。SSH仅loopback并固定主机密钥；数据不上传外部计算服务器。安装记录、依赖、源码版本、编译日志与连接配置均保留；不把VM与原始大数据提交Git。配置位于当前用户LOCALAPPDATA/EcoDataEngine/wrfda_backend.json。

调用链为prepare_uploads → 受限嵌套接收 → 实际WRFDA配置识别 → 依据分析时刻、背景原生网格、OBSPROC窗口与be.dat唯一绑定 → 官方原生3DVAR → 原生完成/代价与数组检查 → 原生分析场及诊断打包。普通融合、随机变异、验证、评分、模型列表、页面布局与控件不变；只修正已有组件结果的一句状态说明，避免已执行同化仍显示“原生运行未执行”。

原始LITTLE_R不能冒充已处理OBSPROC，4DVAR配置不自动改为3DVAR，未确认的辐射配置不擅自关掉。原背景、观测、误差字节不变。工作配置仅记录必要的原生I/O适配：核实旧背景TITLE后使用官方force_use_old_data开关；num_land_cat未声明时从背景NUM_LAND_CAT读取，已有显式声明冲突则拒绝，不重标土地类别。所有修改及依据进入bindings.json，不是调整同化误差权重、CV选择或迭代参数追求效果。

### 两套实际数据结果

|案例|实际原输入与范围|真实结果|
|---|---|---|
|原美国CONUS60|用户原01_WRFDA_真实多源同化资料.zip；2008-02-05 12UTC；89×59×40背景；9061个原OBSPROC报告|实际生成wrfvar_output，状态COMPONENT_EXPORTED；原生完成标记、时刻网格、有限值/正压力/位温及水汽检查通过；29次内迭代，J从29710.458降到21663.249，Jo从29710.458降到18973.580|
|官方北极案例|官方2015教程原包，2013-12-23 00UTC，180×180×40背景，46454个原OBSPROC报告；原包209546803字节，SHA256 3d44c4b33105c2756595e3af2da1c46665b30c1a167be24292664e4a4516782f|常规3DVAR实际生成分析场并同样通过验收；29次内迭代，J从65977.132降到51177.266，Jo从65977.132降到46339.705|

北极作者原配置为辐射教程配置。本次用户已明确允许仅复验常规3DVAR：原TAR及其原配置保持不变，在platform_inputs附单独的已确认任务范围声明，工作副本明确关闭amsua/amsub辐射开关、传感器初始化与VarBC，原非辐射观测完整送入官方程序；保留配置差异。不能把这次称为原辐射方案通过。报告数也不是质控后实际接纳数，原生QC和O-B/O-A明细随包提供。

两例实际app.py页面经Streamlit AppTest完整处理（没有处理函数模拟）均无异常并生成可下载真结果；不是仅调用识别函数，也不冒充浏览器视觉验收。另用SciPy netcdf_file独立回读两份原生分析场，温度/风场维度及有限值通过；所有下载ZIP CRC通过。原美国ZIP SHA与前次审计一致，北极原TAR SHA与下载收据一致。温度、风、水汽及压力等存在实际分析增量；J/Jo是参与同化资料上的原生目标，不是独立测试RMSE，accuracy_improvement仍为NOT_ESTABLISHED。

### 失败保留与运维验收

保留了原生程序退出码0却因旧数据版本、缺LANDUSE.TBL、类别默认值不符而失败的日志，验收不能只看退出码或文件存在。修复采用官方旧数据兼容开关、官方运行表及背景声明，不伪造数据。第二例还揭示OBSPROC平台字段包含FM-36 TEMP S多词类型；按原INFO_FMT固定宽度读取，未丢掉四条报告。TAR中的作者旧机器绝对链接复用现有>=2路径段唯一后缀绑定，仅在包内重定位，不访问宿主机绝对路径；相同规则适用于同类包。

另做后台完全停止后的冷启动公共入口验收：首次发现cloud-init重复初始化导致SSH身份变化，安全校验拒绝连接，未放宽或自动接受未知主机密钥。固定本机受控VM的既定主机身份，仅对一次性隔离appliance在构建完成后停止cloud-init重复配置；再次关机后，平台自动隐藏启动后端并生成通过检查的真实同化结果。cold_start_signedoff.json明确记录initially_stopped和automatically_started。该修复限于新建隔离后端，不更改宿主机系统服务。

相关回归覆盖已安装/未安装后端、辐射范围确认、原字节保护、I/O兼容、退出码0无输出拒绝、固定宽度多词类型、包内绝对链接重定位，以及既有随机变异/工程端口/界面路径；测试签收见pytest_signedoff.log。pip check无冲突。后端安装工具为tools/bootstrap_wrfda_linux.sh、setup_wrfda_vm.py、register_wrfda_backend.py及run_wrfda_job.py，运维步骤已追加现有docs/格式转化模块接入接口.md。

### 边界与证据

本轮完成的是已确认范围的常规观测3DVAR及两套真实输入自动处理，不能推广成任意模型/任意观测同化均支持。未验收4DVAR、卫星辐射同化、侧边界更新和后续WRF预报；没有独立观测验证精度提高。两套官方教程资料按其训练/测试用途使用。

证据集中D:\codex\EcoDataEngine\output\wrfda_3dvar_integration_20261105：initial_github_sync.json、官方tag/commit、运行下载校验、编译与失败日志、page_acceptance.json、independent_readback.json、cold_start_signedoff.json、原生QC/代价/分析场及回归日志。两套可交接结果放D:\codex\交付\WRFDA_3DVAR_真实验收_20261105，包含两份分析结果ZIP、验收摘要与运行说明。模型程序真实运行、分析场有效、后续预报与独立精度是不同验收项。

官方代码：https://github.com/wrf-model/WRF/tree/v4.7.1/var ；官方数据：https://www2.mmm.ucar.edu/wrf/users/wrfda/download/testdata.html ；北极资料：https://www2.mmm.ucar.edu/wrf/users/wrfda/Tutorials/2015_Aug/class/wrfda_testdata.html 。


## 2026-11-18 DeepSeek按需文件工具与Saguling连续运行验收

用户授权把默认完整深读改为由DS按工作规则主动列目录、读表头、查工程配置和依赖。核对现有Scanner、归档成员读取、scientific元数据读取器、DeepSeek transport、three_phase及下游契约和推荐。原流程在浅扫后无条件Scanner().scan全输入，是本次替换的连接点。新流程保持既有GUI控件，由scan_index只读取根目录／外层归档目录，确认发送后three_phase统一分派file_agent。DS实际使用原生function calling，按返回结果决定下一工具。列目录不解码成员，只有DS选中的成员才通过现有materialize_member流式落盘、scientific读元数据或按页读文本。无默认递归解包、全数组加载或完整字段目录预扫描。参考DeepSeek官方tool_calls协议及MCP filesystem的只读根目录／分页工具设计，复用本地Python能力，未添加Node服务或模型框架。

新增公共file_tools与file_agent；service.tool_turn复用已有认证／网络／重试，deepseek_transport增量拼装原生tool_calls。工具仅有list_directory、read_text、read_header，限制在本次授权输入及内部成员，拒绝越界和任意命令。工作规则只指导调查顺序和证据要求，不给定算法赢家。列表150项、文本16KiB分页，显式返回next_offset；资源不清楚保留UNKNOWN。每次工具调用使用独立子进程和TemporaryDirectory，正常及错误退出清理。单任务最多12轮、48次工具调用，单读取120秒，API回复最多8192token；达到边界返回未完成状态。读完选中元数据后复用原有discovery、catalog_decision、兼容性核验及推荐入口。未检查的文件仍未知，工程依赖完整性不因LLM判断自动PASS。

额外发现GUI静态导入会触发大量计算线程的提交内存。控制实验中，同一空窗口在限制线程前约5.49GiB私有提交，进程启动前将BLAS/OMP/MKL/NumExpr设为1后约153MiB。两个启动器添加默认单线程，保留显式环境配置；读取子进程固定单线程。线程限制可能影响后续数值计算的耗时，本轮未比较算法耗时或效果。不是修改科学算法的数学定义。

真实输入为用户03_Saguling_完整工程与多源降水.zip，613367712字节。前后SHA256一致，d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f。通过实际Qt Window、相同公共three_phase入口及真实DeepSeek调用连续运行两次，未使用模拟回复。第一轮9次工具、检查8份文件，77.34秒；第二轮13次工具、检查11份文件，96.84秒。两轮均由模型主动停止，未耗尽轮数；数量差异为模型自主调查差异，没有挑选有利种子。实际查阅原始副本ZIP内file.cio、time.sim、pcp.cli、指定corrected_pcp降水文件及POWER／CMORPH表头，并按真实目录检查引用。未深读完整7z副本，也未把所有711条深读。目录／文本／表头工具及真实父路径均保存在JSON工具轨迹。

最终版本连续两轮的内存和清理结果如下，峰值含测量到的读取子进程，并按0.15秒抽样。

|指标|结果|
|---|---|
|父子进程合计常驻内存峰值|约492.48MiB|
|父子进程合计私有提交峰值|约363.58MiB|
|第1轮结束常驻／私有提交|约290.37／220.29MiB|
|第2轮结束常驻／私有提交|约293.30／223.19MiB|
|两轮结束临时文件|均0；实际工具记录全部确认临时目录已删除|
|原始包变化|无，SHA一致|

此前完整深读测试常驻峰值约1.85GiB、私有提交峰值约7.12GiB，生成约5.73GiB测试产物；本次为按需检查任务，未再跑原生工程处理，不能把两者当相同算法执行成本。当前证明了此包两次按需调查的峰值、释放和临时清理，没有证明无限次运行无泄漏，抽样也可能遗漏短暂峰值。保存的JSON审计、源码快照和测试记录属于主动保留的证据，和已清理的解压临时文件分开计量。

本次验收为数据和工程调查链路。DS结论、已读字段和依赖证据可交给后续平台处理；未执行新的融合、原生写回或SWAT+求解。历史MODEL_DAY_UNCONFIRMED仍需根据真实工程日界证据核验，工作规则未将其猜成UTC或当地日。界面可在原决策记录查看工具反馈；检查结果暂存为任务file-inspection.json，结果导出携带desktop_file_inspection.json或同等审计字段。未手工操作可见窗口，Qt离屏运行不冒充人工GUI验收。

新增及直接关联测试11项通过（路径越界、未授权工具、输入变化、嵌套引用、未请求成员不读取、原生SSE工具参数拼装、选定NetCDF表头、公共入口只消费请求文件、GUI目录模式、既有表布局回归）。旧网页残留测试因fusion_state已随队友覆盖而收集失败，另有3项旧native-header测试失败；两模块哈希与队友原版一致，错误日志保留，不将它们标PASS，也未为本任务恢复旧网页或改模型处理。旧版原始读取测试和本次新增工具测试应分别解释。

初次联调出现最终输出截断、非JSON回复，按明确的简洁摘要规则、JSON响应模式及8192输出协议修复；两次失败记录保留。v3证明API和工具链两次执行，随后使用启动前线程设置的final版本重新做完整两轮验收，不混用旧内存数据。

证据目录output/on_demand_agent_20261118，最终真实结果在final/result_0.json、result_1.json、memory.json、runs.json及tools_*.jsonl，汇总findings.json；协议、网页来源、测试JUnit和修改前／执行源码指纹同目录。源码修改__main__.py、discovery.py、three_phase.py、service.py、deepseek_transport.py、Start_PySide.bat及本机launch_platform.ps1；新增file_agent.py、file_tools.py和定向测试。主运行目录和原PR4工作目录同步本次源码，未推送GitHub。

发布补充：增加读取期间输入变化拒绝测试，共11项通过（tests_release.xml）；桌面初始提示改为目录模式，两个本机快捷启动入口统一进入当前运行目录。launch_platform.ps1线程设置、根目录启动器及发布前源码已保留。


## 2026-11-18 Saguling平台完整链路续验与日期证据复核

本轮复用原包及Qt桌面公共服务，以真实DeepSeek工具调用检查工程和来源，再经公共决策、推荐、数值处理、产品导出以及原工程接入尝试。原始ZIP未改动。此前“日期不对上”表述不准确：time.sim为2001—2023年，覆盖2018年来源；真正未确认的是模型日降水记录的起止小时，而非年份不重合。

**时间调查**：DS按需读取file.cio、time.sim、降水控制和来源表头、原始获取凭证。POWER原始请求明确time-standard=UTC并经已有凭证校验，CMORPH依据实际时间坐标/bounds按UTC处理。工程日降水未提供UTC/当地日的边界定义；经纬度、文件日期与模型运行年限不足以证明日界。未任意平移日总量，未把推测标为已核实。

**修复**：file_agent原先未将用户task_requirements及显式task_goal持续传入后续公共决策，现修复转发；discovery.check不再将DS因缺独立验证参考给出的pending_goal，误当用户未声明用途。用户明确evaluate_gain后仍执行原单位、物理变量、时空支持检查；没有放宽科学契约，也没有修改融合内核、评分或预置赢家。未明确用途仍不自动开放融合。新增定向用例验证两者区别。

**真实执行**：首轮保存了失败记录；第二轮完成DS调查和两源空间对齐、日降水率到日量转换，以及原工程接入。修复用途转发后复用同一次已冻结的DS决策和输入重新走公共检查，并由真实DS推荐后调用selected_processing实际执行单源/多源图对比。使用一个共同UTC日期2018-02-03、POWER较粗目标网格7×5，共35格；这是小规模链路验收，不代表整个包所有日期完成融合。CMORPH其余两天未纳入本次数值对比。

|项目|实际结果|状态|
|---|---|---|
|真实API调查、源时间核实|查到POWER UTC凭证及CMORPH时间定义；工程日界仍未知|部分完成|
|单位转换、空间对齐、图执行|实际单源9个、多源2个通过评价；无执行失败|PASS|
|选择队列26格，RMSE/MAE|单源7.3313/6.5342，多源16.7647/11.8812 mm；最终保留NASA_POWER单源|PASS，不证明融合提升|
|冻结后9格代理复核|单源RMSE5.7176，多源5.2567 mm；未据此改选方案|仅代理诊断|
|产品ZIP和NetCDF回读|3文件，各1×7×5、mm、全部有限且非负，ZIP CRC通过|PASS|
|SWAT+原生写回|完整原包接入尝试明确返回MODEL_DAY_UNCONFIRMED|未通过|
|SWAT+求解|未运行|未运行|

上述误差参考为PRIMARY_GRID_RECONSTRUCTION_PROXY，非独立雨量站真值；不能由9格代理上的误差下降宣称真实降水精度提升。最后selected.nc与NASA_POWER相同属于保留单源的真实结果，不能将导出产品称为最终多源融合获胜。

**成本与保护**：第二轮调查至工程接入166.94秒，抽样峰值RSS1,171,599,360字节、私有提交1,068,904,448字节；后续数值续跑未监测内存，不将其并入此峰值。数值服务自身0.6003秒，45次算法调用（搜索36、部署3、核验6），失败0；单独保留API和处理日志，不将此时间当完整用户耗时。删除仅本轮生成的展开工程副本、expanded_inputs.zip及测试暂存，释放6,132,968,736字节，原包和实际产品、图、指标、审计、失败记录保留；清理后D盘可用347,340,484,608字节。完整工程后端仍会产生大临时副本，已清理，不宣称其达到按需表头检查同等低内存。

**定向测试**：9项通过，包含真实GUI处理函数的目录模式、按需读取和临时目录释放、输入变化拒绝、SSE工具参数重组、用户目标/调查要求转发、明确用途与缺参考的区别。第一次测试因既有.pytest_tmp权限失败，日志保留，改为本轮独立测试目录后9项全通过；未运行无关全量回归。本轮Qt离屏公共流程并非人工可见浏览器验收。

证据：output/saguling_full_20261118/v2/inspection.json、engineering/required_information.json，continued/result.json、processing/group-2/search-1下图与评价轨迹，continued/Saguling_fusion_result.zip，verification.json、cleanup_manifest.json、tests_final_retry.xml及source_hashes_final.json。真实产品ZIP SHA256为579d3a29621bd18e2e7a50145390216bda43aec2becc46c221d1d090f76a77f5。修改前和执行源码均留存，当前运行目录与PR4工作目录同步本轮源码，未推送GitHub。结论：已完成小规模真实数值处理、对照与导出；没有证明融合优于单源，工程日界缺证导致原生写回未完成，不能标全工程任务成功。

## 2026-11-18 Saguling显式UTC工作约定下的原生写回链路验收

用户要求先验证链路可用。本轮不再寻找日界证据、不改科学算法或关闭质量门，而是复用既有ecodata.engineering-semantics.v1资源声明，为实际工程引用的降水端口明确声明time_basis=UTC。本声明只用于链路测试，另附test_assumption.json标记WORKING_ASSUMPTION_ONLY和original_model_day_verified=false，随输出工程保留；confirmed表示本轮主动采用这一测试约定，不表示原作者历史定义已证实。原输入ZIP保持不变，不修改来源原始日期或人为拆分日总量。没有新增页面选项，也没有任务专用算法补丁。

执行入口为公共automatic_preparation.prepare_uploads，经已有归档规范化、工程语义/端口绑定、产品适配、公共platform_search真实搜索、双层质量门、SWAT+原生替换及打包回读。使用两种实际来源（POWER与三天CMORPH），全部28项端口任务共用一个真实搜索执行记录，不将重复端口回执计为28次独立搜索。更新共同日期3天，原降水文件每份8400日；其余日期与原生缺测记录保留。最终选图为spatial.source与spatial.observations进入spatial.residual_idw，实际执行结果PASS；这不是独立雨量精度评价，也不据此宣称融合优于原工程。

|验收项|结果|
|---|---|
|按显式约定进入工程端口处理|PASS，原先MODEL_DAY_UNCONFIRMED不再阻止本轮已声明约定的任务|
|公共数值处理、搜索及质量门|PASS，保留真实请求、图、评价和部署记录|
|原生降水更新|PASS，27份文件的前后SHA不同|
|独立原生回读|PASS，逐份27文件按原生pcp解析并核对输出SHA；公共流程同时校验ZIP CRC及所有非目标成员未改变|
|原始输入不变|PASS，SHA复核一致|
|作者工程原始日界已核实|未证实，本次未改变此结论|
|SWAT+求解/独立精度提高|未运行/未验证|

总执行206.58秒；未新增内存峰值测量，不用以前峰值冒充本轮测量。清理本次两级解包/展开ZIP副本，释放11,118,222,147字节；保留约0.89GB的实际完整更新工程、原始输入及决策证据。没有改动平台生产源码，本轮改动为独立测试操作脚本和声明，复用现成公共机制。

证据目录output/saguling_full_20261118/native_utc_chain：engineering_semantics.json、test_assumption.json、result.json、verification.json、native_readback.json、cleanup.json、cleanup_extra.json；实际完整工程processing/engineering_inputs/EcoData-processed-project.zip，公共搜索轨迹在该目录0/decision/search。脚本test_native_chain.py与check_native_output.py在上一级。本轮结论：在明确测试约定下，数据到数值处理到原生工程写回与回读链路已跑通；科学日界定义、模型运行和精度收益仍分别未验证。


## 2026-11-18 DS推荐组合到真实变异搜索的公共连接验收

### 原缺口与本轮公共改动

原工程入口没有将DS选中的组合传入搜索；另一些入口仅把步骤压平为initial_algorithms，不能表达串联父边，不能证明以推荐图继续变异。本轮保留现有随机变异、误差排名、质量门、原生写回及历史结果，只接入推荐资源与合法探索，不增加算法。

1. platform_search.with_recommendations建立公共建议协议：带变量作用域、步骤顺序、推荐ID和理由；不接收赢家分数或预置fitness。未知作用域明确报缺条件；其他变量的建议不应用到当前端口。建议数量最多16，每条最多8步。
2. desktop.recommendations.validate从已核验任务上下文绑定变量；主页面execute_original把已选择推荐传给service.run_original_workbench，服务通过同一个公共函数传入工程请求。independent_alignment也传入完整组合，不只传算法名。页面未新增控件，工具包与页面处理函数共用此服务。
3. engineering.preparation复用既有GridRegridder执行推荐的保守重网格，按真实网格关系取共同较粗支持、保留时间bounds并记录输入输出指纹；不把未执行的对齐冒充已完成。对齐后的实际数组再进入端口提取和搜索。新增逻辑不含Saguling路径、日期、站名或指定获胜源。
4. spatial_fusion_task按步骤编译真实DAG；blend→残差时，下游实际父边为上游blend输出和合法conditioning观测，不能退回原始输入。缺背景、禁止拟合、不支持步骤、错误串联及未完成前置处理明确拒绝；通过现有类型化执行器端口验证后，按固定留出队列真实评价。失败/预算未执行者不成为推荐父图。
5. 推荐图与普通候选共用去重和预算。每个已通过的可参数变异推荐图获得一次有界探索机会，参数从现有域中选择不同值；随后回到既有best-parent随机变异。所有最终选择仍按真实RMSE，不给DS建议奖励分，不保证其留到最终。无推荐时保持原默认搜索行为。后端仍是现有random_graph_search，不把它称作官方GOLEM优化器或全局最优证明。

### 原包真实验收与关闭推荐对照

复用同一Saguling原包（SHA256 d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f）、前次真实DeepSeek回复及本轮已声明的UTC日工作约定；没有重新抽取有利DS答案，没有更改作者历史日界证据。两个来源分别为POWER与CMORPH，3个共同日期，28个端口回执共用一个实际搜索。所有来源先做真实保守对齐，27个工程位置；63个留出代理目标参与选型，原工程数据并非独立雨量站真值。

|机制验收|真实证据|结果|
|---|---|---|
|推荐有执行消费者|保守对齐产生新数组；DS的blend图最先按recommended_initial执行|PASS|
|推荐图的真实评价|权重0.5，RMSE23.9399584 mm、MAE20.7685484 mm，63目标|PASS|
|以推荐图继续变异|首个变异父fc2d0aa32864cb5c2a7e，子0e36ba06456a4198fcf7，保持同一对源父边，权重0.5→0.75|PASS|
|子图真实效果|子图实际执行，RMSE14.0866447 mm、MAE12.2979029 mm；非模拟分数|PASS|
|关闭推荐行为对照|相同已对齐输入字节、种子11、预算和完整评价协议；首个父图变为IDW，普通algorithm操作；没有recommendation_parameters|PASS|
|最终误差是否更优|开启与关闭均RMSE9.4677636 mm、MAE3.4259807 mm；最终均选IDW(power=3)|未证明额外优势|
|完整原生写回与独立回读|27份pcp文件SHA改变，逐份原生解析通过，ZIP CRC及非目标成员保护通过；原包SHA未变|PASS|
|SWAT+求解及真实环境精度|未运行模型；代理参考不是独立雨量真值，UTC日仍是测试约定|未运行/未验证|

两种搜索均实际评价80个不同图，搜索阶段73个新变异图，其余初始化；各249次算法调用且失败0。沿用工程入口既有上限：objective/total calls=1800、proposal=400000、wall=3600秒；两组实际各触发400000次有界提议，其中399927次未形成新数值评价，不能当作400000次真实执行。数值服务用时开启20.4636秒、关闭20.0706秒，不是端到端全部耗时。关闭推荐对照复用相同预处理输入，不能凭这一对照宣称总成本优势。现有去重后重复提议较多的效率问题未在本轮另行重构或掩盖。

本次证明推荐改变了首先验证的图和变异路径，并且推荐组合经真实变异得到更低的代理误差；没有证明相对既有搜索的最终精度或效率优势。最终保留IDW而非强行融合，是按相同指标选型的真实负结果。没有宣称原始观测、独立测站真值或全局最优。

### 测试、失败、成本与保护

新增及受影响定向回归45项通过：顺序父边、推荐真实执行和实际变异父图、重复图不重复评价、非法串联、拟合权限、未完成保守处理、未知算法拒绝、关闭推荐、变量作用域、真实保守数组转换、页面处理函数与工具包同公共请求、既有空间各关系路径和按需文件工具回归。未跑无关全量回归。Qt处理函数离屏测试不是人工操作可见窗口验收。

保留首轮缺time坐标处理测试失败并修复为合法静态栅格也能使用公共对齐；Windows跨盘硬链接失败后改为清理得掉的临时输入副本。完整工程处理实际成功后，初次对照断言因内存元组与持久化JSON数组类型差异失败；保存结果规范JSON比较显示输入指纹、预算、种子和全部评价协议一致，已修正断言并独立核验现有产品，没有重跑寻找好结果。完整墙钟因该断言未最终落盘，标未记录；两组数值服务成本和所有失败日志保留，不用估算值冒充测量。

本次解包/展开临时副本释放6,132,565,887字节，另删除本次613,367,712字节输入副本；保留原始数据、旧结果、本次完整更新工程和决策轨迹。未新增本轮内存峰值测量。没有改变单位、时区事实、质量门、参考队列或数学内核来追求胜出。

新版本spatial_relationship_fusion_v1_3_verified_recommendations。证据output/ds_recommendation_search_20261118：source_before/source_executed、source_hashes.json、tests3.xml、native_v2/provenance.json、ds_recommendations.json、verification.json、processing/engineering_inputs/0/decision/search下真实图、评价和transitions、recommendation_off下同条件对照、processing/engineering_inputs/EcoData-processed-project.zip。完整源码指纹逐项保存，旧产物不覆盖，未推送GitHub。

支持边界：本轮贯通已有连续量空间融合/工程端口及既有独立格网对齐搜索中的推荐组合；支持当前空间组件所定义的blend、IDW、残差和显式允许的拟合节点及其合法顺序。没有把所有历史时间、神经模型和模型专用后端一律宣称迁移；不支持的推荐必须记录拒绝，不能静默当成已验证。DS建议是可否决的探索起点，不是精度预测模型。


## 2026-11-18 推荐依据展示与用户选择已评价方案

本轮按用户要求增加推荐解释和候选选择，保留当前Qt页面主题、模型列表、默认选型、科学内核与质量门。新增用户选择是本轮明确授权的交互扩展，不恢复旧网页或添加实验专用规则。

### 推荐解释

原页面已有一段推荐理由。本轮在recommendations公共模块依据实际任务上下文构造explanation：来源/字段ID、变量、单位、原始时间统计和采样信息、真实网格间距，与DS推荐理由、公共检查前提、仍缺的信息分开保存。无法核实的字段显示UNKNOWN，不以模型文字充当文件证据。页面每张推荐卡增加“展开数据依据与检查前提”，展开后可查看原记录；平台补充方案的理由保持原标注，不把建议包装为效果保证。未伪造评分、权重依据或动态思考过程；任务执行进度仍由既有真实进度回调显示。

### 用户选择候选与真实产品

结果页新增“已验证候选”下拉框，显示变异/已评价来源、方法、RMSE、默认标记和参数；保留同队列MAE、样本数、代理参考限制。各任务最多显示前10种验证预测行为；已有same_validation_predictions_as用于合并同验证预测的重复图，不能将重复名称计作额外效果。默认工程/结果ZIP与“导出所选方案数据产品”是两个明确分开的按钮。

公共platform_search现在为支持的空间搜索暂存可回放的实际输入及哈希；新增execute_evaluated_candidate只接受已有PASS记录、非空且相同评价支持、相同顺序/角色/元数据和字节哈希的输入。它调用既有冻结图执行器，重新执行用户选定图、通过产品质量检查与NetCDF回读后导出selected-product.nc、graph.json、selection.json。没有在选择后重搜或重算排名，原通用execute_frozen_graph对未知current_input迁移的限制仍保留。输入变更、无评价候选或评价队列不明时拒绝，不能沿用旧分数。

产品包明确记录USER_SELECTED、原默认图、所选图、原验证指标、参考依据、部署轨迹与成本；native_writeback=false、native_model_executed=false，不能冒充已替换默认原生工程。比较当前产品与默认产品，数值相同或不同均如实记录并显示，不把结构更复杂或用户手动选择当作效果提升。临时重放目录在成功/失败后自动清理，当前任务输入暂存随既有任务目录关闭清理；用户主动导出的包保留。

### 验收

联合定向回归49项通过；导出细节补充后，最终5项候选/页面定向测试再次通过，包括新增页面导出处理函数真正使用用户选择、原输入不变、产品生成、原默认结果不变、未知候选拒绝、输入篡改拒绝、公共API同输入隔离、依据与UNKNOWN展示和展开控件。没有跑无关全量回归。

复用前轮Saguling真实运行及真实DeepSeek回复，只补入旧结果遗漏的回放目录引用并从原评价记录重新组织显示，不更改图、分数或评价队列。通过当前公共执行器生成变异候选3f80dca44e83cf46e854的真实产品（IDW→blend，weight=0.25；原代理RMSE9.4892123 mm、MAE3.5262999 mm，63样本），完成ZIP和NetCDF回读，输出3个日期×27位置，非负且有限。此次默认工程未修改，没有重新下载、重做全年准备或重跑大规模搜索。

该例所选方案部署在当前输入上与默认产品数值相同（最大差0），尽管其留出代理评分不同；已明确保留比较结果，不声称额外精度收益。新功能提供可核查选择，不保证任意备选更优。独立实测真值仍缺，原工程UTC日仍是前轮测试约定，SWAT+未运行。

Qt离屏截图已实际渲染并检查文字及布局；离屏字体库未自动发现系统字库，验收脚本显式载入本机字体后重渲，不修改生产主题。未进行人工可见窗口或浏览器操作验收。初次导出遇到既有current_input冻结迁移保护，随后增加经过同输入和已评价记录检查的公共“用户选择部署”入口，未简单删除保护条件；初次失败日志保留。

证据目录output/recommendation_explanations_20261118：source_before/source_executed、source_hashes.json、tests4.xml、tests_export_final.xml、acceptance.json、推荐依据页面.png、候选选择页面.png，以及所选变异方案_数据产品.zip。当前支持具有实际空间搜索、类型化图与回放资源的任务；没有回放资源的旧入口不显示可用选择，不能冒称所有历史模型任务均可更换原生工程方案。源码同步至本机PR4工作目录时仅覆盖与修改前一致的文件，未推送GitHub。


## 2026-11-19 转换后最终检查关卡：SWAT+真实读入验收

本轮复用前轮Saguling真实转换工程和冻结数据，不重跑融合选型、搜索或大规模研究实验；在平台公共转换末尾接通最终关卡。原始用户包、前轮工程、原始输入和历史结果保留。新输出采用独立目录output/final_model_gate_20261119/release。

### 公共调用链与职责

automatic_preparation.prepare_uploads完成原转换后统一调用models.final_check.attach_final_check。最终关卡读取本次实际输出工程，不引用旧机器默认工程：验证归档、写回SHA、模型版本、天气字段/元数据/日期/有限值及缺测规则、CIO/天气站/索引/WGN/对象关联；随后只选择登记且哈希校验通过的匹配模型程序进行有界试读。

SWAT+复用swat_runner.check_project、set_dates、engine_path、start_job及其返回码和新success.fin联合判据；不执行上传的程序、脚本或DLL。既有禁止程序后缀规则提取为共享常量，最终依赖也消费同一规则，不复制新的科学内核。日期仅在临时副本中按既有setter限制为一天，优先选原模拟期内真实更新的首日。原time.sim、file.cio、模型参数和未更新数据不因此改变。试读证据包含程序SHA、输入SHA、原/试读时间控制SHA、日志、返回码和新成功标记；结束清理临时工作目录。

公共service.review_model_output由Qt结果页和工具包共同调用，复用现有DeepSeek transport、预览和输出复核协议。DS收到登记模型要求、实际CIO版本、101项实际活动资源、100条控制引用、31份天气检查结果及本地原生回执。只提供必要元数据/有界日志，不把整个数组发给DS。AI回复不能覆盖本地格式/引用失败或把未试读模型说成通过；无钥匙/网络失败明确保持未完成。相同请求指纹命中已有复核，要求或证据变化后重新核对，保留独立版本回执，避免ZIP重复成员。

### 原包真实结果

|检查层|实际结果|结论|
|---|---|---|
|转换文件格式|31份原生气象文件通过已有表头、字段、日记录、日期连续性、数值和缺测规则检查；27份写回文件SHA与记录一致|PASS|
|工程引用|复用版本感知检查及天气/WGN/对象关联；实际file.cio为60.5.7|PASS（现有检查范围）|
|原生程序|已登记官方60.5.7程序，SHA256 de01a927eebe00b4467ff53424175e247e7808e0bee99790f3d75db1b0a64179|匹配且实际执行|
|模型试读|临时副本仅运行2018-02-03，日志逐项读取降水、温度、辐射、湿度、风速、WGN和天气站，出现执行成功；返回0且新success.fin非空|PASS，model_read_confirmed=true|
|DS实际调用|DeepSeek真实API最终核对完成，补入实际资源与控制引用后missing_data为空；仍提示完整模拟期、通用参考画像范围和科学定义待核实|COMPLETED，非科学全面通过|
|原模拟期及精度|未运行2001—2023完整模拟期；UTC日仍为前轮明确工作假设；没有独立雨量真值评价|未验证|

独立核对最终工程与原转换工程的全部既有成员CRC/大小一致；原生试读101个实际输入逐个SHA与最终工程匹配，唯一例外是已记录的临时时间控制变化。原转换归档SHA未改变。最新验收收据为result_current.json中final_check.final_receipt所指complete.2.json，归档无重复成员，ZIP回读通过。

### 发现并修复的资源与证据问题

初版将平铺工程中的作者旧输出也装入试读副本，涉及3,423,781,737字节；虽试读成功但浪费资源。现复用project_inventory的CIO绑定并沿实际字面文件引用补齐依赖，只装入101项、14,150,741字节；68项未到达成员共3,409,630,996字节不进入模型输入副本。未根据Saguling文件名或固定日期筛选，未知隐式依赖仍由检查/原生失败明确暴露，不能据此宣称所有模型版本和分支已覆盖。排除的作者历史输出在交付原工程中仍保留，没有删除用户资料。

另发现原DS复核只收到通用产品files列表，原生工程场景为空而产生不必要缺项建议。已传入真实活动资源和引用关系，并告知通用产品空列表不等于工程缺文件；重新真实核对后不再列确定数据缺失。DS关于62通用画像与60.5.7工程、完整时段和工作假设的意见保留为待核实信息，不覆盖匹配60.5.7程序的实际读入成功证据。

闭包试读及该轮初次DS复核57.235秒，采样进程族峰值RSS549,879,808字节、私有提交493,543,424字节；不含可见Qt窗口，也不能宣称任意大工程长期无泄漏。资源证据补齐后的最终DS复核15.359秒，9025输入token/1285输出token，单独记账，不冒充一次请求。归档完整性仍需流式CRC读取，不等于把全部历史输出深读成数组。所有native-read临时目录均移除；删除本轮被发布版取代的重复完整工程副本和测试暂存，明细cleanup.json；仅保留发布工程及日志/收据/源码证据。

### 测试和支持边界

最终联合35项定向回归通过；原生运行器及最终关卡37项定向检查通过（两组含重复，不累加为独立用例数）。覆盖格式通过但未运行、工程失败与天气格式分离、日期重复、程序缺失、AI不能提权、同请求缓存、公共转换钩子、任意文件名的工作假设声明、重传历史收据、其他模型旧运行证据保留、非依赖旧输出不加载、上传DLL拒绝，以及既有版本、日期副本、超时、旧成功标记和进程状态检查。旧Streamlit页面测试未运行，未跑无关全量回归。最初合成夹具缺WGN条件及UTF编码别名错误均保留失败日志并修复，未降低生产约束；三个真实短试读均成功，保留修复前后日志，没有按结果调参。

本轮实际完成SWAT+最终格式/工程/匹配程序试读适配；公共入口覆盖其他模型时，尚无对应最终适配者明确UNKNOWN或NOT_RECHECKED，保留其原流程运行记录，不冒称五个模型的新最终试读全部完成。页面分别显示“原生格式检查”“工程引用检查”“模型实际试读”，已通过短试读也明确完整时段和精度未验证。仅回读、仅改后缀或AI认可均不能替代原生试读。

源码：新增models/final_check.py；连接algorithms/automatic_preparation.py、desktop/service.py、output_review.py及现有Qt回调，复用interface/swat_runner.py并共享已有安全后缀常量。source_before/source_executed及source_hashes.json保存版本，程序及请求/输入/输出指纹与实际日志同目录。独立核验release/verification_current.json，运行日志release/current_checks/runner.log，真实DS回复release/enriched_review.json，当前完整工程release/Saguling_最终检查工程.zip。未改算法、质量门和用户原始数据，未推送GitHub。


## 2026-11-19 四个模型接入公共最终检查：有界集成验收

本轮在SWAT+已有最终关卡上扩展WRF、CAMx、CMAQ、CALPUFF分派；不重跑融合、搜索或科学实验。页面与工具包继续沿用automatic_preparation.prepare_uploads → attach_final_check → check_model_output → desktop.service.review_model_output，同一份本地结果进入DS并写入交付副本。无数据包名、日期、站点或赢家规则。原始数据、旧SWAT+成功证据和历史结果保留。

### 修改与支持范围

- models/final_native_check.py复用project_binding/ProjectFiles的实际依赖、CMAQ的有界C-shell静态绑定、CALMET/CALPUFF的既有dependency_plan/read_surface、现有程序登记指纹与外部原生运行契约；仅流式复制激活的NetCDF依赖到临时目录检查表头、维度、字段/单位及TFLAG。这里不是全数组数值/全部化学物种认证。
- WRF复用wrf_external_project实际检查器，提取inspect_project_files供已绑定输入调用；原inspect_project API兼容。检查不等于预报运行。现有外部检查仅支持单域及显式时间分量，嵌套工程返回明确未通过；WRFDA不能代替WRF forecast。当前未注册可匹配任意上传工程的预报运行契约，原生试读保持NOT_RUN，不能称WRF通用运行已完成。
- CAMx复用已登记7.40程序SHA及既有inspect原生契约。程序可用且工程符合现有短运行/网格/化学参数契约时，复制实际绑定输入，改写副本路径，直接启动固定程序并限定超时，绝不运行上传脚本。当前主机登记程序不存在，此执行分支未获本机真实模型运行验收。
- CMAQ复用已登记5.5程序及inspect_manifest。核心气象绑定不等于完整化学运行清单，project明确为UNKNOWN；只有唯一ecodata.cmaq.local-project.v1清单、机制、全部输入及运行契约验收后才能直接调用固定CCTM。缺清单、缺程序分别提示。当前程序及所需登记资源不可用，真实模型运行未验收。
- CALPUFF复用已有5.8依赖闭包与SURF回读，程序SHA固定、控制版本匹配后在新临时目录直接有界调用，要求本次日志及成功终止证据。支持范围不是全部CALPUFF版本/源类型；不运行包内EXE/DLL。二进制气象内部仍需模型验证。
- final_check.py统一分派、保存日志/回执；output_review.py将原生文件检查、工程范围/原因交给DS，分别展示格式、引用、实际试读。旧native_ready不能覆盖新关卡失败/未知；AI不能抬升本地失败。旧native_model_executed仅作为历史证据保留，不冒充本次试读。

### 本轮实际验收结果（不是新融合结果）

| 模型 | 实际资料/范围 | 本地检查 | 模型试读 | 结论 |
|---|---|---|---|---|
| WRF | 轻量包内现有WRF工程，只读解包副本 | 实际依赖及NetCDF已检查；原检查器报告嵌套域不支持、start_minute未显式提供 | NOT_RUN | 公共关卡接通；该真实工程未通过当前检查范围，不推断原工程科学数据错误，不称模型已读入 |
| CALPUFF | 轻量包内原有5.8完整工程副本 | dependency_plan/SURF回读PASS；保留范围说明 | NOT_RUN：本机无固定指纹CALPUFF程序 | 已完成真实公共检查与DS复核；未完成原生运行 |
| CAMx | 桌面02_CAMx原包（含上游tgz） | ZIP/指纹核验；找不到直接可绑定的本次原生工程，未标格式PASS | NOT_RUN | 这是原始上游归档，不是已写回工程；本轮未重新展开数GB工程及重跑融合，最终工程级正例仍未验收 |
| CMAQ | 当前检索路径未找到原有完整CMAQ包 | 公共分派、已有静态绑定及负例定向测试通过；完整真实工程未验收 | NOT_RUN：登记程序不存在，运行清单亦需核对 | 不以合成边界测试冒充真实CCTM验收 |

真实DS请求通过desktop.service.review_model_output对CALPUFF工程副本执行，约7.172秒，ai_review=COMPLETED，verdict=needs_verification，错误为空；本地native仍NOT_RUN。输出CALPUFF_final_gate_copy.zip包含本地final_check及DS complete回执，供下载复核。这是已有工程副本的最终核对，不是本轮新增融合或精度改善；未新做人工GUI/浏览器操作验收。

### 测试、证据与历史保护

64项新增/受影响定向测试通过，0失败0错误（tests_final.xml）：四模型公共分派、WRF真实格式检查器调用、错误维度、报告包拒绝、CAMx缺TFLAG、缺程序/程序哈希错误、旧ready不得覆盖最终失败、SWAT+旧关卡、CMAQ动态绑定、桌面候选交付及推荐搜索相关回归。合成fixtures仅为行为测试。初次pytest因旧.pytest_tmp权限失败，改用独立basetemp；随后旧测试预期NOT_RECHECKED与新关卡的NOT_RUN不一致，更新为“本次未运行但保留previous_native_execution”，未删除测试或改写历史。

本轮证据统一位于output/final_four_models_20261119：source_before（修改前三份源文件）、source_executed、source_hashes.json、acceptance_protocol.json、runtime_availability.json、三项actual/final_check.json、real_api_review.json、page_status.txt、verification.json、tests_final.xml。三份实际输入的SHA复核均一致。只清理本轮解出的WRF/CALPUFF临时检查副本421761077字节，原用户轻量包及桌面CAMx原包未改；保留有回执的CALPUFF交付副本及全部失败记录。未推送GitHub。

复现定向测试（项目根目录，先限制BLAS线程并设置PYTHONPATH=src;integrations/existing_air;integrations/local_swat）：
`.venv/Scripts/python.exe -m pytest tests/test_final_model_check.py tests/test_final_four_models.py tests/test_wrf_external_project.py tests/test_cmaq_dynamic_binding.py tests/test_desktop_candidate_choice.py tests/test_recommendation_search.py --basetemp=output/自行选新目录 -q`

验收边界：四模型不再落入“无最终适配器”的统一空分支，公共检查与DS复核已接入；不能据此宣称四模型均真实试读通过。当前无新增原生模型运行，WRF通用预报运行契约、CAMx/CMAQ完整真实产物及匹配运行环境仍未验收。格式检查、依赖检查、实际模型运行与科学精度四层继续分别记录。


## 2026-11-19 模拟技术引擎：桌面主流程简化与格式说明

按用户最新授权修改当前Qt桌面入口，未重建页面、改科学算法/评分/预算或重跑历史大型实验。界面名称与窗口标题改为“模拟技术引擎”；桌面原“EcoData Engine 1.0.lnk”备份后改名为“模拟技术引擎.lnk”，保持指向当前项目launch_platform.ps1，继续从用户环境变量读取密钥，不要求每次粘贴。

操作主线改为：选择文件/文件夹（或拖入/输入路径）→选择目标模型→一次“开始自动处理”。内部继续使用既有scan_index（只列目录）、file_agent/three_phase、公共建议与契约核验、原有处理与最终模型检查。目录准备完成后自动发起DS按需检查，核验后的可执行推荐自动加入原有处理流程；原始输入只读。旧手动扫描、确认发送、算法ID输入、重复执行按钮不在主操作面展示；JSON、分批记录与API设置折叠到技术记录。推荐理由保留可查看；已经数值评价的其他方案只在实际存在时显示为可选项。没有合法方案、仍有失败、整包工程混有未放行任务或整包与独立任务不能混用时，明确暂停并提供原因，不以自动化绕过原合同与质量门。

格式说明使用公共output_review.format_conversion与conversion_text，结果页和下载包共用。明确原始数据格式、实际输出格式、已记录的来源→目标路径、内容更新与格式改变的区别、实际写回文件清单；ZIP只标为下载容器。后缀标签不等于完整编码认证，具体模型原生结构与试读状态仍由final_check判定，来源不足明确未确认。逐文件来源关系缺失时保留written_files_without_source_mapping，不捏造追溯。导出附desktop_format_conversion.json；重复导出原工程时不覆盖同名历史记录。

复用Saguling历史handoff进行显示核对：保存记录显示NetCDF→SWAT+降水文本（.pcp），实际写回27个文件，明确仅2条任务来源→目标关系可直接追溯；其余文件不人为补来源。此项仅显示旧成果，未重跑融合或模型，也没有赋予新模型运行/精度PASS。

验收：tests_verified.xml记录48项新增及受影响测试通过；末次UI文案与格式修正后tests_last.xml记录12项相关检查通过（含重复项，不相加为独立测试数）。覆盖一次开始后自动进入公共决策、重复提交阻止、缺密钥、合格方案自动继续、未放行整包不执行、无可执行方案不伪装融合成功、格式不变提示、未知格式、下载记录回读与原件不变；亦包含已有最终关卡/桌面候选/按需工具回归。初次新增测试有一处括号语法错误，已修复并通过。测试中的模拟科学后端仅验证UI行为，不作为模型运行或精度证据。

另通过Qt实际start按钮+真实DeepSeek API，使用现有examples/ai_round1/cmaq/A_existing_case开发资料开展有界检查。首次运行20.578秒，DS最终JSON语法无效；失败截图与日志保留failed_ui_01。修复为只在既有12轮预算内允许一次最终说明格式修正，修正阶段不允许重新调用文件工具，连续失败返回中文说明，不能无限重试。修复后同一资料通过真实界面自动衔接，93.609秒，读到4份资料，无异常/超时；没有通过核验的可执行处理方案，正确提示补条件、尚未运行融合，未生成新数据产品。此例证明按需检查与自动衔接/真实暂停，不证明融合转换成功；未换好样本追求成功。已有数值处理回归仍通过。

证据统一保存在output/desktop_flow_20261119：source_before、source_current、source_hashes.json、tests_verified.xml、tests_last.xml、home.png、format_result.png、saguling_format_display.json、ui_acceptance.py、real_api_ui.json及日志/截图、failed_ui_01。截图为Qt offscreen实际渲染/事件循环，并非人工桌面点击验收。已将正常桌面入口重新启动到新代码；调试过程中误继承offscreen环境启动的空闲测试实例已结束，未结束用户数据任务。未推送GitHub，未改原始用户数据。


## 2026-11-19 自动入口衔接修复：Saguling原包页面全链路验收

### 根因与公共修复

| 实际断点 | 公共修改 | 验证与边界 |
|---|---|---|
| DS工具已读控制，但原始字节未进入native_dispatch；未读file.cio被当成空坏文件 | file_tools/FileTools按字节偏移累计文本；只有完整覆盖才登记original_probe_payload，单独保留表头/片段状态 | 完整控制与分页行为测试通过；未读控制返回“尚未完整读取”，不谎报simulation/climate缺失 |
| 已读取原始下载回执却未读取对应数据头；已有POWER UTC核验器缺少回执 | read_header携带精确同名receipt副本；已读且含url/sha256回执至多补核4个对应表头，计入原48次工具预算 | 不遍历全部数据；日界仍由已有哈希绑定URL检查决定，不从文件名或相关性猜测 |
| 目录列过但未打开的依赖被报为实际缺失 | 传递已列目录的存在证据；按需不完整目录下返回尚未核对，完整工程仍在处理时实际绑定 | 元数据存在不当作内容/模型验收PASS |
| 显式任务目标在浅入口丢失，处理建议停在独立格点模块 | 主界面明确“评估多源处理价值并准备模型输入”；浅/深公共catalog保留evaluate_gain；识别实际工程后转到既有工程输入端口服务 | 状态为待工程核验，不是质量PASS；单位/物理量冲突和最终关卡不绕过。工程端口、搜索、写回均复用原实现 |
| 原模型日界历史约定不在用户原包 | 新增通用models.input_context：仅用原文件SHA256和model_id复用已登记科学上下文，增补已有engineering-semantics声明及原工作假设文档 | 不含算法偏好、图或赢家；换文件名可匹配，改内容/目标模型不匹配。此包上下文是从历史真实输入配置迁移的数据记录，不是代码中的Saguling分支。WORKING_ASSUMPTION_ONLY全程保留 |

原始POWER回执明确time-standard=UTC并与原文件SHA一致，因此可作为来源UTC证据；原生站点日界仍没有被历史文献/元数据证实。历史配置中的confirmed字段表示当时采用该声明，不能升级为作者时间定义已证实；本轮科学状态明确为WORKING_ASSUMPTIONS。

### 实际页面执行（未模拟DS、数值或模型）

使用原用户03_Saguling_完整工程与多源降水.zip，经当前Qt Window的开始按钮事件 → scan_index → file_agent真实DeepSeek → discovery/推荐 → run_original_workbench → prepare_uploads → 公共端口处理/真实变异搜索 → 原生写回 → final_check → 真实DS最终复核 → 页面下载按钮导出。没有调用单独Saguling转换器或向候选生成提供真值。GUI事件循环为offscreen Qt自动验收，截图与导出来自实际组件；不是人工浏览器验收。

第一轮67.828秒未形成候选：DS读POWER回执但漏其表头，仅有一份可用科学数据头；失败记录保存在attempt_01，未删除。修复上述依赖补核后，第二轮302.047秒完成全链路：按需检查10份资料，14次只读工具调用，形成3项工程处理建议。没有换数据、改误差标准或加种子追效果。

- 真实格点来源：NASA POWER与NOAA CMORPH（后者3日成员），实际统计日与单位按已有adapt_products处理，再经公共端口空间绑定。站点日界采用历史UTC [00:00,24:00)联调假设，明确不作为独立科学确认。
- 公共数值搜索评价99张图，单源53、多源46；按已有预算/提议上限停止（400000次提议），没有扩大预算。账本记录307次新节点调用：search 300、deploy 3、validation_audit 4，失败0，服务墙钟约23.578秒；缓存遍历不冒充新调用。此项只验工程链路，不是新增GOLEM优势实验。
- 选中残差IDW、power=3路线，包含spatial.source→spatial.residual_idw及实际站点观测父边，phase=mutation；输入索引1对应CMORPH，另有原工程站点条件观测。两套格点都参加比较，不等于最终一定将两个格点直接加权混合。未把工程站点的作者订正产品当成全新独立实测。
- 选择队列63项：单源最低RMSE=9.4677635985，多源所选RMSE=9.1319103375、MAE=3.8019922207；未用于选型的18项留出复核，多源RMSE=4.0599524998、单源RMSE约3.358，多源未优于单源。reference_basis=PRIMARY_GRID_RECONSTRUCTION_PROXY，accuracy_improvement=NOT_ESTABLISHED。不得宣传真实精度提升。
- 实际更新27份原生降水文件、合计81个原生日期/站点值；来源NetCDF日降水经处理写入SWAT+ .pcp，非共同日期保留原值/缺测状态。新ZIP CRC与27个写回after-SHA全部回读一致。
- 最终格式PASS、工程引用PASS，31份天气文件检查通过。匹配固定SHA官方60.5.7程序完成2018-02-03单日试运行：返回0、新success.fin、updated_date_exercised=true、model_read_confirmed=true；日志确认读取降水、温度、辐射、湿度、风、WGN与站点。原2001—2023完整模拟及预测精度未验收。
- 最终DS复核真实完成，verdict=needs_verification，原因包括科学约定/完整模拟仍待核实；没有推翻本地格式和试读PASS，也没有把它改成无条件ready。

重要推荐边界：本轮3项建议是工程适配/端口绑定/启动搜索的编排步骤。它们确实引导了公共工程入口，但当前数值搜索编译器不接受engineering.*作为空间算法节点，recommendation_validation保留REJECTED、confirmed_initial为空。本轮真实数值图由已有公共搜索候选产生；不能宣称这些工程包装步骤已作为DS数值初始图通过验证，或据此证明DS提高搜索效果。该限制不阻止已验收的融合处理/写回/模型试读，但后续若要评价DS数值推荐效果，需单独修复该层协议，不能拿本次结果充数。

### 依据展示与成本、保护

保留现有模拟技术引擎样式，新增“决策依据与执行链”区域，按实际读取展示字段/单位、处理判定、工作约定及最终反馈；完整文本可滚动，避免裁切。连接重试/生成等待细节收进技术记录，首页保留重要步骤。结果页用实际原生格式替代ZIP容器标签，明确NetCDF→SWAT+降水文本、原生写回、模型试读和科学边界；技术目录名改为工程任务名。不用虚构评分、提升百分比或AI置信度装饰页面。

成功轮累计10次线上响应（去重响应ID）、291661服务Tokens，成本见costs.json；不宣称低Token成本，也不拿节点执行耗时替代全流程302.047秒。第一轮失败成本与记录保留。临时工具目录与整个本次Qt处理目录均已清理，保留约893.9MB必要交付包；原用户ZIP最终SHA仍为d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f。

证据统一在output/flow_bridge_20261119：ui_acceptance.py、real_api_ui.json/日志/原始运行截图、processing_evidence（真实图、预测/产品、成本与评分）、final_native_evidence（程序日志与新标记）、verification.json、完整处理结果.zip、source_before/source_current/source_hashes.json。data/approved_input_contexts保存指纹绑定的历史科学声明；源代码无案例名称或站点/日期条件分支。evidence_panel_current.png/result_current.png是用本次真实回执重新渲染的展示修正版，不冒充另一轮数值运行。

48项新增与受影响定向测试通过，0失败/0错误（tests_final.xml），覆盖原始控制字节、分页、回执侧证、浅入口目标传递、内容/模型作用域、公共工程委派仍待验收、旧按需工具/桌面/最终关卡/推荐搜索。展示调整后又跑16项相关测试（tests_display.xml，通过，含重复项不另加为独立测试数）。未重跑历史大型研究实验、未改算法内核与质量门、未推送GitHub。

本轮验收：公共衔接PASS；原包Qt页面真实全链路PASS；格式/写回回读PASS；匹配程序单日读取PASS；原始数据保护和临时清理PASS。历史日界真实性未证实、独立精度改善未证明、完整原模拟期未运行、DS工程包装建议作为数值种子未接通，分别保留，不能与功能运行成功混同。


## 2026-11-19 推荐页手动继续误拦截修复

用户同一Saguling原包在推荐页继续时遇到“整包工程仍含未放行任务”。此前真实302秒全链路验证走自动继续，漏验推荐页手动按钮。源码确认放行布尔值原先只在continue_automatically设置，手动continue_recommendations依赖默认/被重置的False，因此存在有有效核验却误拦截的状态路径；不能归咎用户数据或认定新坏包。

本轮只改Qt编排状态：统一update_task_checks保存实际核验记录，完成检查时即登记；自动和手动继续共用require_engineering_checks，依据当前记录重新计算，不信任单独布尔缓存。数据/目标重置清空记录；无勾选回退路径也不能绕过检查。真正阻塞时错误显示具体候选ID、操作和核验原因，而非笼统要求隔离输入。科学处理、搜索、原生写回和最终质量关卡不改。

19项受影响测试通过（output/continuation_gate_20261119/tests.xml，0失败0错误），实际Qt按钮覆盖：同一批准记录自动继续与手动点击进入同一执行入口、缓存布尔被误清零仍可按实际记录继续、真实阻塞原因显示、数据重置后失效和无勾选不得旁路。数值后端在此行为测试中使用spy，未再次重跑5分钟融合或模型；不得将本轮机械状态测试表述为新的完整数值实验。此前原包成功工程、失败记录与原始文件均保留。修改前及当前源码和指纹同目录保存。


## 2026-11-19 具体缺项与失败原因的持续显示

用户指出数据/工程缺条件时只看到笼统暂停。此轮只接通诊断展示与导出，未重跑融合/模型或改质量门。新增desktop/issues.py，归一化已有checked/failures、IncompleteProject.completeness.findings、SearchNotReady.report、source_admission.required_information、requirements_check、模型输入缺项、最终格式/工程/原生运行结果和DS待核实信息。沿用后台原始依据及action；已知block_code只给与该定义一致的补充要求，其他原因明确“后台未提供具体补充步骤”，不捏造缺文件/单位/时区。

界面全局增加“问题与待核实事项”表，切换页面后仍可见，逐项显示检查阶段、任务/文件及字段、状态、具体问题和补充要求。候选/任务ID通过当前真实candidates与scan绑定来源文件；完整路径在提示和下载清单中保留。输入/目标变化时清空陈旧问题；暂无可执行方案而后台没有具体原因时明确标未知，不能笼统说数据损坏。异常弹窗仍可提醒，但不是唯一证据渠道；即使没有产物也可单独下载JSON问题清单，禁止覆盖原始输入/写入原始目录。

selected_processing不再只把结构化异常压成一串str，逐任务保留issues。结果ZIP追加desktop_issues.json，同名旧记录使用修订名保留。格式通过、模型未运行、科学待核实分别列项，不因其中一项失败改写其他PASS；部分成功的产物和失败清单同时保留。

本轮定向回归：tests_final.xml为49项通过0失败0错误，随后问题绑定展示与继续按钮复核tests_last.xml为22项通过0失败0错误（含重复测试，不相加为独立测试数）。覆盖实际公共异常类型的具体文件/行动透传、时间定义缺项、未知不得编造、跨页持续显示、独立下载/原件保护、部分失败、原生未运行不覆盖格式PASS、原始准入缺项、候选到来源真实身份绑定及去重、旧手动/自动继续回归。数值后端未新执行；没有以界面测试充当融合或模型运行证据。

output/problem_display_20261119保留源码快照、指纹、测试XML和issues_preview.png。截图明确标注合成缺项功能验收，不是当前Saguling原包又出错。当前已知限制：后台完全没有提供字段/文件原因时，本轮只能准确展示诊断不足，不能可靠推断具体缺失项。未推送GitHub，既有真实原包成功结果保留。


## 2026-11-19 质量筛查误拦截与工程上下文衔接：双入口真实复验

此前“quality_screen：逐格比较只需核验对齐……”是平台把质量筛查建议混入处理目标放行判断的衔接错误，不能据此断言原包损坏。上一轮只测自动继续，未完整覆盖手动继续和DS不同读文件路径，历史PASS的范围应据此限定。

最小修复：desktop/discovery.py把quality_screen登记为待执行质量核验，不预判质量PASS；blocking_checks/deferred_quality_check由自动、手动按钮和问题展示共同消费，只有同一候选存在公共工程执行路线时才委派给真实质量门。单独质量建议、其他候选的路线、真正缺时间/科学语义仍阻塞。three_phase.py不再另维护质量建议分支；__main__.py、issues.py统一调用。theme.py仅补错误弹窗背景/文字对比度。

实际可见Qt复验v4又发现：DS只读说明、数据回执和网格表头而未读file.cio时，已登记原包被误送到通用格点处理，缺原生写回。因此v4明确FAIL，保留failed.json、outcome.json和截图。随后只在native_dispatch.py复用input_context.find，重新校验原文件SHA及目标模型，把已登记engineering-semantics声明送入已有工程准入路径；不伪造本轮已读控制文件，不预先给原生PASS。discovery保留显式evaluate_gain目标，不因DS仅建议quality_screen就丢弃用户目标；DS明确拒绝/直接使用不会被覆盖。无匹配历史上下文的新数据仍需自己的科学证据。

本轮真实验收output/quality_gate_gui_20261119/visible_run_v5：原始Saguling ZIP、真实DeepSeek、可见Qt窗口和真实按钮点击，自动处理与推荐页手动继续分别跑完整处理及导出。自动完成273.969秒；手动处理从283.953秒开始，到462.094秒留档；两个导出全部结束470.844秒。非模拟后端。两次均为输入及输出质量PASS、原生格式PASS、工程引用PASS、SWAT+实际单日试读PASS；27份.pcp、81/81个目标日值更新，31份气象表格式核验。NetCDF降水经现有平台流程转为SWAT+降水文本(.pcp)，更新工程副本，原文件保留。

两份ZIP分别893895374、893900704字节，独立CRC全读无损坏，27份变更的实际字节SHA全部匹配写回清单；两入口所选图9591823b972565a8f0b8及指标完全相同。选型队列63样本，RMSE9.1319103375、MAE3.8019922207；99图（53单源/46多源），每轮307次真实节点调用，其他缓存与预算见实际summary/total_cost。不因工程接通宣称精度提高：参考为格点代理，精度改善NOT_ESTABLISHED；选中的是背景加训练站残差校正，不能称两个外部格点必然均被加权融合。工程推荐包装节点尚不能作为数值搜索种子，confirmed_initial为空，此限制未由本次修复解决。

模型程序60.5.7的SHA256 de01a927eebe00b4467ff53424175e247e7808e0bee99790f3d75db1b0a64179；在副本以2018-02-03单日实际试读，退出0、新success.fin、updated_date_exercised=true。仅有界试读，不是2001—2023全期模拟通过；原manifest旧字段native_model_executed=false是准备阶段状态，实际试读证据以随后final_check.native为准。最终DS仍needs_verification，版本参考差异与历史模型日界UTC工作约定未证实，均保留在问题清单，未伪装历史事实。

原包最终SHA仍d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f；工具临时目录及本轮ecodata-desktop任务目录已清理。保留两份必要导出及失败记录。成功双入口共去重10条线上响应，服务234601 Tokens（api_costs.json）；包含共享的首次扫描，不把手动重用扫描重复记账，失败v4成本不混入该数字。未推送GitHub，未改科学内核、随机变异、评分或质量门。

定向测试tests03.xml：34项通过、0失败/错误，覆盖同候选质量委派、不同候选不得借用、真正缺条件仍拒绝、自动/手动同消费者、输入改变后上下文失效、原始控制字节/回执传递、具体失败展示与旧简单流程。tests02.xml为前一轮24项通过，含重复，不累加。source_before/source_current及source_hashes.json保留修改前后；verify_delivery.py、verification.json、真实auto/manual截图、各图及质量/模型证据可复核。

交互验收边界：这是可见原生Qt窗口的QTest实际按钮点击，路径输入及保存目的地由验收程序填入；DS、数值处理、模型和质量门未mock。原生文件选择器及人工鼠标操作未验收，电脑自动化工具初始化报路径错误，不能称完成了该项。v1—v3为自动化模态对话框设置失败，未作为数值成功证据。v5自动与手动处理/导出PASS，OS级文件选择交互未验收，独立环境精度提升未证明。


## 2026-11-19 推荐方案确认步骤恢复

用户反馈开始后直接跳第三步，未得到选择算法方案的机会。核实当前continue_automatically在检查后默认勾选全部可执行项并立即continue_recommendations，选项并未删除，而是被自动跳过。现仅改Qt编排：保留按需检查、共同质量验收与默认推荐勾选，检查完成后停在第二步，显示推荐依据和选择控件，由用户点击“按当前方案继续处理”才进入执行。查看方案提示和流程说明同步更正。第三步既有“其他已验证方案”选择/导出不变：该处仅能在真实数值执行后展示；第二步推荐不等于已验证精度。未改科学算法、搜索、质量门或历史结果。

19项受影响定向测试通过（output/recommendation_pause_20261119/tests.xml，0失败/错误），验证检查结束不执行且停第二步、默认推荐可编辑、用户取消一项后实际仅提交保留项、待核验质量记录不丢失、真正缺条件仍阻塞及既有输入/导出行为。科学后端以spy隔离，本轮不重复真实融合/模型试读；之前真实双入口470.844秒证据仍是此前自动跳转版本，不能冒称新交互也已完整真实重跑。修改前后源码和指纹保留在同目录。新流程需重新打开Qt进程加载；未强制关闭用户现有窗口，未推送GitHub。


## 2026-11-19 第三页真实处理进度与空黑底修复

用户反馈第三页只有“正在生成原版工程输出包”和大块黑色区域。ResultCards原先仅在产物返回后填卡片，QScrollArea未指定浅色背景，系统深色主题下空区变黑；搜索真实候选评价也未回调页面。现保持原布局与算法：显式设置浅色滚动面板，新增有界300行只读过程记录；第三页收到show_progress即显示，完成渲染保留，重新任务reset清空。工程开始明确推荐仍需契约核验、只承诺预算内最佳。

空间搜索在实际完整候选执行后发出图指纹、真实节点列表、阶段、成功/失败、RMSE/MAE/样本及当前最佳；重复已评图不重复声称新执行。实际推荐契约回执、变异开始和停止统计同样经既有公共progress传到Qt，不虚构推荐已成为种子、不改数值算法、搜索策略或预算。其他后端仅显示其实际已有回调，不冒称所有后端都已有逐图进度。格式/工程/模型最终关卡沿用原回调。

21项受影响定向测试通过（output/process_display_20261119/tests02.xml）；首次测试因测试夹具缺model字段失败，保留tests.xml，修正夹具后复测，未屏蔽生产异常。新增检验实际小型合成数据候选的数值指标进入页面、缓存重复不新增事件、完成后保留/新任务清空。progress_preview.png为当前Qt离屏渲染、两条真实小型合成候选计算的功能截图，明确不是Saguling新运行、不是环境精度证据。已人工查看截图，黑底消失、文字可读。本轮没有重新调用DS或完整SWAT模拟，未强制关闭用户旧窗口；需重新打开加载改动。

另读取用户桌面问题清单，7项全部为待核实：单日试读不能代替全期模拟、版本参考差异、通用摘要覆盖不足、原始日界假设及站点映射复核。清单明确31天气文件无失败且format/project PASS、model_read_confirmed true，不能把这些提醒误说成7项执行失败，也不能仅凭摘要空列表断言工程缺文件。


## 本轮最小任务条件 自动补齐及工程推荐搜索闭环

本轮用户授权优先完成目标分流、非关键缺项补齐及推荐到搜索接通。不是取消所有科学条件，不让DS凭空填写地理位置、时间基准、流域拓扑、观测值或模型状态。只改公共适配/编排，不重写随机变异、评分或融合内核，未下载新数据或训练模型。新证据统一output/minimum_task_contract_20261119，protocol.json显式版本minimum_task_and_engineering_recommendations_v1。

### 目标和最低输入

Qt现有目标菜单增加“仅数据处理（融合/对齐，不生成模型工程）”，不是增加一套页面。该目标跳过模型画像与原生工程要求，仍走公共识别、真实算法及质量门。已有模型组件服务本来就可不带完整工程，本轮补齐独立组件在页面无配对推荐时的确认路径：不得因没有第二份数据、没有融合候选而拦住单份组件转换；确认后仍由公共写出器核验必需科学字段，未在界面给预先PASS。data_only、component_conversion、model_project的最终复核范围分开，组件未请求全工程检查记NOT_REQUESTED，模型未运行记NOT_RUN，不以缺少无关土壤/温度等资料把组件格式成功改成失败。

新增SWAT+单站降水组件绑定复用现有SWATPlusWriter._weather_file/_cli和真实_records回读，不新增科学算法。最小输入为真实precipitation/time、mm单位、明确逐日累计/86400秒及日时间基准、实际EPSG:4326站点经纬度和海拔。可仅交付station.pcp与pcp.cli；不强求温度、土壤、landuse或file.cio。目录中没有工程时不会借用Ames或本机流域。单日数据也支持。不能按默认值补造日降水或位置；缺关键数据明确失败。其他模型继续已有组件契约，本轮未证明五模型任意缺字段均可自动补全。

models/minimum_inputs.py为公共非关键配置补齐：非物理项目标签、稳定输出站点标识、实际时间覆盖、标量坐标属性，以及写出器按真实文件生成索引。已声明值优先；与实际元数据冲突拒绝，不静默覆盖。逐项auto_completion保存字段、值和依据，真实progress显示；这不是DS自由编造物理缺值。DS工作规则也区分独立融合/组件/全工程，允许建议登记补齐项，但确定性服务决定是否安全执行。generic_requirements增加可读的最低输入与不需要的资料清单。

### 工程推荐到真实搜索

原问题是把engineering.port_binding/port_search等流程动词送入只接受数值算法的编译器；旧记录REJECTED、confirmed_initial空不应抹去。本轮recommendations.make_request从现有COMPONENTS给工程融合提供具体spatial.blend/residual_idw/idw组合；不把通用port_search包装成某个数学算法，也不预设赢家。推荐匹配不能仅因共同工程前缀而把全部不同算法混为匹配。

自主检查入口原先只from_decisions补候选，未调用已有recommendation线上协议。本轮抽出公共recommend_plan复用service.preview/send与目录验证，页面实际调用；失败允许原已核验候选回退，但明确标PLATFORM_FALLBACK，不能声称DS已选。推荐理由仍为AI建议，不是事实或精度证据。每个工程准备步骤只有公共端口实际对齐、质量通过并附输入指纹后，search_bridge才提供completed_preparation；数值编译器检查这些证据，再将具体算法绑定真实父边和训练观测。仅准备步骤记PREPARATION_ONLY，不能冒充已数值评分的算法。

### 真实及定向验收

1. 真实数据独立融合：accept_data_only.py复用历史已核验POWER/CMORPH规范化产品，按实际公共保守重网格对齐共同日窗口，不带模型工程；页面公共process_search_upload和工具包submit_file_request图及数值产品一致。real_data_only/acceptance.json为PASS，代理队列78样本，RMSE11.1208191076、MAE9.1723382176。这里只证明数据处理能独立执行，没有独立观测证明精度提高，不混同完整Qt上传验收。

2. 最小真实转换：从用户原包内实际单站.pcp提取2018-02-03至05真实值，按现有时间/坐标工作约定标准化为小NetCDF验收输入，再由prepare_uploads自动判定独立降水组件。real_precipitation_component_v2保留原成员SHA、输入与回执；COMPONENT_EXPORTED，format PASS，project NOT_REQUESTED，native NOT_RUN。没有上传完整工程、温度或土壤。真实DS最终复核按component_conversion返回ready，error为空；ready仅限组件。原日界仍采用历史工作约定，非作者定义核实。首个提取脚本错把嵌套成员当顶层，StopIteration后改用实际嵌套ZIP读取；该失败不伪装数据缺失。

3. 原Saguling完整工程：visible_run_v1为FAIL，DS只读工程文档未读科学表头，无法形成候选。保留失败记录。修复仅补有界最低表头检查（至多两份已列出科学结构、总补核上限4、总调用上限48，不深读数组）。visible_run_v2在补具体推荐调用前真实通过375.766秒，属于中间版本；最终visible_run_v3在真实DS具体推荐服务接通后，通过第二步确认按钮、真实数值执行、原生写回、格式/工程检查、SWAT+60.5.7单日试读及ZIP导出，总276.281秒。

最终v3三条DS数值推荐均PASS：blend fc2d0aa32864cb5c2a7e，residual_idw 6ba3cd62c0365efc048d，IDW a0b17d051bb8095d5541，均先在固定63样本队列评分。transitions前三条以对应推荐为父图进行参数变异，其中两条新执行，一条复用已真实评价的子图，不虚记第三次新执行。最终共99图，400000原始提议上限不变；选中9591823b972565a8f0b8，RMSE9.1319103375、MAE3.8019922207；最终真实节点调用307。保持原随机变异后端，不冒称本轮重建了GOLEM或证明全局最优。新推荐确实进入执行与变异，但最优图与旧轮相同，不能据此宣称新机制精度胜出。

v3结果ZIP CRC全读通过，27份降水文件的实际字节SHA与写回清单全部相符；原包SHA保持d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f。模型试读仍仅2018-02-03一日，不是完整模拟期验收；历史科学假设与独立精度未证实继续披露。v3共去重7次线上响应、87961服务Tokens；本轮所有保留尝试及独立组件复核共25次、508526 Tokens，详costs.json，不隐藏失败调用。任务临时目录已清理，保留有效交付和失败证据。

### 测试 范围与来源

tests06.xml为73项定向通过；随后任务范围/组件页修复tests08.xml为33项通过，最终tests10.xml为30项通过（存在重叠，不相加为独立数）。最终新增按钮测试确实由Qt确认入口调用公共真实转换，只有最后DS复核用模拟响应；独立真实组件DS复核另有证据。初次单组件按钮测试发现旧身份判断只识别强语义属性而漏掉Registry正式字段名，修复为仅作为组件候选识别，科学契约仍由写出器验收，不据字段名预判正确物理含义。其他失败包括旧页面/工具包测试把随机final_check收据UUID当科学差异，以及测试错误读取preview.body而非summary；分别校验真实载荷/两端检查状态、改正夹具后复测，失败XML保留。

桌面全工程验收为可见Qt/QTest实际按钮；路径与导出目的地由脚本提供，原生文件选择器仍未验收。源码最终快照source_current及hash在所有修复后保存；全工程v3过程中后续只修改独立组件/无模型的提示与确认路径，未重新跑全工程，不能说每份最终文件都在同一进程重新加载。完整工程、最小组件与无工程数据融合的证据分别列明。

官方/成熟成果核对并复用：SWAT+ Editor官方GitHub文档 https://swat-model.github.io/swatplus-editor-documentation/edit-inputs/weather/、官方程序化入口说明 https://swat-model.github.io/swatplus-editor-documentation/developers/ 与源仓库 https://github.com/swat-model/swatplus-editor，降水格式 https://docs.swat.tamu.edu/62.0.0/input-reference/pcp/。其气象索引与完整工程引用分离的定义支撑组件输出；本轮优先复用本库已有官方格式写出器，未新下载依赖、抄入优化器或给特定数据包设条件分支。

本轮验收：独立数据处理公共入口PASS；SWAT最低降水组件与安全补齐PASS；具体推荐实际评分及作为变异父图PASS；完整原包写回/短时试读/导出PASS；任意五模型缺项自动补齐未验证、完整期水文运行未运行、独立精度提高未证明。用户不需要完整工程也能做相应数据处理，但不能让DS把未知流域拓扑当成真实工程。未推送GitHub；旧协议、用户原数据与历史结果未覆盖。


## GitHub同步与队友更新合并

本轮将当前本地实际平台更新整理为代码提交，未上传用户原始数据、运行环境、密钥、大型结果包或本机验证工具安装目录。发现队友PR4最新c347426尚未进本地运行版，因此保留该提交并合并其统计窗口、对象面积换算与确定性校验，未以本地旧文件覆盖。两处冲突分别保留主页面preprocess说明与operation_labels、three_phase的质量建议传递与preprocess处理，未修改科学算法或重跑数值研究实验。

合并后的48项定向测试通过，0失败/错误，覆盖最小组件、线上推荐协议、质量/确认按钮、真实小型执行及队友统计前处理；不是全库回归或新原包完整运行。证据output/github_sync_20261120/tests.xml与teammate_merge.json。保留原运行源码到live_before_teammate_merge，合并代码同步回本地平台；已有两个脏Git工作区未覆盖。远端主分支推送结果以本轮sync_receipt.json为准；旧实验与数据不变。


## 不同原生分辨率真实双源通过仅数据处理入口验收

用户要求两份分辨率不齐的数据直接走昨日新增的仅融合入口。本轮未改平台源码、未换算法或调参，未下载新数据；从原Saguling包只读复制两份原始字节及POWER回执：CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_20180203.nc（373379字节，0.25°×0.25°、单日、变量cmorph）与POWER_regional_original.nc（21273字节，0.5°纬×0.625°经、7日、变量PRECTOTCORR）。未人工预对齐、预改数组/单位或提供模型工程，原始SHA保存在input_manifest.json，验收后逐一复核未变。二者都是公开格点降水产品，不是独立站点实测真值；本轮不能验证“粗实测约束细场”的真实精度。

output/unequal_resolution_data_only_20261120保存预定protocol、输入、副本指纹、当前源码指纹、脚本、visible_run_v1完整轨迹和verification.json。可见Qt选择data_only，真实DS按需读表头及原始下载时间回执，展示第二步方案后QTest实际点击继续，走公共selected_processing/independent_alignment/local_search/platform_search。模型工程未上传且未要求；没有通过后台脚本预置最终图。路径及保存目的地由验收脚本提供，原生文件选择器未验收。

平台自行将原日降水率按86400秒区间积分为mm日累计，统一到共同UTC日期2018-02-03，采用公共保守重网格（本环境实际后端xarray-regrid，未安装ESMF，不假称ESMF执行）将CMORPH对齐到POWER粗网格，输出1日×7纬×5经，仍为0.5°×0.625°。原数组未人工修改。对齐任务分别导出两份产品，融合任务执行23候选/100次提议：19单源、4多源，均在相同26个可见格点留出代理样本上评价。最佳单源RMSE7.3312545168、MAE6.5341892531；最佳多源RMSE10.5151597757、MAE8.3842796521。按已有评分最终保留单源图6fb8052607fc23ed9473（observations→IDW(power1)→残差IDW(power3)），不是最终交付双源融合。部署产品与已对齐POWER输入逐像元最大绝对差0.0；不能把该产物描述成已引入细源信息或生成更细分辨率。

完整页面处理及导出114.359秒，0失败任务，6条去重真实线上响应、38479服务Tokens。ZIP重新CRC全读无损坏，三份NetCDF逐一回读，单位mm、有限及非负检查通过；输入指纹均未变，Qt任务临时目录关闭后已清理。DS输出复核ready仅限数据处理交付，不是环境精度PASS。仅有一天、参考为PRIMARY_GRID_RECONSTRUCTION_PROXY且不是独立实测，不能据此推广“融合不如单源”或“粗网格更准确”。

结论分开记录：不同原始分辨率的双源可由仅数据入口识别、对齐、真实比较与导出PASS；真实多源候选已执行PASS；本次最终多源方案NOT_SELECTED；更细目标网格输出NOT_EXECUTED；粗实测作为整体约束后的细场恢复及独立精度NOT_EVALUATED。当前默认入口选择共同粗网格，基础流程跑通不等于用户前述完整细尺度需求已经满足。本轮没有因为结果不理想更换种子或追加试跑，没有启动研究对照；未重跑无关回归、未推送本轮新报告到GitHub。

## 目标细网格作为公共硬约束：仅融合入口真实验收（2026-11-20）

本轮响应用户要求：目标分辨率不能只是最终插值导出参数，候选必须在同一目标网格上执行、评价和交付。保留前一轮默认粗网格结果，不覆盖历史数据、源码证据与协议；未新增算法、未调评分/种子/搜索预算、未下载数据、未启动大型实验。

实现位置：spatial/regridding.py提供target_grid_spec、resolve_target_grid、require_target_grid公共契约；algorithms/spatial_fusion_task.py在构建候选前统一目标坐标，每折真实运行完整细网格、检查形状/有效覆盖/物理范围，再通过既有球面面积重叠算子回聚合到原始粗参考像元评分。粗输入或错误尺寸不能直接取得fitness或入选；最终产品直接使用冻结图在该目标网格上的输出，没有选图后插值放大的步骤。参考始终保留原始粗像元，不将插值结果作为细尺度真值。页面任务声明经file_agent、recommendations、discovery、independent_alignment传入同一公共搜索任务；不消费目标契约的适配器明确阻塞，不能退回默认网格。页面增加仅数据模式的目标网格间距字段，修改即作废旧核验，明确填写才启用新规则。当前该严格重建路径支持同变量cell_mean规则经纬网格、EPSG:4326、共同覆盖；站点/协变量及其他CRS的目标观测算子未在本轮扩展，明确拒绝，不能宣称全任务均支持。无目标声明仍保留历史默认行为。

真实输入沿用output/unequal_resolution_data_only_20261120/inputs内CMORPH与POWER原始字节及哈希绑定UTC回执，分别原生0.25°×0.25°和0.5°纬×0.625°经；本轮明确目标0.1°×0.1°，比两者均细。不是按输入文件名选算法。原坐标基准尚无独立证明，EPSG:4326是公开记录的地理坐标工作约定，产品单位与地理坐标字段先由已有产品契约核验；明确CRS不应被改写成WGS84来绕过检查。

验收路径是可见Qt页面、真实DeepSeek、实际点击第二步确认/处理/导出、公共搜索与真实科学内核；路径和保存目的地由QTest脚本提供，未冒称原生文件选择器/人工浏览器验收。新证据统一在output/target_grid_20261120：protocol.json、source_before、source_after、tests01—05.xml、visible_run_v1/v2/v3、verify.py与verification.json；最终以v3和verification.json为准。初次v1融合成功但单独对齐缺CRS传递而失败，脚本最初仅检查是否有产品，success字段过宽，不能当整体验收依据；已改为同时检查failed_tasks。v2无任务异常但独立回读发现POWER边缘310个NoData，未标完整通过。原因是轻量重网格后端在外半像元边缘不保证有效输出。v3按几何条件将完整目标像元限制在共同原坐标中心支持范围内，不填造边缘，不根据误差裁区域；该修正缩小覆盖，并导致参考队列变化，不能将新RMSE与旧轮直接比较。全部失败及部分成功记录保留。

最终v3页面全流程89.954秒，0失败任务；23个候选实际评价（19单源、4多源），100次提议，72次搜索真实节点调用。23个通过候选均记录相同1×30×25目标网格检查，目标750像元；原粗参考回聚合算子15×750，共同留出评分队列11个真实存在的粗产品像元。最终选择单源重建图6fb8052607fc23ed9473（观测→IDW→残差IDW），代理RMSE5.6213491104、MAE4.7496665452；最佳双源图0e36ba06456a4198fcf7，代理RMSE12.1970960247、MAE9.3891033045。未强迫双源获胜。这里“单源”是目标细网格上的重建路线，不是原粗网格直接交付。目标网格硬约束不等于必须使用多个来源；若以后用户明确要求使用各来源，需要另行声明来源使用契约，不能通过低RMSE抵消该要求。

完整结果ZIP的两份对齐产品与一份选中产品均回读为1日×30纬×25经、0.1°×0.1°、mm；所有像元有限且非负，坐标完全一致，ZIP CRC全读通过，原始输入SHA256均未改变。产品导出为NetCDF→NetCDF，单位日率先按已核实日区间转mm累计；本轮仅融合/对齐，不生成模型工程、不运行SWAT+。source_after保留当前源码及指纹；v3执行后只清理了不参与数值的未用局部导入/变量，并增加回归测试，未改变所执行科学计算。公共重网格实际使用现有xarray-regrid后端，未声称ESMF执行。预处理、搜索节点、缓存与部署成本在summary/verification中分开记录，不能将节点执行时间当作全流程耗时。

定向测试：63项受影响回归通过（tests02.xml）；最终6项目标网格测试通过（tests05.xml，含公共真实搜索、错误候选尺寸拒绝、无目标不自动新增约束、页面参数变更失效/非法参数、边缘NoData回归）。其中测试用温度小场为合成功能数据，不作为环境精度证据。未运行无关全量回归或949图穷举。

验收结论：目标网格硬约束、候选实际细网格执行、原生粗参考回聚合评分、仅融合页面完整处理、同坐标产品回读与原始数据保护均PASS。最终双源方案NOT_SELECTED；细尺度独立精度、融合优越性、真实新增细节NOT_ESTABLISHED。两份输入均是格点产品，不是独立站点真值，且只共同覆盖一天，11个粗参考样本证据很有限。0.1°是已执行的输出网格，不应被描述为已证明的0.1°观测准确度。本轮未自动推送GitHub。

## 通用证据自适应融合评价与仅融合入口验收（2026-11-20）

### 复用、缺口与修改边界

已核实并复用：platform_search公共入口、SpatialTask/SpatialSearch、同一固定评价队列、已有球面面积重叠算子、sklearn坐标分组、单位/时间契约、目标网格硬门槛、NumericalDAGExecutor、随机变异driver、调用/缓存预算、冻结后复核、页面结果卡及统一ZIP打包。此前缺口不是“没有RMSE”，而是主来源粗格点重构代理单独决定细网格方案，无法评价另一个来源的细节信息。本轮修改评价编排，不新增科学算法、不改插值/残差/融合内核、不根据来源数量奖励、不下载数据、不开展正式算法优越性对照。

新公共模块ecodata/evaluation/fusion_protocol.py（fusion_evidence_v1）根据证据选择评价范围，spatial_fusion_task版本为spatial_relationship_fusion_v1_4_evidence_adaptive：

| 条件 | 真实消费者与行为 | 结论边界 |
|---|---|---|
| 已有合法观测 | 复用原空间留出及观测尺度契约，输出OBSERVATION_HOLDOUT；观测不参与自身拟合 | 不自动宣称来源与站点独立；上游独立性仍须证据 |
| 存在已核实的原生cell_mean网格且共同范围足够 | 所有来源先按既有面积算子变粗，以按几何分辨率确定的原细产品为隐藏参考；RMSE反馈真实搜索 | 仅产品重建代理，不是环境真值；不能直接证明原目标尺度的细节精度 |
| 支持域未知、区域/样本不足，无法构建合法退化实验 | 返回具体missing_evidence；只保留明确标注的粗产品重建代理，不宣称可据此排序细尺度精度 | 这是受限降级，不是已验证的细尺度收益；不存在凭空生成的评分参考 |

平台已有EngineeringMetric仍用于明确声明的工程指标任务；本轮没有把所有物理变量强行塞入一个加权总分，也没有替没有评价依据的任务制造精度排名。新模式自动用于已支持的cell_mean空间路径和观测路径；不满足契约的变量/累计/点支持仍依赖原适配器核验，未扩展未审查的物理算子。legacy_primary_grid_v1显式开关仅保留旧协议复现。适配器不维护数据包专属规则；原生分辨率、单位、统计含义和状态依赖由公共契约消费。

方法参考已核验的GitHub案例：https://github.com/liangjiandeng/DLPan-Toolbox 的退化尺度评价思路，以及 https://github.com/rvalavi/blockCV 的空间隔离思路。实际复用项目已有面积权重和sklearn分组实现；未复制影像专用指标、未引入新训练模型或优化器。降雨不是全色影像，不能声称直接复现其验证协议。

### 冻结协议、隔离与真实执行

新证据目录output/fusion_evaluation_20261120，保留protocol.json、source_before/source_after、输入/代码指纹、测试XML、visible_run_v1/v2、评价预测NPZ、metrics.csv、verification.json及当前卡片渲染记录。输入仍是原CMORPH、POWER及哈希绑定UTC回执；原始SHA256逐一复核未变。目标继续0.1°×0.1° EPSG:4326工作约定，共同有效内部范围，真实应用产品1日×30纬×25经；原坐标基准并未被本轮独立证实。所有候选仍须真实生成完整应用细网格、经过形状/覆盖/非负/范围检查；不存在最后才放大的替代执行。

评价退化采用预先固定的名义2倍，因共同边界和整数网格数，实际轴间距由面积覆盖计算并记录，不伪称严格2倍。两来源退化为POWER 3×2、CMORPH 7×6；原细参考168个像元。退化算子使用所有原观测生成粗输入是重建实验的明确构造，绝不是细参考真值直接输入算法：评价执行器仅接收变粗后的两个来源及变粗主来源的conditioning；任何细参考值、选型/复核误差都不进入该执行器输入。参考仅用于独立误差计算。该退化不是已知真实仪器误差模型，跨尺度可迁移性未证实。

沿用坐标分组seed11，129个样本用于本轮选型，39个外部空间块样本仅在最终图冻结后复核；这里“外部块”不是新独立实测，也不是新独立年份。粗主产品留出11样本的RMSE另存coarse_reference_diagnostic且used_for_selection=false。新RMSE是唯一数值主目标；MAE、Bias、覆盖和失败率独立列出。可观测参考掩码固定，失败不能只删困难样本参与排名。误差极接近时继续按原简化规则选图。来源消融是同一已执行候选集中排除某来源的最佳合法路线，不是声称已穷举所有重训组合。

预算未按结果修改：130搜索调用、180总调用、100提议、45秒搜索墙钟。沿用公共composition_optimizers.random_graph_search；这是当前已用随机变异机制，不冒称本轮重新启动了官方GOLEM对照。新评价的真实调用计入同一账本，未为多源增加预算。第一次v1数值计算成功，但NPZ使用子目录名违反公共下载接口的平铺文件名契约，整体验收FAIL。修正为通用平铺名称后v2完整通过；18张共同图的指标逐项完全相同，证明没有用修复打包之机调整数值结果。

### 实际结果：两类依据必须分开

| 评价范围 | 最佳单源RMSE | 最佳多源RMSE | 单源MAE | 多源MAE | 样本数 |
|---|---:|---:|---:|---:|---:|
| 退化尺度产品重建选型 | 12.469866 | 12.390965 | 6.593596 | 7.031880 | 129 |
| 图冻结后的退化尺度外部块复核 | 25.881368 | 30.316031 | 14.909727 | 16.796942 | 39 |
| 这两个选型胜出图的粗主产品诊断，不参与选择 | 42.356029 | 32.087446 | 29.304114 | 22.521315 | 11 |

选型Bias分别0.996389和-0.492232；外部块Bias分别-3.431517和-9.404563。通过候选在统一队列覆盖率100%、失败率0。主指标上多源只改善0.6327%，MAE反而更高；12个选型空间块中只有2个改善。冻结复核多源RMSE上升17.1346%，不支持稳定收益。没有像元级bootstrap、没有把12个邻近块称作独立重复，没有显著性或置信区间宣称；只有一天共同数据，时序稳健性未评价。

同一选型队列来源消融：去掉POWER，最佳已执行路线RMSE12.469866（CMORPH单源）；去掉CMORPH，最佳已执行路线RMSE21.414639（POWER单源）。这只能说明CMORPH对恢复CMORPH参考结构更有信息，不能证明其真实降水一定更准确，也不能把模拟/产品与实测的可信度按名称硬排序。精度仍依赖独立观测与支持尺度。

最终选中多源图dea7c689b883e00a5734，实际父边为CMORPH来源场与POWER观测→IDW(power=2)估计进入blend(weight=0.75)。它以冻结选型RMSE选中；即使复核更差，也没有偷偷改回单源或调整参数。最佳单源7d1e002146f28d3bb60b。所有图真实执行，保存18份逐图预测/参考/选型与外部块索引NPZ，独立verify.py重新计算每图RMSE与MAE，确认objective就是RMSE，没有多源加分；没有缓存命中冒充新图。18图、100次提议，76次搜索真实节点调用、4次部署调用，0失败调用；新增退化准备0.015929秒，已另列，不将内核耗时冒充全平台成本。v2含真实DS按需读取/建议/输出核对和导出的完整Qt流程130.265秒；输出核对发生有界API重试，日志保留。v1失败流程87.343秒亦保留，不隐藏失败成本。

### 页面、下载、测试与交付边界

可见Qt实际按钮操作完成“仅数据处理”输入、方案确认、搜索、处理、输出检查与下载；路径/保存目的地由QTest提供，未冒称原生文件选择器或人工浏览器测试。实际结果卡由公共output_review.cards消费同一evaluation_protocol，显示参考来源/尺度、129/39队列、两个来源消融、粗主产品诊断不参与选择和0.63%差异；多源/单源候选仍可由用户选择导出。ZIP含融合评价JSON和18份NPZ，原3份NetCDF回读同一0.1°网格、mm、有限且非负，ZIP CRC全读通过；不是模型原生工程转换，不运行SWAT+。

v2后只修正一处仍写“粗参考决定选图”的旧说明，并将仅数据模式的成功处理说明移出“模型缺项”，没有改数学评分、候选、随机数或产品；当前公共结果卡基于同一真实结果另行渲染回放，证据为evaluation_cards.png/json/txt，不冒称重新跑数值。旧ZIP与旧截图保留，evaluation_protocol.json和逐图数值证据为评分依据。无需因显示文字修正再跑搜索。

定向检查按最新复测记录合并59个唯一用例均通过：包括原目标网格、空间任务、候选导出、评分依据页面消费、参考隔离、外部标签不能改选型、无来源数量奖励、单源/多源共同队列、缺证据/未知、旧协议显式复现、真实父边、打包与回读及页面流程。早期失败XML保留：原测试把主分数样本数等同粗诊断数的假设已按新协议分开验证；下载回归初始只提供1个文件触发原接口要求至少2文件，修正测试夹具后通过，没有放宽生产打包门槛。未运行无关全量回归、949图穷举、正式搜索优越性实验，也未自动推送GitHub。

最终状态：公共证据自适应评价/隔离PASS；目标细网格与真实执行PASS；来源消融与两种评价差异展示PASS；页面完整流程及产品回读PASS；融合稳定收益NOT_ESTABLISHED；独立实测精度NOT_EVALUATED；通用于任意物理支持/任意模型的精度判断NOT_SUPPORTED。无独立真值时仍有产品特定参考偏差，新协议减少“只重构粗主来源”的片面性，不意味着消除了所有评价偏差。

### 冻结结果原因定位：为何选型改善0.63%、复核恶化17.13%

本轮仅查源码、原生坐标和已保存逐图预测，没有重新执行搜索/预测，没有调参数、换划分或改排名；源代码逐项SHA与verification.json相同。诊断保存output/fusion_evaluation_20261120/frozen_result_diagnosis.json。

冻结多源图实际为0.75×CMORPH重建＋0.25×POWER的IDW估计，并非可随位置变化的可信度或偏差模型。选型12个空间块仅2块改善，合计平方误差减少253.04，刚好使总体RMSE低0.63%；其余10块均变差，MAE也变差。这是固定总体RMSE的局部补偿效应，不是指标算错，更不等于普遍改善。

坐标分组未读取降水值，不存在据误差挑复核点的证据；但只有一天、一次空间划分，未验收选型/复核降水结构的代表性。选型参考均值14.536mm、最大124.4mm；复核均值38.933mm、最大166.1mm，强降水分布明显不同。复核多源新增平方误差9719.44，其中块12贡献9258.86，占95.26%；该块参考均值112.378mm。参考166.1mm的一个像元，单源估计102.715mm，多源仅79.597mm，说明POWER分支在该事件的强雨区压低了已偏低的峰值。此处参考是CMORPH产品，不是雨量站真值。

另发现评价范围的具体缺口：退化评价保留了输入halo的共同范围，而没有再次裁到应用目标范围；168个参考中心有48个在实际目标范围外（选型39、复核9）。这意味着评价空间范围与交付范围不完全一致，应后续统一修复；但复核这些范围外像元仅贡献238.74的新增平方误差，约2.46%，不能解释主要的17%退化。本轮仅记录，不改变封存结果。

结论：已经能定位为“本次全局固定权重混合压低强雨＋微弱总体RMSE收益不稳＋单日空间划分代表性不足”；另有评价/交付范围未严格一致的问题。没有证据据此断言两来源永远无融合价值，也不能把17%都归咎于划分或格式。复核与选型还共享构造粗观测约束，并非全新独立实测盲测。后续应先修正通用评价范围一致性，再用预先冻结、多日期的空间验证检验稳定性；不能利用这39个已暴露样本反复调整权重凑改进。

## 评价/输出空间范围统一与现有三日冻结方案验证（2026-11-20）

本轮仅修范围契约与固定方案核验，保留所有旧结果。根目录AGENTS中的历史转换限制已由用户本轮明确授权的公共评价修复覆盖；没有新增科学算法、下载数据、调权重、改变原始资料或启动搜索。

### 公共修复与行为验收

spatial/regridding.py新增contained_cell_grid，按已有cell-edge几何选择完整落在目标范围内的参考像元；仅中心落入但边缘越界的像元不参与。SpatialTask原粗参考裁切和evaluation/fusion_protocol.py细参考裁切共用该函数，不按文件名、变量数值或误差分支。输入halo仍可作为计算输入，但不再自动成为评价区域。新协议fusion_evidence_v2_target_domain记录应用边界、评价边界、裁前/裁后数量、完整参考坐标和outside_cells；队列指纹包含协议版本、坐标、日期和有限值掩码，避免同样像元数而不同区域错误复用。NPZ增加reference_coordinates，支持直接回查。

真实边界核对：应用与评价均为纬度[-8,-5]、经度[106.25,108.75]；细产品参考168→120像元，越界0；实际输出仍为0.1°、30×25。新裁切后按原坐标分组规则和seed11划分，93个开发区/27个外部区，两者空间身份不交叉；3天始终使用同一划分。由于几何总体改变，分组会随之改变，不能把新的数值与旧129/39队列直接比较并宣称提升。新划分在查看新误差前固定，不根据降水数值或结果重抽。外部区仍是产品重建复核，不是独立实测盲测。

定向测试49项通过（output/fusion_domain_v2_20261120/tests02.xml），包括：中心在内而像元边缘越界排除；完整支持不足明确拒绝；粗/细参考共用目标契约；值改变不改变范围和划分；相同数量不同坐标使队列指纹失效；固定图逐日指标、0提议且不能调用随机搜索；既有评价隔离、目标网格、空间处理和真实执行回归。未跑无关全量回归或大型实验。

### 现有日期盘点及原结论更正

原来选取的两份文件确实只有2018-02-03一天共同日期，但不能推断整个原包只有一天。只读检查原03_Saguling_完整工程与多源降水.zip的完整NetCDF成员，找到CMORPH 2018-02-03、04、05三份原始文件；POWER原文件覆盖02-01至02-07。因此本批原包/所选目录可匹配3天，而非1天；02-01、02、06、07只有POWER，不能伪造对应来源或称为双源验证。没有重新下载、补造样本或改时间标签。

通过既有inspect_product、adapt_products、哈希绑定POWER原始请求回执、单位转换和区间核验读取合并，保留原生格点和缺失状态，再经公共parse_inputs、evaluate_bank与execute_graph验证。沿用此前EPSG:4326工作约定，不把坐标基准当本轮新证实事实。首次准备因验收客户端遗漏公共回执契约的resource/receipt_sha256字段而停止，发生在数值评价前；补齐真实回执元数据后在attempt2新目录继续，没有绕过公共核验。失败记录及首次产物保留。

### 固定图、真实数值与负结果

本轮提前冻结原单源7d1e002146f28d3bb60b及原多源dea7c689b883e00a5734，后者仍为75% CMORPH重建＋25% POWER的IDW(power=2)；原图记录逐项核对相同。公共evaluate_bank新增逐日、全目标域及冻结外部区报告；selection_performed=false，提议0，不选择新赢家。两个固定图均通过相同原始数据、固定质量门、完整目标网格与同一队列；不是在各日期重新调参后挑最好的结果。

| 日期 | 全目标域单源RMSE | 全目标域固定融合RMSE | 单源MAE | 固定融合MAE | 相同参考数 |
|---|---:|---:|---:|---:|---:|
| 2018-02-03 | 19.321811 | 21.169750 | 10.561360 | 11.165449 | 120 |
| 2018-02-04 | 12.379652 | 15.823497 | 8.712289 | 11.003162 | 120 |
| 2018-02-05 | 8.157698 | 10.414999 | 4.601081 | 7.923263 | 120 |
| 三日合并 | 14.061012 | 16.401358 | 7.958243 | 10.030625 | 360 |

三日合并Bias分别-0.525979、-1.754411。两条图覆盖率100%、失败率0；当前固定融合在三天全域均更差。单独列出的外部27像元/日复核RMSE：02-03单源23.892761/融合27.025681；02-04为12.885219/18.832637；02-05为12.273354/12.941660，也均未改善。全部日期与区域均报告，没有只保留有利部分。固定方案原本开发日期02-03已暴露；02-04/05来自此前已提供原包，不承诺从未被历史工程读取，因此统一称回顾性产品重建验证，不称新盲测。

### 可复现产物与范围

新目录output/fusion_domain_v2_20261120保留修改前快照、attempt2/protocol.json、原始成员SHA与真实时间清单、单位/时区适配记录、canonical输入、两次有界固定图调用、逐图NPZ、逐日CSV、源码快照及verification_final.json。首次数值核验2图16节点调用；为新增全域逐日报告完成受影响复测，再执行同2图16调用，原指标逐项相同；无新候选或权重。两个实际产品部署分别1和4次节点调用。首轮准备/实际核验/两个产品部署12.759秒，受影响复测与独立回读另有日志；不把节点时间当作总成本。

verify_final.py独立重算逐日RMSE/MAE，核对图身份、0提议、93/27隔离、全部参考像元的完整边界及原始子集SHA；读取两份实际NetCDF均为3日×30纬×25经、mm、有限且非负。原包SHA d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f与历史一致。复用公共package_results打包，ZIP CRC全读通过；下载文件frozen_controls_3days.zip含两份真实产品、预测、协议、日期清单、指标和回读记录。不运行模型、不重新调用DS或做页面整轮重复搜索；这是公共固定方案核验入口的真实数值验收，不冒称本轮重新做了浏览器/GUI验收。

最终结论：公共空间范围一致性修复PASS，当前案例参考支持域与输出支持域完全一致；本批原包全部3个共同日期真实验证PASS，额外4个POWER独有日期不适用于双源验证；旧图/权重/原始数据保护PASS。结果不支持当前固定融合优于单源，但不能推广为所有融合无价值；3个连续日期和产品代理不足以建立独立实测精度或统计稳定性。本轮结束，不调权重补救、不新增数据、不推送GitHub。

### ecf2a15同步后功能与决策链核查（2026-11-22）

用户要求核实仅融合、最少必要输入转换及推荐/变异决策是否被新版界面覆盖。对照dae836f与ecf2a15：minimum_inputs.py、component_service.py、spatial_fusion_task.py、fusion_protocol.py、composition_optimizers.py均无变更。当前真实Qt菜单探针确认“仅数据处理（融合/对齐，不生成模型工程）”仍在选择目标模型菜单；选择后可登记目标网格。当前功能定向测试27项通过，记录output/current_capability_audit_20261122/tests.xml；包含无工程降水真实写出SWAT+组件、非关键标签补齐、关键时间/坐标冲突拒绝、页面组件确认、仅数据路径和共享推荐契约。测试中的小数据是合成功能输入，不是新增环境精度实验。没有发现这两个功能被删除，因此未回退界面或修改算法。

推荐调用实际顺序为file_agent/three_phase按需识别→materialize绑定数据字段→discovery.check本地核验与可执行入口→recommendations.make_request生成allowed_recipes→DS在目录内提议。recommendation_only/advisory_only条目可能在条件尚待确认时显示，但步骤为空且界面禁止作为可执行项勾选；不能把显示推荐等同已经具备融合条件或证明融合有效。进入搜索的具体推荐还须validate_recommendations契约检查和真实评分，只有通过者进入推荐父代池。

当前这条空间/仅融合路径使用共享random_graph_search随机图变异，包含来源、参数、结构及删除/重组，真实执行后在统一目标/队列上取RMSE最低合格图，受预算及去重约束；不是全库任意算法排列，也不是本轮已证明的全局最优GOLEM。仅数据模式默认不套目标模型画像，模型特定软引导依赖明确目标模型契约。框架可用于工程筛选，但复杂度近似优先当前仅采用浮点容差，缺少实际最小收益/稳健性通过门槛，因此0.63%微弱优势仍可能选中复杂方案；3日负结果和代理参考限制保持。复核不反馈反复调权，不能将预算内选优说成精度优势已证实。最少输入转换仅承诺所请求的已支持组件，不等于DS能补造完整工程；关键物理定义、位置、时间或必要目标资源缺失仍应明确阻塞。

### 独立仅转换入口与算法选择说明（2026-11-22）

用户授权增加直观的仅转换入口。本轮在现有首页目标模型旁增加“仅转换（不融合、不搜索）”勾选项；保留原仅数据处理及原模型流程。勾选后需明确目标模型，直接登记目录并到第二步确认转换范围，不要求先配置DS，不调用融合推荐、自动融合、评分或变异。切换模式作废旧输入/方案状态，防止借用旧结果；仅数据处理与仅转换不能静默混用。

复用desktop/service.run_original_workbench(declared_only=True)、normalize_archives、component_service.prepare_components、minimum_inputs及final_check.attach_final_check。仅增加公共组件分支的明确缺项提示与最终本地核验，不重写任何转换内核，不构造完整工程；网页/工具包可调用同一已有公共服务参数。最低关键单位、时间、坐标和任务资源仍必须真实具备，非关键补齐继续使用原机制。已知组件契约之外返回缺项/不支持，不回退到融合或搜索；不宣称任意数据均能无配置转换。

新增test_conversion_only_entry.py验证五项行为：实际原生组件写出与ZIP回读；首页无API→第二步确认→真实组件转换；模式互斥；关键时间信息缺失拒绝；无目标模型拒绝。测试显式禁止调用融合入口/搜索/线上send，以确认不是仅隐藏按钮。首次测试钩子错误引用prepare_uploads所在模块，已修正为automatic_preparation，失败日志保留，没有放宽生产条件。与最少输入、确认流程和候选导出回归合计25项通过；补充界面与选型展示回归21项通过，两组按测试身份去重共41项。未重复无关全量回归或降水实验。

可见Qt/QTest另完成两次有界功能验收：输入明确标注的合成日降水1/2/3mm、真实SWAT+ .pcp写出、格式检查PASS、工程NOT_REQUESTED、导出按钮实际点击并ZIP全读；未发DS请求、未融合/搜索、原输入SHA不变。路径及保存目的地程序提供，不冒称原生文件选择器或环境精度验收。首轮发现空方案区使用暗色默认背景，已复用现有卡片容器显示转换步骤/待检查条件，并修正容器背景；最终visible_v2/confirmation.png已回看。输出验证记录在output/conversion_only_20261122（source_before、tests01/02/03.xml、visible_v1/v2、conversion-result.zip）。当前真实格式功能验收为SWAT+降水组件；其他模型仍按原有已注册组件契约开放，不冒称此次逐一重跑。

普通融合流程第二步方案勾选仍保留，第三步可导出已评价候选的选择机制仍在，相关回归通过；某些任务合法配方只有一种，不能承诺全库任意选算法。仅转换模式不显示融合算法候选是范围约束，不是功能丢失。“选优不够稳”仍指小幅验证改善不能稳定迁移到其他日期/区域，不是界面随机崩溃。本轮没有修改原评分、权重或变异策略，不能把新入口验收当成选优稳健性已经解决。变更仅限__main__.py、service.py、新增定向测试和本总报告；未自动推送GitHub。


### 跨变量尺度对齐与日期—区域评价闭环（2026-11-22）

本轮按用户授权修改公共对齐与评价机制，保留仅融合、仅转换、最少必要输入转换、全部旧协议和输出；没有下载数据、扩展算法库、训练新模型或重跑历史大型对照。适用AGENTS的旧合并任务限制与本轮明确授权冲突的部分，以本轮授权为准。当前源目录不是Git工作区，本轮尚未同步GitHub。

**复用及来源。** 复用SemanticCrossScaleStackPlugin、GridRegridder、SemanticTimeResampler、Registry中已有scale.semantic_stack、公共platform_search、空间关系执行器、固定空间块及独立评价队列。查阅并沿用已安装的[xarray-regrid](https://github.com/xarray-contrib/xarray-regrid)关于连续量、保守平均、分类变量的不同处理；参考[blockCV](https://github.com/rvalavi/blockCV)和[scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)的数据依赖分组原则。没有复制新的优化器，也没有宣称地理分块本身就消除了相关性。

|改进|公共修改及真实消费者|验收|
|---|---|---|
|跨变量尺度匹配|algorithms/cross_scale.align_variables与platform_search.process_cross_variable；desktop/discovery按DS已建议的align_space/align_time做实际契约核验，recommendations把真实可执行组合送DS，selected_processing/pair_processing和工具包调用同一公共函数|PASS：同网格保持、不一致网格重映射、不同单位保留、不允许跨物理量直接平均；页面处理函数/工具包产品一致|
|目标网格|原pair_processing.grid_plan的几何实现移到spatial/regridding.common_pair_grid供公共层复用；明确目标优先，否则仅采用共同覆盖的较粗工程网格|PASS；不能自动推断精度最优分辨率；双轴粗细交叉、曲线网格或未声明垂向层仍须明确适配|
|时间支持|复用原SemanticTimeResampler，各变量保留瞬时/平均/累计语义；只取实际共同时间，区间量校验time_bounds|PASS：不同变量不要求换成相同单位或物理量；UNKNOWN和不匹配区间拒绝，不猜时区或填造扫描|
|多日期多区域选优|spatial_selection_protocol.blocked_validation由spatial_fusion_task.execute实际消费。新默认selection_policy=date_region_v1，按日历日期×地理块等权平均块RMSE作为真实objective；总体RMSE、MAE、Bias和覆盖率另报|PASS：不是按小时条数假装多日期；密集区域不自动占更高权重；失败预测不能丢弃难样本；保留区标签改变不影响适配/选型|
|公平与兼容|硬质量门、目标网格、同一有限参考掩码、训练状态和预算保持；可显式selection_policy=pooled_rmse_v1恢复旧汇总目标|PASS：无多源奖励，无按复核结果反复调参；新默认目标会改变后续部分排名，旧结果不改写|
|可见解释|service.fusion_evidence、result_summary、output_review显示实际选优指标与汇总RMSE的区别，传递逐日/区域结果及配对变化；DS公共工作规则说明不同变量可对齐、不可直接平均|PASS：共享显示函数已接入；本轮未做当前桌面完整人工点击验收，不能等同视觉验收|
|压缩工程NetCDF读取|ProjectFiles.netcdf统一暂存当前选定成员，用现有netCDF4路径后端并在上下文结束清理；工程来源、端口、组件、task_dispatch和原生字段读取复用|PASS：修复WRF压缩成员上的HDF5反复seek卡顿；错误路径清理测试通过，非包名补丁|

**协议与评价含义。** output/cross_variable_validation_20261122/protocol.json在数值运行前固定种子11、24次提议、150次搜索内核调用、230次总调用和60秒上限。新目标为 J = mean_{有实际参考的日期、区域}(RMSE_block)。不同来源数量不进入J；所有候选使用完全相同的参考掩码与块。仍以真实执行反馈选型，DS只能在已登记适用方法中推荐，不能提交误差或赢家。仅3个相邻日期不具备独立统计重复；报告块间变化，不将块标准差当置信区间，不承诺更稳就一定更准。没有新设“多源必须胜过单源”的门槛。原总体RMSE和原旧协议可复现语义保留，但新J与总体RMSE不是同一个数值。

**真实降水链路。** 复用已规范化的NASA POWER和CMORPH全部3个共同日期2018-02-03至05、同一0.1度目标网格及坐标分组，参考为全部来源先降采样后的原细产品，不是独立雨量计真值。原先已暴露的数据仅作探索性开发验证。DS官网调用成功，推荐3条已有组合，三条均经真实契约及数值验证通过后进入推荐父代。共享随机图变异实际评价17图、24次提议，72次搜索内核调用、1次部署调用，失败调用0；不是官方GOLEM的新优越性实验，也不是全局最优证明。DS推荐耗时8.34秒，搜索服务账本约1.35秒（不含DS与输入准备）。原始输入不变；输出3×30×25格点，非负、有限并实际回读。

|同一选型队列279个样本|日期×区域等权RMSE（实际J）|总体RMSE|MAE|覆盖率|
|---|---:|---:|---:|---:|
|最佳已执行单源|9.400280|13.008556|7.396436|100%|
|最佳已执行多源|10.846348|16.751142|8.660685|100%|

选中单源7d1e002146f28d3bb60b。冻结后的区域复核81样本：单源RMSE17.200079，多源24.065388；不反向用于选图。多源只在36个日期×区域块中的9个块降低误差，不能声称稳定融合增益。不同于历史固定0.75/0.25混合，本轮最佳已执行多源是另一条真实评估路线，不能把不同候选和不同目标的误差变化解释成单一机制因果优势。保留旧17%负结果及原报告。最终源码下另外只执行已冻结的2图检查（0提议），指标与本次搜索完全一致；后续新增配对展示仅重算保存的真实预测，不增加搜索或改选图。

**真实跨变量案例。** 使用data/basin_ecology_2013_2017/prepared_v1/standardized.nc中既有真实地形协变量dem与slope，按行列顺序取第一个两者均有限的8×8窗口（只依据覆盖，不依据误差）。通过公共服务分别重网格至明确0.02度目标，各输出2×2，保留m和degree，保守平均保持各自物理变量。输入及输出SHA256、选窗规则、实际算子、单位和回读记录见real_cross_variable_acceptance.json。该数据为既有准备产品，不冒称新原始实测；该案例证明对齐和数据依赖，不证明跨变量融合改善精度，也不宣称这个目标尺度最优。

**桌面WRF原包。** WRF4.8_Jan2000_12h_native_inputs.zip共3个成员：namelist.input、wrfinput_d01、wrfbdy_d01。初始尝试在压缩成员的h5netcdf维度读取中卡顿/异常，有限诊断发现多个公共消费者仍直接使用ZIP流；失败日志保留。公共ProjectFiles路径读取修复后，原包从同一桌面处理函数完成ENGINEERING_BOUND，并保留实际绑定及最终检查记录。不是融合或转换完成：该包未包含另一个独立来源及验证观测，本次没有数值融合、没有原生字段更新或WRF运行。当前输出为绑定报告包，不含原生工程副本，最终格式UNKNOWN、工程FAIL、模型NOT_RUN；该FAIL是交付包不能作为可运行工程验收，不能据此断言原始wrfinput/wrfbdy文件损坏。本轮不复制原文件来制造“已转换”。原ZIP SHA256前后相同。详见wrf_native_audit.json、wrf_probe_result_6.json及acceptance_final.json。

**测试及保护。** tests05.xml（44项）、tests06.xml（58项）、tests07.xml（11项）去重103项全部通过；另先通过28项受影响空间关系回归，存在重叠，不相加宣传。覆盖跨变量约束、DS建议进入公共门槛但非法混合被拒绝、来源不变、时间支持、固定评价队列、外层隔离、实际父边、组件及既有转换入口。最初pytest默认临时目录权限错误、4项测试预期异常类型错误及WRF失败均保留，不屏蔽。source_before/source_after、source_fingerprints.json、artifact_manifest.json保存修改与产物证据；早期数值运行后的显示代码变化另由最后2图一致检查核实，不声称每次历史运行用了最终显示源码。

复现定向测试：设置PYTHONPATH=src;integrations/existing_air;integrations/local_swat、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1、QT_QPA_PLATFORM=offscreen，用.venv/Scripts/python.exe -m pytest tests/test_cross_variable_blocked_validation.py tests/test_fusion_evaluation_protocol.py，指定一个新的--basetemp目录。真实搜索请求、推荐、预算、图、逐图预测、产品和独立复核位于three_day_search；真实跨变量产品位于real_cross_variable；原始数据只读。重新运行客户端必须换输出目录，不覆盖本轮证据。

**结论。** 跨变量语义对齐和日期—区域真实评分已进入公共路径；有限真实搜索与产品回读通过，融合没有获胜。不同物理变量的对齐不等于自动发现因果互补，更不等于有精度收益。当前只支持有明确契约的规则经纬网格与时间窗口，不强行将任意WRF曲线网格、多层变量或UNKNOWN语义套入同一算子。当前缺少独立观测及更多独立日期，不能给出普适融合精度或统计稳定性结论。WRF只完成绑定/问题显式返回，未完成融合与运行；不将这一项标为成功处理。未启动正式研究、未自动提交GitHub。


### CAMx建议局部恢复与按需表头加速（2026-11-22）

用户提供桌面问题清单的5条结果，并要求减少整轮退回。读取原清单、适用AGENTS、当前总报告及公共调用者后，本轮只修公共识别/绑定/任务准入与I/O；未改科学算法、评分或历史结果。允许“不适用建议局部跳过、其他已满足契约的步骤继续”，不等于允许DS编造单位、CRS、风矢量、观测或真实质量PASS。所有实际处理仍由原公共服务逐项验收；无法执行的操作保持未执行。

|原问题|本轮处理|真实证据及范围|
|---|---|---|
|C1 t2↔TMP没有单位|轻量read_header原来手写空单位字段，绕过已有tabular/NOAA复合解码器。现复用tabular与noaa_global_hourly，公共discovery依据唯一、同物理量分量绑定TMP.air_temperature（degC）；可以消费正常DS协议明确的air_temperature声明，保留原字段与重绑定记录|原包重测C1 fusion与compare_station_grid均通过时空配对并进入station_grid候选；这是适用性核验，不是融合已执行或精度提高|
|C2 u10↔WND|WND是方向/风速等复合编码，不能当数值列，更不能把风速直接当u10。无合法同物理量分量时自动跳过这条错误建议|原包实际检查标为SKIPPED_UNSUPPORTED_PROPOSAL，原始风资料保留；未新增风矢量转换内核|
|C3臭氧Sample Measurement单位|已从真实表头/前缀明确找到Units of Measure列和Parts per million样本，提取unit_column/sample evidence，不再简单声称用户文件未给单位|该CSV实际2,086,401,212字节；本轮只取16KiB前缀。全评价队列的单位、站点/时区/空间契约尚未闭环，不把样本单位冒称全表一致，不宣称臭氧融合可执行|
|C4经纬度配对、C5坐标X融合|在公共discovery标为不适用建议自动排除，坐标仍保留用于定位。issues与页面区分“已自动跳过”和“待核实”，不再把它们全算成待用户补填|原包C4排除；C5由相同坐标门与定向测试覆盖，未另提取巨大初始场作无意义重测|
|整包被无关建议一起否决|页面已识别project/declared工程时，允许继续调用原公共工程核验服务；原服务仍决定每个实际操作的单位/状态/质量与写回条件。没有工程证据的空选择/未知任务仍拒绝|新增Qt处理函数测试证明可进入公共执行函数，不能据此声称产生产品；已有必要阻塞与空状态测试保留|

扫描慢的可确认原因：原file_tools.selected会先完整解出待读CSV，才取前16KiB；每轮工具使用独立临时目录，重复解内层归档并重复扫描tar目录；DS多轮网络等待是另一项成本，不能从本地耗时推定其已消失。现在文本表头在ZIP/TAR中直接读取前缀，保留真实成员大小与原始引用；复用本轮内层归档、tar元数据索引和相同请求的表头结果。会话finally清理缓存，相关输入签名变化拒绝旧缓存。缓存不跨任务滥用；7z保留原读取支持，不虚称它也具备前缀随机读取。扫描器还修复了把不存在属性填为空串导致所有普通变量被误标为quality_flag的问题，仅传真实存在的属性。

桌面原包02_CAMx_两日嵌套工程与站点观测 - 副本.zip实测：NOAA表头约2.11秒，重复表头约0.0007秒；嵌套EPA约2GB CSV表头约2.20秒；955MB气象tgz目录约8.36秒、目标气象NetCDF表头约22.04秒。当前版本一次选定NOAA/气象/EPA表头及C1/C2/C4公共核验总计约42.96秒，其中配对核验约8.90秒；没有DS在线重放、全包融合、搜索或模型运行。当前EPA单位列证据另一次真实读取约2.08秒。均记录会话临时目录已清理。上述是特定本地选定读取的计时，不能承诺完整DS流程恒定几十秒；公共审计另分local_tool_seconds、api_and_orchestration_seconds和cache_hits，避免把API思考算成表头I/O。

定向测试去重55项全部通过（tests04—08.xml），覆盖前缀与真实成员大小、共享NOAA解码、重复请求缓存与类型一致性、内层容器/索引复用、签名变化、正常/异常清理、单位列样本不冒充全表单位、缺失flag属性不误标、未知条件仍阻塞对应操作、已绑定工程局部继续、页面提示区分跳过。初次测试发现缓存JSON往返改变Path类型，已改deepcopy并复测；失败记录保留。修改前后快照、原问题清单、当前行为与源码指纹见output/camx_continuation_scan_20261122。

本轮结论：温度配对的识别/绑定问题已修复，错误风/坐标建议自动排除，已绑定工程不再因无关候选整体退回；表头路径已取得真实局部加速。EPA全队列单位及观测适配仍待完成，不得称5个项目全部完成融合。未运行本次全GUI+DS+融合+原生写回，所以不宣称原包整链已通过。未改原始数据、未覆盖历史结果、未同步GitHub；重新打开桌面平台后再扫描才使用新逻辑。目标是尽量不中断可完成的工作；无法恢复的关键科学条件不能靠默认值强行PASS。


### CAMx原包温度实际运行与评价机制核验（2026-11-22）

用户要求用桌面CAMx原包实际运行并检查评价体系。本轮读取适用AGENTS与最新报告，复用上轮真实表头/原始定位和公共discovery、recommendations、selected_processing、station_grid及输出打包器；没有重扫无关变量、展开2GB臭氧数据、增加科学算法、修改原始包或覆盖旧结果。原AGENTS禁止变动评分的旧合并边界，与本轮明确核验评价、必要修复的授权冲突处以最新请求为准。输出目录output/camx_evaluation_run_20261122，旧算法源码及真实旧运行均保留。

**发现并修复的边界。** 当前站点路径原先前半时间拟合均值偏差，后半时间同时用来选择背景/订正和报告收益；该成绩是选型成绩，不是独立复核。本轮在查看新复核结果前冻结protocol.json：共同队列按真实时刻排序，前50%拟合、中间25%选型、最后剩余时刻只复核，至少9个真实时刻；分区边界和算法不依误差调整。完整科学实现仍是已有逐站常数均值残差，不新增算法或调参。另发现最近匹配容差将DatetimeIndex.asi8直接当纳秒，微秒存储会把一小时误作3.6秒；公共matching_tolerance显式as_unit('ns')后计算半步长，保留真实时间标签，不猜时区。

精确匹配与最近匹配的原始队列不同，现由公共common_queue固定所有已登记合法匹配方法共同的station/grid_time与同一观测标签，再拟合与评价。最近匹配本例获得25个站点/网格时刻，精确匹配9个，主比较共同9个；最近匹配只保留36%的已配对时刻进入主比较，已单独记录，不能把共同9点的100%预测覆盖宣传为全文件100%覆盖。缺观测不当真值；背景缺失、未配对、同距聚合及原始匹配数在queue_audit中明列。本例无背景缺失。原同距同站点时刻均值聚合的科学口径保留并记录matched_observations，不把重复报文当独立重复。

**真实执行与来源。** 原包02_CAMx_两日嵌套工程与站点观测 - 副本.zip中的12km t2与NOAA Pittsburgh站TMP.air_temperature，公共解码/质控、摄氏转K、原生坐标定位及实际时间配对后计算。当前选型使用2016-06-10起9个共同站点/网格时刻：4个拟合、2个选型、3个最终后段。DS官网推荐真实成功（6.91秒），仅接收已登记配对组合和元数据，不接收验证真值。为完整对照，公共执行器实际执行精确/最近两种合法配对及相同背景/订正方案，而非把DS建议当效果结论。此station_grid路线没有变异搜索，不称GOLEM或全局最优验收。

旧协议先实际执行并保留old_summary.json、old_protocol_results.zip：后半5点背景RMSE0.776829、订正0.945657，按该后半数据选择背景；不再把这组数字当独立复核。修正协议的结果如下，温差K与摄氏温差数值相同：

|阶段|实际样本|背景RMSE|订正RMSE|说明|
|---|---:|---:|---:|---|
|中段验证|2|0.480795|0.140083|只据此冻结选择订正|
|最终后段|3|0.922856|1.215468|订正误差增加31.71%，未反向改选图|
|次日冻结补核|6|1.474479|1.925894|订正误差增加30.62%，不重拟合、不重新选型|

后段背景/订正MAE分别0.748110/0.895188，Bias为−0.194008/−0.814455，R²为−0.906710/−2.307526；次日MAE1.317492/1.549315，Bias−0.926771/−1.547218，R²0.836155/0.720475。共同有效队列上覆盖率100%、执行失败率0%，不等于总体匹配覆盖率100%。全部机器指标见metrics.csv，真实逐行产品为Parquet，压缩包与表格均回读。

次日补核先冻结second_day_protocol.json，读取实际20160611气象文件，核验原生网格签名一致，仅应用已冻结约−0.620447 K的站点偏差和已选方案，剔除与首日复核重叠的午夜，得到6点；没有将次日标签用于再训练。该补核有一个导出索引断言错误：原DataFrame保留过滤前索引而index=False Parquet不保存它，真实数值计算及产品已完成；之后从保留产品回读，用公共apply_frozen_correction逐值重放确认，未重新配对/拟合获取有利结果。原失败日志保留，恢复结果只报告精确匹配路线，不伪称另一条路线也重新执行成功。

**结论及评价解释。** 本次均值订正不稳定：拟合段推导向下校正约0.62 K，中段偏暖故表现改善；后段及次日背景已偏冷，继续减值反而增加偏差。这个数值分解支持“当前常数偏差不随时段适应”，不能证明数据完全无融合价值，更不能证明所有融合方法无效。只有1站、2个选型点、3个后段点和6个补核点，时间相关且历史已暴露，无统计显著性、空间泛化或全新盲测结论。新结果明确标NOT_CONFIRMED；页面按验证误差排名，展示后段误差但不按其重新排序，并说明选中不代表复核收益成立。

**交付范围。** 实际完成公共任务的读取、DS推荐、站点配对、局部订正对照、三段评价、次日冻结补核、Parquet/ZIP导出与回读。选中的局部表格并不等于已更新整个CAMx工程：本轮没有原生网格写回、没有启动CAMx模型，也没有执行臭氧/风分量融合。已验证的是温度这条已通过配对的路线，不是全包全部变量均通过。最终可读产品包CAMx_temperature_evaluation_verified.zip含真实表格和核验记录；旧包与失败记录保留，不覆盖历史结果。

定向测试去重25项通过，包含最终时段标签变化不改变拟合状态/选型、验证可拒绝无益订正、同队列标签一致性、失败预测不得丢弃、us/ns容差一致、页面按验证排序且公开复核未通过。源码修改仅desktop/station_grid.py与output_review.py，另加针对性测试。原运行约63.46秒，修正后公共任务执行约33.20秒，DS耗时另列；这些不是公平速度对照，存在读取缓存和路径差异。次日首次脚本在计时汇总前被索引断言中断，未伪填总耗时。原始包大小/mtime签名前后一致，操作只读，不称完整原包官方校验。未做当前GUI人工全流程或模型程序验收，未同步GitHub。

## CAMx温度分支接回公共变异服务与页面验收（2026-11-23）

本轮对应“温度为什么没走变异、是平台问题吗、把这些问题修正”。旧 `output/camx_evaluation_run_20261122` 的固定订正、失败记录、次日补核和结论全部保留，不把其改称变异结果。新协议、修改前后源码、实际产品和失败日志集中在 `output/station_mutation_integration_20261123/`。只完成有界工程修复，不启动历史全量实验，不恢复已移除的官方GOLEM，也不新增科学算法。

**原因及修改。** 原 `desktop.station_grid.run` 直接循环预登记的时空配对＋均值偏差订正，未进入公共搜索；正常空间搜索确实要求至少三个监测位置，原包只有一个NOAA站。前者是编排缺口，后者是合理的空间留点条件，不能通过把时间当空间或放宽站数来绕过。本轮将原固定入口保留为显式 `run_fixed` 历史/诊断路径，生产fusion入口改为“真实配对 → 共同评价队列 → 公共 paired_station_calibration 适配 → 公共 field_mutation → 原 composition_optimizers.random_graph_search → NumericalDAGExecutor”。这是单位置时间校准，不是区域空间插值验收。新增 `algorithms/paired_station_search.py` 只承担数据、训练状态、端口和独立评价适配；未复制优化器、画像映射或科学内核。

|已定位缺口|公共修改位置|验收及边界|
|---|---|---|
|站点温度融合绕过变异|desktop/station_grid.py；algorithms/platform_search.py、paired_station_search.py|原CAMx/NOAA温度路线实际进入公共搜索；空间至少3站要求保留。纯配对/转换不冒称误差搜索|
|推荐只停留在固定流程，没有成为变异父图|algorithms/field_mutation.py、composition_optimizers.py|推荐先过Registry适用性、实际验证误差和质量门；合格图进入公共变异父队列。真实mutation_trace记录推荐父图→结构插入→执行反馈。多父推荐若无端口绑定明确拒绝，不编造依赖|
|拟合状态不在真实DAG父边及缓存依赖中|algorithms/autonomous_decision.py|状态边界节点resource.state → 订正节点state端口；实际执行轨迹含状态指纹。修改状态会使缓存失效并改变真实输出；状态来源/训练截止仍由已有PriorSeriesCalibration核验|
|只有初始化/缓存也可能被误认为变异完成|algorithms/field_mutation.py|另计mutation_actual_new_executions；本适配器要求至少一次真正新变异执行，否则返回MUTATION_NOT_EXECUTED，不以已有产品代替完成。初始化截断限额与种子洗牌写入记录，不声称覆盖全部图空间|
|页面线程只接收str，公共数值服务发送dict事件|desktop/__main__.py Job|公共后台事件统一序列化后发给页面；批次反馈仍保留原对象协议。定向测试证明事件不再导致类型错误|
|结果卡旧字段不适配公共数值结果、显示内容容易误导|desktop/output_review.py|实际样本分母denominator映射为展示count，避免渲染KeyError；展示新变异次数、验证误差、未参与选型的后段复核、未证实收益、未原生写回/未运行模型。已用真实结果渲染当前Qt控件并回读ZIP|

**信息隔离及候选。** 沿用前轮同队列：精确配对和最近时间配对先取相同真实站点/时间标签；不以两条配对重复记录冒充独立样本。9个公共时刻分为前4点拟合、中2点验证、后3点仅复核。均值残差仍为约−0.620447 K，以既有 `bias.prior_series_affine` 的斜率1和训练截距状态表示；可识别时提供既有ratio状态，不捏造月系数。未来标签不进入预测源、源身份、拟合状态、特征或候选偏好；完整搜索测试将后段标签加10000后，选图指纹和验证指标不变，只有最终复核改变。无可靠模型软评分证据时本适配器明确soft_guidance=False，仍调用既有硬兼容、真实误差和随机图变异，不宣称实现了温度精度预测。

DS建议复用上轮保存的真实线上响应 `camx_evaluation_run_20261122/ds_receipt.json`，来源明确记录为历史真实响应复用，本轮没有新增DS调用或token。时间配对/单位转换/坐标匹配为已实际完成的前置步骤；“station.bias.holdout”科学含义绑定既有先验订正内核，不把该名称另写为选择规则。推荐不能保证获胜，最终仍按真实验证误差与公共质量门选择。

**原包真实有界结果。** 预先固定seed=11，预算objective_limit=80、total_call_limit=120、proposal_limit=256、wall_seconds=30；初始提议限64，为后续变异保留额度。最终当前数值版本记录位于 `final_acceptance/real_summary.json`、`real_public_client/group-1/mutation/station-1/search/`。实际4个候选评价、3次科学内核调用、1个真正新变异执行、256次筛选/生成尝试，按资源上限停止。两条推荐在共同队列上编译成相同订正图，第二次明确计为缓存复用。真实变异从推荐订正父图出发，将既有ratio候选接为missing_fallback另一父边；下游真实读入父输出，当前无缺失情况下输出等价、没有额外收益，该复杂图未替代更简单的订正图。没有宣称预算内找到全局最优。

|真实指标|最终记录|
|---|---:|
|选中订正图验证RMSE，2点|0.140082534 K|
|选中图后段复核RMSE，3点|1.215467684 K|
|原背景同后段RMSE，3点|0.922856292 K|
|选中图相对背景后段RMSE变化|+31.7071%（更差）|
|后段收益验收|NOT_CONFIRMED|

因此：编排缺口已修复，变异确实运行；该有限数据下变异没有提供精度增益，常数订正在后段仍变差。没有根据复核结果更换种子、权重、站点或重新选图。历史已暴露且样本极少，这不是盲测、显著性或空间泛化证据。

**成本、产品及保护。** 最终公共客户端总耗时约43.47秒，搜索内部约9.41秒；训练状态拟合单列在paired_protocol.json，算法/缓存/失败/部署调用成本见total_cost.json，时间差包括输入读取、适配与导出，不能宣称为纯算法加速。前一次同协议运行和失败日志也保留（首次完整客户端约49.96秒），最终重验是初始化记录字段补齐后的受影响检查，不是增加种子筛选好成绩。产品为K单位、5个应用时刻、1×1真实站点位置的NetCDF（含验证和复核时刻预测），经xarray回读、有限性/单位检查及ZIP CRC；不等于完整CAMx网格产品。最终产品包 `final_acceptance/CAMx_station_mutation_products.zip`。原包size/mtime签名前后一致，只读访问；没有计算并冒称官方整包哈希。

**定向测试与界面范围。** `tests02.xml` 52项、最终受影响 `tests06.xml` 10项、`tests08.xml` 3项，去重57项通过；包括公共推荐父图实际变异、真实状态父边/缓存失效、未来标签完整搜索隔离、共同队列、微秒/纳秒容差、旧站点融合契约、候选导出、Qt控件、预算及现有变异行为。最终结果卡文字变更另由tests09及真实产品回读渲染核对。保留tests04失败：新增检查误读不存在的selected_graph键，修正为公共结果best.key；保留tests07失败：旧进度测试把构造阶段已有“评价依据”也算成候选执行事件，现单独验证初始化消息并比较执行增量，缓存命中仍不得新增执行消息，未删除失败断言或屏蔽异常。初次pytest缓存目录权限警告和第三方弃用警告保留；后续显式关闭pytest自身缓存，不改变产品缓存/数值逻辑。

当前Qt结果卡已使用真实最终产品渲染，截图为 `real_result_cards.png`，只称控件级验收；没有人工从桌面重新选包点击全流程，不冒称浏览器或模型运行验收。页面execute_selected、工具客户端selected_processing.run和适配器共享同一submit_search；Job事件类型与页面结果卡也有定向执行证据。原温度路线不再静默跳过搜索。

**同类入口核查。** `route_audit.json` 明确分层：普通空间/独立参考/站点搜索已有公共调用；本轮只读核查其调用者，不冒称全部重跑。带已提供不确定度的固定加权融合、纯格式转换、统计和跨变量对齐属于明确固定操作，不因此自动获得“搜索完成”身份；无合法评价依据不能制造误差优化。仅一个站的空间泛化、臭氧完整队列、风分量科学转换、整网格写回及CAMx程序运行仍未在本轮完成；不把这些未验收内容改为PASS。后续若要验证区域融合价值，应提供足够空间观测和独立评价支持，不由本轮自动下载或训练补造。

复现：设置PYTHONPATH为src与既有integrations后，运行 `python -m pytest tests/test_paired_station_mutation.py tests/test_station_grid_evaluation_isolation.py tests/test_desktop_process_progress.py`。真实客户端分别为run_real.py和run_real_final.py，输出目录不可覆盖，若需复现应显式提供新的目录。verify_delivery.py仅回读已完成产品、核验指纹和渲染，不重新搜索。当前集成未自动推送GitHub，未启动正式科研实验。

## 分块初筛与冻结方案一次复检接入公共评价服务（2026-11-23）

**范围与依据。** 本轮修复“少量时刻选型后，换时段失效仍被推荐”的评价与交付缺口。选择固定校准状态下的分块验证＋独立后段一次复检：借鉴 [scikit-learn内外层选型/评价隔离](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) 和 [sktime分窗回测实现](https://github.com/sktime/sktime/blob/main/sktime/forecasting/model_evaluation/_functions.py)，复用平台已有 reliable_selection.aggregate、DatasetReference/SearchValidation、真实DAG执行与质量门，没有复制优化器，没有增加预测算法、学习模型或大型依赖。这里不是每折重新拟合的完整嵌套交叉验证，也不是对效果的事先保证。

新协议和证据集中于 `output/blocked_confirmation_integration_20261123/`；修改前后源码、各次失败和真实运行全部保留。原CAMx包、上轮所有结果及原9点公共队列不变；没有增加随机种子或根据复检调参数。候选和评价逻辑均在公共模块，页面不维护独立评分规则。

|改进|修改及实际消费者|核验|
|---|---|---|
|按时间块评价，不再只给单段分数|evaluation/blocked_confirmation.py；公共 BlockedSearchValidation、BlockedReference；field_mutation的实际终端评价入口|按真实经过时间分3个窗口，不按行号或目标数值划分；所有有限参考点必须预测，不能只算成功子集。各窗口RMSE等权平均反馈搜索，另列整体RMSE/MAE/Bias和分母；缺窗口时仅作探索诊断|
|复用已有多折汇总|调用 reliable_selection.aggregate，RMSE权重1、物理单位尺度1，不借用NDVI专用校准阈值|受控功能输入证明：关闭分块时选择整体RMSE更小的B，启用分块后选择跨窗口平均RMSE更小的A；真实Executor计算，不是仅写日志。人工测试算法只存在于测试内，不加入平台算法库|
|冻结后一次复检|field_mutation在读取外部复检前写selection_freeze.json；paired_station_search调用公共review|复检时刻必须与初筛不重叠且严格在后；变量、单位、队列和分母必须匹配。禁止用同一批标签既选型又声称独立复检。只评价一个冻结候选及其固定背景对照，不重新挑候选或重抽样|
|推荐与执行成功分开|公共review及paired_station_search默认协议blocked_confirmation_v1|要求两阶段每个窗口至少2个真实有效时刻、整体分块损失严格改善、各块RMSE及MAE均不恶化、质量门通过。最少支持仅是预先固定的保守工程规则，不是统计置信门槛。复检无收益、证据不足、质量失败和搜索未完成分开记录|
|不可靠增强不成为默认交付|推荐关卡不通过时保留已通过质量门的source基线，另存candidate_product与冻结候选图|不是把复检最好的另一候选重新选为赢家。没有安全合格背景时明确失败，不用未验收来源兜底。候选仍保留供探索查看，默认产品为背景，不改原始输入|
|页面与下载一致|desktop/output_review.py、result_cards.py、station_grid.py|显示“背景保留、增强未推荐”及原因；默认保留不写成“本次最优”。下载包同时含明确分开的背景产品、探索候选、图、冻结与复检记录。实际Qt结果处理函数与导出处理函数用真实产物通过，不冒称人工全流程点击验收|

**冻结规则。** 协议版本blocked_confirmation_v1，3个等实际时长窗口，每块最低2个不同有效时刻，独立后段同样核验；零损失附近不计算不稳定相对收益率，仅采用机器舍入容差比较，未手填经验收益分数。原数据仍前50%拟合、中25%选型、余下后段复检；固定拟合状态在验证各窗口间不更新，复检标签不进入状态、特征或候选生成。窗口离散程度明确不称置信区间，窗口也不当作独立重复样本。当前最少支持规则很保守，但通过也只表示在这些观测窗口内成立，不能保证未来季节、区域或模型效果。

**原CAMx/NOAA真实验收。** 最终证据为 `real_verified/summary.json`、`public_client/group-1/mutation/station-1/` 与 `CAMx_blocked_review_products.zip`。使用原DS真实响应复用，不新增线上调用；seed11，原budget80次搜索内核/120总调用/256提议/30秒保持不变。本次仍实际评价4个候选、科学内核调用3次、新变异执行1次。算法没有被换成有利的新方法，冻结候选仍为先验均值偏差订正。

|项目|结果|
|---|---|
|原公共观测队列|9个时刻，4拟合／2初筛／3后段复检|
|初筛窗口有效时刻|1／0／1，达不到每窗口最低2时刻|
|后段窗口有效时刻|1／1／1，同样不足；且2个窗口误差变大|
|候选初筛整体RMSE|0.140082534 K（探索诊断，不是可靠收益证明）|
|候选后段整体RMSE|1.215467684 K|
|背景同后段整体RMSE|0.922856292 K|
|增强推荐状态|EVIDENCE_INSUFFICIENT；后段stable_gain=false，改善未证实|
|默认交付|BASELINE_RETAINED；背景后段RMSE仍为0.922856292 K|

正确变化是拒绝不可靠的增强推荐，不是把融合本身改好了。候选相对背景变差31.7%的事实仍保留；不能把回退到背景称为新融合精度提升。新机制也不声称这份只有一个站、少量历史已暴露时刻的数据具备统计确认能力。初筛可在有限预算内继续作探索搜索，但不能凭产生了产品就自动获得推荐身份。

**行为测试与兼容。** 去重45项通过：tests06.xml为25项，tests04.xml为15项，tests07.xml为5项；包含块评分真正改变执行器选型、同队列与失败预测、微秒/纳秒一致、重复/非递增/数值伪时间拒绝、不规则时间按实际时长划块、初筛与复检重叠拒绝、单块恶化不得被整体平均掩盖、证据不足不得通过、质量门、预算中止、冻结先于读取复检、改变后段标签不改变搜索选图、默认背景产品真实回读、状态父边和缓存、旧变异与候选导出/进度。另完成当前Qt Window.show_result与export_now真实函数验收、ZIP CRC及默认产品逐值对比背景；截图current_result_page.png。未进行人工从桌面选包点击验收，未运行CAMx模型、未原生整网格写回。

保留最初tests01失败：测试构造的参考与输入共用数组，修改预测同时修改了后来创建的旧评价参考；修正测试夹具深拷贝后复测，不屏蔽断言。另有一次PowerShell生成客户端的引号错误，发生在真实执行前，未生成结果，已改用保存的Python客户端。最终验收脚本verify_delivery.py只回读已完成结果、验证默认导出、渲染与存档，不再搜索。旧接口SearchValidation和未传自定义评价器的GraphMutation保持原行为；站点历史复现可显式传evaluation_protocol=legacy_single_v1，其旧评分和交付行为有单独回归，不改写原历史报告。

**支持边界与成本。** 新评价器是公共接口，可用于有合法时间参考的字段任务；默认推荐关卡本轮实际接入站点—网格时间校准公共适配器，页面/工具包调用同一submit_search。既有空间留站、无真值产品重建代理、纯转换等不被强行改成时间分块精度检验；没有参考观测时，不能伪造这套精度复检。本轮没有用一个CAMx样例冒称所有任务均已完成新关卡验收。

最终客户端总耗时约12.38秒，搜索内部约2.27秒，拟合与调用明细另见paired_protocol.json及total_cost.json；读取缓存和环境状态不同，不用于宣称速度收益。三次有界真实运行分别保留于real_acceptance、real_final、real_verified：后两次用于时间标签/隔离校验及最终版本确认，不改种子、预算、窗口或收益要求，所有负结果保留。未修改第三方源码，未重跑历史大型对照或穷举，未自动同步GitHub。修改前后源码与SHA256、产品回读、当前GUI处理函数和下载证据统一见acceptance.json。

复现定向测试：`python -m pytest tests/test_blocked_confirmation.py tests/test_paired_station_mutation.py tests/test_station_grid_evaluation_isolation.py tests/test_desktop_process_progress.py`，沿用项目src与integrations的PYTHONPATH。真实客户端run_real_verified.py使用不可覆盖的新目录；再次复现应显式指定另一个新目录，不覆盖已保存的本轮记录。


## 通用融合入口的任务分块复检与安全交付验收（2026-11-23）

**本轮范围与结果。** 修复上一节只将新推荐关卡默认接到站点时间校准的限制。复用原有时间、空间留站、日期×区域评分、任务匹配缺失模拟、真实图执行和质量门，不新增科学算法、不下载数据、不调参争取多源获胜。公共服务版本为 `unified_fusion_review_v1`，基础关卡规则仍为 `blocked_confirmation_v1`；历史文件与旧结果全部保留。证据集中于 `output/unified_fusion_review_20261123/`，机器验收入口 `acceptance.json`。

|入口/任务|公共衔接及实际消费者|本轮证据|结论及边界|
|---|---|---|---|
|站点—网格时间校准|paired_station_search → platform_search.submit_search → 公共review；子任务交付状态向父任务汇总|原CAMx/NOAA真实队列重新有界运行，paired_CAMx|功能PASS；9时刻支持不足，背景保留，不推荐增强|
|站点、站点+背景、协变量空间估计|SpatialTask已有留位置和日期×区域指标 → 冻结候选/单源对照 → review_spatial → 公共finalize|3个真实西澳站点的90日降水与ERA5；station_grid/auxiliary的定向行为测试|公共路径PASS；真实3站不足以留出可靠独立区域，EVIDENCE_INSUFFICIENT；不能宣称空间精度改善|
|多格点产品、仅融合|既有面积降尺度重建协议与外层空间留块 → review_spatial；不是拿未变化原产品与自身比误差|原Saguling两份降水产品、3个现有共同日期；页面函数和JSON工具包真实执行|PASS；产品重建参考，不能替代独立雨量真值；冻结最优与单源对照相同，NO_STABLE_SELECTION_GAIN|
|多源字段缺失恢复|task_validation.reserve_review整体封存外层日期/位置，遮挡别名及重算特征，重新生成内层任务匹配队列；field_mutation真实执行冻结图及对照|真实WA站点与ERA5的恢复任务；独立三季边界夹具实际执行外层图|PASS；真实短序列只形成两个合法任务匹配案例，不能强造第三个，明确证据不足；功能夹具不作环境精度证据|
|合法外部验证与复检参考|SearchValidation在公共入口转为任务适用BlockedSearchValidation，冻结来源对照后一次后段复检；真值仅在评价侧|真实WA前30日可见、随后30日选型、最后30日复检，背景全90日|PASS；验证与复检均有3个合法时间块；未通过收益稳定性关卡，保留冻结背景|
|固定不确定度加权融合|pair_processing直接融合后调用同一模块engineering_review；不编造搜索或真值|确定性覆盖/不确定度正负边界测试|工程检查PASS；没有独立真值时明确ENGINEERING_ONLY，不能称误差下降|
|旧Profile/Autonomous等诊断/归档入口|统一finalize必须返回复检状态；显式legacy_v1保留旧数学与产品，状态LEGACY_NOT_REVIEWED|旧paired协议数学回归、公共入口路由测试|兼容PASS；缺合法隔离队列的旧模式仍是探索性，不能冒称新独立复检已完成|
|页面、工具包与模型包装|search_page.process_search_upload及submit_file_request共用submit_search；local_search、模型native_meteorology/native_reference/engineering.search_bridge经既有公共链委托，不另立评价规则|真实页面函数/工具包同seed同请求的图、指标、关卡及产品逐值一致；当前Qt show_result/export_now|函数级PASS；未人工浏览器全程上传点击，未重跑五个模型原生模拟|

**实施及安全边界。** 公共实现位于 `evaluation/blocked_confirmation.py`；编排接入 `platform_search.py`，任务适配在 `spatial_fusion_task.py`、`field_mutation.py`、`task_validation.py`、`paired_station_search.py`。复用原图缓存与执行计账，新增外层审计走同一总调用额度。空间复检要求位置不重叠，时间复检要求严格后段，时空任务同时满足；不能随机拆同站同日像元当独立复检。完整共同评价队列必须全部预测成功，块数量、身份、参考指纹和分母必须一致。要求整体分块损失严格改善且每块RMSE/MAE不恶化；这是保守工程筛选规则，不是统计置信界。依据任务类型选择轴，未将小时控制节点限制推广为全平台时间限制。

多源恢复的外层参考在搜索前整体从输入、同值别名和特征上下文封存；仅在能够重建至少两个合法内层案例时保留外层。重新规划若失去任务支撑即返回证据不足，不降级成随机点验证。为防新公共入口初始化先耗尽全部提议，现代默认初始化最多使用min(64,提议额度/3)，保留同一种子洗牌，内层循环达到上限即停；显式归档协议保持原行为。不是扩预算或增加算法。字段应用契约另修复一处名称混淆：明确声明的physical_variable作为standard_name，字段别名不再被误当物理量；真实物理量冲突仍拒绝。

**真实结果与负结果。** 最终固定seed11、80次搜索科学调用上限、120次总调用、256次提议、30秒；3站空间案例提前声明为80/180/100/30秒。全部保持原数据、日期窗口和质量门，没有为了收益换种子或重抽参考。

|案例|实际结果|默认交付|
|---|---|---|
|CAMx原9时刻队列|候选后段RMSE 1.215467684 K，背景0.922856292 K；仍变差31.7%，并且各时间块支持不足|BASELINE_RETAINED；不是融合精度改好了|
|WA真实90日任务匹配恢复|只得到两个合法匹配案例，达不到“2内层+1外层”|证据不足；保留质量合格基线与探索候选|
|WA真实外部时间留块|分块选型平均RMSE变化−0.065788408 mm，但前两个块变差；冻结方案复检平均块RMSE变化+1.092398871 mm，未稳定改善|NO_STABLE_SELECTION_GAIN；保留预先冻结的来源s1，不用复检重新选另一张图|
|Saguling两产品3日空间重建|选图与冻结单源对照相同，选型及复检目标差均为0|NO_STABLE_SELECTION_GAIN；细网格产品保持目标范围，参考是产品而非独立实测|
|西澳3站空间融合|真实执行69次搜索内核、5次部署、12次审计，总86次；只有3个位置，无法形成规定的独立外层位置支持|EVIDENCE_INSUFFICIENT；背景保留，增强另存|

上述WA实例说明新关卡能拒绝“平均初筛略好、分块不稳定且后段更差”的推荐；不证明未知未来场景一定准确，也不证明任一融合算法优于单源。真实站点日降水使用已保存metadata的24h ending 09:00 Australia/Perth（01UTC）定义；空间端口原要求按区间起点标记，客户端将真实区间的端标签统一改记同一区间起点并提供86400秒及bounds，数值与覆盖时段未改，没有伪造UTC零时日界。第一次客户端未补bounds、第二次漏interval_seconds，均保留失败记录，未修改科学内核放宽门槛。

**页面、交付与成本。** 不可靠增强不成为默认产品，另存exploratory-candidate；共同基线必须满足原质量门和完整目标范围，不能最后插值扩大范围凑回退。`delivered_graph`、`delivered_best`与实际产品绑定，稳定性诊断也读取交付图；Qt候选卡片显示“默认保留”，不再把回退称为“本次最优”。完整摘要、冻结图、失败、指标、review_service及成本均写入结果下载；给DS的新增复检信息只传汇总，不将参考日期、坐标、真值或训练状态路径送入候选提示。

最终公共客户端墙钟分别约12.63秒（CAMx）、5.09秒（WA恢复）、2.70秒（WA显式留块）、6.41秒（Saguling）；受缓存、测试并行及文件读写影响，不宣称速度提升。`summary.review_cost`记录含数据准备/后端/最终关卡的公共服务墙钟，`total_cost`继续分列真实科学调用、拟合、部署、审计和缓存；来源边界节点本身可真实执行而没有科学内核调用，不能把该0计数伪称没执行，也不能计作新增算法运行。原完整包不重新下载、不改写，CAMx复用已有真实配对表及原NOAA坐标。

**测试与失败保留。** 最终去重110项定向测试通过，0项未解决失败。详见tests05/06/07/08.xml及acceptance.json，覆盖同队列、空间/时空隔离、未知与不完整参考、真实父边、别名遮挡及特征隔离、输出范围、预算、缓存、旧协议、候选导出及页面进度。tests05/06新功能夹具失败为180帧特征准备已耗尽20秒服务预算，科学执行尚未开始；固定该功能测试的60秒额度后tests07通过，未修改真实验收预算。另一次断言错误地要求来源边界图必须增加科学内核调用，改为严格检查真实执行trace、PASS及实测指标，不强迫复杂图。早期夹具季节支持不足、缺bounds/interval_seconds、客户端导出把嵌套路径送入只接收平面文件名的下载函数等失败均留档；客户端导出改为既有公共包协议的平面产物，不为单包放宽安全检查。第三方弃用、空梯度诊断及既有pytest缓存目录权限警告保留，没有屏蔽失败测试。

当前Qt结果/下载函数用真实Saguling产物验收并回读ZIP，截图 `client_acceptance_final/current_qt_result.png`，导出 `desktop_export.zip`；只是函数级GUI验收，不冒称人工浏览器上传全流程。字体预览最初缺字，验收进程显式载入本机微软雅黑后重新渲染；未改平台布局。页面函数和工具包的两次真实搜索保留于client_acceptance/page和toolkit，后续纯显示修复仅回读这些结果，new_searches=0，不冒充新数值执行。

**复现与未验收。** 使用项目.venv及src/integrations的PYTHONPATH；逐用例命令、输入哈希、预算见run_real.py、run_real_spatial_stations.py和protocol.json。真实客户端目录采用不可覆盖mkdir，复现须另设目录，避免覆盖本轮证据。最终源码与SHA256见source_after及source_fingerprints.json；原总报告快照另存source_before/docs。未运行模型模拟、未重跑原生写回、未进行新增科研优越性对照、未自动上传GitHub。当前完成的是生产公共融合路径的统一关卡与按任务复检；旧诊断模式、缺参考、短日期或稀少位置仍明确未具备精度推荐证据，不能把“可执行”升级成“已证实融合更好”。


## 不同数据上的复检正反对照与日累计契约补齐（2026-11-23）

**目的与固定范围。** 依据“选方便、效果直观的数据，只用平台能力完善检验机制”的要求，选择本机现有三个西澳站点实测降水/ERA5、NCEP温度场，以及两处LGHAP AOD小区域。先固定2018年前180日、NCEP 2013年首6个00UTC时次、AOD前三个已有日期及区域；seed11，80次搜索科学调用、120次总调用、256次提议、40秒。未更换种子、挑选好时段或修改评分门槛。此前样本及结果保持不变。本轮为已暴露数据上的有界工程验收，不是新盲测。

所有科学候选、状态拟合、图执行、质量门及评价均调用公共 `platform_search.submit_search`、Registry和既有算法；客户端仅整理文件/坐标/真实参考、传入协议、保存输出。没有实验专用算法匹配或评分规则，没有新算法、训练大型模型、额外下载或DS调用。

**真实数值结果。** 先用公共缺失恢复入口测试可见站点序列＋背景，随后因发现累计量契约缺口，补测已有偏差订正公共入口。两个任务的候选域/拟合方式不同，不能混成同一算法的重复实验。

|数据及任务|结果|关卡及默认交付|
|---|---|---|
|Broome降水＋ERA5，后段缺失恢复|初筛平均块损失改善，但复检候选与背景相同，RMSE均0.029814 mm|CONFIRMATION_FAILED；没有严格后段增益，背景保留|
|Perth降水＋ERA5，后段缺失恢复|初筛总体改善但有块退化；后段背景4.241776 mm、候选9.287842 mm|NO_STABLE_SELECTION_GAIN；拒绝候选，背景保留|
|Kalgoorlie降水＋ERA5，后段缺失恢复|候选与背景后段RMSE均0.444665 mm|NO_STABLE_SELECTION_GAIN；没有收益不强荐|
|NCEP sigma0.995温度场，单产品空间重建|选型整体RMSE2.492011 K；空间留块复检2.608086 K，冻结图和单源对照相同|NO_STABLE_SELECTION_GAIN；这是重建产品的代理误差，不是地面实测精度|
|Broome日累计降水，已有偏差订正|基线及候选后段RMSE均0.029814 mm；固定预算内无新增有效变异|SEARCH_INCOMPLETE / MUTATION_NOT_EXECUTED；保留基线，机制执行不冒充全部通过|
|Perth日累计降水，已有偏差订正|冻结比例订正后段RMSE5.413682 mm，背景4.241776 mm，变差约27.6%|CONFIRMATION_FAILED；最终交付背景4.241776 mm|
|Kalgoorlie日累计降水，已有偏差订正|QA筛除后配对队列与前一任务略不同，基线及候选后段均0.434891 mm；无新增有效变异|SEARCH_INCOMPLETE；按固定协议保留基线，不加预算凑成功|
|LGHAP BTH及Yangtze，各12×12、3日期|CF尺度0.01和65535缺测由xarray解码一次；日产品的真实时间合成窗口未证实|公共入口明确拒绝UNKNOWN，未进行数值评价；没有伪装成瞬时观测硬跑|

ERA5列为已有站点查询位置预测，保留原metadata中的原生格点位置和“24h ending 09:00 Australia/Perth = 01UTC”定义，不把模型值说成实测。NCEP明确为sigma0.995再分析层，不冒充独立地面测温。原始下载的gauge.dly/era5_hourly.json SHA256与既有metadata逐项一致；派生CSV的实际哈希单列，不用原始下载哈希冒充CSV哈希；输入和源码哈希见acceptance.json。雨量评价只用真实且QA合格的标签，未知自然缺失不作真值。日累计适配器按QA合法配对后做半段拟合、四分之一验证、余下复检，精确队列另存paired_protocol；因此与首次按日历90/45/45遮挡的任务不强行合并统计。

**正反功能对照。** 为验证机制不是“全部拒绝”，使用真实NCEP前96个温度值作为参考，按提前固定的规则人为给输入加入+2 K偏差；反例仅在最后四分之一把偏差改为−2 K。二者通过同一公共入口、同预算、同种子，无预置完整图或手工指定赢家，使用已有训练均值偏差订正和变异。稳定偏差：基线RMSE2 K→冻结候选0 K，通过RECOMMENDED_WITHIN_OBSERVED_WINDOWS，交付0 K；后段偏差反转：候选4 K、基线2 K，CONFIRMATION_FAILED并交付基线2 K。正例因构造而可订正，只是机制敏感性证据，不是现实融合精度提升、更不是统计泛化证明。

**实际修复（公共模块）。**

|发现的缺口|修改|行为证据|
|---|---|---|
|paired入口默认瞬时量，不能合法接日累计|`paired_station_search.prepare`只接受明确instantaneous/mean/accumulation；区间量必须提供start/end、等长正区间、无重叠且一致标起点或终点；逐字段保留time_bounds，状态身份包含区间语义；不从单位/文件名猜|微秒/纳秒等价、未知/缺区间/重叠/语义冲突拒绝；日累计真实三站执行及合成正例公共数值验收。未知窗口的AOD仍拒绝，不为运行放宽|
|通过推荐时delivered_graph只有key，回放缺图；回退缺delivered_best，可能展示错分数|同一适配器保存实际完整图、实际交付评分及合法应用输入；拟合状态单独保存，预测输入不含后段真值|真实温度序列正反对照的交付图均经公共execute_frozen_graph回放，逐值等于原交付产品；真实父边读取同一合法state，不重新选图或传入参考|
|字段/外部参考的推荐路径可能只看好误差而忽略搜索未完成|`blocked_confirmation.review_field/finalize`将运行状态纳入最终验收；未完成明确SEARCH_INCOMPLETE，不能仅靠低误差推荐|定向反例：候选误差0但MUTATION_NOT_EXECUTED，仍拒绝并保存合格基线及探索候选；不调门槛|
|UNKNOWN时间错误提示被笼统写成区间平均问题|`spatial_fusion_task`明确回传当前实际语义及支持范围|两个AOD实际文件都返回UNKNOWN及所缺时间支持，不伪装数据错误已解决|

保留历史API默认瞬时量及旧数学行为；用户显式提供的累计/平均区间才走新核验。非负要求继续由公共质量门执行，不能让负降水订正因误差低而入选。未添加“下雨选某算法”等偏好，也未改变各块都不恶化、整体严格改善等原验收要求。

**图、测试与成本。** `confirmation_metrics_all.csv`保留8个案例×3种产品的24行真实计算指标（不同任务及单位分列，受控例明确标注），`review_comparison_v2.png/pdf/svg`展示原方案、冻结候选及交付差别，右图显著标明人为偏差功能对照。图像已回读，文字边界检查通过；仅描述固定案例，无显著性或置信区间。逐例记录、全部负结果和成本分别见runs_verified、interval_calibration、controls_final、delivered_graph_replays.json。两处AOD属于条件未满足，未完成数值验收；两个日累计站点属于新增变异未完成，不能改称PASS。

去重31项新增/受影响定向测试通过，0失败，详见tests01/02.xml。包含原温度链兼容、真实误差反馈、后段改变不影响选图、低预算、旧协议、推荐图记录、回退实际评分、区间语义及无状态串用。原始数据不写回；本轮没有模型模拟、原生格式写回或人工GUI全流程验收。首轮NCEP序列化出现继承整数编码警告，客户端清除历史存储编码并用解码数值回读核对，不改变参考值；随后一轮客户端变量名遮盖文件名导致格式识别失败，已修正，失败保留于runs_final，不伪称算法失败。首次回读比对整份指标字典包含参考载体元数据差异，改为逐项严格核对物理指标、分母和块损失；没有修改平台评分。另在归档校验时发现metadata的gauge_sha256/era5_sha256对应原始dly/hourly.json，并非派生CSV；核对既有采集源码后按正确文件逐项匹配通过，未修改数据或metadata。旧目录和图v1均保留。

**验收结论。** 机制在这些有界案例中能认可构造条件明确的改善，也能阻止后段变差方案成为默认交付；实际降水本轮未证明稳定融合收益。三站/单年份/单产品不提供普适可靠性或检出率估计。仍偏保守，可能漏掉总体改善但局部轻微退化的方案；不能根据这轮结果放松阈值。作为第一版工程验证的证据比上轮完整，但没有变成“自动保证精度”的机制，也没有证明多源必胜。复现客户端run_cases.py、run_interval_calibration.py仅传数据和协议，所有科学操作仍在平台；不可覆盖mkdir要求复现另设目录。源码前后快照与成本均集中保存，不自动同步GitHub或追加大规模实验。


## 合法同期站点驱动的逐时段状态更新与原数据复测（2026-11-23）

**任务与结论。** 接通已有同期站点观测、更新订正状态至公共融合/真实执行机制，不为三个站点写判断分支。不增加科学算法ID；本轮完成的是既有残差IDW、比例订正的状态契约及数据父边接线。正式搜索仍有未完成项：原三站均在固定预算内没有新增独立变异，保留MUTATION_NOT_EXECUTED/SEARCH_INCOMPLETE，不伪造完整搜索成功。更新后的真实候选没有稳定精度收益，全部保留原背景交付。

**复用与最小扩展。** 原SpatialSearch已经按时段用保留站点做残差IDW及比例/仿射估计；之前单站时间入口只有第一阶段拟合后冻结的标量状态。本轮从spatial_fusion_task原公式原样提取fit_ratio_frames到quality_aware并让两条路径共用，等价性测试保留；复用fusion.point_weights的米制距离IDW及weighted_values、fusion.residual的差值核。新增fusion.observation_state只是公共状态适配器，不是新选择器、效果预测模型或新优化器；未重写科学内核、改变算法偏好或扩候选池。

公共make_context消费既有observations角色及明确parameters.observation_state_update；配对适配器接受目标paired_observations与独立observations资源，将权限传给同一服务。研究/工具包/页面处理函数共用submit_search和同一Executor，不导入实验脚本。既有空间网格路径保留；新增动态状态适配器仅支持明确的单点序列，不把一个像元平均值偷偷当站点，也不宣称所有原生模型已重新运行。

**实际数据流与约束。** prior训练状态、原背景和合法其他站点分别形成真实端口，经resource.observation_state_update输出time索引状态，再进入原bias.prior_series_affine/ratio。原静态标量状态保持兼容；同一时段合法观测的残差经既有IDW更新截距，满足原配对数/正分母条件时经原比例公式更新系数。无可用残差或比例不可识别时回退原先验，不填造观测、不借相邻或未来时段。monthly_ratio没有这套动态契约，明确不适用，未假装消费了新观测。

必须具备同变量/同单位、同背景产品、明确坐标、区间及质量标记；目标ID与同位置别名拒绝，batch资源不得包含任何被评价目标。背景在其他站点的配对值必须由合法配对资源提供，不拿目标格点值冒当其他位置背景。历史拟合不得跨入应用区间；输入mean/accumulation严格核验同一实际窗口，微秒/纳秒表示一致。非负门继续使用原实现。动态系数必须通过真实状态生产者父边进入下游，不能通过另一分支或伪造的原始向量状态借能力。

状态更新读取的只是当前试验中明确允许的其他站点；被预测站的验证/复检值完全不进入更新。调用模式是contemporaneous_analysis，即事后同期分析，未证实传感器发布延迟，不称真实预报或生产实时采集。新资源进入下一次请求会按观测与策略指纹重算，不是后台自动联网拉取、持续训练大型模型。历史拟合状态保持只读，新状态按时段重建并记录系数、所用站点、支持数量、回退及距离。

**信息条件。** 本轮逐站留一，每次只封存当前目标站的中后段，另两个站在该次请求显式声明为可用同期输入。当前目标的CSV、QA排除、训练/验证/复检时间和共同分母与上一轮逐项一致；所有旧结果保留。新增合法输入信息意味着这不是与旧单站实验同信息条件的纯算法优越性比较。如果三站中后段都属于同一个全局封存测试集，则不能把其中任何站重新作为可用观测；batch权限校验会拒绝这种借用。本轮数据均已暴露，不宣称全新盲测。

**原始三站重测。** 同seed11、80次搜索调用/120总调用/256提议/40秒，IDW沿用既有默认power2，未按后段效果调参；三等时长块、原固定损失与所有验收规则不变。原第一阶段拟合仍仅用目标站前半数据。下表为真实后段整体RMSE（mm），不同站的自然缺失后有效队列各自固定，不把它们当独立重复研究。

|目标站|背景|旧冻结订正候选|同期状态候选|实际交付|同期比例可更新时段/应用总时段|
|---|---:|---:|---:|---:|---|
|Broome|0.029814|0.029814|0.029814|0.029814|31/90|
|Perth|4.241776|5.413682|6.045197|4.241776|12/90|
|Kalgoorlie|0.434891|0.434891|0.758896|0.434891|28/90|

新增候选确实消费其他站点并按时段更新，不是仍在套固定系数。三站残差更新各90期；比例更新分别31、12、28期，其余因原比例核的分母/有效配对条件不满足而回退先验，不能把“收到了两个站点”说成“每期都识别到了有效比例”。按现有EPSG:4326工作坐标约定计算，站间距离约536—1672km；空间代表性和共享偏差均未证实，不能把遥远站点干湿变化当作本地雨量规律。这里的负结果不证明多源数据普遍无价值，也不能靠添加算法或手调权重保证翻盘。

此外三次搜索均保留SEARCH_INCOMPLETE：真实更新节点和候选已数值执行，但预算内新独立变异数为0。原规则要求新变异的标志没有关闭；没有为了凑PASS制造等价图、扩算法池或增加好种子。窗口比较和后段指标仍保留，增强未获推荐；默认产品逐值等于质量合格背景。不能把控制节点执行计作新科学算法或图变异，也不能因未生成新图就否认状态确已更新。

**行为与端到端验收。** 去重53项定向测试通过，0项未解决测试失败：原空间方法回归、旧静态状态/旧协议、区间与单位、来源/别名/全batch目标隔离、错误CRS和未知预报模式、后段参考改变不影响选图、真状态父边、无可用观测回退、后期观测改变不影响早期系数、输入改变时更新缓存且原背景合法复用等。合成功能例专门让同期偏差在后段反向，状态能随合法其他站点改变而更新；这不是实际环境增益证据，也不读取目标后段真值生成状态。最初两次新测试失败为状态适配函数局部变量漏绑定，以及把约2.5e-15的浮点误差硬断言成0；分别修复局部变量和采用1e-12的测试舍入容差，未修改实际评分或质量门。

同一Perth真实请求经页面处理函数和JSON工具包产生相同图、指标、复检与状态报告；冻结候选通过公共execute_frozen_graph无参考、无再选型回放，产品逐值一致。原背景、先验、观测的真实父边及更新系数可见，实际新执行、缓存、失败、拟合及状态更新成本分账。未人工点击GUI/浏览器全流程，未运行原生模型或重新格式写回，不把API处理函数验收冒充以上验收。

**源码、产物和成本。** 证据集中output/observation_state_integration_20261123/。release为最终三站有界重测，run_real.py只传原文件和结构化权限；其protocol保存执行源码SHA256，原目标CSV指纹未变。delivered_client与对应日志保存页面/工具包及回放证据，comparison.csv及acceptance.json提供逐例数字、支持范围和全部限制。拟合时间在paired_protocol，真实状态变换和科学节点在total_cost，公共服务墙钟在review_cost，未据此声称效率优势。各轮修复后的有界复核均保留不同目录，不当作独立样本或用来挑选最好的结果。原始数据、旧协议、旧输出和本轮负结果均保留；不自动同步GitHub。

**算法库是否需要再改。** 当前不需要再加一批算法。原平台注册构造函数经AST比较未变，没有新增算法ID；实际补的是原算法的状态形状/有效期契约、观测资源父边、权限校验和缓存。优先验证合法的附近站点在目标时段是否提供信息，再决定有无必要引入新方法；本轮不因结果差就追加算法。功能集成已有证据，精度收益未证明，完整新变异搜索未通过，这三项必须分别表述。

## 五模型及仅融合、仅转换的分步补救与继续执行（2026-11-23）

**处置原则与实现。** 本轮将既有安全补全、来源准入、工程端口处理、组件转换和最终检查连接到公共 `core/continuation.py`。明确输出 AUTO_COMPLETED、FALLBACK、SKIP_STEP、CONTINUE_WITH_WARNING、BLOCK_STEP；BLOCK_STEP只表示记录所指步骤/来源/变量未满足条件，不再自动等于整项工作失败。原始文件不写回；产品写到新目录或工程副本。没有修改科学算法、评分、变异或验证集规则，没有增加算法ID。

|原缺口|修改位置|实际行为与验收|状态|
|---|---|---|---|
|一个来源/变量缺项牵连所有来源|`models/engineering/preparation.py`|逐来源、逐变量准入；缺单位/时间语义的变量隔离，合法变量保留；来源到端口的单位/时间不兼容只排除该映射|PASS，合成功能测试|
|一个端口搜索条件不足就放弃全部端口|同上及`models/project_inventory.py`|逐端口记录失败后继续；温度分支真实数值搜索、写回并回读，压力分支单位冲突停止，原压力及辅助输入字节保留；无产品时输出具体缺项诊断而非伪成功|PASS，真实计算/合成边界输入|
|组件已生成却被工程绑定问题丢弃|`models/component_service.py`|保留已通过写出器及回读的组件；工程绑定失败单列，绝不冒称工程已更新。复用原最小配置补全，不猜经纬度、单位和观测值|PASS，四模型故障注入及真实组件验收|
|WRF组件入口落入开发机固定配置|同上CATALOG及TASK_MODES|仅补既有`wps_sst.export_sst`的公共组件绑定；显式当前科学配置可从NetCDF生成WPS中间文件，不需要另一台机器的配置或完整WRF工程|PASS，真实OISST子集。未宣称所有旧WRF入口已迁移|
|一条建议未通过就阻止其他已核验操作|`desktop/discovery.py`、`__main__.py`|第二步保留具体拒绝原因；至少一条已核验操作时允许其执行，未核验建议不能获得执行权限；最终后端仍独立验契约|PASS，Qt处理函数定向测试|
|后续研究/附加步骤异常掩盖已有产品|`desktop/selected_processing.py`、`models/task_dispatch.py`|独立任务分别记录；保留成功产品及失败列表；全部关键条件不足时可下载诊断包，明确无数据产品|PASS，故障注入、打包回读及原数据任务回归|
|转换、工程齐备、运行模型混为同一结果|`conversion_bridge.py`、`final_check.py`、`desktop/issues.py`、`output_review.py`|转换器失败仍保留已准备产品；组件检查不要求完整工程；最终格式、工程、模型运行分别显示。PASS限定到明确的组件哈希/序列化或原生天气检查范围，不代表完整模型兼容|PASS，受影响定向测试|

**自动补全与DS边界。** 复用原有SWAT+最小配置补全：非物理输出标签、从实际坐标和时间覆盖取得的站点元数据/日期、由实际输出生成的索引。既有写出器允许的默认值保持其契约范围，不新增气象值或虚构工程。DS可以沿既有审核入口判断用途、提出跳过非必要建议；公共处置器只在该建议已被判定为可选时消费SKIP_STEP。必需条件即使DS建议跳过仍BLOCK_STEP。证据不足的单位、CRS、区间、变量物理定义、合法训练状态和原生运行关键依赖不能由DS编造补齐。未新增第二套API后端，本轮未做线上DS自动补救调用验收。

**有界实际执行。** 证据集中 `output/unified_continuation_20261123/`，本轮未下载、未覆盖历史数据或结果，没有运行大型研究或五个原生模型求解。`real_v2/cases.json`、各目录result/manifest及下载包保存实际结果：

|入口/模型|已有输入及真实性|实际结果|范围与限制|
|---|---|---|---|
|仅转换/SWAT+|官方土壤有效子集CSV|COMPONENT_EXPORTED，格式/回读记录存在|不要求file.cio，不等于完整流域工程|
|仅转换/CAMx|已标注合成的经纬网排放CSV|COMPONENT_EXPORTED，NetCDF组件|格式功能测试，不作环境精度证据|
|仅转换/CMAQ|真实EDGAR CO巴黎子集NetCDF|COMPONENT_EXPORTED，原有排放写出器执行|完整CCTM工程与模型求解未请求|
|仅转换/CALPUFF|真实NOAA USCRN降水CSV|COMPONENT_EXPORTED，原有降水写出器执行|不要求CALMET/CALPUFF可执行程序，未运行模型|
|仅转换/WRF|真实NOAA OISST三日子集NetCDF及已有明确配置|COMPONENT_EXPORTED，3个`.wps`中间文件及格式报告|这是WPS中间输入，非wrfinput；没有运行geogrid/metgrid/real/wrf|
|仅融合|原Perth站2018年前180日观测＋同期ERA5，固定90/45/45时段|公共`submit_search` PASS，产品回读；NO_STABLE_SELECTION_GAIN、BASELINE_RETAINED|没有稳定增益就保留质量合格基线；未强加多源奖励或改规则|

五个转换案例分别约0.66、0.97、1.09、0.34、0.81秒；仅融合客户端约14.52秒，公共服务记录约14.24秒，实际科学调用6次，原始budget及分项成本见protocol和fusion_result。没有据此宣称速度或精度优势。仅融合请求的前90日观测为合法可用输入，中45日仅选型、后45日仅冻结复核；未把自然缺失当真值，全部年份均已暴露，仍是工程/探索性复核。

**测试与失败保留。** 去重140项新增及受影响定向测试通过，包含五模型公共端口/页面处理函数、科学拒绝、真实父边、状态与既有质量门回归、字段级隔离、温度处理/压力保留、仅组件导出、缺程序不抵消已完成转换、无可执行任务诊断导出、DS不能豁免必要条件及下载包回读。结果分散运行于tests_run3.xml、tests_new.xml、tests_final.xml、tests_bridge2.xml、tests_fields.xml，不重复计数；未执行无关全量回归。

早期测试的三项setup error来自项目旧`.pytest_tmp`权限，随后使用本轮独立basetemp，未删除旧目录。首次功能运行暴露诊断包未集中登记端口缺项和测试误把纯格式说明卡当数据成果，已补齐缺项汇总并检查卡片没有选中、数值指标或已执行标记，失败留存。真实验收real_v1保留WRF旧开发机配置缺失、客户端遗漏显式search_validation标签两项失败；修复公共WRF绑定并明确标注合法验证标签后使用real_v2，不更换数据或评价规则。最后一次字段隔离补充仅影响错误变量分支，35项定向测试另存tests_fields；真实组件与仅融合案例不进入这些变化分支。

另尝试收集旧`test_workbench_execution_messages.py`时失败：它从未修改的旧workbench_page导入当前不存在的fusion_state。该历史页面测试未通过、未删除、未屏蔽，也不计入140项通过；本轮没有恢复已经迁移掉的旧页面API。当前桌面处理函数及公共组件页相关测试已通过。没有人工GUI/浏览器点击全流程或线上DS验收，不拿AppTest/Qt函数测试冒称以上验收。

**完成边界。** 对现有已支持的科学任务，非必要建议、缺完整工程、缺模型程序不再当然阻止独立融合或组件转换；关键缺失在对应来源/变量/端口/转换步骤明确停下，其他可执行工作继续。仍不能承诺任意损坏文件、未知物理含义或缺少必要参考/状态都能产出可信产品；这种情况下保留诊断，不捏造完成。裸WRF输入若没有当前可验收转换契约，尚可能进入旧未迁移路线；本轮验收的是明确本次科学配置的公共组件路径，不能称任意WRF输入均自动支持。不在此轮扩展新格式或自动建立完整模型工程。

修改前快照before、源码差异changes.diff、最终执行源码executed_source、source_fingerprints.json、acceptance.json及输入指纹集中保留。复现真实小案例使用accept_real.py并更换OUT为新的未存在目录，避免覆盖；定向测试使用项目`.venv/Scripts/python.exe -m pytest`与独立basetemp，PYTHONPATH包含src及原existing_air/local_swat集成目录。后续没有自动启动科研实验或同步GitHub。

## 自动补救、估计值标注及五模型公共入口验收（2026-11-24）

**目标及结论。** 上轮实现“某步骤失败不牵连其他步骤”，本轮补上“失败前先寻找可验证的补救依据”。新增公共`core/recovery.py`作为数据准备环节，复用原只读资料访问、DeepSeek传输、时间恢复内核、GridGapRecovery、模型组件写出器和continuation处置记录。当前五模型公共组件路径、仅融合及仅转换均可进入该环节；没有按数据包、站点或年份写判断分支。原始文件只读，补救只作用于内存副本和新产品。不承诺任意格式、任意工程或所有缺失模式均能恢复。

**公共链路及最小接线。**

|位置|本轮补齐的行为|验收证据|
|---|---|---|
|`core/recovery.py`|同文件元数据、已读配套说明及明确的源单位声明形成作用于具体变量的事实；DS引用显式编号，公共检查拒绝越界编号、冲突及无依据新值|新增测试及release两项线上元数据补全|
|`algorithms/automatic_preparation.py`、`models/component_service.py`|五模型共用补救会话；选定输入后才读取和修复，之后仍执行原写出器、质量门、最终检查|offline_release五模型真实数值/格式执行|
|`models/engineering/preparation.py`|工程来源先补元数据；数值补救在角色明确后进入公共搜索，不提前填补可能成为验证基准的主来源|五模型端口回归、真值隔离测试|
|`desktop/discovery.py`、`pair_processing.py`|第二步单位缺项先复用同一补救函数；前置核验与真实处理不再各用一套元数据|前置核验→实际读取的行为测试|
|`desktop/file_agent.py`、`service.py`、`selected_processing.py`、`__main__.py`|沿用现有页面和连接凭据，公共会话跨相应操作复用；不新增页面、算法选择器或客户端补救规则|原按需工具/Qt函数/公共组件页回归|
|`algorithms/platform_search.py`|背景/辅助输入可按合法支持补缺，primary不自动填值；已带补值掩码的primary位置不成为验证真值；参考和观测角色不自动估计|信息隔离、原有统一复检回归及真实仅融合执行|
|`desktop/issues.py`、`output_review.py`|展示来源、字段、补全或工作假设、估计数量与剩余问题；下载包带说明和位置掩码|结果打包回读测试、实际组件ZIP|

科学算法、变异、排序、评价规则及第三方源码未修改，没有新增算法ID或训练模型。空间补缺调用既有`reconstruction.grid_gap_linear`的真实契约与内核；时间补缺调用既有temporal_kernels。没有把平滑滤波器冒充缺失恢复，也没有用预测收益替代真实评价。

**可自动补救的范围。**

1. 元数据缺在字段层、但同文件属性或有作用对象的说明中存在时，可补回单位、时间语义、空间支持、CRS等允许字段。只使用明确的源单位，不把模型期望的目标单位当作原始单位。多来源目录的无归属说明不能向另一来源“捐赠”单位。
2. DS负责审核和解释证据，不能生成观测、流域拓扑、训练状态、单位或坐标的无依据新值。每个补救会话最多一次DS请求、不自动重试，流式响应预算30秒；连接及底层网络超时另受既有传输约束。原按需扫描有其独立预算，不能把这理解为整个操作只调用一次DS。
3. DS失败或未选中某项不等于自动否定文件中的明确、无冲突声明；平台可以按直接证据继续。明确的WORKING_ASSUMPTION_ONLY可作为工作约定使用，页面标注“未证实原始定义”。科学冲突、非法CRS、缺少证据仍不能靠DS意见豁免。
4. 默认时间补救范围固定为每条实体序列缺失比例不超过20%、连续缺口不超过3条；区分记录数与实际时间。瞬时量复用线性恢复；平均/累计量还须明确等长度区间及时间基准，估计的是缺失窗口的值，不拆分累计量。单侧支持仅允许一个已有端点记录的最近邻估计，明确标注外推，不新增时间标签。
5. 默认空间补救只处理支持明确的连续、直角网格，单平面缺失比例不超过20%、连续缺口不超过3格，至少四个原始有效格点。时间记录数和空间格点数分别登记；不是普适科学阈值。分类、方向、未知QA/不确定性等不强行套标量插值；持续全缺测位置不被空间恢复涂成有效域。
6. 原有效数值不覆盖，估计位置以`__recovered`掩码、压缩位置索引及CSV原行号记录；模型原生格式不能增加随意列时，标注保存在配套文件。最终单位、范围、非负、状态依赖及原生格式检查继续有效。

超过上述补救范围不是整项工作退出，而是保留缺失并由原契约决定对应步骤能否交付，其他独立步骤继续。不能确认必要物理含义、缺完整关键变量、依赖或有效支持时仍可能停止该步骤；本轮没有把“任何数据都不退回”作为虚假承诺。

**真实与受控案例。** 证据集中`output/automatic_recovery_20261124/`。使用现有文件，没有下载或改写原包。为验证补救，预先固定删除一个字段单位或遮挡一个值，均明确属于基于真实数据的缺项对照，不能说原始数据本来就有这些缺失。

|入口|数据及缺项|实际结果|结论边界|
|---|---|---|---|
|CMAQ仅转换|已有EDGAR CO巴黎子集，移除变量单位|按同文件/配套证据补回单位，排放组件转换及回读通过|不是完整CCTM运行|
|WRF仅转换|已有NOAA OISST三日子集，移除SST单位|补回Celsius及当前配置明确的CRS，生成WPS中间文件|不是wrfinput生成或WRF求解|
|CALPUFF仅转换|已有NOAA USCRN小时降水，遮挡第6条记录（0起始索引5）|估计1个值并标注，原降水写出器及回读通过|估计不是实测，不声称改善误差|
|SWAT+仅转换|已有Perth日降水前30日，遮挡第6条记录（0起始索引5）|估计1个值，复用最小配置补全，生成pcp/cli|不要求完整流域工程，未运行SWAT+|
|CAMx仅转换|原有明确标注的合成排放格式样例，无需修补|原路径转换通过，无缺项不强行修改|不是环境精度证据|
|CAMx补缺补充|上述合成格式扩展为20小时，固定一个单元一个缺值|真实公共入口按实体分组补1值，原生排放写出及回读通过|仅为功能边界测试，见tests_camx2.xml|
|仅融合|Perth观测＋ERA5，原90/45/45时间划分；背景0起始索引15（第16条记录）遮挡1值|补背景1值，公共搜索、独立复检及产品回读通过；保留基线|不修改评价规则，不强推多源，不证明融合增益|
|空间补缺补充|EDGAR原3×3网格遮挡中心1格；显式地理/WGS84及单元均值工作支持|调用既有空间恢复后原生排放写出通过|工作支持未独立核实原始坐标基准，见release_spatial_v2|

线上矩阵`release`中CMAQ、WRF各一次DeepSeek请求，均COMPLETED，返回usage分别605、538 Tokens；这只是该矩阵，不是整个调试期的总用量。最终代码另在`offline_release`完成五模型及仅融合复核，证明无API时明确文件证据仍可继续，未冒称该轮调用了线上DS。运行源码SHA256随矩阵保存并核对；原文件SHA256、组件文件、掩码和ZIP CRC均复核。没有因为结果负面换区域、换种子或调阈值；这些不是正式算法优越性实验。

**评价隔离与成本。** 自动估计不能增加“已观测真值”的分母。primary/观测/参考/测试资源不被预先补成评价标签；带估计掩码的primary会排除对应位置。工程来源在确定角色之前只补元数据，避免将估计值当作内部留出基准。背景补救发生在合法预测输入侧，原验证、冻结复检和质量规则保持。公共补救耗时、其中DS耗时、说明读取IO及整个作用域墙钟分别记录；搜索和科学调用仍用原计数。补值掩码、记录写包成本属于整体墙钟，不据此宣称效率优势。最早一次非法DS编号的响应未完整保留usage，历史总Token统计不完整，明确保留该缺口。

**测试、失败与版本。** 去重166项新增及受影响定向测试通过，未运行无关全量回归。最终补救专项34项见tests_release.xml；其他通过记录包括tests5/7/final/final_core/isolation/verified/last及CAMx补充。覆盖声明冲突、来源归属、DS越界引用/不可用/预算、未知不伪装为零、微秒/纳秒、区间边界、原始有效值不变、局部与大块缺失、端点估计、分类/方向/QA排除、持续NoData、主来源真值隔离、前置核验与执行共享消费者、位置掩码下载及组件回读。

早期失败全部保留：实际时间差曾按纳秒解释导致us输入不补，已改用Timedelta实际秒数；证据文件名曾包含计时导致页面/工具包不一致，已改科学内容指纹；DS证据列表起初未显式编号，CMAQ越界引用被拒绝后按文件证据完成，已补显式ID及枚举；客户端对pandas只读数组赋值失败，改为副本；合成CAMx时钟漏了原UTC偏移，修复测试构造，未修改科学时区规则；时间边界测试误要求重复units属性，现同时核对实际时间、边界及原始编码值。另保留release_spatial中DS未选明确工作声明后未能完成的记录，经“DS建议/明确声明/硬约束”分离修复后release_spatial_v2完成。不是通过降低原模型检查或增加好种子获得通过。

**仍未验收或不支持。** 未人工点击GUI/浏览器完成本轮全流程，不把Qt/AppTest或公共处理函数称为人工浏览器验收；没有运行五个完整原生模拟程序，也没有补造工程拓扑。自由文本不能形成明确、作用来源可查证的事实时，仍保留UNKNOWN；当前补救不等于理解任意说明书。曲线网格、结构性无效域、未知QA、完全缺少的关键变量或训练状态，需要其已有合法适配或资源，不能凭空补全。源数据、旧协议、原报告和旧负结果保留；本轮不自动同步GitHub或启动正式研究。

修改前快照before、最终源码final_source、source_fingerprints.json、changes.diff、acceptance.json及各版本真实结果集中保存。复现客户端`run_real.py <新目录名>`只提供数据和固定缺项；科学补救、DS、转换、搜索及评价都在平台公共模块，目录存在即停止而非覆盖。使用项目.venv及原src/existing_air/local_swat路径；API密钥继续走现有环境/页面凭据，不写进文件。


## 队友抽样更新同步、自动补救保留与变异后独立验收（2026-11-24，桌面20261124-r2）

**范围与版本。** 实际运行目录为D:\codex\EcoDataEngine，并非仅更新EcoDataEngine-GitHub协作副本。冻结队友ui-redesign提交964778a（完整SHA见本轮plan.json），以共同祖先dae836f三方整合72项变动，保留本地新增功能及全部历史结果。冲突按实际职责合并，没有将旧main直接覆盖运行目录。新增分格式采样、holdout_plan、sampling_risk、recipe_trials、release_gate、独立补证和桌面展示共用队友公共模块；未复制第三方算法或新增数学内核。自动补救core/recovery.py与上一轮交付逐字节相同，保留五模型、仅融合、仅转换各公共调用及估计标注、原始参考隔离。安装队友已声明而本机缺失的ruptures 1.1.10；未更换评分权重、样本或增加搜索预算追求收益。

凭据只更新Windows用户环境DEEPSEEK_API_KEY，未写入源码、请求文件或报告。此前HTTP402属于本机旧凭据，不能作为用户新凭据无余额的证据。替换后真实线上分类、按需工具、资料判断、算法推荐及桌面最终复核完成，SIMULATED不能代替这些记录。账户余额期间变化，不能用余额差推算本轮费用；较早客户端序列化失败前的一次推荐未保存完整usage，不声称历史Token/费用统计完整。

### 公共连接及安全边界

|环节|本轮落实|证据/边界|
|---|---|---|
|识别到推荐|队友轻量类型判断→按需文件工具→按格式隔离样本→公共recommendations推荐和本地recipe_trials|saguling_live_v2/inspection.json为真实线上记录；源包SHA核对，未全包发送给DS|
|推荐验收到变异|保留with_recommendations、Registry契约、真实Objective；不把DS解释当误差|perth_live/recommendation.json、search/station-1/search/recommendation_validation.json，真实推荐编译PASS并进入已有数值路径|
|提前划开两次验收|holdout_plan在推荐回检前从尚未暴露的时间保留区分出recipe_review_units和mutation_confirmation_times|tests/test_mutation_final_acceptance.py行为测试，原locked_holdout保留用于外发重叠检查；单个留出单位明确INSUFFICIENT|
|搜索消费保留区|两个网格适配器将公共预留计划传入spatial_fusion_task；指定确认日期排除出search_times，禁止与推荐回检日期重叠|真实数值测试检查实际执行trace；不是只记录字段。减尺度代理路径也在冻结后重新执行确认日期，不能沿用搜索期缓存预测冒充新日期|
|最终交付|blocked_confirmation统一复用原评分和质量结论，新增冻结图指纹/最终图身份回执；paired_station在自身最终回检后刷新回执，桌面展示交付图而非未确认赢家|post_mutation_acceptance.json；缺冻结证据不标确认，未通过增强则保留原关卡允许的基线或已验收产品，另保留探索候选|
|补救|保留作用来源可核实的元数据补齐、小缺口估计及掩码，参考/测试标签不补造|real_components五模型＋仅融合矩阵；自动补救34项专项已包含在本轮定向去重记录|
|实际页面|快捷方式指向本运行目录；窗口标题明确20261124-r2；保留仅转换和仅数据处理|desktop_launch.json、desktop_acceptance.json；通过Windows真实界面操作，不把Qt单元测试称为手工验收|

固定方案的超几何抽检仍使用队友sampling_risk与既有公共质量分数。默认零缺陷样本方案针对当前**留出抽样总体的单位**；大总体5%检出概率95%约需59单位，不能转述为95%概率保证全域像元正确。变异后的最终复核沿用blocked_confirmation分块误差门，**不是**把推荐前抽检结论直接继承给新图，也没有声称该分块门具备相同的95%概率保证。不具备可分割时间保留区的路线保留其既有任务型独立参考协议或明确证据不足，不伪造新保留集。全量材料用于推荐后不能再充当同次独立确认材料。

### 真实执行及负结果

1. **Perth站点＋ERA5：链路PASS，增强效果FAIL。** 使用既有ASN00009021资料中2018年固定180个真实日累计，沿用其24h截止当地09:00（UTC01:00）的既有定义，前90拟合、中45选图、后45仅复核。DS仅获拟合期6组真实可见配对和必要科学信息，不获后段标签。通过公共recommendations调用线上DS，推荐的既有station.bias.holdout经真实契约编译为prior-series订正，并被公共paired_station/field_mutation实际评价和变异。固定seed11、80/120/256及30秒原有预算；原数据哈希保留。最终候选后45日RMSE 5.4136818456 mm，原背景4.2417763575 mm；原分块复核CONFIRMATION_FAILED，交付BASELINE_RETAINED。selected graph和原背景产物均保存，不用后45日重选图或调参数。独立性仅指本次声明的时间/用途隔离；年份和资料此前已暴露，仍属探索性工程验收。该条是站点约束背景订正，不冒称验证了任意多源加权融合精度。
2. **Saguling原613,367,712字节包：线上识别/推荐PASS，融合交付未通过。** 新凭据完成真实分类、按需读取、资料判断和推荐。初次客户端在保存含tuple键的gated字典时失败（不是科学算法失败），保留失败日志；随后按已保存真实工具调用重建本地读取，复用本次真实inspection，不重花整轮识别API。新推荐仅再请求一次，未换数据/阈值/好种子。融合方案recipe_numeric_rejected，0项放行，**NO_PRODUCT**；源包SHA仍为d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f。该包共同日期少、预留不足两个单位，不能同时证明推荐验收与变异最终验收。saguling_resume_v3保留被拒指标、理由与无产品回执；其ZIP只有记录不能冒称成功融合。未因功能目标强行放行或为本包写补丁。
3. **保留五模型组件路径：PASS（不等于五模型运行）。** 复用上一轮run_real.py公共客户端，在本轮real_components目录重新执行CMAQ EDGAR、WRF NOAA SST、CALPUFF NOAA小时降水、SWAT+ Perth降水和CAMx已有格式样例；前四基于真实资料，CAMx明确为既有合成功能fixture。原控制缺项仍能补救、标注并经过原生写出/回读；仅融合背景一处缺失补救、数值搜索、复核和产品回读PASS，默认保留基线。没有启动五个完整模拟程序。
4. **最终r2真实桌面：PASS。** 从桌面模拟技术引擎快捷方式启动，保留仅转换开关。先测试原WRF daily目录：因没有ecodata.model-task.v1且公共自动识别未证明唯一组件任务而明确拒绝；不能把已有普通config.json自动当完整任务声明。随后将同一真实NetCDF原字节复制到wrf_gui_input，并用已有科学config生成受支持的model-task.v1（指定目标组件，不指定算法赢家），未改原数据或为包增加解析分支。通过真实界面读取→确认→公共转换→最终DS复核→下载，生成3个WPS intermediate v5 SST文件；NetCDF→WPS格式、原生格式PASS、工程NOT_REQUESTED、模型试读NOT_RUN分开显示。自动补救按源配置把sst.crs补为EPSG:4326且在结果标注，下载包包含补救回执。最终wrf-desktop-export-r2.zip CRC通过，逐文件SHA和格式映射核对。初版组件格式说明只读工程changes而未读output_inventory，导致已产出WPS却显示格式未知；已在公共output_review补齐单组件来源、经过验证的输出清单及哈希消费，未以ZIP后缀猜数据格式。

### 测试、成本、历史保护与复现

定向用例按文件/class/name去重**242项PASS**，以每个用例最后一次受影响执行为准（directed_test_summary.json），不是全仓库一次全量测试，也不把重复轮次累计为独立证据。涵盖抽样总体/隔离、声明冲突、资料外发、冻结图不能继承父图验收、未知资源、减尺度最终日期真实执行、分支状态、估计值不成为真值、产品格式和GUI交互。保留全部早期失败：默认.pytest_tmp权限错误改为本轮新basetemp；缺ruptures按原依赖安装；Windows默认GBK读取UTF-8测试结果改为显式UTF-8；日志动画和抽屉布局并发导致差1像素，改为最新范围直接跟随；自写新测试误把execute无返回值当结果，改读取真实rows而非更改执行器。未删除失败测试、屏蔽异常或放宽数值门。

证据目录output/teammate_sampling_integration_20261124_214021包括before、plan/merged_files、final_source、source_fingerprints、各轮JUnit和日志、real_components、saguling_live_v2（完整真实线上识别）、saguling_resume_v3（负结果）、perth_live（真实DS→变异→冻结复检）、两个版本桌面下载包、desktop_acceptance及desktop_launch。原包/原文件未修改；旧协议与旧结论不覆盖。前期旧凭据HTTP402及客户端失败完整保留，不解释成新凭据不可用。执行成本见各API receipt、progress日志、total_cost、paired_protocol及预算；无完整历史usage的轮次单列，不伪称总费用优势。

复现环境为本项目.venv、src及integrations/existing_air与local_swat。定向测试使用独立--basetemp，避免复用被锁旧pytest目录；客户端脚本在证据目录，仅提供真实输入、固定配置和公共函数调用，不维护实验专用算法匹配或打分。run_paired_live.py对应线上推荐和固定180日验证；resume_live.py明确复用本次inspection，不能称重新做了线上识别；run_real.py来自前轮公共验收客户端。运行需使用新的输出目录，保留本轮不可变记录。启动仍用桌面快捷方式或launch_platform.ps1，密钥从Windows用户环境加载，不要求用户再次粘贴。当前窗口最终版本20261124-r2。

**结论。** 队友更新已落到实际运行平台；自动补救、仅融合/仅转换及变异后冻结数值复核保留且有真实执行证据。当前可交付的是工程链路和安全拒绝/基线保留，不是所有数据包必能融合或精度必提高。Saguling融合本轮未通过；完整五模型模拟、全任务通用统计置信保证和创新优越性实验未执行。本轮不自动推送GitHub、不启动正式大规模研究。


## 推荐评分失败后继续合法候选变异：公共衔接修复与原包验收（2026-11-25）

**问题与改动。** 原DS方案的“科学适用性”与“数值收益”不是同一判断。原Saguling推荐的保守重网格→缺失回退→等权融合已通过字段/单位/时间契约，但推荐抽检代理参考上RMSE23.6957798151、单源基线9.89906141367，按原标准拒绝正确；此前把这一次方案拒绝等同整项任务停止，没有继续公共变异。该分数只证明此方案在此产品重建代理上不合格，不证明所有融合无价值，也不是独立雨量站精度。

recommendations.review_holdouts现调用公共continue_rejected_search：只有明确recipe_numeric_rejected、所有失败项均recipe_score_below_single_source且有评分的情况，才另列SEARCH_ONLY选项。原失败推荐、指标、外发记录不改；契约/覆盖失败、UNKNOWN、缺参考不冒充可搜索。release_gate.allowed仍严格表示原方案放行；selectable表示可提交搜索，require默认仍严格，仅搜索入口显式allow_search=True。页面使用同一权限，并明确显示“原推荐未通过，继续合法候选变异（待最终复核）”，不会默认勾选或标成DS已通过。selected_processing和independent_alignment共用既有platform_search/local_search，失败推荐不再作为initial_algorithms或recommended_recipes的种子/偏好。搜索失败不可降级执行原拒绝方案；最终质量门、分块复核、基线保留规则和固定预算不变，未新增科学算法或包名分支。

**支持边界。** 本次搜索衔接覆盖已有independent_grid_search公共双源日累计网格路径，不依赖Saguling名称/日期/网格。其他适配器尚未证明能够跳过失败配方直接变异，因此保持原关卡；特别是不将indexed_table_recipe等直接组合路径冒称已完成同样衔接。非数值收益失败仍需其具体契约/资源，不通过降低评分放行。

**原包真实结果。** output/rejected_recipe_search_20261125/real_original为新证据目录，保留旧NO_PRODUCT及所有历史结果。原613367712字节ZIP SHA256仍为d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f。复用上一轮真实线上inspection及DS推荐（明确记录来源），重新读取原包、重新抽检拒绝、通过新公共搜索选项执行，没有新增API请求或伪造推荐。本轮37.00148秒，实际评价18图（其中11图为mutation阶段），100次提议后预算停止；真实科学节点search76次、deploy1次、失败调用0次。缓存/提议/执行分别见counts、transitions和total_cost；当前平台使用composition_optimizers.random_graph_search公共变异后端，不将其称为新官方GOLEM优越性实验。

预算内最佳及交付仍为NASA_POWER单源，验证代理RMSE6.1953983729 mm、MAE4.3627947048、Bias0.3719997934，78个共同评价样本，覆盖1.0、失败率0。该78样本多日期/区域重建队列与原推荐8点评分队列不同，禁止直接用两组数字宣称修复提升。最终post_mutation_acceptance=BASELINE_RETAINED，recommended=false，numerical_review=EVIDENCE_INSUFFICIENT；只有一个未暴露日期单位，不能同时用于推荐和变异最终独立确认，frozen_graph_verified=false，不改称通过。结果证明流程不再在推荐失败处终止，并真实搜索和生成产品；不证明多源收益或全局最优。

产品NetCDF回读：precipitation，mm，time3×lat7×lon5及bounds2；desktop-results.zip CRC PASS，包含实际产品和评价记录。原始数据不动，未运行SWAT+完整模拟器或额外模型写回；本轮验收走仅数据处理公共入口。未再次进行人工桌面点击全流程，页面控件通过真实Qt组件定向测试，不冒称人工浏览器/桌面验收。

**定向验证。** tests1.xml/tests2.xml/tests3.xml按classname/name去重54项PASS，包含8项新增案例：数值拒绝可搜索而不可直接执行、原失败记录不变、幂等、未评分/覆盖/待确认/未支持路线拒绝、已有合法方案无需额外选项、伪造不完整admission拒绝，以及实际Qt失败卡不可选/搜索卡可选且不默认勾选。受影响release、candidate_choice、recommendation_search和mutation_final_acceptance回归通过；未跑无关全量回归。旧pytest_cache权限warning保留，后续本轮使用-p no:cacheprovider及独立basetemp；第三方弃用warning未屏蔽。acceptance.json记录真实断言、产品回读、原包指纹、计数和限制；before保存修改前源码/报告，final_source及source_fingerprints.json保存交付源码。复现命令为项目.venv运行output/rejected_recipe_search_20261125/run_original.py（须另设新目录避免覆盖）；客户端只提供原包、已记录真实推荐和公共调用，不含打分或匹配规则。

**验收结论。** 推荐失败→其他合法候选变异→真实评价→产品：PASS；评分标准及原失败记录保护：PASS；Qt搜索选项与直接执行权限隔离：PASS；融合改善及独立确认：证据不足/未获增强推荐；其他尚无失败后变异契约的适配器：未接入本次继续机制。未自动推送GitHub、未重跑大型历史实验。


## 推荐拒绝后的融合入口统一衔接与跨入口验证（2026-11-25，all-fusion-rejected-search-v2）

**为何上一轮只有日降水。** 当时只有independent_grid_search证明了可不执行被拒配方、直接运行其他合法图；站点路径仍携带失败推荐，windowed路径可能退回失败模板，indexed表格路径则只有固定配方。上一章的范围确有限制，本轮扩展公共适配与执行，不把旧单例当成已完成泛化。

**公共实现。** recipe_trials.SEARCH_CONTINUATION_ROUTES/continuation_steps登记四个有真实搜索路径的适配器；recommendations.continue_rejected_search复用同一数值拒绝判据，按既有配方保留合法准备操作、删除被拒融合/偏差订正种子。release_gate.selectable和require(allow_search=True)仍区分“可提交搜索”与“配方可执行”，allowed默认门不变。selected_processing统一分派，各客户端不另写算法排序/权重/循环。station_grid与indexed_table_recipe共用local_search.run_paired→platform_search→paired_station_search→既有field_mutation；普通网格复用local_search.submit→platform_search→spatial_fusion_task。保留原随机变异预算/科学内核/最终质量和分块标准，没有调权重使融合赢。

|入口|本轮改动|实际证据|结论/限制|
|---|---|---|---|
|日累计双源网格 independent_grid_search|共享原继续机制和统一权限，不重跑无关旧搜索|上一轮rejected_recipe_search_20261125/acceptance.json；本轮公共权限和质量回归|保留PASS，未声称新重复实验|
|站点＋网格 station_grid|失败配方不再作为推荐种子；共用paired搜索和导出；缺独立预留禁止增强推荐|station_v4/CAMx_NOAA_station，原CAMx原生气象与NOAA站点|失败配方→真实新变异1次→NetCDF/回读，PASS；最终BASELINE_RETAINED|
|双源表格/一维时间 indexed_table_recipe|补齐已有成对订正内核的公共图绑定，与站点共用run_paired；没有坐标则使用真实time一维轴，不填造经纬度|real_v3/CAMx_NOAA_table，两份真实配对温度表，来自同一CAMx/NOAA实际读取|真实新变异6次、产品及ZIP回读PASS；不是新独立数据集，最终BASELINE_RETAINED|
|普通时空网格 windowed_grid_recipe|合法准备后直接搜索；取消失败时退回原失败模板；消费公共关卡交付的基线；保留源物理定义，单元均值/通量强制已有保守对齐契约|grid_v5/NCEP_grid_derived_control，真实NCEP原场及其明确粗化表示|11图/100提议，其中4张新变异；产品/回读PASS。两表示同源，属于工程受控验证，不是独立多源收益证据|

原生工程、确定性配对、不确定性加权等未产生recipe_numeric_rejected的路线，不在缺参考/不支持时伪造数值放行；直接调用platform_search的入口本来已在公共搜索后执行最终复核，无需复制一个“失败继续”循环。普通窗口变异仍依赖已声明瞬时量、EPSG:4326和至少三个真实完整共同时间；静态/其他投影/其他统计语义无匹配数值适配时只标对应步骤条件不足，不能把失败配方偷偷交付。累计量继续走已有日累计适配，不把所有时序强行解释为瞬时量。单位、物理定义、覆盖、状态或UNKNOWN失败仍拒绝，不是“所有失败都允许继续”。

**推荐为何会过不了评分。** DS推荐来自登记配方和有限可见资料，能说明适用性，尚不具备可靠的收益预测。Saguling既有配方的等权融合不等于已估计双方误差；站点均值偏差也不保证适用另一个变化时段。真实CAMx/NOAA配方抽检RMSE0.9456565238 K，背景0.7768292245 K（5个样本），因此实际数值不合格；不能以“能运行”替代不劣于基线。表格抽检的补缺RMSE1.4096457514 K，基线0.3369319045 K；该组融合RMSE1.0781015225、基线1.1126764824，说明不同操作有不同结果，不能把整套失败全归因于融合。没有调参救这几个数字，也不将这些小样本视为正式精度结论。

此外确实发现平台接口缺口：站点分组试跑的原始表未经过正式读取已有的NOAA复合字段解码；表内无关重复列/相同REPORT_TYPE重复列又使试跑失败；窗口图绑定遗漏原始physical_definition。这些不是数据价值不足。现在复用生产noaa_global_hourly.add_variables解码；只取关键输入列，完全相同重复列核对后去重并记matching_audit，冲突关键列明确拒绝，不平均冲突值；windowed.Grid保留实际源定义，定义缺失/冲突不捏造。变异候选仍可能真实执行失败，日志和计数保留，不筛选掉失败种子。

**最终验收不因继续搜索放松。** paired公共服务接收推荐隔离状态，缺独立保留区明确EVIDENCE_INSUFFICIENT并保留质量基线。表格优先排除推荐用过的分组；确实只有同一组时允许作为开发资料继续探索，明确记录recommendation_groups_reused_for_development，不将其重新命名为独立测试，最终不能据此增强推荐。新图仍训练前段、选择中段、复核后段；已暴露历史资料不算新盲测。即使某个数值候选更好，回执不满足也不确认。run_paired和windowed均要求继续任务的交付要么经过最终验收，要么是关卡保留的质量基线；没有这两者则明确失败，绝不退回被拒配方。没有新独立雨量站/精细真值就不宣称降尺度精度或融合提高。

**页面与成本。** 页面的搜索选项沿用同一search_admission，无额外匹配策略。已有可执行继续选项时，send_next_automatic_fallback不再自动重复向DS发全量扰动资料；问题清单准确提示可选择同任务合法候选变异，原推荐失败仍保留。当前有Qt控件行为测试，未人工再点完整桌面/浏览器。此次新增验证没有API请求，不声称重新生成了线上推荐；站点配方来自既有冻结任务，表格/网格使用注册的固定功能验收配方。各实际搜索成本、缓存、失败、停止见各summary/total_cost和mutation_trace；本轮未计时完整历史整理全过程，不声称总成本优势。预算保持原公共默认：站点/表格80/120/256/30秒，窗口130/180/100/45秒，不增加好种子。模型完整运行及原生工程写回未验收，本轮交付为融合处理产品。

**测试、失败与保护。** 82项新增及受影响用例按classname/name、最后受影响执行去重PASS（test_summary.json），不是全仓库全量回归。覆盖四入口同一权限、必需准备与失败种子分离、UNKNOWN/无评分/覆盖失败不放行、关闭冗余自动推荐、无坐标序列不造坐标、后段真值不能改变拟合状态、真实新变异、独立性不足不得因低误差晋级、真实父边、最终回读、冲突重复字段拒绝及失败模板不回退。保留早期失败：real_v1控制网格裁剪后仅7行不能2×精确粗化，按实际维度修正12行；real_v2漏源定义、real_v3双线性不属于单元均值的合法候选域，最终grid_v5使用已有保守契约并显式拒绝不合法组合；站点codec及相同重复列修复见station_v4。tests4新增fixture漏bytes元数据，修正fixture后tests5/7通过；未删失败测试、未改数值门。shell创建v3客户端的一次引号语法错误未启动科学计算，后改用here-string执行。

证据目录output/fusion_continuation_all_20261125包含before、protocol、全部中间失败/负结果、station_v4/real_v3/grid_v5实际产品、acceptance.json、route_matrix.json、JUnit、最终源码与指纹及changes.diff。原CAMx ZIP的路径/大小/mtime清单与旧冻结签名一致，原NCEP只读；新表和粗化控制文件仅写在本轮输出目录，未覆盖原数据、旧实验或报告历史。后续仅更新protocol/source说明的元数据不改变已验证数值路径，定向回归通过；executed source版本与交付diff留存。没有自动推送GitHub、下载新数据或启动大型研究。

**结论。** 本轮不再限于Saguling：四个产生完整配方数值评分的融合适配器共用拒绝后合法搜索机制，三个新增入口有真实数值执行及产品证据。精度增益未得到独立确认，三个新增验收案例最终均保留基线。推荐是待验证候选而非效果承诺；数值拒绝保留并继续寻找，而不是把拒绝当成功或把整项任务停止。


**最终源码统一复跑补记。** 所有上述代码冻结后，使用run_final.py对同三个小案例做最后一次受影响端到端回归，最终记录统一为real_final/CAMx_NOAA_station、real_final/CAMx_NOAA_table、real_final/NCEP_grid_derived_control，均PASS；前三轮中间路径仍保留为排错证据。实际新变异分别1、6、4次，网格总评价11图/100提议；全部最终保留基线。acceptance.json已指向该统一最终运行及NetCDF/ZIP回读断言；input_integrity.json核对原包冻结签名。交付守卫和未定位序列说明均已包含在此次执行源码中。executed_source_fingerprints.json与最终源码对应，不以中间样例替代最终版本验收。未改预算、种子、指标或通过门槛。


## 融合验收V2：总体收益、局部风险与证据充分性分离（2026-11-25）

**范围。** 依据用户授权修改公共ecodata.evaluation.blocked_confirmation，替换“任一块RMSE或MAE退化即拒绝”的默认验收。数据解析、算法数学内核、候选池、搜索目标、原固定预算、数据遮挡和原始文件不改；所有旧输出保留。本轮是评价工程改进和有界行为验证，不是证明新融合算法更优。请求中既有blocked_confirmation_v1名称继续兼容原时间划分协议；实际验收receipt.version明确为blocked_confirmation_v2_aggregate_risk。原严格判据保留review_legacy及policy_version=blocked_confirmation_v1，用于明确回放；不会将历史JSON改成新结论。

**公共规则。** 候选与基线必须仍是同一参考、同一队列、同一分母，保留原时间/空间隔离检查，质量失败不放行。总体口径仍为原等权分块RMSE，不另加多源奖励。总体初筛不改善、最终总体退化或已声明局部风险超限，仍保留基线。总体两阶段改善但不确定性、局部容忍依据或独立预留不足，返回TRIAL_AVAILABLE（可试用、收益/风险未确认），而非把证据不足写成效果失败。仅证据和风险满足时返回当前窗口内推荐，不保证未来效果。

compare记录aggregate_gain、配对分块不确定性、退化块比例、最差RMSE/MAE退化和明确阈值违例。采用已有SciPy bootstrap(paired=True)，对固定分组的损失差和对应计数共同重采样，保持原等权目标；4999次、seed20261125、95% percentile区间在protocol.json中提前固定。空间评价把同一区域不同日期归为同一重采样单位；时空评价将同日各区域共同重采样；时间评价使用时间窗口，不能把像元或同日多个区域冒充独立重复。记录留一组范围用于诊断，并明确区间依赖分块假设，不是经校准的未来覆盖保证。区域/日期之间可能相关、小组数有限，故不宣称统计独立或普适显著性。

**阈值依据与边界。** 没有填“允许10%退化”之类默认数字。局部严重损失的review_risk_limits须有匹配物理单位、非负限值和可审计basis；来源是明确任务约束或冻结开发协议，不能由最终误差反向挑选。未声明容忍依据且确有局部退化时保持UNASSESSED_TOLERANCE，只可试用，不声称安全。若所有观察块都改善，则记录NO_OBSERVED_DEGRADATION，也不推断未见区域安全。测试中的0.2 K是明确标注的合成任务边界，不是现实模型默认阈值。原64×机器精度仅防浮点误差，原3块/每块2个时间的条件仅为工程最低支持；95%区间不等于允许5%数据不合格。未建立仪器/模型层面的通用容忍标定，未声称已做到。依据见SciPy官方文档 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html 及OTexts训练/测试时序评估 https://otexts.com/fpp3/toolbox.html#time-series-cross-validation；这些资料支撑实现和隔离方式，不替平台证明环境效果。

**各入口实际消费者。** review_spatial（站点空间、多网格、日累计网格等）、review_field（多源/辅助/缺失处理）、finalize显式参考和paired_station_search均调用同一review；风险要求从公共request转发，不在客户端另写分数。desktop的local_search、independent_alignment、windowed_grid_recipe转发同一review_risk_limits。原始观测仍不进入预测/提议；bootstrap使用已有独立评价结果，不反向进入搜索。apply_recommendation_isolation只影响证据等级，不再把真实“后段变差”覆盖成笼统“样本不足”。没有真值的工程指标仍为ENGINEERING_ONLY，缺评价资源仍未评价，不用结构分数冒充精度。

默认交付仍可保留质量基线，候选另存且标明可试用；paired/table沿用候选下载，两个网格适配器在TRIAL_AVAILABLE时将真实trial-candidate.nc加入结果包。页面输出已增加可试用/局部风险超限说明，公共解释只含汇总，不外发参考块ID。质量、原生格式、完整模型运行和环境精度依然分开，不因可下载就称推荐完成。

### 现有案例：只重算验收，不重跑旧搜索

|案例|旧默认判断|新判断|理由|
|---|---|---|---|
|CAMx/NOAA站点|EVIDENCE_INSUFFICIENT|CONFIRMATION_FAILED，同时证据不足|选型RMSE0.481→0.140 K，后段0.923→1.215 K；真正退化仍拒绝，不被模糊证据状态掩盖|
|同批数据的表格入口|EVIDENCE_INSUFFICIENT|CONFIRMATION_FAILED，同时证据不足|同一批实际值的不同入口，不当成独立数据集；后段退化不因改规则通过|
|NCEP同源粗化控制|NO_STABLE_SELECTION_GAIN|TRIAL_AVAILABLE|总体初筛/复核改善，但7个区域组的初筛条件区间跨零；最终12块中1块局部退化且无现实容忍阈值，不再一票写成无效，也不强行确认|

saved_comparison/comparison.json保存旧/新同一数值结果的回放。对NCEP当前版本又通过selected_processing做一次有界真实端到端：11图、100提议、4张新变异，输出仍默认基线，但products.zip同时包含真正trial-candidate.nc，NetCDF及ZIP CRC通过（existing_entry/receipt.json）。此案例来自同一真实NCEP场的不同表示，仍仅作功能控制，不是独立多源增益证据。

### 冻结后资料验证：西澳2022年三个站点＋ERA5

protocol.json及source_freeze.json在读取这些年内观测数值前固定。使用现有ASN00003003、ASN00009021、ASN00012038的2022全年真实逐日资料，沿用原metadata中前24小时截至当地09:00＝UTC01:00的定义，保留明确时间边界，不解释成瞬时量。只排除原始QA无效或双源无有效值的记录，不能补造评价真值。相同公共paired入口，前半拟合、中四分之一选型、末段仅冻结后一次复核；每站seed11，80/120/256/30秒原预算，不因失败补种子、扩预算或调整规则。资料与以往集合相同、年份公开且项目可能历史读取过，不称全新盲测；本轮2022值未参与规则制定，源码前后指纹一致。

|站点|真实有效日对|冻结后候选/背景RMSE（mm）|本轮结果|
|---|---:|---|---|
|ASN00003003|365|3.4999844720 / 3.4999844720|SEARCH_INCOMPLETE；188次尝试但0张真正新变异，资源预算停止，保留源；不冒称搜索完成或融合无效|
|ASN00009021|365|1.9570190032 / 1.9300540655|CONFIRMATION_FAILED；后段差约1.40%，保留基线|
|ASN00012038|361（4日QA/配对排除）|1.5730435879 / 1.7509181265|当前窗口推荐；后段改善约10.16%，原严格规则对此候选也通过，不能将收益归因于改了评分|

前一个搜索确实完成过初始候选执行，但未完成真正新变异，因此按原完成标准标PARTIAL_OR_BLOCKED，未为使表格好看重试。最后两站有真实变异和真实独立评价侧调用。三站共用气象/区域背景可能相关，不当成三次独立科研重复；95%条件区间不是这三站之外的未来成功概率。此批结果说明新规则没有全部放行，不能证明算法库或融合对所有数据有效。

**验证记录与成本。** 最终tests7.xml一次运行76项新增/受影响定向测试全PASS，不跑无关全量。包括总体优但局部轻微退化可在明确风险限内通过、无风险依据可试用、总体差拒绝、超限拒绝、单位/依据错误拒绝、分组重复不虚增样本、旧规则回放、原始/预测/独立参考隔离、真实父边、质量门、缓存/预算、Qt页面/工具包及产品回读。保留早期失败：旧断言仍期待一票否决/不足一律失败，按新需求更新并增加旧版本回放；新增合成fixture缺预测units，补齐测试输入，不放松生产检查；np.bool_不能直接JSON序列化，修正为原生bool；相关通过记录见tests3/6/7，失败XML留存。只有最初测试失败前的日志不能代表最终源码。

证据集中于output/fusion_review_v2_20261125：before、protocol及SHA、frozen_source/source_freeze、method_sources、saved_comparison、existing_entry、unseen_2022/results、全部搜索轨迹/冻结图/真实产品、source_freeze_verified、acceptance及JUnit。没有新数据下载、API调用、算法扩库、重复北京全量或949图实验；原始资料指纹回核未变，旧结果不覆盖。每站成本及停止/失败记在各search/summary和total_cost，results.json记录含准备和评价的客户端墙钟时间；配对重采样成本包含在流程时间内，但未单独测量其占比，不声称总成本降低。人工浏览器/桌面整轮点击、完整原生模型运行未验收；Qt控件和真实公共处理入口不是人工浏览器验收。

**交付结论。** 判断规则已由逐块零退化转为总体收益＋局部风险＋证据分层；可以产生“推荐、可试用未确认、保留基线”三类结论，新增类别已能下载真实候选。并非自动更换算法使精度提高；实际退化仍不推荐、缺局部容忍依据不编造阈值、搜索未完成如实保留。若后续制定现实局部容忍阈值，应先由任务/开发资料确定并在新资料前冻结，不能拿本轮2022复核反复调阈值。


## Saguling原包当前版本逻辑一致性检查（2026-11-25）

使用613367712字节原ZIP，复用同一SHA绑定的历史真实DS识别/推荐，不声称新做线上识别；公共selected_processing实际重跑数值，不改阈值、种子、算法或原始数据。新目录output/saguling_consistency_20261125，未覆盖旧结果。

数值链PASS：原推荐仍recipe_numeric_rejected，另有SEARCH_ONLY合法搜索选项，18张实际评价图/100提议，真实搜索节点调用76次；最终及交付均SINGLE_SOURCE、BASELINE_RETAINED，代理RMSE6.1953983729 mm，78个共同样本，独立复检EVIDENCE_INSUFFICIENT、recommended=false。产品为2018-02-03至05三日、7×5网格、mm，非负/有限值、NetCDF回读及结果ZIP CRC通过。此入口实际是NetCDF→NetCDF的对齐/日累计处理，不是已转换为SWAT+输入。页面结果对象显示单源和未确认提升；未进行本轮人工界面点击。数值客户端总耗时74.1487秒，非纯搜索耗时。

但不能判整个平台完全自洽。底层independent_alignment.original_report仍用fusion_executed=task_id in searches，把运行过融合搜索误记成实际融合；真实交付单源却写fusion_executed=true。该标记不是本次数值选型依据，不能据此声称已融合。见numerical/result.json和acceptance.json。本轮检查未修改源码或掩盖旧状态。

另通过service.run_original_workbench对同一原包测试SWAT+工程入口。公共机制自动复用同SHA历史UTC日界工作假设，明确原模型日界未证实，未将假设写成事实。该工程入口实际预算objective/total1800、proposal400000、wall3600秒，不是页面小规模130/180/100/45秒。解包后确实进入实际搜索；为遵守有界检查而停止长任务，记录STOPPED_NOT_ACCEPTED。原生写回、最终格式关卡和模型实际读入均未验收，不沿用旧UTC联调结果冒充当前成功。两个入口任务支持/预算不同，分数不能混比，不能说前一入口选图自动写回后一入口。保存部分搜索请求/轨迹及停止记录，不称完整工程产品。

完整工程展开临时文件约5.25GB，并产生约885MB重打包副本。尝试仅删除本轮archive_intake/expanded_inputs.zip，先确认绝对路径且限定在本轮输出目录，但自动审批两次拒绝，仅返回blocked by policy、无更具体原因；未进一步换工具删除，全部暂存，释放0字节，具体大小见cleanup.json。未将此情况写成自动清理成功。原ZIP最终SHA仍d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f，源码和旧结果保留。

结论：推荐→评分拒绝→继续变异→保留基线的数值逻辑可复现；融合状态标记和工程入口预算仍有缺口，总体PARTIAL_CONSISTENCY_NOT_FULL_PASS。没有把无提升说成融合成功、未完成写回或模型运行标PASS。本轮无源码修改、无无关全量测试；后续修复应落在公共状态和预算机制。


## 平台临时副本自动清理与DS推荐原因核查（2026-11-25）

**本轮实际范围。** 用户明确要求清理前次测试副本，并将任务后清理变成平台能力。原6,132,575,206字节archive_intake/expanded_inputs.zip再次以已核实的绝对路径提交删除，但自动审批仍拒绝，仅返回blocked by policy，没有更具体原因；删除未执行，原副本暂存，未伪报空间释放。未换工具或让新清理器认领这批未登记历史目录来绕过拒绝。该外部阻塞不等于Windows或平台文件系统已损坏。

**公共清理实现。** models.archive_inputs新增normalized_archive_scope，复用原normalize_archives数学/格式读取器，记录本次进入前已存在的路径及原始输入。只清理此调用新建的archive_intake和expanded_inputs.zip，保护原始输入、既有文件、符号链接和被结果引用的路径；不扫描历史output，不删除最终产品、日志、manifest、拟合状态或必要的复现预测。try/finally覆盖正常完成、异常和能够传播BaseException的协作取消。删除失败单列PARTIAL/具体原因，不掩盖原处理异常；保留temporary_cleanup.json与ZIP内回执。

automatic_preparation._prepare_with_context将输入归一化、处理及封包纳入该生命周期，在最终原生检查前释放中间副本；desktop.service仅转换分支也使用同一scope。五模型正常准备、仅融合/仅转换走已有公共调度，是否创建解包副本由实际输入决定，不按包名特判。底层normalize_archives仍保留原API、由低层独立调用者管理生命周期；页面按需文件工具原先已有TemporaryDirectory，不重写。用户下载产品仍保留，不能把需要下载的成品当垃圾清掉。

**限制。** 强制杀进程或断电时Python finally无法保证执行；本轮不宣称已具备崩溃后历史目录安全回收。前次副本正来自被强制停止的工程测试，不属于新调用所有权。桌面原有关闭/新任务清理整项TemporaryDirectory的机制保留，没有新增取消按钮或重写调度器，也不保证回收所有NumPy进程内存；本轮解决的是磁盘临时解包副本。用户需要重启运行中的桌面进程以加载新源码。

**实际验证。** 48项新增及受影响定向测试PASS，tests1/2/3.xml按classname/name去重；涵盖成功、异常、协作取消、归一化半途失败、预存在目录和原始输入保护、结果引用保护、清理权限失败不替换原异常、嵌套容器及五模型公共契约回归。小型合成压缩包只作功能边界；没有使用它们证明环境精度。另将既有真实NOAA SST的原字节及已有合法WPS任务配置封装为嵌套容器，通过prepare_uploads真实执行：输出3个WPS intermediate v5文件，格式PASS、工程NOT_REQUESTED、模型试读NOT_RUN；平台自动删除两个本次中间副本、释放11,308字节，ZIP CRC通过，原始文件指纹不变（real_wrf/receipt.json）。此例验证生命周期和真实导出，不冒称5GB压力测试或完整WRF模拟。

**DS为何反复“推荐失败”。** 只读核对真实记录发现：最近Saguling数值重测复用了同一份2026-11-24线上推荐，不是每次重新调用DS后都失败；客户端均记录reused_inspection/REUSED_REAL_ONLINE_RECOMMENDATION。该请求C1:fusion被标engineering_requires_fusion=true，mandatory过滤后allowed_recipes只有R1：保守对齐→缺失回退→spatial.blend；每源仅4个扰动样本。于是DS在这例几乎没有算法选择自由，只能解释唯一配方，不能将此包装成充分智能选优。实际trial_recipe的融合权重为既有固定0.5，未由DS或独立误差估计出来。

本地配方评分参考为aligned[primary]，primary取较粗网格；降采样后重建该产品，并与同队列最佳单源作比较。它能验证产品重建，不能充当独立雨量观测、证明哪一源真实降水最准；对参考产品有天然偏向。只有3个共同日期、1个预留日期，又不足以证明推荐前抽检与变异后复检均独立。因此配方被拒并不等于DS没有任何作用，更不证明真实多源无价值。DS仍适合识别、语义核对、解释和提出候选；收益必须由任务匹配的真实执行/参考验证。菜单只有一种时应承认只有一种待验方案；无独立降水真值时应仅报代理结果。本轮整理诊断，未扩算法、改权重、修改评分阈值或重新调参凑提升，未声称已经修复完整推荐选型设计。

证据集中output/automatic_scratch_cleanup_20261125，包含before、scope、全部JUnit、recommendation_audit.json、真实WPS回执、acceptance和最终源码/指纹；唯一总报告追加，旧结果保留。原6.13GB人工删除仍外部阻塞，今后正常公共流程的本次自有副本自动清理已实现并验证，二者不得混称全部完成。

## 推荐初筛、新评价与补充推荐调用衔接筛查修复（2026-11-25）

本轮直接检查实际运行目录，不以旧报告或函数名代替调用核查。新证据目录为 output/recommendation_chain_repair_20261125；保留修改前源码、两次阶段性数值重测、失败测试、当前源码和指纹。未同步GitHub，未改原始资料、科学内核、融合权重、收益阈值、固定预算或历史结论。

### 查实的问题与修复

|问题|实际修改|行为或真实证据|结论|
|---|---|---|---|
|有合法继续搜索选项时，页面跳过队友的补充材料重新推荐|__main__.send_next_automatic_fallback移除has_search_continuation短路；保留原eligible、字段授权、API条件和automatic_fallback_started单次标记|test_ready_continuation_does_not_skip_supplementary_recommendation；原有页面补充发送测试|PASS（行为测试，不是重新在线发送全量资料）|
|旧网格初筛先对齐到粗网格，再以粗主源为代理；与现有搜索的原生细产品重建协议不一致|fusion_protocol.fixed_recipe_cohort复用现有prepare，网格初筛经公共算子执行；independent_alignment将真实原生两源传入，不先损失细源网格|原生细/粗网格与输入交换测试、实际trial_recipe测试；public_protocol_numerical/recommendations.json|PASS：共享观察算子、参考定义和目标范围规则；初筛与搜索的时段仍按隔离协议不同，不声称评价队列相同|
|外发后清空旧验收，新候选替换列表导致失败依据不易追溯|independent_review.invalidate/start保留previous_recommendation_review；页面在invalidate前保存原推荐快照|原失败不变、快照深拷贝、未知新证据不能继承PASS、一次补充不无限循环测试|PASS|
|运行过融合搜索就写fusion_executed=true，实际交付可能单源|independent_alignment按实际delivered_best.source_family记录；另列fusion_search_executed；缺来源身份返回UNKNOWN而非伪造融合|真实原包18图记录：搜索true、交付SINGLE_SOURCE、fusion_executed=false|PASS|
|初筛count是输出有效像元数，不是评分分母|评分count改为真实指标分母之和，output_valid_count单列；缺评分时保留unavailable_evaluation_protocols|acceptance.json：评分126，输出有效像元35，两者不混用|PASS|
|只有一个配方仍可能被理解为充分智能选型；代理初筛与最终复核含义混淆|公共explanation_text明确唯一待验方案、代理范围及不能代替冻结后独立复核|页面共用说明函数行为测试|PASS（说明消费者测试；未做本轮手工桌面点击）|

公共链路实际为 recommendations.review_holdouts → recipe_trials.review → 对应真实适配器；数值拒绝后保留原拒绝并提供独立SEARCH_ONLY权限。页面的补充推荐不再被该权限短路；全量材料外发仍会使已暴露的评价失去独立资格，新DS答复必须等待新的合法验收，不能凭投票继承旧结果。补充响应不足时不伪造候选。真实变异后仍走既有blocked_confirmation及冻结图mutation_acceptance，推荐前抽样结论不能直接继承给新图。

### 原包真实结果与证据边界

原613,367,712字节Saguling包SHA仍为d316a261665f5322e8b12a5e7a90e178c81ccd164a0497df72f76837bad35c6f。public_protocol_numerical复用2026-11-24线上DS记录，当前公共入口重新执行真实数值，39.8776秒客户端墙钟，18图/100提议/76真实节点调用；非新一次线上推荐，更不是整轮全量补充DS验收。

新初筛使用fusion_evidence_v2_target_domain，几何选出的参考为原生较细NOAA_CMORPH，而非旧粗NASA_POWER代理；仅在共同目标范围内评价，所有预测输入先经既有面积算子降尺度，隐藏细参考不传给配方。新初筛固定等权配方RMSE15.4215618291 mm、单源基线7.1448631461 mm，126个代理评价点，仍按原门槛拒绝。旧记录23.6957798151/9.89906141367、8点原样保留；参考、算子与分母已变，禁止用两批数值比作精度提升。统一算子不能消除产品上游相关性和尺度迁移假设，也不能凭代理结果证明真实降水融合有无价值。

搜索与最终默认交付仍为SINGLE_SOURCE，RMSE6.1953983729 mm、MAE4.362795，78个搜索期代理样本；这与初筛的126点/不同日期不是同一指标队列，不混比。最终EVIDENCE_INSUFFICIENT、BASELINE_RETAINED；仅3个共同日期及不足的独立预留仍无法支持稳定增益声明。真实产品NetCDF回读、有限/非负、ZIP CRC通过。verify_current.py又以真实适配产物并复用实际搜索输入中已接受的CRS绑定，单独复核最后count字段修复；不从坐标数值另猜坐标基准。

### 继续筛查的其他入口与未完成范围

- station_grid初筛用真实留出站点观测，变异用公共paired服务与分块确认；不能因为配置名仍含blocked_confirmation_v1便断言运行旧判断，实际导入的是当前公共review。
- indexed_table_recipe初筛包含掩码本地值及主序列重建代理；变异用同一paired服务，复用推荐组只作开发并明确隔离不足。区间累计/平均不在该插值入口的支持范围，不把它们自动解释为瞬时量。
- windowed_grid_recipe保留基于实际Affine/CRS的Rasterio窗口重建代理；真实变异目前要求明确瞬时量、EPSG:4326和至少3个完整共同时间。该投影/点支持路径不能机械套用只接受地理cell_mean的公共面积协议，本轮未声称所有栅格和投影入口已统一其观察算子。其参考仍是产品代理，不是环境实测真值。
- 这些适配器仍须各自合法参考和支持契约；没有实测时不能把一致性或产品重建改名成真实精度。没有新的未暴露复核资料时，只能保留基线/探索性结果，不可强制PASS。
- 本轮没有重新线上发送完整扰动副本、没有手工桌面/浏览器整轮验收、没有工程原生写回或模型运行；不能将原包数值路径通过替代上述事项。初筛共享算子只修复实际查实的网格断点，不声称全平台所有适配器和所有客户端均已完成最新版本端到端验证。

定向测试tests7.xml为84项PASS，随后tests10.xml为8项PASS，按class/name取最后一次结果去重85项PASS；不累加重复执行，不跑无关全量。保留tests3的原生网格变量绑定NameError（已修正）、tests5/8/9新fixture缺CRS/连续量/物理语义字段的失败记录（补齐真实测试契约，未放宽生产检查），以及验证脚本读取归一化前产物缺CRS的失败（改为复用实际执行输入已接受的绑定）。源码快照、changes.diff、chain_audit.json、source_fingerprints.json、acceptance.json、真实产品和所有失败均保留。当前是上述具体修复及有界数值验收完成，不是“平台所有评价均无问题”或融合精度优势已证明。

## Saguling原包整轮线上DS与实际桌面窗口验收（2026-11-25）

### 本轮完成范围和真实调用

使用同一613,367,712字节Saguling原ZIP，从实际Window开始按钮发起文件结构读取、线上分类、按需检查、资料判断、算法推荐、本地初筛、补充推荐、合法候选变异、冻结复核、线上输出核对及页面下载。最终live_qt_v3未复用历史DS回复，14个去重线上响应，prompt 222,493、completion 17,115、total 239,608 Tokens；这是最终一轮，不包含此前失败验收轮次费用或Tokens，不以当前价格推算人民币。API回执和实际建议保留，密钥不写入报告或结果。

本轮实际显示Windows Qt桌面窗口，由Qt验收客户端操作原控件并驱动普通Window处理函数/QThread；不替换DS、不模拟数值误差、不手写最终图。外部computer-use工具初始化报failed to write kernel assets / 系统找不到指定的路径，重试仍失败，因此**外部鼠标键盘人工式验收未完成**，不能把Qt自动操作称为该项通过。当前桌面快捷方式实际指向D:\codex\EcoDataEngine\launch_platform.ps1，与此次Window和公共模块同目录，详见desktop_shortcut.json。

|阶段|实际证据|结论|
|---|---|---|
|初始线上DS识别和推荐|live_qt_v3/desktop_records/file-inspection.json、algorithm-recommendation.json、逐阶段events|已执行；没有用旧线上结果代替|
|推荐方案真实初筛|R1保守重网格→缺失回退→等权混合，公共fusion_evidence_v2_target_domain，126个代理评价点|数值FAIL：RMSE15.4215618291 mm，基线7.1448631461 mm；MAE12.4261029106对4.6027544985 mm|
|补充材料和新线上回复|仅在已获准任务共同空间范围及原生一格邻域内发送749条扰动记录，2分片均真实收到响应|已执行，但未形成可放行新配方；不是“新推荐通过”|
|拒绝后继续搜索|页面勾选legal_candidate_search，使用公共selected_processing→local_search→platform_search|PASS：12图/100提议/52真实节点调用；缓存与新执行按已有记录区分，不称全局最优|
|变异后冻结复核|当前blocked_confirmation_v2_aggregate_risk与冻结图回执|关卡已运行，EVIDENCE_INSUFFICIENT、BASELINE_RETAINED；未证明融合增益|
|线上输出核对|output_review.json含实际线上回执|已执行，verdict=needs_inputs；其意见不是模型实际试读|
|页面展示与下载|final.png显示选中来源、代理范围、格式和未完成模型事项；原export_now保存81,048字节ZIP|PASS（实际Qt窗口自动验收）；ZIP CRC、NetCDF回读、有限/非负通过|
|完整原生SWAT+模型输入及模拟|本路径实际NetCDF→NetCDF，7×5×3日，mm|未执行；不能冒称已产生pcp或模型读入成功|

### 发现并实际修复的衔接问题

1. 补充推荐此前遍历整个获准全球变量。fallback_evidence.task_scoped_fields复用已准入网格任务的坐标关系，按共同空间范围及重网格所需一格邻域生成证据索引；numerical_rows只读对应窗口并保留原始索引。数据规模没有人为接入上限，未按数值选“好样本”，只是限定任务相关空间范围；原文件未改。该限定当前消费于已有independent_grid_search任务，其他格式仍走其原有审核路径，不声称任意投影或格式的全量范围均已重写。
2. 新范围字段在内存使用tuple而JSON回读成为list，导致补充发送前的清单严格比较误拒。改为明确JSON列表，增加真实生成ZIP→清单回读→发送协议的定向测试；未关闭摘要或清单一致性检查。第一次真实验收的具体错误“待发清单与获准副本不一致”保存在live_qt，不能归咎于Saguling。
3. 补充资料外发会正确撤销旧验收资格，但原页面随后丢失继续搜索入口。recommendations.supplementary_search_continuation复用原continue_rejected_search：仅对确定数值拒绝的已有合法任务重新提供SEARCH_ONLY；被拒配方仍不作为种子，新DS回复也不继承PASS。同时把已暴露资料的mutation_reservation明确改为INSUFFICIENT，不拿外发过的参考再声明独立提升。页面回调消费此公共函数，无包名、日期或本包算法偏好分支。

### 推荐为何仍未通过：具体原因而非笼统失败

- 本任务推荐菜单只有R1一条完整配方，DS自己也说明“唯一允许的配方”。这次线上推荐证明的是适用性解释，不是从多个方案中预知最优效果。
- R1实际使用原有固定0.5权重；本轮未拟合或调权重。它能执行，但在新的细产品重建代理上比同队列单源差约115.84%，因此按既有评分被拒。代理参考不等于独立雨量站真值，不能据此宣称真实降水多源永远无价值。
- 补充回复没有产生可放行的新方案。F1把recipe写成包含id的嵌套对象，而现有投票协议只接受目录ID字符串，未被当作合法选票；F2明确recipe=null，理由是单源分片缺少另一来源/共同权重依据。完整原文保存在receipts.jsonl。没有把无法解析的建议改写成成功推荐；这也暴露了补充响应格式兼容及成对上下文表达仍有改进空间，本轮未宣称解决这些效果选型能力缺口。
- 即使没有形成新DS配方，公共合法候选搜索已继续实际运行。相同78点搜索队列上，SINGLE_SOURCE RMSE6.1953983729、MAE4.3627947048 mm；实际涉及两个来源的MULTISOURCE最佳已执行路线RMSE8.9122301959、MAE6.6507010872 mm。看似含多个节点但来源身份相同的图不冒称多源增益。没有使用初筛126点与搜索78点直接比较改善幅度。
- 只有3个共同日期，且补充资料已用于推荐，当前没有新的独立确认队列。冻结后复核如实返回不足、默认保留NASA_POWER来源场；既不是跳过变异，也不是已经证实真实降水精度下降。最终候选输出及警告不冒称融合成功。
- 最后DS提到SWAT+所需字段，是对当前NetCDF产品与模型画像的核对。此分支没有完整工程成员绑定及模型程序试读记录，其列表不能直接当作原ZIP确实缺土壤/工程文件的证据。模型原生转换、工程完整性和模型运行与本次数据处理完成分开。

### 全部验收轮次、输出和保留

live_qt第一次真实运行到补充发送前，发现上述清单错误；修复后保留失败。live_qt_v2完成补充、变异和线上最终核对，但验收客户端用同一个周期定时器进入静态保存对话框，无法再入，未完成下载；这是测试客户端缺陷，已停止明确属于该客户端的两个Python进程并保留所有已有结果。修正为独立singleShot操作保存对话框后，以相同原包重新线上从头运行live_qt_v3，不通过换种子、换数据或扩大预算得到成功。

v3导出实际成功，保存对话框保留默认文件名“模拟技术引擎-结果.zip”，验收脚本因只查desktop-result.zip先误记EXPORT_FAILED。未改写原acceptance.json；export_location.json登记真实下载文件位置与SHA，复制完全相同字节作为desktop-result.zip便于核验。verify_live.py独立检查实际ZIP、NetCDF、14次新响应、补充发送、真实搜索反馈、复核、输出线上响应及原包SHA，verified_acceptance.json中9项功能检查PASS。其PASS只指这条Qt数据处理/下载链路，不能替代推荐效果、独立验证、模型原生输出或外部鼠标键盘验收。

最终产品为2018-02-03至05三日、lat7×lon5、precipitation单位mm；输入和输出都是NetCDF，内容做了时间区间处理和目标网格对齐。模型未运行。最终一轮约227.375秒客户端墙钟，不能把多次失败及开发测试的时间隐藏成一次总成本优势。

53项新增与受影响测试按class/name取最后一次PASS（tests3/4/5.xml），无无关全量回归。测试覆盖任务范围索引、ZIP清单往返、保持原始索引、完整发送授权、失效后不继承验收、失败后继续合法变异、字段/状态隔离和最终复核。源代码执行前后指纹核对一致；executed_source、executed_source_fingerprints.json、failure_reasons.json、所有失败、源码前快照、API回执和实际桌面图均集中保存，不覆盖旧实验。

正常完成的v3桌面临时目录在退出后自动清理已核实；v2因停止客户端留下9,796,110字节测试目录。仅对本轮自建、已核实所有权的目录提出一次删除，自动审批仍以blocked by policy拒绝，未换工具绕过，未伪报清理成功。此残留与历史6.13GB目录不是同一项，分别保留边界；failed_client_cleanup.json记录本轮情况。所有原始文件、旧协议和历史产品保持。


## 第二步界面收敛与Saguling整轮验收（2026-11-26追加）

本轮仅修改src/ecodata/desktop/__main__.py的桌面流程与展示，复用公共放行、自动补充推荐、合法候选搜索和最终复核机制；没有降低评分阈值、修改算法或覆盖历史实验。

第二步后台自动完成检查及必要补充推荐，处理中禁用继续操作。主区只显示共享放行机制允许继续的方案；单个方案默认选中，提供明确“继续处理”按钮。失败、待核验和建议项收进“查看检查详情”，保留原始依据、此前数值失败与当前独立性限制。没有可执行方案时显示具体阻塞原因并禁用继续；工程核验与融合放行仍分别处理。允许合法候选搜索不等于原推荐通过，更不保证融合胜出；隐藏失败选项即使被程序勾选也不能绕过公共门禁。

定向测试按tests4.xml、tests5.xml、tests6.xml中class/name最后一次结果去重：{'PASS': 47, 'FAIL': 0, 'SKIP': 0}。覆盖后台忙状态、默认选中、失败隐藏但保留、明确阻塞、共享最终门禁、工程独立核验、仅转换兼容及历史失败依据。此前失败测试记录保留，不运行无关全量回归。

原Saguling包通过可见Qt桌面窗口和实际控件处理函数完成新一轮线上DS检查、补充推荐、合法候选变异、冻结复核、输出核对及保存对话框下载。12条线上响应；实际搜索12张图、100次提议、52次数值节点调用。下载live/desktop-result.zip为80,428字节；ZIP CRC、NetCDF回读、单位mm、非负/有限值、原包SHA不变均通过。输出为三日7×5网格NetCDF，输入也是NetCDF；不是SWAT+原生pcp写回，未运行模型。最终BASELINE_RETAINED，独立效果证据仍不足，不声称融合改善。正常退出的桌面临时目录自动清理已核实。

完整线上链的执行源码保存在executed_source；其后仅调整详情背景、状态换行与历史失败说明显示，最终源码保存在final_source，两版本差异见executed_to_final.diff。最终展示使用本轮真实记录回放并截图检查，不重复API或数值计算。step2_ready.png、step2_details.png、step2_default_size.png为最终展示；visual_check.json明确为展示回放。外部计算机控制工具初始化失败，未完成独立人工鼠标键盘验收；不能将Qt自动化称为人工验收。

全部证据集中output/step2_clear_flow_20261125：before快照、两类源码差异、delivery_summary.json、verified_acceptance.json、测试XML、live实际调用/搜索/产品记录及最终界面截图。旧结果保留，本轮未向GitHub推送。


### 第二步改动后定向自检（2026-11-26）

本次只读审查UI状态、行与勾选映射、自动补充回调、公共放行与实际提交关系；未重新运行线上DS或整轮实验，未修改生产代码。重新执行31项第二步/续行门禁/拒绝推荐后合法搜索定向测试，全部通过，记录output/step2_selfcheck_20261126.xml。首次默认pytest临时目录因Windows拒绝访问导致3项setup错误；未删除旧目录，改用本轮独立basetemp复测通过。现存体验限制：详情仍有技术用语，接口异常后不保证自动续完；无选择时继续按钮仍可能可点、由提交校验提示勾选。无证据证明这些有限测试覆盖全部任务。Saguling效果与模型运行结论保持原记录，不因UI自检升级。


## 桌面受管临时目录自动回收（2026-11-26）

复用现有TemporaryDirectory生命周期和normalized_archive_scope归档清理，仅增加core/temporary.py公共受管目录：默认位于LOCALAPPDATA/EcoDataEngine/temporary-v1，保存独立token、schema、目录名、PID及进程创建时间。桌面窗口启动调用sweep_abandoned；只回收标记有效且原进程已退出的目录，PID复用不阻止旧目录回收。活跃进程、进程状态无法核实、无标记、标记冲突、符号链接/目录联接和越界路径保留。文件被占用则记录失败，下次进程退出再启动重试，不绕过系统权限。没有按名称清扫旧历史目录。

桌面扫描、按需文件工具、独立复核、降水分片、临时重网格、候选回放等原TemporaryDirectory调用改用同一公共生命周期；显式dir参数仍按原有作用域使用，通常位于受管任务内。归档标准化副本原清理机制保持不变，科学算法、门禁和选优没有修改。保存取消/失败保留待下载产品；默认结果保存成功后清理本轮run/task工作目录、主要结果与准备状态引用，执行gc并关闭该轮再次导出/候选回放，保留结果页面摘要及用户保存的ZIP；下一轮重新识别执行。输出若被保存到内部工作目录，不删除该目录并明确提示另存。正常关闭清理，清理失败不阻止关窗，留下标记供后续启动重试。

本轮49项去重定向测试全部通过（tests1/2/3.xml）；涵盖真实子进程os._exit强制退出后回收、活跃进程不删、PID复用、占用重试、无标记/链接保护、启动实际调用、成功/失败/内部路径保存、原有按需工具和归档清理及第二步回归。未运行大型搜索或付费API。

使用上一轮Saguling真实结果包，经当前Qt导出处理函数、真实公共ZIP导出和新清理回调验证：输出ZIP CRC通过，NetCDF回读lat7/lon5/time3，原包SHA不变，工作副本删除，已保存结果保留；记录export_acceptance_final.json。此为现有产品保存/清理验收，不是重新运行融合，也不是外部人工鼠标验收。首次验收客户端遗漏scan.path导致KeyError，补用真实inventory后通过，未绕过assert_current。

限制：历史无归属标记的副本、用户主动保存的结果、开发output证据不会自动删除；已有显式外部dir或导出同目录原子写入的临时文件仍依赖原finally清理，强制断电残留不按未知路径自动回收。内存仅释放平台主要引用和回收对象，Python/Qt/数值库保留的缓存不保证RSS立即恢复；本轮不声称进程零内存占用或任意历史垃圾全部清除。全部源码前快照、执行源码指纹、测试与实际产品回读在output/automatic_cleanup_20261126，历史实验保持，未推送GitHub。


## CAMx桌面流程、密钥记忆与自动补充预算修复（2026-11-26，20261126-r5）

本轮以桌面原包02_CAMx_两日嵌套工程与站点观测 - 副本.zip和实际问题记录核查。Windows快捷方式指向D:/codex/EcoDataEngine/launch_platform.ps1；初始窗口为旧进程20261124-r2。保存其检查记录后关闭，重新启动并通过外部Windows控制工具确认最终窗口20261126-r5、原包路径及“已记住DeepSeek密钥”。最终版本未重新跑完整付费链；不得将前阶段r4线上过程冒称r5逐字相同源码的完整验收。

| 原问题 | 公共修复位置 | 真实行为证据 | 状态与边界 |
| --- | --- | --- | --- |
| API只在会话内记住 | core/credentials.py、desktop/__main__.py | 官网/models验证成功后写Windows凭据管理器；去掉进程环境变量、创建新窗口仍恢复 | PASS；不明文写项目，不以保存成功代替每次网络可用 |
| 第一页只有泛化提示 | desktop/file_agent.py、__main__.py | 实际列目录、读指定表头/依赖、缓存复用及耗时发到主页面最近12项时间线 | PASS；不伪造百分比和阶段完成 |
| 无关上下文仍哈希大包 | models/input_context.py | 不存在同模型上下文时不读完整原包；同模型仍核对SHA | PASS；不是跳过输入变更检查 |
| 描述JSON被当成原科学文件已检查 | file_agent.uninspected_described_sources及现有工具循环 | 有变量描述但没实际表头时，在原12轮/48工具预算内最多一次补查请求；r4真正读到NOAA与met原生NetCDF | PASS；不能定位仍UNKNOWN，不凭描述伪造原文件 |
| NOAA复合编码被当浮点数导致假失败 | sample_values.field_values、sampling_plan、sampling_risk | 复用io.noaa_global_hourly.decode的尺度、质量码、缺失编码；同原CSV原留出从tabular_unit_incomplete变checked，16个有效样本 | PASS；不填造观测，不修改阈值 |
| 比较任务走逐字段通用抽检 | recipe_trials、station_grid、recommendations | 真实站点/原生网格联合试跑；没有BIAS节点不拟合订正。r4精确9条、最近24条验收released；另实际运行输出9条Parquet并回读 | PASS；仅比较与适用性，不是融合增益 |
| 个别检查失败挡住独立合法任务 | __main__.continue_automatically及共享放行机制 | 有真实可继续方案/工程时不被无关失败全局否决；原因留详情；没条件仍不能假通过 | PASS定向行为；r4线上无关失败分支未触发最终修正，最终函数另有定向测试 |
| 自动全量补充出现362联网分片 | fallback_evidence.automatic_budget/transmit、__main__自动补充 | 真实362分片清单在最终公共预算被拒绝且不发请求；UI行为测试确认失败保留、合法方案继续、busy清除 | PASS；每输入自动最多4分片并预留汇总，总5逻辑请求，网络有限重试另计。人工明确批准的全量发送不改为默认自动 |
| 风速标量与u/v分量混用 | station_grid.task_for | 有依据的不相容操作退出，原背景及其他任务保留 | PASS；没有合成/方向契约不擅自构造风速 |

### 本轮真实过程与成本边界

旧记录按需检查约222.232秒，其中本地87.772秒、API及编排134.460秒。r3线上检查100.525秒，但只读到两个观测表，不能作为与旧流程相同科学范围的加速证明；该轮暴露未读模型科学文件的缺口。补查修复后r4按需检查172.062秒，其中本地48.991秒、API及编排123.071秒，实际读到NOAA与原生met/camx.2d.12km.20160610.nc。日期、来源和读取范围不同，不声称稳定提速比例；复杂嵌套归档读取与DS思考仍需时间。

r4线上推荐与本地核验已完成，两个温度配对方案released；另一个订正方案真实评分失败后允许合法候选搜索。随后现有自动全量补充生成362分片，发现后主动停止属于本次验收的PID25608/17000，保留已收到的51条分片回执、批准清单和7MB证据副本。该轮不是完整成功。最终增加联网预算后，用同一实际清单验证不允许发送，并测试跳过补充不会暂停合法方案；不通过再发362批证明预算。实际已消耗回执Tokens见actual_budget_fix.json，未核实货币费用，不写为零。

当前最终版的NOAA公共抽检与预算通过定向及真实记录检查，最终桌面已重新启动、密钥恢复和输入路径经外部工具核实。另一次当前公共站点试跑真实从原包读取并输出station-grid-1.parquet（9条），19.355秒、原包文件签名未变。这个局部产品证明读取与配对真实执行；本轮没有重新完成CAMx整包融合、原生模型写回或CAMx模拟，不宣称已证明融合提高。

### 测试、保留与未解决项

去重测试结果：{'PASS': 143, 'FAIL': 6}。保留tests1至tests10.xml全部记录；6项station_grid_fusion旧测试失败，代表random用修改前源码覆写导入同样复现17/731合成缺失位置的数值不一致。问题与既有自动补值提前填入主源有关；剩余5项未逐一做before复验，不把它们隐藏为PASS，本轮未改算法内核或降低断言。新增密钥、主进度、真实读入失败不可放行、比较不拟合订正、无关失败不否决、复合质量码、自动补充预算等测试均通过。

源码前快照、final_source及指纹、原问题、线上记录、部分失败回执、真实产品、采样修复与预算检查均保存output/camx_desktop_flow_20261126，详delivery.json。sample_values.py前版本与此前归档逐字匹配后复制，其余已修改生产模块执行前直接快照。测试中的Windows凭据使用独立临时目标并删除，不替换用户真实凭据；真实验证只用本机已配置密钥，未写入源码或报告。原始数据及历史结果保持，自动临时清理机制保留，未推送GitHub。


### 原CAMx整包公共流程实际试跑（2026-11-26）

按用户“试一遍、只用平台自身能力”，调用与页面相同的desktop.service.run_original_workbench，复用已有原包检查记录，未重新请求DS推荐，未手工替换输入或拼接输出。原包02_CAMx_两日嵌套工程与站点观测 - 副本.zip为3,292,398,499字节，SHA256为40b48781983fe52611dba3ff709259f66c2ce1ab3bd1128493462aa7556fa85f。

真实结果FAIL：automatic_preparation -> normalized_archive_scope -> archive_inputs._Reader.copy在嵌套展开阶段触发“压缩资料展开超过10 GiB接收上限”。该公共模块仍从interface.engineering_upload导入10 GiB MAX_TOTAL_BYTES，累计展开量受固定上限约束。这次是平台残留接收限制，不是本轮检出了缺文件或科学定义错误；不能据此保证越过限制后所有后续步骤都会通过。本轮用户要求试跑，未擅自解除全平台资源保护或编写外部数据处理绕过。

总耗时约51.60秒，受监测进程合计RSS峰值203,087,872字节（约193.7 MiB）。平台finally清理回执COMPLETED，释放10,580,619,702字节解包工作副本；本轮受管临时目录不存在已核实，原包前后SHA256完全一致。融合、原生写回、原生产品回读及CAMx程序执行均未运行，不能标记整包跑通。

本轮没有修改生产源码，没有重复全量测试或启动研究实验。调用客户端、真实进度、完整异常栈、资源记录、公共源码指纹与逐阶段acceptance.json保存在output/camx_full_platform_20261126。此前温度配对9/24条通过的有限结论保留，不升级为整包完成。


## 公共按需归档读取、分块写回与CAMx原包验收（2026-11-26，selective-intake-v1）

本轮复用现有ProjectFiles、工程绑定、C-shell静态解释器、meteorology_schedule、Pipeline局部残差处理、原生NetCDF块切分、最终检查及受管临时目录。未修改科学算法、评分阈值、GOLEM源码或原始输入；未运行研究对照或重新请求DS。此前失败结果保留。本轮是桌面公共处理函数run_original_workbench和原下载函数验收，不是再次手工操作完整GUI，不宣称浏览器验收。

| 原缺口 | 本轮修改 | 实际证据与结论 |
| --- | --- | --- |
| 嵌套资料先全部展开再重打包，触发10GiB限制 | models/selective_resources.py提供作用域内的虚拟资源索引；archive_inputs与ProjectFiles共享。顶层ZIP原地读取，嵌套压缩容器为可寻址访问暂存；TAR顺序扫描头部，叶数据按需复制。复制块1MiB，最终头部缓存最多8MiB | 原包84个唯一叶资源，75个因工程依赖、观测或同路径重复校验而实际暂存；年度EPA观测叶文件未暂存。不再生成expanded_inputs.zip。TAR仍需顺序解码才能索引，不宣称只读ZIP目录就能定位所有TAR叶文件 |
| 现成多运行段模块没有公共入口、缺少实际导入 | automatic_preparation接入已有绑定计划；project_binding复用已有模板绑定；native_meteorology连接prepare_schedule；修正不存在的_open_native导入，直接用已安装netCDF4.Dataset | 原两天、两网格均实际走到处理和写回；另一个名称/年份/网格的小型合成工程通过公共入口，不以该合成例证明环境精度 |
| 缓冲区坐标与内部网格未统一核验 | native_axes依据原生X/Y单位、NCOLS/NROWS及BUF元数据核验中心坐标；geometry复用现有投影实现 | 原真实网格及缓冲区通过；声明冲突拒绝。通过容差核验后使用元数据中心，避免float32公里存储转换后的微小舍入影响精确格点坐标 |
| 原生回读整变量进入内存 | 局部更新按格点写入，回读复用existing_air_converters.native_netcdf_update._chunks，以不超过4MiB的数值数据块逐块遍历全部变量 | 大于单块的NetCDF定向测试及真实4个更新文件逐变量比较通过；未更改其他变量、时间标记、缓冲区或未处理支持域 |
| 多日导出控制被视为多个冲突工程 | 从camx_templates提取既有绑定逻辑给真实namelist复用；schedule校验同目录、明确时序和实际前序产物依赖；独立工程不合并 | 仅重跑受影响的公共最终关卡后，41份NetCDF格式PASS、工程实际依赖绑定PASS；运行时重启产物仍标为待模型生成，没有伪造缺失产物 |
| 临时索引和成员副本清理 | 复用core.temporary所有权标记和finally释放，失败清理记PARTIAL并保留原因，旧输入和产品受保护；接收回执在退出作用域后写入 | 原包完整流程临时根目录已不存在；按需资源清理回执COMPLETED，释放14,055,908,283字节。完整任务的其他工作副本也随外层受管目录自动清理，不由人工删原始输入 |

原输入为桌面02_CAMx_两日嵌套工程与站点观测 - 副本.zip，3,292,398,499字节，前后SHA256均为40b48781983fe52611dba3ff709259f66c2ce1ab3bd1128493462aa7556fa85f。完整公共流程及首次导出用时436.83秒，实测进程合计RSS峰值292,024,320字节，约278.5MiB。实际暂存压缩容器3,292,325,155字节、叶资源10,763,583,128字节；这些是工程依赖/冲突核验的磁盘工作副本，不是同等内存驻留。完整原生输出仍需包含控制文件引用的较大依赖，因此产品约3.31GB；不能把必要产品或所有工程依赖承诺为零磁盘占用。

实际处理范围：NOAA有效且恰逢原生时刻的温度观测，与CAMx二维t2场进行已有局部残差订正。四份文件分别更新9、9、7、7个格点时刻值，共32处，均为NETCDF3_CLASSIC，原生回读PASS；同一观测在不同网格使用，不能当作32个独立实测样本。不向全域外推，不改风场和三维动力字段。本次是已有格式内数值写回：观测CSV经公共NOAA解码、摄氏度转K后进入订正，输出仍为CAMx原生NetCDF3；不是把原生NetCDF改称另一种新格式。本轮没有独立精度改善实验，没有重跑算法优越性研究，也没有证明订正优于背景场。

最终关卡：格式PASS、工程依赖绑定PASS、CAMx程序试读NOT_RUN。本机未安装已登记并锁定指纹的D:\EcoDataEngine\work\camx740\src.v7.40\CAMx.v7.40.noMPI.NCF4.gfortran。原生文件回读通过不等于CAMx已经实际积分，第二天重启依赖仍需第一天模型运行生成。完整模型可运行性与环境精度未验收。

测试：最终tests9.xml为42项通过、0失败；之后对运行计划独立性收紧的6项受影响测试再次通过，不能相加当成48个独立测试。覆盖按需头部读取、选中成员、原输入不变、不安全路径、正常/异常/取消清理、清理失败保留原异常、真实父依赖、模板与真实控制回读、坐标缓冲区、稀疏写回和有界块比较。早期失败记录tests1至tests8、attempt1/attempt2均保留：同路径重复资源先严格失败，后复用相同内容合并原则修复；另发现现存多网格模块导入缺失并修复。测试中强制锁定清理的用例会产生tempfile终结器告警，清理失败未标成功；NumPy写入弃用警告不被屏蔽。没有执行无关全量回归，历史站点融合6项不一致不因本轮定向测试通过而宣称已解决。

证据统一位于output/selective_intake_20261126：before、final_source、diff、tests*.xml、execution.json、progress.jsonl、resources.json、records/native_validation.json、final_recheck、delivery.json及CAMx-platform-result.zip。完整运行先完成，随后仅对受影响最终输出关卡复检，87.02秒；末尾的头部缓存上界、平面多网格入口和退出后回执位置做定向复测，没有重新执行整轮数值融合。最终下载追加复检版记录，保留较早失败关卡回执以便追溯，不把旧回执改写成PASS。

当前边界：此次公共按需索引支持ZIP及常规ZIP/TAR嵌套，深度最多5、叶成员最多100000，真实可用磁盘不足时明确停止所需暂存，不取消资源安全。TAR链接/特殊成员在新按需路径明确拒绝；7z仍走原有路径，尚未迁移为按需读取。没有声称五种模型所有归档格式全部验收。没有推送GitHub；已写入D:\codex\EcoDataEngine实际项目，已启动的桌面进程需重新打开才能导入新代码。

交付后复核：当前源码与final_source全部指纹一致，最终ZIP成员名无重复，已追加desktop_issues.2.json等最新导出记录，处理原包的临时根目录已不存在。尝试清理本轮pytest夹具时，自动审批以“blocked by policy”拒绝删除命令，未执行；约55,594,363字节测试夹具因此保留在本验收目录test_tmp*，不能将其写为已清理。该拦截不影响此前平台finally实际完成的约14GB按需资源回收与外层工作目录释放。


## 2026-11-26 公共复合观测解码与桌面搜索—原生写回衔接（r8）

本轮复用公共读取、已有科学算法、变异服务、质量门、工程绑定和原生writer，修复真实桌面流程中发现的断点。没有降低评分标准、改变算法内核、给CAMx文件名写规则或覆盖历史实验。实际运行代码为D:/codex/EcoDataEngine；Git协作目录为D:/codex/EcoDataEngine-GitHub，二者来源不能混称。

| 核实的问题 | 公共修改位置 | 实际验证及边界 |
|---|---|---|
| 留出检查能解码复合观测，正式表格质量门却把“数值,QA”当普通浮点，误报没有有效数值 | desktop/release_gate.py、fallback_evidence.py复用sample_values.field_values及已登记NOAA解码器 | 温度、风速、海平面气压统一按原有比例、缺测码和QA解码；未知复合格式仍拒绝。没有去掉质量码或填造有效值 |
| 广义wind_velocity别名绕过原有速度/分量检查 | desktop/station_grid.py依据实际观测standard_name和两端物理身份检查 | 标量风速与u10/v10不能直接融合；只跳过对应任务，不影响温度分支。没有新增风矢量合成算法 |
| 显式evaluate_gain目标被DS的仅比较/对齐答复提前结束 | desktop/discovery.py复用既有目标补全、compatibility和数值核验 | 同一真实数据结构在evaluate_gain下继续检查融合候选，在grid_comparison下不强制融合；DS明确拒绝/直接使用不被本修复覆盖。没有指定算法赢家 |
| 推荐证据不足只能停留在待核验 | desktop/recommendations.py复用既有SEARCH_ONLY继续机制 | 仅已有适用路线和已登记“评分支持不足”状态可进入有界探索；不放行原推荐、不作为种子偏好，不绕过真实Objective及最终独立复核。覆盖失败、未知适配器等仍不能放行 |
| 纯数字station被公共识别转成浮点字符串，实际预测站号与上游不相等 | recognition.py既有标识字段集合补station、site_no | 界面r7真实失败保留；r8保持72520094823而非72520094823.0；含前导零站号亦经真实公共数值执行验证，不靠删“.0”或猜测ID修补 |
| 站点预测仅导出点产品，未接入工程原生写回 | desktop/station_grid.py、local_search.py、selected_processing.py；models/engineering/native.py新增最小公共产品绑定适配 | 传递真实上游格点索引、背景值、源内容指纹及最终冻结产品；单位、索引、背景一致性和冲突检查后复用prepare_components/update_project。比较诊断标为comparison_passthrough，不能盖过selected_product；独立工程不静默合并，缺契约保留具体失败 |
| 部分点产品变量清单被误用为整个工程缺项依据 | desktop/output_review.py复用实际最终工程关卡结果 | 工程PASS时不从部分产品臆断缺PM25；native_ready及模型实际运行仍不提升为PASS |

桌面原包02_CAMx_两日嵌套工程与站点观测 - 副本.zip为3,292,398,499字节，SHA256仍为40b48781983fe52611dba3ff709259f66c2ce1ab3bd1128493462aa7556fa85f。r8在真实外部PySide界面选择CAMx、调用线上DS、展示可继续的方案并点击执行，再通过真实保存对话框下载。未用后台脚本、预置最终图或模拟误差替代此轮界面验收。截图和可访问性记录见验收目录ui/16至19。

最终C1温度路线：原推荐数值评分未通过，自动转入公共合法候选变异；4个完整候选记录、3次kernel调用、1次新的变异执行、256次有界提议，搜索本体3.191秒，RESOURCE_BUDGET_EXHAUSTED停止。不能把候选记录、缓存调用和新的变异执行相加。最优探索路线含实际父边上的拟合状态及bias.prior_series_affine；候选与最终背景在5个应用时刻均有差别，最大差0.620447K，证明并非只改日志。但复核CONFIRMATION_FAILED：选择侧仅2个块、复核3个块，选择侧RMSE变化约-0.340712K，复核约+0.147078K，证据量不足且跨块不稳定。最终BASELINE_RETAINED，保留探索候选但不当作已推荐增强，不声称融合提高精度或全局最优。

最终产品经公共工程上下文绑定到原生t2父资源的5个真实格点时刻。交付的是冻结背景，因此changed_values=0，原生文件SHA256与父资源一致；这不是把未通过复核的融合产物强行写回。原生输出仍为NETCDF3_CLASSIC，t2维度[25,1,149,152]、单位K，24个变量可读；没有把细网格、点预测或格式通过当成环境精度证据。最终关卡41个NetCDF格式PASS、工程引用PASS；CAMx匹配程序缺失，模型实际试读NOT_RUN，第二天重启资源仍记录为前一天模型生成依赖。

界面下载CAMx-desktop-r8-result.zip为3,312,948,740字节，15个外层成员、55个原生工程包成员。独立验收对外层和内层ZIP逐成员CRC检查通过，再重新打开下载包中的实际原生文件；文件指纹与冻结交付一致，NETCDF3格式、尺寸和单位回读通过。此独立下载检查27.375秒，不能当作搜索成本。本轮最终关卡自身56.859秒；识别阶段168.015秒，其中本地工具75.546秒、API及编排92.469秒，0次该阶段缓存命中。该识别阶段记录8条线上回执、273846 tokens，不代表包含推荐、补充推荐和输出审核的完整账单。主流程从界面执行到完成约8分钟，主要还包括按需工程依赖暂存、压缩、检查；不能用3秒搜索宣称整轮只需3秒。离散RSS观测最大约559.8MiB，不是完整连续峰值测量。

清理验收：按需索引84个成员，记录分块暂存10,763,549,713字节、压缩容器3,292,325,155字节，按需临时根目录清理COMPLETED。界面下载后明确显示“工作副本已清理”，实际桌面任务目录也已不存在；原始输入与用户下载产品保留。工作目录的清除不代表关闭桌面进程或把进程驻留内存全部归零。电脑21:06实际重启造成的上一轮中断及外部DNS/超时/流中断均保留，不把重试记为独立成功实验，也没有更换种子追求好结果。

定向测试：r7修复集66项通过；站号修复后的r8新增及受影响24项再测通过；识别/表格格式受影响20项通过。存在重叠，不相加成110个独立用例。测试包含数字/前导零ID经过真实搜索与原生索引绑定、未知复合值拒绝、非法风分量匹配、目标补全、仅探索准入、真实父资源及背景冲突拒绝、诊断分支不覆盖最终产品、原始文件不变。未运行无关全量回归或大型研究实验；NumPy及既有依赖弃用警告未屏蔽。最早的夹具失败、错误风分量匹配、仅比较的r6及r7绑定失败均保存；这些旧结果不能当作本轮完整融合成功的证据。

当前验收结论：公共解码PASS；真实界面推荐→失败后合法变异→候选数值评价→最终验收→冻结产品原生写回→独立回读→下载/自动清理PASS_WITH_LIMITATIONS；融合增益未获支持；模型求解未运行。本结论只覆盖实际选中的温度资源及已核验工程依赖，不宣称两日所有变量均融合，也不以一个CAMx包证明所有工程完全通用。证据集中于output/desktop_decode_20261126，紧凑副本随代码保存至docs/acceptance/desktop_decode_20261126；原包和3.31GB产品不入Git。执行源码指纹、协议、失败记录及验收结果分别见r8-source-fingerprints.json、protocol-r8.json、acceptance.json及r8-download-verification.json。

## 2026-11-27 全Registry发现、DS显式构图与开放变异集成（registry-graph-v1）

本轮以实际运行目录D:\codex\EcoDataEngine为准，复用当前公共图变异后端、Registry、语义识别/能力映射、NumericalDAGExecutor和独立评价。旧GOLEM已被前序版本移除，本轮没有重新引入、复制优化器或将结构分数冒充数值误差；没有新增科学算法、训练模型、下载数据、重跑北京全量或949图穷举。证据集中于output/registry_graph_20261127，历史协议与结果保留。

核实的边界：原桌面推荐主要由allowed_recipes限定，registry_advice只展示目录、steps为空，不进入执行。通用场变异器已有较广的Registry分派，空间留点适配器仍只有窄的固定组件语法。因此“目录可见”“实际可执行”“该入口能构图”“已经实测”不能混称。

| 原缺口 | 公共修改位置 | 实际证据与判定 |
|---|---|---|
| 全库目录缺执行条件、未逐项筛查 | registered_graph.catalogue/discover；field_mutation | 121项注册全部发现；99项available，另22项有明确状态；包含适配器/检查条目，不等于121个独立内核。逐项契约、资源及三个真实上下文的兼容原因见algorithm_integration_matrix.csv。有限父绑定检查不宣称穷尽空间；available也不等于所有算法均已实测 |
| DS只能选固定配方 | platform_search.recommend_and_submit、with_recommendations；desktop.recommendations/service | 支持显式nodes/parents/parameters/output，自主引用实际来源，最多8个推荐处理节点；无需处理可保留输入。原始线上响应、拒绝和一次有界契约修正分开保存。质量检查条目由自动关卡消费，不能作为处理节点；不提供真值或预期赢家 |
| 推荐图没有实际进入执行与变异 | field_mutation.initialize_recommendations；paired_station_search；桌面station_grid/independent_alignment/indexed_table_recipe | 图先经过当前Registry及真实父输入检查，实际执行、数值评价与最终质量门后才能成为变异父图；拒绝的推荐不能变成奖励。搜索仍能引入推荐图以外的适用算法；执行失败继续记录并探索其他合法候选 |
| 初始登记顺序及预算耗尽妨碍变异 | field_mutation.initialize/propose/execute | 基于实际根兼容性安排初始优先级，种子控制的生产者轮转，所有available处理插件仍保留在变异域。初始化预留数值调用；提议预算结束只禁止继续提新图，不取消此前合法图的评价，调用/时间预算仍严格执行。最后一条合法提议实际执行的行为测试通过 |
| 观测/辅助/训练资源只藏在上下文 | autonomous_decision.Executor | 参数和辅助资源经resource.context真实父边进入计算，拟合状态保留状态端口及来源/训练时间检查；参考标签不进入资源边。缓存键计入实际资源指纹，执行契约升级v2_explicit_resources；资源边不代表算法一定使用了协变量，不据此宣称融合贡献 |
| 空间搜索只有几种估计方法 | spatial_registry_binding、spatial_fusion_task | 按Registry真实observation_interpolation契约发现站点插件，复用其科学实现；按日期仅传保留站点，真实目标几何执行后回取对应位置。新增节点及连接可被变异选择，未伪造站点观测。真实NCEP小窗出现station.pykrige_ordinary→spatial.blend的变异图；MetPy Barnes/Cressman、PyKrige ordinary/universal均有实际调用；regression因缺协变量明确失败 |

统一调用为：上传/解析数据和结构化目标→SemanticFeatureExtractor与现有RequirementMapper/AlgorithmSelector→完整Registry契约及实际兼容检查→DS显式图提议→公共图准入→真实父边执行与质量/独立数值评价→公共随机图变异→冻结产品及回读。补齐了实际目标网格、时间队列和非负要求进入推荐载荷的连接；不是新增“某缺口必须选某算法”的规则。页面处理函数、工具包submit_file_request及研究客户端可调用同一recommend_and_submit；桌面已为站点—网格、索引观测表、独立日降水网格三条路线传递同一显式图协议。搜索准入显示为SEARCH_ONLY，不冒称推荐已通过数值验收；数值检查发生在实际搜索初始化阶段、变异之前。

### 固定小数据验收及真实负结果

先冻结seed=11、12次搜索调用上限、总调用30、提议80、搜索墙钟40秒；三个公开真实数据子集分别为NCEP 6×6格点×6时刻的2×2块均值重构、NCEP单格32个6小时时刻的隔时刻隐藏恢复、已有WA站点与ERA5前90日（前45日实测可见、中22日验证、后23日复核）。前两项参考为产品本身，不是独立地面真值；降水保留既有单位与日界说明。主数值目标沿用公共日期/区域分块损失，表中RMSE是物理单位的合并指标，不能将两者混为同一数值。

| 最终版本真实案例 | 数值结果及产品 | 新变异执行 | 验收边界 |
|---|---|---:|---|
| 连续空间重构 | PASS；RMSE 2.330044 K；11次内核调用；输出回读完成；选中framewise.raster_task.pchip | 1 | DS的bilinear→非负投影图在真实父输入处因外缘缺测不满足有限栅格契约而被拒绝；公共搜索继续找到合法路线，并非强行放行推荐 |
| 时间缺失恢复 | 推荐temporal.raster_linear真实评价通过、产品可读；RMSE 3.779440 K；2次内核调用 | 0 | 原始状态BUDGET_EXHAUSTED_BEFORE_MUTATION保留。80次固定提议内没有新的数值变异执行，不能标为“时间变异验收通过”，没有追加种子/预算凑成功 |
| 站点与ERA5多源任务 | PASS；RMSE 62.930337 mm；3次内核调用；最终保留单一背景源，产品可读 | 1 | DS可合法选择不组合；不存在多源硬加分。该结果不支持融合增益或算法优越性 |

另有一轮真实NCEP空间留点代理验收，固定既有预算130/180次调用、100次提议、45秒、seed=11。公共服务实际生成并运行PyKrige估计→blend的多算法图，执行轨迹记录下游绑定上游field；不是手工预置最终图。此例只验证产品空间重构及连接，不等于地面精度或降尺度信息增加。合成边界测试另验证改变真实父边产物会改变下游、错误父边/伪造状态/未使用分支拒绝，以及每个新增空间Registry叶节点在受控抽样中均可达。

real-v7保存本轮线上DS请求/响应和用量回执。最终real-v9按同一输入、完整载荷指纹严格回放这些已获取建议，所有数值Objective重新执行，不复用旧数值排名、不冒充新API调用。中途的结构响应协议错误、缺少明确验证split、目标信息漏传、预算前置耗尽等失败分别保存在real-v1至v8及日志中，不作为独立重复实验。源文件只读；NCEP输入前后SHA256一致。推荐/发现开销、数值搜索调用、失败与产品回读分别记录；不以减少图数宣称总成本优势。源码修改前副本在before；额外Git基线文件在baseline_git（来源7e35e35，不冒称修改前运行目录逐字节副本）；最终源码与依赖指纹另存。

### 未接通及未验收项（不能改称全部完成）

1. 全库发现PASS，通用场图分派按实际compatibility条件开放；尚未逐一数值运行所有99个available条目。缺权重、依赖、GPU、许可或模型专用资源的条目逐项保留原状态，不因目录可见就放进执行。空间regression/3D及地形、学习重构等缺对应协变量/垂向/权重时仍不可执行。
2. windowed_grid_recipe、pair_processing、observation_gap_search、original_engineering、native_reference_search等旧专用入口保留原执行，但其推荐协议尚未全部迁移为任意Registry图，详见pending_bindings.json。不宣称这些入口都已取消固定配方；尤其窗口读取的对齐准备仍依赖已登记步骤，未用错误空步骤伪装支持。
3. 当前版本的外部桌面/浏览器完整上传、提交和下载未重新人工验收；页面处理函数/工具包共用公共入口及相关Qt回归不等同于浏览器验收。已打开的旧进程需要重启才能载入修改，不宣称其已热更新。
4. 最终新增及受影响定向测试81项通过、0失败/跳过；未跑无关全量回归。时间新变异与上述专用入口仍有明确未完成项；未检验算法优越性、效果预测、操作收益学习或全局最优。没有自动启动正式研究实验或推送GitHub。

本轮交付为可运行的开放图协议、全库逐项发现、公共数值反馈和真实正负验收证据；不以空间/多源PASS替代时间新变异验收，也不以一个公开图入口代替全部旧入口迁移。复现客户端为tools/verify_registry_graph.py，汇总为tools/summarize_registry_graph.py；机器可读总验收acceptance.json、算法接入表、拒绝记录、实际图、产品和指纹均在上述独立目录。


## 2026-11-27 旧融合入口统一到准备后Registry推荐与真实图变异（prepared-registry-search-v2）

本节承接上一节明确保留的旧入口缺口，保留全部历史结果。当前用户明确授权统一推荐与搜索，故覆盖旧队友交付文档“仅转换、不改搜索”的历史任务边界；没有重建页面、修改科学内核、扩充算法库或恢复已退役GOLEM。本轮后端仍是现有PortraitMutation，不冒称官方GOLEM。

### 原问题与最小连接

旧入口先以固定步骤得到推荐/批准，再把少数方法ID传入专用runner；公共Registry虽可见，却没有统一负责“准备后数据→DS显式父边图→数值执行→变异”。简单调用外层推荐会暴露尚未隔离的原始数据语义，且部分配对空间端口不等于原始网格端口。

现在registered_graph.recommendation_session建立请求内推荐上下文，apply_prepared_recommendation是唯一准备后消费者。平台文件入口recommend_and_submit、桌面selected_processing和工程workbench均进入同一上下文；任务适配器仍只做真实单位/窗口/格式准备，到了field_mutation或spatial_fusion_task再用隔离后的数据构造载荷。service.prepared_graph_provider只负责线上DS传输，公共服务不导入tools。

公共载荷包含全Registry发现、实际允许的端口、保留训练/条件观测摘要、配对目标位置及UNKNOWN。结构建议最多修正两轮；失败记录原因后继续合法开放搜索，不放宽最终能力、质量或独立复检。旧推荐ID/固定配方不再充当自主搜索的初始化白名单；新的DS图必须按真实父边执行、计分，后续仍由既有搜索变异。缓存仅限请求内且按完整无参考载荷指纹命中，作用变量受请求范围约束。

### 入口接线及证据边界

| 入口 | 当前连接 | 行为/数值证据与限制 |
|---|---|---|
| local、station_grid | 桌面处理→公共空间服务→准备后推荐→真实图搜索 | local真实NCEP旧处理函数完整执行；站点公共消费者真实数值功能测试 |
| pair_processing | 不再把旧weighted直接当最终融合；自主模式把两个真实来源送公共服务 | 共享准入/隔离测试；非自主历史路径保留 |
| windowed_grid_recipe | 对齐准备后去掉旧固定blend终点，转公共搜索；静态图不编造时间 | 新增静态公共入口及旧窗口入口数值测试；EPSG:4326、完整对齐连续栅格；原有瞬时及明确日累计契约复用 |
| indexed_table_recipe、independent_grid_search | 继承外层请求上下文，在已有公共数值入口消费 | 九路准入矩阵、原有联合抽样和工程桥定向回归；不是九套真实数据均重跑 |
| observation_gap_search、native_reference_search、original_engineering | 复用已有公共调用，新增共享请求内推荐上下文 | 原工程搜索桥数值测试和会话绑定测试；不等于所有模型程序已实际运行 |

静态适配static_raster_search只绑定现有面积重叠聚合、field_mutation、Executor及产品回读，没有新优化器或科学算法。它把原产品降尺度后作为明确的产品重建代理评价，参考不进入预测输入；无独立确认时保留基线，增强候选另存，不宣称环境精度提升。缺值、类别、未对齐来源、不同显式目标网格等不符合此代理契约时明确拒绝，不能强行造真值。区间平均或任意累计窗口缺合法评价适配时仍未开放；“入口统一”不等于所有科学任务已经支持。历史固定配方协议以内部prepared_graph_service=False保持复现，不新增页面选项；搜索准入SEARCH_ONLY与最终允许交付仍严格区分。

### 当前真实有界执行

证据目录output/legacy_entry_unification_20261127，最终线上案例real_local_v4。复用现有真实NCEP温度6时刻×6×6子集，通过桌面selected_processing→旧local批处理→公共服务，不直接绕过旧入口。输入SHA256为869f400eba80d4be6d3bb33f10c28600f06cca45c5d3b47e8932d1dabf97d427，前后不变。DS真实响应63736 tokens（62855输入、881输出），推荐阶段9.766秒，原始响应及拒绝信息保留。DS本次合法建议保持source:0；没有人为要求其必须推荐复杂图。

同一既有预算130次搜索调用/180总调用、100提议、45秒，seed=11；实际评价11图，其中7张变异候选（4通过、3失败），搜索算法节点调用73，部署2，验收审计12，共87次。搜索数值服务计时2.743秒，另列线上推荐开销。变异实际涉及Barnes、Cressman、PyKrige及带真实父边的估计→blend，已超出旧eligible仅含IDW的范围。最佳选择队列RMSE=2.136368 K、MAE=1.716690 K、覆盖率1；这是产品重建代理，非实测精度。本次独立冻结复检结论NO_STABLE_SELECTION_GAIN，服务标记BASELINE_RETAINED，不推荐为融合增强；本例仅一个真实来源，不用于证明多源收益。

real_local首次线上记录保留；v2客户端未解包scan_index返回值、v3客户端物理变量声明冲突，均在进入有效线上搜索前失败，保留日志，不作为独立重复试验或成功案例。最终客户端tools/check_legacy_registry_entry.py只提供真实数据、配置及回读检查，没有匹配规则或搜索实现。无2017数据、北京全量对照、949穷举或大型研究实验。

### 定向回归与剩余限制

最终final-targeted2.xml：75项通过，0失败、0错误、0跳过；仅新增及受影响测试。新增测试验证九路SEARCH_ONLY不冒充正式放行、旧入口实际消费准备后图、保留观测与隐藏标签隔离、API失败仍走合法搜索、会话异常清理，以及静态无虚构时间和产品回读。修复静态纬度降序，以及非自主历史静态路径被错误触发变异的回归；保留失败运行记录。旧NIST固定配方测试显式选择历史协议，新默认协议另有真实执行断言，未降低评分标准。

当前版本手工桌面上传/点击/下载未重新验收；上述是实际桌面处理函数和文件输出，不冒称人工界面验收。已打开进程需重启，不宣称热更新。全部99个available算法不等于本轮逐个数值执行；有数据/权重/垂向/协变量条件的条目仍需真实契约。源码快照、已知范围、API用量、实际图、轨迹、失败、成本及测试在本目录，历史证据不覆盖。

“没有执行出新的变异方案”的准确含义：预算内未完成一张此前未评价的新变异图的数值执行；原始算法被运行、命中缓存或只产生结构提议都不算。前轮时间案例仍保持该结论，本轮没有重跑它来挑好结果；本轮真实local空间例已执行新变异，但也不能据此宣称时间案例或全局最优已证明。


## 2026-11-27 有界多尺度评价升级（multiscale_review_v1）

按本轮授权直接更新当前公共服务，不改科学算法、不放宽质量门、原RMSE和独立复检标准，不覆盖历史结果。证据在output/multiscale_review_20261127；旧源码在before，新源码在source_final。此版本是十分钟内的有界工程升级，不冒称所有研究方法均已集成。

1. DatasetReference统一追加参考约束的梯度RMSE（每网格间隔）与归一化二维频谱形状差；所有候选使用真实同坐标/同单位参考。事件阈值明确时计算1/3/5像元尺度FSS；未声明阈值、网格不规则、站点表或数据不足明确标记。缺失像元不填零凑频谱，结构更锐利不自动加分。空间重建搜索在真实选优掩码内调用同一结构评价，不读取空间留出真值计算搜索特征。以上是额外诊断，尚未校准为综合选优权重；主排名仍沿用原误差Objective，不能称已解决所有细尺度选型偏差。
2. 公共空间日期划分改为整日分组，至少六个日期时尽量保留每阶段至少三个日期；日期不足仍允许处理，但不宣称跨日期验证充分。不再把同一天的小时记录拆进选优与确认。降尺度重建有足够日期时也在冻结后重新执行独立日期与独立空间块；不再只复用选优日期的空间复检。显式已有保留队列仍优先并检查冲突。
3. 原公共复检增加逐块基线/候选真实指标、优劣数量及全部范围记录，不仅给PASS/FAIL；局部退化仍如实记录，没有更换阈值。公共空间搜索冻结后同时评价最多两条质量合法的DS图，数值不反馈搜索。导出并回读baseline.nc与ds_recommendation_0.nc；失败的图不作为可用产品。service打包范围JSON及两份产品，现有结果卡展示选优和独立复检范围对照。接口支持不等于所有历史后台均重新人工验收。

真实验收：现有NCEP温度12日期×8×8子集，经桌面处理函数、线上DS、公共变异、复检及产品回读。9日选优/3日确认，无日期重叠。11张图、100次提议，共91次算法/验收调用。DS方案选优108块中40块误差较小、68块较大；独立复检12块中5块较小、7块较大。基线和DS产品均真实输出、回读有限值完整。仍保留基线，不宣称推荐整体更好或真实降尺度精度提高。本例只有一个真实产品来源，不是多源收益实验。原始全文件指纹不变；初次客户端未规范0..360经度被契约拒绝，保留失败记录，后续按已有经度表示规则变换，没有按误差挑地区或调参数。

定向验证：targeted2.xml为62项通过，display-tests2.xml另9项通过（有重复用例，不当作独立总数）；包括同场结构零误差、平滑损失、阈值缺失/单位冲突、隐藏空间真值不进入结构评价、整日隔离、公共评价实际消费者与Qt结果卡。原固定银行测试的10日样本选优数从40改为35，是留出3整日后7日×5站的必然变化，未改其评分合格条件。Qt离屏显示测试不冒称手工浏览器验收。

限制：实际NCEP空间选择掩码有洞，因此频谱按契约标记不可计算；只报告合法梯度诊断。降水FSS完成机制边界测试，本轮没有新增真实降水FSS实验。没有适用第三来源时不启用三重配准；没有细尺度参考依然不能认定全量细节真实准确。本轮未加入贝叶斯整批放行，也没有无限重推直到通过。日期与区域范围外仍为UNKNOWN；原生模型程序未运行。修改已覆盖运行目录；已经打开的旧进程仍需要重启加载。


## 2026-11-27 复用开源评价与参考多目标筛选（reference_pareto_v1）

本节修正前一版本“结构仅作诊断”的未完成部分，沿用公共搜索/执行/质量与独立复检。先明确：当前evidence_adaptive降尺度路径的主参考已是所有来源降级后恢复的原生较细产品，并非仅回聚到粗输入；coarse_diagnostics不参与其主排名。不同入口不能笼统等同为粗尺度一致性评分。原有产品重建仍非独立实测真值。

### 实际复用而非另写指标

直接安装并固定scores 2.7.0（https://github.com/nci/scores），平台调用其官方scores.spatial.fss_2d；同一参考网格、单位和事先声明阈值下评价1/3/5像元窗口，zero_padding=False。无阈值不计算，无事件不当作免费满分。独立接口测试通过调用拦截和逐值比对，证明使用的是官方库。原先自行拼装的uniform_filter版FSS已替换。频谱目前仍是既有NumPy二维归一化频谱诊断，不冒称已接pysteps RAPSD。pysteps无本机二进制轮子，源码安装先遇Windows编码问题，第二轮构建依赖未完成即停止；没有修改其科学源码或声称其已成功运行。scores通过纯Python轮子安装成功，没有为单个数据包写接口。

复用已有pymoo 0.6.2的NonDominatedSorting，在公共reliable_selection.reference_frontier对同一真实参考队列上的误差、梯度误差及有充分共同证据的频谱/FSS作非支配筛选。不同量纲不靠手填权重凑综合分。未知结构、不同坐标/掩码指纹和不完整指标不计为零收益，也不将算法直接判非法。帕累托集保留真实取舍：有的误差更低，有的结构更接近参考，不能凭一个RMSE把前者称为全面最优。

### 真实消费者与输出

公共空间搜索在原误差等价集内优先保留非支配方案；在DS父代用完后，按稳定图身份给已真实评价的结构备选一次变异机会，仍受原预算控制，记录structure_parent_keys。新默认为reference_pareto_v1；显式rmse_v1保留原模式。空间与通用场图结果均提供reference_frontier，结构不再只写日志。通用场图后端本轮增加了共同的终端集合筛选，尚未声称所有优化器都改造成原生多目标优化器；现有误差Objective仍负责其内部搜索。

选择后冻结误差优选及梯度结构优选，在同一独立确认队列上评价，并保留structure_preferred.nc、baseline.nc及合法的DS产品，质量失败者不导出。默认交付仍服从原独立关卡；若有取舍，页面明确标为“默认误差方案；另有结构优选”，不说综合最优。scope_reports.json通过公共审计打包路径提供，避免将报告JSON当作输入数据表。没有更低RMSE抵消硬约束、奖励多源、利用确认误差重新挑图或降低检验标准。

### 有界真实验收与负结果

证据目录output/open_source_evaluation_20261127，最终数值目录real_ncep_v3，原输入NCEP12日×8×8不变，9日选优/3日确认。共有11图、100提议、94次实际算法/审计调用；结构备选2c053fcc74e3d7f35328和a0b17d051bb8095d5541实际进入父代队列，已保存对应变异边和数值反馈。

| 方案 | 选择队列RMSE（K） | 梯度误差（K/网格间隔） | 解释 |
|---|---:|---:|---|
| Barnes误差优选 | 2.222185 | 2.502740 | 较低数值误差 |
| Universal Kriging结构优选 | 2.684560 | 2.431319 | 梯度更接近产品参考，但RMSE较高 |

这两者都在非支配集合，分别导出且回读通过，不宣称后者整体胜出。第一次线上DS已真实返回；后续修复打包和最终数值验收只在完整请求SHA相同时回放同一个线上响应（新API调用0），每轮数值执行均重新运行，不伪装为新的线上建议。首轮报告被当作表格导致下载链失败的日志保留，已迁移到公共审计打包并通过完整处理函数重测。最终数值后的少量排除别名、输出说明补充已另做边界测试，当前版本对冻结真实评价行重算的前沿与数值运行一致。

另用保存的真实降级重建预测数组验证scores，固定273.15 K阈值（物理冻结点约定）。Barnes在1/3/5像元FSS分别为0.898182/0.959641/0.968527，DS来源方案为0.886861/0.961495/0.971128。不同尺度有取舍；该记录是已暴露历史场的事后完整场库接口检查，没有参与新选优，也不当独立确认或降水收益证据。

final-tests2.xml：75项通过；display-tests.xml另14项通过；final-smoke.xml另11项通过，各集合存在重复，不能相加当独立测试数。覆盖官方函数实际调用、多目标集合改变、结构父代真实消费者、UNKNOWN不占便宜、候选顺序不影响结果、公共打包、隐藏参考隔离及受影响空间/平台回归。尚未重新人工桌面上传验收，不用Qt/处理函数代替人工验收。

边界：没有细参考仍无法证明细尺度真实精度；没有适用第三来源不启用三重配准；未接影像专用QNR；未引入收益学习、大模型训练、贝叶斯整批放行或正式算法优越性对照。库接通不等于提供了不存在的评价证据。当前代码已覆盖运行目录，已打开旧进程需重启。历史结果、协议和源数据不覆盖。


## 2026-11-28 队友更新兼容合并（WRF/贝叶斯抽检＋中间尺度评分）

以da7128d为本机基准，先保存backup/before-teammate-sync-20261128，再快进到d23f0d4（包含58c2aa2贝叶斯更新），合并cf713f6中间尺度更新。未强推、未删除既有功能和历史结果。读取现有AGENTS及队友接入说明；本轮用户明确授权同步这些功能，故不受旧“仅转换任务”历史范围限制。

唯一文本冲突位于desktop/output_review.py：旧版写“误差与梯度结构取舍”，中间尺度分支写“不同空间指标取舍”。合并为“默认交付方案；另有多尺度优选”，说明误差、结构或中间尺度存在取舍且不能称全面最优。保留UNKNOWN处理、真实父边、独立留出、双/多方案产品及开源scores/pymoo消费者。

检查边界：新版贝叶斯整批准入与保留的传统字段检查分属不同协议；不能把某个固定配方抽检通过继承给另一张任意图。新图仍需独立数值执行和最终验收。固定抽检模式、准备后Registry图模式分别按请求适配，不宣称每个入口都同时运行全部协议。中间尺度评分只用本阶段完整参考单元；无中间网格或缺依据保持未评价，不用粗聚合误差替代细尺度精度。WRF新增单时刻多站补缺路径有文档约定的命名、变量、投影和时间输入条件，不能泛称所有WRF工程均支持该任务；融合可执行与优于单源分开。

合并后181项新增/受影响定向测试通过，0失败/错误/跳过。涵盖WRF补缺、站点两阶段推荐、贝叶斯后验与重新推荐、正式抽检、共享Registry推荐、隐藏参考隔离、多尺度与中间尺度、CAMx/候选选择/自动继续和Qt相关处理函数。本轮没有付费线上DS调用，没有重跑原CAMx/Saguling或大型研究实验，也未人工点击桌面整轮；测试通过不是任意数据均能运行或精度提升的保证。

同步前逐项核对47个改动路径：运行目录与旧主版本一致，没有覆盖额外本地修改。运行目录旧文件保存于output/teammate_sync_20261128/runtime_before；合并代码通过字节比较同步到D:/codex/EcoDataEngine，测试从合并后的Git目录源码运行。新进程会读取新代码，已打开的进程需要关闭重开。同步清单、源码SHA256和JUnit在本轮验收目录。
