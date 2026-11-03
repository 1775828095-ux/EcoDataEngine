# 三张画像表与短组合选择

本模块是平台公共入口的显式 `Profile` 模式。它复用模型需求、算法契约、端口语法、真实执行器和质量门，默认三个科学节点，明确请求可扩至六个，不调用进化优化器。历史 GOLEM 模式保持原行为。2026-10-23起采用joint_portrait_v1：属性注册与参数校准见[当前接入说明](D:/codex/EcoDataEngine/docs/registered_joint_scoring_20261023.md)，旧学习模型仅供诊断。

## 初始字段与证据

字段定义由人审核，单位和物理语义来自可靠元数据/已审核模型配置，统计量由程序计算。画像不是一份人为填写的性能排行榜。

|表|当前字段来源|未知项|
|---|---|---|
|数据|来源指纹、DataSemantics的结构/类型/单位/时空支持/缺失统计、现有12项栅格特征、处理历史|噪声真假、动力合理性没有依据则未知；现有栅格特征不适用含缺失或非二维输入时不填造|
|算法|AlgorithmSpec的ID/版本/能力/输入输出/资源/倍率/声明证据；保留粗成本等级|未验收的副作用和经验成本为None，不能用粗成本推断精度|
|模型|RequirementStore中的七个版本化配置、所选角色/字段的准备契约、来源、显式应用权重和阈值|敏感性为None；剩余角色明确列出；单字段产品不等于完整模型输入|

七个模型为CALPUFF、WASP、传统SWAT 2012、WRF、CAMx、CMAQ、WRF-Chem。排除EPICC。SWAT不替换成SWAT+。配置中尚未有明确字段/单位的角色会拒绝，而非推测。没有覆盖原生文件和原生程序的验收缺口。

每次实际科学节点输出都重新提取语义/缺失状态，写入端口metadata.stage_portrait及节点轨迹；下游继续使用真实父产物，并重新执行既有compatibility。画像记录不替代科学契约。

## 匹配与评分

1. 选择一个模型、输入角色和已审核字段；确认单位、物理变量与类型，冲突拒绝。只自动提升1h/1D这样的机器可读需求，不从自由文字猜频率。其他时间/空间目标继续按明确请求传入。
2. 复用公共能力映射和实际端口配方，逐图做require_graph最终验收；合法生成范围受既有任务适配器限制，不能任意串联注册表所有算法。
3. 默认最多64个有界配方、三个科学节点，显式保存截断，不宣称全局穷举。指定model_requirements后按三表规则排序；无声明偏好时明确按成本排序，不用学习模型替代。模板别名可能共享数值行为，不视为独立科学方法。
4. 有参考或明确工程指标时，每张候选完整图真实执行并读回产品；按三表顺序交付首个通过质量门和明确阈值的产品。J=sum(normalized_weight[k]*loss[k]/scale[k])用于评价和开发校准，abs_bias使用绝对偏差；误差最优另存hindsight_best，仅作事后诊断。缺必要指标报错，覆盖率不冒用RMSE。
5. 无评价资源时只给推荐，执行次数为0，不隐式替换成覆盖率，也不输出虚假的最佳数值图。
6. 可选PlanRankingService继续保留旧指标与名次预测、画像观测及支持域检查，全部只作诊断；预测、旧默认策略及固定回退均不能改变联合评分的排序和交付。参考、评价损失和文件名不进入诊断特征。
7. 新排序模式与旧AOD单阶段收益服务分开。旧模型和历史证据不修改；不能把整图差异伪装成一个节点的收益。

## 页面与工具包

页面选择“模型需求与画像 · 短组合有限枚举”，选择七个模型之一及角色/字段。单位与变量必须真实匹配上传文件。页面处理函数和工具包都调用platform_search.submit_search。

已有请求JSON增加：

```json
{
  "mode": "Profile",
  "request": {
    "model_profile": {
      "id": "wrf_chem",
      "role": "wrf_initial_boundary",
      "field": "air_temperature"
    },
    "preferences": {
      "weights": {"RMSE": 1.0},
      "scales": {"RMSE": 1.0},
      "thresholds": {},
      "tie_tolerance": 0.0
    },
    "model_requirements": {
      "priorities": {"continuity": 1.0},
      "hard": ["range_preservation"],
      "evidence": "本次任务要求连续插值且禁止超出输入值域；这是任务声明，不是默认WRF-Chem敏感性"
    },
    "profile_options": {"max_steps": 3, "candidate_limit": 64}
  }
}
```

这只是增加字段的片段，不是完整可运行请求。还需实际输入、任务、变量/单位/语义和目标坐标或频率。实测RMSE必须有合法验证参考；无参考可把RMSE作为学习模型的预测目标，但不生成实测误差或通过精度阈值的承诺。尺度1.0在此仅为格式示例，不是推荐权重或容差。

输出：request.json、portraits.json、candidate_ranking.json、逐候选产品/轨迹、summary.json。无参考推荐模式不生成产品。完整可运行真实数据请求位于output/profile_matching_20261022/real_request.json；原目录不可覆盖，应指定新输出目录。

## 旧诊断模型的训练与边界

显式调用平台公共API `platform_search.train_profile_matching(run_folders,model_id,holdout_groups=[...])`，复用PlanRankingService。请求通过profile_options.evidence_role="development"、group="独立来源组"积累记录。只接收完整跑完该有界候选池、产物指纹匹配的记录，标签与逐候选result.json交叉核对并按记录权重重新计算task_loss。相同来源内容不能跨组，重复客户端不增加样本。至少两个训练组与一个分离留出组，所有预处理只在训练部分拟合；这只是最低工程检查，不是统计充分性保证。

同时输出不可覆盖的model.joblib、report.json和portrait_evidence.json。后者仅从训练组填充观测，不把整图效果归因到单节点或将留出组写入画像。装载时核对模型、画像证据SHA及版本；未见组合/上下文或超范围返回None。支持或弃权只影响诊断字段，均不改变三表评分。新的规则参数校准使用calibrate_profile_scoring，不能用这套随机森林训练替代。

## 三表评分的实际消费者（portrait_rules_v3_registered_joint）

`portrait_rules.score_graph`直接消费portraits.json中的data.matching_facts、algorithms.matching_properties、model.matching_requirements，并沿实际图的父连接传播性质，不相加节点分数。只认可实现类白名单及其数学性质，不根据算法名字猜性质。

|需求维度|依据与范围|数据激活强度|
|---|---|---|
|range_preservation|有限二维规则网格内部：最近邻复制原值、双线性为凸组合；其他算法未知|V=平均相邻绝对差/输入值域，常数场为0|
|continuity|同上范围：双线性连续；非恒定场最近邻不保证连续；裁剪连续函数保持连续|V|
|no_mixing|定义为复制输入样本值：最近邻有保证；双线性不提供此保证；负值裁剪会撤销该保证|V|
|nonnegative|上述凸组合在原场非负时保持非负；max(x,0)和QP有输出保证|可见输入负值比例|
|coarse_consistency|实际QP对明确提供/构建的观测算子约束并核对残差；不是物理总质量守恒|1|

priority权重归一化。每维贡献=权重×激活强度×(+1有保证，-1无保证)。这是结构适配的操作定义，不是实测误差、概率或科学收益。未知性质保留[-权重×强度,+权重×强度]，未知激活则保留[-权重,+权重]；点分为None，不伪填0。零激活明确标INACTIVE。按区间下界保守排序，再按成本和步骤数破同分；区间不是统计置信区间。每个贡献保存数据字段、模型任务声明、父连接及实现/数学依据。

hard中的保证必须被证明，否则候选在执行前拒绝（包括未知），最终数值回读后再次检查非负/值域；连续性/复制原值由实现契约证明，QP自身验证残差。硬coarse_consistency会进入已有require_coarse_consistency映射，生成算子资源并筛出实际投影组合；缺少合法空间支持时拒绝，不猜守恒语义。

组合输出状态按父节点传播：投影不自动继承未知父性质，不重复奖励同一性质，不假定正协同收益。完整执行后，用同次已执行且合格的单算法最小task_loss减去组合task_loss记录gain；没有可比较单算法则None。只对参考误差任务标记准确率口径，工程覆盖收益不冒充精度改善。

运行中间输出继续写stage_portrait及stage_matching_facts，作为真实状态证据；当前有界候选图预先生成，不宣称已经实现执行中动态重规划。未评审算法、噪声/动力/峰值性能仍为未知，可通过后续分组真实记录补充上下文观测，不自动晋升为通用保证。历史画像由显式加载的新版本模型接入，未加载时不静默使用临时测试权重。

## PCHIP、三次插值、冻结池与无参考推荐

画像schema为portraits_short_plans_v3_reference_free；旧版本模型保留，不静默转换，需从合法开发记录重训新版本。

- PCHIP：依据现有ShapePreservingRaster和SciPy PchipInterpolator实现，在有限、轴递增的二维网格内部，各一维保形阶段不过冲；连续性、值域保持以及源场非负时的非负保持进入实际评分。只声明二维连续，不声明各向同性二维C1或未知细峰恢复。
- 三次插值：依据现有XarrayAggregationPlugin及xarray-regrid→xarray.interp(cubic)，有限二维内部且每轴至少四个样本时给出连续插值保证；值域不提供保证，非负保持仍未知。噪声、峰值、数值精度无实验证据仍为None。
- 范围外、轴或点数条件不满足时，不套用内部性质。评分沿实际_port_bindings读取source0/field，不能把QP算子/粗观测资源当作上游预测。

按本轮空间方案冻结五个组件可在request中使用：

```json
{
  "requirements": {"try_coarse_consistency": true},
  "profile_options": {
    "max_steps": 2,
    "candidate_limit": 64,
    "candidate_pool": {
      "algorithms": [
        "raster_task.spatial.nearest",
        "raster_task.spatial.bilinear",
        "raster_task.pchip",
        "raster_task.spatial.xarray_cubic",
        "raster_task.coarse_consistency"
      ]
    }
  }
}
```

这是补充字段，不是完整请求。需合法完整二维粗场、明确平均支持、目标坐标及覆盖外边界，使公共入口能构建粗观测算子。只提供算法池而缺必要资源会阻止提交，不静默更换候选。不要为原始支持UNKNOWN的数据伪造官方像元平均语义；人工退化平均只能标在实验派生粗场。

冻结在生成前过滤实际适配器，使用原有类型语法生成四种单方法及各自接QP的八图；硬coarse_consistency再筛为四图。插值原生ID和raster_task包装统一到显式最近邻边界延拓的科学方法，保存requested/effective/aliases。此规范化不是声称二者域外行为本来相同。PCHIP保留其现有坐标裁剪边界规则。不会通过输出数值碰巧相同而合并不同方法。

candidate_pool.json保存规范池、版本/实现指纹、别名、边界规则、语法/合法数量；相同规范池的别名次序和种子不改变候选集合。达到生成cap时拒绝，要求显式提高上限；不以部分枚举冒充完整冻结池。缺省不设candidate_pool时保留历史有界池行为。

学习特征只取可见数据/算法、明确源与目标网格/频率/语义和冻结池；不包含评价类型、评价指标名称、参考指纹或数组。RMSE/MAE/abs_bias指标头不依赖是否提供当前参考。目标网格/池或数据范围未被训练覆盖仍弃权，未放宽支持门。

objective保留独立标签语义：reference_error/RMSE与declared_task_metric/coverage不能混训，预测时仅请求匹配的objective头；无参考recommendation_only不返回这个泛化objective。以RMSE/MAE/abs_bias作为无参考预测目标即可使用合法训练模型：

```json
{
  "preferences": {"weights": {"RMSE": 1.0}, "scales": {"RMSE": 1.0}},
  "profile_options": {
    "ranking_model_id": "实际已发布的新版本模型ID",
    "acknowledge_limited_development": true
  }
}
```

与前面的冻结池字段合并，不要丢失其余profile_options。页面和submit_file_request均不传evaluation即可。返回recommendations及可用的诊断预测；best=None、real_executions=0、actual_accuracy=NOT_ASSESSED。模型不支持时诊断为None；推荐始终由三表评分/明确成本顺序决定，覆盖率不能满足RMSE实测目标。

使用排序模型需profile_options中显式ranking_model_id与acknowledge_limited_development=true。当前没有发布通用可信模型。测试中的训练仅验证实现，临时功能测试模型不作为环境精度证据。

## 借鉴来源与复用范围

- sklearn官方仓库https://github.com/scikit-learn/scikit-learn，BSD-3-Clause；复用项目既有安装的DictVectorizer、RandomForestRegressor，不复制或自行实现随机森林，不拉取main覆盖环境。
- Ranking Trick论文https://proceedings.mlr.press/v293/vazquez25a.html：借鉴排名标签目标，不声称复制了论文作者的完整框架。
- AutoFolio https://github.com/automl/AutoFolio：参考特征/性能矩阵设计，未复制其代码、未安装旧依赖。
- 本地capability_scoring.py复用证据管理思想与现成工具，旧AOD模型权重不用于新排序。
- TypedGolemFactory复用配方和图表示；NumericalDAGExecutor、UnifiedAlgorithmPipeline、RegistryEvaluator复用实际计算和最终验收。OptGraph只是既有图容器，本模式evolution_used=false。

## 尚未覆盖

专用hourly_recovery及站点/协变量状态端口暂走原公共入口，Profile模式会明确拒绝，不能宣称新排序已覆盖这些路径。模型原生验收、动态敏感性学习、所有空间缺失形态特征、所有任意算法串联、可信跨区域排序与精度优势均未完成。早期接线轮不运行正式遮挡对照；后续有界实证须按各轮冻结协议解释。

## 历史记录：2026-10-23 扩展实验与旧模型策略

以下记录解释旧实验，不是当前选择政策。joint_portrait_v1已禁止这些预测顺序或固定回退替代三表评分；支持域只影响诊断。应用偏好变化后，旧任务内排名诊断也保留None，不跨目标解释旧名次。

本轮限定模型`spatial_next_release_20261023`复用2013年NCEP开发记录训练，属于LIMITED_DEVELOPMENT，不是通用气象排序模型。数据与验收范围见`spatial_next_protocol_20261023.md`及相应结果报告。

页面JSON与工具包仍设置`profile_options.ranking_model_id`和`acknowledge_limited_development=true`，同时采用对应冻结候选池和明确的RMSE偏好。模型报告中的default_policy自动供公共入口采用，无需页面或实验客户端另写排序规则。旧模型没有default_policy时保留原metric/rules行为。

- `learning_order=metric`：按预测指标加权损失排序；`task_rank`：复用已有任务内排名头。后者要求训练/推荐的应用偏好指纹完全相同，不能在改变目标后沿用旧排名。
- `fallback_policy=development_fixed`：仅在已知上下文内，采用拟合期真实效果最好的合法固定参考；状态为DEVELOPMENT_FIXED_FALLBACK，不算学习命中。未知上下文没有该参考，仍使用规则回退。没有预测就保持predicted_metrics=None。
- support_diagnostics公开匹配样本数、拒绝原因和逐项数值边界。逐特征支持域与整池支持条件未放宽。
- `publish_profile_policy`保存新的不可覆盖模型ID，复制并核对既有权重/完整画像证据，以已记录验证文件的路径和SHA记录策略来源，不覆盖原模型或重新拟合审计标签。
- 完整历史证据仍在模型文件中；响应最多摘录每列表16条，同时给出history_index路径、SHA及总数。摘录只影响展示与存储，不影响训练、支持检查、规则和评分。

新增IDW、MetPy线性、普通克里金的栅格适配器，内部将可见粗均值放在中心作为点值代理再调用已有数值内核，外部不需真实站点数据。该近似及最近邻边界扩展明确记录；不宣称它们已成为严格面积支持的块克里金，也不把不存在的环境权重、DEM或协变量补造出来。
