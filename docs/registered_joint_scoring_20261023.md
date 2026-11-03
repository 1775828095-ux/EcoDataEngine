# 版本化属性注册与三表评分参数校准

## 当前选择政策

公共`submit_search(..., mode="Profile")`使用`joint_portrait_v1`。顺序为：算法/资源与语义筛选→端口组合及硬需求验收→三表联合评分区间下界→声明成本→科学节点数→稳定键。

旧PlanRankingService的指标、排名预测和训练观测可以显式加载查看，但`learning_status.affects_selection=false`。旧`task_rank`和`development_fixed`配置也不再替代三表评分或启用固定回退。旧模型及历史结果不覆盖。

有参考时：真实执行、质量检查与明确评价阈值仍决定产品能否交付；`best`是联合评分顺序中首个通过的候选。`recommended_key`在执行前冻结；`hindsight_best`仅为已执行候选的事后误差最优诊断，不作为推荐成绩。默认无参考只推荐，不执行最终候选、不验收精度；显式启用station_probe属性时另做训练站分折测量，次数和成本单列，仍不接触独立参考。

没有声明软评分偏好时，系统明确返回`RULE_COST_ORDER_NOT_ACCURACY`；不会根据模型名字编造敏感性或默认为已有精度排序。

## 属性注册表

默认定义位于`src/ecodata/algorithms/portrait_attributes_v1.json`。新增版本由平台公共API发布到`data/attribute_registries/<version>.json`，使用排他创建，不能覆盖。每个快照记录版本、父快照SHA、创建时间、依据和全部定义；加载核对SHA和schema。

每个属性包含：

|字段|作用|
|---|---|
|id、definition、unit|稳定标识、操作定义和单位|
|tables|涉及data/algorithm/model中的哪些表|
|role|scoring或diagnostic；诊断项不能偷偷参与评分|
|scoring_enabled|是否参与软评分；关闭不取消硬要求|
|hard_allowed|是否已有可用的硬保证消费者；与软评分启停独立|
|provider、property、activation|真实测量/保证消费者、最终图性质、数据激活方法|
|evidence、history|来源依据和追加修订理由|

当前可配置消费者为：

- `guarantee`：复用现有五项已审核保证及其作用域，沿真实图连接传播；可以注册有清晰定义的别名，不自动创造新物理保证。
- `declared_cost`：从实际科学节点累加已有声明成本等级，用`-cost/(1+cost)`映射为有界效用；这不是实测耗时或准确率。
- `unavailable`：注册尚无测量器的属性，保证与点分保持未知，不用人工猜值补齐。

初始五项硬保证词汇不能删除或改成无效消费者。可选属性退出评分时保留定义，设`scoring_enabled=false`并追加原因，形成可恢复的停用记录。新增版本不得移除父版本已有记录。

现有消费者的属性可以配置增删/启停；全新科学测量仍须实现和验收消费者。注册表不接受任意Python表达式，不把新增一行配置当作识别能力已经实现。

## 使用公共入口

已有完整请求可增加：

```json
{
  "scoring": {
    "registry_version": "builtin-v1",
    "parameters": {
      "continuity": {"multiplier": 1.0, "activation_threshold": 0.0}
    }
  },
  "model_requirements": {
    "priorities": {"continuity": 0.6, "no_mixing": 0.4},
    "hard": ["range_preservation"],
    "evidence": "本次明确声明的字段准备要求；不是推断WRF敏感性"
  }
}
```

这只是请求增量。仍需真实输入、任务、变量/单位/语义、模型角色/字段、目标坐标和候选池。示例权重仅说明接口，不是已验证科学权重。

发布新属性版本：

```python
from copy import deepcopy
from ecodata.algorithms import attribute_registry
from ecodata.algorithms.platform_search import publish_attribute_registry

definitions = deepcopy(attribute_registry.load()["attributes"])
item = next(a for a in definitions if a["id"] == "continuity")
item["scoring_enabled"] = False
item["history"].append("本次停用软奖励的明确理由")
publish_attribute_registry("reviewed-v2", definitions,
    parent_version="builtin-v1", evidence="本次审查/开发证据来源")
```

之后页面JSON、工具包及文件客户端统一指定`scoring.registry_version="reviewed-v2"`。原版本继续可用；不能同ID覆盖。停用连续性加分不会允许一个不满足硬连续性要求的候选通过。

## 联合评分与参数

每个项仍有明确的下游权重、可见数据激活、实际组合性质和依据。保留原有有界符号适配定义，新增参数不改变保证真值：

`term = normalized_user_priority × multiplier × activated_risk × signed_suitability`

`multiplier`范围[0,2]，默认1；`activation_threshold`范围[0,1]，默认0，低于阈值时该软项不激活。用户原始优先级与评价指标权重不被校准改写，实际有效权重和参数写入逐项轨迹。

未知保证/未知激活产生上下界和`contribution=None`；总点分为None。零激活、参数零化和属性停用明确标为INACTIVE或DISABLED，不代表未知性质被测成零风险。区间不是统计置信区间，“无保证”不等于“实测误差更大”。

组合读取最终状态，不把每个节点的优点累加。硬约束独立检查，不参与参数搜索；数值回读继续复核非负和值域等可检验条件。

## 参数校准

平台公共API：

```python
from ecodata.algorithms.platform_search import calibrate_profile_scoring

report = calibrate_profile_scoring(
    development_run_folders,
    "reviewed-calibration-v1",
    holdout_groups=["held-development-group"],
    registry_version="builtin-v1",
    parameter_grid={
        "continuity.multiplier": [0.5, 1.0, 2.0],
        "nonnegative.activation_threshold": [0.0, 0.1]
    }
)
```

仅可调已声明、已启用、具有测量/保证支持的项。不是每个示例参数都适用于任意开发记录；不请求该项、没有支持或使用不同注册版本会拒绝。

输入必须是显式分组development、完整合法候选池的真实执行记录，至少两个训练组和一个分离留出组。所有候选需执行/回读成功，产品SHA、逐候选result.json及按冻结指标重新计算的task_loss均需吻合；不静默删除失败候选去制造完整标签。相同来源内容不能跨组，重复记录不增样本。

校准要求明确RMSE/MAE/abs_bias评价偏好，不接受含义随评价器变化的通用objective。已有规范五维历史记录可按其真实字段适配；不能借旧记录填造新增属性。

对已有候选损失矩阵重新计算同一个评分器的选择遗憾，按训练组等权平均选择参数；并列优先保留接近初值的参数。最多六个参数轴、256个网格点，另保留初始参数。校准本身实际图执行次数为0，不训练替代选择器。

留出组只评价冻结后的参数，不选参数。报告保存全部试验、选择键、来源指纹、训练/留出结果、未知数量、数据激活方差及任务内贡献区分度；这些诊断不自动删除属性或修改硬规则。最低组数是工程门，不是样本充分性或泛化保证。

输出到`output/score_calibrations/<id>.json`，不可覆盖。使用时：

```json
{"scoring":{"registry_version":"builtin-v1","calibration_id":"reviewed-calibration-v1"}}
```

校准ID与直接parameters不能同时指定。加载检查指纹，并精确核对属性版本、候选池、模型角色/字段、目标网格、硬软需求和指标偏好；不匹配明确报错，不悄悄使用旧参数或另一套学习排序。当前范围只表示同一声明上下文，不能宣称跨域效果有保证。

## 端到端验收及限制

新增`tests/test_registered_joint_scoring.py`实际覆盖：属性启停改变数值产品、注册新消费者与未知项、可选属性停用后恢复原排序、硬要求在软项关闭后仍筛选并构建投影、权重变化改变同一数据的组合连接、校准矩阵重用、留出隔离、校准版本加载到页面/文件数值执行、旧学习逆向排序不能替代规则、指纹与非法证据拒绝。

本轮最终累计128项不同定向/回归测试通过，包括既有协作、Profile、P1/P2；重复运行不累加。主要XML为`output/registered_joint_release.xml`、`registered_joint_final.xml`、`registered_joint_clients.xml`、`registered_joint_retirement.xml`、`registered_joint_p1.xml`、`registered_joint_acceptance.xml`及最终政策检查`registered_joint_final_policy.xml`。

独立回放工具`tools/audit_registered_joint_scoring.py`不调用评分器本身，重新计算保存的逐项区间、权重、排序、硬过滤和首个合格交付，另核对产品SHA、有限性、单位和执行指标记录。27份决策记录与46份产品全部通过，汇总为`output/registered_joint_independent_audit.json`。最终源码指纹、去重测试及限制记录在`output/registered_joint_release_acceptance.json`。

所有新增开发/留出和数值案例均是合成功能夹具，不发布为生产校准参数，不作为环境精度优势、完整WRF运行或跨域泛化证据。没有安装新MCDA框架、下载数据或运行正式效果实验。自动发现新物理属性/自动写测量器、完整相关性/因果归因和线上自更新权重不在本轮完成范围。


## 多源实验的实测评分扩展（2026-10-23）

新增可选`station_probe`消费者，注册指标仅限RMSE、MAE、abs_bias，不允许作为硬保证。它在公共Profile入口对显式提供的8—100个可用训练站做固定三折GroupKFold，每折都通过公共图执行器运行同一候选图，留出站标签不进入该折拟合端口。测量失败保留原因并产生未知评分，不缩减成功子集、不重抽有利分折。未选择这些属性时不产生探测成本。

对图g、指标k，实测误差为e(g,k)，可用站浓度标准差为s，效用为`-e/(max(s,1e-6)+e)`；声明成本效用沿用`-c/(1+c)`。同一个`parameterized_terms`用任务声明权重、属性乘数计算联合分数，原硬约束不变，开发阶段仍只校准原公式。它是基于当前可用站局部交叉验证的实测适配评分，不是零样本画像预测、另训练的排序器或独立测试精度保证。

`portraits.station_probes`按完整图键记录原始误差、标准差、分折和预测；`station_probes.json`记录可用训练值、失败与耗时，`probe_costs.json`及summary.counts.training_probe_graph_folds单列成本。无独立参考的recommendation-only也会产生这些训练探测，`real_executions=0`仅指最终候选执行为0，不代表总计算为0。

`fusion.station_rf_background/weather/joint`是既有站点分组RF数值核的三个特征配置，复用同样50棵树、深度6、最小叶3、种子11，并非三种新科学算法。GEOS与ERA5保留各自物理定义、单位和时间窗，仅作为协变量学习站点目标；主输入中的零场明确标为geometry prediction_template，不能当观测或用于直接浓度融合。本轮不使用模板参与逆方差融合。

为避免重复开发，EPA任务和新适配器共用feature_forest.py；EPA保留原站点OOF残差行为，新即时估计不重复计算内部OOF，其选择证据来自外层训练站三折。新适配器可与已有GP/普通克里金残差估计、背景叠加、非负投影连接。

已修正：残差叠加不再枚举无效interpolation参数；IDW不作为独立背景再校正；同小时多源基线保留标量time；普通克里金不再为未使用的协变量检查满秩（通用/回归克里金继续检查）。来源指纹包括实际可见观测与协变量，避免相同空模板导致不同遮站任务被校准器误去重；校准上下文固定特征组与GP参数。
