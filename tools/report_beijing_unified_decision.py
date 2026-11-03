"""Print one evidence-backed chapter for appending to the existing report."""
import json
import pandas as pd
import run_beijing_unified_decision as u


def table(frame):
    def cell(v):
        if v is None or pd.isna(v):return "未评价"
        if isinstance(v,float):return f"{v:.4f}"
        return str(v).replace("|","/")
    return "|"+"|".join(frame.columns)+"|\n|"+"|".join("---" for _ in frame.columns)+"|\n"+"\n".join(
        "|"+"|".join(cell(v) for v in row)+"|" for row in frame.itertuples(index=False,name=None))


def main():
    read=lambda name:json.loads((u.OUT/name).read_text(encoding="utf-8"))
    acceptance=read("final_acceptance.json");assert acceptance["status"]=="PASS_FOR_BOUNDED_TEMPORAL_CHAIN"
    runs=pd.read_csv(u.OUT/"search_runs.csv");fixed=read("fixed_validation_results.json");test={r["key"]:r for r in read("test_results.json")["results"]}
    costs=pd.read_csv(u.OUT/"factory_generation_accounting.csv");guide=pd.read_csv(u.OUT/"guidance_consumption.csv")
    summary=runs.groupby("mode").agg(n=("seed","size"),val_J=("best_validation","mean"),val_SD=("best_validation","std"),
        followup_J=("test_objective","mean"),followup_SD=("test_objective","std"),followup_RMSE=("test_point_RMSE","mean"),
        scored=("scored","mean"),real=("real_executions","mean"),valid=("unique_valid","mean"),
        proposal_legal=("structural_proposal_valid_fraction","mean"),real_quality_pass=("quality_pass_fraction_of_real","mean"),
        search_seconds=("seconds","mean"),setup_seconds=("platform_setup_seconds","mean"),total_seconds=("platform_total_seconds","mean")).reset_index()
    baselines=[]
    for name,r in fixed.items():
        t=test[r["key"]]
        baselines.append(dict(method=name,validation=r["status"],val_J=r["objective"],val_RMSE=r.get("RMSE"),val_MAE=r.get("MAE"),
            val_Bias=r.get("Bias"),val_coverage=r["coverage"],followup=t["status"],followup_J=t["objective"],followup_RMSE=t.get("RMSE"),followup_coverage=t["coverage"]))
    pairs=pd.read_csv(u.OUT/"paired_environment_intervals.csv")
    judgments=[]
    for other in ["A","B","C-gap","R","linear_nearest"]:
        rows=pairs[pairs.a.eq("C") & pairs.b.eq(other)]
        if rows.empty:
            judgments.append(f"C相对{other}：存在无解/失败策略，不能只对成功子集给出整组精度区间。")
            continue
        r=rows.iloc[0]
        judgment=("本探索性队列上的配对J更低" if r.ci95_high<0 else
                  "本探索性队列上的配对J更高，未获得优势" if r.ci95_low>0 else "区间跨零，稳定优势未确立")
        judgments.append(f"C相对{other}：平均J差{r.objective_difference:+.4f}，95%区间[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]；{judgment}。")
    perseed=runs[["mode","seed","scored","real_executions","unique_valid","best_validation","test_objective","seconds","platform_total_seconds"]].sort_values(["mode","seed"])
    enum=read("enumeration_summary.json");passing=[r for r in enum["results"] if r["status"]=="PASS"]
    total=dict(runs[["scored","real_executions","quality_failures","incomplete_feedback","callbacks","graph_cache_hits","deferred_callbacks","proposals","proposal_rejections","posterior_rejections"]].sum())
    nonuniform=guide.groupby("mode").nonuniform.agg(["sum","count"]).reset_index()
    resource=read("resource_summary.json");old=pd.read_csv(u.OLD/"search_runs.csv")
    fit_cost=read("training_fit_cost.json")
    comparison=[]
    for mode in ["A","B","C","C-gap","R"]:
        before=old[old["mode"].eq(mode)];after=runs[runs["mode"].eq(mode)]
        comparison.append(dict(mode=mode,old_local_val_J=before.best_validation.mean(),new_val_J=after.best_validation.mean(),
            old_local_followup_J=before.test_objective.mean(),new_followup_J=after.test_objective.mean()))
    pd.DataFrame(comparison).to_csv(u.OUT/"historical_comparison.csv",index=False)
    out=f"""

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

{table(nonuniform)}

逐批候选、每个输出端口缺口、权重、概率、操作、图指纹及最终数值反馈在 `search/*/decision.json`、`summary.json`、`factory_trace.json`、`executions/`。没有向策略传入H₀、2017成绩、文件名、人工预期赢家或旧穷举排名。

### 38.4 机制验收、发现的问题及处理

受控测试验证：单侧/双侧改变相应能力需求与路径覆盖；past_only改变实际匹配；开关软偏好改变概率而非合法性；无关描述/文件名不改变选择依据；状态在错误分支不能抵消当前state缺失；条件分支不是能力并集；未知契约保留UNKNOWN；微秒/纳秒存储不改变实际小时特征；软支持采用位置频数。生产真实上传与北京实验调用同一决策API，且已有栅格真实页面/下载测试通过。界面仍只支持其规定栅格，不宣称新增北京上传界面。

2015年两条真实路线“最近邻”和“线性＋最近邻回退”执行、质量检查、预测NPZ回读通过且输出不同。相关初始机制回归74项通过；最终受影响回归见下节。

保留的异常记录：旧测试曾依赖旧源码字符串，已改为检查公共调用/开关并增加行为测试，没有删除失败案例。5次及20次评价冒烟中A种子11未找到合格图，但实际执行/质量反馈正常；前20图与历史A逐图逐状态一致，旧A第25次才成功。这是有限预算未命中，不是生成器Bug；失败JSON保留，未增加好种子或改规则。

第一新版本v3运行时发现时间数组同时使用`datetime64[us]`（685例）和`datetime64[ns]`（288例），提取器直接比较底层整数导致采样间隔错误记UNKNOWN。停止该版本，保留7份完整搜索摘要和在途记录；公共提取器统一`.as_unit('ns')`后测得真实1小时。插值内核未改，但不混用错误日志；新v4重新冻结和执行。没有删除v3，没有据结果调规则。资源监控首次命令转义失败，改为复用原监控脚本的明确命令行参数；结束时发现监控器也匹配了自己的`--entry`参数，已改为匹配实际脚本位置，并按执行进程固定PID排除{resource['excluded_other_pid_samples']}条其他PID采样，原始行全部保留。XML合并首条命令转义失败后改用无嵌套转义的同等命令，未改变任何测试结果。以上不是算法调优，不隐藏失败。

### 38.5 冻结协议、公平预算及数据隔离

原12站CSV、训练状态、2015/2016/2017公开输入、掩码、私有评价队列及权重逐文件SHA一致；没有重新匹配或下载。2013—2014拟合；2015仅用于既有校准/机制验收；2016为真实搜索队列201189目标；2017为冻结后46476目标的历史暴露探索性复核，不是新盲测。人工参考H₀及主指标J未变，所有方法同队列、同质量门，不在成功子集排名。

同一949种规范图、同一未完成初始图、10种子11/22/33/44/55/66/77/88/99/110。每运行最多50次独特Objective计分、500外层提议、1000原始回调、600秒软墙钟，原提前停止保持；不完整反馈也占计分，不用缓存补次数。所有组独立冷执行器/内核缓存，未声称清空操作系统文件缓存；共同训练状态的原记录拟合成本{fit_cost['seconds']:.6f}秒，本轮直接复用，不冒称重新拟合。

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

{table(pd.DataFrame(baselines))}

新增的解释性强对照“linear_nearest”只是原池已有的线性＋最近邻回退合法图，不新增算法、不替代H₀。不能只胜过偏弱H₀就声称自动组合有效。

### 38.7 全部搜索结果与成本

{table(summary)}

proposal_legal是外层提议通过共同结构/端口检查的比例；real_quality_pass是完整真实执行通过最终质量门的比例，不把二者或重复的后验检查次数混作同一分母。

实际计数：{json.dumps({k:int(v) for k,v in total.items()},ensure_ascii=False)}。计分、完整真实执行、质量失败、不完整反馈、缓存、拒绝与延后回调分开记录。初始化/子树产生的真实工厂构建总次数{int(costs.constructions.sum())}，不冒充额外Objective评价；每组计数详见CSV。

部分运行未达到50次计分：A种子99为7次；C种子22/33/110分别31/37/8次，C-gap种子22/33/44/110分别31/37/40/42次。保留统一的提议/回调上限和GOLEM原提前停止、无法补足下一代的正常退出记录，没有补次数或补种子；不把这些运行称为“空间已穷尽”。`common_budget_prefix.csv`和`common_execution_time_prefix.csv`另提供各配对种子实际共同达到的计分、执行和时间前缀对照。存在重复提议与局部停滞，属于当前机制的实际效率限制，不隐藏、不用重启凑更好成绩。

搜索总时间{runs.seconds.sum():.2f}秒，输入装载/特征提取/规则映射/匹配等setup总计{runs.platform_setup_seconds.sum():.2f}秒，逐运行平台总计{runs.platform_total_seconds.sum():.2f}秒。公共决策服务内部时间不含外层特征遍历，不能只报告那一个很小数；setup已包含该部分。逐运行total计时到摘要组装，最终摘要/工厂日志写出等还有额外开销，进程生命周期资源采样保留，不能解释为完整端到端耗时的精确下界之外无成本。

进程OS生命周期峰值工作集{resource['process_lifetime_peak']/1024**3:.3f}GiB；从进程创建到最后一次正确PID采样的墙钟{resource['last_sample_wall_since_process_start']:.2f}秒（含搜索、穷举和复核；采样停止有约2秒粒度，不作精确完成时长）。资源监控晚于最初若干运行启动，原始采样范围与说明见 `resource_summary.json`，不冒充全系统资源测量。保存完整嵌套轨迹会增加内存，不能只报算法内核成本。best-so-far同时保存计分次数、真实执行次数、搜索墙钟和加上setup的平台墙钟；未外推提前停止后的成绩。

逐种子完整结果（无删除失败或差种子）：

{table(perseed)}

### 38.8 穷举、配对不确定性与创新判断

949图全部真实执行，其中{len(passing)}通过、{949-len(passing)}质量失败；最优验证J={min(r['objective'] for r in passing):.6f}。完整输出的支持后验与实际成功/失败逐图交叉核验一致。此空间可穷举，不据此声称GOLEM必要或最优。

解释性强固定流程linear_nearest达到同一穷举最优，说明当前候选池内无需昂贵搜索就可取得该上限。C相对A的改善不能全部归因于完整语义引导，因为单独B已有很大改善；新C的能力/支持证据可以提高有效提议，却没有输入→精度预测模型，也不能保证把预算集中到最低误差图。软支持与原场景加权J不是同一个目标，这是可能的限制而非经本实验单独证实的因果解释；C与C-gap的对照用于检验其实际净收益，不按测试成绩重新定义偏好。

2017复核配对差异：负值表示左侧方案J更低。沿用按站点和7天时间块的双向配对重抽样1000次；同一连续缺口按起点周整体处理，遮挡复本不是独立环境样本；不同搜索种子只表示算法随机性，不当成不同环境。以下为探索性点式95%区间，无多重比较校正，不作确认性显著结论。

{table(pairs[['a','b','objective_difference','ci95_low','ci95_high']])}

{chr(10).join(judgments)}

上述结论只针对当前队列和点式区间，不能因结构合法率上升就推论精度提升。若强固定linear_nearest不劣于搜索，应优先考虑简单流程；不为追求优势追加预算、调参或扩大候选池。

### 38.9 新旧C的可比范围

{table(pd.DataFrame(comparison))}

相同数据、掩码、J和949共同语法允许作上述历史数值比较，但新C不等于旧局部C：前者由共享规则/能力/兼容性/端口状态驱动，支持频数用于软偏好；后者是任务局部端口缺口和去重支持风险。新C-gap本身也使用公共路径能力缺口，不机械当作旧gap-only实现。历史时间不在同一时段采集、旧版未完整列setup成本，不能用新旧墙钟推断因果加速。

### 38.10 定向回归、文件验收与复现

最终相关测试{acceptance['tests']}项，失败{acceptance['failures']}、跳过{acceptance['skipped']}；没有重复无关全量回归。核验{acceptance['actual_prediction_files_read_and_hashed']}份实际预测文件SHA/回读、949规范空间、50运行预算、每次回调账目、原12CSV、复用掩码和训练状态、旧协议/冻结/结果哈希以及新源码/冻结指纹；冷缓存强固定路线两次真实执行逐值一致。环境沿用第37章，不安装新依赖，第三方GOLEM核心不变；旧GOLEM声明依赖与当前主数值栈的版本警告仍保留。

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
"""
    print(out)


if __name__=="__main__":main()
