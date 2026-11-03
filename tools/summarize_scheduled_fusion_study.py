"""Evidence-only report; no tuning, searching or test-dependent policy changes."""
from pathlib import Path
import json,hashlib,statistics,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/scheduled_fusion_20261025'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    result=load(OUT/'results.json');audit=load(OUT/'independent_audit.json');accept=load(OUT/'delivery/acceptance.json')
    assert audit['status']==accept['status']=='PASS'
    cli=load(OUT/'delivery/cli_entry_verified/summary.json');file=load(OUT/'delivery/file_entry/summary.json')
    assert cli['recommended_key']==file['recommended_key'] and cli['best']['numeric_fingerprint']==file['best']['numeric_fingerprint']
    tests=set();executions=0
    for p in [OUT/'regression.xml',OUT/'compatibility.xml',OUT/'boundary_tests.xml',OUT/'cli_tests.xml']:
        if not p.exists():continue
        root=ET.parse(p).getroot()
        for s in root.findall('.//testsuite'):assert int(s.attrib['errors'])==int(s.attrib['failures'])==0
        for c in root.findall('.//testcase'):tests.add((c.attrib['classname'],c.attrib['name']));executions+=1
    m=result['summary'];baseline={name:statistics.mean(r['rmse'] for r in result['baselines'] if r['baseline']==name) for name in ('fixed','fixed_stationary')}
    singles=load(OUT/'delivery/single_source_controls.json')
    baseline.update({name:statistics.mean(r['metrics']['RMSE'] for r in singles if r['source']==name) for name in ('fine_slow','coarse_fast')})
    alns_better_random=m['alns']['success_rate']>m['random']['success_rate']
    claimed='本轮ALNS比纯随机更容易在预算内命中验证全局最优。' if alns_better_random else '本轮未证明ALNS比纯随机更容易在预算内命中验证全局最优。'
    final=dict(status='PASS',legal_structures=1296,stationary_structures=36,summary=m,test_baselines=baseline,unique_tests=len(tests),test_executions=executions,
        independent_audit=audit,scope='Four validation/test pairs; controlled same-origin data, finite-domain validation global optimum, no universal or WRF claim')
    (OUT/'final_summary.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    table='\n'.join(f"| {name} | {m[k]['successes']}/20 | {m[k]['median_hit_evaluations'] if m[k]['median_hit_evaluations'] is not None else '未命中'} | {m[k]['expected_evaluations_to_success'] if m[k]['expected_evaluations_to_success'] is not None else '未命中'} | {m[k]['mean_test_rmse']:.6f} |" for k,name in [('alns','三画像＋ALNS'),('basic','三画像＋普通变异'),('random','纯随机合法域搜索'),('none','仅三画像初始2图')])
    report=f'''# 分时逐源组合与验证误差驱动变异：实验交付（2026-10-25）

## 已执行机制

严格沿用“合法域完整建立→三画像初筛→变异候选真实执行→质量检查→真实验证误差反馈”，新增标签专用SearchValidation端口，公共文件、页面处理及CLI共用服务。最终测试标签不能成为SearchValidation；独立评价与搜索验证时间重叠会拒绝。搜索真实运行时不访问穷举表。

变异接入原生ALNS7/AlphaUCB破坏—修复，奖励依据验证RMSE，真实误差并列时才以画像、步骤数和索引破平。未使用GOLEM优化器，未新训黑箱排序器。没有将最终测试误差拿来选择图或改权重。旧的画像代理评分负结果保持不变。

## 组合数及关系

每时段：细源时间线性插值1 × 粗源两种空间方法及可选滤波前/后顺序6 × 细源主导/粗源主导/细节投影3 × 融合后滤波开关2 =36。两时段独立选择，共 **1296种**，其中上下时段相同的36种。全部经过结构检查，每个开发和验证案例都全部真实执行作穷举对照。执行通过与精确数值不同数见独立审计，不能把结构数当作算法库独立方法总数。

逐源不同处理、先后顺序、主辅角色、融合前后处理和分时策略均进入实际图。新增的滤波只是对现有SciPy三乘三中位数核的适配；滤波有可能抹去真细节，不预设其优势，未知画像不填造。独立审计已找到只改变第一时段的空间处理/滤波顺序、使第一时段数值改变而第二时段不变的真实产品对。

1296是冻结的代表性任务子空间，不是整库所有DAG；没有涵盖任意长度、所有参数连续值或全部环境任务。两个时段的切分点固定，未搜索任意切分位置。本轮只有一种细源时间插值，不能据此宣称所有时间算法已比较。

## 数据与信息权限

NCEP真实温度受控降采样：相对细源12×12/48小时（3个锚点），相对粗源6×6/6小时（17个时刻），恢复14个完整细场。原生约2.5°，同一再分析构造互补源，非独立传感器；面积支持为受控观测假设，非原生传感器支持证明，也未运行WRF。

2013四个开发组确定最强固定整图和最强上下时段相同图。2014四组搜索验证与四组后续日期测试，两两不重叠；每验证组5种子。所有方法/案例的选择写入frozen_selections.json后才读取测试标签。开发仅选择基线，本轮没有训练新学习模型。参考协议与全部日期见frozen_protocol.json。

每策略最多64次唯一真实评价、256提议；仅初筛执行2图。ALNS/普通变异共享画像初始2图；纯随机忽略画像排序，从完整合法域均匀取初始图并扩展。所有方案以同一验证队列的真实RMSE选优。相同中间链可以缓存，重复提议不会算新增执行，未用预计算误差表替代候选执行。

## 结果

| 方法 | 64次预算内命中验证全局最优 | 命中者首次执行次数中位数 | 含失败的汇总期望执行成本 | 独立测试平均RMSE/K |
|---|---:|---:|---:|---:|
{table}

命中容差为1e-9K。首次命中次数是事后与穷举核对的基准，搜索当时并不知道已最优；实际仍运行到预算或停止条件，实际执行数见final_summary.json。不能将首次命中速度说成在线认证速度。仅初筛预算实际为2次。期望执行成本为首次命中次数加失败实际耗尽次数、除以成功次数；是跨任务描述性汇总，须结合独立审计per_case逐案例结果阅读，不能只比较命中者中位数。四个案例为分析组，五种子不是五个独立数据集。不同案例并行运行，墙钟受共享资源影响，不能以并行耗时宣称算法加速；主报唯一执行次数。

{claimed} 所有未命中种子保留，不补换种子、不增加预算直到出优势。

独立测试基线：开发期最强固定整图 **{baseline['fixed']:.6f} K**，最强上下时段相同图 **{baseline['fixed_stationary']:.6f} K**；补充单细源 **{baseline['fine_slow']:.6f} K**，单粗源双线性 **{baseline['coarse_fast']:.6f} K**。单源为选图后附加工程基线，不参与任何调参。搜索命中验证最优和测试误差降低是两件事，不将其中一项冒充另一项。

## 全局最优与保证

对冻结合法集合G，穷举真实执行得到g*=argmin_G RMSE_validation。每条搜索轨迹以首次达到该值的真实执行次数衡量效率，未达到按失败保留。**这可证明有限候选域内的验证目标全局最优，不能保证独立测试误差最小。**

64次自适应变异不承诺覆盖1296种。需要确定性保证时，使用delivery/request_exact_validation.json：先变异，再完整覆盖剩余候选；遍历完成且有合法图后给出EXACT_VALIDATION_RMSE_OPTIMUM_IN_DECLARED_DOMAIN。保证来自完整覆盖，不单独归功于变异，也不超出已声明域。

## 验收和直接使用

{len(tests)}项不同回归/定向测试通过，包含标签隔离、完整计数和真实误差反馈。独立回读{audit['decisions']}次科学决策、{audit['real_numeric_products_checked']}份实际数值产品，重算RMSE、推荐、穷举最优、首次命中和开发固定基线；全部通过。最终测试产品另行回读核验。文件和页面处理结果相同，实际CLI也相同；未声明浏览器逐控件验收。

已生成可用请求：output/scheduled_fusion_20261025/delivery/request.json（预算搜索）、request_exact_validation.json（完整覆盖保证）、page_request.json（页面补充需求）。平台识别来自真实输入；任务候选语法由本协议明确声明，没有暗含预期胜出图。

```powershell
$env:PYTHONPATH = 'D:/codex/EcoDataEngine/src'
./.venv/Scripts/python.exe -m ecodata.algorithms.platform_search output/scheduled_fusion_20261025/delivery/request.json output/scheduled_fusion_new_request
```

输出目录必须为新目录。完整证据位于output/scheduled_fusion_20261025：frozen_protocol.json、source_frozen、source_release、results.json、independent_audit.json、final_summary.json、search_curves.csv，以及全部失败日志。原始故障（测试临时目录不存在、计数检查误读结构状态名、最终产品布尔属性的NetCDF序列化、python -m产生的验证类重复身份）均保留失败日志，并经定向测试及真实入口修复验收。序列化修正在所有选图冻结之后、首次读取测试标签之前，未改变选图；CLI仅转发至规范模块，未放松标签用途或类型检查。源码差异及原因见orchestration_repair和cli_type_repair的JSON/diff；没有根据测试成绩改科学规则。

本轮未证明独立来源融合、原生WRF响应、所有算法参数组合适用或无条件搜索最优。三画像、AlphaUCB和具体算子的独立贡献不能从本轮整套方法比较分别推断。本轮搜索需要可遮挡真值形成验证反馈；无可用验证标签时仍只能按画像推荐，不能套用验证全局最优保证。结论只适用于冻结任务和预算。
'''
    boundary=OUT/'boundary_diagnostic_fix.json'
    if boundary.exists():report+='\n## 时段诊断修正\n\n'+load(boundary)['explanation']+'\n'
    (ROOT/'docs/分时逐源组合_实验交付_20261025.md').write_text(report,encoding='utf-8')
    master=ROOT/'docs/EcoData Engine项目总交付报告.md';text=master.read_text(encoding='utf-8')
    assert '## 第57章' not in text
    chapter=f'''\n\n## 第57章 完整分时逐源组合与真实验证误差反馈实验（2026-10-25）

已按用户确认路线在公共平台接入SearchValidation专用反馈：三画像先完整筛选，再以真实验证RMSE接受变异，最终测试不反馈。原生ALNS/AlphaUCB继续复用，新增现有SciPy空间中位数适配器，未知画像不填造。每时段细源时间处理1×粗源空间方法/滤波先后6×主辅或投影3×后处理2=36，两时段独立共1296结构，固定同策略36；不是整库全部可能组合。四个开发及四个验证案例全域真实执行，另有四组后续日期测试。

四验证案例×五种子、各64次唯一评价：ALNS命中{m['alns']['successes']}/20，普通变异{m['basic']['successes']}/20，纯随机{m['random']['successes']}/20，仅初筛{m['none']['successes']}/20；未命中保留。独立测试平均RMSE：ALNS {m['alns']['mean_test_rmse']:.6f}K、普通变异 {m['basic']['mean_test_rmse']:.6f}K、纯随机 {m['random']['mean_test_rmse']:.6f}K、仅初筛 {m['none']['mean_test_rmse']:.6f}K、开发期最强固定 {baseline['fixed']:.6f}K。{claimed} 不把命中验证最优等同测试最优或普适显著优势。

全局参照明确为冻结1296种合法图的最低真实验证RMSE。预算变异不保证全覆盖；完整覆盖配置先变异后穷举剩余，可认证有限域验证最优，不能归功于变异或宣称真实未知误差全局最优。{len(tests)}项不同测试通过，独立审计{audit['decisions']}决策、{audit['real_numeric_products_checked']}产品及最终测试产品通过。文件、页面处理及CLI同结果；图/顺序实际数值差异及分时局部性有产品证据。

数据仍为同一真实NCEP场的受控互补分辨率构造（12×12/48小时与6×6/6小时），非独立传感器、非原生高分辨率、未运行WRF。没有新训黑箱模型。详见docs/分时逐源组合_实验交付_20261025.md及output/scheduled_fusion_20261025全部冻结配置、源码和负结果。第56章和更早负结果保持不变。
'''
    master.write_text(text+chapter,encoding='utf-8')
    print(json.dumps(dict(status='PASS',tests=len(tests),summary=m,baselines=baseline),ensure_ascii=False))
if __name__=='__main__':main()
