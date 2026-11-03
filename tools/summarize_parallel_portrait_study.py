"""Deliver only saved, independently audited results; preserve negative findings."""
from pathlib import Path
import csv,hashlib,json

import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/parallel_portrait_20261025'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    r=load(OUT/'results.json');audit=load(OUT/'independent_audit.json');entry=load(OUT/'delivery/acceptance.json')
    assert audit['status']==entry['status']=='PASS'
    names=dict(full='三路联合＋ALNS',no_data='去掉数据适用性',no_model='去掉模型适配性',no_collaboration='去掉组合协同',
        unscored_alns='无三路评分＋ALNS',random='随机32图',random_cost_allowance='随机104图',score_only='仅三路首选图')
    lines=['# 三路并行评分与多区域时空实验交付（2026-10-25）','',
        '实现与实验均已完成；验收通过不等于所有机制都取得效果优势。旧版全部同分及泛化负结果保留。','',
        '## 机制与边界','',
        '公共 Composition 复用既有注册算法、观测算子、场景风险函数、真实执行与ALNS；新增三路评分，不训练替代排序器。先对完整1296种组合联合评分，再初始化和变异。提供SearchValidation时，变异接受和最终选择依据真实验证RMSE；不提供时依据三路评分推荐。','',
        '- 数据适用性：遮挡可见内部细网格锚点，实际执行组合，按 1/(1+RMSE/锚点空间标准差) 评分。',
        '- 模型适配性：复用明确场景中粗观测一致性、细节保留的风险下降，映射至[0,1]。这不是已测WRF敏感性。',
        '- 组合协同：比较组合与其自身各完整单源分支的误差，按相对改善评分；不将单算法标签加分冒充协作收益。','',
        '每时段36张完整局部链真实探测，两个时段72次。中间状态和顺序副作用实际记录。缺少可夹持内部锚点、失败或不支持的响应保留未知区间[0,1]；局部探测不推造跨时段协同。','',
        '## 数据、训练与对照','',
        '使用同一真实NCEP温度场受控构造互补分辨率，非独立传感器。每例33个六小时时次、8×8细网格五个48小时锚点、4×4粗网格全时次，评价28个隐藏细网格时次。两个固定时段，各36局部策略，共1296完整图；不是整个算法库的全部可能图。','',
        '8个2013年开发案例校准三路权重；四个不重叠区域×两个2014年时期形成8组验证—后续测试案例。东部两个区域不参与校准。每个案例3个搜索种子，种子不是独立数据集。开发和验证各8例全部真实执行1296图。','',
        '全机制权重仅在等权及(.5,.25,.25)三种排列中选择；逐路消融在剩余两路的三种预定权重中校准。只用开发组初始两张图的相对遗憾选择，保留留一案例诊断，不根据测试修正规则。开发最强固定从1296图跨开发平均误差选择，另保留最强36种不分时方案。','',
        '全机制及三项消融、无评分ALNS和标准随机均32次完整评价/128提议；仅评分只执行1图。额外随机给予104次完整评价，用于考察72次额外局部探测带来的成本影响。局部探测和完整图成本不同，不宣称严格等时对照。','',
        '## 真实结果','',
        '| 方法 | 测试平均RMSE/K | 相对固定差异/K | 验证最优命中 | 平均局部探测/完整评价 |',
        '|---|---:|---:|---:|---:|']
    for method,s in r['summary'].items():
        lines.append(f'| {names[method]} | {s["mean_test_rmse"]:.6f} | {s["mean_difference_vs_fixed_K"]:+.6f} | {s["validation_hits"]}/{s["runs"]} | {s["mean_probe_graphs"]:.0f}/{s["mean_objective_graphs"]:.0f} |')
    lines+=['',f'开发期最强固定测试平均RMSE：**{r["mean_fixed_rmse"]:.6f} K**。', '',
        '冻结的三路权重：`'+json.dumps(audit['calibrated_weights'],ensure_ascii=False)+'`。','',
        '## 评分辨识与消融解释','',
        f'独立审计的{len(audit["discrimination"])}个开发/验证案例中，分数种类范围为'+str(min(x['unique_scores'] for x in audit['discrimination']))+'—'+str(max(x['unique_scores'] for x in audit['discrimination']))+'（每例1296候选）。不再全部同分。分数与负真实误差的Spearman相关逐例见independent_audit.json；分数不同不能替代有效性证据。','',
        '全机制相对各逐路消融的测试RMSE差异（负数表示全机制较好）：','']
    full=r['summary']['full']
    for m in ('no_data','no_model','no_collaboration','unscored_alns','random','random_cost_allowance'):
        delta=full['mean_test_rmse']-r['summary'][m]['mean_test_rmse']
        lines.append(f'- 对{names[m]}：{delta:+.6f} K。')
    diagnostic=load(OUT/'development_top_rank_diagnostic.json')[0]
    a,b=diagnostic['full'],diagnostic['no_collaboration']
    lines+=['',f'开发组具体冲突（{diagnostic["case"]}）：全三路首选图真实误差{a["validation_rmse"]:.6f}K，而去协同首选为{b["validation_rmse"]:.6f}K。后者数据分和模型分均较高，但协同分较低；相对自身较弱分支的改善奖励反而把前者推到更前。见development_top_rank_diagnostic.json。该开发负结果未触发对测试规则的临时修改。']
    val_oracles=[load(p) for p in (OUT/'oracles').glob('val_*/summary.json')]
    calibration=load(OUT/'calibration.json')
    same_fixed=sum(d['recommended_key']==calibration['fixed'] for d in val_oracles)
    init_hits=sum(x['hit'] and x['first_hit']<=2 for x in r['runs'] if x['method']=='no_collaboration')
    lines+=['',f'8个验证案例中有{same_fixed}个的穷举最优就是同一开发固定图。去协同组{init_hits}/24次在初始两图中已命中，不能将其收益归功于变异。本轮全三路未超过固定，主要对应预算内仍有5次未找到该验证最优，而不是已发现验证更优图却在测试反转。该任务尚未证明不同案例需要不同最优组合。', '',
        f'全三路相对无评分ALNS降低测试RMSE约{100*(1-full["mean_test_rmse"]/r["summary"]["unscored_alns"]["mean_test_rmse"]):.2f}%，相对随机32图约{100*(1-full["mean_test_rmse"]/r["summary"]["random"]["mean_test_rmse"]):.2f}%；相对最强固定仍高{100*(full["mean_test_rmse"]/r["mean_fixed_rmse"]-1):.2f}%。这些是冻结案例上的描述性效果，三路机制额外包含72次可见探测，不是零成本规则的单独优势。']
    lines+=['','逐例结果、所有种子、失败和描述性案例bootstrap区间见results.json。8个案例共享区域和时期结构，不能将24次运行当24个独立数据集；不据此宣布普适或统计显著优势。','',
        '## 验收证据','',
        f'- 独立回读{audit["real_numeric_products_checked"]}份开发/验证/搜索产品、{audit["test_products_checked"]}份不同最终测试产品；重算RMSE、全域最优和固定方案。',
        f'- 独立重算{audit["real_probe_responses_recomputed"]}份局部响应的三路公式，依据原始可见锚点和独立球面面积矩阵核验模型状态。',
        '- 71项不同定向/兼容测试通过，覆盖数据与模型偏好改变推荐、未知保持未知、独立参考不影响选图、旧模块兼容。',
        '- 文件、页面处理器及实际CLI在不提供隐藏参考/搜索标签时输出相同产品；不冒充浏览器逐控件验收。',
        '- 所有候选选择先写入frozen_selections.json，再执行最终测试；正式评分代码与训练规则按source_manifest.json冻结。','',
        '## 直接使用','',
        '无隐藏参考推荐：`output/parallel_portrait_20261025/delivery/request_no_reference.json`。带独立验证反馈：`delivery/request_validated_search.json`。均通过公共平台入口，无实验局部选择规则。输出目录请用新目录。','',
        '```powershell',"Set-Location -LiteralPath 'D:/codex/EcoDataEngine'","$env:PYTHONPATH = 'D:/codex/EcoDataEngine/src'",'./.venv/Scripts/python.exe -m ecodata.algorithms.platform_search output/parallel_portrait_20261025/delivery/request_no_reference.json output/parallel_new_request','```','',
        '## 必须保留的限制','',
        '本轮是有足够可见锚点的连续互补时空重建范围，不能宣布所有特征、算法或模型均具备可靠响应。遮掉内部锚点会把该探测点的48小时供数间隔扩大为96小时，探测难度不同于实际目标；权重校准不能消除这种分布差异。','',
        '模型路线仍为明确输入偏好，未运行WRF；噪声、锋面、动力一致性等未知敏感项不填造。实际主目标是验证RMSE，三路评分属于初始化和提议引导。没有相应消融和独立效果证据，不能把ALNS的收益全部归于三路评分。','',
        '评分支持区间不是泛化置信区间。没有足够观测时会退为未知，应补证据而不是放宽门槛。完整穷举只认证冻结候选域的验证最优，不保证最终测试全局最优。','']
    path=ROOT/'docs/三路并行评分_多区域实验交付_20261025.md';path.write_text('\n'.join(lines),encoding='utf-8')
    # A readable paired-case table, independent seeds averaged within each case.
    with (OUT/'case_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['case','method','mean_test_RMSE_K','fixed_RMSE_K','difference_K'])
        fixed={x['case']:x['rmse'] for x in r['baselines'] if x['method']=='fixed'}
        for method,s in r['summary'].items():
            for case,error in s['case_rmse'].items():writer.writerow([case,method,error,fixed[case],error-fixed[case]])
    master=ROOT/'docs/EcoData Engine项目总交付报告.md';text=master.read_text(encoding='utf-8')
    title='## 第58章 三路并行评分与独立时空对照（2026-10-25）'
    assert title not in text,'Do not duplicate final report chapter'
    addition='\n\n'+title+'\n\n'
    addition+='新增公共composition.parallel_scoring：数据适用性、模型输入适配、组合协同三路并行，以输入中可见锚点遮挡的真实局部执行补齐响应，不训练替代排序器，未知维持区间。先全域评分后变异，探测72图成本单列。\n\n'
    addition+=f'8开发案例校准权重，8组四区域/双时期验证—测试案例；开发/验证1296全域真实执行。全机制测试RMSE {full["mean_test_rmse"]:.6f}K，开发最强固定{r["mean_fixed_rmse"]:.6f}K；验证命中{full["validation_hits"]}/{full["runs"]}。随机32图{r["summary"]["random"]["mean_test_rmse"]:.6f}K，随机104图{r["summary"]["random_cost_allowance"]["mean_test_rmse"]:.6f}K。逐路消融及所有负结果保留。\n\n'
    addition+='分数不再全部相同；是否选优依照真实测试/消融结果判断，不以链路验收替代效果。71项不同测试、独立数值与评分审计、实际文件/页面处理器/CLI通过。数据仍为同源受控分辨率构造，非独立传感器、非WRF响应；不承诺测试全局最优。详见三路并行评分_多区域实验交付_20261025.md及output/parallel_portrait_20261025。\n'
    master.write_text(text+addition,encoding='utf-8')
    print(json.dumps(dict(status='PASS',report=str(path),mean_test_rmse=full['mean_test_rmse'],fixed=r['mean_fixed_rmse'])),flush=True)


if __name__=='__main__':main()
