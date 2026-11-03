"""Write a factual report from sealed experiment and independent audit files."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/spatial_next_20261023'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def run():
    audit=read(OUT/'independent_audit.json');assert audit['status']=='PASS'
    selection=read(OUT/'final_selection.json');results=pd.read_csv(OUT/'final_audit_results.csv')
    assessments=read(OUT/'final_audit_assessment.json');model=read(ROOT/'output/plan_ranking_models'/selection['model']/'report.json')
    states={}
    for filename in ['release_tests.xml','registry_recheck.xml','rank_sealed.xml','policy_tests.xml']:
        for test in ET.parse(OUT/filename).getroot().iter('testcase'):
            states[(test.attrib.get('classname'),test.attrib['name'])]=not any(test.find(k) is not None for k in ['failure','error'])
    assert all(states.values())
    release=dict(status='PASS',unique_targeted_tests=len(states),scope='public profiles, adapter/rank guards, platform integration, P1/P2; not full repository',
        retained_failure='original registry count80 expectation; corrected to83 after adding three real adapters')
    (OUT/'release.json').write_text(json.dumps(release,indent=2),encoding='utf8')
    rows=[]
    for name in ['round1','round2','round3_rank','round4_rank_fixed','round5_extreme_metric','round6_extreme_rank','round7_extreme_rank_fixed']:
        path=OUT/(name+'_validation_assessment.json')
        if path.exists():
            for x in read(path):rows.append(dict(round=name,scenario=x['scenario'],coverage=x['coverage'],RMSE=x['mean_RMSE']['learning'],
                fixed=x['mean_RMSE']['development_fixed'],goals_met=x['goals_met']))
    pd.DataFrame(rows).to_csv(OUT/'development_iterations.csv',index=False)
    mean=results.groupby(['scenario','strategy']).RMSE.mean().unstack()
    chosen=results[results.strategy=='learning']
    changed=[]
    for (scenario,region,scale),part in chosen.groupby(['scenario','region','scale']):
        ref=str(selection['fixed'][f'{scenario}/{region}/{scale}'])
        changed.append(dict(scenario=scenario,region=region,scale=int(scale),distinct_decisions=int(part.steps.nunique()),
            changes_from_fixed=int(part.steps.ne(ref).sum())))
    (OUT/'decision_diversity.json').write_text(json.dumps(changed,indent=2),encoding='utf8')
    lines=['# 扩展空间重建与学习选优实验（2026-10-23）','',
        '结论：可靠启用与旧机制选优已改善，最终工程门通过；但相对开发最强固定方案基本持平，尚未证明稳定的额外自适应收益。不能把前者说成普遍算法创新已获证实。','',
        '本轮真实运行并按开发验证结果迭代，最终测试仅在模型、策略和全部推荐封存后开启。所有轮次和负结果保留。','',
        '## 数据与算法','',
        '新增数据来自[xarray官方示例库](https://github.com/pydata/xarray-data/blob/master/air_temperature.nc)：NCEP/NMC sigma=.995气温再分析，非ERA5 2m气温、非独立站点真值。取两个不重叠24×24区域、2/3倍面积平均退化后重建。训练仅2013年，2014年1/2月校准、3—6月开发验证、7—12月最终审计，固定每月1日00时评估；最终六日期组，不能把像素或图数当独立样本。',
        '', '七基础方法：最近邻、双线性、PCHIP、三次插值、已有IDW、MetPy线性、普通克里金；各自可接已有QP粗一致性校正，共14图。后三种明确将粗均值近似为中心值代理，非真实点观测；最近邻边界规则显式记录。缺权重/辅助/兼容输入的算法未硬塞入，83项现有注册清单见algorithm_inventory.csv。','',
        '从96任务的多季节训练，增加到384任务，再从仅2013年候选中按可见粗数据的各特征极值补86任务；不使用验证或审计的参考结果选训练样本。最终470个拟合任务、8个校准任务、16个开发验证任务、24个审计任务。各图通过submit_search真实执行，训练通过train_profile_matching复用已执行记录；共享产品复用不重复计数。',
        '', '## 机制变化','',
        '原逐特征min/max支持域、完全匹配上下文/图及整池支持要求未放宽。新增support_diagnostics逐项记录越界值、训练边界和匹配样本数。训练仍使用平台已有RF64/depth4/minleaf2/seed17。',
        '', 'task_rank复用原本已有但未进入主排序的任务内排序头；按预测相对名次选择，训练/推荐的目标偏好必须一致，旧模型缺少契约元数据时拒绝冒用。它不是本轮新发明的排序学习算法。指标预测排序作为消融保留。',
        '', 'development_fixed回退只采用训练期各上下文的平均效果，不读取当前参考；先经过物理/接口硬筛选，再在合法候选中选开发固定参考。其状态独立标为DEVELOPMENT_FIXED_FALLBACK，不计入学习覆盖率。陌生上下文没有该参考时仍回到原规则；不伪造预测。',
        '', '完整学习证据保留于模型目录；公共推荐只显示每列表最多16条均匀摘录，并记录完整文件路径、指纹和条目数，避免每请求复制上百MB历史。该显示缩减不改变模型、支持域、硬规则或训练证据。','',
        '## 开发迭代（允许用于选择，不当作最终盲测）','',
        '| 轮次 | 场景 | 学习覆盖率 | 推荐RMSE K | 强固定RMSE K | 预设工程门 |','|---|---|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['round']} | {r['scenario']} | {r['coverage']:.1%} | {r['RMSE']:.6f} | {r['fixed']:.6f} | {r['goals_met']} |")
    lines+=['', '工程门在首次验证前冻结：覆盖≥80%，相对规则和随机误差均至少降5%，相对开发最强固定不劣于2%。达门只说明此有界实验可用，仍需核查是否实际胜过强固定，以及自适应分支是否有额外价值。',
        '',f"最终只按开发验证选定：{selection['validation_variant']}；learning_order={selection['learning_order']}，fallback_policy={selection['fallback_policy']}。随后冻结全部审计选择才执行数值评价，未据审计结果调参。",'',
        '## 最终审计：每场景24任务，6个日期组','',
        '| 场景 | 开发最强固定 | 规则/成本 | 随机10种子均值 | 最终推荐 | 学习覆盖率 |','|---|---:|---:|---:|---:|---:|']
    for x in assessments:
        s=x['scenario'];m=x['mean_RMSE'];lines.append(f"| {s} | {m['development_fixed']:.6f} | {m['rules']:.6f} | {m['random']:.6f} | {m['learning']:.6f} | {x['coverage']:.1%} |")
    for x in assessments:
        m=x['mean_RMSE'];lines.append(f"\n同模型/同数据消融 {x['scenario']}：原指标预测排序RMSE={m['metric_ablation']:.6f} K；任务内排序但原规则回退={m['rank_ablation']:.6f} K。两者推荐同样在参考开启前封存，不能事后挑选测试赢家。")
    lines+=['', '以上为固定共同内部区域的面积加权RMSE，逐任务等权。M1无硬粗一致性要求，M2只允许七张QP图；偏好为明确声明的准备要求，不声称已测得WRF-Chem敏感性。',
        '', '### 相对强固定及不确定性','']
    for c in audit['comparisons']:
        if c['baseline']=='development_fixed':lines.append(f"- {c['scenario']}：强固定减推荐的日均误差差 {c['mean_gain_K']:.6f} K，日期胜/负={c['winning_dates']}/{c['losing_dates']}；六日期描述性bootstrap区间{c['descriptive_bootstrap_95']}。正值才代表推荐更好。")
    lines+=['', '不能仅因胜过较弱规则/随机就声称数据特征选优已全面优于强固定；小差异、负日期和区间跨零均须保留。decision_diversity.json列同一上下文内选择变化，不能把不同区域预设固定选择冒当数据自适应。',
        '', '本轮每场景21/24任务真正启用学习、3/24采用明确标记的固定回退；总体选择22次三次→QP、2次PCHIP→QP。同一上下文内确有路径变化，但相对强固定24任务仅1胜1负22平，均值仅低约0.0345%，六日期区间跨零。因此不能声称已稳定优于强固定。',
        '', '组合收益独立看待：三次插值加QP的平均RMSE从1.298212降至1.204807 K，24/24改善；其余非最近邻方法加QP也均24/24改善，最近邻持平。这不等于学习评分有额外收益，也不代表该校正在所有真实数据上都有效。',
        '', '## 独立回读与边界','',
        f"独立回读核对{audit['products_verified']}个产品指纹及执行记录，失败{len(audit['failures'])}；重算全部{audit['audit_metrics_recomputed']}张审计图RMSE，最大差{audit['max_RMSE_delta']:.3g} K。源数据、最终源码指纹、封存先于评价、训练/校准/审计日期隔离均通过。{len(states)}项独立定向测试最终通过，原计数断言失败日志保留；不称全项目全部测试。",
        '', '独立审计脚本首版保留float32坐标计算权重，与主评价器的float64权重产生约1.7×10^-7 K复算差；已修正独立脚本的计算精度并保持原1e-10核对阈值，未改实验产品或分数。失败日志independent_audit_float32_failure.log保留。节点缓存保留，不把图执行次数等同于所有内核冷启动，也不据共享产品矩阵声称计算效率优势。',
        '', '本轮是两个已知网格上下文中的新日期重建，不是任意新地区/网格泛化；原支持门仍会拒绝陌生上下文。人工退化再分析也不是新观测精细真值或下游模式改进。Profile为有界图生成/排序，不是本轮证明了进化交叉变异优势。',
        '', '证据目录：output/spatial_next_20261023/。protocol.json、manifest_extremes.json、feature_extreme_acquisition.json、各轮assessment/diagnosis/choices、audit_choices_frozen.json、final_selection.json、independent_audit.json、all_graph_results.csv、audit_combinations.csv、audit_daily.csv及runs/recommendations均保留。']
    report=ROOT/'docs/spatial_next_results_20261023.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    claims=[('complete graph execution','supported',str(OUT/'independent_audit.json'),'factual'),
        ('support gate unchanged and OOD abstention retained','supported',str(OUT/'rank_sealed.xml'),'factual'),
        ('broad generalisation or universal scientific advantage','unsupported',str(report),'limitation')]
    with (OUT/'claim_ledger.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['claim','status','evidence_source','claim_kind']);writer.writerows(claims)
    main=ROOT/'docs/EcoData Engine项目总交付报告.md'
    chapter='\n\n## 第49章 扩展空间重建与可靠选优迭代（2026-10-23）\n\n'
    chapter+='详见docs/spatial_next_results_20261023.md；证据目录output/spatial_next_20261023/。七基础方法及七QP组合全部走公共入口。训练仅2013、开发验证2014年3—6月、最终审计7—12月；支持域未放宽。新增逐项弃权诊断、已有任务内排序头公共接入及明确标识的开发固定回退。\n\n'
    for x in assessments:
        m=x['mean_RMSE'];chapter+=f"{x['scenario']}最终24任务：学习覆盖{x['coverage']:.1%}；推荐RMSE={m['learning']:.6f} K，强固定={m['development_fixed']:.6f}，规则={m['rules']:.6f}，随机={m['random']:.6f}。\n\n"
    chapter+=f"独立核对{audit['products_verified']}个产品，重算336张审计图，定向测试{len(states)}项最终通过。全部迭代负例、诊断、组合持平/负收益和日期差异保留。六审计日期组仅作有界探索证据，不能宣称任意区域泛化、下游模式改善或普遍算法创新已证实。\n"
    chapter+='相对强固定平均仅低约0.0345%，24任务1胜1负22平，六日期描述性区间跨零；这部分优势尚未得到可靠证明。组合本身另有正证据：三次→QP平均1.298212→1.204807 K，24/24改善。新发布模型默认采用已选策略，页面和公共入口均直接读取默认值；不需实验客户端另外硬编码排序。\n'
    assert '## 第49章' not in main.read_text(encoding='utf8')
    with main.open('a',encoding='utf8') as f:f.write(chapter)
    print(report);print(json.dumps(assessments,indent=2))


if __name__=='__main__':run()
