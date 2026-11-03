"""Summarize frozen artifacts without changing scientific decisions."""
from pathlib import Path
from collections import Counter
import hashlib,json,zipfile,shutil,platform
import numpy as np
from importlib.metadata import version
from run_unseen_portrait_response import ROOT,OUT,ARMS,load
from ecodata.algorithms.numerical_search import save


def main():
    audit=load(OUT/'independent_audit.json');assert audit['status']=='PASS'
    assert load(OUT/'independent_data_contract.json')['status']=='PASS'
    effects=load(OUT/'effect_summary.json');costs=load(OUT/'execution_costs.json')
    controls=load(OUT/'controls_frozen.json');prepared=load(OUT/'prepared.json')
    mechanism=load(ROOT/'output/portrait_response_20261028/mechanism_acceptance.json')
    lines=['# 多源响应评分与联合变异：修复验收和新站点实验交付（2026-10-28）','',
      '事实：公共入口机制验收通过；以下效果来自冻结版本的新站点测试，负结果全部保留。没有在测试后改参数、换种子或增加搜索预算。','',
      f"核心结论：完整链路对开发共同固定改善{effects['comparisons']['global_fixed']['improvement_percent']:.2f}%，但对同预算随机为{effects['comparisons']['random']['improvement_percent']:.2f}%、对去响应为{effects['comparisons']['no_response']['improvement_percent']:.2f}%、对去联合修复为{effects['comparisons']['local_only']['improvement_percent']:.2f}%（负数即退步）。因此不能宣称本轮响应评分或ALNS带来选优优势，也不能把自动链路相对固定/单源的整体改善全部归因于组合或评分。",'',
      '## 已完成的公共机制','',
      '- 同分接受改为验证误差、科学节点数、实测成本估计、确定性键的严格排序；同误差增肥不接受。质量检查固定执行，不再作为可反复添加的科学节点。',
      '- 科学状态按完整输出数值、坐标、单位、区间、缺失掩码、来源/校正标志及训练状态命名空间去重，只忽略已知日志属性；选短代表，未评分内部代表可无额外主执行晋升。不是只比验证位置。',
      '- 复用可见日遮挡与响应归一化，从额外遮挡后的全部根输入重新执行多源组合；独立缓存，隐藏真值仅在评分控制器。数据路为1/(1+遮挡RMSE/训练标准差)，模型准备路复用注册属性状态评分并归一化；按声明权重合成区间。未知不补造。',
      '- 状态评分复用risk(x)=clip((x−acceptable)/(bad−acceptable),0,1)，属性收益为处理前风险减处理后风险，按注册需求的重要度加权后用(u+1)/2归一化；前后状态未知时保留收益区间。探测额度耗尽后数据路恢复为[0,1]未知区间，不能把未探测组合说成已测量优劣。',
      '- 本轮沿用旧日恢复实验的0.5数据/0.5模型准备/0协同先验。总量偏差只是输入代理，不是SWAT响应；没有把“相对自身改善”当协同奖励，也没有学习出模型敏感性。',
      '- 联合破坏真实移除当前处理子图，留下根与修复洞；修复从真实已执行逐源组件重建兼容连接。原有替换、参数、插入删除、重连及交换仍保留。使用既有ALNS/AlphaUCB，没有引入GOLEM。',
      '- ALNS和随机共用历史最好父图、相同邻域与两种修复策略。随机为随机算子/修复选择对照，不是从有限整图池抽样。探测、搜索、失败、部署的新增注册算法调用统一计账；缓存不算新调用。','',
      '50项相关定向回归通过；临时目录权限导致的一次环境失败及早期诊断仍保留，最终使用独立临时目录复测。另修复UNKNOWN响应缓存重复读取的边界，失败探测不重算、也不冒充完整评分。旧Robit公共链路验收：','',
      '| 场景 | 总调用 | 不同响应评分 | 科学重复 | 联合试验 | 最终科学节点 |',
      '|---|---:|---:|---:|---:|---:|']
    for r in mechanism['checks']:
        lines.append(f"| {r['case']}{'/自验证' if r['self_validation'] else ''} | {r['total_new_calls']} | {r['response_score_intervals']} | {r['scientific_duplicates']} | {r['joint_trials']} | {r['selected_nodes']} |")
    lines+=['','## 新数据与冻结协议','',
      '三站为GHCN-Daily Perth Airport、Broome Airport、Kalgoorlie-Boulder Airport，配合同地点最近ERA5网格降水代理；不是两个独立地面观测，也不是空间降尺度。站点仅按每年有效率≥80%筛选，未按效果选择。埃塞俄比亚备选站因覆盖不足排除；原始标记、下载请求和哈希保留。',
      'GHCN PRCP按十分之一毫米换算；自然缺失、不合格质量标记和部分时段/推定零标记不当真值。按BOM通用早9点累计日界，与ERA5前24小时按该约定对齐；三站在西澳，使用UTC+8。未取得逐站逐日TOBS，9am属于明确采用的日界假设，审计能证明实现符合该假设，不能替代原站观测时间核验。',
      '2014—2017拟合；2018—2019开发/可见响应与固定对照选择；2020—2021验证选图；2022—2024测试。每站连续3日、14日两类缺口，每年约20%，只遮原本有效值。所有阶段隐藏值均从站点根输入移除。此为离线恢复，允许利用其他日期可见观测，不能解释为未来预测。',
      '正式3站×2情境×3种子×4组=72次；种子11/29/47。各次上限180新增调用（探测最多60）、120秒服务墙钟、40000提议；不限制成几个固定完整组合，无完整空间穷举或全局最优认证。正在执行的内核只能返回后计时，墙钟不是可中断硬保证。',
      f"开发共12次；从全部{controls['discovered_recipes']}个发现配方真实跨开发案例重放，{controls['all_cases_feasible']}个全案例可执行。固定对照为开发共同最强、逐案例最强、站点单源、ERA5单源、原值ERA5、上轮固定转移，均不使用测试效果选择。固定最强指已发现开发配方，不是全库最优。", 
      '完整/随机/去响应/去联合修复四组共享初始规则和总成本；去响应节省的探测额度可用于搜索，避免免费探测。108份最终预测先冻结哈希再计测试。','',
      '## 独立测试结果','',
      '主指标：先按每年隐藏日计算RMSE，再对年份、情境、种子和站点等权，单位mm/day。','',
      '| 方法 | 主RMSE | MAE | Bias |', '|---|---:|---:|---:|']
    names=dict(full='完整ALNS＋响应＋联合修复',random='同邻域随机算子',no_response='ALNS去响应',local_only='ALNS去联合修复',
       global_fixed='开发共同最强固定',station_only='开发站点单源固定',era5_only='开发ERA5单源固定',era5_raw='ERA5原值',legacy_fixed='上轮固定转移',case_fixed='开发逐案例最强固定')
    for key,r in effects['results'].items():lines.append(f"| {names[key]} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['Bias']:.6f} |")
    lines+=['','改善百分数为正表示完整方案更好：','', '| 对照 | 完整方案改善 | 赢站数/3 | 配对站点差描述性95%区间 |','|---|---:|---:|---|']
    for key,r in effects['comparisons'].items():
        interval=r['descriptive_station_bootstrap_95']
        lines.append(f"| {names[key]} | {r['improvement_percent']:.3f}% | {r['station_wins']} | [{interval[0]:.6f}, {interval[1]:.6f}] |")
    lines+=['','只有3个独立站点；种子、缺口类型、年份是重复测量，区间仅描述，不支持稳定普遍优越性的强推断。全部逐任务/逐种子/逐站结果见test_metrics.json与effect_summary.json。失败与不利结果不删除。','',
       '## 实际成本和轨迹','', '| 方法 | 平均新增调用 | 探测调用 | 平均秒 | 失败调用 | xsdba调用 | 选中节点均值 | 响应差异均值 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    trajectories={}
    for arm in ARMS:
        rows=[r for r in costs if r['arm']==arm]
        avg=lambda field:float(np.mean([r[field] for r in rows]))
        probe=float(np.mean([r['phases'].get('probe',0) for r in rows]))
        lines.append(f"| {names[arm]} | {avg('calls'):.2f} | {probe:.2f} | {avg('seconds'):.2f} | {sum(r['failed_calls'] for r in rows)} | {sum(r['fit_adjust_calls'] for r in rows)} | {avg('selected_nodes'):.2f} | {avg('response_intervals'):.2f} |")
        traces=[t for folder in (OUT/'runs/validation').iterdir() if load(folder/'run_metadata.json')['arm']==arm for t in load(folder/'mutation_trace.json')]
        trajectories[arm]=dict(mutations=len(traces),accepted=sum(t['accepted'] for t in traces),joint_trials=sum(t['family']=='joint' for t in traces),
             chosen_with_measured_response=sum(t['edit']['matching'].get('status')=='MEASURED_VISIBLE_MASK' for t in traces),
             selected_node_range=[min(r['selected_nodes'] for r in rows),max(r['selected_nodes'] for r in rows)],
             validation_improved_runs=sum(r['final_validation']<r['initial_validation'] for r in rows),scientific_duplicates=sum(r['scientific_duplicates'] for r in rows))
    save(OUT/'trajectory_summary.json',trajectories)
    selections=[]
    for folder in (OUT/'runs/validation').iterdir():
        summary=load(folder/'summary.json');records=load(folder/'graphs.json');seen=set()
        def visit(key):
            if key in seen:return
            seen.add(key)
            for parent in records[key]['parents']:visit(parent)
        visit(summary['best']['key'])
        selections.append(dict(**load(folder/'run_metadata.json'),
            source_roots=sorted({records[k]['parameters']['source'] for k in seen if records[k]['operation']=='source'}),
            operations=dict(Counter(records[k]['operation'] for k in seen if records[k]['operation']!='source'))))
    save(OUT/'selected_structure_audit.json',selections)
    lines+=['','各组最终同时使用两源的次数（不强迫融合）：'+
        '；'.join(names[a]+f" {sum(r['arm']==a and len(r['source_roots'])==2 for r in selections)}/18" for a in ARMS)+'。具体节点与逐源分支见selected_structure_audit.json。']
    lines+=['','选中科学节点范围：'+'；'.join(names[a]+str(trajectories[a]['selected_node_range']) for a in ARMS)+
        '。同误差增肥被拒绝不等于限制所有复杂图；有验证误差改善时仍可增长，不把它当作泛化改善证明。']
    diagnostic=ROOT/'output/unseen_portrait_response_20261028_diagnostic_v1'
    revision=load(diagnostic/'revision_notice.json')
    lines+=['',f"版本修正记录：正式测试评分前，源码复查发现UNKNOWN响应重复读取会返回不完整评分字典。修复仅统一未知缓存语义，没有依据效果调参；重新通过定向测试和旧数据机制门后重新冻结、重跑。本轮之前的{revision['completed_search_runs']}次完成运行保留为diagnostic_v1，不混入最终对照；已知搜索调用{revision['completed_search_calls']}次、开发固定重放{revision['control_replay_calls']}次，另{len(revision['interrupted_runs'])}次被中断，尚未落盘的调用只能报0—{revision['interrupted_calls_unknown_upper_bound']}的上界，不能声称精确成本。测试当时尚未评分。旧轨迹、源码快照、失败和中断均保留。"]
    lines+=['',f"共同准备拟合{len(prepared['common_training_fits'])}次，准备耗时{prepared['total_preparation_seconds']:.2f}秒（包含读写）。固定开发重放另{controls['replay_adapter_calls']}次适配器调用/{controls['replay_seconds']:.2f}秒；不能隐去，也不计为搜索优势。独立审计另见audit_cost.json。并发计时只作本机描述。单次xsdba注册调用内部包括训练和调整，耗时完整计入；同调用数不等于同CPU工作量，不声称精确等算力。", 
      '所有72次搜索的父图策略、同分接受、总账、探测期隔离均验收；108份预测独立真实重放并以NumPy重新按年计算测试RMSE一致。最终值有限、非负、可见站点原值保护，数据与源码哈希未变。30份仅降水SWAT+格式输出（每份1096日）回读通过；没有编造其他天气，也没有运行SWAT径流模型。','',
      '审计脚本V2仅纠正一个过严断言：最佳父图可能是零科学节点的原始源，联合修复应核对移除数等于父图实际节点数，而非要求每次都大于零。原审计脚本和冻结清单保留，搜索、评分、数据和预算未改；见audit_assertion_amendment.json。','',
      '天气导出复用既有SWATPlusWriter；首版研究导出辅助脚本误引用Adapter类的方法，已用export_unseen_precipitation_handoff.py纠正本地绑定，原冻结脚本和失败日志保留。只影响格式导出，不改任何搜索、预测或测试指标。','',
      '## 可以与不能证明什么','',
      '可以证明：本轮公共链路中的多源响应评分确实产生差异并影响优先级；联合修复不是换名的单节点编辑；同分增肥被拒绝；计算成本完整公开。效果强弱必须以以上固定和随机/消融结果分别判断。',
      '不能证明：三画像已学习真实SWAT敏感性、所有任务都可无标签自动评估、随机必然较差、ALNS保证全局最优，或新机制在所有流域有稳定收益。评分负责引导提议，最终选图仍以隔离验证误差为主；这一路线没有偷偷换成学习排序模型。',
      '研究建议：仅当多站点对固定与随机的效果和成本同时有支持时再扩大主张。若无优势，保留本轮负结果，下一轮另行冻结协议；不得在这份测试集上循环调至有利。','',
      '## 文件与来源','',
      '- 证据目录：output/unseen_portrait_response_20261028；旧数据机制：output/portrait_response_20261028/mechanism_v3。',
      '- source_manifest.json/source_frozen、input_hashes.json、protocol.json、controls_frozen.json、predictions_frozen.json及independent_audit.json为冻结与验收入口。',
      '- 公共实现：field_mutation.py、composition_optimizers.py、autonomous_decision.py、dag_response.py、scientific_state.py、search_accounting.py；旧日恢复和画像函数复用。',
      '- https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt',
      '- https://www.bom.gov.au/climate/cdo/images/about/rainfall-log.pdf',
      '- https://open-meteo.com/en/docs/historical-weather-api','']
    report=ROOT/'docs/三画像响应与联合变异_实验交付_20261028.md';report.write_text('\n'.join(lines),encoding='utf-8')
    save(OUT/'environment.json',dict(python=platform.python_version(),packages={n:version(n) for n in ['alns','numpy','pandas','xarray','scipy','xsdba','h5netcdf']}))
    master=ROOT/'docs/EcoData Engine项目总交付报告.md';heading='## 第67章 多源响应评分、科学去重与联合变异验收'
    if heading not in master.read_text(encoding='utf-8'):
        shutil.copy2(master,OUT/'master_before_ch67.md')
        r=effects['results'];compar=effects['comparisons']
        text='\n\n'+heading+'（2026-10-28）\n\n'
        text+='复用平台PortraitMutation公共入口，修复同误差增肥接受、完整科学状态去重，新增多源遮挡响应评分和真实联合破坏—修复。保留来源/校正/单位/区间/拟合状态，未知不填；独立遮挡缓存从根重执行。ALNS与随机统一最好父图及180新增注册算法调用总账，探测/失败/部署计入。50项定向回归与旧Robit三场景真实链路验收通过。\n\n'
        text+=f"新数据为三处西澳GHCN雨量站＋ERA5降水；2014—2017拟合、2018—2019开发、2020—2021验证、2022—2024测试。3/14日遮挡、3种子、4搜索组共72次，另保留开发固定/单源/旧转移；108份预测先冻结后测试。主年度RMSE：完整{r['full']['RMSE']:.6f}、随机{r['random']['RMSE']:.6f}、去响应{r['no_response']['RMSE']:.6f}、去联合{r['local_only']['RMSE']:.6f}、共同固定{r['global_fixed']['RMSE']:.6f}、逐案例固定{r['case_fixed']['RMSE']:.6f} mm/day。完整对随机改善{compar['random']['improvement_percent']:.3f}%，对共同固定{compar['global_fixed']['improvement_percent']:.3f}%，对逐案例固定{compar['case_fixed']['improvement_percent']:.3f}%；负数为退步，全部保留。\n\n"
        text+='完整成本和逐种子负结果见详细报告。模型准备画像权重仍为声明先验、总量偏差为输入代理，不是SWAT实测响应；仅3站，不能推断普遍优越或认证全局最优。108预测独立重放、信息隔离、父图/接受/总账/哈希审计通过；30份仅降水输出每份1096日回读通过，不伪造其他天气、不运行原生SWAT。详见docs/三画像响应与联合变异_实验交付_20261028.md及output/unseen_portrait_response_20261028。\n'
        with master.open('a',encoding='utf-8') as f:f.write(text)
    zip_path=ROOT/'output/三画像响应与联合变异_完整交付_20261028.zip'
    files=[p for p in OUT.rglob('*') if p.is_file()]
    files+=[p for p in diagnostic.rglob('*') if p.is_file()]
    files+=list((ROOT/'data/wa_precipitation_2014_2024').rglob('*'))
    mech=ROOT/'output/portrait_response_20261028'
    files+=[p for p in (mech/'mechanism_v3').rglob('*') if p.is_file()]
    files+=[p for p in (mech/'before').rglob('*') if p.is_file()]
    files+=[mech/n for n in ['before_hashes.json','mechanism_acceptance.json','mechanism_source_hashes.json','tests_final_v4.log','tests_final_v4.xml']]
    files+=[report,master,ROOT/'docs/三画像响应与联合变异_冻结协议_20261028.md']
    files+=[ROOT/'tools'/n for n in ['run_response_mechanism_acceptance.py','acquire_unseen_precipitation.py','report_unseen_portrait_response.py','audit_unseen_data_contract.py','audit_unseen_portrait_response_v2.py','export_unseen_precipitation_handoff.py']]
    files+=[ROOT/'tests'/n for n in ['test_dag_response_mechanisms.py','test_field_mutation.py','test_series_fusion.py',
        'test_parallel_composition_scoring.py','test_composition_response.py','test_registered_joint_scoring.py']]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in hashes:z.write(ROOT/name,name)
        z.writestr('PACKAGE_SHA256.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in hashes.items())
    save(OUT/'delivery_receipt.json',dict(status='PASS',files=len(hashes),zip=str(zip_path),bytes=zip_path.stat().st_size,
        sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),all_zip_members_readback=True))
    print('REPORT AND VERIFIED PACKAGE',str(report),str(zip_path),len(hashes),flush=True)


if __name__=='__main__':main()
