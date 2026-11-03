"""Deliver the audited result without changing choices or discarding failures."""
from pathlib import Path
import json
from run_current_input_effect_retest import ROOT,OUT,BEFORE,cases,load


def run():
    audit=load(OUT/'independent_audit.json');assert audit['status']=='PASS'
    effect=load(OUT/'effect_summary.json');before=load(BEFORE/'effect_summary.json');runs=load(OUT/'runs_audit.json')
    names=dict(random='当前输入自主选择＋随机变异',none='当前输入自主选择＋不变异',
        development_fixed='原2015开发期最强固定组合',fine_time_linear='细源时间线性插值',
        fine_time_nearest='细源时间最近邻',coarse_only='仅粗源重建',anchor_residual='原始锚点残差',
        anchor_residual_physical='锚点残差＋范围投影')
    lines=['# 修复后同数据自主随机变异效果复测（2026-10-29）','',
        '复用昨日黄山小流域Landsat约30m、MODIS约250m/8天NDVI，三个48×48区域，交错/连续中段隐藏约一半细源日期，共6个案例。未下载或换数据，未新增算法，未修改生产评分、权重和数值内核。',
        '', '## 冻结方案与实际决策','',
        '随机组通过平台公共文件入口进入process_current_input，由当前可见数据自动遮挡评价、三画像筛选、随机变异、真实执行和部署。每个案例种子11/29/47；不提供指定处理图、人工胜出算法或候选模板池。实际optimizer.backend=random，不是ALNS。提议可因接口/物理约束提前拒绝；只有真实执行并通过完整目标质量门的图才有合法数值成绩，缓存和科学状态等价复用不伪称新执行。',
        '', '不变异组为公共PortraitMutation的显式mutation_control=none消融：同一完整算法库、相同当前输入、相同搜索/确认遮挡和初始化，停止于初始化选优；不改成固定人工算法。六个固定对照完整保留，原最强固定图仍按2015开发期冻结，不依2018成绩挑选。共36次自动选择，以及36次固定图部署。',
        '', '搜索沿用上一轮宽裕上限：每次1800注册数值调用、400000提议、600探测调用、3600秒服务保护。包含原失败、探测和原部署调用；不变异自然耗用较少。6个并行工作进程。',
        '', '测试误差未打开时，发现长搜索耗尽部署预留时间的资源收尾失败。冻结统一部署补充协议：全部36个随机/不变异运行均由公共GraphMutation.rank和deploy_ranked读取保存的真实评价、按原平台排序自动部署，每次额外10000调用/3600秒，与原固定图部署额度相同。不增加任何搜索、候选、训练、权重或参数，不人工替换图。原有成功与失败记录均保留；额外部署全部计入成本。不能声称同总预算效率优势，也不能把补充部署后的完整率冒充原服务预算内完整率。',
        '', '历史训练/开发画像和状态复用，不在测试标签上拟合或校准。当前输入原本可见部分划出互斥搜索/确认集合，两者均从选择根输入移除，副本同源遮挡；天然缺失不作真值。确认误差不参与选图。整个决策阶段由Python文件访问审计阻止读取外部development/validation/test_labels。全部产品和决策哈希封存之后，单独读取隐藏日期参考统一评价。',
        '', '这是2018已被研究者查看过后的回顾复测。旧版在2016选图后冻结迁移；新版按当前2018可见部分自主重选。因此前后差异是整套运行方式的变化，不能归因于某一行修复或三画像单独贡献，也不属于新盲测。',
        '', '## 隐藏日期实际效果','',
        '主指标：先对每个隐藏日期计算固定QA支持上的RMSE，再对日期、案例、种子等权平均。越低越好。缺少预测即失败，不跳过困难像元或失败种子。',
        '', '|方案|隐藏日期等权RMSE|MAE|状态|','|---|---:|---:|---|']
    for arm,row in effect.items():
        value=f"{row['RMSE']:.8f}|{row['MAE']:.8f}|完整" if row['status']=='PASS' else f"—|—|不完整：{row['passed']}/{row['expected']}"
        lines.append(f'|{names[arm]}|{value}|')
    lines+=['','## 是否提升','']
    for row in audit['comparisons']:
        ref=row['reference']
        label=names.get(ref,{'before_legacy_random':'修复前旧库随机变异','before_expanded_random':'修复前同扩充库随机变异',
            'before_expanded_none':'修复前扩充库不变异'}.get(ref,ref))
        if row['status']=='PASS':
            detail=f"，{row['case_wins']}/{row['cases']}案例更低" if 'case_wins' in row else ''
            lines.append(f"- 相对{label}：RMSE改善{row['improvement_percent']:.4f}%{detail}；负值表示变差。")
        else:lines.append(f'- 相对{label}：至少一侧不完整，不给出排除失败的总体改善率。')
    lines+=['','|案例|修复后随机（3种子均值）|修复后不变异|原开发固定|修复前同扩充库随机|',
        '|---|---:|---:|---:|---:|']
    for c in cases():
        pair=next(r for r in audit['paired_before_after'] if r['case']==c)
        vals=[effect[m].get('by_case',{}).get(c) for m in ['random','none','development_fixed']]
        vals.append(pair.get('before'))
        lines.append('|'+c+'|'+'|'.join('—' if value is None else f'{value:.8f}' for value in vals)+'|')
    lines+=['','修复前同扩充库随机只有16/18部署成功，原结果保留，不补填整体均值。完整匹配案例仅作分案例描述，不能删除原失败案例后另算“总体改善”。',
        '', '|方案|三种子整体RMSE|种子间标准差|总注册调用|总服务秒数|','|---|---|---:|---:|---:|']
    for m in ['random','none']:
        row=effect[m];cost=audit['costs'][m];values=', '.join(f'{v:.8f}' for v in row.get('seed_RMSE',[])) or '不完整'
        sd='—' if row.get('seed_SD') is None else f"{row['seed_SD']:.8f}"
        lines.append(f"|{names[m]}|{values}|{sd}|{cost['calls']}|{cost['service_seconds']:.2f}|")
    lines+=['','## 负结果、真实使用与边界','',
        f"原3600秒服务预算内自动处理失败{audit['original_budget_failures']}项，详见original_budget_failures.json；最终产品表对应统一追加部署后的结果。资源收尾缺陷本轮未修改生产源码，不能据此说平台所有收尾边界已修复。"]
    for failure in load(OUT/'failures.json'):
        lines.append(f"- {failure['case']} / {failure['method']} / {failure['stage']}：{failure.get('error',failure.get('details',{}).get('status','FAILED'))}")
    unsupported=sorted({r.get('response_unsupported') for r in runs if r.get('response_unsupported')})
    if unsupported:
        lines+=['','本轮栅格的真实数值恢复误差用于选优，但既有DAGResponse的完整实测三路响应仍限于日点序列；栅格响应保持UNKNOWN，未编造响应分或自动补学模型偏好。故不能把本轮结果解释成三画像收益机制全部获证。具体状态：'+ '; '.join(unsupported)]
    lines+=['','随机变异不保证全局最优；本轮未穷举开放图空间，不宣称全局认证。重复节点、参数拟合与能力互补的贡献未单独隔离。独立确认仅为已知观测遮挡误差，最终主结论使用隔离的隐藏日期误差。',
        '', '三个区域空间相依，同一流域、一个测试年；种子和像元不是独立流域，不报告伪独立显著性。NDVI融合重建不是SWAT+/WRF实际模型收益。',
        '', '## 独立核对','',
        f"核对{audit['jobs_checked']}/{audit['expected_jobs']}次自动运行，{audit['products_checked']}份产品独立复算主指标及原观测保护；原输入、源码和历史对照哈希未变。全部对照/种子使用相同内部遮挡，固定参考支持一致，决策阶段外部真值读取为零。失败共{audit['failure_count']}项，均保留。",
        '', '协议、冻结源码、36次公共入口配置/轨迹/逐图调用、固定图部署、封存产品、指标、负结果和独立审计：output/current_input_effect_retest_20261029/。']
    path=ROOT/'docs/修复后同数据自主重选效果复测_20261029.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(path)


if __name__=='__main__':run()
