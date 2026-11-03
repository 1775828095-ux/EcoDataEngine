"""Summarize completed, independently audited same-data comparisons."""
from pathlib import Path
import json
from run_library_update_retest import ROOT,OLD,OUT,cases,load


def run():
    audit=load(OUT/'independent_audit.json');assert audit['status']=='PASS'
    results=load(OUT/'effect_summary.json');failures=load(OUT/'deployment_failures.json')
    names={'legacy_random':'更新前＋随机变异','expanded_random':'更新后＋随机变异',
        'legacy_none':'更新前＋不变异','expanded_none':'更新后＋不变异',
        'development_fixed':'原开发期最强固定组合','fine_time_linear':'细源时间线性插值',
        'fine_time_nearest':'细源时间最近邻','coarse_only':'仅粗源重建',
        'anchor_residual':'原始锚点残差','anchor_residual_physical':'锚点残差＋范围投影'}
    lines=['# 算法库更新：同数据复测（2026-10-29）','',
        '复用昨日半日期隐藏实验：同一流域三个48×48区域，Landsat约30m与独立MODIS约250m/8天NDVI；交错及连续中段隐藏两种方式，共6个案例。原训练/开发/验证/测试边界保留。',
        '', '本轮是已看过2018结果后的回顾复测，不是新的盲测。2016选图；全部选图及2018输出封存后，才在本轮读取2018参考。没有用测试成绩调整算法、规则或权重。',
        '', '## 对照与执行','',
        '更新前库成员从昨日冻结源码导出108个注册ID；更新后为完整公共注册表118个ID。包含不可用状态的注册ID，不等于本任务可调用方法数。两组共用当前数值内核、搜索、画像及质量门，差别是是否纳入新增10个节点。实验客户端只冻结旧库对照，不给新库指定流程。',
        '', '两库均运行随机变异及不变异初始化选优，种子11/29/47，共72次搜索。真实算法调用上限1800、提议上限400000、服务保护上限3600秒、画像探测调用上限600；数值调用额度为昨日十倍。本轮以精度为目标，不以计算效率宣称优势，不把更大预算的提升单独归因算法库。',
        '', '每个计分图真实执行；未完成图不冒填RMSE。原开发期固定组合及所有固定对照保持不变，失败种子、非法候选和部署失败均保留。并行调整前的中断轨迹在diagnostic_initial_parallelism，不计作正式运行，也不声称其成本为零。',
        '', '## 测试结果','',
        '|方案|隐藏日期等权RMSE|MAE|状态|','|---|---:|---:|---|']
    for key,value in results.items():
        lines.append('|'+names.get(key,key)+'|'+(f"{value['RMSE']:.8f}|{value['MAE']:.8f}|完整" if value['status']=='PASS' else f"—|—|不完整：{value.get('passed',0)}/{value['expected']}通过")+'|')
    lines+=['','## 更新后的收益','']
    for c in audit['comparisons']:
        if c['status']=='PASS':
            lines.append(f"- {names[c['candidate']]}相对{names[c['reference']]}：RMSE改善{c['improvement_percent']:.4f}%，{c['case_wins']}/{c['cases']}案例更低；负值表示更差。")
        else:lines.append(f"- {names[c['candidate']]}与{names[c['reference']]}存在未完成结果，不给出选择性均值。")
    lines+=['','|案例|更新前随机变异|更新后随机变异|原最强固定|','|---|---:|---:|---:|']
    for case in cases():
        case_result=next(r for r in audit['per_case_comparison'] if r['case']==case)
        expanded=(case_result['RMSE']['expanded'] if case_result['status']=='PASS' else None)
        vals=[results['legacy_random'].get('by_case',{}).get(case),expanded,results['development_fixed'].get('by_case',{}).get(case)]
        lines.append('|'+case+'|'+'|'.join('—' if x is None else f'{x:.8f}' for x in vals)+'|')
    complete=[r for r in audit['per_case_comparison'] if r['status']=='PASS']
    lines+=['',f"逐案例也要求三个种子齐全：{len(complete)}/6案例完整，其中更新后{sum(r['improvement_percent']>0 for r in complete)}个改善、{sum(r['improvement_percent']<0 for r in complete)}个变差。不对剩余完整案例另算一个排除失败案例的总体均值。",
        '', '更新后随机搜索两次部署失败均在patch2_contiguous：平台选中的DINEOF图从验证期输入迁移到测试期后，不再满足时刻数/有效观测要求。执行时契约检查正确阻止调用，但现有冻结图部署入口没有自动改选其他合法图；本轮不人工替换。另一失败是原始锚点残差固定对照越过声明范围，单独保留。']
    lines+=['','## 新能力实际使用','', '|新增算法|成功注册调用次数|最终选图出现次数（更新后）|','|---|---:|---:|']
    added=load(OUT/'protocol.json')['added_algorithms']
    for key in added:
        calls=sum(r['count'] for r in audit['newly_added_calls'] if r['algorithm'].removeprefix('framewise.')==key and r['status']=='PASS')
        selected=sum(r['count'] for r in audit['selected_operations'] if r['library']=='expanded' and r['algorithm'].removeprefix('framewise.')==key)
        lines.append(f'|{key}|{calls}|{selected}|')
    complexity=load(OUT/'selected_graph_complexity.json')
    expanded_random=[r for r in complexity if r['library']=='expanded' and r['mode']=='random']
    repeated=sum(bool(r['repeated_operations']) for r in expanded_random)
    largest=max((max(r['repeated_operations'].values(),default=1) for r in expanded_random),default=0)
    lines+=['',f'更新后随机搜索的选图中，{repeated}/{len(expanded_random)}包含重复使用同类操作，单图同一操作最多{largest}个节点。它们由平台生成并真实执行，未人工删改；反复平滑不能被解释成同等数量的不同互补能力。单凭本轮总体收益，不能区分新增单方法、重复平滑、参数选择与多能力协作的贡献。',
        '','上述次数包含真实画像探测及搜索执行，不能当作独立样本。注册不等于适用：STARFM反射率契约不适用于NDVI；AROSICS目前要求至少64×64影像并显式允许配准，本案例为48×48；均值校正需要历史训练配对。本轮不为让它们参加而改变变量含义或放宽条件。其他方法可在前置处理后成为可用候选，未按初始可用性删库。',
        '', '## 核对与结论边界','',
        f"独立核对{audit['jobs_checked']}/{audit['expected_jobs']}次搜索的候选库和统一额度；{audit['independent_primary_metric_checks']}份通过结果重新计算主指标；{audit['visible_observation_protection_checks']}份输出核对原可见观测保护。输入及源码哈希保持不变，所有方法使用固定参考支持。部署/搜索失败记录数：{len(failures)}。",
        '', '三个区域空间相依，同属一个流域；种子和像元不增加独立流域数量。本轮不能证明普遍优势、全局最优、三画像评分单独有效或下游SWAT+/WRF获益。用户上传后自动识别跨文件关系的入口能力不在本轮修改范围。',
        '', '证据目录：output/library_update_retest_20261029。包含冻结协议/源码、72次完整轨迹、逐图真实调用、选图、输出、负结果、逐案例指标、独立审计和原始中断诊断。']
    path=ROOT/'docs/算法库更新同数据复测_20261029.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT',path)


if __name__=='__main__':run()
