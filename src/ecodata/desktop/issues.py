"""Present existing validator evidence without inventing diagnoses or repairs."""


def item(stage,target,problem,action=None,status='待处理'):
    return dict(stage=stage,target=str(target or '当前输入'),status=status,problem=str(problem),
        action=str(action or '后台未提供更具体的补充步骤；请根据此项原始原因核对相应文件或科学定义，不能自动猜测。'))


def checks(rows,failures=()):
    from .discovery import deferred_quality_check
    result=[]
    actions={
        'time_support_unknown':'提供该来源的时间统计含义、原始请求回执或明确区间定义，不能只比较日期标签。',
        'unit_unknown':'补充目标字段的原始单位或可信元数据，不根据数值大小猜测。',
        'unit_conflict':'核对两来源单位和物理量；确认存在合法换算后再处理。',
        'quantity_conflict':'核对变量实际物理含义；不同物理量不能直接作为同变量融合。',
        'disjoint_time':'提供覆盖共同时间窗口的数据，或明确调整任务时间范围；平台不补造重叠记录。',
    }
    for row in rows:
        if row.get('status') in {'通过','无需新增处理'} or row.get('next_step')=='original_workbench':continue
        if deferred_quality_check(row,rows):continue
        if row.get('disposition','').startswith('SKIPPED_'):
            result.append(item('自动排除的建议',row.get('candidate'),row.get('reason'),'无需补填；保留原数据，继续其他可执行步骤。','已自动跳过'))
            continue
        result.append(item('输入与任务核验',row.get('candidate'),row.get('reason') or row.get('status','核验结果未知'),
            row.get('action_required') or actions.get(row.get('block_code')),row.get('status','待处理')))
    for row in failures:
        if not isinstance(row,dict):
            result.append(item('检查或执行',None,row));continue
        if row.get('issues'):
            for detail in row['issues']:
                detail=dict(detail)
                if detail.get('target') in {None,'当前输入'}:detail['target']=row.get('task') or '当前输入'
                result.append(detail)
            continue
        result.append(item(row.get('stage','检查或执行'),row.get('task') or row.get('ids') or row.get('batch'),
            row.get('reason','后台未提供失败原因'),row.get('action_required')))
    return result


def exception(exc):
    rows=_exception(exc)
    from ecodata.core.recovery import descriptions
    return [item('自动补救',e['source'],e['reason'],'其他合法步骤可继续；原始数据保留。',e['status'])
        for e in descriptions(getattr(exc,'automatic_recovery',None))]+rows


def _exception(exc):
    decisions=getattr(exc,'checked_decisions',None)
    if isinstance(decisions,list) and checks(decisions):return checks(decisions)
    report=getattr(exc,'completeness',None)
    if isinstance(report,dict):
        from ecodata.interface.engineering_completeness import LABELS
        rows=[item('工程依赖',r['item'],r['basis'],r.get('action'),LABELS.get(r['status'],r['status']))
              for r in report.get('findings',[]) if r.get('blocks_run')]
        if rows:return rows
    report=getattr(exc,'report',None)
    if isinstance(report,dict):
        return [item('处理条件',report.get('status'),report.get('reason') or str(exc),report.get('action'))]
    return [item('当前操作',None,str(exc))]


def recommendation_checks(rows):
    from .fallback_evidence import eligible
    from .release_gate import allowed, explanation, has_search_continuation

    result = []
    for row in rows:
        release = row.get("execution_release")
        if row.get("baseline_block_reason"):
            result.append(item("算法验收", row["task"], row["baseline_block_reason"],
                               "在方案卡片查看融合候选的分时段试跑记录。"))
            continue
        if not release or row.get("advisory_only") or allowed(row):
            continue
        if release["status"] == "released" and row.get("requires_final_confirmation"):
            result.append(item("算法验收", row["task"], "留出评分已通过，等待 DeepSeek 最终确认。",
                               "在方案卡片点击「重新请求最终推荐」。"))
        elif release.get("blockers"):
            action = ("此推荐不放行；可勾选同任务的合法候选变异，最终仍须复检。"
                      if has_search_continuation(rows, row["task"]) else
                      "工作台会在抽检驳回后自动提交全量扰动副本，完成后查看复验状态。"
                      if eligible(row) and release["status"] == "failed" else
                      "在方案卡片补充证据或提供新的独立数据。" if eligible(row) else
                      "在方案卡片查看数据依据与检查前提。")
            result.append(item("算法验收", row["task"], explanation(release), action))
    return result


def output(result,review=None):
    rows=checks([],result.get('failed_tasks',[]))
    manifest=result.get('manifest',{})
    from ecodata.core.recovery import descriptions
    for entry in descriptions(result.get('automatic_recovery') or manifest.get('automatic_recovery')):
        rows.append(item('自动补救',entry['source'],entry['reason'],'原始文件不变；查看下载包中的补救依据及位置掩码。',entry['status']))
    continuation=result.get('continuation') or manifest.get('continuation',{})
    for entry in continuation.get('decisions',[]):
        status={'BLOCK_STEP':'仅该步骤受阻','SKIP_STEP':'已自动跳过','AUTO_COMPLETED':'已自动补齐',
                'FALLBACK':'已安全回退','CONTINUE_WITH_WARNING':'提示，不中断其他步骤'}.get(entry['action'],entry['action'])
        rows.append(item(entry['stage'],entry['target'],entry['reason'],
            '其他已核验步骤继续；原始数据保留。'+(' 本步骤条件未满足，未伪造完成。' if entry['action']=='BLOCK_STEP' else ''),status))
    gate=manifest.get('final_check',{})
    admission=manifest.get('preparation',{}).get('source_admission',{})
    requirements=[*admission.get('required_information',[]),*manifest.get('requirements_check',{}).get('missing_or_conflicting',[])]
    for entry in requirements:
        if entry.get('message'):rows.append(item('输入条件',entry.get('resource') or entry.get('resources') or entry.get('code'),entry['message'],entry.get('action')))
    for missing in result.get('local',{}).get('model_input',{}).get('whole_model_missing',[]):
        rows.append(item('模型输入',None,missing))
    for stage,key in [('原生格式','format'),('工程引用','project')]:
        section=gate.get(key,{})
        errors=section.get('errors',[]) or section.get('validation',{}).get('errors',[])
        for error in errors:rows.append(item(stage,None,error))
        if section.get('status') in {'FAIL','UNKNOWN','ERROR'} and not errors:
            rows.append(item(stage,None,section.get('reason') or '检查器尚未给出可验收结论'))
    native=gate.get('native',{})
    if native.get('status') in {'NOT_RUN','FAIL','ERROR'}:
        rows.append(item('模型试读',native.get('engine') or native.get('engine_id'),native.get('reason') or '原生试读未通过，详见模型日志',status='未完成'))
    if review:
        for key,stage in [('missing_data','待补数据'),('missing_information','待核实信息')]:
            for reason in review.get(key,[]):rows.append(item(stage,None,reason,status='待核实'))
    return rows


def unique(rows):
    seen=set();result=[]
    for row in rows:
        key=(str(row.get('stage','')),str(row.get('candidate_id',row.get('target',''))),str(row.get('problem','')),str(row.get('action','')))
        if key not in seen:seen.add(key);result.append(row)
    return result


def bind_sources(rows,candidates,scan):
    """Attach existing catalogue identities; do not infer causes from filenames."""
    from pathlib import PurePosixPath
    bound=[]
    for original in rows:
        row=dict(original);pair=candidates.get(row['target']) or candidates.get(row['target'].split(':',1)[0])
        if pair and scan:
            try:
                paths=[scan['rows'][endpoint[0]]['name'] for endpoint in pair[:2]]
                fields=[endpoint[2]['name'] for endpoint in pair[:2]]
                row['source_paths']=paths
                row['candidate_id']=row['target']
                row['target']+=' · '+' ↔ '.join(PurePosixPath(path).name+' ['+field+']' for path,field in zip(paths,fields))
            except (KeyError,IndexError,TypeError):row['source_binding']='未能从当前目录恢复文件映射'
        bound.append(row)
    return bound
