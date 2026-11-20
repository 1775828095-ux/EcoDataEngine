"""Readable desktop report for completed local searches."""
import json

from .output_review import collect_steps


def candidate_options(result,limit=10):
    """Only already measured candidates, never an invented improvement label."""
    from .output_review import searches
    options=[]
    for task,search in searches(result).items():
        if not search.get('search_folder'):continue
        seen=set()
        for row in search.get('rankings',[]):
            behavior=row.get('same_predictions_as') or row['key']
            if behavior in seen:continue
            seen.add(behavior)
            options.append(dict(task=task,search_folder=search['search_folder'],key=row['key'],
                metrics=row['metrics'],steps=row.get('steps') or [row['algorithm']],parameters=row.get('parameters',{}),
                selected=row['selected'],phase=row['phase'],
                reference_basis=search.get('evidence',{}).get('reference_basis','UNKNOWN')))
            if len(seen)>=limit:break
    return options


def export_evaluated_candidate(folder,key,destination):
    """Deploy a user-chosen, passed graph; keep the default engineering ZIP intact."""
    from pathlib import Path
    from tempfile import TemporaryDirectory,NamedTemporaryFile
    import hashlib,math,zipfile
    from ecodata.algorithms.platform_search import SearchInput,execute_evaluated_candidate
    folder=Path(folder).resolve();destination=Path(destination).resolve()
    def load(path):return json.loads(path.read_text(encoding='utf8'))
    rows=load(folder/'candidate_results.json');summary=load(folder/'summary.json')
    selected=next((r for r in rows if r['key']==key and r['status']=='PASS'),None)
    if selected is None:raise ValueError('所选方案没有真实通过的数值评价记录')
    if not selected.get('fixed_support_sha256') or selected['fixed_support_sha256']!=summary['best'].get('fixed_support_sha256'):
        raise ValueError('所选方案与默认方案的评价队列不一致')
    if not selected['metrics']['count'] or not all(math.isfinite(selected['metrics'][m]) for m in ('RMSE','MAE')):
        raise ValueError('所选方案缺少有效评价指标')
    protocol=load(folder/'request.json');inputs=[];protected={folder/'product.nc'}
    replay=folder/'replay_inputs'
    if (replay/'index.json').exists():
        records=load(replay/'index.json')
    else:
        replay=folder.parent/'inputs'
        records=[dict(path=r['source_filename'],filename=r['source_filename'],role=r['role'],
            metadata=r.get('declarations'),sha256=r['sha256']) for r in protocol['inputs']]
    for record in records:
        path=(replay/record['path']).resolve()
        if not path.is_relative_to(replay.resolve()):raise ValueError('候选输入引用越界')
        protected.add(path)
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('候选输入指纹已变化，不能复用旧评分')
        inputs.append(SearchInput(record['filename'],record['role'],raw,record.get('metadata')))
    if destination in protected or destination.is_relative_to(folder):
        raise ValueError('不能覆盖搜索证据或输入')
    with TemporaryDirectory(prefix='ecodata-candidate-') as temporary:
        receipt=execute_evaluated_candidate(inputs,folder,key,Path(temporary)/'deployment')
        if receipt.get('status','PASS')!='PASS':raise ValueError('所选方案产品未通过质量核验')
        import numpy as np
        import xarray as xr
        with xr.open_dataset(folder/'product.nc') as default,xr.open_dataset(receipt['product']) as chosen:
            left,right=xr.align(default[protocol['request']['variable']],chosen[protocol['request']['variable']],join='exact')
            values=right.values-left.values;finite=values[np.isfinite(values)]
            comparison=dict(same_values=bool(np.array_equal(left.values,right.values,equal_nan=True)),
                max_abs_difference=float(np.max(np.abs(finite))) if len(finite) else None,
                scope='同一输入上的产品数值差异，不是独立精度评价')
        report=dict(selection='USER_SELECTED',automatic_winner=summary['best']['key'],chosen_key=key,
            validation_metrics=selected['metrics'],reference_basis=summary['selection_comparison'].get('selection_reference_basis',summary['selection_comparison']['reference_basis']),
            same_evaluation_support=True,original_engineering_archive_changed=False,native_writeback=False,native_model_executed=False,
            note='所选方案的数据产品；不替换默认工程包，不代表独立实测精度提升',deployment=receipt,
            default_product_comparison=comparison)
        with NamedTemporaryFile(dir=destination.parent,suffix='.zip',delete=False) as stream:stage=Path(stream.name)
        try:
            with zipfile.ZipFile(stage,'w',zipfile.ZIP_DEFLATED) as archive:
                archive.write(receipt['product'],'selected-product.nc')
                archive.writestr('selection.json',json.dumps(report,ensure_ascii=False,indent=2,default=str))
                archive.writestr('graph.json',json.dumps(receipt['graph'],ensure_ascii=False,indent=2))
            with zipfile.ZipFile(stage) as archive:
                if archive.testzip():raise ValueError('所选产品ZIP回读失败')
            stage.replace(destination)
        finally:stage.unlink(missing_ok=True)
    return destination


def ranked_candidates(folder, search):
    path = folder / "candidate_results.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf8"))
    passed = [row for row in rows if row["status"] == "PASS"]
    graphs=json.loads((folder/"graphs.json").read_text(encoding="utf8"))
    selected=search["best"]["key"]
    # The original winner is authoritative, including ties within its numerical tolerance.
    passed.sort(key=lambda row: (
        row["key"] != selected, row["metrics"]["RMSE"], row["algorithm_nodes"], row["key"]))
    return [
        dict(rank=rank, key=row["key"], family=row["source_family"],
             phase=row["phase"], metrics=row["metrics"], algorithm=graphs[row["key"]]["operation"],
             steps=collect_steps(row["key"], graphs), parameters=graphs[row["key"]].get("parameters", {}),
             same_predictions_as=row.get('same_validation_predictions_as'),
             selected=row["key"]==selected)
        for rank,row in enumerate(passed,1)
    ]


def describe(result):
    from .recommendations import recipe_name

    lines = ["本地处理完成 · ZIP 可导出"]
    searches = result.get("grid_searches") or result.get("batch_searches") or {}
    if result.get("search"):
        searches = {"本次任务":dict(best=result["search"]["best"],
                    selected_graph=result["search"]["selected_graph"],
                    evidence=result["fusion_evidence"], rankings=result["rankings"],
                    confirmed_initial=result["search"].get("confirmed_initial", []),
                    portrait_strategy=result["search"].get("portrait_strategy"),
                    pareto_comparison=result.get("pareto_comparison"))}
    family_labels = {"SINGLE_SOURCE":"单源", "MULTISOURCE":"多源"}
    def algorithm_name(operation, family):
        if family == "SINGLE_SOURCE":
            single_names = {"spatial.residual_idw": "残差反距离权重校正",
                            "spatial.blend": "同源估计加权组合",
                            "spatial.source": "保留单源"}
            if operation in single_names:
                return single_names[operation]
        return recipe_name([operation])

    for task, search in searches.items():
        best = search["best"]
        evidence = search["evidence"]
        records = search["selected_graph"]["records"]
        selected = records.get(best["key"],{})
        op = selected.get("operation","未知")
        parameters = selected.get("parameters",{})
        family = family_labels.get(evidence.get("selected_family"),"未确定")
        lines.extend(["",f"{task} · 本次搜索找到的最优方案：{family} / {algorithm_name(op,evidence.get('selected_family'))}",
                      f"选中方案 RMSE：{best['metrics']['RMSE']:.4g}"])
        if parameters:
            lines.append("参数："+json.dumps(parameters,ensure_ascii=False))
        for seed in search.get("confirmed_initial", []):
            status = "评分通过" if seed["status"]=="PASS" else "评分未通过"
            lines.append(f"AI 初始候选 · {recipe_name([seed['algorithm']])} · {status} · {seed['key'][:10]}")
        strategy=search.get("portrait_strategy")
        if strategy:
            lines.append(f"模型策略 · {strategy['version']} / {strategy['scenario']} · "
                         f"引导 {strategy['portrait_proposals']} 次变异提议，改变 {strategy['changed_choices']} 次候选选择")
        for label,metrics in evidence["selection_scores"].items():
            if metrics:
                lines.append(f"{family_labels[label]}最佳：RMSE {metrics['RMSE']:.4g} · "
                             f"MAE {metrics['MAE']:.4g} · 样本 {metrics['count']}")
        if evidence.get("reference_basis")=="PRIMARY_GRID_RECONSTRUCTION_PROXY":
            lines.append("评分参考：格点代理。结果用于本次选优，实测精度仍待验证。")
        elif evidence.get('reference_basis')=='REDUCED_SCALE_PRODUCT_RECONSTRUCTION':
            protocol=evidence['evaluation_protocol']
            lines.extend(['评分依据：两个来源都先按面积聚合变粗，再恢复未提供给算法的原细产品；不是对主来源自我打分。',
                '参考来源：'+protocol['reference_source']+'（按原生网格粗细确定，不按误差挑选）；它仍是产品代理，不是独立实测真值。',
                f"评分样本 {protocol['selection_count']}；冻结后复核样本 {protocol['test_count']}；粗尺度误差另列，不与重建误差加权混排。",
                '多源没有额外奖励；目标细网格与质量门仍为共同硬约束。'])
            for item in evidence.get('source_ablations',[]):
                metric=item.get('metrics')
                lines.append('去掉来源 '+item['omitted_source']+'：'+(f"最佳已执行路线 RMSE {metric['RMSE']:.4g} · MAE {metric['MAE']:.4g}" if metric else '未评价'))
            paired=evidence.get('paired_selection')
            if paired:
                change=paired['RMSE_change_percent']
                lines.append(('多源相对最佳单源：RMSE变化 '+f'{change:+.2f}%' if change is not None else '单源误差为0，不计算相对百分比')+
                    f"；改善的空间块 {paired['improved_blocks']}/{len(paired['blocks'])}。负数表示误差下降。")
                lines.append('空间块差异仅作描述；独立日期不足，不能据此声称稳定或显著增益。')
            for label,score in evidence.get('coarse_diagnostics',{}).items():
                if score:
                    lines.append(f"{family_labels.get(label,label)}的粗主产品诊断（不参与选图）：RMSE {score['RMSE']:.4g} · 样本 {score['count']}")
        elif not evidence.get("independence_verified"):
            lines.append("参考数据的独立性未确认。")
        holdout=evidence.get("holdout_scores", {})
        if any(holdout.values()):
            lines.append("xskillscore · 方案冻结后的留出复核")
            for label,key in (("单源","single_source_baseline"),("多源","multisource_candidate")):
                score=holdout.get(key)
                if score:
                    lines.append(f"  {label} · RMSE {score['RMSE']:.4g} · MAE {score['MAE']:.4g} · "
                                 f"偏差 {score['Bias']:.4g} · 样本 {score['count']}")
        comparison=search.get("pareto_comparison") or {}
        if comparison.get("candidates"):
            lines.append("pymoo · 同批选优样本上，无法同时改善两项表现的代表方案")
            representatives={}
            for row in comparison["candidates"]:
                score=row["metrics"]
                representatives.setdefault((row["family"],score["RMSE"],score["MAE"]),row)
            for row in list(representatives.values())[:5]:
                score=row["metrics"]
                lines.append(f"  {family_labels.get(row['family'],row['family'])} · {algorithm_name(row['algorithm'],row['family'])} · "
                             f"RMSE {score['RMSE']:.4g} · MAE {score['MAE']:.4g}")
        lines.append("原版评分排名（前 10 项；完整排名见导出报告）")
        for row in search["rankings"][:10]:
            score=row["metrics"]
            lines.append(f"{row['rank']:>2}. {family_labels.get(row['family'],row['family'])}  "
                         f"{algorithm_name(row['algorithm'],row['family'])} · {row['key'][:10]}  "
                         f"RMSE {score['RMSE']:.4g}  MAE {score['MAE']:.4g}"
                         + ("  ← 原版选中" if row["selected"] else ""))
    lines.extend(["","转换与导出理由",result["local"].get("reason","")])
    if result["mode"].startswith("independent_"):
        lines.append("将降水率积分为共同 UTC 日期的日累计量，按本次已核验的目标网格对齐；分辨率与掩码见产品回读记录。")
    source=" / ".join(dict.fromkeys(result["source_formats"]))
    lines.append(f"格式 · {source} → {result['target_format']} · {len(result['files'])} 个数据文件")
    local=result["local"]
    if not local.get("native_ready"):
        lines.append("模型使用 · 已通过本地读回，可用作数据准备产物。"
                     + (local.get("missing") or "目标模型工程的完整输入和配置仍待核验。"))
    return "\n".join(lines)
