"""Readable desktop report for completed local searches."""
import json

from .output_review import collect_steps


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
        lines.append("将降水率积分为共同 UTC 日期的日累计量，再保守对齐到较粗网格，以统一空间尺度。")
    source=" / ".join(dict.fromkeys(result["source_formats"]))
    lines.append(f"格式 · {source} → {result['target_format']} · {len(result['files'])} 个数据文件")
    local=result["local"]
    if not local.get("native_ready"):
        lines.append("模型使用 · 已通过本地读回，可用作数据准备产物。"
                     + (local.get("missing") or "目标模型工程的完整输入和配置仍待核验。"))
    return "\n".join(lines)
