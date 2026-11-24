"""Result-card data and a bounded, public-schema-only output review."""
from math import isfinite
from pathlib import PurePosixPath


FAMILY_NAMES = {"SINGLE_SOURCE": "单源", "MULTISOURCE": "多源"}
SELECTION_REFERENCES = {
    'OBSERVATION_HOLDOUT': '留出的观测样本',
    'HELD_STATION_OBSERVATIONS': '留出的站点观测',
    'PRIMARY_GRID_RECONSTRUCTION_PROXY': '主产品网格重建（代理评价）',
    'REDUCED_SCALE_PRODUCT_RECONSTRUCTION': '降尺度产品重建（代理评价）',
    'COARSE_PRODUCT_RECONSTRUCTION_ONLY': '粗网格产品重建（代理评价）',
}
METHODS = {
    "spatial.source": ("保留来源场", "直接使用来源场在目标位置上的估计"),
    "spatial.idw": ("反距离权重插值（IDW）", "用邻近位置按距离加权估计目标位置"),
    "spatial.residual_idw": ("残差反距离权重校正", "把参考样本与背景场的差异按距离传播到目标位置"),
    "spatial.blend": ("加权组合", "按候选权重合并输入估计"),
    "spatial.affine": ("线性偏差校正", "按保留样本估计偏移和比例"),
    "spatial.aux_affine": ("辅助变量线性估计", "用辅助变量和保留样本建立线性估计"),
    "spatial.ratio": ("比例校正", "用保留样本估计比例关系"),
}
FORMATS = {
    ".nc": "NetCDF", ".nc4": "NetCDF", ".csv": "CSV", ".tsv": "TSV", ".xlsx": "Excel",
    ".xls": "Excel", ".tif": "GeoTIFF", ".tiff": "GeoTIFF", ".parquet": "Parquet",
    ".json": "JSON", ".geojson": "GeoJSON", ".txt": "文本", ".pcp": "SWAT+ 降水文本（.pcp）",
    ".cli": "SWAT+ 气象索引（.cli）", ".dat": "模型数据（具体编码待原生核验）", ".in": "模型配置", ".inp": "模型输入",
}
PUBLIC_AXES = {
    "time": "时间", "tstep": "时间", "lat": "纬度", "lon": "经度",
    "x": "横向坐标", "y": "纵向坐标", "row": "网格行", "col": "网格列",
    "lay": "垂直层", "station": "站点", "record": "记录", "point": "点位",
}


def collect_steps(key, graphs):
    operations = []
    seen = set()

    def visit(node):
        if node in seen or node not in graphs:
            return
        seen.add(node)
        row = graphs[node]
        for parent in row.get("parents", []):
            visit(parent)
        if row["operation"] in METHODS:
            operations.append(row["operation"])

    visit(key)
    # Source nodes are inputs when a numerical operation is present.
    return [op for op in operations if op != "spatial.source"] or ["spatial.source"]


def searches(result):
    if result.get("search"):
        return {"本次任务": dict(rankings=result["rankings"], evidence=result["fusion_evidence"],
                                confirmed_initial=result["search"].get("confirmed_initial", []),
                                seed_usage=result.get('seed_usage'))}
    return result.get("grid_searches") or result.get("batch_searches") or {}


def conversion_reason(result, output, axes):
    if result.get('manifest',{}).get('native_writeback_verified') and result.get('manifest',{}).get('changes'):
        return f'已将处理结果写回工程副本的{output}；原始文件保留。实际格式和模型试读分别核验。'
    mode = result.get("mode")
    if mode in {"independent_grid_search", "independent_alignment"}:
        return ("将降水率按经核验的 UTC 日区间积分成日累计量，再保守对齐到共同网格。"
                "NetCDF 把时间、经纬度和降水变量存进同一文件，便于逐日、逐格比较和后续提取。")
    if mode == "native_reference_search":
        return ("按原版质量规则筛选卫星 LAI，在 CMAQ 网格尺度上比较单源与融合，"
                "保留 MCIP 的逐小时变化。NetCDF 保留时间、网格坐标和 LAI 单位，便于后续核验。")
    if output == "NetCDF":
        spatial = ({"纬度", "经度"} <= axes or {"网格行", "网格列"} <= axes
                   or {"横向坐标", "纵向坐标"} <= axes)
        if spatial and "时间" in axes:
            if set(result.get("source_formats", [])) & {"CSV", "TSV", "Excel"}:
                return ("将表格中的时间和位置列保存为 NetCDF 的坐标轴，把各观测列保存为变量。"
                        "这样可按时间或区域读取，也便于核对模型网格和单位。")
            return "NetCDF 将时间、空间坐标、变量和单位一起保存，可按时间或区域读取，便于匹配模型网格。"
        if "时间" in axes:
            return "NetCDF 把时间轴、变量和单位随文件保存，便于按时间读取和核对输入。"
        return "NetCDF 把已核验的维度、变量和单位随文件保存，便于后续程序读取。"
    if output in {"CSV", "TSV", "Excel", "Parquet"}:
        return f"{output} 以表格保存结果，便于按列检查、筛选和导入后续工具。"
    if output == "GeoTIFF":
        return "GeoTIFF 将栅格值与地理定位信息随文件保存，便于在 GIS 中定位和使用。"
    return "按目标流程交付已生成的文件，具体可用性见输出包检查。"


def format_conversion(result):
    """Explain actual written products; a ZIP container is not a data encoding."""
    manifest=result.get('manifest',{})
    changes={r['path'] for r in manifest.get('changes',[]) if r.get('path')}
    source=list(dict.fromkeys(f for f in result.get('source_formats',[]) if f not in {'ZIP','TGZ','TAR','JSON','Markdown'}))
    rows=[];seen=set()
    for task in manifest.get('preparation',{}).get('tasks',[]):
        port=task.get('port',{});target=port.get('resource')
        if target not in changes:continue
        for origin in task.get('sources',[]):
            name=origin.get('resource')
            if not name:continue
            before=FORMATS.get(PurePosixPath(name).suffix.lower(),'编码待确认')
            after=FORMATS.get(PurePosixPath(target).suffix.lower(),'编码待确认')
            key=(name,target)
            if key in seen:continue
            seen.add(key)
            rows.append(dict(source=name,source_format=before,target=target,target_format=after,
                status='SAME_FORMAT_CONTENT_UPDATED' if before==after else 'CONVERTED' if '编码待确认' not in (before,after) else 'UNCONFIRMED',
                basis='已写回changes及实际任务来源；格式标签来自文件后缀，精确原生结构另见最终检查'))
    targets=list(dict.fromkeys(FORMATS.get(PurePosixPath(r['name']).suffix.lower(),'编码待确认')
                 for r in result.get('local',{}).get('artifacts',[]) if r.get('name')))
    if not targets:targets=list(dict.fromkeys(r['target_format'] for r in rows))
    if not targets:targets=list(dict.fromkeys(FORMATS.get(PurePosixPath(n).suffix.lower(),'编码待确认') for n in changes))
    # A validated source format list is preferable to guessing from a ZIP name.
    input_formats=source
    if rows:source=list(dict.fromkeys(r['source_format'] for r in rows))
    explicit=result.get('target_format')
    if not targets and explicit and explicit not in {'ZIP','原版工程交付包','模型专用产物'}:targets=[explicit]
    return dict(schema='ecodata.format-conversion.v1',source_formats=source,input_document_formats=input_formats,target_formats=targets,files=rows,changed_files=sorted(changes),
        scientific_status=manifest.get('final_check',{}).get('scientific_definitions',{}).get('status'),
        written_files_without_source_mapping=sorted(changes-{r['target'] for r in rows}),
        native_writeback_verified=manifest.get('native_writeback_verified'),delivery_container='ZIP',
        scope='说明实际文件编码和写回，不代表模型已读入或融合精度提升')


def conversion_text(report):
    source='、'.join(report['source_formats']) or '未取得足够格式证据'
    target='、'.join(report['target_formats']) or '未确认；不能仅凭ZIP认定原生格式'
    lines=['格式与写回说明',f'原始数据格式：{source}',f'实际输出格式：{target}']
    for before,after in dict.fromkeys((r['source_format'],r['target_format']) for r in report['files']):
        lines.append(f'转换路径：{before} → {after}')
    if report['source_formats'] and report['source_formats']==report['target_formats']:
        lines.append('输入与输出格式相同；数据处理或内容更新不等于编码格式改变。')
    if report['native_writeback_verified']:lines.append('已核验模型原生文件写回；模型实际试读状态另列。')
    if report.get('scientific_status')=='WORKING_ASSUMPTIONS':lines.append('科学边界：采用明确的历史联调假设，不等于原始时间定义已证实或精度提升。')
    if report['changed_files']:
        lines.append(f"实际写回文件：{len(report['changed_files'])}个（完整路径见下载包中的格式转换记录）")
    lines.append('ZIP是下载容器，不是模型数据格式。')
    return '\n'.join(lines)


def describe_output(result, profile):
    from ecodata.models.adaptation import field_requirements

    local = result.get("local", {})
    manifest = result.get("manifest", {})
    artifact_rows = local.get("artifacts", [])
    archive_members = result.get("archive_members", [])
    archive_counts = {}
    for name in archive_members:
        kind = FORMATS.get(PurePosixPath(name).suffix.lower(), "其他成员")
        archive_counts[kind] = archive_counts.get(kind, 0) + 1
    output_files = []
    axes = set()
    for number, row in enumerate(artifact_rows, 1):
        public_axes = sorted({PUBLIC_AXES[k.casefold()] for k in row.get("dimensions", {})
                              if k.casefold() in PUBLIC_AXES})
        axes.update(public_axes)
        output_files.append(dict(id=f"O{number}", format=FORMATS.get(
            PurePosixPath(row["name"]).suffix.lower(), "其他产物"),
            axes=public_axes, variable_count=len(row.get("fields", [])),
            has_ioapi_tflag="TFLAG" in row.get("fields", []), readback=True))
    direct = manifest.get("native_ready") is True or local.get("native_ready") is True
    gate=manifest.get('final_check')
    if gate:
        direct=direct and all(gate.get(k,{}).get('status')=='PASS' for k in ('format','project')) and bool(gate.get('native',{}).get('model_read_confirmed'))
    source = list(dict.fromkeys(result.get("source_formats", [])))
    format_report=format_conversion(result)
    if format_report['source_formats']:source=format_report['source_formats']
    output = result.get("target_format") or (' / '.join(format_report['target_formats']) if format_report['target_formats'] else "模型专用产物" if direct else "原版工程交付包")
    fields = {k.casefold() for row in artifact_rows for k in row.get("fields", [])}
    blocks = []
    profile = profile or {}
    role_specs = profile.get("preparation_scenario", {}).get("inputs") or profile.get("inputs", [])
    component_only=manifest.get('status')=='COMPONENT_EXPORTED'
    for role in ([] if component_only else role_specs):
        if not role.get("required"):
            continue
        required = [f for f in field_requirements(role) if f.get("required", True)]
        missing = [f["name"] for f in required if not fields.intersection(
            name.casefold() for name in [f["name"], f.get("standard_name") or f["name"],
                                         *(f.get("aliases") or [])])] if artifact_rows else []
        blocks.append(dict(role=role.get("role", role["name"]), title=role["name"],
                           missing=missing, structure_needs_check=not required or not artifact_rows))
    local_notes = [f"{item['task']} 处理失败：{item['reason']}"
                   for item in result.get("failed_tasks",[])]
    if gate:
        local_notes.extend('最终格式检查：'+str(e) for e in gate.get('format',{}).get('errors',[])[:10])
        local_notes.extend('最终工程检查：'+str(e) for e in gate.get('project',{}).get('validation',{}).get('errors',[])[:10])
    if not direct:
        preparation = manifest.get("preparation", {})
        admission = preparation.get("source_admission", {})
        local_notes.extend(item["message"] for item in admission.get("required_information", [])
                           if item.get("message"))
        local_notes.extend(item["message"] for item in manifest.get("requirements_check", {}).get(
            "missing_or_conflicting", []) if item.get("message"))
        match = local.get("model_input") or {}
        local_notes.extend(map(str, match.get("whole_model_missing", [])))
    return dict(model=(profile or {}).get("id"), native_ready=direct, zip_readable=True,
                component_only=component_only,
                format_conversion=format_report,
                final_check=manifest.get('final_check'),
                files=output_files, archive_formats=archive_counts, source_formats=source, target_format=output,
                requirements=blocks, notes=list(dict.fromkeys(local_notes)),
                conversion_reason=conversion_reason(result, output, axes))


def cards(result, check):
    format_line = " / ".join(check["source_formats"])
    format_line = f"{format_line} → {check['target_format']}" if format_line else check["target_format"]
    from .recommendations import recipe_name

    items = []
    for task, search in searches(result).items():
        proxy = search["evidence"].get("reference_basis") in {'PRIMARY_GRID_RECONSTRUCTION_PROXY','REDUCED_SCALE_PRODUCT_RECONSTRUCTION','COARSE_PRODUCT_RECONSTRUCTION_ONLY'}
        for row in search["rankings"]:
            methods = row.get("steps") or [row["algorithm"]]
            labels = [METHODS.get(op, (op, ""))[0] for op in methods]
            alignment = search.get("alignment_steps") or ()
            if alignment:
                labels.insert(0,recipe_name(alignment))
            elif result.get("mode") == "independent_grid_search" or search.get("backend")=="independent_grid_search":
                alignment = ('spatial.regrid.conservative',)
                labels.insert(0, "保守重网格")
            elif search.get("backend")=="native_reference_search":
                alignment = ('reference.native_grid_support',)
                labels.insert(0, "卫星质量筛选与网格聚合")
            benefit = "，再".join(dict.fromkeys(METHODS.get(op, ("", ""))[1] for op in methods
                                               if op in METHODS))
            params = " · ".join(f"{name}={value:g}" for name, value in (row.get("parameters") or {}).items()
                                if name in {"power", "weight"} and isinstance(value, (int, float)))
            audit = []
            if row["selected"]:
                evidence=search['evidence'];protocol=evidence.get('evaluation_protocol') or {}
                if protocol.get('mode')=='REDUCED_SCALE_PRODUCT_RECONSTRUCTION':
                    audit.extend(['评价：所有来源先面积聚合变粗，再恢复原细产品；参考只在评价侧使用',
                        '参考 '+protocol['reference_source']+'（按网格粗细选取，不是独立实测真值）',
                        f"选型 {protocol['selection_count']} 样本；方案冻结后复核 {protocol['test_count']} 样本；多源无额外加分"])
                    for omitted in evidence.get('source_ablations',[]):
                        score=omitted.get('metrics')
                        audit.append('去掉 '+omitted['omitted_source']+'：'+(f"RMSE {score['RMSE']:.4g}" if score else '未评价'))
                    for name,score in evidence.get('coarse_diagnostics',{}).items():
                        if score:audit.append(f"{FAMILY_NAMES[name]}粗主产品诊断 RMSE {score['RMSE']:.4g}（不参与选图）")
                    paired=evidence.get('paired_selection')
                    if paired and paired['RMSE_change_percent'] is not None:
                        audit.append(f"多源相对最佳单源RMSE变化 {paired['RMSE_change_percent']:+.2f}%（负数为改善）；改善空间块 {paired['improved_blocks']}/{len(paired['blocks'])}")
                    audit.append('仅为降尺度产品重建试验；跨尺度有效性、独立实测精度及稳定增益未证实')
                elif protocol.get('missing_evidence'):
                    audit.append('细尺度评价条件不足：'+'；'.join(protocol['missing_evidence']))
                for family, score in search["evidence"].get("selection_scores", {}).items():
                    if score:
                        audit.append(f"{FAMILY_NAMES[family]}最佳 RMSE {score['RMSE']:.4g}")
                for group, score in search["evidence"].get("holdout_scores", {}).items():
                    if score:
                        group_name = {"single_source_baseline": "单源", "multisource_candidate": "多源"}[group]
                        audit.append(f"留出复核 {group_name} RMSE {score['RMSE']:.4g}")
                for seed in search.get("confirmed_initial", []):
                    algorithm = METHODS.get(seed["algorithm"], (seed["algorithm"], ""))[0]
                    status = "评分通过" if seed["status"] == "PASS" else "评分未通过"
                    audit.append(f"AI 初始候选 {algorithm}（{status}）")
            reference=SELECTION_REFERENCES.get(search['evidence'].get('reference_basis'))
            alternative=None
            if row['selected']:
                alternative=next((candidate['metrics'] for candidate in search['rankings']
                                  if not candidate['selected'] and candidate.get('metrics')),None)
            items.append(dict(task=task, title=" → ".join(labels), family=FAMILY_NAMES.get(row["family"], ""),
                              rank=row["rank"], selected=row["selected"], metrics=row["metrics"], audit=audit,
                              methods=methods, parameters=params, proxy=proxy, format_line=format_line,
                              selection_reason='在通过核验的候选中按 RMSE 选优；误差在容差内时优先步骤更少的方案。',
                              selection_reference=reference,
                              seed_usage=search.get('seed_usage'),steps=[*alignment,*methods],alternative=alternative,
                              reason=(benefit + "。" if benefit else "") + check["conversion_reason"]))
    for entry in result.get("original_report", []) if result.get("mode")=="selected_operations" else []:
        reports=entry['report']
        if entry.get('backend')=='statistical_preprocessing' and isinstance(reports,dict):
            reports=[reports]
        for report in reports if isinstance(reports,list) else []:
            if entry.get('backend') in {'statistical_preprocessing','structure_conversion'} and report.get('methods'):
                items.append(dict(task=entry['task'],title=recipe_name(report['methods']),selected=False,
                    metrics={},methods=report['methods'],parameters='',proxy=False,performed=True,
                    format_line=format_line,reason='该处理方案已生成可回读产物，未执行候选评分。'))
                continue
            if entry.get("backend") in {"independent_grid_search", "independent_alignment"} and "alignment" in report:
                alignment = report["alignment"]
                ranked = sorted(alignment.get("candidates", ()), key=lambda grid: (
                    grid["rmse"], grid["mae"], grid["source_grid"] != alignment["selected_source_grid"]))
                for rank, grid in enumerate(ranked, 1):
                    resolution = f"{grid['latitude_step']:g}° × {grid['longitude_step']:g}°"
                    items.append(dict(task=entry["task"], title=f"保守重网格 · {resolution}",
                        family=f"来源 {grid['source_grid']} 网格", rank=rank,
                        selected=grid["source_grid"] == alignment["selected_source_grid"],
                        metrics=dict(RMSE=grid["rmse"], MAE=grid["mae"], count=grid["n"]),
                        methods=["spatial.regrid.conservative"], parameters="", proxy=True, format_line=format_line,
                        selection_reason="按相同参考格点上的重建 RMSE、MAE 选择工作网格。",
                        selection_reference="原生参考格点重建（代理评价）",
                        reason="在共同留出日期和相同原生参考格点上比较对齐结果。"))
                if not alignment.get("candidates"):
                    items.append(dict(task=entry["task"], title="保守重网格", selected=True, metrics={}, performed=True,
                        methods=["spatial.regrid.conservative"], parameters="", proxy=False, format_line=format_line,
                        reason=("已按指定目标网格对齐，本步骤没有网格比较分数。" if alignment["selection_basis"] == "DECLARED_TARGET_GRID"
                                else "可比评分证据不足，按共同覆盖范围选用原生工作网格，本步骤没有网格比较分数。")))
                if gap := report.get("gap_filling"):
                    items.append(dict(task=entry["task"], title="备用源逐缺口补位", selected=True, metrics={}, performed=True,
                        methods=[], parameters="", proxy=False, format_line=format_line,
                        reason=f"补位 {gap['filled']} 个像元记录，剩余缺测 {gap['unresolved']} 个。"))
                continue
            if entry.get("backend")=="station_grid":
                methods=report["methods"]
                title=recipe_name(methods)
                score=report.get("heldout_correction")
                baseline=report.get("heldout_baseline")
                if not score:
                    items.append(dict(task=entry["task"],title=title,selected=False,metrics={},methods=methods,
                        parameters="",proxy=False,format_line=format_line,performed=True,
                        reason=f"配对成功，共 {report['matched_records']} 条。"))
                    continue
                selected=report["selected"]
                for rank,(name,metrics) in enumerate(sorted(
                        [("background",baseline),("corrected",score)],key=lambda item:item[1]["rmse"]),1):
                    items.append(dict(task=entry["task"],title=title if name=="corrected" else "原始格点单源基线",
                        family="订正试跑" if name=="corrected" else "单源",rank=rank,selected=name==selected,
                        metrics=dict(RMSE=metrics["rmse"],MAE=metrics["mae"],count=metrics["count"]),
                        methods=methods,steps=methods if name=='corrected' else ['spatial.source'],
                        alternative=(dict(RMSE=baseline['rmse'],count=baseline['count']) if name=='corrected'
                                     else dict(RMSE=score['rmse'],count=score['count'])),
                        parameters="",proxy=False,format_line=format_line,
                        selection_reason='比较同一批留出时刻的 RMSE；订正没有降低 RMSE 时保留单源基线。',
                        selection_reference='已配对站点的时间留出样本',
                        reason="在同一批站点留出时刻比较误差，适用范围为已配对站点。"))
                continue
            weighted = str(report.get("method","")).startswith("fusion.uncertainty_weighted.")
            aligned=entry.get('backend')=='independent_alignment'
            if report.get("operation") not in {"align_time","align_space"} and not weighted and not aligned:
                continue
            if aligned:
                reason='按共同 UTC 日期和目标网格对齐，分别保留两份来源。'
            elif weighted:
                reason = "用原始 uncertainty 降低较不确定来源的权重，输出合成值和合成不确定度。"
            elif report.get('operation')=='align_time':
                reason='统一共同覆盖时段的时间间隔，保留时间轴与单位。'
            else:
                reason='统一网格坐标，保留各来源，便于逐格对照。'
            items.append(dict(task=entry["task"],title='保守重网格' if aligned else recipe_name(report["name"]),selected=False,metrics={},performed=True,
                methods=['spatial.regrid.conservative'] if aligned else report['name'],
                parameters="",proxy=False,format_line=format_line,reason=reason))
    for row in result.get("research_records",[]):
        status = {"PASS":"已执行","PARTIAL":"部分方案完成","BLOCKED":"条件不足"}.get(row["status"],row["status"])
        reason = row.get("reason") or f"原版服务返回 {row['status']}，已读回 {len(row.get('readback',[]))} 份 NetCDF。"
        items.append(dict(task=row["label"],title=status,selected=False,metrics={},methods=[],
                          performed=row['status'] in {'PASS','PARTIAL'},
                          parameters="",proxy=False,format_line=format_line,reason=reason))
    if not items:
        title = "保守重网格" if result.get("mode") == "independent_alignment" else "输出格式与转换"
        items.append(dict(task="", title=title, selected=False, metrics={},
                          methods=['spatial.regrid.conservative'] if result.get('mode')=='independent_alignment' else [],parameters="",
                          performed=result.get('mode')=='independent_alignment',
                          proxy=False, format_line=format_line, reason=check["conversion_reason"]))
    return items


def preparation_summaries(result):
    """Describe operations verified by the original handoff, including no-score runs."""
    from .recommendations import recipe_name

    manifest=result.get('manifest') or {}
    preparation=manifest.get('preparation') or {}
    admission=preparation.get('source_admission') or {}
    summaries=[]
    adaptation=preparation.get('adaptation') or admission.get('adaptation') or []
    for row in adaptation:
        if row.get('status')!='ADAPTED_AWAITING_ADMISSION' or not row.get('product'):
            continue
        titles={
            'coordinate_equivalence':'坐标排序与等价整理',
            'strict_time_concatenation':'时间分片顺序拼接',
            'rate_interval_integral':'降水率按时间区间积分',
        }
        operations=[titles[item['operation']] for item in row.get('operations',[]) if item.get('operation') in titles]
        detail=f"已回读 {row.get('time_records','未知数量')} 条时间记录，单位 {row.get('units') or '未注明'}。该预处理没有候选评分。"
        summaries.append(dict(task=row.get('source_id'),title=' → '.join(operations) or '降水适配',
                              reason=detail,performed=True,steps=['engineering.precipitation_adaptation']))
    for task in preparation.get('tasks',[]):
        execution=task.get('result') or (task.get('execution') or {}).get('result') or {}
        if execution.get('status')=='UNCHANGED_COMPLETE_INPUT':
            summaries.append(dict(task=task.get('field'),title='保留完整单源',performed=True,
                reason=execution.get('reason') or '通过数据检查后保留原有数据和分辨率。',steps=['spatial.source']))
        elif task.get('operation') and (manifest.get('component_format_generated') or manifest.get('native_writeback_verified')):
            summaries.append(dict(title=recipe_name([task['operation']]),performed=True,
                reason='原生文件已写出并回读，清单记录的是格式或写回校验。'))
    if manifest.get('status')=='PROJECT_UPDATED' and preparation.get('route')=='bound_multirun_multigrid_meteorology_v1':
        summaries.append(dict(title='同刻观测局部残差校正',performed=True,
            reason='已在有观测支持的区域完成局部校正并回读工程副本。'))
    if manifest.get('status')=='CONVERTED':
        summaries.append(dict(title='模型原生格式转换',performed=True,
            reason='转换器已提供原生输出文件及格式校验记录。',steps=['format.normalize']))
    if manifest.get('status')=='ENGINEERING_INSPECTED':
        blocked=(admission.get('weather') or {}).get('blocked_search') or {}
        reason=blocked.get('reason') or '本次交付记录为工程资料盘点，没有可确认的工程融合或写回成果。'
        summaries.append(dict(task='原版工程',title='工程盘点',performed=True,reason=reason))
    return summaries


def selection_summaries(items, failures, recommendations, *, result=None):
    """Summarize recorded outcomes without treating AI recommendations as winners."""
    from .recommendations import recipe_name

    summaries = []
    for row in items:
        if row['selected'] and row['metrics']:
            summaries.append(dict(task=row['task'], title=row['title'], metrics=row['metrics'],
                reason=row['selection_reason'], reference=row.get('selection_reference'),
                parameters=row['parameters'],seed_usage=row.get('seed_usage'),
                steps=row.get('steps',row.get('methods',[])),alternative=row.get('alternative'),proxy=row['proxy']))
        elif row.get('performed'):
            summary=dict(task=row['task'],title=row['title'],reason=row['reason'],performed=True,
                         steps=row.get('steps',row.get('methods',[])))
            if summary not in summaries:
                summaries.append(summary)
    result=result or {}
    for task, row in result.get('other_searches',{}).items():
        reason=('按同一评价条件下的实测目标值排序；误差等价时优先结构更简单的方案。'
                if row.get('validation_best_delivered') is not False else
                '交付方案由后端通过部署核验后选定；可能与评价阶段的第一名不同。')
        summaries.append(dict(task=task,title=recipe_name(row['steps']) if row['steps'] else '保留来源场',
            metrics=row['metrics'],reason=reason,performed=True,steps=row['steps'] or ['spatial.source'],
            search_note='已记录实际变异执行。' if row['mutation_executed'] else '未记录新的变异候选执行。'))
    summaries.extend(preparation_summaries(result))
    for failure in failures:
        task = (failure.get('task') if isinstance(failure,dict) else None) or '本次任务'
        recipes = dict.fromkeys(recipe_name(row['steps']) for row in recommendations if row['task'] == task)
        reason = (failure.get('reason') if isinstance(failure,dict) else str(failure)) or '后台未返回具体失败原因。'
        if reason in {"'time_bounds'", '"time_bounds"'}:
            reason = '本地处理时未找到时间区间字段 time_bounds。'
        summaries.append(dict(task=task, title='处理失败 · 未选出算法', reason=reason, failed=True,
            proposed='；'.join(recipes)))
    if not summaries:
        summaries.append(dict(title='未产生算法评分',
            reason='本次结果没有可展示的候选选优记录。实际执行的操作见下方处理记录。'))
    explainable=[row for row in summaries if row.get('steps')]
    for number,row in enumerate(explainable[:8],1):
        row['review_id']=f'P{number}'
    return summaries


def public_plans(summaries):
    """Send only stable operation names and aggregate scores for finished plans."""
    from .recommendations import ALGORITHM_NAMES

    allowlist={**ALGORITHM_NAMES,**{key:label for key,(label,_) in METHODS.items()}}

    def scores(values):
        return {key:values[key] for key in ('RMSE','MAE','count','objective')
                if type(values.get(key)) in (int,float) and isfinite(values[key])}

    plans=[]
    for row in summaries:
        if not row.get('review_id'):
            continue
        steps=row['steps']
        if not all(step in allowlist for step in steps):
            continue
        metrics=scores(row.get('metrics') or {})
        alternative=scores(row.get('alternative') or {})
        plan=dict(id=row['review_id'],steps=[allowlist[step] for step in steps],scores=metrics,
                  grid_proxy=row.get('proxy',False))
        if row.get('reference') in (*SELECTION_REFERENCES.values(),'已配对站点的时间留出样本'):
            plan['evaluation_reference']=row['reference']
        if metrics.get('count') and alternative.get('count')==metrics['count'] and 'RMSE' in alternative:
            plan['alternative_RMSE']=alternative['RMSE']
        plans.append(plan)
    return plans


def review_request(check, items, *, summaries=()):
    """Only fixed format/category labels and bundled model-profile fields can leave the machine."""
    known_formats = set(FORMATS.values()) | {"Text", "原版工程交付包", "模型专用产物", "其他产物"}
    final=check.get('final_check') or {}
    native=final.get('native',{})
    weather=final.get('format',{}).get('weather_files',{})
    format_detail=check.get('format_conversion') or {}
    return dict(stage="output_check", target_model=check["model"], zip_readable=check["zip_readable"],
                task_scope='component_conversion' if check.get('component_only') else 'data_only' if not check['model'] else 'model_project',
                final_model_check=dict(model_requirements=final.get('model_requirements'),
                    format_status=final.get('format',{}).get('status'),format_errors=final.get('format',{}).get('errors',[])[:10],
                    format_scope=final.get('format',{}).get('scope'),
                    native_files_checked=final.get('format',{}).get('native_files',{}),
                    project_status=final.get('project',{}).get('status'),
                    project_reason=final.get('project',{}).get('reason'),
                    project_errors=final.get('project',{}).get('validation',{}).get('errors',[])[:10],
                    declared_version=final.get('project',{}).get('validation',{}).get('declared_version'),
                    weather_files_checked=len(weather),weather_failures=sum(r['status']=='FAIL' for r in weather.values()),
                    native_resources=final.get('input_dependency_scope',{}).get('resources',[])[:200],
                    omitted_native_resources=max(0,len(final.get('input_dependency_scope',{}).get('resources',[]))-200),
                    active_control_bindings=final.get('input_dependency_scope',{}).get('control_bindings',[])[:200],
                    native={key:native.get(key) for key in ('status','reason','engine_id','program_sha256','returncode','fresh_success_marker','model_read_confirmed','trial_start','trial_end','full_simulation_verified')},
                    native_log_tail=native.get('log_tail','')[-1800:],
                    scientific_status=final.get('scientific_definitions',{}).get('status'),
                    working_assumptions=[{key:a.get(key) for key in ('status','rule','original_model_day_verified')}
                        for a in final.get('scientific_definitions',{}).get('assumptions',[])]),
                local_native_ready=check["native_ready"], files=check["files"][:60],
                omitted_file_count=max(len(check["files"])-60, 0),
                archive_formats=check["archive_formats"],
                source_formats=[f for f in check["source_formats"] if f in known_formats],
                target_format=check["target_format"] if check["target_format"] in known_formats else "其他产物",
                actual_output_formats=[fmt for fmt in format_detail.get('target_formats',[]) if fmt in known_formats],
                model_requirements=[dict(role=r["role"], missing_field_matches=r["missing"],
                                         structure_needs_check=r["structure_needs_check"])
                                    for r in check["requirements"]],
                methods=sorted({op for item in items for op in item["methods"] if op in METHODS}),
                selected_plans=public_plans(summaries),
                physical_reference="grid_proxy" if any(item["proxy"] for item in items) else "unspecified")


def validate_review(output, *, plan_ids=()):
    if not isinstance(output, dict) or not isinstance(output.get("conversion_benefit"), str):
        raise ValueError("输出复核响应缺少转换说明")
    if output.get("verdict") not in {"ready", "needs_inputs", "needs_verification"}:
        raise ValueError("输出复核响应缺少可用性判断")
    clean = {"verdict": output["verdict"], "conversion_benefit": output["conversion_benefit"].strip()[:420]}
    for key in ("missing_data", "missing_information"):
        value = output.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("输出复核响应缺少数据或信息检查")
        clean[key] = list(dict.fromkeys(item.strip()[:170] for item in value if item.strip()))[:8]
    explanations={}
    rows=output.get('plan_explanations')
    for row in rows if isinstance(rows,list) else []:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or row['id'] not in plan_ids:
            continue
        benefit,limits=row.get('benefit'),row.get('limits')
        if isinstance(benefit,str) and benefit.strip() and isinstance(limits,str) and limits.strip():
            explanations.setdefault(row['id'],dict(benefit=benefit.strip()[:420],limits=limits.strip()[:250]))
    if explanations:
        clean['plan_explanations']=explanations
    return clean


def availability_text(check, review=None):
    model = {"swat_plus":"SWAT+", "wrf":"WRF", "cmaq":"CMAQ", "camx":"CAMx",
             "calpuff":"CALPUFF"}.get(check["model"], "目标模型")
    if not check["model"]:
        lines = ["本次仅进行数据处理；未请求模型工程转换或运行。"]
    elif check["native_ready"]:
        lines = [f"模型可用性 · 已通过 {model} 本地原生输入校验"]
    else:
        lines = [f"模型可用性 · 尚不能确认可直接供 {model} 使用"]
    gate=check.get('final_check')
    if gate:
        labels={'PASS':'通过','FAIL':'未通过','UNKNOWN':'未确认','ERROR':'检查异常','NOT_RUN':'未运行','NOT_REQUESTED':'本次未请求（仅组件转换）','NOT_RECHECKED':'原流程已有运行记录，本关卡未复核'}
        native=gate.get('native',{})
        lines=[f"原生格式检查 · {labels.get(gate['format']['status'],gate['format']['status'])}",
            f"工程引用检查 · {labels.get(gate['project']['status'],gate['project']['status'])}",
            f"模型实际试读 · {labels.get(native['status'],native['status'])}"]
        if native.get('model_read_confirmed'):
            lines.append('已由匹配版本模型完成短时试运行；原模拟期完整运行和精度未验证。')
        elif native.get('reason'):lines.append('试读说明 · '+native['reason'])
        if gate.get('project',{}).get('reason'):lines.append('工程说明 · '+gate['project']['reason'])
        if gate.get('scientific_definitions',{}).get('status')=='WORKING_ASSUMPTIONS':
            lines.append('科学定义 · 含明确工作假设，不等于作者原始定义已证实。')
    missing = [f"{r['title']}（{', '.join(r['missing'])}）" for r in check["requirements"] if r["missing"]]
    if missing and not check["native_ready"]:
        lines.append("数据待补或待映射 · " + "；".join(missing))
    pending = list(check["notes"])
    unresolved = [r["title"] for r in check["requirements"] if r["structure_needs_check"]]
    trial_passed=bool(gate and gate.get('native',{}).get('model_read_confirmed'))
    if unresolved and not check["native_ready"] and not trial_passed:
        pending.append("需核对的输入结构包括 " + "、".join(unresolved))
    if not check["native_ready"] and check["model"] and not check.get('component_only'):
        pending.append('原模拟期完整运行、未覆盖的条件和科学定义仍需核验' if trial_passed else '目标模型的原生文件、输入绑定和完整工程配置仍需核验')
    if review:
        if review["verdict"]=="ready" and not check["native_ready"]:
            pending.append("完整模型输入仍以本地校验为准")
        if review["verdict"]!="ready" and check["native_ready"]:
            pending.append("输出包复核还有待确认事项")
        if review["missing_data"]:
            lines.append("待核对数据 · " + "；".join(review["missing_data"]))
        pending.extend(review["missing_information"])
    if pending:
        lines.append("待确认信息 · " + "；".join(dict.fromkeys(pending)))
    if not missing and not pending and (not review or not review["missing_data"]):
        lines.append("数据和配置信息 · 当前检查未发现缺项")
    return "\n".join(lines)
