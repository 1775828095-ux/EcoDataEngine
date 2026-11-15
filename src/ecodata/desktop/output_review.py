"""Result-card data and a bounded, public-schema-only output review."""
from pathlib import PurePosixPath


FAMILY_NAMES = {"SINGLE_SOURCE": "单源", "MULTISOURCE": "多源"}
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
    ".json": "JSON", ".geojson": "GeoJSON", ".txt": "文本", ".pcp": "SWAT 降水",
    ".cli": "SWAT 气象索引", ".dat": "模型数据", ".in": "模型配置", ".inp": "模型输入",
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
                                confirmed_initial=result["search"].get("confirmed_initial", []))}
    return result.get("grid_searches") or result.get("batch_searches") or {}


def conversion_reason(result, output, axes):
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
    source = list(dict.fromkeys(result.get("source_formats", [])))
    output = result.get("target_format") or ("模型专用产物" if direct else "原版工程交付包")
    fields = {k.casefold() for row in artifact_rows for k in row.get("fields", [])}
    blocks = []
    profile = profile or {}
    role_specs = profile.get("preparation_scenario", {}).get("inputs") or profile.get("inputs", [])
    for role in role_specs:
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
                files=output_files, archive_formats=archive_counts, source_formats=source, target_format=output,
                requirements=blocks, notes=list(dict.fromkeys(local_notes)),
                conversion_reason=conversion_reason(result, output, axes))


def cards(result, check):
    format_line = " / ".join(check["source_formats"])
    format_line = f"{format_line} → {check['target_format']}" if format_line else check["target_format"]
    from .recommendations import recipe_name

    items = []
    for task, search in searches(result).items():
        proxy = search["evidence"].get("reference_basis") == "PRIMARY_GRID_RECONSTRUCTION_PROXY"
        for row in search["rankings"]:
            methods = row.get("steps") or [row["algorithm"]]
            labels = [METHODS.get(op, (op, ""))[0] for op in methods]
            alignment = search.get("alignment_steps") or ()
            if alignment:
                labels.insert(0,recipe_name(alignment))
            elif result.get("mode") == "independent_grid_search" or search.get("backend")=="independent_grid_search":
                labels.insert(0, "保守重网格")
            elif search.get("backend")=="native_reference_search":
                labels.insert(0, "卫星质量筛选与网格聚合")
            benefit = "，再".join(dict.fromkeys(METHODS.get(op, ("", ""))[1] for op in methods
                                               if op in METHODS))
            params = " · ".join(f"{name}={value:g}" for name, value in (row.get("parameters") or {}).items()
                                if name in {"power", "weight"} and isinstance(value, (int, float)))
            audit = []
            if row["selected"]:
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
            items.append(dict(task=task, title=" → ".join(labels), family=FAMILY_NAMES.get(row["family"], ""),
                              rank=row["rank"], selected=row["selected"], metrics=row["metrics"], audit=audit,
                              methods=methods, parameters=params, proxy=proxy, format_line=format_line,
                              reason=(benefit + "。" if benefit else "") + check["conversion_reason"]))
    for entry in result.get("original_report", []) if result.get("mode")=="selected_operations" else []:
        for report in entry["report"] if isinstance(entry["report"],list) else []:
            weighted = str(report.get("method","")).startswith("fusion.uncertainty_weighted.")
            if report.get("operation") not in {"align_time","align_space"} and not weighted:
                continue
            reason = ("统一共同覆盖时段的时间间隔，保留时间轴与单位。" if report["operation"]=="align_time" else
                      "统一网格坐标，保留各来源，便于逐格对照。")
            if weighted:
                reason = "用原始 uncertainty 降低较不确定来源的权重，输出合成值和合成不确定度。"
            items.append(dict(task=entry["task"],title=recipe_name(report["name"]),selected=False,metrics={},
                methods=[],parameters="",proxy=False,format_line=format_line,reason=reason))
    for row in result.get("research_records",[]):
        status = {"PASS":"已执行","PARTIAL":"部分方案完成","BLOCKED":"条件不足"}.get(row["status"],row["status"])
        reason = row.get("reason") or f"原版服务返回 {row['status']}，已读回 {len(row.get('readback',[]))} 份 NetCDF。"
        items.append(dict(task=row["label"],title=status,selected=False,metrics={},methods=[],
                          parameters="",proxy=False,format_line=format_line,reason=reason))
    if not items:
        title = "保守重网格" if result.get("mode") == "independent_alignment" else "输出格式与转换"
        items.append(dict(task="", title=title, selected=False, metrics={}, methods=[], parameters="",
                          proxy=False, format_line=format_line, reason=check["conversion_reason"]))
    return items


def review_request(check, items):
    """Only fixed format/category labels and bundled model-profile fields can leave the machine."""
    known_formats = set(FORMATS.values()) | {"Text", "原版工程交付包", "模型专用产物", "其他产物"}
    return dict(stage="output_check", target_model=check["model"], zip_readable=check["zip_readable"],
                local_native_ready=check["native_ready"], files=check["files"][:60],
                omitted_file_count=max(len(check["files"])-60, 0),
                archive_formats=check["archive_formats"],
                source_formats=[f for f in check["source_formats"] if f in known_formats],
                target_format=check["target_format"] if check["target_format"] in known_formats else "其他产物",
                model_requirements=[dict(role=r["role"], missing_field_matches=r["missing"],
                                         structure_needs_check=r["structure_needs_check"])
                                    for r in check["requirements"]],
                methods=sorted({op for item in items for op in item["methods"] if op in METHODS}),
                physical_reference="grid_proxy" if any(item["proxy"] for item in items) else "unspecified")


def validate_review(output):
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
    return clean


def availability_text(check, review=None):
    model = {"swat_plus":"SWAT+", "wrf":"WRF", "cmaq":"CMAQ", "camx":"CAMx",
             "calpuff":"CALPUFF"}.get(check["model"], "目标模型")
    if not check["model"]:
        lines = ["模型可用性 · 尚未指定目标模型"]
    elif check["native_ready"]:
        lines = [f"模型可用性 · 已通过 {model} 本地原生输入校验"]
    else:
        lines = [f"模型可用性 · 尚不能确认可直接供 {model} 使用"]
    missing = [f"{r['title']}（{', '.join(r['missing'])}）" for r in check["requirements"] if r["missing"]]
    if missing and not check["native_ready"]:
        lines.append("数据待补或待映射 · " + "；".join(missing))
    pending = list(check["notes"])
    unresolved = [r["title"] for r in check["requirements"] if r["structure_needs_check"]]
    if unresolved and not check["native_ready"]:
        pending.append("需核对的输入结构包括 " + "、".join(unresolved))
    if not check["native_ready"] and check["model"]:
        pending.append("目标模型的原生文件、输入绑定和完整工程配置仍需核验")
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
