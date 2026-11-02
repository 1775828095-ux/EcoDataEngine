"""Automatic model preparation; physical compatibility remains a core gate."""
import hashlib
import io
import json

import geopandas as gpd
import streamlit as st

from ecodata.interface.downloads import package_results
from ecodata.models import RequirementStore
from ecodata.models.adapters import get_adapter
from ecodata.models.contracts import ModelSchema
from ecodata.models.scenario import (scenario_schema, recognise_inputs, suggest_roles,
                                     plan_scenario, execute_scenario, apply_field_mappings)
from ecodata.recognition import inspect_dataset
from ecodata.fusion.planning import DataRoleClassifier
from ecodata.interface.engineering_upload import scan_model_uploads


def _render_wasp_package(files):
    """Simple WASP path: recognise, classify, build and download one package."""
    from ecodata.models.adapters.wasp_package import WASPDataPackageExporter

    inputs, read_failures = [], []
    for item in files:
        try:
            inputs.extend(recognise_inputs([item]))
        except Exception as exc:
            read_failures.append({"文件": item[0], "原因": str(exc)})
    if read_failures:
        st.warning(f"有 {len(read_failures)} 个文件无法识别，其余文件继续分析。")
        with st.expander("查看无法识别的文件"):
            st.dataframe(read_failures, hide_index=True)
    if not inputs:
        st.error("没有可识别的数据文件。")
        return
    rows = []
    for item in inputs:
        info = inspect_dataset(item.filename, item.dataset)
        role = item.dataset.attrs.get("data_role", "unclassified")
        target = {"tributary_inflow": "tributary_flow.csv",
                  "forcing": "environmental_forcing.csv",
                  "water_quality_observation": "tributary_concentration.csv / observations.csv"}.get(role, "需要确认")
        rows.append({"文件": item.filename, "自动识别角色": role, "可生成": target,
                     "变量": "、".join(info.variables), "状态": "✅" if target != "需要确认" else "⚠ 需要确认"})
    st.markdown("#### 3. 自动识别与 WASP 数据角色")
    st.dataframe(rows, hide_index=True)
    st.caption("segment_id 只使用工程模板、明确映射或可核验空间关系；不会自动猜测。")
    template_upload = st.file_uploader("可选：上传 segment GeoJSON 工程模板",
                                      type=["geojson"], key="wasp_project_template")
    project_template = None
    # Streamlit returns one UploadedFile here; ignore incompatible mocked/list values.
    if template_upload and not isinstance(template_upload, (list, tuple)):
        from ecodata.models.engineering import ModelProjectTemplate
        try:
            segments = gpd.read_file(io.BytesIO(template_upload.getvalue()))
            project_template = ModelProjectTemplate.from_segments(
                "WASP", segments, provenance={"file": template_upload.name, "kind": "user_project_template"})
            st.success(f"已读取 {len(segments)} 个 segment；CRS：{segments.crs}")
        except Exception as exc:
            st.error(f"工程模板不能使用：{exc}")
            return
    with st.expander("高级设置"):
        max_distance_m = st.number_input("最近 segment 最大距离（m）", min_value=1.0,
                                         value=5000.0, step=100.0)
        max_initial_days = st.number_input("初始条件最大时间差（天）", min_value=0.0,
                                           value=7.0, step=1.0)
        auto_confirm_threshold = st.selectbox("自动确认阈值", ["HIGH", "MEDIUM"], index=0,
                                              help="默认仅自动确认空间包含或专家模板等 HIGH 证据。")
    if st.button("自动分析并生成方案", key="wasp_analyse"):
        st.session_state["wasp_package_result"] = WASPDataPackageExporter().build(
            inputs, project_template=project_template, max_distance_m=max_distance_m,
            max_initial_time_gap=f"{max_initial_days:g}D",
            auto_confirm_threshold=auto_confirm_threshold)
    result = st.session_state.get("wasp_package_result")
    if not result:
        return
    generated = [name for name in result.files if name.endswith(".csv")]
    st.markdown("#### 4. 可生成的 WASP 标准数据")
    st.write("、".join(generated) if generated else "没有可生成的数据文件")
    st.json(result.status)
    validation = json.loads(result.files["validation_report.json"])
    mapping = json.loads(result.files["mapping.json"])
    st.markdown("##### 工程用途与 segment 映射")
    st.dataframe(mapping["mappings"], hide_index=True)
    uncertain = [item for item in mapping["mappings"] if not item.get("confirmed")]
    if uncertain:
        st.warning(f"有 {len(uncertain)} 项需要工程模板或人工确认；其他标准化结果仍可下载。")
        for item in uncertain:
            if item.get("segment_id"):
                st.checkbox(f"确认 {item['station_id']} → {item['segment_id']}",
                            key=f"wasp_confirm_{item['station_id']}_{item['segment_id']}")
    if validation["failures"]:
        st.error("存在失败项；请先补充明确单位或修复识别问题。")
    elif validation["status"] == "WARN":
        st.warning("数据包已生成，但仍有 segment 映射或缺失输入需要确认。")
    else:
        st.success("WASP 标准数据包已生成并通过当前质量检查。")
    st.download_button("下载 wasp_package.zip", result.zip_bytes(), "wasp_package.zip", "application/zip")


def render_model_page():
    st.subheader("模型数据准备")
    st.caption("选择模型 → 上传数据 → 自动分析 → 确认方案并执行。不会凭空补造缺失数据。")
    st.markdown("#### 1. 目标模型")
    profiles = RequirementStore().all()
    ids = [item["id"] for item in profiles]
    model_id = st.selectbox("目标模型", ids, index=ids.index("cmaq") if "cmaq" in ids else 0,
                            format_func=lambda value: next(item["name"] for item in profiles if item["id"] == value), key="scenario_model")
    profile = next(item for item in profiles if item["id"] == model_id)
    schema = scenario_schema(profile)
    st.markdown("#### 2. 上传数据")
    upload_mode = st.radio("导入方式", ["选择工程总文件夹（推荐）", "选择文件或ZIP"], horizontal=True)
    supported_types = ["csv", "txt", "rdb", "gz", "isd", "xlsx", "parquet", "nc", "nc4", "tif", "tiff", "h5"]
    if upload_mode == "选择工程总文件夹（推荐）":
        if model_id == "wasp":
            st.caption("请选择同时包含 usgs、nbc、noaa、ri_dem、wwtp 的上一级文件夹。references 会自动忽略。")
        else:
            st.caption("请选择包含本次模型数据的工程总文件夹；系统会递归读取支持的数据文件。")
        # A project directory commonly contains PDF/JSON notes and sidecar files.
        # Let the scanner classify them instead of having the browser reject them.
        upload_generation = st.session_state.get("scenario_upload_generation", 0)
        if st.button("清空已选文件", key="clear_scenario_directory"):
            st.session_state["scenario_upload_generation"] = upload_generation + 1
            st.session_state.pop("wasp_package_result", None)
            st.rerun()
        uploads = st.file_uploader("选择模型工程总文件夹", accept_multiple_files="directory",
                                   key=f"scenario_uploads_directory_{upload_generation}")
    else:
        uploads = st.file_uploader("上传模型准备文件或工程ZIP（可多选）", type=[*supported_types, "zip"],
                                   accept_multiple_files=True, key="scenario_uploads")
    if not uploads:
        st.session_state.pop("scenario_result", None)
        st.session_state.pop("scenario_analysis_fingerprint", None)
        return
    scan = scan_model_uploads(uploads, classify_wasp=model_id == "wasp",
                              expand_archives=upload_mode == "选择文件或ZIP")
    st.markdown("##### 工程目录扫描结果")
    st.write(f"有效文件 {len(scan.accepted)} 个；忽略 {len(scan.ignored)} 个；浏览器重复提交已排除 {len(scan.duplicates)} 个。")
    if scan.duplicates:
        st.info("检测到同一批文件被重复加入，平台已经自动去重，不影响后续分析。需要重新选择时请点击“清空已选文件”。")
    if model_id == "wasp":
        st.dataframe([{"目录": name, "接收文件数": count} for name, count in scan.group_counts.items()], hide_index=True)
    with st.expander("查看接收、忽略和重复清单"):
        if scan.accepted:
            st.write("接收")
            st.dataframe(scan.accepted, hide_index=True)
        if scan.ignored:
            st.write("忽略")
            st.dataframe(scan.ignored, hide_index=True)
        if scan.duplicates:
            st.write("重复")
            st.dataframe(scan.duplicates, hide_index=True)
    files = scan.files
    if not files:
        st.error("文件夹或ZIP中没有当前可读取的数据文件。")
        return
    try:
        if model_id == "wasp":
            _render_wasp_package(files)
            return
        raw_inputs = recognise_inputs(files)
        from ecodata.evaluation.quality_gates import RawInputQualityGate
        input_quality = {item.filename: RawInputQualityGate().validate(
            item.dataset, intended_operation="model_adaptation", requires_spatial=False) for item in raw_inputs}
        blocked_inputs = [name for name, report in input_quality.items() if report["status"] == "BLOCK"]
        warned_inputs = [name for name, report in input_quality.items() if report["status"] == "WARN"]
        if blocked_inputs:
            st.error("原始数据质量阻止：" + "、".join(blocked_inputs))
        elif warned_inputs:
            st.warning("原始数据质量存在警告：" + "、".join(warned_inputs))
        else:
            st.success("原始数据质量：通过")
        with st.expander("查看原始数据质量检查"):
            st.json(input_quality)
        suggested = suggest_roles(profile, raw_inputs)
        assignments = dict(suggested)
        declarations, mappings, options = {}, {}, {}
        st.markdown("#### 3. 自动识别与模型匹配结果")
        st.caption("默认自动匹配。只有识别不明确或需要调整时，才点击“修改识别结果”。")
        with st.expander("修改识别结果 / 高级设置"):
            st.caption("修改后必须重新分析；角色、变量映射仍需通过物理兼容性检查。")
            with st.expander("模型输入要求"):
                st.dataframe([{"类别": role["name"], "变量/单位": "；".join(f"{f['name']} [{f['unit']}]" for f in role["fields"]),
                               "格式": ", ".join(role["formats"]), "维度": " × ".join(role["dimensions"]), "必需": role["required"]}
                              for role in schema["inputs"]], hide_index=True)
            for item in raw_inputs:
                info = inspect_dataset(item.filename, item.dataset)
                needed_names = {str(name).casefold() for role in schema["inputs"] for field in role["fields"]
                                for name in [field["name"], field.get("standard_name", field["name"]), *field.get("aliases", [])]}
                missing_units = [v for v in info.variables if item.dataset[v].dtype.kind in "biuf" and v.casefold() in needed_names and not info.units.get(v)]
                edit_metadata = st.checkbox(f"手动修改 {item.filename} 的元数据", key=f"scene_metadata_{item.filename}")
                if edit_metadata:
                    missing_units = [v for v in info.variables if item.dataset[v].dtype.kind in "biuf"
                                     and v not in {info.x_coordinate, info.y_coordinate, info.time_coordinate}]
                value = {"units": {}}
                if info.x_coordinate and info.y_coordinate and (not info.crs or edit_metadata):
                    value["crs"] = st.text_input(f"{item.filename} 源 CRS（修改需确认）", value=info.crs or "", key=f"scene_crs_{item.filename}")
                for variable in missing_units:
                    unit = st.text_input(f"{item.filename} / {variable} 单位", value=info.units.get(variable, ""), key=f"scene_unit_{item.filename}_{variable}")
                    if unit:
                        value["units"][variable] = unit
                if info.structure == "table" and "qc_flag" not in item.dataset:
                    value["observations_valid"] = st.checkbox(f"确认 {item.filename} 观测可用于本次试验（不是自动质控）", key=f"scene_valid_{item.filename}")
                if info.time_coordinate and (edit_metadata or not item.dataset.attrs.get("temporal_semantics") and not any(item.dataset[v].attrs.get("temporal_semantics") for v in item.dataset.data_vars)):
                    choices = ["", "instantaneous", "mean", "accumulation"]
                    current = item.dataset.attrs.get("temporal_semantics", "")
                    semantics = st.selectbox(f"{item.filename} 时间语义（无法识别时确认）", choices, index=choices.index(current) if current in choices else 0, key=f"scene_time_{item.filename}")
                    if semantics:
                        value["temporal_semantics"] = semantics
                declarations[item.filename] = value
            adapter = get_adapter(ModelSchema.from_profile(profile))
            for role in schema["inputs"]:
                if role.get("record_only"):
                    continue
                choices = ["", *[item.filename for item in raw_inputs]]
                assignments[role["role"]] = st.selectbox(f"修改 {role['name']} 的来源", choices,
                    index=choices.index(suggested.get(role["role"], "")),
                    format_func=lambda value: value or "未匹配 / 需要确认", key=f"scene_role_{model_id}_{role['role']}")
                selected = next((item for item in raw_inputs if item.filename == assignments[role["role"]]), None)
                if selected:
                    automatic = adapter.field_mapping(selected.dataset, role)
                    mappings[role["role"]] = {}
                    for field in role["fields"]:
                        default = automatic.get(field["name"], "")
                        choices = list(dict.fromkeys(["", *selected.dataset.data_vars, *([default] if default else [])]))
                        mappings[role["role"]][field["name"]] = st.selectbox(f"字段映射：{role['name']} / {field['name']}", choices,
                            index=choices.index(default), key=f"scene_map_{model_id}_{role['role']}_{field['name']}_{selected.filename}")
            for option in schema.get("advanced_options", []):
                labels = option.get("labels", {})
                options[option["name"]] = st.selectbox(option["label"], option["values"],
                    format_func=lambda value, labels=labels: labels.get(value, value), key="scene_" + option["key"])
        fingerprint = hashlib.sha256(json.dumps([profile, [(name, hashlib.sha256(content).hexdigest()) for name, content in files],
            declarations, assignments, mappings, options], sort_keys=True, default=str).encode()).hexdigest()
        if st.session_state.get("scenario_fingerprint") != fingerprint:
            st.session_state.pop("scenario_result", None)
        if st.button("自动分析并生成方案"):
            st.session_state["scenario_analysis_fingerprint"] = fingerprint
        if st.session_state.get("scenario_analysis_fingerprint") != fingerprint:
            st.info("请点击“自动分析并生成方案”。修改文件或高级设置后需重新分析。")
            return
        inputs = recognise_inputs(files, declarations)
        inputs = apply_field_mappings(profile, inputs, assignments, mappings)
        analysis, plan = plan_scenario(profile, inputs, assignments)
        file_rows = []
        for item in inputs:
            uses = [match for match in plan["matches"] if match["file"] == item.filename]
            info = inspect_dataset(item.filename, item.dataset)
            measurements = {source for match in uses for source in match.get("mapping", {}).values() if source in item.dataset}
            if not measurements:
                measurements = {v for v in info.variables if DataRoleClassifier.physical_kind(item.dataset, v)}
            classified = sorted({DataRoleClassifier().classify(item.dataset, v) for v in measurements} - {"unclassified"})
            if not classified and item.dataset.attrs.get('data_role') not in {None,'unclassified'}:
                classified=[item.dataset.attrs['data_role']]
            file_rows.append({"文件": item.filename, "识别内容": "、".join(info.variables),
                              "模型用途": "、".join(m["name"] for m in uses) or "需要确认 / 暂无匹配用途",
                              "数据角色": "、".join(classified) or "待确认",
                              "状态": "✅ 可处理" if uses and all(m["can_process"] for m in uses) else "⚠ 需要确认或补充"})
        st.dataframe(file_rows, hide_index=True)
        st.dataframe([{"模型需求": m["name"], "文件": m["file"],
                       "状态": "✅ 已满足" if m["can_process"] and m["classification"] == "satisfied" else
                       "🔄 可以自动转换" if m["can_process"] else "❌ 缺失" if m["classification"] == "missing" else "⚠ 需要确认 / 不兼容",
                       "原因": "；".join(m.get("reasons", []) + m.get("missing", [])) or m.get("reason", "")}
                      for m in plan["matches"]], hide_index=True)
        st.caption("以上是模型配置范围内的匹配，不代表原生模型输入已验证。")
        with st.expander("识别详情与原生模型缺项"):
            st.json(analysis)
            st.write(plan["native_gaps"])
        st.markdown("#### 4. 处理方案 + 执行")
        st.write("系统计划执行以下操作：")
        st.dataframe(plan["steps"], hide_index=True)
        if assignments.get("background") and assignments.get("observation") and plan["can_process"]:
            st.info("将按模型配置执行：背景场 + 同变量监测站融合。算法详情见高级设置；地形与气象不会直接与浓度相加。")
        if plan["errors"]:
            st.warning("；".join(plan["errors"]))
        if plan.get("unassigned_files"):
            st.warning("以下文件不会参与执行，请确认用途：" + "、".join(plan["unassigned_files"]))
        if plan.get("partial_adapter"):
            st.info("数据尚不齐全：只准备已匹配的数据，不执行多源融合，也不生成完整模型输入。")
        st.caption("点击执行即确认当前识别与处理方案；原生运行还需排查上述模型缺项。")
        can_execute = (plan["can_process"] or plan["can_process_partial"]) and not plan.get("role_conflicts") and not blocked_inputs
        if st.button("执行模型数据准备", disabled=not can_execute):
            st.session_state.pop("scenario_result", None)
            result = execute_scenario(profile, inputs, assignments, **options)
            st.session_state["scenario_result"] = result
            st.session_state["scenario_fingerprint"] = fingerprint
        result = st.session_state.get("scenario_result")
        if result:
            st.write("转换后自查：" + result.quality["validation_status"])
            with st.expander("信息损失与模型兼容性自查"):
                st.json(result.quality["validation_report"])
                st.json(result.quality["completion"])
            if result.quality["completion"]["standardized_preprocessing_completed"]:
                st.success("配置范围内的标准化预处理完成；不代表原生模型输入已验证。")
            else:
                st.warning("仅完成部分数据准备，仍有缺失/失败项；不能作为完整模型输入。")
            if result.quality.get("negative_fused_cells"):
                st.warning(f"原始融合有 {result.quality['negative_fused_cells']} 个负浓度像元，按明确策略截为0；原始值、背景和标记都保留。")
            st.dataframe([{"变量": field, **stats} for field, stats in result.quality["fields"].items()], hide_index=True)
            with st.expander("结果数据预览"):
                st.write(result.output)
            for name, content in result.files.items():
                st.download_button(f"下载 {name}", content, file_name=name,
                    mime="application/x-netcdf" if name.endswith(".nc") else "application/json", key=f"scene_download_{name}")
            archive = package_results(list(result.files.items()), result.plan, result.analysis)
            st.download_button("下载全部模型准备结果(zip)", archive, file_name="model_preparation.zip", mime="application/zip")
    except Exception as exc:
        st.session_state.pop("scenario_result", None)
        st.error(f"无法完成模型数据准备：{exc}")
        if getattr(exc, "report", None):
            st.json(exc.report.as_dict())
            from ecodata.evaluation.self_check import validation_files
            for name, content in validation_files(exc.report).items():
                st.download_button(f"下载失败诊断 {name}", content, name, "application/json")
