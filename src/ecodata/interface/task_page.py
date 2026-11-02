"""The single user-facing conversion task."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st
from rasterio.crs import CRS

from ecodata.core.data_types import SEMANTIC_TYPES
from ecodata.interface.workflow import QP_METHODS, TrialRequest, recommended_spatial_resolutions
from ecodata.interface.downloads import package_results
from ecodata.io.structures import FORMATS, GEOMETRY, StructureRequest, describe_hdf, table_frame, vector_raster_grid
from ecodata.models import RequirementStore
from ecodata.models.adaptation import field_requirements, prepare_input
from ecodata.optimization.downscaling import QPOptions
from ecodata.recognition import recognise_file
from ecodata.task import ConversionTaskEngine


STORE = RequirementStore()
ENGINE = ConversionTaskEngine()


def _read_upload(upload):
    options = {}
    try:
        return recognise_file(upload.name, upload.getvalue())
    except Exception as original:
        if not upload.name.lower().endswith((".h5", ".hdf5", ".hdf")):
            raise original
        inventory = describe_hdf(upload.getvalue())
        st.warning(f"{upload.name} 不是 NetCDF-compatible HDF5，请明确选择 dataset。")
        st.dataframe(inventory, hide_index=True)
        values = [item["path"] for item in inventory if len(item["shape"]) == 2]
        axes = [item["path"] for item in inventory if len(item["shape"]) == 1]
        value = st.selectbox("HDF5 数值 dataset", ["请选择", *values], key=f"hdf_value_{upload.name}")
        x = st.selectbox("HDF5 x 坐标 dataset", ["请选择", *axes], key=f"hdf_x_{upload.name}")
        y = st.selectbox("HDF5 y 坐标 dataset", ["请选择", *axes], key=f"hdf_y_{upload.name}")
        crs = st.text_input("HDF5 CRS", key=f"hdf_crs_{upload.name}")
        if "请选择" in {value, x, y} or not crs:
            return None
        options.update(hdf_value=value, hdf_x=x, hdf_y=y, crs=crs)
        return recognise_file(upload.name, upload.getvalue(), **options)


def _edit_requirements() -> None:
    with st.expander("编辑或新增模型需求（JSON）"):
        profiles = STORE.all()
        selected = st.selectbox("编辑对象", ["新建", *[item["id"] for item in profiles]], key="requirement_edit")
        template = STORE.template()
        value = template if selected == "新建" else STORE.get(selected)
        text = st.text_area("需求配置", json.dumps(value, ensure_ascii=False, indent=2), height=260, key=f"requirements_json_{selected}")
        if st.button("保存模型需求"):
            try:
                STORE.save(json.loads(text))
                st.success("模型需求已保存；下次任务可直接选择。")
            except Exception as exc:
                st.error(f"无法保存：{exc}")


def _operation(dataset, output_format: str, *, has_xy: bool) -> tuple[str, list[str]]:
    is_vector = GEOMETRY in dataset
    is_table = dataset.attrs.get("structure") == "table"
    options = ["preserve"]
    if is_vector:
        options += ["rasterize"]
    elif is_table and has_xy:
        options += ["table_to_grid", "table_points"]
    elif not is_table:
        options += ["grid_points", "polygonize"]
    if is_vector and output_format in {"GeoTIFF", "NetCDF", "HDF5-NetCDF"}:
        recommended = "rasterize"
    elif is_table and has_xy and output_format in {"GeoTIFF", "NetCDF", "HDF5-NetCDF"}:
        recommended = "table_to_grid"
    elif not is_table and not is_vector and output_format in {"CSV", "TSV", "Excel", "Parquet", "JSON", "GeoJSON", "Shapefile ZIP"}:
        recommended = "grid_points"
    else:
        recommended = "preserve"
    return recommended, options


def render_task_page() -> None:
    st.subheader("数据转换任务")
    st.markdown("**上传数据 → 自动识别 → 选择目标模型或自定义输出 → 自动生成处理流程 → 执行 → 下载结果**")
    target_mode = st.radio("输出目标", ["自定义输出", "环境模型"], horizontal=True)
    if target_mode == "环境模型":
        from ecodata.interface.model_page import render_model_page
        render_model_page()
        return
    uploads = st.file_uploader("上传一个或多个数据文件", type=["csv", "tsv", "xlsx", "parquet", "json", "geojson", "nc", "nc4", "h5", "hdf5", "hdf", "tif", "tiff", "zip"], accept_multiple_files=True)
    if not uploads:
        st.info("支持 CSV、Excel、Parquet、NetCDF、HDF5、GeoTIFF、GeoJSON 和 Shapefile ZIP。")
        _edit_requirements()
        return
    if not isinstance(uploads, (list, tuple)):
        uploads = [uploads]
    parsed, errors, input_failures = [], [], []
    for upload in uploads:
        try:
            item = _read_upload(upload)
            if item:
                parsed.append(item)
        except Exception as exc:
            errors.append(f"{upload.name}：{exc}")
            input_failures.append({"输入文件": upload.name, "错误原因": "识别失败：" + str(exc)})
    if errors:
        st.error("\n".join(errors))
    if not parsed:
        return
    st.markdown("#### 1. 自动识别结果")
    rows = []
    for _, info in parsed:
        rows.append({"文件": info.filename, "格式": info.format, "结构": info.structure, "变量": "、".join(info.variables),
                     "单位": "；".join(f"{k}={v or '待确认'}" for k, v in info.units.items()), "时间": info.time_coordinate or "无",
                     "空间坐标": f"{info.x_coordinate or '-'} / {info.y_coordinate or '-'}", "CRS": "已识别" if info.crs else "待确认",
                     "数据类型": "；".join(f"{k}={v}" for k, v in info.semantic_types.items() if k in info.variables),
                     "需确认": "、".join(info.confirmations_required) or "无"})
        rows[-1]["源空间分辨率 km"] = f"{info.spatial_resolution_km:.6g}" if info.spatial_resolution_km else "未识别"
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    primary_index = st.selectbox("主数据文件", range(len(parsed)), format_func=lambda i: parsed[i][1].filename)
    dataset, info = parsed[primary_index]
    ordered = [dataset, *[item for i, (item, _) in enumerate(parsed) if i != primary_index]]
    from ecodata.evaluation.quality_gates import RawInputQualityGate
    input_quality = RawInputQualityGate().validate(dataset, intended_operation="conversion")
    if input_quality["status"] == "PASS":
        st.success("原始数据质量：通过")
        input_quality_confirmed = True
    elif input_quality["status"] == "WARN":
        st.warning("原始数据质量：警告。原始值不会被自动删除或修改。")
        input_quality_confirmed = st.checkbox("我已查看警告，继续使用原始数据", key="confirm_input_quality")
    else:
        st.error("原始数据质量：阻止。请修正关键结构、字段或坐标问题后再处理。")
        input_quality_confirmed = False
    with st.expander("查看原始数据质量检查"):
        st.json(input_quality)
    with st.expander("数据预览", expanded=True):
        st.write(dataset)
        try:
            st.dataframe(table_frame(dataset).head(100), hide_index=True)
        except Exception:
            pass

    st.markdown("#### 2. 目标要求")
    model = None
    model_mapping = {}
    model_content = ()
    role_assignments = {}
    _edit_requirements()

    has_xy = bool(info.x_coordinate and info.y_coordinate)
    is_spatial = has_xy or GEOMETRY in dataset or dataset.attrs.get("structure") == "grid"
    if dataset.attrs.get("structure") == "table" and not has_xy:
        available_formats = [item for item in FORMATS if item not in {"GeoTIFF", "GeoJSON", "Shapefile ZIP"}]
        st.info("未识别到经纬度或几何字段：本任务按非空间监测表处理，已隐藏 CRS、网格和空间分辨率设置。")
    else:
        available_formats = list(FORMATS)
    output_default = model.get("output_format", "NetCDF") if model else "NetCDF"
    output_format = st.selectbox("输出格式", available_formats, index=available_formats.index(output_default) if output_default in available_formats else 0)
    if model:
        model_match["标准化输出格式"] = output_format
        st.caption("当前生成标准化预处理格式；模型原生格式要求保存在匹配报告中。")
    recommended, operations = _operation(dataset, output_format, has_xy=has_xy)
    labels = {"preserve": "保持当前结构，仅转换文件容器", "table_to_grid": "表格 → 规则网格", "table_points": "表格 → 点矢量",
              "grid_points": "网格 → 像元中心点", "rasterize": "矢量 → 栅格", "polygonize": "分类栅格 → 多边形"}
    if GEOMETRY in dataset:
        operation = recommended
        st.caption("数据结构转换：" + labels[operation])
    elif len(operations) == 1:
        operation = operations[0]
        st.caption("数据结构：普通表格，保持表格结构（没有可用空间坐标，因此无需选择空间结构转换）。")
    else:
        operation = st.selectbox("数据结构转换", operations, index=operations.index(recommended), format_func=lambda key: labels[key])
    fields = [name for name in dataset.data_vars if name != GEOMETRY]
    raster_mask = operation == "rasterize" and not info.rasterizable_fields
    excluded = {info.x_coordinate, info.y_coordinate, info.time_coordinate, "uncertainty"}
    measurements = [name for name in fields if name not in excluded and info.semantic_types.get(name) in {"continuous", "integer_count", "categorical"}]
    if not measurements:
        measurements = [name for name in fields if name not in excluded]
    if model:
        measurements = list(model_content) if model_content else [value for value in model_mapping.values() if value]
    if operation == "rasterize":
        st.caption(f"几何类型：{'、'.join(info.geometry_types)}｜原始范围：{info.extent}")
        with st.expander("属性字段识别结果"):
            st.json(info.field_types)
        if raster_mask:
            variables = ("value",)
            st.info("没有可直接栅格化的有限数值属性，自动生成 value=1 覆盖掩膜；未覆盖区域为 NoData。")
        else:
            variables = (st.selectbox("栅格化字段（已排除文字、标识符和含缺失值的字段）", info.rasterizable_fields),)
    else:
        variables = tuple(st.multiselect("需要处理的变量（可多选）", fields, default=measurements, help="系统已默认勾选识别到的数值指标。只转换文件格式时，未勾选字段也会原样保留。"))
    if not variables:
        st.warning("请至少选择一个处理变量。")
        variable = fields[0]
    else:
        variable = variables[0]
    semantic_types = {"categorical"} if raster_mask else {info.semantic_types.get(name, "continuous") for name in variables}
    semantic = next(iter(semantic_types)) if len(semantic_types) == 1 else "multidimensional_array"
    if raster_mask:
        st.caption("输出类型：覆盖掩膜（1=覆盖，NoData=未覆盖），不是新增环境观测值。")
    else:
        st.caption("识别类型：" + "；".join(f"{name}＝{SEMANTIC_TYPES[info.semantic_types.get(name, 'continuous')].name_zh}" for name in variables))
    units = {}
    if operation == "rasterize":
        units[variable] = "1" if raster_mask else info.units.get(variable, "")
        st.caption("变量单位：" + (units[variable] or "原文件未提供，输出不虚构单位"))
    elif variables:
        st.write("变量单位")
        unit_columns = st.columns(min(2, len(variables)))
        for index, name in enumerate(variables):
            units[name] = unit_columns[index % len(unit_columns)].text_input(name, value=info.units.get(name, ""), key=f"unit_{info.filename}_{name}", placeholder="实际单位；未知填 unknown，无量纲填 1 或 无量纲")
            if units[name]:
                dataset[name].attrs["units"] = units[name]
    unit = units.get(variable, "")
    if operation != "rasterize" and any(not value.strip() for value in units.values()):
        st.warning("部分变量没有单位。进行时空处理前请填写实际单位；无法确定可填 unknown，无量纲可填 1 或 无量纲。仅转换文件格式可以留空。")
    x, y, time_name = info.x_coordinate or "", info.y_coordinate or "", info.time_coordinate or ""
    crs = ""
    if is_spatial:
        if info.crs:
            crs = info.crs
            st.caption("CRS：" + CRS.from_user_input(crs).to_string() + "（自动读取）")
        else:
            crs = st.text_input("CRS", value=info.crs or "", help="只有空间数据需要；经纬度通常为 EPSG:4326。")
    bounds, width, height, overlap = None, 100, 100, "error"
    all_touched, raster_ready = False, True
    if operation in {"table_to_grid", "table_points"}:
        columns = list(dataset.data_vars)
        x = st.selectbox("x / 经度字段", columns, index=columns.index(x) if x in columns else 0)
        y = st.selectbox("y / 纬度字段", columns, index=columns.index(y) if y in columns else 0)
        if operation == "table_to_grid":
            time_name = st.selectbox("时间字段（可留空）", ["", *columns], index=["", *columns].index(time_name) if time_name in columns else 0)
    if operation == "rasterize":
        coordinate_unit = "CRS 坐标单位"
        if crs:
            vector_crs = CRS.from_user_input(crs)
            coordinate_unit = vector_crs.linear_units if vector_crs.is_projected else "度"
        default_resolution = max(info.extent[2]-info.extent[0], info.extent[3]-info.extent[1])/100 if info.extent else 1.0
        resolution = st.number_input(f"目标栅格分辨率（{coordinate_unit}/像元）", min_value=0.00000001,
                                     value=max(default_resolution, 0.00000001), format="%.8f", key=f"raster_resolution_{info.filename}")
        st.caption("使用原 CRS，不自动重投影；范围自动覆盖所有要素，边缘向外补齐完整像元。")
        try:
            bounds, width, height = vector_raster_grid(info.extent, resolution)
            st.caption(f"输出范围：{bounds}｜网格：{width}×{height}")
        except Exception as exc:
            raster_ready = False
            st.error(str(exc))
        strategy = st.selectbox("栅格化策略", ["像元中心", "所有接触像元"], help="点和线采用 Rasterio 对应绘制规则；所有接触像元可保留细小面要素。")
        all_touched = strategy == "所有接触像元"
        if not raster_mask:
            overlap = st.selectbox("重叠要素策略", ["error", "first", "last"],
                                   format_func=lambda value: {"error": "重叠时报错（不擅自覆盖）", "first": "先出现要素优先", "last": "后出现要素优先"}[value])
        else:
            overlap = "last"
    target_time = ""
    station_field = x
    if info.time_coordinate and GEOMETRY not in dataset:
        use_time = st.checkbox("进行时间分辨率转换", value=bool(model and model.get("temporal_resolution")))
        if use_time:
            time_name = info.time_coordinate
            default_frequency = model.get("temporal_resolution") or "1h" if model else "1h"
            target_time = st.text_input("目标时间分辨率", value=default_frequency, disabled=bool(model and model.get("temporal_resolution")))
            if dataset.attrs.get("structure") == "table" and operation == "preserve":
                id_fields = [name for name in fields if info.semantic_types.get(name) == "string_identifier" and name != time_name]
                preferred = next((name for name in id_fields if any(token in name.lower() for token in ("station", "code", "id", "编码"))), id_fields[0] if id_fields else fields[0])
                station_field = st.selectbox("站点唯一标识字段", id_fields or fields, index=(id_fields or fields).index(preferred))
                st.caption("系统会在每个站点自身已有观测时间范围内处理，不跨站点、不向两端外推。")
    elif model and model.get("temporal_resolution"):
        st.error("目标模型要求时间分辨率，但输入中未识别到时间字段。")
    temporal_semantics = dataset[variable].attrs.get("temporal_semantics", dataset.attrs.get("temporal_semantics")) if variable in dataset else None
    if target_time and "time" in dataset.dims:
        temporal_options = ["", "instantaneous", "average", "accumulated"]
        temporal_semantics = st.selectbox("时间语义", temporal_options,
            index=temporal_options.index(temporal_semantics) if temporal_semantics in temporal_options else 0,
            format_func=lambda value: {"": "原文件未声明，请确认", "instantaneous": "瞬时值", "average": "时段平均值（需要时间边界）", "accumulated": "时段累计值（需要时间边界）"}[value])
    source_space = target_space = 0.0
    spatial_method = "auto"
    needs_conservative_semantics = False
    spatial_semantics = dataset[variable].attrs.get("spatial_semantics", dataset.attrs.get("spatial_semantics")) if variable in dataset else None
    optimization_options = None
    training_observations = None
    training_filename = ""
    if is_spatial and GEOMETRY not in dataset:
        detected_space = dataset.attrs.get("grid_resolution_km") or info.spatial_resolution_km
        if not detected_space or detected_space <= 0:
            detected_space = None
            st.warning("源空间分辨率未识别：" + (info.spatial_resolution_reason or "输入未提供可换算的像元尺寸或源空间分辨率，请在需要空间处理时填写。"))
        if info.pixel_width_km and info.pixel_height_km:
            st.caption(f"像元宽度 {info.pixel_width_km:.6f} km；像元高度 {info.pixel_height_km:.6f} km。")
        source_space = st.number_input("源空间分辨率 km（空间转换时必填）", min_value=0.000001, value=float(detected_space) if detected_space else None,
                                       format="%.6f", placeholder="未识别，需要空间处理时请填写", key=f"source_space_{info.filename}_{dataset.attrs.get('source_sha256', '')}") or 0.0
        target_default = float(model.get("spatial_resolution_km") or 0.0) if model else 0.0
        recommendation = 0.0
        if source_space and not (model and model.get("spatial_resolution_km")):
            options = recommended_spatial_resolutions(dataset, source_space)
            recommendation = st.selectbox("推荐目标空间分辨率", [0.0, *options],
                                          format_func=lambda value: "自定义（在下面填写，0 表示不处理）" if value == 0 else f"{value:.6g} km（{'细化' if value < source_space else '聚合'}）",
                                          key=f"resolution_recommendation_{info.filename}_{source_space}")
            if recommendation:
                target_default = recommendation
        target_space = st.number_input("目标空间分辨率 km（0 表示不处理）", min_value=0.0, value=target_default, format="%.6f",
                                       key=f"target_resolution_{info.filename}_{recommendation}", disabled=bool(model and model.get("spatial_resolution_km")))
        if target_space:
            method_labels = {
                "auto": "自动选择（按数据语义）", "nearest": "最近邻", "bilinear": "双线性插值",
                "conservative": "保守重网格", "constrained": "普通约束 QP",
                "peak_aware": "原始峰值感知 QP", "adaptive_smoothing": "自适应平滑 QP",
                "adaptive_peak": "自适应平滑＋可信峰值约束 QP",
            }
            methods = ["auto", "nearest", "bilinear", "conservative"]
            if semantic == "continuous" and len(variables) == 1:
                methods += ["constrained", "peak_aware", "adaptive_smoothing", "adaptive_peak"]
            with st.expander("高级设置：算法选择", expanded=False):
                spatial_method = st.selectbox("空间转换算法", methods, format_func=lambda value: method_labels[value])
            needs_conservative_semantics = spatial_method == "conservative" or (spatial_method == "auto" and target_space > source_space and semantic == "continuous")
            if needs_conservative_semantics and spatial_semantics not in {"cell_mean", "flux"}:
                spatial_semantics = st.selectbox("像元含义（保守重网格必需）", ["", "cell_mean", "flux"],
                                               format_func=lambda value: {"": "请选择", "cell_mean": "单元平均值", "flux": "通量"}[value],
                                               help="保守重网格需要明确物理含义；不了解像元含义时请使用最近邻或双线性。")
            if spatial_method in QP_METHODS:
                required_station_fields = {"station_id", "time", "latitude", "longitude", "variable", "value", "unit", "qc_flag", "split"}
                station_candidates = []
                for index, (candidate, candidate_info) in enumerate(parsed):
                    try:
                        frame = table_frame(candidate)
                    except Exception:
                        continue
                    if required_station_fields.issubset(frame.columns):
                        station_candidates.append((index, candidate_info.filename, frame))
                if station_candidates:
                    selected_station = st.selectbox(
                        "训练监测站文件",
                        range(len(station_candidates)),
                        format_func=lambda index: station_candidates[index][1],
                        help="只允许使用 split=train 且 qc_flag=VALID 的站点；验证集不会参与优化。",
                    )
                    _, training_filename, training_observations = station_candidates[selected_station]
                else:
                    st.error("所选 QP 需要另行上传训练监测站 CSV，必须包含 station_id、time、latitude、longitude、variable、value、unit、qc_flag、split。")
                with st.expander("QP 高级优化参数", expanded=False):
                    st.caption("一般保持推荐值。最大迭代次数是求解上限，不代表迭代越多结果一定越准确。")
                    initial_method = st.selectbox("初始高分辨率场", ["bilinear", "nearest", "conservative"])
                    solver = st.selectbox(
                        "优化求解器",
                        ["OSQP", "CLARABEL", "SCS"],
                        help="OSQP 是默认QP求解器；Clarabel为内点法求解器；SCS为一阶锥优化求解器。三者都使用同一套约束模型。",
                    )
                    max_iter = int(st.number_input("最大迭代次数", min_value=100, max_value=200000, value=20000, step=1000))
                    eps_abs = float(st.number_input("绝对收敛容差", min_value=1e-8, max_value=1e-2, value=1e-5, format="%.8f"))
                    eps_rel = float(st.number_input("相对收敛容差", min_value=1e-8, max_value=1e-2, value=1e-5, format="%.8f"))
                    observation_tolerance = float(st.number_input("观测误差容差", min_value=0.0, value=0.35, step=0.05))
                    alpha = float(st.number_input("平滑权重 alpha", min_value=0.0001, value=1.0, step=0.1))
                    beta = float(st.number_input("站点观测权重 beta", min_value=0.0, value=20.0, step=1.0))
                    gamma = float(st.number_input("初始场保持权重 gamma", min_value=0.0, value=0.8, step=0.1))
                    peak_threshold = float(st.number_input("可信峰值阈值", min_value=0.0, value=45.0, step=1.0))
                    soft_penalty = float(st.number_input("软约束惩罚系数", min_value=0.0001, value=80.0, step=5.0))
                optimization_options = QPOptions(
                    initial_method=initial_method, solver=solver, alpha=alpha, beta=beta, gamma=gamma,
                    peak_threshold=peak_threshold, observation_tolerance=observation_tolerance,
                    soft_constraint_penalty=soft_penalty, max_iter=max_iter,
                    eps_abs=eps_abs, eps_rel=eps_rel,
                )
    fusion_ready = True
    fuse = st.checkbox(
        "对多个兼容网格进行数据融合",
        disabled=len(parsed) < 2 or not is_spatial or GEOMETRY in dataset or spatial_method in QP_METHODS,
        help="QP 上传中的训练站文件不是融合数据源；如需融合后优化，请先完成融合任务，再对结果运行 QP。",
    )
    if fuse:
        fusion_method = st.selectbox("融合方式", ["auto", "weighted", "residual", "ordinary_kriging", "universal_kriging", "external_drift_kriging"], format_func=lambda value: {"auto": "自动检查并推荐（不强行融合）", "weighted": "多源反方差加权融合", "residual": "背景场 + 监测站残差融合", "ordinary_kriging": "GSTools Ordinary Kriging", "universal_kriging": "GSTools Universal Kriging（线性空间 drift）", "external_drift_kriging": "GSTools External Drift Kriging（辅助场约束）"}[value])
        covariates = ()
        if fusion_method == "auto":
            from ecodata.fusion.planning import FusionPlanner
            recommendation = FusionPlanner().plan(ordered, variable, "auto")
            st.caption("推荐方案：" + recommendation["method"] + "；" + recommendation["relation_check"])
            fusion_ready = recommendation["ready"]
            for reason in recommendation["reasons"]:
                st.error("当前不能自动融合：" + reason)
        if fusion_method == "external_drift_kriging":
            candidates = sorted({name for ds, desc in parsed for name in desc.variables
                if name != variable and desc.semantic_types.get(name) == "continuous" and ds[name].ndim >= 2})
            covariates = tuple(st.multiselect("辅助变量（DEM/气象等连续场，不与目标变量相加）", candidates))
        if fusion_method.endswith("kriging"):
            st.info("上传目标网格、训练站（split=train、qc_flag=VALID、station_id）及可选辅助场。独立验证站不得参与拟合；先使用转换任务统一目标时间/分辨率。辅助关系不可靠时拒绝 External Drift。")
        st.caption("加权融合自动对齐到主数据的 CRS、网格、单位和共同时间范围；每个源必须声明同一 physical_variable 并提供 uncertainty。残差融合目前仅支持同一时间点、EPSG:4326 的背景场与站点。")
        rows = []
        for source, desc in parsed:
            rows.append({"文件": desc.filename, "CRS": desc.crs or "缺少", "范围": str(desc.extent),
                         "时间分辨率": desc.temporal_resolution or "待确认/静态", "空间分辨率km": desc.spatial_resolution_km,
                         "单位": desc.units.get(variable, "缺少"), "物理变量": source[variable].attrs.get("physical_variable", "未声明") if variable in source else "站点表或缺少字段",
                         "uncertainty": "已有" if "uncertainty" in source else "无（仅残差融合不要求）"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.info("处理计划：检查变量语义 → 单位换算 → 共同时间范围对齐 → CRS/网格对齐 → 有效观测加权融合。无重叠、语义不兼容或缺少必要元数据时停止，报告具体原因。")
        if len(variables) != 1:
            st.error("当前融合任务必须明确选择一个物理变量；其他字段仍会保留在原始输入中。")
        if fusion_method == "weighted":
            from ecodata.fusion.alignment import fusion_compatibility
            compatibility = fusion_compatibility(ordered, variable)
            fusion_ready = compatibility["ready"]
            if fusion_ready:
                st.caption("这些数据具有相同物理语义、可换算单位与空间信息；将检查实际时间/空间交集后融合。")
            for reason in compatibility["reasons"]:
                st.error("当前不能融合：" + reason)
    else:
        fusion_method = "weighted"
        covariates = ()

    if source_space:
        dataset.attrs["grid_resolution_km"] = float(source_space)
    structure = StructureRequest(operation=operation, variable=variable, x=station_field if target_time and operation == "preserve" else x, y=y, time=time_name, crs=crs, unit=unit,
                                 bounds=bounds, width=width, height=height, overlap=overlap, all_touched=all_touched, raster_mask=raster_mask)
    processing = TrialRequest(target_time_frequency=target_time.strip() or None, target_spatial_resolution_km=target_space or None,
                              spatial_method=spatial_method, variable=variable, variables=variables, temporal_semantics=temporal_semantics or None, unit=unit,
                              target_model=model["id"] if model else "custom", output_format=output_format, semantic_type=semantic,
                              optimization_options=optimization_options, spatial_semantics=spatial_semantics, fusion_method=fusion_method, covariates=covariates)
    plan = ENGINE.plan(structure, processing, fuse=fuse, model=model)
    st.markdown("#### 3. 自动处理流程")
    st.dataframe(pd.DataFrame([item.as_dict() for item in plan]), hide_index=True, width="stretch")
    model_missing = bool(model and (missing or model_match["unsupported"] or model_match["unit_issues"]))
    qp_missing_training = spatial_method in QP_METHODS and training_observations is None
    ready = input_quality_confirmed and bool(variables) and raster_ready and fusion_ready and not model_missing and not (fuse and len(variables) != 1) and not qp_missing_training
    if target_time and "time" in dataset.dims and not temporal_semantics:
        ready = False
    if (target_space or target_time) and any(not value.strip() for value in units.values()):
        ready = False
    if target_space and (not source_space or (spatial_method == "conservative" and spatial_semantics not in {"cell_mean", "flux"})):
        ready = False
    if needs_conservative_semantics and spatial_semantics not in {"cell_mean", "flux"}:
        ready = False
    fingerprint = hashlib.sha256(repr((structure, processing, fuse, role_assignments, training_filename, model_mapping, model_content)).encode()).hexdigest()
    fingerprint = hashlib.sha256((fingerprint + repr([(item.filename, data.attrs.get("source_sha256")) for data, item in parsed]) + str(primary_index)).encode()).hexdigest()
    batch = len(uploads) > 1 and not fuse and model is None and spatial_method not in QP_METHODS
    if batch:
        st.caption("多个独立文件将使用当前设置分别转换，完成后可下载全部结果 ZIP。每个文件必须具备所选字段。")
    if st.button("执行数据转换", type="primary", disabled=not ready):
        st.session_state.pop("task_result", None)
        st.session_state.pop("task_exports", None)
        st.session_state.pop("task_zip", None)
        try:
            with st.spinner("正在执行统一处理流程……"):
                batch_results = []
                batch_failures = list(input_failures)
                if batch:
                    for source, source_info in parsed:
                        try:
                            missing_fields = set() if raster_mask else set(variables) - set(source.data_vars)
                            if missing_fields:
                                raise ValueError(f"缺少所选字段：{', '.join(sorted(missing_fields))}")
                            converted = ENGINE.execute([source], structure, processing)
                            converted.export(output_format)
                            batch_results.append((source_info.filename, converted))
                        except Exception as exc:
                            failed = {"输入文件": source_info.filename, "错误原因": str(exc)}
                            if getattr(exc, "report", None):
                                failed["validation_report"] = exc.report.as_dict()
                            batch_failures.append(failed)
                    if not batch_results:
                        raise ValueError(f"所有文件转换失败：{batch_failures}")
                if model:
                    # Recheck using user-confirmed units, then apply mapping before the existing Pipeline.
                    current_info = replace(info, units={name: dataset[name].attrs.get("units", "") for name in info.variables})
                    prepared, model_match = prepare_input(dataset, current_info, model, model_mapping, model_content)
                    if not model_match["ready"]:
                        raise ValueError("所选输入类型仍有缺失字段、单位或结构问题，请先确认映射。")
                    rename = {source: target for target, source in model_mapping.items() if source}
                    target_variable = rename.get(structure.variable, structure.variable)
                    model_structure = replace(structure, variable=target_variable,
                                              unit=prepared[target_variable].attrs.get("units", ""),
                                              x=rename.get(structure.x, structure.x), y=rename.get(structure.y, structure.y),
                                              time=rename.get(structure.time, structure.time))
                    model_processing = replace(processing, variable=rename.get(processing.variable, processing.variable),
                                               variables=tuple(rename.get(name, name) for name in processing.variables),
                                               unit=prepared[rename.get(processing.variable, processing.variable)].attrs.get("units", ""))
                result = ENGINE.execute(
                    [prepared, *ordered[1:]] if model else ordered,
                    model_structure if model else structure, model_processing if model else processing, fuse=fuse, model=model,
                    training_observations=training_observations,
                ) if not batch else batch_results[0][1]
                result.report["输入文件角色"] = role_assignments
                result.report["双层质量控制"] = {"原始数据输入检查": input_quality}
                result.report["数据识别报告"] = [item.as_dict() for _, item in parsed]
                result.report["识别失败文件"] = input_failures
                if operation == "rasterize":
                    result.report["矢量识别"] = info.as_dict()
                    result.report["栅格化设置"] = {"范围": bounds, "列数": width, "行数": height,
                                                "策略": strategy, "重叠规则": overlap, "覆盖掩膜": raster_mask}
                if model:
                    result.report["模型需求匹配"] = model_match
                    from ecodata.evaluation.self_check import PostConversionValidator, require_usable
                    audit = PostConversionValidator().validate(dataset, result.output, kind="model_adaptation", history=result.history,
                        field_mapping={source: target for target, source in model_mapping.items() if source},
                        model_check=result.report["validation_report"].get("model_compatibility_check"))
                    require_usable(audit)
                    result.report["validation_report"] = audit.as_dict()
                    result = replace(result, original=dataset)
                if batch:
                    exports = []
                    for index, (source_name, converted) in enumerate(batch_results, 1):
                        content, name, mime = converted.export(output_format)
                        filename = f"{index:02d}_{Path(source_name).stem}_{name}"
                        exports.append((filename, content, mime))
                    conversion_report = [{"输入文件": name, "报告": converted.report} for name, converted in batch_results]
                    recognition_report = [item.as_dict() for _, item in parsed]
                    archive = package_results([(name, content) for name, content, _ in exports], conversion_report, recognition_report, failures=batch_failures)
                    result.report["批量失败文件"] = batch_failures
                    st.session_state["task_exports"] = exports
                    st.session_state["task_zip"] = archive
            st.session_state["task_result"] = result
            st.session_state["task_fingerprint"] = fingerprint
        except Exception as exc:
            st.error(f"转换失败：{exc}")
            if getattr(exc, "report", None):
                st.json(exc.report.as_dict())
                from ecodata.evaluation.self_check import validation_files
                for name, content in validation_files(exc.report).items():
                    st.download_button(f"下载失败诊断 {name}", content, name, "application/json", key="failed_" + name)
    result = st.session_state.get("task_result")
    if result is None:
        return
    if st.session_state.get("task_fingerprint") != fingerprint:
        st.info("设置已改变，请重新执行。")
        return
    st.markdown("#### 4. 结果与下载")
    try:
        payload, filename, mime = result.export(output_format)
        if model:
            filename = f"{model['id']}_{model['selected_input']}_standardized{Path(filename).suffix}"
        if output_format == "GeoTIFF":
            from ecodata.io.structures import validate_geotiff
            result.report["GeoTIFF输出检查"] = validate_geotiff(payload)
            st.json(result.report["GeoTIFF输出检查"])
    except Exception as exc:
        st.error(f"输出检查失败：{exc}")
        if getattr(exc, "report", None):
            from ecodata.evaluation.self_check import validation_files
            st.json(exc.report.as_dict())
            for name, content in validation_files(exc.report).items():
                st.download_button(f"下载失败诊断 {name}", content, name, "application/json", key="export_failed_" + name)
        return
    validation = result.report["validation_report"]
    result.report.setdefault("双层质量控制", {})["融合或转换结果检查"] = validation
    if standards := validation.get("standards_compliance"):
        if standards["status"] != "PASS":
            st.warning(f"CF/ACDD 检查：{standards['status']}。文件可无损保存，但尚未满足所检查的标准；不能当作已验证模型输入。详细原因见自动质量报告。")
        else:
            st.caption("CF/ACDD 文件规范检查通过；不代表模型原生输入已验证。")
    message = f"转换结果：{validation['status']}，运行时间 {result.runtime_seconds:.3f} 秒"
    if validation["status"] == "WARN":
        st.warning(message + "。存在信息变化，请查看自查报告。")
    else:
        st.success(message)
    check_labels = {"numerical_integrity": "数值完整性", "variables": "变量", "units": "单位", "time": "时间", "CRS": "CRS", "missing_values": "缺失值", "metadata": "元数据", "physical_range": "物理范围", "conservation": "守恒性", "structure": "数据结构", "model_compatibility": "模型需求"}
    st.dataframe([{"检查": check_labels.get(name, name), "状态": level} for name, level in validation["checks_by_category"].items()], hide_index=True)
    for failure in result.report.get("批量失败文件", []):
        st.warning(f"{failure['输入文件']} 未完成：{failure['错误原因']}。其原因已写入 ZIP，未标记为成功。")
    with st.expander("自动质量检查"):
        st.json(validation)
        st.json(result.report.get("自动质量检查", {}))
    st.write(result.output)
    st.dataframe(table_frame(result.output).head(100), hide_index=True)
    left, right = st.columns(2)
    left.download_button("下载转换结果", payload, filename, mime)
    right.download_button("下载处理与模型匹配报告", result.report_bytes(), "ecodata_report.json", "application/json")
    for name, content in result.validation_files().items():
        st.download_button(f"下载自查 {name}", content, name, "application/json", key="validation_" + name)
    archive = st.session_state.get("task_zip")
    if archive:
        st.download_button("下载全部结果(zip)", archive, "results.zip", "application/zip")
        for name, content, mime in st.session_state["task_exports"]:
            st.download_button(f"下载 {name}", content, name, mime, key=f"batch_download_{name}")
    with st.expander("完整处理记录"):
        st.json(json.loads(result.report_bytes()))
