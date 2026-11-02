"""Small explicit-format page; all processing goes through the shared Pipeline."""
import json

import streamlit as st

from ecodata.interface.workflow import load_netcdf, run_structure_trial
from ecodata.io.structures import FORMATS, GEOMETRY, StructureRequest, describe_hdf, read_structure, spatial_axes, table_frame


def render_structure_page() -> None:
    st.subheader("格式与数据结构转换")
    st.caption("上传 → 确认字段与坐标 → 选择转换规则 → 检查结果 → 下载。仅改格式不要求修改时间或空间分辨率。")
    upload = st.file_uploader("上传待转换文件", type=["csv", "tsv", "xlsx", "parquet", "json", "geojson", "nc", "nc4", "h5", "hdf5", "hdf", "tif", "tiff", "zip"], key="structure_upload")
    if upload is None:
        st.info("支持表格、NetCDF、NetCDF-compatible HDF5、二维原生 HDF5、GeoTIFF，以及 GeoJSON/Shapefile ZIP。Shapefile 请把 .shp/.shx/.dbf/.prj 一起打包。")
        return
    content = upload.getvalue()
    try:
        options = {}
        if upload.name.lower().endswith((".h5", ".hdf5", ".hdf")):
            try:
                load_netcdf(content)
            except Exception:
                inventory = describe_hdf(content)
                st.dataframe(inventory, hide_index=True)
                value_paths = [item["path"] for item in inventory if len(item["shape"]) == 2]
                axes = [item["path"] for item in inventory if len(item["shape"]) == 1]
                if not value_paths or not axes:
                    st.error("当前原生 HDF5 需要二维数值 dataset 和一维 x/y 坐标 dataset。")
                    return
                value = st.selectbox("HDF5 数值 dataset", ["请选择", *value_paths])
                x = st.selectbox("HDF5 x 坐标 dataset", ["请选择", *axes])
                y = st.selectbox("HDF5 y 坐标 dataset", ["请选择", *axes])
                hdf_crs = st.text_input("HDF5 坐标系 CRS（如 EPSG:4326）")
                if "请选择" in {value, x, y} or not hdf_crs.strip():
                    st.info("请明确选择 HDF5 数值及坐标，软件不会猜测 dataset 含义。")
                    return
                options = dict(hdf_value=value, hdf_x=x, hdf_y=y, crs=hdf_crs.strip())
        dataset = read_structure(upload.name, content, **options)
    except Exception as exc:
        st.error(f"无法读取：{exc}")
        return
    st.write("已识别维度", dict(dataset.sizes))
    with st.expander("输入预览", expanded=True):
        st.write(dataset)
    operations = {"仅转换文件格式（保留数据）": "preserve"}
    is_table = dataset.attrs.get("structure") in {"table", "vector"}
    if GEOMETRY in dataset:
        operations["矢量 → 栅格（选择属性与网格）"] = "rasterize"
    elif is_table:
        operations["表格 → 规则网格（明确坐标列）"] = "table_to_grid"
        operations["表格 → 点矢量（明确坐标列）"] = "table_points"
    else:
        operations.update({"网格 → 像元中心点": "grid_points", "分类栅格 → 区域多边形": "polygonize"})
    operation = operations[st.selectbox("转换任务", list(operations))]
    fields = [name for name in dataset.data_vars if name != GEOMETRY]
    if not fields:
        st.error("数据没有可转换的属性/变量")
        return
    variable = st.selectbox("处理字段", fields, index=fields.index("value") if "value" in fields else 0)
    unit = st.text_input("字段单位（留空保留原值，不自动换算）", value=str(dataset[variable].attrs.get("units", "")))
    crs = st.text_input("坐标系 CRS（空间转换必须确认）", value=dataset.attrs.get("crs", ""), help="可输入 EPSG:4326 或 WKT。已有 CRS 不能通过改标签直接替换。")
    x, y, time = "longitude", "latitude", ""
    bounds = None
    width, height, overlap, all_touched = 100, 100, "error", False
    if operation in {"table_to_grid", "table_points"}:
        columns = list(dataset.data_vars)
        x = st.selectbox("x / 经度列", columns, index=columns.index("longitude") if "longitude" in columns else 0)
        y = st.selectbox("y / 纬度列", columns, index=columns.index("latitude") if "latitude" in columns else 0)
        if operation == "table_to_grid":
            time = st.selectbox("时间列（可不选）", ["", *columns])
            st.caption("相同时间/坐标重复会报错；缺格保留为缺失，不擅自插值。文本和质量字段仍保留。")
    if operation == "rasterize":
        bounds_text = st.text_input("栅格外边界 xmin,ymin,xmax,ymax（使用输入 CRS 单位）")
        try:
            bounds = tuple(float(v.strip()) for v in bounds_text.split(",")) if bounds_text else None
        except ValueError:
            st.error("边界需要用英文逗号分隔的四个数字")
        width = st.number_input("栅格列数", min_value=2, max_value=2000, value=100)
        height = st.number_input("栅格行数", min_value=2, max_value=2000, value=100)
        overlap = st.selectbox("重叠要素规则", ["error", "first", "last"], format_func=lambda v: {"error": "发现重叠则提示", "first": "先出现的要素优先", "last": "后出现的要素优先"}[v])
        all_touched = st.checkbox("所有相交像元均赋值（默认按中心点/线绘制规则）")
    output_format = st.selectbox("目标文件格式", FORMATS, key="structure_format")
    slices = []
    if not is_table and (output_format == "GeoTIFF" or operation == "polygonize"):
        try:
            spatial = spatial_axes(dataset)
            for dim in dataset[variable].dims:
                if dim not in spatial:
                    index = st.selectbox(f"选择 {dim} 切片", range(dataset.sizes[dim]), format_func=lambda i, d=dim: str(dataset[d].values[i]) if d in dataset.coords else str(i))
                    slices.append((dim, index))
            st.caption("本次只导出所选二维切片；其他时刻或垂直层不合并到同一波段。")
        except Exception as exc:
            st.error(str(exc))
    if output_format in {"GeoJSON", "Shapefile ZIP"} and GEOMETRY not in dataset and operation not in {"grid_points", "polygonize", "table_points"}:
        st.info("矢量输出需要几何：网格请选择像元中心点或分类多边形。表格可先转成网格，再生成点。")
    request = StructureRequest(operation=operation, variable=variable, x=x, y=y, time=time, crs=crs.strip(), unit=unit.strip(), slices=tuple(slices), bounds=bounds, width=int(width), height=int(height), overlap=overlap, all_touched=all_touched)
    key = (dataset.attrs["source_sha256"], repr(options), repr(request), output_format)
    if st.button("执行格式转换", type="primary"):
        st.session_state.pop("structure_result", None)
        try:
            result = run_structure_trial(dataset, request, output_format)
            st.session_state["structure_result"] = result
            st.session_state["structure_key"] = key
        except Exception as exc:
            st.error(f"无法完成转换：{exc}")
    result = st.session_state.get("structure_result")
    if result is None or st.session_state.get("structure_key") != key:
        return
    st.success(f"转换完成：{result.report['导出文件']}，耗时 {result.runtime_seconds:.3f} 秒")
    st.write(result.output)
    try:
        st.dataframe(table_frame(result.output).head(100), hide_index=True)
    except Exception as exc:
        st.info(f"表格预览未展开：{exc}")
    for note in result.report["注意事项"]:
        st.info(note)
    with st.expander("字段、坐标与处理报告"):
        st.json(json.loads(result.report_bytes()))
    payload, filename, mime = result.export(output_format)
    st.download_button("下载转换文件", payload, filename, mime)
    st.download_button("下载元数据与处理报告", result.report_bytes(), "ecodata_report.json", "application/json")
