"""Validate actual exported files, not only in-memory conversion results."""
import numpy as np
import pandas as pd

from pyproj import CRS as ProjCRS
from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import read_structure, table_frame, projected_coordinate_names, GEOMETRY


def dataset_quality(dataset):
    stats = {}
    for name, field in dataset.data_vars.items():
        values = field.values
        missing = pd.isna(values)
        numeric = values.dtype.kind in "biuf"
        finite = np.isfinite(values) if numeric else ~missing
        stats[name] = {"dtype": str(field.dtype), "count": int(field.size),
                       "missing": int(missing.sum()), "NaN_ratio": float(missing.mean()) if field.size else 1.,
                       "valid_ratio": float(finite.mean()) if field.size else 0.,
                       "unit": field.attrs.get("units"),
                       "min": float(values[finite].min()) if numeric and finite.any() else None,
                       "max": float(values[finite].max()) if numeric and finite.any() else None}
    from ecodata.recognition import inspect_dataset
    info = inspect_dataset("output", dataset)
    return {"dimensions": dict(dataset.sizes), "variables": stats, "CRS": dataset.attrs.get("crs"),
            "resolution_km": info.spatial_resolution_km, "extent": info.extent,
            "time_resolution": info.temporal_resolution,
            "time_range": [str(dataset.time.min().values), str(dataset.time.max().values)] if "time" in dataset.coords and dataset.time.size else None}


def processing_quality(source, output):
    """Area-integral diagnostics only when physical semantics make them meaningful."""
    from ecodata.io.structures import spatial_axes, raster_plane
    from ecodata.evaluation.metrics import area_integral
    diagnostics = {}
    if output.attrs.get("fusion_method"):
        return {"conservation": "融合改变场值，不把融合前后差异冒充守恒误差"}
    for name in set(source.data_vars) & set(output.data_vars):
        semantics = output[name].attrs.get("spatial_semantics", output.attrs.get("spatial_semantics"))
        if semantics not in {"cell_mean", "flux"}:
            continue
        if not source.attrs.get("crs") or not output.attrs.get("crs"):
            diagnostics[name] = {"conservation": "缺少 CRS，不能计算面积积分"}
            continue
        x, y = spatial_axes(source)
        ox, oy = spatial_axes(output)
        before, after = source[name], output[name]
        extra = [dim for dim in before.dims if dim not in {x, y}]
        if len(extra) > 1 or (extra and (extra[0] not in after.dims or not np.array_equal(before[extra[0]], after[extra[0]]))):
            diagnostics[name] = {"conservation": "时间/层次不同，不能直接比较空间积分"}
            continue
        errors = []
        for index in range(before.sizes[extra[0]] if extra else 1):
            a = source[[name]].isel({extra[0]: index}, drop=True) if extra else source[[name]]
            b = output[[name]].isel({extra[0]: index}, drop=True) if extra else output[[name]]
            av, at, ac = raster_plane(a, name)
            bv, bt, bc = raster_plane(b, name)
            ab = (at.c, at.f + at.e*av.shape[0], at.c+at.a*av.shape[1], at.f)
            bb = (bt.c, bt.f + bt.e*bv.shape[0], bt.c+bt.a*bv.shape[1], bt.f)
            if ac != bc or not np.allclose(ab, bb) or not np.isfinite(av).all() or not np.isfinite(bv).all():
                diagnostics[name] = {"conservation": "范围/CRS 不同或存在缺失值，不报告不可比的守恒误差"}
                break
            if ac.is_projected:
                factor = ac.linear_units_factor[1]**2
                ai, bi = av.sum()*abs(at.a*at.e)*factor, bv.sum()*abs(bt.a*bt.e)*factor
            else:
                af = a[name].rename({axis: dest for axis, dest in ((x, "lon"), (y, "lat")) if axis != dest}).sortby("lat").sortby("lon")
                bf = b[name].rename({axis: dest for axis, dest in ((ox, "lon"), (oy, "lat")) if axis != dest}).sortby("lat").sortby("lon")
                ai, bi = area_integral(af), area_integral(bf)
            errors.append(float(abs(bi-ai)/abs(ai)) if ai else float(abs(bi)))
        else:
            diagnostics[name] = {"area_integral_relative_error": errors, "max_error": max(errors), "zero_source_uses_absolute_error": True}
            if output.attrs.get("spatial_regridding_method") == "conservative" and max(errors) > 1e-6:
                raise SchemaError(f"质量检查失败：{name} 保守重网格面积积分误差超过 1e-6")
    return diagnostics or {"conservation": "未声明可积/守恒语义，不虚构守恒指标"}


def validate_export(expected, content, filename, output_format):
    if not content:
        raise SchemaError("质量检查失败：输出文件为空")
    expected = projected_coordinate_names(expected)
    if not expected.data_vars or any(size == 0 for size in expected.sizes.values()):
        raise SchemaError("质量检查失败：结果没有变量或记录")
    numeric = [field.values for field in expected.data_vars.values() if field.dtype.kind in "biuf"]
    selected = expected.attrs.get("export_variable")
    if selected in expected and expected[selected].dtype.kind in "biuf":
        numeric = [expected[selected].values]
    if numeric and not any(np.isfinite(values).any() for values in numeric):
        raise SchemaError("质量检查失败：结果数值全部为 NaN/无效值")
    restored = read_structure(filename, content)
    checks = {"nonempty_file": True, "readable": True}
    notes = []
    if output_format == "GeoTIFF":
        from ecodata.io.structures import validate_geotiff, raster_plane
        check = validate_geotiff(content)
        plane, transform, crs = raster_plane(expected, selected or "value")
        field = next(iter(restored.data_vars))
        np.testing.assert_allclose(restored[field].values, plane, rtol=0, atol=0, equal_nan=True)
        checks.update(values=True, CRS=ProjCRS.from_user_input(restored.attrs["crs"]).equals(ProjCRS.from_user_input(crs), ignore_axis_order=True))
        if not checks["CRS"]:
            raise SchemaError("质量检查失败：GeoTIFF CRS 丢失或改变")
        if not np.allclose(check["transform"], list(transform)[:6], rtol=1e-12, atol=1e-9):
            raise SchemaError("质量检查失败：GeoTIFF transform 改变，像元位置不一致")
        if expected[selected or "value"].attrs.get("units") != restored[field].attrs.get("units"):
            raise SchemaError("质量检查失败：GeoTIFF 变量单位丢失")
        checks.update(transform=True, units=True)
        return {"passed": True, "checks": checks, "raster": check, "output": dataset_quality(restored)}
    spatial_vector = output_format in {"GeoJSON", "Shapefile ZIP"}
    if spatial_vector:
        if GEOMETRY not in restored or not restored.attrs.get("crs"):
            raise SchemaError("质量检查失败：矢量 geometry 或 CRS 丢失")
        notes.append("GeoJSON 坐标按 RFC 7946 转为 WGS84；几何不作逐字节比较。")
    expected_frame = table_frame(expected)
    actual_frame = table_frame(restored)
    fields = list(expected.data_vars)
    if spatial_vector:
        fields = [name for name in fields if name != GEOMETRY]
        if output_format == "GeoJSON" and {"longitude", "latitude"}.issubset(expected_frame) and "longitude" not in actual_frame:
            import json
            coordinates = [json.loads(value)["coordinates"] for value in actual_frame[GEOMETRY]]
            actual_frame["longitude"] = [point[0] for point in coordinates]
            actual_frame["latitude"] = [point[1] for point in coordinates]
    if len(expected_frame) != len(actual_frame):
        raise SchemaError("质量检查失败：导出回读记录数改变")
    if missing := set(fields)-set(actual_frame.columns):
        raise SchemaError(f"质量检查失败：输出变量丢失 {sorted(missing)}")
    if output_format in {"NetCDF", "HDF5-NetCDF"}:
        if dict(expected.sizes) != dict(restored.sizes):
            raise SchemaError("质量检查失败：科学数据维度改变")
        for axis in restored.coords:
            if axis in expected.coords and restored[axis].ndim == 1:
                restored = restored.sortby(axis)
                expected = expected.sortby(axis)
                if not np.array_equal(expected[axis].values, restored[axis].values):
                    raise SchemaError(f"质量检查失败：坐标/时间 {axis} 改变")
        # NetCDF may restore dimension insertion order differently. Compare the
        # same coordinate traversal rather than unrelated flattened row orders.
        dim_order = tuple(expected.dims)
        expected_frame = expected.to_dataframe(dim_order=dim_order).reset_index()
        actual_frame = restored.to_dataframe(dim_order=dim_order).reset_index()
        if expected.attrs.get("crs") and not restored.attrs.get("crs"):
            raise SchemaError("质量检查失败：CRS 丢失")
        if expected.attrs.get("crs") and not ProjCRS.from_user_input(expected.attrs["crs"]).equals(ProjCRS.from_user_input(restored.attrs["crs"]), ignore_axis_order=True):
            raise SchemaError("质量检查失败：CRS 改变")
        for name in fields:
            if expected[name].dtype.kind in "biuf" and restored[name].dtype.kind not in "biuf":
                raise SchemaError(f"质量检查失败：{name} 数值类型变为字符串")
            if expected[name].attrs.get("units") != restored[name].attrs.get("units"):
                raise SchemaError(f"质量检查失败：{name} 单位属性丢失")
        for key in expected.attrs:
            if key not in restored.attrs:
                raise SchemaError(f"质量检查失败：元数据 {key} 丢失")
    else:
        notes.append("表格文件不完整携带单位、CRS 和来源属性；必须同时保存 JSON 报告。")
    for name in fields:
        before, after = expected_frame[name], actual_frame[name]
        if expected[name].dtype.kind in "biuf":
            values = pd.to_numeric(after.replace("", np.nan), errors="raise").to_numpy(dtype=float)
            if not np.allclose(before.to_numpy(dtype=float), values, rtol=1e-10, atol=1e-12, equal_nan=True):
                raise SchemaError(f"质量检查失败：{name} 数值或缺失值改变")
        elif expected[name].dtype.kind in "Mm":
            if not np.array_equal(pd.to_datetime(before).to_numpy(), pd.to_datetime(after).to_numpy()):
                raise SchemaError(f"质量检查失败：{name} 时间改变")
        elif not spatial_vector:
            if not before.fillna("").astype(str).equals(after.fillna("").astype(str)):
                raise SchemaError(f"质量检查失败：{name} 文字/质量标记改变")
    return {"passed": True, "checks": dict(checks, records=True, variables=True, values=True, missing_values=True),
            "output": dataset_quality(restored), "notes": notes, "accuracy_metrics": "无独立真值，不计算 MAE/RMSE"}
