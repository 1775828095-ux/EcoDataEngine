"""Explicit structure conversion using xarray, Rasterio and PyShp."""
from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import h5py
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapefile
import xarray as xr
from rasterio.crs import CRS
from rasterio.features import is_valid_geom, rasterize, shapes
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult

MAX_CELLS = 2_000_000
GEOMETRY = "__geometry__"
FORMATS = ("CSV", "TSV", "Excel", "JSON", "Parquet", "NetCDF", "HDF5-NetCDF", "GeoTIFF", "GeoJSON", "Shapefile ZIP")


def dumps(value) -> str:
    def default(item):
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (pd.Timestamp, np.datetime64)):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"无法保存元数据类型：{type(item).__name__}")
    return json.dumps(value, ensure_ascii=False, default=default, allow_nan=False)


def check_geometry(geometry: dict, *, two_dimensions: bool = False) -> None:
    if not geometry or not is_valid_geom(geometry):
        raise SchemaError("存在空或无效 geometry")
    if geometry.get("type") == "GeometryCollection":
        for item in geometry["geometries"]:
            check_geometry(item, two_dimensions=two_dimensions)
        return
    def coordinates(item):
        if item and isinstance(item[0], (int, float)):
            if not np.isfinite(item).all() or (two_dimensions and len(item) != 2):
                raise SchemaError("当前空间操作要求有限的二维坐标；不能静默丢弃 Z/M")
        else:
            for child in item:
                coordinates(child)
    coordinates(geometry["coordinates"])


def table_dataset(frame: pd.DataFrame, *, crs: str = "") -> xr.Dataset:
    if frame.empty or not frame.columns.is_unique:
        raise SchemaError("表格必须非空且列名不能重复")
    axis = "__row__"
    while axis in frame:
        axis += "_"
    dataset = xr.Dataset.from_dataframe(frame.reset_index(drop=True).rename_axis(axis))
    dataset.attrs.update(structure="vector" if GEOMETRY in frame else "table", row_dimension=axis)
    if crs:
        dataset.attrs["crs"] = CRS.from_user_input(crs).to_wkt()
    return dataset


def table_frame(dataset: xr.Dataset) -> pd.DataFrame:
    if int(np.prod(list(dataset.sizes.values()))) > MAX_CELLS:
        raise SchemaError("展开表格超过 200 万行，请先选择时间或垂直层切片")
    frame = dataset.to_dataframe().reset_index()
    axis = dataset.attrs.get("row_dimension")
    return frame.drop(columns=[axis], errors="ignore") if axis else frame


def describe_hdf(content: bytes) -> list[dict]:
    result = []
    with h5py.File(io.BytesIO(content), "r") as handle:
        def visit(name, item):
            if isinstance(item, h5py.Dataset):
                result.append({"path": name, "shape": list(item.shape), "dtype": str(item.dtype)})
        handle.visititems(visit)
    return result


def read_structure(name: str, content: bytes, *, hdf_value: str = "", hdf_x: str = "", hdf_y: str = "", crs: str = "", record_read_source: bool = True) -> xr.Dataset:
    suffix = Path(name).suffix.lower()
    if suffix == '.rdb':
        from ecodata.io.raw_readers import USGSRDBReader
        dataset = USGSRDBReader().read_bytes(content).dataset
    elif suffix in {'.gz', '.isd'}:
        from ecodata.io.raw_readers import NOAAISDReader
        dataset = NOAAISDReader().read_bytes(content).dataset
    elif suffix == '.txt':
        from ecodata.io.raw_tables import read_monitoring_text, attach_table_evidence
        layout=json.loads(Path(__file__).with_name('monitoring_semantics.json').read_text(encoding='utf8'))['fixed_layout']
        frame,evidence=read_monitoring_text(content,layout)
        dataset=attach_table_evidence(table_dataset(frame),evidence)
    elif suffix=='.csv' and content.lstrip(b'\xef\xbb\xbf').startswith(b'Latitude:') and b'Daymet' in content[:2000]:
        from ecodata.io.daymet import read_daymet_precipitation
        dataset=read_daymet_precipitation(content)
    elif suffix in {".csv", ".tsv"}:
        # CSV carries no type schema. Preserve identifiers including leading zeros;
        # numeric conversion is explicit when a numeric operation is requested.
        frame = pd.read_csv(io.BytesIO(content), sep="\t" if suffix == ".tsv" else ",", dtype=str, keep_default_na=False)
        dataset = table_dataset(frame)
        if b'LEGEND FOR FIELD NAMES' in content[:200]:
            from ecodata.io.raw_tables import read_monitoring_text, attach_table_evidence
            frame,evidence=read_monitoring_text(content)
            dataset=attach_table_evidence(table_dataset(frame),evidence)
    elif suffix == ".xlsx":
        book = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
        from ecodata.io.raw_tables import select_excel_sheet
        frame, evidence = select_excel_sheet(book)
        dataset = table_dataset(frame)
        dataset.attrs['reader_evidence']=json.dumps(evidence,ensure_ascii=False)
        if "_ecodata_metadata" in book.sheet_names:
            metadata = pd.read_excel(book, sheet_name="_ecodata_metadata")
            saved = json.loads("".join(metadata["json"].astype(str)))
            dataset.attrs.update({k: v for k, v in saved["attributes"].items() if k not in {"structure", "row_dimension"}})
            for key, attrs in saved["variables"].items():
                if key in dataset:
                    dataset[key].attrs.update(attrs)
    elif suffix == ".parquet":
        frame = pd.read_parquet(io.BytesIO(content))
        dataset = table_dataset(frame)
        saved = frame.attrs.get("ecodata_metadata", {})
        dataset.attrs.update({k: v for k, v in saved.get("attributes", {}).items() if k not in {"structure", "row_dimension"}})
        for key, attrs in saved.get("variables", {}).items():
            if key in dataset:
                dataset[key].attrs.update(attrs)
    elif suffix in {".json", ".geojson"}:
        payload = json.loads(content.decode("utf-8-sig"))
        if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
            if payload.get("crs"):
                raise SchemaError("带旧式 crs 的 GeoJSON 请先转换成 RFC 7946 WGS84；不能猜测坐标系")
            topology = gpd.GeoDataFrame.from_features(payload.get("features", []), crs="EPSG:4326")
            if not topology.empty and (topology.geometry.is_empty.any() or not topology.geometry.is_valid.all()):
                raise SchemaError("GeoJSON 包含空或拓扑无效 geometry")
            rows = []
            for feature in payload.get("features", []):
                row = dict(feature.get("properties") or {})
                if GEOMETRY in row:
                    raise SchemaError(f"属性名 {GEOMETRY} 为保留字段")
                geometry = feature.get("geometry")
                check_geometry(geometry)
                row[GEOMETRY] = dumps(geometry)
                if "id" in feature:
                    if "__feature_id__" in row:
                        raise SchemaError("属性与 feature id 保留字段冲突")
                    row["__feature_id__"] = feature["id"]
                rows.append(row)
            dataset = table_dataset(pd.DataFrame(rows), crs="EPSG:4326")
        else:
            if not isinstance(payload, list):
                raise SchemaError("表格 JSON 必须为 records 数组，或使用 GeoJSON FeatureCollection")
            dataset = table_dataset(pd.DataFrame(payload))
    elif suffix in {".tif", ".tiff"}:
        with MemoryFile(content) as memory, memory.open() as source:
            if source.transform.b != 0 or source.transform.d != 0:
                raise SchemaError("旋转/倾斜 GeoTIFF 需要先重投影到规则网格，不能直接当作规则经纬度")
            if source.crs is None:
                raise SchemaError("GeoTIFF 缺少 CRS，无法可靠定位")
            if source.width * source.height * source.count > MAX_CELLS:
                raise SchemaError("超过本地转换的 200 万数值上限，请先裁剪或分块")
            raw = source.read(masked=True)
            if raw.dtype.kind in "iu" and (np.any(raw > 2**53) or np.any(raw < -(2**53))):
                raise SchemaError("GeoTIFF 整数超过 float64 可精确表示范围，当前入口不能无损读取")
            values = raw.astype(float).filled(np.nan)
            x = source.transform.c + (np.arange(source.width) + .5) * source.transform.a
            y = source.transform.f + (np.arange(source.height) + .5) * source.transform.e
            saved = json.loads(source.tags().get("ecodata_metadata", "{}"))
            variable = saved.get("export_variable", "value")
            dataset = xr.Dataset({variable: (("band", "y", "x"), values)}, coords={"band": np.arange(1, source.count + 1), "y": y, "x": x})
            if source.count == 1:
                dataset = dataset.isel(band=0, drop=True)
            dataset.attrs.update(saved.get("attributes", {}))
            dataset.attrs.update(structure="grid", crs=source.crs.to_wkt(), spatial_x="x", spatial_y="y", raster_tags=dumps(source.tags()))
            pixel_width, pixel_height = abs(source.transform.a), abs(source.transform.e)
            dataset.attrs.update(raster_transform=dumps(tuple(source.transform)[:6]), pixel_width=pixel_width, pixel_height=pixel_height)
            if not np.isfinite([pixel_width, pixel_height]).all() or min(pixel_width, pixel_height) <= 0:
                dataset.attrs["spatial_resolution_reason"] = "GeoTIFF transform 中的像元宽高无效，无法计算分辨率"
            elif not source.crs.is_projected:
                dataset.attrs["spatial_resolution_reason"] = "CRS 使用经纬度等非投影坐标，像元大小不是固定的米数，不能直接换算为固定公里分辨率"
            else:
                try:
                    unit_name, unit_factor = source.crs.linear_units_factor
                    width_km, height_km = pixel_width * unit_factor / 1000, pixel_height * unit_factor / 1000
                    if not np.isfinite([width_km, height_km]).all() or min(width_km, height_km) <= 0:
                        raise ValueError("无效的坐标单位换算系数")
                    dataset.attrs.update(pixel_coordinate_unit=unit_name, pixel_width_km=width_km, pixel_height_km=height_km)
                    for axis, standard in (("x", "projection_x_coordinate"), ("y", "projection_y_coordinate")):
                        dataset[axis].attrs.update(units=unit_name, standard_name=standard)
                    if np.isclose(width_km, height_km, rtol=0.001):
                        dataset.attrs["grid_resolution_km"] = (width_km + height_km) / 2
                        dataset.attrs["spatial_resolution_method"] = "transform 像元宽高按 CRS 线性单位换算；宽高相差不超过0.1%时取均值"
                    else:
                        dataset.attrs["spatial_resolution_reason"] = "像元宽高不同，已识别双向分辨率；当前空间处理使用单一分辨率，需要用户确认"
                except (ValueError, rasterio.errors.CRSError) as exc:
                    dataset.attrs["spatial_resolution_reason"] = f"CRS 的线性坐标单位无法换算为公里：{exc}"
            dataset[variable].attrs.update(saved.get("variables", {}).get(variable, {}))
            band_units = []
            for band in source.indexes:
                tags = {key.lower(): value for key, value in {**source.tags(), **source.tags(band)}.items()}
                band_units.append(source.units[band - 1] or tags.get("units") or tags.get("unit") or tags.get("unittype") or "")
            if all(band_units) and len(set(band_units)) == 1:
                if not dataset[variable].attrs.get("units"):
                    dataset[variable].attrs["units"] = band_units[0]
            dataset.attrs["original_band_descriptions"] = dumps(source.descriptions)
            for key, item in saved.get("scalar_coordinates", {}).items():
                if key not in dataset:
                    dataset = dataset.assign_coords({key: np.asarray(item["value"], dtype=item["dtype"])})
                    dataset[key].attrs.update(saved.get("variables", {}).get(key, {}))
    elif suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.namelist()
            shp = [item for item in members if item.lower().endswith(".shp")]
            if len(shp) != 1:
                raise SchemaError("Shapefile ZIP 必须恰好包含一组 .shp/.shx/.dbf/.prj")
            stem = shp[0][:-4]
            lookup = {item.lower(): item for item in members}
            def part(ext):
                key = (stem + ext).lower()
                if key not in lookup:
                    raise SchemaError(f"Shapefile 缺少 {ext}")
                return archive.read(lookup[key])
            encoding = part(".cpg").decode().strip() if (stem + ".cpg").lower() in lookup else "utf-8"
            if encoding == "65001":
                encoding = "utf-8"
            with shapefile.Reader(shp=io.BytesIO(part(".shp")), shx=io.BytesIO(part(".shx")), dbf=io.BytesIO(part(".dbf")), encoding=encoding) as reader:
                if reader.shapeType not in {1, 3, 5, 8}:
                    raise SchemaError("当前 Shapefile 仅支持二维几何；Z/M 需要专门的维度映射")
                rows = []
                for item in reader.iterShapeRecords():
                    row = item.record.as_dict()
                    if GEOMETRY in row:
                        raise SchemaError("Shapefile 属性名与几何保留字段冲突")
                    row[GEOMETRY] = dumps(item.shape.__geo_interface__)
                    rows.append(row)
            dataset = table_dataset(pd.DataFrame(rows), crs=CRS.from_wkt(part(".prj").decode()).to_wkt())
            if "ecodata_metadata.json" in members:
                saved = json.loads(archive.read("ecodata_metadata.json"))
                dataset.attrs.update({k: v for k, v in saved.get("attributes", {}).items() if k not in {"crs", "structure", "row_dimension"}})
                for key, attrs in saved.get("variables", {}).items():
                    if key in dataset:
                        dataset[key].attrs.update(attrs)
    elif suffix in {".nc", ".nc4", ".netcdf", ".h5", ".hdf5", ".hdf"}:
        if hdf_value:
            if not hdf_x or not hdf_y or not crs:
                raise SchemaError("原生 HDF5 必须明确指定数值 dataset、x/y dataset 和 CRS")
            with h5py.File(io.BytesIO(content), "r") as source:
                field, x_data, y_data = source[hdf_value], source[hdf_x], source[hdf_y]
                if field.ndim != 2 or x_data.ndim != 1 or y_data.ndim != 1 or field.shape != (y_data.size, x_data.size):
                    raise SchemaError("原生 HDF5 当前要求二维数值数组及长度匹配的一维 x/y 坐标")
                if field.size > MAX_CELLS or not np.issubdtype(field.dtype, np.number):
                    raise SchemaError("HDF5 数值 dataset 必须为数值类型且不超过 200 万单元")
                dataset = xr.Dataset({"value": (("y", "x"), field[()])}, coords={"x": x_data[()], "y": y_data[()]})
                dataset["value"].attrs.update(json.loads(dumps(dict(field.attrs))))
                for axis, obj in (("x", x_data), ("y", y_data)):
                    dataset[axis].attrs.update(json.loads(dumps(dict(obj.attrs))))
                dataset.attrs.update(structure="grid", crs=CRS.from_user_input(crs).to_wkt(), spatial_x="x", spatial_y="y", hdf_dataset=hdf_value, hdf_attributes=dumps(dict(source.attrs)))
                dataset = xr.decode_cf(dataset)
        else:
            from ecodata.interface.workflow import load_netcdf
            dataset = load_netcdf(content)
            if not dataset.attrs.get("crs"):
                for variable in dataset.data_vars.values():
                    mapping = variable.attrs.get("grid_mapping")
                    if mapping in dataset:
                        wkt = dataset[mapping].attrs.get("crs_wkt") or dataset[mapping].attrs.get("spatial_ref")
                        if wkt:
                            dataset.attrs["crs"] = CRS.from_wkt(wkt).to_wkt()
                            break
    else:
        raise SchemaError(f"不支持的输入格式：{suffix}")
    if record_read_source:
        dataset.attrs["source_file"] = name
        dataset.attrs["source_sha256"] = hashlib.sha256(content).hexdigest()
    return dataset


def metadata(dataset: xr.Dataset) -> dict:
    scalar = {
        key: {"dtype": str(value.dtype), "value": str(value.values) if value.dtype.kind in "Mm" else value.item()}
        for key, value in dataset.coords.items() if value.ndim == 0
    }
    return {"attributes": dict(dataset.attrs), "variables": {key: dict(value.attrs) for key, value in dataset.variables.items()}, "dimensions": dict(dataset.sizes), "scalar_coordinates": scalar}


def spatial_axes(dataset: xr.Dataset) -> tuple[str, str]:
    x = dataset.attrs.get("spatial_x") or ("lon" if "lon" in dataset.coords else "x")
    y = dataset.attrs.get("spatial_y") or ("lat" if "lat" in dataset.coords else "y")
    if x not in dataset.coords or y not in dataset.coords:
        raise SchemaError("需要明确的 x/y 或 lon/lat 坐标；表格请先选择“表格 → 规则网格”")
    return x, y


def projected_coordinate_names(dataset: xr.Dataset) -> xr.Dataset:
    """Correct projected axis labels without changing coordinate values."""
    if not dataset.attrs.get("crs") or not CRS.from_user_input(dataset.attrs["crs"]).is_projected:
        return dataset
    rename = {old: new for old, new in (("lon", "x"), ("lat", "y")) if old in dataset.dims}
    if not rename:
        return dataset
    if any(new in dataset.variables for new in rename.values()):
        raise SchemaError("投影坐标名称冲突：同时存在 x/y 和 lat/lon，请确认坐标字段")
    result = dataset.rename(rename)
    result.attrs = dict(dataset.attrs)
    for key in ("spatial_x", "spatial_y"):
        result.attrs[key] = rename.get(result.attrs.get(key), result.attrs.get(key)) or ("x" if key == "spatial_x" else "y")
    for axis, standard in (("x", "projection_x_coordinate"), ("y", "projection_y_coordinate")):
        if axis in result.coords:
            result[axis].attrs = dict(result[axis].attrs, standard_name=standard)
    return result


def validate_geotiff(content: bytes) -> dict:
    """Reopen the completed file, rejecting unusable raster outputs."""
    try:
        with MemoryFile(content) as memory, memory.open() as source:
            if not source.width or not source.height or not source.count:
                raise SchemaError("GeoTIFF 宽高或波段数量为零")
            if source.crs is None:
                raise SchemaError("GeoTIFF 缺少 CRS")
            transform = source.transform
            if not np.isfinite(tuple(transform)).all() or transform.a == 0 or transform.e == 0:
                raise SchemaError("GeoTIFF transform 无效")
            values = source.read(masked=True)
            valid = (~np.ma.getmaskarray(values)) & np.isfinite(values.data)
            count = int(valid.sum())
            if not count:
                raise SchemaError("GeoTIFF 没有有效像元：数据全部为 NaN 或 NoData")
            if np.any(~np.isfinite(values.data) & ~np.ma.getmaskarray(values)):
                raise SchemaError("GeoTIFF 非有限像元未正确标记为 NoData")
            return {"width": source.width, "height": source.height, "bands": source.count,
                    "CRS": source.crs.to_string(), "transform": list(transform)[:6],
                    "valid_pixels": count, "nodata": "NaN" if source.nodata is not None and np.isnan(source.nodata) else source.nodata,
                    "dtype": source.dtypes[0], "success": True}
    except SchemaError:
        raise
    except Exception as exc:
        raise SchemaError(f"GeoTIFF 回读检查失败：{exc}") from exc


def raster_plane(dataset: xr.Dataset, variable: str):
    x, y = spatial_axes(dataset)
    field = dataset[variable]
    if set(field.dims) != {y, x}:
        raise SchemaError("栅格输出需要二维切片，请明确选择时间、波段或垂直层")
    field = field.transpose(y, x).sortby(x).sortby(y, ascending=False)
    for axis in (x, y):
        coords = field[axis].values.astype(float)
        steps = np.diff(coords)
        if len(coords) < 2 or not np.isfinite(coords).all() or not np.allclose(steps, steps[0]) or steps[0] == 0:
            raise SchemaError("栅格坐标必须一维、等间隔、无重复且每个方向至少 2 个中心点")
    if not dataset.attrs.get("crs"):
        raise SchemaError("缺少 CRS，请明确填写坐标系后再导出空间格式")
    dx = float(field[x][1] - field[x][0])
    dy = float(field[y][0] - field[y][1])
    transform = rasterio.Affine(dx, 0, float(field[x][0]) - dx/2, 0, -dy, float(field[y][0]) + dy/2)
    if field.dtype.kind not in "biuf":
        raise SchemaError("栅格字段必须为数值类型")
    if field.dtype.kind in "iu" and (np.any(field.values > 2**53) or np.any(field.values < -(2**53))):
        raise SchemaError("整数超过 float64 可精确表示范围，请保留 NetCDF 整数类型")
    values = field.values.astype(float)
    if np.isinf(values).any():
        raise SchemaError("栅格包含无穷值，请先明确处理规则")
    if not np.isfinite(values).any():
        raise SchemaError("无法输出 GeoTIFF：所选二维数据全部为 NaN/缺失值，没有有效像元")
    return values, transform, CRS.from_user_input(dataset.attrs["crs"])


def vector_raster_grid(extent, resolution: float):
    """Cover the vector extent at an exact native-CRS pixel size."""
    if extent is None or not np.isfinite(resolution) or resolution <= 0:
        raise SchemaError("无法确定矢量范围，或目标分辨率不是有效正数")
    left, bottom, right, top = extent
    if not np.isfinite(extent).all() or left > right or bottom > top:
        raise SchemaError("矢量空间范围无效")
    width = max(2, int(np.ceil((right-left)/resolution)))
    height = max(2, int(np.ceil((top-bottom)/resolution)))
    if width * height > MAX_CELLS:
        raise SchemaError(f"目标分辨率将生成 {width}×{height} 像元，超过 200 万上限；请增大分辨率数值")
    if right == left:
        left -= resolution
    if top == bottom:
        bottom -= resolution
    return (left, bottom, left+width*resolution, bottom+height*resolution), width, height


@dataclass(frozen=True)
class StructureRequest:
    operation: str = "preserve"
    variable: str = "value"
    x: str = "longitude"
    y: str = "latitude"
    time: str = ""
    crs: str = ""
    unit: str = ""
    slices: tuple[tuple[str, int], ...] = ()
    bounds: tuple[float, float, float, float] | None = None
    width: int = 100
    height: int = 100
    overlap: str = "error"
    all_touched: bool = False
    raster_mask: bool = False


class StructureConverter:
    name = "structure.convert"

    def __init__(self, request: StructureRequest):
        self.request = request

    def validate_input(self, bundle: EcoDataBundle) -> None:
        if self.request.operation not in {"preserve", "table_to_grid", "table_points", "rasterize", "grid_points", "polygonize"}:
            raise SchemaError("未知结构转换")
        if self.request.operation != "preserve" and self.request.variable not in bundle.dataset and not (self.request.operation == "rasterize" and self.request.raster_mask):
            raise SchemaError("请选择有效的数值/分类字段")

    def apply(self, bundle: EcoDataBundle) -> TransformResult:
        self.validate_input(bundle)
        started = perf_counter()
        r = self.request
        ds = bundle.dataset.copy(deep=True)
        if r.crs:
            if ds.attrs.get("crs") and CRS.from_user_input(ds.attrs["crs"]) != CRS.from_user_input(r.crs):
                raise SchemaError("输入已有 CRS 与填写值不同；不能通过更改标签代替重投影")
            ds.attrs["crs"] = CRS.from_user_input(r.crs).to_wkt()
        if r.unit and not (r.operation == "rasterize" and r.raster_mask):
            if ds[r.variable].attrs.get("units") not in {None, "", r.unit}:
                raise SchemaError("单位不同需要数值换算，格式转换不会只改单位标签")
            ds[r.variable].attrs["units"] = r.unit
        if r.slices:
            ds = ds.isel(dict(r.slices))
        notes = []
        if r.operation == "table_to_grid":
            frame = table_frame(ds)
            if not {r.x, r.y, r.variable}.issubset(frame):
                raise SchemaError("请正确选择表格坐标列和数值列")
            if GEOMETRY in frame:
                raise SchemaError("矢量请使用栅格化，不把几何对象当普通网格字段")
            for column in (r.x, r.y, r.variable):
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[[r.x, r.y]].to_numpy()).all():
                raise SchemaError("坐标列不能缺失或包含非有限数值")
            dims = [r.y, r.x]
            if r.time:
                frame[r.time] = pd.to_datetime(frame[r.time], errors="raise")
                if frame[r.time].isna().any() or frame[r.time].dt.tz is not None:
                    raise SchemaError("时间不能为空；带时区数据请先明确统一到 UTC 无时区表示")
                dims.insert(0, r.time)
            if len(set(dims)) != len(dims) or r.variable in dims:
                raise SchemaError("x、y、time 与数值列必须是不同字段")
            if frame.duplicated(dims).any():
                raise SchemaError("相同时间/坐标存在重复记录，请先确定聚合规则，不能静默覆盖")
            count = int(np.prod([frame[c].nunique() for c in dims]))
            if count > MAX_CELLS:
                raise SchemaError("展开网格超过 200 万单元，请缩小范围或先分组")
            original = ds
            ds = frame.set_index(dims).to_xarray()
            ds.attrs.update(original.attrs)
            ds.attrs.update(structure="grid", spatial_x=r.x, spatial_y=r.y)
            ds.attrs.pop("row_dimension", None)
            for key in ds.variables:
                if key in original:
                    ds[key].attrs.update(original[key].attrs)
            if count > len(frame):
                notes.append(f"缺少 {count-len(frame)} 个网格位置，保留为缺失值，未插值填充。")
        elif r.operation == "table_points":
            frame = table_frame(ds)
            if not ds.attrs.get("crs") or r.x == r.y or not {r.x, r.y}.issubset(frame):
                raise SchemaError("生成点矢量需确认不同的 x/y 列及 CRS")
            x = pd.to_numeric(frame[r.x], errors="raise")
            y = pd.to_numeric(frame[r.y], errors="raise")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise SchemaError("点坐标不能为缺失或非有限值")
            frame[GEOMETRY] = [dumps({"type": "Point", "coordinates": [float(a), float(b)]}) for a, b in zip(x, y)]
            original = ds
            ds = table_dataset(frame, crs=original.attrs["crs"])
            ds.attrs.update({k: v for k, v in original.attrs.items() if k not in {"structure", "row_dimension"}})
            for key in original.data_vars:
                ds[key].attrs.update(original[key].attrs)
        elif r.operation == "rasterize":
            if GEOMETRY not in ds:
                raise SchemaError("栅格化需要 GeoJSON 或 Shapefile 矢量")
            if not r.bounds or len(r.bounds) != 4 or not np.isfinite(r.bounds).all():
                raise SchemaError("请明确填写栅格边界 xmin,ymin,xmax,ymax")
            left, bottom, right, top = r.bounds
            if left >= right or bottom >= top or r.width < 2 or r.height < 2 or r.width*r.height > MAX_CELLS:
                raise SchemaError("边界顺序或行列数无效；网格应至少 2×2 且不超过 200 万单元")
            if not ds.attrs.get("crs"):
                raise SchemaError("栅格化缺少 CRS")
            if r.overlap not in {"error", "first", "last"}:
                raise SchemaError("不支持的重叠规则")
            frame = table_frame(ds)
            values = np.ones(len(frame)) if r.raster_mask else pd.to_numeric(frame[r.variable], errors="raise").to_numpy()
            if not np.isfinite(values).all():
                raise SchemaError("栅格化属性必须为有限数值；文字类别请先建立明确的数值编码")
            if values.dtype.kind in "iu" and (np.any(values > 2**53) or np.any(values < -(2**53))):
                raise SchemaError("栅格化整数属性超过 float64 可精确表示范围")
            features = [(json.loads(g), float(v)) for g, v in zip(frame[GEOMETRY], values)]
            for geometry, _ in features:
                check_geometry(geometry, two_dimensions=True)
            transform = from_bounds(*r.bounds, r.width, r.height)
            if r.overlap == "error":
                counts = rasterize([(g, 1) for g, _ in features], out_shape=(r.height, r.width), transform=transform, all_touched=r.all_touched, merge_alg=rasterio.enums.MergeAlg.add, dtype="int32", skip_invalid=False)
                if np.any(counts > 1):
                    raise SchemaError("多个要素覆盖同一像元，请明确选择先出现或后出现的要素优先")
            if r.overlap == "first":
                features.reverse()
            values = rasterize(features, out_shape=(r.height, r.width), transform=transform, fill=np.nan, dtype="float64", all_touched=r.all_touched, skip_invalid=False)
            ds = xr.Dataset({r.variable: (("y", "x"), values)}, coords={"x": left+(np.arange(r.width)+.5)*(right-left)/r.width, "y": top-(np.arange(r.height)+.5)*(top-bottom)/r.height}, attrs=dict(ds.attrs))
            if r.raster_mask:
                ds[r.variable].attrs.update(units="1", long_name="Vector coverage mask", data_kind="categorical")
                notes.append("自动生成覆盖掩膜：要素覆盖处 value=1，未覆盖处为 NoData；未改写原始属性。")
            else:
                ds[r.variable].attrs.update(bundle.dataset[r.variable].attrs)
            if r.unit:
                ds[r.variable].attrs["units"] = r.unit
            ds.attrs.update(structure="grid", spatial_x="x", spatial_y="y")
            ds.attrs.pop("row_dimension", None)
            notes.append("按指定范围栅格化；范围外要素被裁剪，未覆盖像元为 nodata；其他属性不作为栅格值。")
        elif r.operation == "grid_points":
            x, y = spatial_axes(ds)
            if not ds.attrs.get("crs"):
                raise SchemaError("生成点矢量需要明确 CRS")
            frame = table_frame(ds)
            if len(frame) > MAX_CELLS:
                raise SchemaError("点数量超过本地转换上限")
            frame[GEOMETRY] = [dumps({"type": "Point", "coordinates": [float(a), float(b)]}) for a, b in zip(frame[x], frame[y])]
            original = ds
            ds = table_dataset(frame, crs=original.attrs["crs"])
            ds.attrs.update({k: v for k, v in original.attrs.items() if k not in {"structure", "row_dimension"}})
            for key in original.variables:
                if key in ds:
                    ds[key].attrs.update(original[key].attrs)
        elif r.operation == "polygonize":
            values, transform, crs = raster_plane(ds, r.variable)
            valid = np.isfinite(values)
            if not valid.any() or not np.all(values[valid] == np.round(values[valid])):
                raise SchemaError("区域多边形化仅支持整数分类栅格；连续量请使用像元中心点")
            if np.any(np.abs(values[valid]) > np.iinfo(np.int32).max):
                raise SchemaError("分类值超出 int32 范围")
            rows = [{r.variable: int(v), GEOMETRY: dumps(g)} for g, v in shapes(np.where(valid, values, 0).astype("int32"), mask=valid, transform=transform)]
            original = ds
            ds = table_dataset(pd.DataFrame(rows), crs=crs.to_wkt())
            ds.attrs.update({k: v for k, v in original.attrs.items() if k not in {"structure", "row_dimension"}})
            ds[r.variable].attrs.update(original[r.variable].attrs)
            notes.append("仅所选分类字段被多边形化；nodata 不生成面。")
        parameters = {**asdict(r), "input_file": bundle.dataset.attrs.get("source_file"), "input_sha256": bundle.dataset.attrs.get("source_sha256")}
        record = TransformRecord(self.name, parameters, int(sum(v.size for v in bundle.dataset.data_vars.values())), int(sum(v.size for v in ds.data_vars.values())), perf_counter()-started, tuple(notes))
        return TransformResult(EcoDataBundle(ds, bundle.history + (record,)), record, tuple(notes))


def export_structure(dataset: xr.Dataset, output_format: str) -> tuple[bytes, str, str]:
    dataset = projected_coordinate_names(dataset)
    saved = metadata(dataset)
    if output_format in {"NetCDF", "HDF5-NetCDF"}:
        # h5netcdf keeps int64 and Unicode fields; never truncate to NetCDF3 types.
        clean = dataset.copy(deep=True)
        for variable in clean.variables.values():
            if variable.dtype.kind == "O" and pd.isna(variable.values).any():
                raise SchemaError("含缺失值的混合/文字字段无法无损写入当前 NetCDF 路径，请使用 Parquet 或先明确缺失值编码")
        for attrs in [clean.attrs, *(v.attrs for v in clean.variables.values())]:
            for key, value in list(attrs.items()):
                if value is None or isinstance(value, (dict, tuple, list, bool)):
                    attrs[key] = dumps(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.nc"
            clean.to_netcdf(path, engine="h5netcdf")
            content = path.read_bytes()
        return content, "ecodata_result.nc" if output_format == "NetCDF" else "ecodata_result.h5", "application/x-netcdf"
    if output_format == "GeoTIFF":
        variable = dataset.attrs.get("export_variable", "value")
        values, transform, crs = raster_plane(dataset, variable)
        saved["export_variable"] = variable
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", height=values.shape[0], width=values.shape[1], count=1, dtype="float64", crs=crs, transform=transform, nodata=np.nan) as target:
                target.write(values, 1)
                target.set_band_description(1, variable)
                target.update_tags(ecodata_metadata=dumps(saved))
            content = memory.read()
        check = validate_geotiff(content)
        if check["valid_pixels"] != int(np.isfinite(values).sum()):
            raise SchemaError("GeoTIFF NoData 检查失败：有效像元数量与输入不一致")
        return content, "ecodata_result.tif", "image/tiff"
    frame = table_frame(dataset)
    if output_format in {"GeoJSON", "Shapefile ZIP"}:
        if GEOMETRY not in frame:
            raise SchemaError("矢量输出需要 geometry；请先选择“网格 → 像元中心点”或“分类栅格 → 区域多边形”")
        crs = CRS.from_user_input(dataset.attrs.get("crs", ""))
        features = []
        for row in frame.to_dict(orient="records"):
            row = {key: _json_property(value) for key, value in row.items()}
            geometry = json.loads(row.pop(GEOMETRY))
            if output_format == "GeoJSON" and crs != CRS.from_epsg(4326):
                geometry = transform_geom(crs, "EPSG:4326", geometry)
            feature = {"type": "Feature", "geometry": geometry, "properties": row}
            if "__feature_id__" in row:
                if output_format == "Shapefile ZIP":
                    raise SchemaError("GeoJSON feature id 需明确映射为短属性名后再导出 Shapefile，或继续使用 GeoJSON")
                feature["id"] = row.pop("__feature_id__")
            features.append(feature)
        if output_format == "GeoJSON":
            return dumps({"type": "FeatureCollection", "features": features}).encode(), "ecodata_result.geojson", "application/geo+json"
        return _shapefile_bytes(features, crs, saved), "ecodata_result.zip", "application/zip"
    if output_format in {"CSV", "TSV"}:
        sep, extension = (",", "csv") if output_format == "CSV" else ("\t", "tsv")
        return frame.to_csv(index=False, sep=sep).encode("utf-8-sig"), f"ecodata_result.{extension}", "text/plain"
    if output_format == "JSON":
        rows = [{key: _json_property(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]
        return dumps(rows).encode(), "ecodata_result.json", "application/json"
    buffer = io.BytesIO()
    if output_format == "Excel":
        if len(frame) > 1_048_575:
            raise SchemaError("超出 Excel 工作表行数上限")
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="data", index=False)
            text = dumps(saved)
            pd.DataFrame({"json": [text[i:i+30000] for i in range(0, len(text), 30000)]}).to_excel(writer, sheet_name="_ecodata_metadata", index=False)
        return buffer.getvalue(), "ecodata_result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if output_format == "Parquet":
        frame.attrs["ecodata_metadata"] = saved
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue(), "ecodata_result.parquet", "application/vnd.apache.parquet"
    raise SchemaError(f"不支持输出格式：{output_format}")


def _json_property(value):
    if isinstance(value, dict):
        return {key: _json_property(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_property(item) for item in value]
    return None if pd.isna(value) else value


def _shapefile_bytes(features: list[dict], crs: CRS, saved: dict) -> bytes:
    if not features:
        raise SchemaError("Shapefile 不能导出空要素集合")
    geometry_types = {f["geometry"]["type"] for f in features}
    if len(geometry_types) != 1 or not geometry_types.issubset({"Point", "LineString", "Polygon"}):
        raise SchemaError("Shapefile 当前要求同一种二维 Point、LineString 或 Polygon；其他几何请用 GeoJSON")
    for feature in features:
        check_geometry(feature["geometry"], two_dimensions=True)
    columns = list(features[0]["properties"])
    if not columns or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,9}", key) for key in columns):
        raise SchemaError("Shapefile 属性名需为不超过 10 字符的英文标识且至少一个字段；请明确改名或使用 GeoJSON，不能静默截断")
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with shapefile.Writer(shp=shp, shx=shx, dbf=dbf, encoding="utf-8") as writer:
        for key in columns:
            # Reject values that DBF would silently truncate or round.
            values = [f["properties"].get(key) for f in features]
            if any(v is None or isinstance(v, (list, dict)) for v in values):
                raise SchemaError("Shapefile 当前不无损支持 null 或嵌套属性，请使用 GeoJSON")
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                if all(isinstance(v, int) for v in values):
                    if any(len(str(v)) > 18 for v in values):
                        raise SchemaError("Shapefile 整数过长，请使用 GeoJSON")
                    writer.field(key, "N", size=20, decimal=0)
                else:
                    if any(float(f"{v:.10f}") != v or len(f"{v:.10f}") > 24 for v in values):
                        raise SchemaError("Shapefile 数值精度无法无损保存，请使用 GeoJSON")
                    writer.field(key, "N", size=24, decimal=10)
            elif all(isinstance(v, bool) for v in values):
                writer.field(key, "L")
            else:
                if not all(isinstance(v, str) for v in values) or max(len(v.encode()) for v in values) > 254:
                    raise SchemaError("Shapefile 字符字段类型混合或超过 254 字节，请使用 GeoJSON")
                writer.field(key, "C", size=max(1, max(len(v.encode()) for v in values)))
        for feature in features:
            writer.shape(feature["geometry"])
            writer.record(*[feature["properties"][key] for key in columns])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for ext, content in ((".shp", shp.getvalue()), (".shx", shx.getvalue()), (".dbf", dbf.getvalue()), (".prj", crs.to_wkt().encode()), (".cpg", b"UTF-8")):
            target.writestr("result" + ext, content)
        target.writestr("ecodata_metadata.json", dumps(saved))
    return archive.getvalue()
