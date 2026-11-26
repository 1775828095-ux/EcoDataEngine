from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult


def normalize_grid_axes(dataset: xr.Dataset, axes=("lat", "lon")):
    """Reverse descending axes together with all dependent arrays."""
    reversed_axes = []
    for axis in axes:
        if axis not in dataset.dims:
            continue
        values = np.asarray(dataset[axis].values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise SchemaError(f"空间坐标 {axis} 必须是一维有限数值")
        differences = np.diff(values)
        if np.all(differences > 0):
            continue
        if np.all(differences < 0):
            dataset = dataset.isel({axis: slice(None, None, -1)})
            reversed_axes.append(axis)
        else:
            raise SchemaError(f"空间坐标 {axis} 存在重复或非单调排列，无法自动翻转")
    return dataset, reversed_axes


@dataclass(frozen=True)
class RegridConfig:
    method: str
    target_lat: tuple[float, ...]
    target_lon: tuple[float, ...]
    variable: str = "value"


class GridRegridder:
    name = "spatial.regrid"

    def __init__(self, config: RegridConfig):
        self.config = config

    def validate_input(self, data: EcoDataBundle) -> None:
        dataset = data.dataset
        if self.config.method not in {"nearest", "bilinear", "conservative"}:
            raise SchemaError(f"Unsupported regridding method: {self.config.method}")
        if self.config.variable not in dataset:
            raise SchemaError(f"Variable '{self.config.variable}' is missing")
        if not {"lat", "lon"}.issubset(dataset[self.config.variable].dims):
            raise SchemaError("Spatial regridding requires rectilinear lat and lon dimensions")
        if len(self.config.target_lat) < 2 or len(self.config.target_lon) < 2:
            raise SchemaError("Target grid requires at least two coordinates per axis")
        if not (_strictly_increasing(dataset["lat"].values) and _strictly_increasing(dataset["lon"].values)):
            raise SchemaError("Source lat/lon coordinates must be strictly increasing")
        if not (_strictly_increasing(self.config.target_lat) and _strictly_increasing(self.config.target_lon)):
            raise SchemaError("Target lat/lon coordinates must be strictly increasing")
        kind = dataset[self.config.variable].attrs.get("data_kind", dataset.attrs.get("data_kind"))
        if self.config.method != "nearest" and kind != "continuous":
            raise SemanticError("Bilinear and conservative regridding require continuous data")
        spatial = dataset[self.config.variable].attrs.get(
            "spatial_semantics", dataset.attrs.get("spatial_semantics")
        )
        if self.config.method == "conservative" and spatial not in {"cell_mean", "flux"}:
            raise SemanticError("Conservative regridding requires cell_mean or flux spatial semantics")

    def apply(self, data: EcoDataBundle) -> TransformResult:
        normalized, reversed_axes = normalize_grid_axes(data.dataset)
        data = EcoDataBundle(normalized, data.history)
        self.validate_input(data)
        started = perf_counter()
        variable = self.config.variable
        selected = data.dataset[[variable]]
        target = xr.Dataset(coords={"lat": np.asarray(self.config.target_lat), "lon": np.asarray(self.config.target_lon)})
        from ecodata.engines.grid import GridEngine
        output, backend = GridEngine().regrid(selected, target, self.config.method)
        output.attrs.update(data.dataset.attrs)
        output.attrs.update(
            {"processing_stage": "spatial_regridded", "spatial_regridding_method": self.config.method}
        )
        output[variable].attrs.update(data.dataset[variable].attrs)
        for axis in ("lat", "lon"):
            output[axis].attrs.update(data.dataset[axis].attrs)
        record = TransformRecord(
            method=f"{self.name}.{self.config.method}",
            parameters={"method": self.config.method, "variable": variable, "reversed_axes": reversed_axes, **backend},
            input_rows=int(data.dataset[variable].size),
            output_rows=int(output[variable].size),
            elapsed_seconds=perf_counter() - started,
        )
        bundle = EcoDataBundle(output, data.history + (record,))
        return TransformResult(bundle, record)


def _strictly_increasing(values) -> bool:
    values = np.asarray(values, dtype=float)
    return values.ndim == 1 and np.isfinite(values).all() and np.all(np.diff(values) > 0)


def target_grid_spec(value):
    """Explicit user grid requirement; never an algorithm preference."""
    if not isinstance(value,dict) or set(value)!={'crs','latitude_step','longitude_step','extent'}:
        raise ValueError('目标网格需crs、latitude_step、longitude_step、extent')
    from pyproj import CRS
    if not CRS.from_user_input(value['crs']).equals(CRS.from_epsg(4326)) or value['extent']!='intersection':
        raise ValueError('此入口的目标网格目前支持EPSG:4326及共同覆盖intersection')
    for key in ('latitude_step','longitude_step'):
        if isinstance(value[key],bool) or not isinstance(value[key],(int,float)) or not np.isfinite(value[key]) or value[key]<=0:
            raise ValueError('目标网格间距必须是正数，单位为度')
    return dict(value)


def resolve_target_grid(datasets,spec):
    """Common supported center domain; uses geometry only, never field values."""
    spec=target_grid_spec(spec);coords={}
    for axis,key in (('lat','latitude_step'),('lon','longitude_step')):
        # Keep complete target cells inside the common observed center domain.
        # The lightweight backend does not guarantee outer-half-cell support.
        low=max(float(normalize_grid_axes(ds)[0][axis][0]) for ds in datasets)
        high=min(float(normalize_grid_axes(ds)[0][axis][-1]) for ds in datasets);step=spec[key]
        count=int(np.floor((high-low)/step+1e-9))
        if count<2:raise ValueError('目标网格在共同覆盖内不足两个像元：'+axis)
        coords[axis]=low+step*(np.arange(count)+.5)
    return xr.Dataset(coords=coords,attrs=dict(crs='EPSG:4326',grid_origin='common_center_domain_lower_edge'))


def require_target_grid(dataset,target,variable):
    from pyproj import CRS
    if not CRS.from_user_input(dataset.attrs.get('crs','')).equals(CRS.from_epsg(4326)):
        raise ValueError('候选输出坐标基准不符合目标网格')
    for axis in ('lat','lon'):
        if axis not in dataset.coords or not np.array_equal(dataset[axis].values,target[axis].values):
            raise ValueError('候选输出不符合目标网格硬约束：'+axis)
    if dataset[variable].dims!=('time','lat','lon'):
        raise ValueError('候选输出维度不符合目标网格硬约束')
    return dict(status='PASS',shape=list(dataset[variable].shape),
        latitude_step=float(target.lat[1]-target.lat[0]),longitude_step=float(target.lon[1]-target.lon[0]))


def contained_cell_grid(source,target):
    """Keep whole rectilinear reference cells inside the target cell-edge footprint.

    A center inside the domain is insufficient: the entire observation support
    must be covered. Geometry only; never inspect values when selecting cells.
    """
    from ecodata.algorithms.raster_reconstruction_task import edges
    out=source
    for axis in ('lat','lon'):
        bounds=edges(out[axis]);limit=edges(target[axis])
        keep=(bounds[:-1]>=limit[0]-1e-9)&(bounds[1:]<=limit[-1]+1e-9)
        out=out.isel({axis:np.flatnonzero(keep)})
        if out.sizes[axis]<2:
            raise ValueError('Target grid cannot fully cover enough original reference cells: '+axis)
    return out


def common_pair_grid(datasets, variable, conservative):
    grids = [normalize_grid_axes(ds)[0] for ds in datasets]
    axes = ("lat", "lon")
    if not all(set(axes) <= set(ds[variable].dims) and
               all(ds[a].dims == (a,) and ds[a].size >= 2 for a in axes) for ds in grids):
        raise ValueError("该重采样需要原版支持的规则经纬网格")
    differences = [[np.diff(ds[a].values) for a in axes] for ds in grids]
    if not all(np.allclose(step, step[0]) for pair in differences for step in pair):
        raise ValueError("目标分辨率需要规则网格间距")
    steps = [[step[0] for step in pair] for pair in differences]
    target = next((i for i in (0, 1) if np.all(np.asarray(steps[i]) >= np.asarray(steps[1-i])-1e-10)), None)
    if target is None:
        raise ValueError("两个坐标轴粗细交叉，须指定目标网格")
    base, source = grids[target], grids[1-target]
    for j, axis in enumerate(axes):
        half = steps[target][j]/2 if conservative else 0
        margin = steps[1-target][j]/2 if conservative else 0
        keep = ((base[axis]-half >= float(source[axis][0])-margin-1e-9) &
                (base[axis]+half <= float(source[axis][-1])+margin+1e-9))
        base = base.isel({axis:np.flatnonzero(keep.values)})
        if base[axis].size < 2:
            raise ValueError("共同网格覆盖不足两行两列")
    return base, source, target

