from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext, AlgorithmResult, AlgorithmSpec, Compatibility
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord
from ecodata.spatial.regridding import GridRegridder, RegridConfig


class SemanticCrossScaleStackPlugin:
    """Build a dimensionless feature stack after semantics-aware spatial alignment."""

    spec = AlgorithmSpec(
        id="scale.semantic_stack",
        name="Semantic cross-scale feature stack",
        version="1",
        category="multi_source_feature_preparation",
        status="available",
        input_kinds=("rectilinear_grids",),
        semantic_types=("multidimensional_array",),
        requires_auxiliary=True,
        physical_constraints=("variable_semantics", "unit_preservation", "categorical_fraction", "conservation_when_declared"),
        compute="medium",
        license="project / xarray-regrid backend",
        backend="EcoData semantic routing + xarray-regrid",
        notes="Creates aligned raw variables and normalized model features. A finer output grid does not imply finer physical information.",
    )

    @staticmethod
    def _datasets(context: AlgorithmContext) -> tuple[xr.Dataset, ...]:
        return tuple(source.dataset for source in context.sources) + tuple(context.auxiliaries)

    def compatibility(self, context: AlgorithmContext) -> Compatibility:
        reasons = []
        if context.variable != "feature_stack":
            reasons.append("该插件生成 feature_stack，不生成任意目标物理变量")
        if context.target_lat is None or context.target_lon is None:
            reasons.append("缺少目标 lat/lon 网格")
        if len(self._datasets(context)) < (1 if context.parameters.get("alignment_only") else 2):
            reasons.append("跨尺度特征栈至少需要两个来源")
        crs_values = set()
        usable = 0
        for dataset in self._datasets(context):
            crs = dataset.attrs.get("crs")
            if crs:
                crs_values.add(str(crs))
            for name, field in dataset.data_vars.items():
                if {"lat", "lon"}.issubset(field.dims):
                    usable += 1
                    kind = field.attrs.get("data_kind")
                    if kind not in {"continuous", "categorical"}:
                        reasons.append(f"{name} 缺少受支持的 data_kind（continuous/categorical）")
                    if not field.attrs.get("units"):
                        reasons.append(f"{name} 缺少单位；分类变量应使用 units=1")
        if usable < (1 if context.parameters.get("alignment_only") else 2):
            reasons.append("至少需要两个带 lat/lon 的可用变量")
        if len(crs_values) != 1 or any(not dataset.attrs.get("crs") for dataset in self._datasets(context)):
            reasons.append("所有来源必须明确且使用同一 CRS")
        return Compatibility(not reasons, tuple(dict.fromkeys(reasons)), ("网格加密只改变表达网格，不代表真实信息尺度提高",))

    @staticmethod
    def _is_finer(source: xr.DataArray, target_lat, target_lon) -> bool:
        source_step = abs(float(np.median(np.diff(source.lat.values)))) * abs(float(np.median(np.diff(source.lon.values))))
        target_step = abs(float(np.median(np.diff(target_lat)))) * abs(float(np.median(np.diff(target_lon))))
        return target_step < source_step

    @staticmethod
    def _regrid(field: xr.DataArray, method: str, target_lat, target_lon) -> xr.DataArray:
        dataset = field.to_dataset(name=field.name)
        dataset.attrs.update(crs=field.attrs.get("crs", ""), data_kind=field.attrs.get("data_kind"), spatial_semantics=field.attrs.get("spatial_semantics"))
        bundle = EcoDataBundle(dataset)
        output = GridRegridder(RegridConfig(method, tuple(target_lat), tuple(target_lon), field.name)).apply(bundle).data.dataset[field.name]
        if not np.isfinite(output.values).all():
            raise SchemaError(f"{field.name} 尺度匹配后出现 NaN；源数据必须覆盖完整目标范围")
        return output

    def run(self, context: AlgorithmContext) -> AlgorithmResult:
        compatibility = self.compatibility(context)
        if not compatibility.eligible:
            raise SchemaError("跨尺度特征准备不可执行：" + "；".join(compatibility.reasons))
        started = perf_counter()
        target_lat = np.asarray(context.target_lat, dtype=float)
        target_lon = np.asarray(context.target_lon, dtype=float)
        raw_variables = {}
        feature_arrays = []
        feature_names = []
        transformations = []
        seen_names = set()

        for dataset in self._datasets(context):
            for name, original in dataset.data_vars.items():
                if not {"lat", "lon"}.issubset(original.dims):
                    continue
                if name in seen_names:
                    raise SchemaError(f"跨来源变量名重复：{name}；请在输入阶段明确重命名")
                seen_names.add(name)
                field = original.transpose("lat", "lon").copy()
                field.attrs["crs"] = dataset.attrs["crs"]
                kind = field.attrs["data_kind"]
                finer_target = self._is_finer(field, target_lat, target_lon)

                if kind == "categorical":
                    categories = [value.item() if hasattr(value, "item") else value for value in np.unique(field.values[np.isfinite(field.values)])]
                    categories = [int(value) if isinstance(value, float) and value.is_integer() else value for value in categories]
                    for category in categories:
                        fraction_name = f"{name}__fraction_{category}"
                        indicator = xr.where(field == category, 1.0, 0.0).rename(fraction_name)
                        indicator.attrs.update(units="1", data_kind="continuous", spatial_semantics="cell_mean", source_variable=name, category_value=category)
                        method = "nearest" if finer_target else "conservative"
                        aligned = self._regrid(indicator, method, target_lat, target_lon)
                        raw_variables[fraction_name] = aligned
                        feature_arrays.append(aligned)
                        feature_names.append(fraction_name)
                    transformations.append({"variable": name, "semantic": "categorical", "method": "nearest one-hot" if finer_target else "conservative category fractions", "information_scale_increased": False})
                    continue

                semantics = field.attrs.get("spatial_semantics")
                if finer_target:
                    method = "bilinear"
                elif semantics in {"cell_mean", "flux"}:
                    method = "conservative"
                else:
                    raise SchemaError(f"{name} 向更粗网格聚合前必须声明 spatial_semantics=cell_mean/flux")
                aligned = self._regrid(field, method, target_lat, target_lon)
                raw_variables[name] = aligned
                feature_arrays.append(aligned)
                feature_names.append(name)
                transformations.append({"variable": name, "semantic": "continuous", "method": method, "information_scale_increased": False})

        normalized = []
        normalization = {}
        for name, field in zip(feature_names, feature_arrays):
            values = np.asarray(field.values, dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            scale = std if std > 0 else 1.0
            normalized.append(((field - mean) / scale).expand_dims(feature=[name]))
            normalization[name] = {"mean": mean, "std": scale, "source_units": field.attrs.get("units", "")}
        feature_stack = xr.concat(normalized, dim="feature").astype(np.float32)
        output = xr.Dataset(raw_variables, coords={"lat": target_lat, "lon": target_lon})
        output["feature_stack"] = feature_stack
        output["feature_stack"].attrs.update(units="1", long_name="normalized semantics-aware cross-scale features")
        output.attrs.update(
            crs=self._datasets(context)[0].attrs["crs"],
            processing_stage="semantic_cross_scale_feature_preparation",
            normalization=json.dumps(normalization, ensure_ascii=False),
            resolution_claim="Aligned grid resolution is not a claim of recovered physical detail.",
        )
        elapsed = perf_counter() - started
        record = TransformRecord(self.spec.id, {"transformations": transformations}, sum(field.size for field in feature_arrays), feature_stack.size, elapsed, compatibility.warnings)
        report = {"status": "PASS", "features": feature_names, "transformations": transformations, "runtime_seconds": elapsed, "physical_detail_claim": False}
        return AlgorithmResult(EcoDataBundle(output, (record,)), self.spec.id, report)


def align_variables(datasets, variables, *, target_grid=None, target_frequency=None):
    """Public cross-variable alignment, never averaging unlike quantities.

    Reuses registered scale/temporal kernels. Without a user/model target the
    common coarser grid is an engineering default, not a learned optimal scale.
    """
    from pyproj import CRS
    from ecodata.spatial.regridding import normalize_grid_axes, resolve_target_grid
    from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler
    if len(datasets) != len(variables) or len(datasets) < 2:
        raise SchemaError('Cross-variable alignment requires at least two explicitly bound fields')
    prepared=[]
    for ds,name in zip(datasets,variables):
        if name not in ds:raise SchemaError('Bound variable absent: '+name)
        if not ds.attrs.get('crs') or not CRS.from_user_input(ds.attrs['crs']).equals(CRS.from_epsg(4326)):
            raise SchemaError('Cross-variable rectilinear alignment requires confirmed geographic EPSG:4326; use declared projection adapter first')
        ds=normalize_grid_axes(ds)[0]
        attrs=ds[name].attrs
        if not attrs.get('units') or attrs.get('data_kind') not in {'continuous','categorical'}:
            raise SchemaError('Unit/data kind UNKNOWN: '+name)
        if attrs.get('spatial_semantics') not in {'cell_mean','flux','point','grid_estimate','categorical'}:
            raise SchemaError('Spatial support UNKNOWN or unsupported cell total: '+name)
        if set(ds[name].dims)-{'time','lat','lon'}:
            raise SchemaError('Select a scientifically defined level before cross-variable alignment: '+name)
        if target_frequency and 'time' in ds[name].dims:
            ds=SemanticTimeResampler(SemanticTimeConfig(target_frequency,name)).apply(EcoDataBundle(ds)).data.dataset
        prepared.append(ds)
    dynamic=['time' in ds[name].dims for ds,name in zip(prepared,variables)]
    if any(dynamic) and not all(dynamic):raise SchemaError('Static/dynamic broadcasting requires an explicit task contract')
    if all(dynamic):
        import pandas as pd
        clocks=[pd.DatetimeIndex(ds.time.values) for ds in prepared]
        if any(c.hasnans or c.has_duplicates or not c.is_monotonic_increasing for c in clocks):
            raise SchemaError('Time labels must be nonmissing, unique and increasing')
        times=clocks[0]
        for c in clocks[1:]:times=times.intersection(c)
        if not len(times):raise SchemaError('No common actual timestamps; no invented observations')
        bounds=[]
        for ds,name in zip(prepared,variables):
            semantic=ds[name].attrs.get('temporal_semantics')
            if semantic not in {'instantaneous','mean','accumulation'}:
                raise SchemaError('Cross-variable time support UNKNOWN or unsupported: '+name)
            if semantic!='instantaneous':
                from ecodata.temporal.semantic import _validated_bounds
                _validated_bounds(ds)
                bounds.append(ds.sel(time=times).time_bounds.values)
        if bounds and any(not np.array_equal(bounds[0],b) for b in bounds[1:]):
            raise SchemaError('Equal labels have different interval support; declare a common target frequency')
        prepared=[ds.sel(time=times) for ds in prepared]
    # Reuse the same geometry planner as same-variable pairing, with a private
    # placeholder only for geometry: scientific fields keep their own names/units.
    from ecodata.spatial.regridding import common_pair_grid as grid_plan
    geometry=[ds[[name]].rename({name:'geometry_field'}) for ds,name in zip(prepared,variables)]
    if target_grid:
        target=resolve_target_grid(geometry,target_grid)
        basis='explicit user/model target'
    else:
        if len(geometry)!=2:raise SchemaError('More than two variables require an explicit target grid')
        target,_,_=grid_plan(geometry,'geometry_field',True)
        basis='common supported coarser grid; not accuracy-optimal resolution'
    records=[];outputs=[]
    for ds,name in zip(prepared,variables):
        frames=[];operations=[]
        indices=range(ds.sizes['time']) if 'time' in ds[name].dims else [None]
        for index in indices:
            plane=ds[[name]].isel(time=index,drop=True) if index is not None else ds[[name]]
            identity=all(np.array_equal(plane[a],target[a]) for a in ('lat','lon'))
            if identity:
                result=plane.copy();op=dict(variable=name,method='identity')
            else:
                context=AlgorithmContext(sources=(EcoDataBundle(plane),),variable='feature_stack',
                    semantic_type='multidimensional_array',target_lat=tuple(target.lat.values),target_lon=tuple(target.lon.values),
                    parameters={'alignment_only':True})
                executed=SemanticCrossScaleStackPlugin().run(context)
                columns=[v for v in executed.data.dataset if v==name or v.startswith(name+'__fraction_')]
                result=executed.data.dataset[columns]
                op=executed.report['transformations'][0]
            if index is not None:result=result.expand_dims(time=[ds.time.values[index]])
            frames.append(result);operations.append(op)
        output=xr.concat(frames,dim='time') if all(dynamic) else frames[0]
        if 'time_bounds' in ds:output['time_bounds']=ds.time_bounds
        output.attrs.update(crs=ds.attrs['crs'],source_id=ds.attrs.get('source_id','UNKNOWN'))
        outputs.append(output)
        records.append(dict(variable=name,unit=ds[name].attrs.get('units','UNKNOWN'),operations=operations,
            input_shape=list(ds[name].shape),output_shape=dict(output.sizes)))
    return outputs,dict(status='PASS',target_basis=basis,variables=records,physical_variables_blended=False,
        physical_detail_claim=False,target_shape=dict(target.sizes),reference_accuracy_assessed=False,
        temporal_relation='Common actual labels; each variable retains instantaneous/mean/accumulation semantics, not physical equivalence')
