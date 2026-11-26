"""Prepare selected pairs with the original space, time and fusion engines."""
import hashlib
import io

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.core.schemas import EcoDataBundle
from ecodata.models.alignment_bridge import spatial
from ecodata.models.scientific_schema import compare_units, convert_matched_field
from ecodata.spatial.regridding import GridRegridder, RegridConfig, normalize_grid_axes
from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler

from .scan import read_selected

SPACE = {
    "raster_task.spatial.nearest": "nearest",
    "raster_task.spatial.bilinear": "bilinear",
    "spatial.regrid.conservative": "conservative",
}
TIME = {"temporal.semantic": "linear", "temporal.semantic.nearest": "nearest"}
REGISTERED_SPACE = "spatial.native_coordinates"
WEIGHTED = {"fusion.uncertainty_weighted.nearest":"nearest","fusion.uncertainty_weighted.bilinear":"bilinear"}
SPATIAL_SEEDS = ("spatial.blend", "spatial.residual_idw", "spatial.idw")


def fields(scan, task):
    from ecodata.recognition import recognise_file

    payloads = [read_selected(scan["rows"][i], limit=64*1024**2) for i, _ in task["fields"]]
    datasets = [recognise_file(scan["rows"][i]["name"], raw, record_read_source=False)[0]
                for (i, _), raw in zip(task["fields"], payloads)]
    from ecodata.core.recovery import scan_dataset
    datasets=[scan_dataset(scan,index,ds)[0] for (index,_),ds in zip(task['fields'],datasets)]
    return datasets, [hashlib.sha256(raw).hexdigest() for raw in payloads]


def conversion_task(scan, selections, action, unit_check):
    from ecodata.io.structures import FORMATS

    fields=[]
    for index,field in selections:
        fmt=scan["rows"][index]["format"]
        if fmt not in {*FORMATS,"HDF5","RDB","NOAA ISD","Shapefile"}:
            raise ValueError("原版结构转换器未登记此输入格式")
        fields.append(dict(index=index,name=field["name"],header=field.get("original_header",field["name"]),
                           unit=field["unit"],sheet=field.get("sheet")))
    labels=[scan["rows"][item["index"]]["format"] for item in fields]
    grid_formats={"NetCDF","GeoTIFF","HDF5","HDF5-NetCDF"}
    formats=[["format.netcdf"],["format.hdf5_netcdf"]] if any(fmt in grid_formats for fmt in labels) else [
        ["format.parquet"],["format.excel"]]
    if all(fmt in {"GeoJSON","Shapefile","Shapefile ZIP"} for fmt in labels):
        formats.append(["format.geojson"])
    if action=="convert_units":
        formats=[["units.harmonize",*recipe] for recipe in formats]
    return dict(action=action,route="structure_conversion",fields=fields,recipe_steps=formats,
                target_unit=unit_check.get("target_unit"))


def conversion_source(scan,index):
    from pathlib import PurePosixPath
    import zipfile
    from .scan import local_member

    row=scan["rows"][index]
    name=row["name"]
    if row["format"]!="Shapefile":
        with local_member(row) as path:
            return name,path.read_bytes()
    stem=PurePosixPath(name).with_suffix("").as_posix().casefold()
    members={PurePosixPath(item["name"]).suffix.casefold():item for item in scan["rows"]
             if item["ref"][0]==row["ref"][0] or not row["ref"][1] and not item["ref"][1]
             if PurePosixPath(item["name"]).with_suffix("").as_posix().casefold()==stem}
    required={".shp",".shx",".dbf",".prj"}
    if not required<=members.keys():
        raise ValueError("Shapefile 缺少原版读取器要求的同名配套文件")
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w") as archive:
        for suffix in sorted(required | (members.keys() & {".cpg"})):
            with local_member(members[suffix]) as path:
                archive.write(path,"source"+suffix)
    return "source.zip",buffer.getvalue()


def run_conversion(scan,task_id,task,folder,model_id,progress):
    from ecodata.io.structures import export_structure, read_structure
    from .table_unit_segments import convert
    files,artifacts,reports=[],[],[]
    output_formats=[]
    for index,recipe in enumerate(task["recipes"],1):
        if recipe not in task["recipe_steps"]:
            raise ValueError("转换方案未登记")
        targets={"format.netcdf":"NetCDF","format.hdf5_netcdf":"HDF5-NetCDF",
                 "format.parquet":"Parquet","format.excel":"Excel","format.geojson":"GeoJSON"}
        target=targets[recipe[-1]]
        output_formats.append(target)
        for position,source in enumerate(task["fields"],1):
            row=scan["rows"][source["index"]]
            input_name,content=conversion_source(scan,source["index"])
            with read_structure(input_name,content,record_read_source=False, excel_sheet=source.get("sheet")) as dataset:
                field=source["name"] if source["name"] in dataset else source["header"]
                if field not in dataset:
                    raise ValueError("原版读取器无法绑定扫描字段")
                segments = []
                if "units.harmonize" in recipe:
                    dataset, segments = convert(dataset, field, source, task["target_unit"])
                raw,exported,_=export_structure(dataset,target)
                name=f"conversion-{index}-source-{position}"+exported[exported.rfind("."):]
                with read_structure(name,raw,record_read_source=False) as restored:
                    if field not in restored or restored[field].shape!=dataset[field].shape:
                        raise ValueError("输出回读未保留所选字段和维度")
                    if "units.harmonize" in recipe:
                        np.testing.assert_allclose(dataset[field].values,restored[field].values,equal_nan=True)
                files.append((name,raw))
                artifacts.append(dict(name=name,dimensions=dict(dataset.sizes),fields=list(dataset.data_vars)))
                reports.append(dict(source_format=row["format"],target_format=target,field=field,
                                    target_unit=dataset[field].attrs.get("units"),methods=recipe,unit_segments=segments))
    return dict(files=files,source_formats=[scan["rows"][item["index"]]["format"] for item in task["fields"]],
                target_format=" / ".join(dict.fromkeys(output_formats)),original_report=reports,
                local=dict(artifacts=artifacts,reason="原版单位和格式转换完成，选中字段已回读核对。"))


def prepare(datasets, task):
    out = []
    variable = task["quantity"]
    metadata=task.get("field_metadata",[{} for _ in datasets])
    for dataset,(_,name),field in zip(datasets,task["fields"],metadata):
        header=field.get("original_header")
        if name not in dataset and header in dataset:
            dataset=dataset.rename({header:name})
        data=dataset[name].copy(deep=False)
        if data.dtype.kind not in "iuf":
            raw=data.values.astype(str)
            parsed=np.asarray(pd.to_numeric(raw.reshape(-1),errors="coerce")).reshape(raw.shape)
            if not np.isfinite(parsed).any() or np.any((np.char.strip(raw)!="") & ~np.isfinite(parsed)):
                raise ValueError("所选字段包含不能明确转换的数值")
            data=data.copy(data=parsed)
        if not data.attrs.get("units") and field.get("unit"):
            data.attrs=dict(data.attrs,units=field["unit"])
        if not data.attrs.get("temporal_semantics") and field.get("temporal_semantics"):
            data.attrs=dict(data.attrs,temporal_semantics=field["temporal_semantics"])
        if name != variable and variable in dataset:
            raise ValueError("标准变量名与现有字段冲突")
        input_unit = data.attrs.get("units")
        if input_unit != task["unit"]:
            if task.get("identity_basis")=="deepseek":
                from ecodata.engines.units import UnitEngine
                data=UnitEngine().convert(data,task["unit"])
            else:
                data = convert_matched_field(data, variable, task["unit"], verified_quantity=variable)
        qc = dataset.get("qc_flag")
        if qc is not None and qc.dims == data.dims:
            if not set(np.unique(qc.values.astype(str))) <= {"VALID","INVALID","MISSING"}:
                raise ValueError("质量标记含义未声明，不能自动用于对齐")
            data = data.where(qc == "VALID")
        for key in ("data_kind","spatial_semantics","temporal_semantics"):
            if key in dataset.attrs and key not in data.attrs:
                data.attrs[key] = dataset.attrs[key]
        data.attrs.setdefault("data_kind", "continuous" if data.dtype.kind == "f" else "unknown")
        names = [item for item in ("time_bounds","uncertainty","qc_flag")
                 if item in dataset and item!=name]
        if "station_id" in dataset and data.ndim==1 and "time" not in data.dims:
            names.extend(key for key in dataset.data_vars if key.casefold() in
                         {"station_id","time","time_utc","datetime","timestamp","date","longitude","latitude","lon","lat"}
                         and key!=name)
        bounds = dataset.time.attrs.get("bounds") if "time" in dataset else None
        if bounds and bounds in dataset and bounds != "time_bounds":
            if "time_bounds" in dataset:
                raise ValueError("时间区间边界定义冲突")
            dataset = dataset.rename({bounds:"time_bounds"})
            names.append("time_bounds")
        ds = dataset[[name, *names]].assign({name:data})
        if "uncertainty" in names and task["action"]=="fusion":
            uncertainty = ds.uncertainty
            report = compare_units(variable,task["unit"],variable,uncertainty.attrs.get("units") or input_unit)
            if report["status"] in {"equivalent","convertible"}:
                ds["uncertainty"] = (uncertainty*abs(report["scale"])).assign_attrs(units=task["unit"])
            else:
                ds = ds.drop_vars("uncertainty")
        if name != variable:
            ds = ds.rename({name:variable})
        ds[variable].attrs.update(physical_variable=variable)
        ds[variable].attrs.setdefault("physical_definition", variable)
        out.append(ds)
    return out


def grid_coordinates(datasets):
    from pyproj import CRS

    if any(not ds.attrs.get("crs") for ds in datasets):
        raise ValueError("重采样需要明确 CRS")
    target_crs = CRS.from_user_input(datasets[0].attrs["crs"])
    grids = []
    for ds in datasets:
        renamed = ds.rename({"x":"lon","y":"lat"}) if {"x","y"} <= set(ds.dims) else ds
        grids.append(renamed.assign_attrs(spatial_x="lon",spatial_y="lat"))
    if not all({"lat","lon"} <= set(ds.dims) for ds in grids):
        raise ValueError("栅格重投影需要可绑定到原版规则轴的坐标")
    restore = {"lon":"x","lat":"y"} if target_crs.is_projected else {}
    return grids, restore


def same_crs(datasets):
    from pyproj import CRS
    left,right = [CRS.from_user_input(ds.attrs["crs"]) for ds in datasets]
    return left.equals(right)


def grid_plan(datasets, variable, conservative):
    from ecodata.spatial.regridding import common_pair_grid
    return common_pair_grid(datasets,variable,conservative)


def spatial_methods(datasets, variable):
    supports = {ds[variable].attrs.get("spatial_semantics") for ds in datasets}
    kinds = {ds[variable].attrs.get("data_kind") for ds in datasets}
    if supports <= {"cell_mean", "flux"} and kinds == {"continuous"}:
        methods = ["spatial.regrid.conservative"]
    elif supports <= {"grid_estimate", "point"}:
        methods = ["raster_task.spatial.nearest"]
        if kinds == {"continuous"}:
            methods.append("raster_task.spatial.bilinear")
    elif supports == {"categorical"} or kinds == {"categorical"} and None not in supports:
        methods = ["raster_task.spatial.nearest"]
    else:
        raise ValueError("空间支持未声明，或单元均值/通量与点估计不宜按同一种规则重采样")
    if same_crs(datasets):
        grid_plan(datasets, variable, methods[0]=="spatial.regrid.conservative")
    else:
        if methods[0]=="spatial.regrid.conservative":
            raise ValueError("跨投影单元均值或通量需原版守恒适配及明确目标网格")
        from ecodata.io.structures import raster_plane
        for ds in datasets:
            if set(ds[variable].dims)-{"time","lat","lon"}:
                raise ValueError("原版跨投影重采样只支持平面或时间序列栅格")
            ds = ds.isel(time=0,drop=True) if "time" in ds[variable].dims else ds
            raster_plane(ds,variable)
    return methods


def clock(ds):
    if "time" not in ds.coords or not np.issubdtype(ds.time.dtype, np.datetime64):
        raise ValueError("缺少可比较的日期时间轴")
    time = pd.DatetimeIndex(ds.time.values).as_unit("ns")
    if time.hasnans or not time.is_monotonic_increasing or time.has_duplicates:
        raise ValueError("时间轴必须递增且不重复")
    return time


def time_plan(datasets, variable):
    times = [clock(ds) for ds in datasets]
    steps = [np.diff(axis.asi8) for axis in times]
    if not all(len(step) and (step == step[0]).all() for step in steps):
        raise ValueError("自动选择时间分辨率需要规则采样轴")
    if max(t[0] for t in times) > min(t[-1] for t in times):
        raise ValueError("没有共同时间覆盖")
    frequency = pd.tseries.frequencies.to_offset(pd.Timedelta(max(s[0] for s in steps), unit="ns")).freqstr
    semantics = {ds[variable].attrs.get("temporal_semantics") for ds in datasets}
    if semantics == {"instantaneous"}:
        methods = list(TIME)
    elif len(semantics) == 1 and semantics <= {"mean", "accumulation"}:
        methods = ["temporal.semantic"]
        starts = [pd.Timestamp(ds.time_bounds.values[0,0]) for ds in datasets if "time_bounds" in ds]
        step = max(s[0] for s in steps)
        if len(starts) != 2 or (starts[0]-starts[1]).value % step:
            raise ValueError("均值或累计数据须有一致的目标统计区间边界")
        counts = [len(t) for t in times]
        if any(count * count > 4_000_000 for count in counts):
            raise ValueError("原版有界重采样需要较短的时间子集")
    else:
        raise ValueError("两源须有相同且已声明的时间统计语义")
    for ds in datasets:
        SemanticTimeResampler(SemanticTimeConfig(frequency, variable)).validate_input(EcoDataBundle(ds))
    return frequency, methods


def methods(datasets, task):
    variable = task["quantity"]
    action = task["action"]
    datasets = prepare(datasets, task)
    if action == "align_time":
        if all(ds[variable].ndim==1 and "time" not in ds[variable].dims for ds in datasets):
            return table_time_plan(datasets,variable)[1]
        return time_plan(datasets, variable)[1]
    if action == "align_space":
        grids, _ = grid_coordinates(datasets)
        return spatial_methods(grids, variable)
    if all("uncertainty" in ds for ds in datasets):
        from ecodata.fusion.alignment import fusion_compatibility
        weighted_check = fusion_compatibility(datasets,variable)
        if weighted_check["ready"]:
            task["weighted_methods"] = list(WEIGHTED)
            if all("time" not in ds[variable].dims for ds in datasets):
                return []
    grids = [spatial(ds, variable)[0] for ds in datasets]
    sem = {ds[variable].attrs.get("temporal_semantics") for ds in grids}
    if sem != {"instantaneous"}:
        raise ValueError("通用空间试跑需要已确认的瞬时口径；日累计降水和合成卫星走各自适配")
    if any(ds[variable].attrs.get("data_kind") != "continuous" for ds in grids):
        raise ValueError("原版多源变异评分需要连续物理量")
    if any("time" not in ds[variable].dims for ds in grids):
        raise ValueError("原版变异评分需要已确认的时间轴，静态场可在提供有效 uncertainty 后使用反方差融合")
    if not common_times(grids).size:
        frequency, time_methods = time_plan(grids, variable)
        if max(clock(ds)[0] for ds in grids).ceil(frequency) > min(clock(ds)[-1] for ds in grids):
            raise ValueError("共同范围内没有目标时间标签")
        task["time_methods"] = time_methods
    rectilinear = all(ds.lat.dims == ("lat",) and ds.lon.dims == ("lon",) for ds in grids)
    if rectilinear:
        return spatial_methods(grids, variable)
    if any(ds[variable].attrs.get("spatial_semantics") not in {"grid_estimate", "point"} for ds in grids):
        raise ValueError("曲线网格试跑需要确认点估计的空间支持")
    from ecodata.algorithms.spatial_fusion_task import positions
    xy = [positions(ds)[0] for ds in grids]
    if np.any(np.maximum(xy[0].min(axis=0),xy[1].min(axis=0)) >
              np.minimum(xy[0].max(axis=0),xy[1].max(axis=0))):
        raise ValueError("投影转换后两份资料空间范围不重叠")
    return [REGISTERED_SPACE]


def table_clock(values):
    return pd.to_datetime(values.astype(str).str.replace("_","T",n=1),format="mixed",
                          errors="raise",utc=True).dt.tz_localize(None)


def table_time_plan(datasets, variable):
    from ecodata.io.structures import table_frame

    time_fields, intervals = [], []
    for ds in datasets:
        frame = table_frame(ds)
        axes = [name for name in frame if name.casefold() in {"time","time_utc","datetime","timestamp","date"}]
        if len(axes)!=1 or "station_id" not in frame or ds[variable].attrs.get("temporal_semantics")!="instantaneous":
            raise ValueError("站点表需要明确 station_id、唯一时间列及瞬时观测口径")
        frame[axes[0]] = table_clock(frame[axes[0]])
        if frame.duplicated(["station_id",axes[0]]).any():
            raise ValueError("站点表存在重复时刻")
        steps = [np.diff(pd.DatetimeIndex(group[axes[0]]).as_unit("ns").sort_values().asi8)
                 for _, group in frame.groupby("station_id")]
        intervals.extend(step[0] for step in steps if len(step) and (step==step[0]).all())
        if not steps or any(not len(step) or (step!=step[0]).any() for step in steps):
            raise ValueError("站点表未确认规则采样间隔")
        time_fields.append(axes[0])
    frequency = pd.tseries.frequencies.to_offset(pd.Timedelta(max(intervals),unit="ns")).freqstr
    return (frequency,time_fields),["temporal.wide_table_interpolation"]


def common_times(datasets):
    return clock(datasets[0]).intersection(clock(datasets[1]))


def apply_space(datasets, variable, method):
    method = SPACE[method]
    if not same_crs(datasets):
        from ecodata.fusion.alignment import reproject_fields
        base, source = [normalize_grid_axes(ds)[0] for ds in datasets]
        aligned, report = reproject_fields(source,base,(variable,),method)
        if not np.isfinite(aligned[variable]).any():
            raise ValueError("投影转换后没有共同有效像元")
        if "time_bounds" in source:
            aligned["time_bounds"] = source.time_bounds
        return [base,aligned],dict(target_source=1,transform=report,target_policy="first_source_crs")
    base, source, target = grid_plan(datasets, variable, method=="conservative")
    cfg = RegridConfig(method, tuple(base.lat.values), tuple(base.lon.values), variable)
    converted = GridRegridder(cfg).apply(EcoDataBundle(source))
    aligned = converted.data.dataset
    if not np.isfinite(aligned[variable]).any():
        raise ValueError("对齐后的共同范围没有有效像元")
    if "time_bounds" in source:
        aligned["time_bounds"] = source.time_bounds
    out = [base, base]
    out[1-target] = aligned
    return out, dict(target_source=target+1, transform=converted.record.as_dict())


def apply_time(datasets, variable, method):
    frequency, _ = time_plan(datasets, variable)
    transformed = [SemanticTimeResampler(SemanticTimeConfig(frequency, variable, TIME[method])).apply(EcoDataBundle(ds))
                   for ds in datasets]
    out = [item.data.dataset for item in transformed]
    times = common_times(out)
    if not len(times):
        raise ValueError("重采样后没有共同时间标签")
    return [ds.sel(time=times) for ds in out], dict(frequency=frequency,
        interpolation=TIME[method], transforms=[item.record.as_dict() for item in transformed])


def artifact(ds, name, variable):
    raw = bytes(ds.to_netcdf(engine="h5netcdf"))
    with xr.open_dataset(io.BytesIO(raw)) as opened:
        xr.testing.assert_allclose(ds[variable], opened[variable])
        for key in ("uncertainty","time_bounds","time_qc"):
            if key in ds:
                xr.testing.assert_allclose(ds[key],opened[key])
    return (name,raw), dict(name=name,dimensions=dict(ds.sizes),fields=list(ds.data_vars),
                            units={key:val.attrs["units"] for key,val in ds.data_vars.items() if "units" in val.attrs})


def run(scan, task_id, task, folder, model_id, progress):
    from ecodata.algorithms.platform_search import SearchInput
    from .local_search import pareto_comparison, submit
    from .result_summary import ranked_candidates
    from .service import fusion_evidence

    datasets, hashes = fields(scan, task)
    variable = task["quantity"]
    datasets = prepare(datasets, task)
    if task["action"] == "fusion" and hashes[0]==hashes[1]:
        raise ValueError("同一内容不能作为两个独立来源")
    raw_prepared = datasets
    restore_axes = {}
    has_spatial_search = any(any(step in SPATIAL_SEEDS for step in recipe) for recipe in task["recipes"])
    if task["action"] == "fusion" and has_spatial_search:
        converted = [spatial(ds, variable) for ds in datasets]
        datasets = [ds for ds, _ in converted]
        coordinate_reports = [ops for _, ops in converted]
    else:
        coordinate_reports = []
        if task["action"] == "align_space":
            datasets, restore_axes = grid_coordinates(datasets)
    files, artifacts, reports, searches = [], [], [], {}
    seeds_by_method = {}
    for recipe in task["recipes"]:
        alignments = tuple(step for step in recipe if step not in (*SPATIAL_SEEDS,"units.harmonize","format.normalize"))
        seeds_by_method.setdefault(alignments, [])
        for seed in recipe:
            if seed in SPATIAL_SEEDS and seed not in seeds_by_method[alignments]:
                seeds_by_method[alignments].append(seed)
    for index,(alignments,seeds) in enumerate(seeds_by_method.items(),1):
        method = alignments[-1]
        process = dict(method=method, name=alignments, coordinate_adaptation=coordinate_reports)
        working = datasets
        if len(alignments)>1:
            working,time_report = apply_time(working,variable,alignments[0])
            process["time_alignment"] = time_report
        if method in WEIGHTED:
            from ecodata.fusion.alignment import align_and_fuse
            fused,report = align_and_fuse(raw_prepared,variable,method=WEIGHTED[method])
            output = [fused.dataset]
            process["transform"] = report
            from ecodata.evaluation.blocked_confirmation import engineering_review
            process['recommendation_review']=engineering_review(output[0],variable)
        elif method in SPACE:
            output, report = apply_space(working, variable, method)
            process.update(report)
        elif method in TIME:
            output, report = apply_time(datasets, variable, method)
            process.update(report)
        elif method == "temporal.wide_table_interpolation":
            from ecodata.interface.workflow import TrialRequest,run_wide_table_time_trial
            (frequency,time_fields),_ = table_time_plan(datasets,variable)
            trials = []
            for ds,time in zip(datasets,time_fields):
                values = table_clock(pd.Series(ds[time].values)).to_numpy()
                ds = ds.assign({time:(ds[time].dims,values)})
                trial = run_wide_table_time_trial(ds,TrialRequest(variable=variable,target_time_frequency=frequency),
                                                 station_field="station_id",time_field=time)
                trials.append(trial)
            output = [trial.output for trial in trials]
            report = dict(frequency=frequency,transforms=[trial.history for trial in trials])
            process.update(report)
        elif method == REGISTERED_SPACE and task["action"] == "fusion":
            output = working
        else:
            raise ValueError("未登记的对齐算法")
        if task["action"] == "fusion" and method not in WEIGHTED:
            times = common_times(output)
            if not len(times):
                raise ValueError("两源没有可比较的共同时间")
            inputs = []
            from ecodata.algorithms.spatial_fusion_task import positions
            # Use the coarser grid as primary when the two sources were mapped to it.
            primary = process.get("target_source",1)-1
            for k in (primary, 1-primary):
                role = "primary" if k == primary else "background"
                data = output[k].sel(time=times).assign_attrs(source_id=str(output[k].attrs.get("source_id") or hashes[k]))
                dimensions = positions(data)[1]
                data[variable] = data[variable].transpose("time", *dimensions)
                file,_ = artifact(data,f"source-{k+1}.nc",variable)
                inputs.append(SearchInput(file[0],role,file[1]))
            request = dict(task="multisource",variable=variable,physical_variable=variable,
                unit=task["unit"],semantic_type="continuous",temporal_semantics="instantaneous",
                processing_goal="spatial_fusion",initial_algorithms=seeds,
                observation_crs="EPSG:4326",mutation_control="random")
            search_folder = folder/f"method-{index}"
            search = submit(inputs,request,search_folder,model_id=model_id,progress=progress)
            if search.get("status") != "PASS" or not search.get("product"):
                raise ValueError("原版变异寻优未生成合格产物")
            rankings = ranked_candidates(search_folder,search)
            searches[f"{task_id} · {index}"] = dict(best=search["best"],
                selected_graph=search["selected_graph"],evidence=fusion_evidence(search),
                confirmed_initial=search["confirmed_initial"],rankings=rankings,
                pareto_comparison=pareto_comparison(rankings),alignment_method=method,alignment_steps=alignments,
                portrait_strategy=search.get("portrait_strategy"),soft_guidance=search["soft_guidance"])
            with xr.open_dataset(search["product"]) as product:
                output = [product.load()]
        for n,ds in enumerate(output,1):
            if restore_axes:
                ds = ds.rename(restore_axes).assign_attrs(spatial_x="x",spatial_y="y")
            file, info = artifact(ds,f"method-{index}-product-{n}.nc",variable)
            files.append(file)
            artifacts.append(info)
        reports.append(dict(task=task_id, operation=task["action"], **process))
    return dict(files=files, local=dict(artifacts=artifacts, native_ready=False,
        reason="按每种确认的对齐算法生成结果；融合试跑使用原版变异评分。无验证参考的纯重采样保留各方案。"),
        grid_searches=searches, original_report=reports, search=None, target_format="NetCDF",
        source_formats=[scan["rows"][i]["format"] for i,_ in task["fields"]])


def run_cross_variable(scan,task_id,task,folder,model_id,progress):
    from pathlib import Path
    from ecodata.algorithms.platform_search import SearchInput,process_cross_variable
    if task.get('recipes') != [['scale.semantic_stack']]:
        raise ValueError('Cross-variable alignment requires the verified semantic adapter')
    items=[SearchInput(scan['rows'][i]['name'],'primary' if n==0 else 'auxiliary',read_selected(scan['rows'][i]))
           for n,(i,_) in enumerate(task['fields'])]
    report=process_cross_variable(items,dict(task='cross_variable_alignment',variables=[v for _,v in task['fields']],
        target_grid=task.get('target_grid'),target_frequency=task.get('target_frequency')),folder)
    files=[];artifacts=[]
    for product in report['products']:
        path=Path(product['path']);raw=path.read_bytes()
        with xr.open_dataset(io.BytesIO(raw)) as ds:
            artifacts.append(dict(name=path.name,fields=list(ds.data_vars),dimensions=dict(ds.sizes)))
        files.append((path.name,raw))
    return dict(files=files,source_formats=[scan['rows'][i]['format'] for i,_ in task['fields']],
        target_format='NetCDF',original_report=report,
        local=dict(artifacts=artifacts,reason='不同物理变量分别对齐，保留单位；没有直接平均或声称精度提升。'))
