"""Collocate approved station observations with CF or native CAMx NetCDF grids."""
from contextlib import ExitStack
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

from .scan import local_member

UNIT_STEP="units.harmonize"
FORMAT_STEP="format.normalize"
EXACT="station.time.exact"
NEAREST="station.time.nearest"
MATCH="station.grid.nearest"
BIAS="station.bias.holdout"
NAMES={
    EXACT:"按真实时间戳精确配对", NEAREST:"限制半步长的最近时刻匹配",
    MATCH:"使用原生网格坐标匹配最近格点", BIAS:"站点偏差订正与时间留出评分",
}


def eligible(grid,field,table):
    names={value["name"].casefold() for value in table["fields"]}
    return (grid["format"]=="NetCDF" and len(field.get("dimensions",[]))>=3
            and names & {"station","station_id","site_no","site_id"}
            and names & {"lon","longitude"} and names & {"lat","latitude"}
            and names & {"time","time_utc","datetime","date","timestamp"})


def task_for(scan,pair,action,units):
    if units["status"] not in {"equivalent","convertible"}:
        raise ValueError("需先核对站点与格点的单位")
    left,right=pair[:2]
    grid,station=left,right
    rows=scan["rows"]
    if not eligible(rows[grid[0]],grid[2],rows[station[0]]):
        grid,station=right,left
    if not eligible(rows[grid[0]],grid[2],rows[station[0]]):
        return None
    # A vector component is not speed, even when the dimensional units match.
    from ecodata.models.scientific_schema import identity
    quantity,_=identity(grid[2]['name'],grid[2])
    description=str(grid[2].get('long_name','')).casefold()
    observation_quantity,_=identity(station[2]['name'],station[2])
    if ((units['quantity']=='wind_speed' or observation_quantity=='wind_speed' or station[2].get('standard_name')=='wind_speed') and (quantity in {'eastward_wind','northward_wind'}
            or 'wind component' in description or 'component of wind' in description
            or grid[2]['name'] in {'u10','v10','U10','V10'})):
        raise ValueError('风分量不能作为标量风速直接融合；跳过本项，保留原背景场及其他合法任务。需要已登记的分量合成及方向契约。')
    return dict(action=action,route="station_grid",grid_index=grid[0],table_index=station[0],
                grid_name=grid[2]["name"],table_name=station[2]["name"],
                table_header=station[2].get("original_header",station[2]["name"]),unit=grid[2]["unit"],
                obs_unit=station[2]["unit"],quantity=units["quantity"])

def xyz(lon,lat):
    lon,lat=np.deg2rad(lon),np.deg2rad(lat)
    return np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))


def coordinates(grid,var):
    from netCDF4 import num2date
    if "TFLAG" in grid.variables and var.dimensions[-2:]==("ROW","COL") and getattr(grid,"CPROJ",None)==2:
        from ecodata.models.native_meteorology import geometry, native_times
        lon,lat=geometry(grid,allow_halo=True)
        offsets=((var.shape[-2]-lat.shape[0])//2,(var.shape[-1]-lon.shape[1])//2)
        if any(var.shape[-2+i]!=lon.shape[i]+2*offsets[i] or offsets[i] not in (0,1) for i in (0,1)):
            raise ValueError("CAMx 输出维度与已核验网格或边界圈不一致")
        return np.asarray(lon),np.asarray(lat),native_times(grid),var.dimensions[-2:],offsets
    axes={}
    for name in ("latitude","longitude","time"):
        aliases={"latitude":{"lat","latitude"},"longitude":{"lon","longitude"},"time":{"time"}}[name]
        value=next((v for k,v in grid.variables.items() if (k.casefold() in aliases or getattr(v,"standard_name","")==name)
                    and set(v.dimensions)<=set(var.dimensions)),None)
        if value is None:
            raise ValueError("网格缺少已登记的经纬度或时间坐标")
        axes[name]=value
    lon,lat,t=axes["longitude"],axes["latitude"],axes["time"]
    if t.ndim!=1 or var.dimensions[0]!=t.dimensions[0]:
        raise ValueError("网格时间轴无法绑定到所选变量")
    time=pd.DatetimeIndex(num2date(t[:],t.units,getattr(t,"calendar","standard"),
                                 only_use_cftime_datetimes=False,only_use_python_datetimes=True))
    lonvalues=np.ma.asarray(lon[:],dtype=float).filled(np.nan)
    latvalues=np.ma.asarray(lat[:],dtype=float).filled(np.nan)
    if lon.ndim==lat.ndim==1 and lon.dimensions!=lat.dimensions:
        lonvalues,latvalues=np.meshgrid(lonvalues,latvalues)
        axes=lat.dimensions+lon.dimensions
    elif lon.dimensions==lat.dimensions and lat.ndim==2:
        axes=lat.dimensions
    else:
        raise ValueError("所选变量缺少可定位的二维网格坐标")
    return lonvalues,latvalues,time,axes,(0,0)


def matching_tolerance(times):
    clock=pd.DatetimeIndex(times).as_unit('ns')
    if clock.hasnans or clock.has_duplicates or not clock.is_monotonic_increasing:
        raise ValueError('Matching time labels must be unique and increasing')
    return pd.Timedelta(int(np.min(np.diff(clock.asi8))),unit='ns')/2 if len(clock)>1 else pd.Timedelta(0)


def load(scan,task, *, table_data=None, keep_missing=False, minimum_grid_index=0):
    from netCDF4 import Dataset
    from ecodata.io.structures import read_structure, table_frame
    from ecodata.models.scientific_schema import compare_units

    grid_idx,table_idx=task["grid_index"],task["table_index"]
    rows=scan["rows"]
    grid_name,obs_name=task["grid_name"],task["table_name"]
    duplicate_columns = []
    if table_data is None:
        with local_member(rows[table_idx]) as source:
            with read_structure(rows[table_idx]["name"],source.read_bytes(),record_read_source=False) as table:
                data=table_frame(table)
    else:
        from ecodata.io.structures import table_dataset
        from ecodata.io.noaa_global_hourly import add_variables, COMPONENTS, SIGNATURE
        if obs_name in COMPONENTS and obs_name not in table_data:
            source_column = COMPONENTS[obs_name][0]
            # Ignore unrelated raw columns, but never silently choose between
            # duplicate scientific/identity fields. Reuse the production codec.
            selected = table_data.loc[:, table_data.columns.isin(SIGNATURE | {source_column})]
            for name in selected.columns[selected.columns.duplicated()].unique():
                copies = selected.loc[:, selected.columns == name]
                if not all(copies.iloc[:, 0].equals(copies.iloc[:, i]) for i in range(1, copies.shape[1])):
                    raise ValueError("站点关键字段重复且内容冲突，无法安全解码：" + name)
                duplicate_columns.append(name)
            selected = selected.loc[:, ~selected.columns.duplicated()]
            decoded = table_dataset(selected)
            add_variables(decoded)
            data = table_frame(decoded)
        else:
            data = table_data
    names={key.casefold():key for key in data}
    def column(*values):
        return data[next(names[value] for value in values if value in names)]
    raw_time=column("time_utc","date","time","timestamp","datetime").astype(str)
    # NOAA Global Hourly DATE is UTC. Generic local timestamps must carry an
    # offset or have a column name that explicitly declares UTC.
    if not obs_name.startswith(("TMP.","WND.","SLP.","VIS.")) and "time_utc" not in names:
        zoned=raw_time.str.contains(r"(?:Z|[+-][0-9]{2}:?[0-9]{2})$",regex=True)
        raw_time=raw_time.where(zoned)
    observation=pd.DataFrame(dict(station=column("station_id","station","site_no","site_id").astype(str),
        time=pd.to_datetime(raw_time,format="mixed",utc=True,errors="coerce").dt.tz_localize(None),
        longitude=pd.to_numeric(column("longitude","lon"),errors="coerce"),
        latitude=pd.to_numeric(column("latitude","lat"),errors="coerce"),
        observed=pd.to_numeric(data[obs_name if obs_name in data else task["table_header"]],errors="coerce")))
    report=compare_units(task["quantity"],task["unit"],task["quantity"],task["obs_unit"])
    if report["status"] not in {"equivalent","convertible"}:
        raise ValueError("站点单位无法换算到模型单位")
    observation.observed=observation.observed*report["scale"]+report["offset"]
    valid = ["longitude", "latitude"] if keep_missing else ["observed", "longitude", "latitude"]
    observation=observation.dropna(subset=["station", "time", *valid])
    observation=observation[np.isfinite(observation[valid]).all(axis=1)].reset_index(drop=True)
    if observation.empty:
        raise ValueError("站点表缺少有效时刻、坐标或数值")
    with local_member(rows[grid_idx]) as source, Dataset(str(source)) as grid:
        var=grid.variables[grid_name]
        lon,lat,time,dims,offsets=coordinates(grid,var)
        if len(time)!=var.shape[0] or not len(time) or time.has_duplicates or not time.is_monotonic_increasing:
            raise ValueError("网格时间轴缺失、重复或未排序")
        if var.dimensions[-2:]!=dims or any(size!=1 for size in var.shape[1:-2]):
            raise ValueError("多垂直层变量需先定义可和站点比较的层或高度")
        finite=np.isfinite(lon)&np.isfinite(lat)
        flat=np.flatnonzero(finite)
        if flat.size<3:
            raise ValueError("网格缺少有效地理覆盖")
        # Verify that stations fall inside the actual grid extent before matching.
        xy=np.column_stack((lon[finite],lat[finite]))
        if np.ptp(xy[:,0])>180:
            raise ValueError("跨日期变更线的空间配对需先统一经度轴")
        inside=Delaunay(xy).find_simplex(observation[["longitude","latitude"]].to_numpy())>=0
        observation=observation.loc[inside].reset_index(drop=True)
        if observation.empty:
            raise ValueError("站点在网格地理覆盖之外")
        _,cells=cKDTree(xyz(lon[finite],lat[finite])).query(xyz(observation.longitude,observation.latitude))
        yy,xx=np.unravel_index(flat[cells],lon.shape)
        tolerance=matching_tolerance(time)
        out={}
        for method in (EXACT,NEAREST):
            actual=pd.DatetimeIndex(observation.time)
            indices=(time.get_indexer(actual) if method==EXACT else
                     time.get_indexer(actual,method="nearest",tolerance=tolerance))
            indices[(actual<time[0])|(actual>time[-1])]=-1
            keep=np.flatnonzero(indices>=minimum_grid_index)
            frame=observation.iloc[keep].copy()
            frame["grid_time"]=time[indices[keep]]
            frame["grid_lon"]=lon[yy[keep],xx[keep]]
            frame["grid_lat"]=lat[yy[keep],xx[keep]]
            frame['native_t']=indices[keep]
            frame['native_y']=yy[keep]+offsets[0]
            frame['native_x']=xx[keep]+offsets[1]
            center=(0,)*(var.ndim-3)
            values=[var[(int(indices[i]),*center,int(yy[i])+offsets[0],int(xx[i])+offsets[1])] for i in keep]
            frame["background"]=[float(value) if not np.ma.is_masked(value) else np.nan for value in values]
            missing_background=int((~np.isfinite(frame.background)).sum())
            frame=frame[np.isfinite(frame.background)]
            frame=frame.assign(delta_seconds=(frame.time-frame.grid_time).dt.total_seconds())
            frame["distance_seconds"]=frame.delta_seconds.abs()
            minimum=frame.groupby(["station","grid_time"]).distance_seconds.transform("min")
            frame=frame.loc[frame.distance_seconds==minimum]
            # Ties at the same station/model time are aggregated and counted.
            out[method]=frame.groupby(["station","grid_time"],as_index=False).agg(
                time=("time","first"),observed=("observed","mean"),background=("background","first"),
                grid_lon=("grid_lon","first"),grid_lat=("grid_lat","first"),
                native_t=('native_t','first'),native_y=('native_y','first'),native_x=('native_x','first'),
                longitude=("longitude","first"),latitude=("latitude","first"),matched_observations=("observed","size"))
            out[method].attrs['matching_audit']=dict(identical_duplicate_columns=duplicate_columns,
                valid_observations_in_hull=len(observation),
                time_matched_raw_observations=len(keep),missing_background=missing_background,
                unmatched_observations=len(observation)-len(keep),
                aggregation='Closest observation per station/model time; same-distance ties averaged and counted',
                matched_station_times=len(out[method]),tolerance_seconds=float(tolerance.total_seconds()) if method==NEAREST else 0.)
    return out


def plan(rows,action):
    methods=[]
    for method,frame in rows.items():
        # Bias fitting requires distinct earlier training and later test times.
        if action=="fusion":
            times=pd.DatetimeIndex(frame.grid_time.unique()).sort_values()
            if len(times)<9:
                continue
            cut=max(3,len(times)//2)
            if not set(frame.loc[frame.grid_time.isin(times[:cut]),"station"]).intersection(
                    frame.loc[frame.grid_time.isin(times[cut:]),"station"]):
                continue
        elif frame.empty:
            continue
        methods.append([UNIT_STEP,FORMAT_STEP,method,MATCH]+([BIAS] if action=="fusion" else []))
    if not methods:
        raise ValueError("未找到满足时间与空间条件的站点—网格样本")
    return methods


def score(frame,partition):
    subsets=frame[frame.grid_time.isin(partition)]
    observed=subsets.observed.to_numpy(float)
    if not len(observed) or not np.isfinite(observed).all():raise ValueError('No finite fixed reference queue')
    def metrics(values):
        if not np.isfinite(values).all():raise ValueError('Incomplete finite prediction on fixed reference queue')
        residual=values-observed;den=float(np.sum((observed-observed.mean())**2))
        return dict(rmse=float(np.sqrt(np.mean(residual**2))),mae=float(np.mean(np.abs(residual))),
                    bias=float(np.mean(residual)),r2=1-float(np.sum(residual**2))/den if den>0 else None,
                    count=len(residual),coverage=1.,failure_rate=0.)
    return metrics(subsets.background.to_numpy()),metrics(subsets.corrected.to_numpy())


def common_queue(rows):
    keys=['station','grid_time'];common=None;audits={}
    for name,frame in rows.items():
        if frame.duplicated(keys).any():raise ValueError('Duplicate evaluation station/time')
        index=pd.MultiIndex.from_frame(frame[keys]);common=index if common is None else common.intersection(index)
        audits[name]=dict(raw_matched_records=len(frame),matching=frame.attrs.get('matching_audit',{}))
    if common is None or not len(common):raise ValueError('No common actual evaluation queue')
    result={name:frame.set_index(keys).reindex(common).reset_index().sort_values(keys).reset_index(drop=True)
            for name,frame in rows.items()}
    reference=None
    for name,frame in result.items():
        y=frame.observed.to_numpy(float)
        if reference is None:reference=y
        elif not np.array_equal(reference,y):raise ValueError('Different reference labels on common station/time queue')
        audits[name].update(common_records=len(frame),common_fraction=len(frame)/len(rows[name]))
    return result,audits


def apply_frozen_correction(frame,bias,selected):
    """Apply existing fitted residual state; reference labels are not accessed."""
    if selected not in {'background','corrected'}:raise ValueError('Unknown frozen selection')
    frame=frame.copy();mapping=pd.Series(bias,dtype=float)
    known=frame.station.isin(mapping.index)
    frame['corrected']=(frame.background+frame.station.map(mapping)).where(known,frame.background)
    frame['selected']=frame[selected]
    return frame,int((~known).sum())


def evaluate_correction(frame):
    """Existing mean residual, fitted early; choose on middle, report late only."""
    frame=frame.copy();times=pd.DatetimeIndex(frame.grid_time.unique()).sort_values()
    if len(times)<9:raise ValueError('At least 9 actual unique times required for fit/validation/test separation')
    if not np.isfinite(frame[['observed','background']].to_numpy(float)).all():
        raise ValueError('All fixed observed/background values must be finite')
    cut=len(times)//2;end=cut+len(times)//4
    train,validation,test=times[:cut],times[cut:end],times[end:]
    training=frame[frame.grid_time.isin(train)]
    bias=(training.observed-training.background).groupby(training.station).mean()
    frame,fallback=apply_frozen_correction(frame,bias.to_dict(),'background')
    baseline,corrected=score(frame,validation)
    best='corrected' if corrected['rmse']<baseline['rmse'] else 'background'
    # Selection is frozen before any final-period metric is computed.
    frame['selected']=frame[best]
    frame['split']=np.where(frame.grid_time.isin(train),'fit',np.where(frame.grid_time.isin(validation),'validation','test'))
    final_baseline,final_corrected=score(frame,test)
    report=dict(protocol='station_grid_chronological_v2',selected=best,
        selection_baseline=baseline,selection_correction=corrected,
        heldout_baseline=final_baseline,heldout_correction=final_corrected,
        training_times=[str(t) for t in train],validation_times=[str(t) for t in validation],test_times=[str(t) for t in test],
        training_bias={str(k):float(v) for k,v in bias.items()},
        unknown_station_fallback_records=fallback,test_used_for_selection=False,
        effect_confirmation=('FINAL_PERIOD_GAIN_ONLY_NOT_INDEPENDENCE_PROOF' if best=='corrected' and final_corrected['rmse']<final_baseline['rmse'] else 'NOT_CONFIRMED' if best=='corrected' else 'BACKGROUND_RETAINED'),
        scope='Same-station chronological holdout, not spatial generalization; previously exposed exploratory data',
        statistical_independence_verified=False)
    return frame,report


def corrected_frame(frame):
    times = pd.DatetimeIndex(frame.grid_time.unique()).sort_values()
    cut = max(3, len(times)//2)
    train, held = times[:cut], times[cut:]
    training = frame[frame.grid_time.isin(train)].dropna(subset=["observed"])
    bias = (training.observed - training.background).groupby(training.station).mean()
    frame = frame.assign(corrected=frame.background + frame.station.map(bias))
    return frame, held, training.background.to_numpy()


def station_holdout(scan, task, source, field):
    from .holdout_plan import exposures, table_groups, time_boundary
    from netCDF4 import Dataset

    index = task["grid_index"]
    grid = scan["rows"][index]
    grid_field = next(item for item in grid["fields"] if item["name"] == task["grid_name"])
    events = exposures(scan, index, grid_field)
    minimum = 0
    if events:
        with local_member(grid) as path, Dataset(str(path)) as dataset:
            axis = dataset.variables[task["grid_name"]].dimensions[0]
        minimum = time_boundary(events, axis)
    return {"groups": sorted(table_groups(source, exposures(scan, task["table_index"], field))),
            "minimum_grid_index": minimum}


def freeze_holdout(scan, task):
    from . import indexed_table_recipe

    row = scan["rows"][task["table_index"]]
    if row["format"] not in indexed_table_recipe.FORMATS:
        raise ValueError(f"站点留出计划暂不支持 {row['format']} 表格")
    field = task.get("table_field") or next(item for item in row["fields"] if item["name"] == task["table_name"])
    with ExitStack() as stack:
        source = indexed_table_recipe.table(stack, row, field, task["obs_unit"])
        return station_holdout(scan, task, source, field)


def review(scan, task, steps, *, independent=False):
    from . import indexed_table_recipe, recipe_trials

    row = scan["rows"][task["table_index"]]
    if row["format"] not in indexed_table_recipe.FORMATS:
        return dict(status="pending", basis="joint_recipe_adapter_unavailable", count=0, checks=[],
                    reason=f"站点与网格试跑目前无法按组读取 {row['format']} 站点表。")
    field = task.get("table_field") or next(item for item in row["fields"] if item["name"] == task["table_name"])
    method = next((step for step in steps if step in {EXACT, NEAREST}), None)
    if method is None:
        return dict(status="pending", basis="joint_recipe_adapter_unavailable", count=0, checks=[],
                    reason="站点与网格方案缺少已登记的时间配对步骤，无法试跑评分。")
    with ExitStack() as stack:
        source = indexed_table_recipe.table(stack, row, field, task["obs_unit"])
        keys = sorted(source.groups if independent else source.held)
        minimum = 0
        if not independent and (plan := task.get("locked_holdout")) is not None:
            if station_holdout(scan, task, source, field) != plan:
                raise ValueError("锁定的站点回检范围与当前外发记录不一致")
            keys, minimum = plan["groups"], plan["minimum_grid_index"]

        def trial(index):
            data = pd.DataFrame(source.read(keys[index]), columns=source.header)
            if len(data) < 6:
                return index, None, "insufficient_score_support"
            matched = load(scan, task, table_data=data, keep_missing=True, minimum_grid_index=minimum)[method]
            if len(matched) < 6:
                return index, None, "insufficient_score_support"
            if BIAS not in steps:
                values=matched.background.to_numpy()
                metrics,reason=recipe_trials.score(values,values,matched.observed.to_numpy(),
                    'heldout_station_observations_comparison_only',retained=values,
                    request=task.get('scoring_request'))
                return index,metrics,reason
            frame, held, training = corrected_frame(matched)
            frame = frame[frame.grid_time.isin(held)]
            if not np.isfinite(training).any():
                return index, None, "insufficient_score_support"
            values = frame.corrected.fillna(frame.background).to_numpy()
            metrics, reason = recipe_trials.score(values, frame.background.to_numpy(), frame.observed.to_numpy(),
                                                  "heldout_station_observations", retained=training,
                                                  request=task.get("scoring_request"))
            return index, metrics, reason

        return recipe_trials.audit(len(keys), f"withheld_{source.category}", trial)


def native_update(scan,task,frame,values,delivery_status,task_id,*,purpose='selected_product'):
    """The writer receives indices of the actual sampled native parent."""
    import hashlib
    source=scan['rows'][task['grid_index']]
    native_field=next(f for f in source['fields'] if f['name']==task['grid_name'])
    rank=len(native_field['dimensions'])
    if len(frame)!=len(values):raise ValueError('原生写回索引和实际产品长度不符')
    with local_member(source) as original,original.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(source_sha256=digest,variable=task['grid_name'],unit=task['unit'],task=task_id,
        delivery_status=delivery_status,purpose=purpose,points=[dict(
            index=[int(row.native_t),*[0]*(rank-3),int(row.native_y),int(row.native_x)],
            expected=float(row.background),value=float(value)) for row,value in zip(frame.itertuples(),values)])


def run_fixed(scan,task_id,task,folder,model_id,progress):
    from ecodata.io.structures import export_structure, read_structure, table_dataset

    from .recipe_trials import FILL
    rows=load(scan,task,keep_missing=any(FILL in recipe for recipe in task["recipes"]))
    requested={next(step for step in recipe if step in {EXACT,NEAREST}) for recipe in task["recipe_steps"]}
    rows,queue_audit=common_queue({method:rows[method] for method in sorted(requested)})
    files,artifacts,reports=[],[],[]
    native_updates=[]
    for index,recipe in enumerate(task["recipes"],1):
        if recipe not in task["recipe_steps"]:
            raise ValueError("方案未登记")
        method=next(step for step in recipe if step in {EXACT,NEAREST})
        frame=rows[method].copy()
        report=dict(methods=recipe,matched_records=len(frame),stations=int(frame.station.nunique()),unit=task["unit"])
        if BIAS in recipe:
            frame,evaluation=evaluate_correction(frame)
            report.update(evaluation)
        report['queue_audit']=queue_audit[method]
        ds=table_dataset(frame)
        for name in ("observed","background","corrected","selected"):
            if name in ds:
                ds[name].attrs["units"]=task["unit"]
        raw,_,_=export_structure(ds,"Parquet")
        name=f"station-grid-{index}.parquet"
        with read_structure(name,raw,record_read_source=False) as check:
            if check.sizes!=ds.sizes:
                raise ValueError("站点配对成果回读行数不一致")
        files.append((name,raw))
        artifacts.append(dict(name=name,dimensions=dict(ds.sizes),fields=list(ds.data_vars)))
        reports.append(report)
        if model_id and model_id!='data_only' and BIAS not in recipe:
            native_updates.append(native_update(scan,task,frame,frame.background.to_numpy(),'BASELINE_RETAINED',task_id,purpose='comparison_passthrough'))
    return dict(files=files,source_formats=[scan["rows"][i]["format"] for i in (task["grid_index"],task["table_index"])],
                target_format="Parquet",original_report=reports,local=dict(artifacts=artifacts,
                     reason="已导出站点与网格的配对记录及所选方案的核验结果。"),native_updates=native_updates)



def run(scan,task_id,task,folder,model_id,progress):
    """Production fusion always reaches the public mutation orchestration."""
    if task['action']!='fusion':return run_fixed(scan,task_id,task,folder,model_id,progress)
    rows=load(scan,task)
    methods={next(step for step in recipe if step in {EXACT,NEAREST}) for recipe in task['recipe_steps']}
    rows,queues=common_queue({method:rows[method] for method in sorted(methods)})
    # Same fixed common labels/background values: time-matching options are
    # preparation controls, not distinct new scientific information sources.
    frame=rows[sorted(rows)[0]].copy();frame['unit']=task['unit']
    recipes=[]
    for index,steps in enumerate([] if task.get('search_after_rejection') else task['recipes']):
        if not steps:continue
        if steps not in task['recipe_steps']:raise ValueError('Unregistered confirmed recipe')
        recipes.append(dict(id=f'confirmed-{index+1}',steps=steps,scope=dict(variable=task['quantity']),
            origin=task.get('recommendation_origin','confirmed_registered_recipe')))
    request=dict(task='temporal',adapter='paired_station_calibration',variable=task['quantity'],unit=task['unit'],
        recommended_recipes=recipes,evaluation_protocol=task.get('evaluation_protocol','blocked_confirmation_v1'))
    if task.get('recommended_graphs'):request['recommended_graphs']=task['recommended_graphs']
    if task.get('search_after_rejection'):
        request['recommended_recipes'] = []
        request['recommendation_isolation'] = task.get('mutation_reservation', dict(
            status='INSUFFICIENT', reason='No prospective separate recommendation/mutation reserve'))
    from .local_search import run_paired
    return run_paired(scan, task_id, task, frame, request, folder, progress, queues,
                      source_indices=(task['grid_index'], task['table_index']))
