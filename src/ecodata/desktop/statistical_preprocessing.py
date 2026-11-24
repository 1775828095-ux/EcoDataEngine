"""Execute evidence-backed time-window and area-normalization plans."""
import hashlib
import io
import json
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.core.schemas import EcoDataBundle
from ecodata.engines.units import UnitEngine
from ecodata.temporal.semantic import SemanticTimeConfig,SemanticTimeResampler
from .scan import assert_current,decode,local_member

METHODS={"temporal.window_align":"按已核验年份和统计方式统一时间窗口",
         "units.area_normalize":"按对应对象的已核实面积将体积换算为深度",
         "units.harmonize":"原版物理单位换算"}


class PlanError(ValueError):
    pass


def parse_plan(value):
    if not isinstance(value,dict):
        return None
    start,end=value.get("start_year"),value.get("end_year")
    sources=value.get("sources")
    if (type(start) is not int or type(end) is not int or not 1<=start<=end<=9998
            or value.get("target_period") not in {"yearly","average_annual"}
            or not isinstance(value.get("target_unit"),str) or not value["target_unit"].strip()
            or not isinstance(sources,list) or len(sources)!=2):
        return None
    clean=[]
    for source in sources:
        if (not isinstance(source,dict) or not isinstance(source.get("field"),str)
                or source.get("statistic") not in {"sum","mean","identity"}
                or not isinstance(source.get("evidence"),str) or not source["evidence"].strip()):
            return None
        item={key:source[key].strip() for key in ("field","statistic","evidence")}
        for key in ("area_field","object_field","area_object_field","time_field"):
            if key in source:
                if not isinstance(source[key],str):
                    return None
                item[key]=source[key].strip()
        documents=source.get("evidence_documents",[])
        if not isinstance(documents,list) or not all(isinstance(doc,str) for doc in documents):
            return None
        item["evidence_documents"]=list(dict.fromkeys(documents))
        if "input_period" in source:
            if source["input_period"] not in {"daily","monthly","yearly"}:
                return None
            item["input_period"]=source["input_period"]
        if source.get("integrate_rate") is True:
            item["integrate_rate"]=True
        clean.append(item)
    return dict(start_year=start,end_year=end,target_period=value["target_period"],
                target_unit=value["target_unit"].strip(),sources=clean)


def references(plan):
    return {value for source in plan["sources"] for key,value in source.items()
            if key in {"field","area_field","object_field","area_object_field","time_field"}}


def field_map(scan):
    return {f"D{i}.F{j}":(i-1,row,field) for i,row in enumerate(scan["rows"],1)
            for j,field in enumerate(row["fields"],1)}


def table(scan,index,cache):
    if index in cache:
        return cache[index]
    row=scan["rows"][index]
    if row["format"] in {"NetCDF","HDF5","HDF5-NetCDF","GeoTIFF"}:
        raise PlanError("对象统计入口尚需将网格按已核实的流域或对象范围聚合为序列，请先推荐对应的空间聚合步骤")
    if row["format"]=="SWAT+ 输出表":
        from ecodata.interface.swat_results import output_header
        from ecodata.io.record_layout import split_declared_record
        with local_member(row) as path,path.open("rb") as stream:
            prefix=[decode(stream.readline()) for _ in range(row["swat_output_header"]+2)]
            _,columns,units,tail=output_header(prefix)
            numeric=[columns.index(name) for name in units]
            records=[split_declared_record(decode(line).split(),len(columns),numeric_columns=numeric,
                integer_columns=range(6),optional_tail=2 if tail else 0)[0] for line in stream if line.strip()]
        frame=pd.DataFrame(records,columns=columns)
    else:
        from ecodata.io.structures import read_structure,table_frame
        from .pair_processing import conversion_source
        name,raw=conversion_source(scan,index)
        with read_structure(name,raw,record_read_source=False) as dataset:
            if dataset.attrs.get("structure") not in {"table","vector"}:
                raise PlanError("统计窗口换算入口需要对象表；网格须先按明确范围聚合为对象序列")
            frame=table_frame(dataset)
    if frame.empty:
        raise PlanError("所选统计字段没有数据记录")
    cache[index]=frame
    return frame


def column(binding,lookup,index):
    source,_,field=lookup[binding]
    if source!=index:
        raise PlanError("对象或时间字段必须属于对应的来源表")
    return field.get("original_header",field["name"])


def values(scan,source,lookup,cache,documents):
    index,row,field=lookup[source["field"]]
    if field.get("axis"):
        raise PlanError("坐标和时间轴不能作为待换算的观测量")
    frame=table(scan,index,cache)
    name=field.get("original_header",field["name"])
    result=pd.DataFrame({"value":pd.to_numeric(frame[name],errors="raise")})
    if not np.isfinite(result.value).all():
        raise PlanError("所选字段有缺测或无效数值，需先质量筛查")
    native=row["format"]=="SWAT+ 输出表"
    if source.get("object_field"):
        result["object"]=frame[column(source["object_field"],lookup,index)].astype(str)
    elif native:
        result["object"]=list(map(tuple,frame[["unit","gis_id","name"]].astype(str).values))
    else:
        raise PlanError("请在方案中指定对象编号字段，平台不能按行号推断对象")
    if result["object"].map(lambda value:any(not token.strip() for token in value) if isinstance(value,tuple) else not value.strip()).any():
        raise PlanError("对象编号含空值")
    period=row.get("declared_output_period")
    if native and period in {"yearly","monthly","daily"}:
        year=pd.to_numeric(frame.yr,errors="raise").astype(int)
        month=pd.to_numeric(frame.mon,errors="raise").astype(int) if period in {"monthly","daily"} else 1
        day=pd.to_numeric(frame.day,errors="raise").astype(int) if period=="daily" else 1
        result["time"]=pd.to_datetime(dict(year=year,month=month,day=day))
        if period=="daily" and not np.array_equal(result.time.dt.dayofyear.to_numpy(),pd.to_numeric(frame.jday).to_numpy()):
            raise PlanError("原始日期与年积日不一致")
    elif period!="average_annual":
        fid=source.get("time_field")
        if not fid:
            raise PlanError("请在方案中指定时间字段")
        result["time"]=pd.to_datetime(frame[column(fid,lookup,index)],errors="raise",utc=True).dt.tz_localize(None)
        period=source.get("input_period") or ("daily" if row.get("resolution",{}).get("time_seconds")==86400 else None)
    unit=field["unit"]
    if source.get("area_field"):
        area_index,area_row,area=lookup[source["area_field"]]
        area_frame=table(scan,area_index,cache)
        area_values=pd.to_numeric(area_frame[area.get("original_header",area["name"])],errors="raise")
        scale,offset=UnitEngine().affine(area["unit"],"m^2")
        area_values=area_values*scale+offset
        if not np.isfinite(area_values).all() or (area_values<=0).any():
            raise PlanError("换算面积必须是正的有限数值")
        if area_index==index:
            areas=area_values
        else:
            if not source.get("object_field") or not source.get("area_object_field"):
                raise PlanError("跨文件面积换算需明确两边的对象关联字段")
            source_id=source["field"].split(".")[0]
            area_id=source["area_field"].split(".")[0]
            witnesses=[doc for doc in documents if doc["id"] in source["evidence_documents"]
                       and doc.get("text") and {source_id,area_id}<=set(doc.get("source_ids",[]))]
            if not witnesses:
                raise PlanError("未找到同时引用数值来源与面积来源的工程依据")
            keys=area_frame[column(source["area_object_field"],lookup,area_index)].astype(str)
            mapping=pd.DataFrame({"object":keys,"area":area_values})
            if mapping.groupby("object").area.nunique().gt(1).any():
                raise PlanError("同一对象的面积随记录变化，需先明确时变面积的配对规则")
            areas=result["object"].map(mapping.drop_duplicates("object").set_index("object").area)
            if areas.isna().any():
                raise PlanError("部分对象未找到对应面积")
        result["area_m2"]=np.asarray(areas)
    return result,unit,period


def align(frame,unit,period,row,source,plan):
    first,last=plan["start_year"],plan["end_year"]
    target=plan["target_period"]
    if period=="average_annual":
        window=row.get("declared_output_print_period",{})
        if (not row.get("single_average_annual_window") or window.get("start_year")!=first
                or window.get("end_year")!=last or target!="average_annual" or source["statistic"]!="identity"):
            raise PlanError("平均年度输出需核实同一完整年份窗口，不能拆成逐年真值")
        if frame["object"].duplicated().any():
            raise PlanError("平均年度输出含同一对象的多条记录")
        out=frame.copy()
        out["start_year"],out["end_year"]=first,last
    else:
        frame=frame.loc[frame.time.dt.year.between(first,last)].copy()
        if frame.empty or frame.duplicated(["object","time"]).any():
            raise PlanError("统计窗口内无数据或同一对象时间重复")
        yearly=[]
        output_unit=unit
        for identity,group in frame.groupby("object",sort=False):
            group=group.sort_values("time")
            frequency={"daily":"D","monthly":"MS","yearly":"YS"}.get(period)
            if not frequency:
                raise PlanError("请先核实采样间隔，当前入口支持逐日、逐月和逐年完整序列")
            expected=pd.date_range(f"{first}-01-01",f"{last+1}-01-01",freq=frequency,inclusive="left")
            if not pd.DatetimeIndex(group.time).equals(expected):
                raise PlanError("对象序列未完整覆盖所选年份窗口，需先处理缺测或统计日界")
            method=source["statistic"]
            if period=="yearly":
                if source.get("integrate_rate"):
                    raise PlanError("逐年速率需明确区间均值后再积分，当前方案尚未提供这一依据")
                if (target=="average_annual" and method!="mean") or (target=="yearly" and method!="identity"):
                    raise PlanError("逐年值对齐需明确保留逐年值或对年份取均值")
                result=group[["time","value"]].copy()
            else:
                if method not in {"sum","mean"}:
                    raise PlanError("细时间尺度聚合需明确累计或区间均值")
                start=group.time.to_numpy(dtype="datetime64[ns]")
                ends=expected[1:].append(pd.DatetimeIndex([pd.Timestamp(f"{last+1}-01-01")])).to_numpy()
                data=xr.DataArray(group.value.to_numpy(),dims="time",coords={"time":start},name="value",attrs={"units":unit})
                if source.get("integrate_rate"):
                    integration_unit="m^3" if source.get("area_field") else plan["target_unit"]
                    data=UnitEngine().integrate_rate(data,(ends-start)/np.timedelta64(1,"s"),integration_unit)
                    output_unit=data.attrs["units"]
                    if method!="sum":
                        raise PlanError("区间速率积分后应按累计量聚合")
                data.attrs.update(temporal_semantics="accumulation" if method=="sum" else "mean",data_kind="continuous")
                dataset=data.to_dataset(name="value")
                dataset["time_bounds"]=(('time','bounds'),np.column_stack([start,ends]))
                restored=SemanticTimeResampler(SemanticTimeConfig("YS")).apply(EcoDataBundle(dataset)).data.dataset
                result=pd.DataFrame({"time":pd.date_range(f"{first}-01-01",periods=last-first+1,freq="YS"),
                                     "value":restored.value.values})
            if target=="average_annual":
                yearly.append(dict(object=identity,start_year=first,end_year=last,value=result.value.mean()))
            else:
                yearly.extend(dict(object=identity,start_year=int(timestamp.year),end_year=int(timestamp.year),value=float(value))
                              for timestamp,value in zip(result.time,result.value))
        out=pd.DataFrame(yearly)
        unit=output_unit
    if source.get("area_field"):
        areas=frame[["object","area_m2"]].drop_duplicates()
        if areas["object"].duplicated().any():
            raise PlanError("统计窗口内同一对象的面积有变化，需要明确时变面积的统计规则")
        mapped=out["object"].map(areas.set_index("object").area_m2)
        if mapped.isna().any():
            raise PlanError("统计后的部分对象没有对应面积")
        scale,offset=UnitEngine().affine(unit,"m^3")
        out["value"]=(out.value*scale+offset)/mapped.to_numpy()
        unit="m"
    scale,offset=UnitEngine().affine(unit,plan["target_unit"])
    out["value"]=out.value*scale+offset
    return out


def prepare(scan,plan,pair_fields,documents):
    assert_current(scan)
    plan=parse_plan(plan)
    if not plan or len(set(pair_fields))!=2 or {source["field"] for source in plan["sources"]}!=set(pair_fields):
        raise PlanError("前处理方案需完整指定两边字段、年份窗口、统计方法和目标单位")
    lookup=field_map(scan)
    if not references(plan)<=lookup.keys():
        raise PlanError("前处理方案引用了未扫描的字段")
    cache_key=hashlib.sha256(json.dumps(plan,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cached=scan.setdefault("statistical_preprocessing_tasks",{})
    if cache_key in cached:
        return cached[cache_key]
    frames=[]
    local={}
    for source in plan["sources"]:
        frame,unit,period=values(scan,source,lookup,local,documents)
        frames.append(align(frame,unit,period,lookup[source["field"]][1],source,plan))
    methods=["temporal.window_align"]
    if any(source.get("area_field") for source in plan["sources"]):
        methods.append("units.area_normalize")
    methods.append("units.harmonize")
    task=dict(action="preprocess",route="statistical_preprocessing",plan=plan,frames=frames,
              recipe_steps=[methods],source_indices=[lookup[source["field"]][0] for source in plan["sources"]])
    cached[cache_key]=task
    return task


def run(scan,task_id,task,folder,model_id,progress):
    assert_current(scan)
    if task["recipes"]!=task["recipe_steps"]:
        raise PlanError("前处理步骤与已核实方案不一致")
    files=[]
    artifacts=[]
    for index,frame in enumerate(task["frames"],1):
        result=frame.copy()
        result["unit"]=task["plan"]["target_unit"]
        name=f"preprocessed-source-{index}.csv"
        raw=result.to_csv(index=False).encode("utf-8-sig")
        restored=pd.read_csv(io.BytesIO(raw))
        np.testing.assert_allclose(restored.value,result.value)
        files.append((name,raw))
        artifacts.append(dict(name=name,dimensions={"record":len(result)},fields=list(result.columns)))
    return dict(files=files,source_formats=[scan["rows"][index]["format"] for index in task["source_indices"]],
                target_format="CSV",original_report=dict(plan=task["plan"],methods=task["recipe_steps"][0]),
                local=dict(artifacts=artifacts,reason="已按实际对象和完整年份窗口统一统计口径，再按核实的面积和单位换算；结果已回读核对。"))
