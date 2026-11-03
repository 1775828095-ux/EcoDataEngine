"""Thin CSV/UI client of the platform numerical-search service (not a backend)."""
from pathlib import Path
import io,json,uuid,zipfile
from time import perf_counter
import numpy as np
import pandas as pd


DEFAULT_BUDGET=dict(objective_limit=5,proposal_limit=50,callback_limit=100,wall_seconds=30)


def process_temporal_request(external,request,evaluate,config,folder,*,seed=11,mode="C",progress=None):
    from ecodata.algorithms.temporal_search import submit_temporal_search
    return submit_temporal_search(external,request,evaluate,config,folder,seed=seed,mode=mode,progress=progress)


def download_search_result(folder):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in Path(folder).rglob("*"):
            if path.is_file():archive.write(path,path.relative_to(folder))
    return buffer.getvalue()


def parse_upload(data,reference,history,*,variable,unit):
    # Explicit schema and unit, never inferred from the file name. No pickle uploads.
    from ecodata.algorithms import numerical_search  # initialise existing official runtime
    from ecodata.algorithms.mixed_gap_dag import SeriesCase,Batch,SealedArtifact,readonly,missing_runs,TASK
    from ecodata.algorithms.temporal_search import ReferenceEvaluation
    if unit!="ug m-3":raise ValueError("需要明确单位 ug m-3；不自动推断或转换")
    frame=pd.read_csv(io.BytesIO(data))
    if not {"station","time",variable}<=set(frame):raise ValueError("CSV需要 station、time 和指定变量列")
    if frame.station.isna().any():raise ValueError("缺少空间实体编号")
    refs=pd.read_csv(io.BytesIO(reference)) if reference else None
    if refs is not None:
        if not {"station","time","truth"}<=set(refs):raise ValueError("验证CSV需要 station、time、truth；仅填写合法验证位置")
        if refs.duplicated(["station","time"]).any():raise ValueError("验证位置重复")
        refs.station=refs.station.astype(str);refs.time=pd.to_datetime(refs.time,errors="raise")
        if refs.duplicated(["station","time"]).any():raise ValueError("解析后的验证时间重复")
    cases=[];lengths=[];offsets=[];query=[];truth=[];offset=0;used=set()
    for entity,group in frame.groupby("station",sort=True):
        times=pd.DatetimeIndex(pd.to_datetime(group.time,errors="raise"));values=pd.to_numeric(group[variable],errors="raise").to_numpy(float)
        missing=np.flatnonzero(~np.isfinite(values))
        cases.append(SeriesCase(str(entity),readonly(times.to_numpy()),readonly(values),readonly(missing)))
        lengths.extend(missing_runs(values)[missing]);offsets.append((offset,offset+len(missing)))
        for j,index in enumerate(missing):
            if refs is None:query.append(offset+j)
            else:
                hit=refs[(refs.station==str(entity))&(refs.time==times[index])]
                if len(hit):
                    query.append(offset+j);truth.append(float(hit.truth.iloc[0]));used.add(hit.index[0])
        offset+=len(missing)
    if refs is not None and len(used)!=len(refs):raise ValueError("验证位置必须存在且已从输入移除；不自动制造遮挡")
    meta=dict(unit=unit,contract=TASK)
    batch=Batch(tuple(cases),readonly(np.array(lengths,int)),tuple(offsets))
    external=dict(batch=SealedArtifact("masked_batch",batch,meta),history=SealedArtifact("fitted_history",history or {},meta),
                  query=SealedArtifact("query_positions",readonly(np.array(query,int)),meta))
    return external,ReferenceEvaluation(truth) if refs is not None else None


"""Retired experimental page; use the main platform workbench."""
def render_temporal_search_page():
    import streamlit as st
    st.info("旧GOLEM时间搜索页面已移除，请返回主工作台使用当前自动处理入口。")
