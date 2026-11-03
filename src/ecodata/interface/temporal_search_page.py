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


def render_temporal_search_page():
    import streamlit as st
    st.subheader("小时PM2.5恢复 · GOLEM真实数值搜索")
    st.caption("此兼容页仅支持规则1小时、非负PM2.5恢复；空间、时间聚合和多源请使用主页面的统一算法图搜索。原栅格学习路由不是GOLEM。")
    data=st.file_uploader("处理CSV：station,time,pm25；保留自然缺失，不填真值",type=["csv"],key="ts_data")
    variable=st.text_input("目标变量列",value="pm25",key="ts_variable")
    unit=st.selectbox("单位（须由数据来源确认）",["UNKNOWN","ug m-3"],key="ts_unit")
    access=st.selectbox("允许访问的已保留观测",["both_sides","past_only"],key="ts_access")
    complete=st.checkbox("要求完整预测覆盖",value=True,key="ts_complete")
    prefer=st.checkbox("偏好使用仍存在的观测支持（不保证精度更高）",value=True,key="ts_support")
    st.caption("无时区标签原样保留；不推断小时起止或平均窗口。显式时间标签必须一致。")
    history=st.file_uploader("可选历史状态JSON：实体→medians(24值)、training_end；须早于输入。无状态则排除历史算法",type=["json"],key="ts_history")
    ref=st.file_uploader("可选独立验证CSV：station,time,truth；这些位置须已在输入中缺失",type=["csv"],key="ts_ref")
    role=st.checkbox("确认参考仅为合法validation数据，不是最终test标签",key="ts_ref_role")
    coverage=st.checkbox("无参考时，仅优化明确的覆盖指标，不声称精度最优",key="ts_coverage")
    confirm=st.checkbox("确认保留输入缺失；接受只读质量筛查并查看异常警告，不自动删除或修改原始数据",key="ts_confirm")
    limit=int(st.number_input("最多候选评价数",1,20,5,key="ts_budget"))
    seconds=int(st.number_input("搜索墙钟上限（秒；单个不可中断内核结束后停止）",1,60,30,key="ts_seconds"))
    if st.button("提交一次有界搜索",key="ts_submit"):
        folder=Path(__file__).resolve().parents[3]/"output/platform_temporal_search"/uuid.uuid4().hex
        try:
            if data is None:raise ValueError("请提供真实输入CSV")
            if not confirm:raise ValueError("请确认保留原始缺失并查看质量警告")
            if ref and not role:raise ValueError("需确认验证数据用途")
            started=perf_counter()
            external,evaluation=parse_upload(data.getvalue(),ref.getvalue() if ref else None,json.loads(history.getvalue()) if history else {},variable=variable,unit=unit)
            parsing_seconds=perf_counter()-started
            if evaluation is None and coverage:
                from ecodata.algorithms.temporal_search import CoverageEvaluation
                evaluation=CoverageEvaluation()
            request=dict(operation="temporal_missing_recovery",require_complete_coverage=complete,prefer_observation_support=prefer,time_access=access)
            config=dict(DEFAULT_BUDGET,objective_limit=limit,wall_seconds=seconds)
            status=st.empty()
            result=process_temporal_request(external,request,evaluation,config,folder,progress=lambda event:status.write(event))
            result["upload_parse_seconds"]=parsing_seconds
            from ecodata.algorithms.numerical_search import save
            save(folder/"summary.json",result)
            st.session_state["ts_result"]=(result,str(folder))
        except Exception as exc:
            st.session_state["ts_result"]=({"status":"BLOCKED","error":str(exc)},str(folder))
    # A rerender/download never starts a second search.
    if "ts_result" in st.session_state:
        result,folder=st.session_state["ts_result"]
        st.json(result)
        decision=Path(folder)/"decision.json"
        if decision.exists():st.json(json.loads(decision.read_text(encoding="utf-8")))
        if result.get("best"):st.success("已选中真实执行且通过质量门的图；指标口径见评价协议")
        elif "error" in result:st.error(result["error"])
        else:st.warning("预算内未得到合格完整图；保留失败与停止记录")
        st.download_button("下载处理产品及决策记录",download_search_result(folder),"temporal_search.zip","application/zip",key="ts_download")
