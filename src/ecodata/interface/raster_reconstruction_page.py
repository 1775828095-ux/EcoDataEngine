"""Formal file-upload UI for input-only raster routing; no reference uploader."""
from pathlib import Path
import io
import json
import zipfile
import joblib
import streamlit as st
from ecodata.interface.raster_upload import RasterRequest,process_raster_upload

ROOT=Path(__file__).resolve().parents[3]


def render_raster_reconstruction_page():
    st.subheader("单源栅格自动重构")
    st.caption("只使用上传的粗网格、物理元数据和任务需求；不接收隐藏高分辨率真值。当前支持正方形经纬度连续网格。")
    upload=st.file_uploader("上传粗网格NetCDF",type=["nc","nc4"],key="raster_upload")
    if upload is None:return
    variable=st.text_input("变量名",value="field")
    size=int(st.number_input("目标每边像元数",min_value=4,max_value=200,value=100))
    description=st.text_input("任务描述",value="重建非负连续空间场")
    boundary=st.checkbox("关注区域边界")
    peaks=st.checkbox("关注局部高值和峰位（不能保证恢复未观测的细峰）")
    consistency=st.checkbox("优先回聚合一致性",value=True)
    single=st.checkbox("只允许单算法，不增加一致性QP")
    model_path=st.text_input("开发集冻结的路由模型",value=str(ROOT/"output/three_task_routing_20261020/router.joblib"))
    if st.button("自动筛选并重构",type="primary"):
        try:
            model=joblib.load(model_path)
            request=RasterRequest(variable,size,description,float(boundary),.3,float(peaks),1. if consistency else .3)
            run=process_raster_upload(upload,request,model,mode="automatic_single" if single else "automatic")
            st.success("已执行："+run["selected"])
            st.json(run["trace"])
            field=run["outputs"][run["selected"]]
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("reconstructed.nc",field.to_netcdf(engine="h5netcdf"))
                archive.writestr("decision_and_quality.json",json.dumps(dict(trace=run["trace"],quality=run["reports"]),ensure_ascii=False,default=str))
            st.download_button("下载结果和决策记录",buffer.getvalue(),"raster_reconstruction.zip","application/zip")
        except Exception as exc:st.error("未完成："+str(exc))
