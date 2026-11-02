"""Reviewable fixed-case native CALMET/CALPUFF workflows."""
import json
import re
import streamlit as st
from ecodata.interface.calmet_runner import RUNS,execute

def render_calmet_run_page(converted_zip=None,mode=None):
    from ecodata.interface.independent_cases import render
    if render('calpuff', converted_zip, mode):
        return
    st.title('CALMET / CALPUFF 工程运行')
    st.info('首个登记案例：1990-01-09，固定 UTC−5，小时结束 05–08。气象资料来自历史观测，排放源为官方演示设定；浓度结果不代表真实污染事件。')
    if converted_zip is not None:
        st.info('已接收上方本次转换ZIP；只接受已核验登记案例的精确输入。校验不符时拒绝，不会使用示例冒充本次数据。受体登记为EPA真实站址的地面评价情景，不代表仪器进气口高度。')
    saved_id=st.session_state.get('calmet_current_job','') if converted_zip is not None else st.query_params.get('calmet_job','')
    saved_record=RUNS/saved_id/'result.json' if re.fullmatch('[0-9a-f]{32}',saved_id) else None
    saved_route=None
    if saved_record is not None and saved_record.is_file():
        saved_route=json.loads(saved_record.read_text(encoding='utf-8')).get('route')
    route=st.radio('工程路线',['更新已有工程','从原始观测重建登记工程'],index=1 if saved_route=='rebuild' else 0,key='calmet_route')
    from ecodata.interface.engineering_flow import render_flow
    render_flow('calpuff','existing' if route=='更新已有工程' else 'rebuild')
    st.write('已有路线：原始地面观测→平台转换→更新 SURF.DAT→CALMET→CALPUFF→浓度图。')
    st.write('重建路线：重建地面、PWM/CHH 探空、小时降水输入；MAKEGEO 从已栅格化地形及土地利用重建地理文件，再运行 CALMET 和 CALPUFF。')
    st.caption('两条路线均复用 MM4、海上站和官方编辑后的 Albany 探空。重建路线不是任意新区域的全自动建模，也不包含原始 DEM 的栅格预处理。MAKEGEO 按原案例配置将 88 个覆盖不足网格补为 LU55；所选 4 小时降水均为无雨，未验证正降水情形。坐标为 NAS-C / UTM 19N。')
    st.markdown('资料来源：[SMERGE 原始地面观测](https://www.calpuff.org/calpuff/download/EPA_Files/SMERGE_v5.57.zip)、[READ62 探空](https://www.calpuff.org/calpuff/download/EPA_Files/READ62_v5.54.zip)、[PXTRACT 降水](https://www.calpuff.org/calpuff/download/EPA_Files/PXTRACT_v4.25.zip)。这些是公开教学案例中的观测资料，不是私人客户提供的数据。')
    if st.button('运行登记案例',type='primary'):
        try:
            with st.spinner('正在转换输入、运行原生模型并出图…'):
                result=execute('existing' if route=='更新已有工程' else 'rebuild',converted_zip=converted_zip,mode=mode)
            new_id=result['folder'].replace('\\','/').split('/')[-1]
            if converted_zip is not None:st.session_state['calmet_current_job']=new_id
            else:st.query_params['calmet_job']=new_id
        except Exception as exc:
            st.error('本次运行失败：'+str(exc))
            return
    ident=st.session_state.get('calmet_current_job','') if converted_zip is not None else st.query_params.get('calmet_job','')
    if not ident:return
    if not re.fullmatch('[0-9a-f]{32}',ident):st.error('任务编号无效');return
    folder=RUNS/ident
    if not (folder/'result.json').is_file():st.warning('任务记录不存在');return
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if result['status']!='success':st.error(result.get('error',result['status']));return
    st.success('原生 CALMET 与 CALPUFF 均完成；浓度文件包含 4 个小时、7 个物种。')
    st.write('任务：'+ident+'；实际运行路线：'+result['route'])
    st.metric('SO₂ 最大小时浓度',f"{result['SO2_max_ug_m3']:.6f} µg/m³")
    st.image(str(folder/'SO2_four_hours.png'),caption='CALPUFF 原生浓度文件读取结果；四幅图使用同一色标。')
    if (folder/'SO2_discrete.png').is_file():
        st.image(str(folder/'SO2_discrete.png'),caption='真实EPA站址的地面受体情景；排放为演示设定，非观测浓度对比。')
        st.download_button('下载离散受体浓度CSV',(folder/'SO2_discrete.csv').read_bytes(),'SO2_discrete.csv','text/csv')
    st.download_button('下载 SO₂ 网格 CSV',(folder/'SO2_grid.csv').read_bytes(),'SO2_grid.csv','text/csv')
    st.download_button('下载运行与来源记录',(folder/'result.json').read_bytes(),'calmet_result.json','application/json')
    with st.expander('原生阶段与复用资料'):
        st.json({k:result[k] for k in ['stages','retained_inputs','times','species']})
