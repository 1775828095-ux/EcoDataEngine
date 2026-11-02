"""Two explicit WRF routes with recorded native-stage execution."""
import json
import streamlit as st
from ecodata.interface import wrf_runner as runner


def _results(ident):
    record=runner.status(ident)
    st.write('任务编号：'+ident)
    route=record['route']
    st.caption('下方结果所属路线：'+('已有工程更新' if route=='existing' else '从原始资料重新搭建'))
    stage_file=runner.result_path(ident,'native_'+('from_zero' if route=='from_zero' else 'existing')+'_outputs.json')
    if record['state']=='succeeded':
        st.success(f"原生WRF链执行成功，已完成{record.get('simulation_seconds',180)}秒接入验收运行。")
        if stage_file.is_file():st.dataframe(json.loads(stage_file.read_text())['stages'],hide_index=True)
        st.image(str(runner.result_path(ident,'temperature.png')))
        st.caption('这是短时运行接入验收图；未进行观测比对、率定或预报精度验证。')
        for name,label in [('temperature.png','下载温度图'),('assembly.json','下载字段来源与接入记录'),(stage_file.name,'下载原生阶段与输出记录')]:
            path=runner.result_path(ident,name)
            st.download_button(label,path.read_bytes(),file_name=name,key=ident+name)
    elif record['state']=='failed':
        st.error(record['error'])
    else:
        st.info('正在执行原生预处理和WRF模拟。')
    log=runner.result_path(ident,'runner.log')
    if log.is_file():
        with st.expander('运行日志'):
            st.code(log.read_text(encoding='utf-8',errors='replace')[-10000:])


@st.fragment(run_every=2)
def _live(ident):
    if runner.status(ident)['state']!='running':st.rerun()
    _results(ident)


def render_wrf_run_page(converted_zip=None,input_mode=None):
    from ecodata.interface.independent_cases import render
    if render('wrf', converted_zip, input_mode):
        return
    st.header('WRF 工程运行')
    st.warning('当前仅支持已登记的2023-03-31案例；气压场/组包/地形为30 km网格、180秒，SST为6 km大西洋域、36秒。暂不支持任意新区域或任意物理方案。')
    st.caption('原始FNL来自NOAA/NCEP公开分析资料；配套地理资料和工程配置沿用已核验案例。不是私人客户数据。')
    label=st.radio('工程路线',['已有工程更新','从原始资料重新搭建'],key='wrf_route')
    route='existing' if label=='已有工程更新' else 'from_zero'
    from ecodata.interface.engineering_flow import render_flow
    render_flow('wrf',route)
    if route=='existing':
        st.write('保留已有域和geo_em地理结果 → 接入平台气压场及湿度、地表、土壤配套资料 → metgrid → real → WRF → 出图。')
    else:
        st.write('原始FNL GRIB及登记地理资料 → 重新执行geogrid和ungrib → 接入平台气压场 → metgrid → real → WRF → 出图。')
        st.caption('此入口重新生成工程输入，区域与物理配置仍来自登记案例；不等同于任意流域/区域的自动建模。')
    st.write('平台自动处理WPS主机文件名和引用关系，核对时间、全部气压层、单位、网格及地球半径；有差异直接报错。')
    if input_mode=='geogrid_tiles':st.info('接入本次HGT_M地形或LANDUSEF土地利用瓦片。两条路线均重新执行geogrid；限定已核验GMTED或MODIS21裁剪范围，保留21层混合地类百分比。其他地理和气象配套资料复用，不支持任意新区域。')
    if input_mode=='sst':
        st.info('本次接入2023-03-31 12UTC的NOAA OISST单快照，显式固定36秒；不是逐小时观测。已有路线复用地理与FNL预处理场，新建路线重新执行geogrid/ungrib；两条都执行metgrid、real和WRF。')
    source='本次转换结果' if converted_zip is not None else st.radio('气压场来源',['使用登记案例原始NetCDF，经平台重新转换','上传平台转换结果ZIP'],key='wrf_source')
    if converted_zip is not None:
        st.info('已接收上方本次转换ZIP；运行前严格核对对应登记案例，失败时不换用示例。')
    upload=None
    if source=='上传平台转换结果ZIP':
        upload=st.file_uploader('平台输出ZIP（含5个时刻的TT、UU、VV、GHT全部气压层）',type=['zip'],key='wrf_update')
        st.caption('本案例earth_radius_km应为6371.2294921875；旧固定半径输出会被拒绝，请用原始数据及正确配置重新转换。')
    if st.button('接入并运行WRF',disabled=source=='上传平台转换结果ZIP' and upload is None,type='primary'):
        try:
            with st.spinner('检查输入并组装工程…'):
                payload=converted_zip if converted_zip is not None else upload.getvalue() if upload else None
                ident=runner.start(route,payload,mode=input_mode) if input_mode else runner.start(route,payload)
            st.session_state['wrf_job']=ident
            st.query_params['wrf_job']=ident
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    ident=st.session_state.get('wrf_job') or (st.query_params.get('wrf_job') if converted_zip is None else None)
    if ident:
        try:
            if runner.status(ident)['state']=='running':_live(ident)
            else:_results(ident)
        except (ValueError,FileNotFoundError) as exc:
            st.error('无法恢复该WRF任务：'+str(exc))
