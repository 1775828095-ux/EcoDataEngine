"""Guided raw-input workflow backed by QSWAT+ and the official Editor."""
import json
import hashlib
from pathlib import Path
import streamlit as st
from ecodata.interface import swat_build as builder
from ecodata.interface import swat_runner as runner
from ecodata.core.exceptions import SchemaError


@st.fragment(run_every=3)
def render_build_job(ident,converted_inputs=None,input_mode=None):
    job = builder.status(ident)
    st.write('当前步骤：' + job['stage'])
    directory = Path(job['directory'])
    if job['state'] == 'failed':
        st.error('建工程失败：' + job['error'])
    with st.expander('查看建工程日志与位置', expanded=job['state'] == 'failed'):
        st.caption(str(directory))
        log = directory / 'build.log'
        st.code(log.read_text(encoding='utf-8', errors='replace')[-16000:] if log.exists() else '准备中', language='text')
    if job['state'] == 'ready':
        project_payload=(directory/'project.zip').read_bytes()
        if converted_inputs is not None:
            from ecodata.models.swat_inputs_project import attach_inputs
            import io,zipfile
            st.info('新工程已生成。接下来将上方本次转换参数接入新工程副本；同名文件和对象标识不一致会拒绝，不会跳过本次输入。')
            if not st.checkbox('确认本次参数属于新建流域及计算对象',key='build_parameter_confirm_'+ident):return
            try:
                attached,report=attach_inputs(runner.read_project(project_payload),converted_inputs,input_mode)
                buffer=io.BytesIO()
                with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                    for name,raw in attached.items():z.writestr(name,raw)
                project_payload=buffer.getvalue();st.json(report)
            except (ValueError,OSError,SchemaError) as exc:
                st.error('本次参数接入被拒绝：'+str(exc));return
        st.success('已从原始GIS与平台天气生成工程；尚未运行模型，不代表参数已率定。')
        st.download_button('保存新工程', project_payload, 'swatplus_new_project.zip')
        st.caption('下一步会运行官方SWAT+。程序可能拒绝参数或计算失败，原始资料及失败日志都会保留。')
        charts = st.checkbox('输出所选日期的逐日水量与河道图表（不跳过输出年份）', value=True, key='build_charts_' + ident)
        engine_id = st.selectbox('运行程序', list(runner.ENGINE_LABELS), index=1,
                               format_func=lambda value: runner.ENGINE_LABELS[value], key='build_engine_' + ident)
        if engine_id == 'gnu62':
            st.warning('此GNU构建在Robit同一输入上出现浮点异常，尚未修复。请优先使用官方Editor Intel rev.62；失败后不会自动替换程序或更改物理参数。')
        if st.button('运行新建工程', key='run_new_' + ident):
            try:
                files = runner.read_project(project_payload)
                if charts:
                    from ecodata.interface.swat_results import enable_charts
                    files, _ = enable_charts(files)
                st.session_state['swat_build_active_job'] = runner.start_job(files, runner.ROOT / 'runs/swat_plus', engine_id=engine_id)
                st.rerun()
            except (ValueError, OSError, SchemaError) as exc:
                st.error(str(exc))


def render_build_page(weather_inputs=None,converted_inputs=None,input_mode=None):
    st.write('① 提供天气和当地原始资料 → ② 确认设置 → ③ 自动建立工程 → ④ 运行与查看结果')
    st.warning('从零建工程为试验功能，面向符合下列资料要求的流域，并非Robit专用。其他流域及资料缺失情形尚未完成通用验证；建成和运行成功不等于科学适用或预测准确。')
    with st.expander('已验证案例与限制（固定验收记录，不是本次资料识别结果）'):
        st.write('Robit公开教程资料曾完成GIS建工程、官方Intel rev.62全年运行及出图。验收使用200/200格阈值；教程1000/100格在300米出口吸附限制下失败，GNU构建同输入曾发生浮点错误。该记录不会随上传资料改变，也不证明当前边界与观测站匹配。')
    st.caption('目标：SWAT+62.0.0、QSWAT+4.0.3、Editor4.0.2、QGIS3.40.9。当前接受米制投影DEM、土地利用/土壤栅格、出口点、分类对照、152列土壤参数及完整12个月WGN统计。')
    with st.expander('本入口支持范围：新建与复用工程有什么区别？'):
        st.write('这里按原始DEM、土地利用、土壤和出口构建流域、河网及HRU；可按配置导入土壤、WGN及可选植物/城镇表，河网约束需明确提供图层和刻蚀深度。资料换成其他流域不等于必然失败，但仍需通过投影、分类覆盖、参数引用和出口吸附检查。')
        st.warning('湖泊/水库、沉沙池、取排水及自定义管理过程尚未在本入口完成通用构建与验证。ZIP里附带这些图层不代表它们已经参与建模；当前只处理配置中明确支持的资料。需要这些过程时，不能将本入口结果视为完整工程。')
        st.write('复用已有工程则保留原工程中已建立的对象与连接，在支持范围内接入数据再运行；能够运行作者工程，不等于本入口能从原始资料重新生成其中全部结构。')
    if weather_inputs is None:
        weather = st.file_uploader('① 日天气原始字段 NC', type=['nc'], key='new_weather_nc')
        weather_config = st.file_uploader('天气字段与单位配置 JSON', type=['json'], key='new_weather_config')
    else:
        from io import BytesIO
        weather, weather_config = (BytesIO(value) for value in weather_inputs)
        st.info('已沿用上方本次转换的原始天气和配置，无需重复上传；建工程时由同一转换器校验并写入天气。')
    st.info('这里上传额外的当地GIS与参数资料ZIP：DEM、土地利用、土壤、出口、分类及参数表。刚才下载的天气转换ZIP不是这份资料，不能上传到此框。')
    raw = st.file_uploader('② 当地原始GIS与参数资料 ZIP（不需要现成工程）', type=['zip'], key='new_gis_zip')
    config = st.file_uploader('资料路径与工程设置 JSON', type=['json'], key='new_gis_config')
    st.caption('如需用实测或已有河网约束DEM水流路径，可在ZIP提供与DEM同投影的线图层，并在JSON同时指定burn_streams与burnin_depth_metres（整数米）。该操作会改变工作副本的地形水流路径，不是补齐缺失土地利用或土壤；平台不猜测刻蚀深度。')
    fingerprint = hashlib.sha256()
    if converted_inputs is not None:fingerprint.update(converted_inputs)
    for upload in (weather, weather_config, raw, config):
        payload = upload.getvalue() if upload is not None else b''
        fingerprint.update(len(payload).to_bytes(8, 'big'))
        fingerprint.update(payload)
    signature = fingerprint.hexdigest()
    if st.session_state.get('swat_build_inputs_fingerprint') != signature:
        st.session_state['swat_build_inputs_fingerprint'] = signature
        st.session_state.pop('swat_build_job', None)
        st.session_state.pop('swat_build_active_job', None)
    st.caption('配置中指定ZIP内资料路径、河网/河道阈值、出口吸附距离和坡度分级。当前先保留全部HRU、不合并短河道；不会补缺失分类或天气。暂不支持用户自定义管理规则及任意工程参数覆盖。')
    st.markdown('#### 建工程采用哪些资料与参数？')
    st.table([
        {'来源': '上方上传的天气NC和天气配置', '用途': '转换并写入所提供的日天气变量、站点和日期', '当前如何调整': '修改源数据或天气配置后重新转换；不自动补造缺测'},
        {'来源': 'GIS ZIP中的DEM、土地利用、土壤、出口', '用途': '划分流域、河网和HRU；按设置吸附出口及处理栅格', '当前如何调整': '更换原始资料及建工程设置后重新建立'},
        {'来源': 'GIS ZIP中的usersoil、分类对照及WGN表', '用途': '采用上传的土壤参数、类别映射和12个月天气发生器统计', '当前如何调整': '编辑符合规定格式的表，再重新上传ZIP；不是全部使用默认值'},
        {'来源': '配置指定的plant、urban表（可选）', '用途': '指定时导入上传的植物、城镇参数表；未指定时沿用参考库相应表', '当前如何调整': '在ZIP中提供兼容CSV，并在工程设置JSON中指定plant、urban路径'},
        {'来源': '工程设置JSON', '用途': '采用河网/河道阈值、出口吸附距离和坡度分级', '当前如何调整': '编辑stream_cells、channel_cells、snap_metres、slope_limits后重新上传'},
        {'来源': '官方参考数据库与工具生成规则', '用途': '提供建工程所需的其余配套参数与默认管理配置；不等于当地实际农事记录', '当前如何调整': '本入口暂不支持自定义管理规则及任意参数覆盖；不能假定所有默认项已被上传资料替换'},
    ])
    st.info('上传参数不等于当地实测参数：其性质取决于资料来源。采用参考库也不代表覆盖全部上传参数。当前仅说明各类来源，尚未提供最终工程每个参数的来源与覆盖差异清单。指定的表能否导入以资料校验及建工程日志为准。')
    if config is not None:
        try:
            supplied = json.loads(config.getvalue().decode('utf-8-sig'))
            if not isinstance(supplied, dict):
                raise ValueError('工程设置必须是JSON对象')
            st.markdown('**本次工程设置指定的参数来源（尚未证明文件有效或已被模型读取）**')
            st.table([{'参数类别': key, '本次指定来源': str(supplied.get(key) or '未指定；沿用参考库相应表' if key in ('plant', 'urban') else supplied.get(key) or '未提供，不能靠默认值补齐')}
                      for key in ('usersoil', 'landuse_lookup', 'soil_lookup', 'wgn_stations', 'wgn_months', 'plant', 'urban')])
            st.write('本次划分设置：', {key: supplied.get(key, '未提供') for key in ('stream_cells', 'channel_cells', 'snap_metres', 'slope_limits')})
        except (ValueError, UnicodeError) as exc:
            st.error('无法展示本次配置来源：' + str(exc))
    if raw is not None:
        try:
            import io, zipfile
            with zipfile.ZipFile(io.BytesIO(raw.getvalue())) as archive:
                names = set(archive.namelist())
            if 'README_weather_only.txt' in names or {'station.pcp', 'pcp.cli'} <= names:
                st.error('你上传的是天气输入包，不是当地GIS资料包。请改选含DEM、土地利用、土壤及出口的ZIP；天气已由上方沿用。')
                return
        except zipfile.BadZipFile:
            st.error('当地GIS资料包不是有效ZIP，请重新选择。')
            return
    st.caption('更换资料或参数后需要重新建工程。平台不提供此处的逐项在线编辑、自动率定或准确性保证；流域范围、出口与观测站对应关系仍需核实。')
    confirmed = st.checkbox('我确认上传资料用于同一目标流域，并理解：指定的上传参数表将被导入，其余配套参数和管理配置仍采用参考库或工具默认；这些默认项未经当地观测率定', key='new_defaults_confirm')
    if st.button('检查资料并建立新工程', disabled=not (weather and weather_config and raw and config and confirmed)):
        try:
            settings = json.loads(config.getvalue().decode('utf-8-sig'))
            settings['accept_reference_parameters'] = confirmed
            ident = builder.start_build(raw.getvalue(), settings, weather.getvalue(), json.loads(weather_config.getvalue().decode('utf-8-sig')))
            st.session_state['swat_build_job'] = ident
            st.session_state.pop('swat_build_active_job', None)
        except (ValueError, OSError, KeyError, SchemaError) as exc:
            st.error(str(exc))
    if ident := st.session_state.get('swat_build_job'):
        render_build_job(ident,converted_inputs,input_mode)
    if active := st.session_state.get('swat_build_active_job'):
        from ecodata.interface.swat_run_page import _render_job
        _render_job(active)
