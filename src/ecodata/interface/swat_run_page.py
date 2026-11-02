"""A small local graphical front end to the actual SWAT+ executable."""
from datetime import date, timedelta
import hashlib
from pathlib import Path

import streamlit as st

from ecodata.core.exceptions import SchemaError
from ecodata.interface import swat_runner as runner


def _date(year, day, end=False):
    if day == 0:
        return date(year, 12, 31) if end else date(year, 1, 1)
    return date(year, 1, 1) + timedelta(days=day - 1)


@st.fragment(run_every=2)
def _render_job_live(ident):
    # Stop polling once the process finishes so downloads and charts stay stable.
    if runner.job_status(ident)['state'] != 'running':
        st.rerun()
    _render_job_contents(ident)


def _render_job(ident):
    if runner.job_status(ident)['state'] == 'running':
        _render_job_live(ident)
    else:
        _render_job_contents(ident)


def _render_job_contents(ident):
    status = runner.job_status(ident)
    labels = {"running": "运行中", "succeeded": "程序运行完成", "failed": "程序运行失败",
              "timeout": "已达到时间上限", "stopped": "已停止"}
    label = labels[status["state"]]
    if status["state"] == "succeeded":
        st.success(label + "：退出码0，且本次新生成success.fin；不代表模拟精度得到验证。")
    elif status["state"] == "running":
        st.info(label)
        if status.get('process_control_available') is False:st.caption('这是从磁盘恢复的任务记录，页面没有原进程控制句柄；继续读取原任务日志，不启动替代任务。')
        if st.button("停止本次运行", key="stop_" + ident,disabled=status.get('process_control_available') is False):
            runner.stop_job(ident)
    else:
        st.error(f"{label}；退出码：{status.get('returncode', '未启动')}。请查看下方日志。")
        if status.get("error"):
            st.error(status["error"])
    with st.expander("查看运行日志和保存位置", expanded=status["state"] in {"failed", "timeout"}):
        st.caption("本次独立工作目录：" + status["directory"])
        st.code(runner.log_tail(ident), language="text")
    if status["state"] != "running":
        try:
            st.download_button("下载本次结果与日志", runner.result_zip(ident),
                               file_name="swatplus_run_" + ident[:8] + ".zip", mime="application/zip")
        except (OSError, SchemaError) as exc:
            st.error(f"下载准备失败：{exc}；原文件保存在上述工作目录。")
    if status['state'] == 'succeeded':
        _render_charts(status['directory'], ident)


def _render_charts(directory, ident):
    from ecodata.interface import swat_results
    import pandas as pd
    import io
    import matplotlib.pyplot as plt
    import re
    choices={p.name:p for p in sorted(Path(directory).glob('*.txt')) if re.search(r'_(day|mon|yr)\.txt$',p.name)}
    if not choices:
        st.info('本次没有可绘图的日/月/年输出，请检查print.prt及跳过输出年份设置。')
        return
    preferred = 'channel_sd_day.txt' if 'channel_sd_day.txt' in choices else next(iter(choices))
    chosen = st.selectbox('结果文件', list(choices), index=list(choices).index(preferred), key='chart_file_' + ident)
    path=choices[chosen];signature=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
    @st.cache_data(show_spinner=False,max_entries=16)
    def object_index(signature):return swat_results.output_objects(signature[0])
    @st.cache_data(show_spinner=False,max_entries=16)
    def selected_output(signature,identity):return swat_results.read_output(signature[0],selected_identity=identity)
    try:
        objects=object_index(signature)
        if not objects:raise ValueError('此文件没有可识别的对象数据行。')
        identity = st.selectbox('流域或计算对象', objects, key='chart_object_' + ident)
        data=selected_output(signature,identity)
    except (ValueError,UnicodeError,OSError) as exc:
        st.error(f'{chosen}：{exc}；不自动补值。');return
    st.caption('按所选对象流式读取，校验其全部变量与日期；对象索引不代表其他对象的数值质量已验证。')
    variables = list(data['units'])
    default_variable = variables.index('flo_out') if chosen == 'channel_sd_day.txt' and 'flo_out' in variables else 0
    variable = st.selectbox('结果变量', variables, index=default_variable, key='chart_variable_' + ident)
    dates, values = swat_results.series(data, variable, identity)
    frame = pd.DataFrame({'date': dates, variable: values})
    st.caption(variable + ' · 单位：' + data['units'][variable] + '；仅展示模型输出，未评价预测精度。')
    st.caption(f'当前序列共{len(values)}个数据点；结果文件：{chosen}。对象是否为流域出口需结合连接关系核实。')
    if len(values) == 1:
        st.info('只有一个数据点（例如单年年度结果），显示散点和数值表；没有折线不代表没有输出。')
        st.scatter_chart(frame.set_index('date'))
        st.dataframe(frame, hide_index=True)
    else:
        st.line_chart(frame.set_index('date'))
    st.download_button('下载图表数据CSV', frame.to_csv(index=False).encode('utf-8-sig'), variable + '.csv', key='chart_csv_' + ident)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, values, marker='o' if len(values) == 1 else None)
    ax.set_ylabel(variable + ' [' + data['units'][variable] + ']')
    fig.autofmt_xdate()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    st.download_button('下载图表PNG', buffer.getvalue(), variable + '.png', key='chart_png_' + ident)


def render_swat_run_page(converted_weather=None, weather_inputs=None,converted_inputs=None,input_mode=None):
    st.subheader("SWAT+ · 准备工程、运行和查看结果")
    route = st.radio('你现在有什么资料？', ['已有工程，更新数据并运行', '没有工程，从原始资料开始'], key='swat_route')
    from ecodata.interface.engineering_flow import render_flow
    render_flow('swat_plus','existing' if route=='已有工程，更新数据并运行' else 'from_zero')
    if route == '没有工程，从原始资料开始':
        from ecodata.interface.swat_build_page import render_build_page
        render_build_page(weather_inputs=weather_inputs,converted_inputs=converted_inputs,input_mode=input_mode)
        return
    st.write("操作顺序：① 上传工程 → ② 检查并选择日期 → ③ 启动 → ④ 下载结果。")
    st.info("这是模型操作页面。导入已有完整TxtInOut工程，检查、调整日期、启动官方程序、查看日志并下载结果。单独天气/土壤转换包不是完整工程。")
    st.caption("数据转换仍在‘首轮专项转换 → SWAT+’。已有工程可在此接入新的天气转换ZIP，自动处理文件名和引用，保留原始工程。")
    engine_id = st.selectbox('运行程序', list(runner.ENGINE_LABELS), index=1,
                           format_func=lambda value: runner.ENGINE_LABELS[value], key='run_engine')
    if engine_id == 'gnu62':
        st.warning('此GNU构建在Robit新建工程中曾出现浮点异常；该问题尚未修复。建议使用已在两条流程核验的官方Editor Intel rev.62程序。切换程序会记录实际版本，不会自动改写数据或失败后偷偷重试。')
    try:
        engine = runner.engine_path(engine_id)
        st.caption("运行程序已校验：" + runner.ENGINE_LABELS[engine_id])
        with st.expander("程序版本和位置"):
            st.code(str(engine), language="text")
    except SchemaError as exc:
        st.error(str(exc))
        return
    st.markdown("### ① 选择工程压缩包")
    st.write("请上传一个完整工程的 ZIP。只有降水、温度等天气文件的压缩包，还不能直接运行模型。")
    upload = st.file_uploader("上传完整工程 ZIP（包含file.cio及其实际引用文件）", type=["zip"], key="swat_project_upload")
    if upload is not None:
        payload = upload.getvalue()
        fingerprint = hashlib.sha256(payload).hexdigest()
        if st.session_state.get("swat_project_fingerprint") != fingerprint:
            st.session_state["swat_project_fingerprint"] = fingerprint
            for key in ("swat_start", "swat_end", "swat_version_confirm", "attach_period_mode", "attach_start", "attach_end", "run_weather_confirm"):
                st.session_state.pop(key, None)
        try:
            initial_report = runner.check_project(runner.read_project(payload), engine_id=engine_id)
            declared_version = initial_report.get('declared_version')
            if declared_version:
                st.caption('原工程声明版本：rev.' + declared_version + '。此声明来自原文件，平台未改写。')
            version_errors = [message for message in initial_report['errors'] if '版本' in message or '仅登记' in message]
            if version_errors:
                for message in version_errors:
                    st.error(message)
                st.info('请先在上方选择匹配的运行程序，再接入天气。工程版本声明不会自动修改；没有匹配程序时应保留原工程并补充对应版本环境。')
                return
            if converted_inputs is not None:
                from ecodata.models.swat_inputs_project import attach_inputs
                st.info('已接收上方本次参数转换ZIP；仅接入同名目标表，保留对象标识与顺序，不推测新文件名或改写file.cio。')
                accepted_inputs=st.checkbox('确认本次参数属于该流域及对象；接入后仍须原生运行验证，基本检查不代表所有深层引用有效',key='swat_parameter_confirm')
                if not accepted_inputs:return
                attached,attachment=attach_inputs(runner.read_project(payload),converted_inputs,input_mode)
                from io import BytesIO
                import zipfile
                buffer=BytesIO()
                with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                    for name,raw in attached.items():z.writestr(name,raw)
                payload=buffer.getvalue()
                st.json(attachment)
                st.download_button('保存接好本次参数的工程',payload,'swatplus_updated_parameters.zip')
            from ecodata.models.swat_weather_project import project_stations, read_weather_zip, attach_weather, inspect_weather_project
            if converted_weather is not None:
                from io import BytesIO
                updated_weather = BytesIO(converted_weather)
                st.info('已接收上方本次转换的日天气ZIP，无需下载后重新上传。')
            else:
                updated_weather = st.file_uploader('可选：上传刚从平台下载的日天气转换ZIP', type=['zip'], key='run_weather_zip')
            if updated_weather is not None:
                inspection = inspect_weather_project(payload)
                with st.expander('工程识别结果：实际引用、站点与支持范围'):
                    st.json(inspection)
                dependencies = inspection['weather_dependencies']
                st.caption('下面统计连接表中明确写出的天气站引用；不会按距离重新分配站点。')
                st.dataframe([{'天气站': name, 'WGN统计站': info['wgn'], '显式关联对象数': info['object_count']}
                              for name, info in dependencies['stations'].items()], hide_index=True)
                for warning in dependencies['warnings']:
                    st.warning(warning)
                for error in dependencies['errors']:
                    st.error(error)
                if dependencies['errors']:
                    st.info('工程天气依赖检查未通过；请修正原始资料或提供正确参数，平台不会自动补齐。')
                    return
                station = st.selectbox('将新天气用于哪个现有站点？', project_stations(payload), index=None, key='run_target_station')
                source_dates = runner.check_project(runner.read_project(payload), engine_id=engine_id)['dates']
                if source_dates is None:
                    st.error('无法读取原工程日期，不能选择本次模拟期。')
                    return
                original_start = _date(source_dates['yrc_start'], source_dates['day_start'])
                original_end = _date(source_dates['yrc_end'], source_dates['day_end'], True)
                st.write(f'原工程模拟期：{original_start} 至 {original_end}')
                weather_files = read_weather_zip(updated_weather.getvalue())
                from ecodata.models.swat_weather_project import _records
                weather_dates = _records(weather_files['station.pcp'], 'pcp')
                st.write(f'本次降水覆盖期：{weather_dates[0]} 至 {weather_dates[-1]}；接入时还会核对温度等各变量日期。')
                period_mode = st.radio('本次模拟日期', ['保留原工程模拟期', '自行选择本次模拟期'], key='attach_period_mode')
                simulation_period = None
                if period_mode == '自行选择本次模拟期':
                    attach_start = st.date_input('本次模拟开始日期', value=original_start,
                        min_value=date(1,1,1), max_value=date(9999,12,31), key='attach_start')
                    attach_end = st.date_input('本次模拟结束日期', value=original_end,
                        min_value=date(1,1,1), max_value=date(9999,12,31), key='attach_end')
                    simulation_period = (attach_start, attach_end)
                st.caption('仅按明确选择调整工程副本的时间文件，原始上传工程不变；其余参数不变，不补天气。缩短模拟期也可能影响预热与输出安排，仍需核对其他时序输入。')
                confirmed = st.checkbox('确认新天气适用于该站点，并按上述明确选择的模拟期接入工程副本', key='run_weather_confirm')
                if not (station and confirmed):
                    st.info('请明确选择站点并确认适用性后再检查、运行。')
                    return
                payload, attachment = attach_weather(payload, weather_files, station, simulation_period=simulation_period)
                st.success('天气已自动接入本次工程副本，无需手动改名或复制。')
                rebuilt = attachment['index_repairs']['reconstructed_indices']
                if rebuilt:
                    st.info('按工程原站点引用重建缺失索引：'+'、'.join(item['file'] for item in rebuilt)+'。仅登记已有文件，未补造观测值或改变站点分配；详情见接入清单。')
                with st.expander('查看接入清单：改变与保留的文件（含SHA256）'):
                    st.json(attachment)
                import json
                st.download_button('下载天气接入核查清单JSON', json.dumps(attachment, ensure_ascii=False, indent=2), 'weather_attachment_report.json')
                st.download_button('保存接好天气的工程', payload, 'swatplus_updated_project.zip')
            files = runner.read_project(payload)
            report = runner.check_project(files, engine_id=engine_id)
            st.markdown("### ② 检查工程、选择日期")
            from ecodata.interface.engineering_completeness import swat,render as render_completeness
            render_completeness(report.get('completeness') or swat(files,report),'swat')
            st.write(f"工程内 {len(files)} 个文件；所选程序：{runner.ENGINE_LABELS[engine_id]}；工程声明版本：{report.get('declared_major') or '未声明'}。")
            for warning in report["warnings"]:
                st.warning(warning)
            for error in report["errors"]:
                st.error(error)
            if report["errors"]:
                st.info("请按上方具体原因处理：版本不匹配时选择与原工程匹配的运行程序；缺少文件时补充原工程资料。平台不会改写版本声明或编造参数来通过检查。")
            if not report["errors"]:
                st.success("文件引用和日期基本检查通过；仍需实际运行判断模型是否接受。")
                dates = report["dates"]
                start = st.date_input("开始日期", value=_date(dates["yrc_start"], dates["day_start"]),
                                      min_value=date(1, 1, 1), max_value=date(9999, 12, 31), key="swat_start")
                end = st.date_input("结束日期", value=_date(dates["yrc_end"], dates["day_end"], True),
                                    min_value=date(1, 1, 1), max_value=date(9999, 12, 31), key="swat_end")
                st.caption("仅修改本次副本中的时间文件；气象和其他时序资料必须覆盖所选时段，页面不会补齐。输出项目由已有print.prt决定。")
                accepted = st.checkbox("我的工程适用于所选SWAT+版本，且输入数据覆盖所选时段", key="swat_version_confirm")
                charts = st.checkbox('开启逐日水量与河道图表，显示所选全部日期', value=False, key='run_enable_charts')
                with st.expander("运行时间限制（默认20分钟，可调整）"):
                    minutes = st.number_input("最长运行时间（分钟）", min_value=1, max_value=120, value=20, step=1)
                st.markdown("### ③ 启动程序")
                if st.button("启动 SWAT+", disabled=not accepted):
                    if updated_weather is not None:
                        weather_start, weather_end = map(date.fromisoformat, attachment['weather_period'])
                        if start < weather_start or end > weather_end:
                            raise SchemaError(f'所选日期超出本次新天气覆盖范围：{weather_start}至{weather_end}；不会补值或自动调整日期。')
                    prepared = runner.set_dates(files, start, end, engine_id=engine_id)
                    if charts:
                        from ecodata.interface.swat_results import enable_charts
                        prepared, _ = enable_charts(prepared, engine_id=engine_id)
                    ident = runner.start_job(prepared, runner.ROOT / "runs" / "swat_plus", timeout=int(minutes) * 60, engine_id=engine_id)
                    st.session_state["swat_active_job"] = ident
        except (SchemaError, OSError, ValueError) as exc:
            st.error(str(exc))
    recent = runner.recent_jobs() if converted_weather is None and converted_inputs is None else []
    if recent:
        chosen = st.selectbox("查看本次平台启动后的运行记录", [j[0] for j in recent],
                              format_func=lambda value: value[:8] + " · " + dict(recent)[value])
        st.session_state["swat_active_job"] = chosen
    if ident := st.session_state.get("swat_active_job"):
        st.markdown("### ④ 查看进度、下载结果")
        _render_job(ident)
    with st.expander("支持范围与依据"):
        st.write("本地Windows、已登记的60.5.7/61/62程序、单个平铺工程、逐日step=0、一次一个任务。不同版本输入结构可能不同；60程序仅登记60.5.7，页面拦截声明版本不匹配，不自动迁移。此分支接入已有工程；原始资料建工程请选另一条流程。不保证任意版本工程兼容，运行记录保存实际程序标识与SHA256。")
        st.markdown("[官方62.0.0代码与参考工程](https://github.com/swat-model/swatplus/tree/62.0.0) · [时间配置说明](https://swatplus.gitbook.io/io-docs/introduction-1/simulation-settings/time.sim)")
