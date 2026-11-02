"""Shared raw-input conversion step for existing engineering pages."""
import hashlib
import io
import json
import zipfile


MODES={'cmaq':{'gridded_emissions'},
       'camx':{'gridded_emissions','initial_concentrations','boundary_concentrations'}}


def convert_input(model,content,config):
    from ecodata.round1 import convert
    if not isinstance(config,dict):raise ValueError('数据转换配置JSON必须为对象')
    if model not in MODES or config.get('mode') not in MODES[model]:
        raise ValueError('当前已有工程入口不支持此转换类型；不会忽略或回退原数据')
    files,report=convert(model,content,config)
    report=dict(report,source_sha256=hashlib.sha256(content).hexdigest(),
                config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():
            if name!='conversion_report.json':z.writestr(name,raw)
        z.writestr('conversion_report.json',json.dumps(report,ensure_ascii=False,indent=2))
    return out.getvalue(),report


def render(model,converted_zip=None,mode=None):
    """Return (conversion, mode, ready); requested incomplete updates block runs."""
    import streamlit as st
    from ecodata.interface.input_result_binding import bind_result
    if converted_zip is not None:
        st.info('已接收上方本次转换结果。下一步明确选择工程中的更新目标。')
        return converted_zip,mode,True
    update=st.checkbox('接入一份新的原始数据（由平台转换）',key=model+'_engineering_update')
    if not update:
        st.caption('当前选择：原工程复验，不更新输入。勾选上项可接入新数据。')
        return None,None,True
    st.caption('上传原始数据和原字段、单位、网格配置。使用与首轮专项转换相同的转换器；转换成功后再选择更新目标，不自动修改工程科学设置。')
    source=st.file_uploader('本次新原始数据',type=['nc','csv','zip'],key=model+'_engineering_source')
    config=st.file_uploader('本次数据转换配置JSON',type=['json'],key=model+'_engineering_config')
    state_key=model+'_engineering_conversion'
    bind_result(st.session_state,state_key,model+'_engineering_conversion_binding',
                source.getvalue() if source else b'',config.getvalue() if config else b'')
    if st.button('转换本次数据',disabled=source is None or config is None,key=model+'_engineering_convert'):
        st.session_state.pop(state_key,None)
        try:
            cfg=json.loads(config.getvalue().decode('utf-8-sig'))
            payload,report=convert_input(model,source.getvalue(),cfg)
            st.session_state[state_key]=(payload,cfg['mode'],report)
        except Exception as exc:st.error(str(exc))
    result=st.session_state.get(state_key)
    if result is None:
        st.info('请先完成本次转换；不会使用工程旧输入代替本次更新。')
        return None,None,False
    payload,mode,report=result
    st.success('本次数据已转换。请选择工程中的对应输入，随后运行副本。')
    st.download_button('下载本次转换数据ZIP',payload,model+'_current_conversion.zip',key=model+'_engineering_zip')
    with st.expander('本次转换记录'):st.json(report)
    return payload,mode,True
