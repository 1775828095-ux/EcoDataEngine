import streamlit as st
from ecodata.interface.cmaq_runner import start,status,directory

def render_cmaq_run_page(converted_zip=None,mode=None):
    from ecodata.interface.independent_cases import render
    if render('cmaq', converted_zip, mode):
        return
    st.title('CMAQ 工程运行')
    st.info('登记案例：CMAQ 5.5，CB6R5/AE7，2018-07-01 UTC 00:00–00:05；12NE3 网格100×105、35层。验证五分钟实际积分，图为00:05第一模型层瞬时O₃。')
    st.write('已有工程更新：平台重新转换登记的NO数据，接入已有互补排放。配套资料重建：从官方成品排放重新解码NO，并重新生成不含NO的互补文件，再关联控制脚本运行。两条路线均逐值核对NO和其余物种，防止重复或遗漏。')
    st.warning('不是独立原始排放清单验证或完整从零建模。默认复用MCIP气象；新建路线可明确选择原始GFS重新构建气象。化学初边界、其余排放及化学资料仍复用同案例成品，不支持任意用户新区域。')
    st.caption('沿用原案例 IOAPI_CHECK_HEADERS=N、CTM_EMISCHK=N，不宣称额外严格头检查或所有排放替代物检查通过；保留原生日志警告。原始地面排放第一层分配明确，未推断垂直分配；NO为moles/s。O₃ ppmV转换为ppbv，非小时平均。')
    st.markdown('来源：[CMAS 官方同案例资料目录](https://cmas-cmaq.s3.amazonaws.com/CMAQv5.4_2018_12NE3_Benchmark_2Day_Input/2018_12NE3/)。CMAQ5.5官方CB6R5脚本引用这套v5.4基准输入。本机仅登记前三个输入时刻用于短跑，未使用私人客户资料。')
    if converted_zip is not None:
        st.info('已接收上方本次转换ZIP；当前仅允许与已核验登记案例完全一致的字段、单位、网格、时段和数值。不匹配会拒绝，不用登记数据冒充本次输入。')
        st.caption('排放接入NO并保留其余123物种；边界接入O3并保留原文件其余物种。剖面仅登记同案例35层O3平均敏感性方案：先运行ICON/BCON，再接入O3、保留其余224物种。旧44层剖面会拒绝。')
        if mode in (None,'bcon_profile'):
            st.warning('本次剖面方案明确改变O3空间分布：同案例初始时刻逐层水平平均，BCON静态剖面用于原时标；不是独立观测或原空间场等价验证。')
    ident=st.session_state.get('cmaq_current_job','') if converted_zip is not None else st.query_params.get('cmaq_job','');previous=None
    if ident:
        try:previous=status(ident)
        except (OSError,ValueError):pass
    route=st.radio('工程路线',['更新已有工程','用配套资料重建登记工程'],index=1 if previous and previous['route']=='rebuild' else 0,key='cmaq_route')
    new_meteorology=False
    if route!='更新已有工程':
        new_meteorology=st.checkbox('从原始GFS重新新建气象工程（化学资料仍复用）',key='cmaq_new_meteorology')
        if new_meteorology:st.info('重新运行geogrid、ungrib、metgrid、real、七小时WRF及MCIP，再接入当前转换并运行CCTM五分钟。固定2018-07-01/12NE3/35层；化学初边界和其他排放仍使用同案例成品，不等于全部原始清单从零构建。')
    from ecodata.interface.engineering_flow import render_flow
    render_flow('cmaq','existing' if route=='更新已有工程' else 'rebuild',mode=('bcon_profile' if mode=='bcon_profile' else 'icon_profile') if converted_zip is not None and mode in (None,'bcon_profile') else None)
    if new_meteorology:st.graphviz_chart('digraph { rankdir=LR; a[label="原始GFS＋登记地理库"]; b[label="geogrid→ungrib→metgrid→real→WRF"]; c[label="MCIP＋网格/时段核对"]; d[label="当前转换＋同例化学资料"]; e[label="CCTM→实际结果图"]; a->b->c->d->e; }')
    if st.button('运行 CMAQ 登记案例',type='primary'):
        try:
            ident=start('existing' if route=='更新已有工程' else 'rebuild',converted_zip,mode,True) if new_meteorology else start('existing' if route=='更新已有工程' else 'rebuild',converted_zip,mode)
            if converted_zip is not None:st.session_state['cmaq_current_job']=ident
            else:st.query_params['cmaq_job']=ident
        except Exception as exc:st.error(str(exc));return
    if not ident:return
    @st.fragment(run_every=5)
    def view():
        try:record=status(ident)
        except (OSError,ValueError) as exc:st.error(str(exc));return
        st.write('任务：'+ident+'；实际运行路线：'+record['route'])
        if record['status']=='running':
            st.info('正在校验输入、执行已选择的原生阶段；结果区每5秒更新。')
            if record.get('new_meteorology_requested'):
                stagefile=directory(ident)/'raw_meteorology/stages.json'
                if stagefile.is_file():
                    try:
                        import json
                        st.dataframe(json.loads(stagefile.read_text()),hide_index=True)
                    except (OSError,ValueError):st.caption('阶段记录正在写入，下次刷新读取。')
            return
        if record['status']!='success':st.error(record.get('error','运行失败'));return
        folder=directory(ident)
        st.success(f"原生CCTM完成至UTC 00:05；最终场包含{record['species']}个物种。")
        st.metric('第一模型层 O₃ 最大瞬时浓度',f"{record['max_O3_ppbv']:.6f} ppbv")
        st.image(str(folder/'O3_surface.png'),caption='00:05 UTC瞬时O₃，第一模型层；Lambert公里坐标。')
        st.download_button('下载 CMAQ O₃ 网格CSV',(folder/'O3_surface.csv').read_bytes(),'CMAQ_O3_surface.csv','text/csv')
        st.download_button('下载 CMAQ 运行记录',(folder/'result.json').read_bytes(),'cmaq_result.json','application/json')
        with st.expander('转换、复用资料与原生记录'):st.json(record)
    view()
