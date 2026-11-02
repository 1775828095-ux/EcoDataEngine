import streamlit as st
from ecodata.interface.camx_runner import start,status,directory

def render_camx_run_page(converted_zip=None,mode=None):
    from ecodata.interface.independent_cases import render
    if render('camx', converted_zip, mode):
        return
    st.title('CAMx 工程运行')
    st.info('默认登记案例：2016-06-10 UTC 00:00–00:05，36 km 主网格与12 km嵌套网格；原始GFS新气象选项生成新的36 km单网格场景。运行输送、扩散和化学积分；仅验证五分钟流程，不代表长期模拟或预测精度。')
    st.write('更新已有工程：使用登记工程控制模板，将平台转换的光解表接入原生 CAMx。重新组装：重新解码官方光解成品表，平台转换后，从登记参数清单生成控制文件并关联配套输入。')
    st.warning('默认两条路线复用官方配套气象和化学资料；新建路线可明确选择原始GFS重新构建气象，形成新的36km单网格20层场景。化学初边界、排放及雪龄配套仍复用；光解CSV来自成品表解码，非独立辐射观测。不等于完整原始清单从零，不支持任意用户新区域。')
    st.caption('沿用原案例CB6r5_CF3_COMPLX化学参数和Ignore_Emission_Dates=true；默认双网格，新GFS气象选项为显式单网格。不进行物种补零、清单分配或日期改造。结果为五分钟平均浓度，非小时平均；O₃从ppmv转为ppbv。')
    st.markdown('来源：[CAMx 官方下载页](https://www.camx.com/download/)，本机已登记 CAMx 7.40 官方测试算例及配套资料。此案例未使用私人客户资料。')
    if converted_zip is not None:
        st.info('已接收上方本次转换ZIP。当前仅接收与已核验登记案例完全一致的变量、单位、网格、时段和数值；不同区域或新数值会明确拒绝，不会换回示例数据。')
        st.caption('排放仅转换NO/PEC，其余52个原输入物种保留；初始/边界使用转换的48个物种，保持原案例缺物种行为。')
    ident=st.session_state.get('camx_current_job','') if converted_zip is not None else st.query_params.get('camx_job','');previous=None
    if ident:
        try:previous=status(ident)
        except (ValueError,OSError):pass
    route=st.radio('工程路线',['更新已有工程','用配套资料重新组装登记工程'],index=1 if previous and previous['route']=='rebuild' else 0,key='camx_route')
    new_meteorology=False;meteorology_project='';raw_gfs_files=None
    if route=='更新已有工程':
        meteorology_project=st.text_input('接入本机已验证的新建气象工程任务号（可选）',key='camx_meteorology_project').strip()
        if meteorology_project:st.caption('仅接收本机成功的原始GFS新建任务；核对气象文件哈希，再接入当前转换。生成单网格结果，不回退原双网格示例。')
    if route!='更新已有工程':
        new_meteorology=st.checkbox('从原始GFS重新新建气象工程（新单网格场景，化学资料复用）',key='camx_new_meteorology')
        if new_meteorology:
            st.info('实际运行WPS、real、十二小时WRF及WRFCAMx；数值步长90秒，前六小时起步。2016-06-10 06时采用00时起报的真实六小时预报，非06时分析。初边界按AGL层中心线性插值，不外推、不补零，不宣称柱质量守恒或原双网格复现。')
            if st.checkbox('上传原始GFS文件（限定上述登记时段）',key='camx_upload_gfs'):
                uploaded=st.file_uploader('三份原始GFS GRIB2',type=['grb2','grib2'],accept_multiple_files=True,key='camx_raw_gfs')
                raw_gfs_files={f.name:f.getvalue() for f in uploaded}
                if len(raw_gfs_files)!=len(uploaded):st.error('GFS文件名重复');return
                st.caption('平台保留原字节并校验全部三份内容；不接受换名替代、不回退本机文件。当前仍为固定网格和时段，化学及排放配套复用。')
    from ecodata.interface.engineering_flow import render_flow
    render_flow('camx','existing' if route=='更新已有工程' else 'rebuild')
    if new_meteorology:st.graphviz_chart('digraph { rankdir=LR; a[label="原始GFS＋登记地理库"]; b[label="WPS→real→WRF十二小时"]; c[label="WRFCAMx＋网格核查"]; d[label="当前转换＋配套化学ICBC显式插值"]; e[label="单网格CAMx五分钟→出图"]; a->b->c->d->e; }')
    if st.button('运行 CAMx 登记案例',type='primary'):
        try:
            if meteorology_project:ident=start('existing',converted_zip,mode,meteorology_project=meteorology_project)
            elif raw_gfs_files is not None:ident=start('rebuild',converted_zip,mode,True,raw_gfs_files=raw_gfs_files)
            else:ident=start('existing' if route=='更新已有工程' else 'rebuild',converted_zip,mode,True) if new_meteorology else start('existing' if route=='更新已有工程' else 'rebuild',converted_zip,mode)
            if converted_zip is not None:st.session_state['camx_current_job']=ident
            else:st.query_params['camx_job']=ident
        except Exception as exc:st.error(str(exc));return
    if not ident:return
    @st.fragment(run_every=5)
    def result_view():
        try:record=status(ident)
        except (OSError,ValueError) as exc:st.error(str(exc));return
        st.write('任务：'+ident+'；实际运行路线：'+record['route'])
        if record['status']=='running':
            st.info('正在校验输入并执行已选择的原生阶段，结果区每5秒更新。')
            if record.get('new_meteorology_requested'):
                stages=directory(ident)/'raw_meteorology/stages.json'
                if stages.is_file():
                    try:
                        import json
                        st.dataframe(json.loads(stages.read_text()),hide_index=True)
                    except (OSError,ValueError):st.caption('阶段记录正在更新。')
            return
        if record['status']!='success':st.error(record.get('error','运行失败'));return
        folder=directory(ident)
        st.success('原生CAMx完成至UTC 00:05；'+('新单网格' if record.get('new_meteorology_requested') or record.get('meteorology_project') else '原双网格')+'平均浓度已读取。')
        st.image(str(folder/'O3_two_grids.png'),caption='2016-06-10 UTC 00:00–00:05 平均 O₃；Lambert 投影坐标，统一色标。')
        st.download_button('下载 CAMx 图表PNG',(folder/'O3_two_grids.png').read_bytes(),'CAMx_O3.png','image/png',key='camx_result_png')
        for grid in record['grids']:
            n=grid['grid'];st.write(f"网格 {n}：{grid['shape']}；O₃ 最大值 {grid['max_O3_ppbv']:.6f} ppbv")
            st.download_button(f'下载网格 {n} O₃ CSV',(folder/f'O3_grid{n}.csv').read_bytes(),f'CAMx_O3_grid{n}.csv','text/csv',key=f'camx_csv_{n}')
        st.download_button('下载 CAMx 运行记录',(folder/'result.json').read_bytes(),'camx_result.json','application/json')
        with st.expander('原生控制与配套输入清单'):st.json(record)
    result_view()
