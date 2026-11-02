"""Scoped first-round input converters, sharing the CLI implementation."""
import hashlib
import io
import json
import zipfile
import streamlit as st
from ecodata.round1 import convert
from ecodata.models.swat_weather_project import project_stations, attach_weather


def render_weather_attachment(files):
    st.subheader('接入已有SWAT+工程（可选）')
    st.write('上传已有工程，选择要使用本次天气的站点。平台自动命名、复制天气文件并更新引用；原工程ZIP不变。无需手动改名或复制。')
    st.caption('仅支持SWAT+62兼容的标准天气站表、标准变量索引和逐日工程。保留运行日期、WGN及其他站点；不会补齐完整流域或替您判断天气的地理适用性。')
    project = st.file_uploader('上传已有流域工程ZIP（不是天气转换ZIP）', type=['zip'], key='swat_weather_project')
    if not project:
        return
    try:
        stations = project_stations(project.getvalue())
        station = st.selectbox('把本次天气用于工程中的哪个站点？', stations, index=None,
                               placeholder='请选择与原始天气对应的站点', key='swat_attach_station')
        confirmed = st.checkbox('我确认所选工程站点适用这份天气，保留工程原运行日期', key='swat_attach_confirm')
        fingerprint = hashlib.sha256(project.getvalue() + str(station).encode() + b''.join(files.values())).hexdigest()
        if st.session_state.get('swat_attach_fingerprint') != fingerprint:
            st.session_state.pop('swat_attach_result', None)
        st.session_state['swat_attach_fingerprint'] = fingerprint
        if st.button('自动接入并检查工程', disabled=not(station and confirmed)):
            st.session_state.pop('swat_attach_result', None)
            try:
                st.session_state['swat_attach_result'] = attach_weather(project.getvalue(), files, station)
            except Exception as exc:
                st.error(f'工程接入被拒绝：{exc}')
        if confirmed and (result := st.session_state.get('swat_attach_result')):
            payload, report = result
            st.success('天气已接入工程副本；天气引用和日期覆盖检查通过。尚未运行模型，也不代表完整工程全部有效。')
            st.json(report)
            st.download_button('下载接好天气的工程ZIP（供SWAT+桌面助手使用）', payload,
                               'swat_plus_project_with_weather.zip', 'application/zip')
    except Exception as exc:
        st.error(f'无法读取工程：{exc}')

INPUT_SCOPE = {
    'calpuff': '生成 SURF.DAT 和转换报告，供 CALMET 5.8.5 的地面小时观测输入步骤使用。不是 CALMET.DAT，也不代表 CALPUFF 完整气象、排放与运行配置已备齐。',
    'swat_plus': '分层土壤模式生成soils.sol，供62.0.0土壤数据库读取使用，不生成HRU或估算土壤参数。日天气模式生成 station.pcp、station.tem，可选 station.slr、station.hmd、station.wnd及实际索引；WGN月统计模式生成装有站点和12个月统计值的 weather-wgn.cli。供 SWAT+ 62.0.0 天气资料接入使用，不生成随机天气，不计算气候统计。不提供file.cio，不代表完整流域工程已完成。',
    'wrf': '生成 SST_YYYY-MM-DD_HH.wps，供 WPS 4.6 的 metgrid 海温补充步骤使用。支持单快照、连续逐小时、连续逐日；逐日保留原始UTC时刻，不做时间插值或复制，另附 README_daily_SST.txt。逐日文件仅覆盖所给每日时刻；更密的metgrid时刻须另提供海温或明确有依据的时间处理方法。单快照可按配置生成 WPS_metgrid_SST_constant.zip。不是 met_em 或 wrfinput，不代表完整 WRF 输入链已完成。',
    'camx': '生成 photolysis.CAMx7.40.CB6r5.txt 标准化 ASCII 光解率查找表及报告，供 CAMx 7.40 CB6r5 的 READPHT 读取步骤使用。输入需已有光解率表；不计算 TUV 光解率，不代表排放、气象、初始场和边界条件已完成。',
    'cmaq': '生成 IC_PROFILE_CMAQ55.csv 及报告，供 CMAQ 5.5 ICON 的 profile 模式使用（CB6R3M_AE7_KMTBR）。需 ICON 配合 GRIDDESC 与 MET_CRO_3D_FIN 继续生成初始场；不是 CCTM 初始场或可运行的完整 CMAQ 工程。',
}
INPUT_SCOPE['calpuff'] += ' 高空模式生成逐站UPn.DAT 2.0，供CALMET5.8.5高空观测读取步骤；需要真实探测层，不产生CALMET气象场。'
INPUT_SCOPE['calpuff'] += ' 独立小时降水模式生成PRECIP.DAT 2.0（NONE表头），只转换已给完整各站小时结束时刻的小时雨量/平均率，不拆分日雨量，不生成控制文件。'
INPUT_SCOPE['swat_plus'] += ' HRU属性模式生成hru-data.hru，可选topography.hyd；只写出用户已有属性并核对所给数据库名称，不划分HRU、不生成流域连接。'
INPUT_SCOPE['swat_plus'] += ' 土地利用配置模式生成62.0.0 landuse.lum植被非城市数据库；需已有植物/管理/CN/糙率等参数数据库，不从分类栅格推导参数，城市及工程措施非null暂不支持。'
INPUT_SCOPE['swat_plus'] += ' 含水层参数模式生成62.0.0 aquifer.aqu，只写用户提供的地下水物理参数并核对初始资料名称；不生成aquifer.con或流域连接。'
INPUT_SCOPE['swat_plus'] += ' 显式对象连接模式从用户给定节点及出流关系生成hru.con/aquifer.con/rout_unit.con/chandeg.con，核对直接引用，不计算流域拓扑，不代表完整工程。'
INPUT_SCOPE['wrf'] += ' 多变量模式写等压TT/UU/VV/RH/GHT、已支持地面字段及明确层深的土壤温湿度WPS intermediate-v5；不推算缺失气象或生成geo_em。'
INPUT_SCOPE['camx'] += ' 网格排放模式生成emissions.CAMx7.40.nc，要求已分种且每格每小时完整的排放率；不做清单空间分配、化学拆分或时间分配。'
INPUT_SCOPE['camx'] += ' 排放/初始/边界浓度配置mechanism可选CB6r5_CF3_SIMPLE_NoEvap或CB6r5_CF3_COMPLX，须与工程CHEMPARAM一致；二者使用相同物种标识，但平台不进行化学机制互换。'
INPUT_SCOPE['cmaq'] += ' 网格排放模式生成emissions.CMAQ55.nc与对应GRIDDESC（GRDDED3地面层，配置明确选择cb6r3_ae7_aq或cb6r5_ae7_aq，分别核对官方物种表；不自动转换机制）。逐小时保留原始时刻；静态清单需明确平均期，不生成小时变化。'
INPUT_SCOPE['cmaq'] += ' BCON剖面模式生成BC_PROFILE_CMAQ55.csv，供CMAQ5.5 BCON/profile的BC_PROFILE使用；静态垂直浓度不含时间和地理网格，不是BCON生成的边界场NetCDF。'

MODES = {
 'calpuff': [('地面小时气象',None,['csv']),('高空探测','upper_air',['csv']),('独立小时降水','precipitation',['csv']),('地理参数网格','geography',['nc']),('离散受体接入','receptors',['zip']),('常量点源排放','constant_points',['zip'])],
 'swat_plus': [('日天气',None,['nc']),('WGN月统计','wgn',['csv']),('分层土壤','soil',['csv','xlsx','parquet']),('HRU属性与地形参数','hru_attributes',['zip']),('土地利用配置数据库','landuse_database',['zip']),('含水层参数数据库','aquifer_parameters',['zip']),('显式对象连接','connections',['zip']),('土地利用空间关联','spatial_landuse',['zip']),('固定日历施肥日程','management_schedule',['zip'])],
 'wrf': [('海温SST',None,['nc']),('多变量气象场','pressure_fields',['nc']),('地理栅格瓦片','geogrid_tiles',['nc']),('已有WPS场一致组包','input_package',['zip'])],
 'camx': [('光解率表',None,['csv']),('网格排放','gridded_emissions',['csv']),('三维初始浓度','initial_concentrations',['csv']),('逐小时边界浓度','boundary_concentrations',['csv'])],
 'cmaq': [('ICON化学剖面',None,['csv']),('网格排放','gridded_emissions',['nc']),('BCON静态边界剖面','bcon_profile',['csv']),('显式外圈边界浓度','boundary_concentrations',['nc'])],
}


def render_round1_page():
    st.subheader('首轮专项转换')
    st.info('先读取、映射、校验并写出专用文件；成功后可在下载区下方继续工程流程。只有后续原生任务完成才算该案例运行通过，一个输入类型通过不代表完整工程。')
    model=st.selectbox('专项模型',['calpuff','swat','swat_plus','wrf','camx','cmaq'],format_func=lambda x:{'calpuff':'CALPUFF流程：CALMET5.8.5 / CALPUFF5.8.4受体','swat':'传统SWAT 2012：单站日降水与温度（非SWAT+）','swat_plus':'SWAT+62.0.0：专用输入转换','wrf':'WRF流程：WPS4.6 intermediate-v5','camx':'CAMx7.40：光解率 / 排放 / 初始浓度','cmaq':'CMAQ5.5：剖面 / 排放 / 边界浓度'}[x])
    if model in INPUT_SCOPE:st.info('当前输出与使用范围：'+INPUT_SCOPE[model])
    st.caption('源字段、单位、日期、空间范围及工程引用必须明确；缺项不填零，不补造物理变量或拓扑。完整输入类别覆盖清单见docs/conversion_coverage_status.md。')
    options=MODES.get(model,[('传统SWAT天气',None,['csv'])])
    choice=st.radio({'calpuff':'CALMET转换类型','swat_plus':'SWAT+转换类型','wrf':'WPS转换类型','camx':'CAMx转换类型','cmaq':'CMAQ转换类型'}.get(model,'转换类型'),[o[0] for o in options],key=model+'_mode') if len(options)>1 else options[0][0]
    selected=next(o for o in options if o[0]==choice);mode=selected[1]
    if model=='swat_plus' and mode=='wgn':st.info('weather-wgn.cli装有站点及12个月统计，不是空索引。需多年统计及14项完整字段；可选weather-sta.cli只核对WGN名称，不修改。')
    if model=='swat_plus' and mode=='hru_attributes':st.info('上传ZIP：HRU源CSV和真实引用数据库；JSON指定table、fields、references，可选topography源参数表。必须已有HRU划分与身份，不生成hru.con、file.cio、河网或管理参数。')
    if model=='swat_plus' and mode=='landuse_database':st.info('landuse.lum保存14项名称/引用，无时间或网格字段。只支持植被非城市profile，ZIP需源CSV及实际植物、管理、CN、保育、糙率数据库；缺引用明确报错，不猜测地类参数，不生成file.cio。')
    if model=='swat_plus' and mode=='aquifer_parameters':st.info('上传ZIP：已确定的含水层参数CSV和现有initial.aqu；配置18项字段、15项单位及直接引用。生成aquifer.aqu，不生成含水层连接aquifer.con或file.cio，不推导地下水参数。')
    if model=='swat_plus' and mode=='management_schedule':st.info('生成SWAT+62 management.sch固定月日fert/manu操作日程及用户给定材料/施用数据库。ZIP含操作CSV及真实引用；显式操作映射、施用量单位、顺序，热量触发必须显式0。无自动规则、轮作、灌溉和其他管理操作，不生成file.cio；可核对已有landuse.lum管理引用。')
    if model=='swat_plus' and mode=='management_schedule':st.caption('显式profile=calendar_fert_manure_preserve_automatic、automatic_operations=preserve_source时，可保留原管理文件的自动规则顺序及完整lum.dtl字节；需提供management_file和references.automatic_decision。只保留和核对直接名称，不生成或认证自动规则。')
    if model=='swat_plus' and mode=='spatial_landuse':st.info('上传ZIP：已有HRU属性表、HRU及地类多边形GeoJSON和真实引用数据库。仅支持同一米制投影、每HRU完整单一地类覆盖，按配置写hru-data.hru的lu_mgt。覆盖缺口、混合地类或缺引用报错；不取多数地类、不划分HRU、不生成file.cio。')
    if model=='swat_plus' and mode=='connections':st.info('上传对象CSV、出流CSV及已有属性数据库/天气站表ZIP。可写hru.con、aquifer.con、rout_unit.con、chandeg.con；所有连线必须由用户提供，不划分流域。当前profile需cst/ovfl/rule=0，每类水文流出比例合计1；不生成file.cio。')
    if model=='calpuff' and mode=='constant_points':st.info('CALPUFF5.8.4 group13：上传点源CSV及已有实际控制文件ZIP。必须给烟囱/坐标/每个控制物种排放率（包括源显式0）；写入IPTU=1 g/s。仅常量、无下洗/平台profile，不生成年度分配、变排放、其它控制和完整工程；未运行CALPUFF。')
    if model=='calpuff' and mode=='receptors':st.info('CALPUFF5.8.4离散受体：上传受体CSV和用户已有控制文件ZIP；仅替换NREC及group17b四值X记录。检查UTM/基准与网格，xy输出km，高度输出m。其它气象/排放/控制及外部文件引用未验证，不能据此认为完整工程可运行。')
    if model=='calpuff' and mode=='geography':st.info('生成CALMET5.8.5 GEO.DAT；支持WGS84 UTM及NAD27北半球UTM，原样保留声明基准，不隐式转换。需八个完整静态物理字段及土地分类码，不估算粗糙度或热通量。已有官方成品地理反解再编码的4小时原生接收证据；不是本次用户数据运行，也不是独立原始GIS验证。')
    if model=='calpuff' and mode=='constant_points':st.caption('初始扩散参数默认要求源显式提供；仅配置initial_sigmas_policy=preserve_source_omission时，允许原控制确实省略SIGYZI且源名、数量、顺序均不变，输出继续省略并记录原生默认行为，不补造数值。官方演示点源反解再编码已有4小时接收证据，不等于真实工业排放验收。')
    if model=='calpuff' and mode=='upper_air':st.info('UPn.DAT：供CALMET 5.8.5高空观测读取，不是完整CALPUFF工程。默认拒绝缺测；配置missing_policy="preserve"可保留温度/风的官方缺测码，不补值、不删层。压力、高度、站点、时间及层号仍必须完整；近地面和声明顶层须有有效温度及完整风。平台不插值，CALMET后续按其配置处理。')
    if model=='calpuff' and mode=='precipitation':st.info('PRECIP.DAT：需真实完整逐小时、全部配置站点；小时结束标签及固定标准时区明确。输出mm/hour，前一小时累计量须有3600秒依据，不能把日累计拆分。站点位置、NPSTA、IFORMP=2等在已有CALMET.INP中配置。')
    if model=='wrf' and mode=='input_package':st.info('上传已有原生WPS v5文件ZIP；配置明确UTC时刻列表、间隔、每字段必需层及缺测标记。核对同网格/投影/单位/时刻后逐字节组包。不补时间、字段或层，不插值；需外部geo_em、METGRID.TBL和其余驱动，未运行metgrid。')
    if model=='wrf' and mode=='geogrid_tiles':st.info('生成WPS4.6 geogrid上游index与二进制瓦片ZIP，仅规则经纬度HGT_M或已分类LANDUSEF。需静态真实完整网格、单位及量化误差阈值；土地分类需明确分类系统/特殊类别。不是geo_em，需外部geogrid和已配置GEOGRID.TBL/域覆盖。')
    if model=='wrf' and mode=='pressure_fields':st.info('支持规则经纬度IPROJ0与显式球形Lambert IPROJ3（源坐标m/km，WPS格距写km；不重投影、不旋转风）。WPS多变量文件仅包含映射成功的原始字段，可写指定深度ST/SM土壤温湿度（需层深与体积含水率依据），不生成缺失字段或缺失层，不做时间/空间/垂直插值；缺少required_fields会拒绝。')
    if model=='camx' and mode=='boundary_concentrations':st.info('CAMx7.40 BOUNDARY NetCDF：需完整三维网格浓度（含内部与边角），配置显式选择1小时或6小时间隔、至少两个连续时段、至少3x3网格及明确起止时刻。原时段原样保留，不复制成逐小时。官方读取器先读取全数组再使用侧边；不补造内部值、不插值。还需项目气象、机制及控制文件；转换本身不执行CAMx。')
    if model=='camx' and mode=='initial_concentrations':st.info('生成CAMx7.40 INITIAL三维浓度NetCDF：需每格每层真实浓度、机制物种、网格、层定义、初始时刻和明确valid_until。只重排和换单位，不推算初始场、不填缺物种，不生成气象/边界/控制工程。')
    if model=='cmaq' and mode=='boundary_concentrations':st.info('生成CMAQ5.5 BNDARY3外圈浓度NetCDF（NTHIK=1），需已在目标网格各层各UTC小时的完整边界浓度和真实位置。不从BC_PROFILE推算边界、不执行BCON；MCIP、机制及完整工程仍需提供。')
    if model=='camx' and mode=='gridded_emissions':st.info('输入已分种的逐小时完整网格CSV，气体mol/hour、气溶胶g/hour。配置包含原点/格距/投影/固定标准时区；不会把NOx、PM25总量自动拆成机制物种。')
    if model=='cmaq' and mode=='gridded_emissions':st.info('输入已空间分配的NetCDF地面排放。hourly要求真实连续UTC小时；static_inventory须明确平均期和静态使用假设，不代表逐小时数据。需匹配MCIP网格、垂直定义与DESID物种；不生成MCIP或CCTM工程。')
    if model=='cmaq' and mode=='bcon_profile':st.info('输入静态垂直化学剖面CSV；需明确机制、逐变量单位、层数与来源。生成BC_PROFILE供CMAQ5.5 BCON/profile读取；不生成边界场NetCDF，不含时间和地理网格，目标网格由外部MET_BDY_3D_FIN提供。')
    if model=='swat_plus' and mode is None:
        source_type=st.radio('日天气原始资料形式',['已有单站NetCDF','原始降水与温度CSV'],key='swat_weather_source_type')
        if source_type=='原始降水与温度CSV':
            from ecodata.interface.swat_raw_csv_page import render
            render()
            return
    data=st.file_uploader('导入源数据',type=selected[2],key='round1_data_'+model+'_'+str(mode))
    cfg=st.file_uploader('导入字段、单位与版本配置JSON',type=['json'],key='round1_config_'+model)
    station_index=st.file_uploader('可选：已有weather-sta.cli（只检查引用，不修改）',type=['cli'],key='wgn_station_index') if model=='swat_plus' and mode=='wgn' else None
    st.caption('示例配置位于examples/ai_round1；原始A、格式参考与合成测试已分别标注。请按你的真实数据修改配置。')
    fingerprint=hashlib.sha256((model+choice).encode()+(data.getvalue() if data else b'')+(cfg.getvalue() if cfg else b'')+(station_index.getvalue() if station_index else b'')).hexdigest()
    if st.session_state.get('round1_fingerprint')!=fingerprint:st.session_state.pop('round1_result',None)
    st.session_state['round1_fingerprint']=fingerprint
    if cfg:
        try:st.json(json.loads(cfg.getvalue().decode('utf-8-sig')))
        except (ValueError,UnicodeError):st.error('配置JSON无效')
    if st.button('执行专项转换',disabled=not(data and cfg)):
        st.session_state.pop('round1_result',None)
        try:
            config=json.loads(cfg.getvalue().decode('utf-8-sig'))
            if not isinstance(config,dict):raise ValueError('配置JSON必须为对象')
            new_modes={'constant_points','management_schedule','input_package','spatial_landuse','receptors','geogrid_tiles','initial_concentrations','boundary_concentrations','geography','upper_air','precipitation','wgn','soil','hru_attributes','landuse_database','aquifer_parameters','connections','pressure_fields','gridded_emissions','bcon_profile'}
            if mode is not None and config.get('mode')!=mode or mode is None and config.get('mode') in new_modes:
                raise ValueError(f'界面转换类型与配置不一致：{choice}需mode={mode if mode else "原有配置"}')
            files,report=convert(model,data.getvalue(),config,weather_station_index=station_index.getvalue() if station_index else None)
            st.session_state['round1_result']=(files,report)
        except Exception as exc:st.error(f'转换被拒绝：{exc}')
    if result:=st.session_state.get('round1_result'):
        files,report=result
        st.success('已完成本类文件写出；请核对报告。本提示不表示完整工程或仿真运行成功。')
        st.json(report);archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for name,content in files.items():
                z.writestr(name,content);st.download_button('下载 '+name,content,name,key='round1_download_'+name)
        st.download_button('下载全部转换文件',archive.getvalue(),model+'_converted.zip')
        render_project_continuation(model, mode, archive.getvalue(), data.getvalue(), cfg.getvalue(), fingerprint)


def render_project_continuation(model, mode, archive, source, config, fingerprint):
    """Continue only with the successful conversion currently displayed above."""
    if model == 'swat':
        return
    st.divider()
    st.subheader('下一步：将本次转换结果用于模型工程')
    st.caption('上传原始数据和配置 → 转换与检查 → 下载转换结果 → 选择工程路线 → 预处理 → 原生运行 → 查看结果')
    st.write('转换ZIP是模型所需输入的一部分；新建工程仍需配套资料，已有工程仍需核对引用、版本、网格与时段。')
    if st.session_state.get('continuation_fingerprint') != fingerprint:
        st.session_state['continuation_fingerprint'] = fingerprint
        for key in ('swat_active_job', 'swat_build_job', 'swat_build_active_job', 'wrf_job',
                    'run_target_station', 'run_weather_confirm', 'swat_version_confirm',
                    'camx_current_job','cmaq_current_job','calmet_current_job','swat_parameter_confirm'):
            st.session_state.pop(key, None)
    if not st.checkbox('继续办理本次数据的工程流程', key='continue_' + fingerprint):
        return
    if model == 'swat_plus' and mode is None:
        from ecodata.interface.swat_run_page import render_swat_run_page
        render_swat_run_page(converted_weather=archive, weather_inputs=(source, config))
    elif model == 'swat_plus':
        from ecodata.interface.swat_run_page import render_swat_run_page
        render_swat_run_page(converted_inputs=archive,input_mode=mode)
    elif model == 'wrf' and mode in ('pressure_fields','input_package'):
        from ecodata.interface.wrf_run_page import render_wrf_run_page
        render_wrf_run_page(converted_zip=archive)
    elif model == 'wrf' and mode=='geogrid_tiles':
        from ecodata.interface.wrf_run_page import render_wrf_run_page
        render_wrf_run_page(converted_zip=archive,input_mode=mode)
    elif model == 'wrf' and mode is None:
        from ecodata.interface.wrf_run_page import render_wrf_run_page
        render_wrf_run_page(converted_zip=archive,input_mode='sst')
    elif model == 'camx':
        from ecodata.interface.camx_run_page import render_camx_run_page
        render_camx_run_page(converted_zip=archive,mode=mode)
    elif model == 'cmaq' and mode in (None,'bcon_profile','gridded_emissions','boundary_concentrations'):
        from ecodata.interface.cmaq_run_page import render_cmaq_run_page
        render_cmaq_run_page(converted_zip=archive,mode=mode)
    elif model == 'calpuff':
        from ecodata.interface.calmet_run_page import render_calmet_run_page
        render_calmet_run_page(converted_zip=archive,mode=mode)
    else:
        route = st.radio('本次数据的工程路线', ['从原始必要资料新建工程', '对接已有工程'], key='continuation_route_' + model)
        requirements = {
            'swat_plus': '完整流域GIS、土壤/土地利用数据库、连接、管理与天气资料',
            'wrf': '计算域、地理资料、完整气象驱动、WPS配置及物理方案',
            'calpuff': 'CALMET/CALPUFF控制文件、地理网格、高空气象、地面气象、排放与受体',
            'camx': '一致的网格和化学机制、气象、排放、初边界场与控制文件',
            'cmaq': 'GRIDDESC、MCIP气象、化学机制、排放、初边界场与运行配置',
        }
        st.write('还需准备：' + requirements[model] + '。')
        if route == '对接已有工程':
            st.write('需核对本次输出与已有工程的变量、单位、时间、坐标、网格及文件引用，再替换工程副本中的对应输入。')
        else:
            st.write('需先建立区域、参数、输入引用和运行配置，再接入本次输出并执行该模型的预处理。')
        st.warning('此转换类别尚未接入通用工程构建/运行接口；当前结果已可下载，但这里不能直接运行。侧栏登记案例使用其自身资料，不代表本次数据已接入。')
