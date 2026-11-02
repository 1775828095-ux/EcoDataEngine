"""Bounded configurable original GFS/FNL project, without registered-value equality."""
from pathlib import Path
from datetime import datetime,timedelta
import hashlib,json,subprocess,threading,uuid,zipfile,io

ROOT=Path(__file__).resolve().parents[3]
RUNS=ROOT/'outputs/wrf_projects'
LOCK=threading.Lock();ACTIVE=None


def validate_settings(value):
    required={'start','end','interval_seconds','duration_seconds','latitude','longitude','dx','nx','ny','levels','top_pressure','time_step','physics_suite'}
    if set(value)!=required:raise ValueError('工程设置字段缺少或不支持：'+str(set(value)^required))
    value=dict(value)
    start=datetime.strptime(value['start'],'%Y-%m-%d_%H:%M:%S');end=datetime.strptime(value['end'],'%Y-%m-%d_%H:%M:%S')
    integers={'interval_seconds':(3600,21600),'duration_seconds':(300,21600),'dx':(6000,50000),'nx':(25,101),'ny':(25,101),'levels':(30,65),'top_pressure':(1000,10000),'time_step':(5,180)}
    for key,(low,high) in integers.items():
        if type(value[key]) is not int or not low<=value[key]<=high:raise ValueError('工程设置超出当前验收范围：'+key)
    span=int((end-start).total_seconds())
    if span<=0 or span>86400 or span%value['interval_seconds'] or value['duration_seconds']>span or start.minute or start.second or end.minute or end.second:
        raise ValueError('时段、间隔或积分长度不一致；当前只接受整点且不超过一天')
    if value['physics_suite']!='CONUS':raise ValueError('当前仅实现显式CONUS物理方案，不替换其他方案')
    if not (15<=value['latitude']<=55 and -130<=value['longitude']<=-55):raise ValueError('当前低分辨率地理资料与区域验收范围仅覆盖北美及邻近海域')
    if value['time_step']>value['dx']/1000*6:raise ValueError('时间步长超过当前入口的保守分辨率限制')
    if value['duration_seconds']%300:raise ValueError('当前图表输出间隔5分钟，积分长度需为300秒的整数倍')
    if max(value['nx']-1,value['ny']-1)*value['dx']>1_200_000:raise ValueError('当前短时验收单轴区域长度不超过1200km')
    return value


def directory(ident):
    import re
    if not re.fullmatch('[0-9a-f]{32}',ident):raise ValueError('无效WRF新建任务号')
    return RUNS/ident


def save(folder,record):
    from ecodata.interface.independent_cases import save as atomic_save
    atomic_save(folder,record)


def status(ident):return json.loads((directory(ident)/'status.json').read_text(encoding='utf-8'))


def export_project(project_id):
    """Platform export only: package existing bytes, no scientific edits."""
    source=directory(project_id)
    if status(project_id).get('state')!='succeeded':raise ValueError('只能导出成功工程')
    mapping={'wrf/namelist.input':'wrf/namelist.input','real/namelist.input':'real/namelist.input',
             'wrf/wrfinput_d01':'real/wrfinput_d01','wrf/wrfbdy_d01':'real/wrfbdy_d01',
             'wps/namelist.wps':'wps/namelist.wps','wps/geo_em.d01.nc':'wps/geo_em.d01.nc'}
    mapping.update({p.relative_to(source).as_posix():p.relative_to(source).as_posix() for p in (source/'original_wps').glob('*.wps')})
    for name in ('raw_pressure.nc','config.json','platform_conversion.zip'):
        if (source/name).is_file():mapping[name]=name
    buf=io.BytesIO();manifest={}
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for target,original in mapping.items():
            raw=(source/original).read_bytes();z.writestr(target,raw)
            manifest[target]=dict(source=original,sha256=hashlib.sha256(raw).hexdigest())
        z.writestr('platform_export.json',json.dumps(dict(source_job=project_id,files=manifest,
            scientific_data_modified=False,scope='Previously platform-created project, not an unfamiliar author archive'),indent=2))
    return buf.getvalue()


def start_external(payload,converted_zip,wrf_namelist,real_namelist):
    global ACTIVE
    from ecodata.interface import independent_cases as cases
    from ecodata.interface.wrf_external_project import validate_run_contract
    template=cases.SOURCE/'evidence/WRF_Colorado'
    files,report,cfg=validate_run_contract(payload,converted_zip,wrf_namelist,real_namelist,
        (template/'namelist.input.wrf').read_bytes(),(template/'namelist.input.real').read_bytes())
    with cases.LOCK,LOCK:
        if ACTIVE or cases.ACTIVE:raise ValueError('已有原生任务，请等待完成；模型任务串行运行')
        ACTIVE=uuid.uuid4().hex;ident=ACTIVE;cases.ACTIVE=ident
    folder=directory(ident)
    try:
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'supplied_project.zip').write_bytes(payload)
        (folder/'supplied_conversion.zip').write_bytes(converted_zip)
        (folder/'settings.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
        (folder/'external_inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        record=dict(id=ident,state='running',stage='保留用户配置并准备外部工程副本',route='external',settings=cfg,
                    project_sha256=hashlib.sha256(payload).hexdigest(),conversion_sha256=hashlib.sha256(converted_zip).hexdigest(),
                    scientific_settings_changed=False,accuracy_evaluated=False,
                    scope=report['execution_contract'])
        save(folder,record)
        threading.Thread(target=external_work,args=(folder,record,files,report,converted_zip),daemon=True).start()
        return ident
    except Exception:
        with cases.LOCK,LOCK:
            ACTIVE=None
            if cases.ACTIVE==ident:cases.ACTIVE=None
        raise


def external_work(folder,record,files,report,converted_zip):
    global ACTIVE
    from ecodata.interface import independent_cases as cases
    try:
        from ecodata.interface.wrf_assembly import read_fields
        from ecodata.interface.wrf_runner import read_update_zip
        for name in ('wps','original_wps','converted'):(folder/name).mkdir()
        mapping={}
        def write(target,source):
            (folder/target).write_bytes(files[source]);mapping[target]=dict(source=source,sha256=hashlib.sha256(files[source]).hexdigest())
        write('wps/namelist.wps',report['wps_companions']['namelists'][0])
        write('wps/geo_em.d01.nc',report['wps_companions']['geogrid'][0])
        write('supplied_namelist.wrf',report['execution_contract']['wrf_namelist'])
        write('supplied_namelist.real',report['execution_contract']['real_namelist'])
        for source in report['wps_companions']['intermediate_files']:
            stamp=read_fields(files[source])[0].time
            if stamp[14:]!='00:00':raise ValueError('当前受控运行要求整点WPS资料；不重排或聚合')
            write('original_wps/'+stamp[:13]+'.wps',source)
        for i,(name,raw) in enumerate(read_update_zip(converted_zip).items()):(folder/'converted'/f'update_{i:02d}.wps').write_bytes(raw)
        (folder/'external_file_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf8')
        worker=Path(__file__).with_name('wrf_project_worker.py')
        science=cases.SOURCE.parent/'四模型页面路线验收_20261029/linux_science_modules'
        command=['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','env','PYTHONPATH='+cases.linux(science),'OPENBLAS_NUM_THREADS=1','python3',cases.linux(worker),cases.linux(folder)]
        for stage in ('external_prepare','run'):
            record['stage']='外部工程：'+stage;save(folder,record)
            with (folder/(stage+'_launcher.log')).open('wb') as log:
                result=subprocess.run(command+[stage],stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            if result.returncode:raise ValueError('外部工程阶段失败：'+stage+'；原资料与日志保留')
        native=json.loads((folder/'native_result.json').read_text())
        for kind in ('real','wrf'):
            if (folder/kind/'namelist.input').read_bytes()!=(folder/('supplied_namelist.'+kind)).read_bytes():
                raise ValueError('用户配置原文保留校验失败：'+kind)
        cases.plot(folder,'wrf',native)
        record.update(state='succeeded',stage='外部工程接入→metgrid→real→WRF积分与出图完成',native=native,namelist_bytes_preserved=True)
    except Exception as exc:record.update(state='failed',error=str(exc))
    finally:
        save(folder,record)
        with cases.LOCK,LOCK:
            ACTIVE=None
            if cases.ACTIVE==record['id']:cases.ACTIVE=None


def start(gribs,settings,converted_zip=None):
    global ACTIVE
    cfg=validate_settings(settings)
    if not 2<=len(gribs)<=9:raise ValueError('需要2至9个原始GFS/FNL文件；不要上传WPS或WRF成品')
    inputs=[]
    for name,raw in gribs:
        if not isinstance(raw,bytes) or not 1000<=len(raw)<=512*1024**2 or raw[:4]!=b'GRIB' or raw[7] not in (1,2):raise ValueError('输入不是支持的原始GRIB1/GRIB2或超过512MiB：'+name)
        inputs.append((raw,{'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'grib_edition':raw[7]}))
    if sum(len(raw) for raw,_ in inputs)>1024**3:raise ValueError('原始资料总量超过1GiB入口限额')
    from ecodata.interface import independent_cases as cases
    with cases.LOCK,LOCK:
        if ACTIVE or cases.ACTIVE:raise ValueError('已有原生任务，请等待完成；模型任务串行运行')
        ACTIVE=uuid.uuid4().hex;ident=ACTIVE;cases.ACTIVE=ident
    folder=directory(ident)
    try:
        folder.mkdir(parents=True,exist_ok=False)
        for i,(raw,meta) in enumerate(inputs):(folder/f'raw_{i:02d}.grib').write_bytes(raw)
        record=dict(id=ident,state='running',stage='原始资料检查与WPS',settings=cfg,inputs=[m for _,m in inputs],scope='Configurable North America GFS/FNL GRIB1/2, explicit CONUS physics; not all global regions or arbitrary weather products',accuracy_evaluated=False)
        (folder/'settings.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');save(folder,record)
        threading.Thread(target=work,args=(folder,record,converted_zip),daemon=True).start()
        return ident
    except Exception:
        with cases.LOCK,LOCK:
            ACTIVE=None
            if cases.ACTIVE==ident:cases.ACTIVE=None
        raise


def start_existing(project_id,converted_zip):
    """Reuse an accepted local custom domain; never silently use the old demo."""
    global ACTIVE
    from ecodata.interface import independent_cases as cases
    if converted_zip is None:raise ValueError('已有工程接入必须提供本次平台转换ZIP')
    source=directory(project_id);prior=status(project_id)
    if prior.get('state')!='succeeded':raise ValueError('只能接入已经成功的新建工程，不回退登记案例')
    cfg=validate_settings(prior['settings'])
    from ecodata.interface.wrf_runner import read_update_zip
    read_update_zip(converted_zip)
    # Vtable is only consumed by ungrib, which this reuse route does not run.
    # Older successful projects use a Linux symlink not readable by Win32.
    names=['settings.json','config.json','raw_pressure.nc','wps/geo_em.d01.nc','wps/namelist.wps']
    names += [p.relative_to(source).as_posix() for p in sorted((source/'original_wps').glob('*.wps'))]
    if len(names)<7:raise ValueError('已有工程缺完整预处理场')
    pins={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}
    with cases.LOCK,LOCK:
        if ACTIVE or cases.ACTIVE:raise ValueError('已有原生任务，请等待完成；模型任务串行运行')
        ACTIVE=uuid.uuid4().hex;ident=ACTIVE;cases.ACTIVE=ident
    folder=directory(ident)
    try:
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'settings.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
        (folder/'supplied_conversion.zip').write_bytes(converted_zip)
        record=dict(id=ident,state='running',stage='核验已有新建工程及本次转换',route='existing',settings=cfg,
                    reused_project=project_id,reused_file_sha256=pins,supplied_conversion_sha256=hashlib.sha256(converted_zip).hexdigest(),scope='Reuse accepted local custom WRF project geogrid and complete WPS companions, attach supplied pressure conversion, rerun metgrid/real/WRF. Not arbitrary uploaded executables or arbitrary WRF versions.',accuracy_evaluated=False)
        (folder/'reuse_project.json').write_text(json.dumps(dict(source=cases.linux(source),pins=pins),indent=2),encoding='utf8')
        save(folder,record)
        threading.Thread(target=work,args=(folder,record,converted_zip),daemon=True).start()
        return ident
    except Exception:
        with cases.LOCK,LOCK:
            ACTIVE=None
            if cases.ACTIVE==ident:cases.ACTIVE=None
        raise


def work(folder,record,converted_zip):
    global ACTIVE
    from ecodata.interface.independent_cases import linux
    try:
        worker=Path(__file__).with_name('wrf_project_worker.py')
        from ecodata.interface.independent_cases import SOURCE
        science=SOURCE.parent/'四模型页面路线验收_20261029/linux_science_modules'
        command=['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','env','PYTHONPATH='+linux(science),'OPENBLAS_NUM_THREADS=1','python3',linux(worker),linux(folder)]
        for stage in (('reuse_prepare','run') if record.get('route')=='existing' else ('prepare','run')):
            if stage=='run':
                record['stage']='平台气压场转换与原生接入';save(folder,record)
                from ecodata.round1 import convert
                config=json.loads((folder/'config.json').read_text())
                files,report=convert('wrf',(folder/'raw_pressure.nc').read_bytes(),config)
                if converted_zip is not None:
                    from ecodata.interface.wrf_runner import read_update_zip
                    files=read_update_zip(converted_zip)
                else:
                    with zipfile.ZipFile(folder/'platform_conversion.zip','w',zipfile.ZIP_DEFLATED) as z:
                        for name,raw in files.items():z.writestr(name,raw)
                updates=folder/'converted';updates.mkdir()
                for i,(name,raw) in enumerate((n,b) for n,b in files.items() if n.endswith('.wps')):(updates/f'update_{i:02d}.wps').write_bytes(raw)
                (folder/'conversion_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            with (folder/(stage+'_launcher.log')).open('wb') as log:
                result=subprocess.run(command+[stage],stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            if result.returncode:raise ValueError('WRF新建阶段失败：'+stage+'；查看本任务阶段日志')
        from ecodata.interface.independent_cases import plot
        native=json.loads((folder/'native_result.json').read_text())
        assert native['success'];plot(folder,'wrf',native)
        record.update(state='succeeded',stage='已有工程→本次转换→WRF积分及出图完成' if record.get('route')=='existing' else '原始GRIB→平台转换→WRF积分及出图完成',native=native)
    except Exception as exc:
        record.update(state='failed',error=str(exc))
    finally:
        save(folder,record)
        from ecodata.interface import independent_cases as cases
        with cases.LOCK,LOCK:
            ACTIVE=None
            if cases.ACTIVE==record['id']:cases.ACTIVE=None


def render(converted_zip=None,mode=None):
    import streamlit as st
    st.header('自定义WRF工程（试验）')
    st.warning('支持北美及邻近海域的单网格短时试验，固定显式CONUS物理方案；不代表任意全球区域、嵌套或其他再分析产品已通过。低分辨率地理库也不适合精细城市研究。')
    st.write('原始GRIB与区域/时段设置 → geogrid、ungrib → 平台气压场转换 → metgrid、real → WRF积分 → 原生结果图与CSV')
    st.caption('完整原始GRIB须含气压层温度、风、高度、湿度及地表/土壤配套场。已有转换包仅替换TT/UU/VV/GHT；缺少配套场会拒绝，不会使用登记案例天气。2011-05-10以前采用官方Vtable注明的PRMSL映射，并记录映射依据。')
    if mode is not None and mode!='pressure_fields':
        st.error('此入口接受气压场转换；其他转换类别不能作为完整三维气象输入。');return True
    route=st.radio('自定义工程路线',['从原始GRIB新建','接入已成功的本机自定义工程','接入用户上传的外部工程（试验）'],key='wrf_custom_route')
    if route=='接入用户上传的外部工程（试验）':
        st.write('上传工程 → 明确REAL/WRF配置 → 检查依赖与转换兼容性 → 在副本运行metgrid、real、WRF → 查看并下载结果。')
        st.warning('仅支持当前受控WRF4.8.0单网格CONUS科学设置与完整WPS资料。不同科学设置会拒绝，绝不覆盖成模板；检查通过仍需实际运行验收。')
        archive=st.file_uploader('外部WRF工程ZIP（含REAL与WRF配置；展开不超过256MiB）',type=['zip'],key='wrf_external_archive')
        update=st.file_uploader('可选：用于兼容性检查的平台气压场转换ZIP',type=['zip'],key='wrf_external_update') if converted_zip is None else None
        from ecodata.interface.input_result_binding import bind_result
        bind_result(st.session_state,'wrf_external_job','wrf_external_base_binding',archive.getvalue() if archive else b'',converted_zip or (update.getvalue() if update else b''))
        if archive is not None:
            from ecodata.interface.wrf_external_project import inspect_project,read_archive,validate_run_contract
            try:
                contents=read_archive(archive.getvalue())
                candidates=[n for n in contents if n.rsplit('/',1)[-1]=='namelist.input']
                chosen=st.selectbox('选择WRF积分配置（原文保留）',candidates,index=None,key='wrf_external_main_nml')
                real_chosen=st.selectbox('选择REAL预处理配置（原文保留）',candidates,index=None,key='wrf_external_real_nml')
                bind_result(st.session_state,'wrf_external_job','wrf_external_choice_binding',chosen,real_chosen)
                if not chosen:
                    st.info('先明确选择WRF积分配置；多个目录不自动猜选。');return True
                supplied=converted_zip if converted_zip is not None else update.getvalue() if update is not None else None
                report=inspect_project(archive.getvalue(),supplied,chosen)
                from ecodata.interface.engineering_completeness import render as render_completeness
                render_completeness(report['completeness'],'wrf')
                st.write('原生输入结构候选：'+('符合本次结构检查' if report['native_structure_candidate'] else '存在缺项或不兼容'))
                for message in report['errors']:st.error(message)
                for message in report['warnings']:st.warning(message)
                st.write('本次新气压场逐时刻合并检查：'+('已检查（未运行模型）' if report['replacement_checked'] else '未通过或尚未提供齐全资料'))
                with st.expander('外部工程文件、日期与依赖检查报告'):st.json(report)
                st.download_button('下载外部工程检查报告JSON',json.dumps(report,ensure_ascii=False,indent=2),'wrf_external_inspection.json',key='wrf_external_report')
                eligible=False
                if supplied is not None and real_chosen:
                    from ecodata.interface import independent_cases as cases
                    template=cases.SOURCE/'evidence/WRF_Colorado'
                    try:
                        validate_run_contract(archive.getvalue(),supplied,chosen,real_chosen,(template/'namelist.input.wrf').read_bytes(),(template/'namelist.input.real').read_bytes())
                        eligible=True
                        st.info('受控运行兼容检查通过；平台会保留两份配置原文，使用已安装程序及配套表。')
                    except ValueError as exc:st.error('运行兼容检查：'+str(exc))
                if st.button('接入并运行上传工程',disabled=not eligible,key='wrf_external_start'):
                    st.session_state.pop('wrf_external_job',None)
                    st.session_state['wrf_external_job']=start_external(archive.getvalue(),supplied,chosen,real_chosen)
            except Exception as exc:st.error('外部工程识别失败：'+str(exc))
        ident=st.session_state.get('wrf_external_job')
        if ident:
            @st.fragment(run_every=3)
            def show_external():
                folder=directory(ident);record=status(ident)
                st.write('本次外部工程任务：'+ident)
                if record['state']=='succeeded':
                    st.success('本次外部工程实际运行及出图完成，两份用户配置原文一致；未评价预测精度。')
                    st.image(str(folder/'result.png'))
                    for ext in ('csv','png'):st.download_button('下载外部工程结果'+ext.upper(),(folder/('result.'+ext)).read_bytes(),'wrf_external_result.'+ext,key='wrf_ext_'+ext)
                elif record['state']=='failed':st.error(record['error'])
                else:st.info(record['stage'])
                with st.expander('外部工程任务、阶段与输入保留证据'):
                    st.json(record)
                    if (folder/'stages.json').exists():st.json(json.loads((folder/'stages.json').read_text()))
            show_external()
        return True
    if route=='接入已成功的本机自定义工程':
        st.write('保留原工程区域和科学设置 → 核验geo_em与完整WPS场 → 接入本次平台转换 → metgrid → real → WRF → 出图。')
        st.caption('当前接收本机成功任务号，不接收可执行脚本。仅支持当前GFS/FNL单网格入口创建的工程；不是任意WRF版本工程导入。')
        project_id=st.text_input('已有自定义工程任务号',key='wrf_custom_existing_id').strip()
        if project_id:
            try:st.download_button('导出此工程的原始文件与配置ZIP',export_project(project_id),'wrf_existing_project.zip',key='wrf_existing_export')
            except (ValueError,OSError) as exc:st.caption('工程导出暂不可用：'+str(exc))
        supplied=st.file_uploader('本次平台气压场转换ZIP',type=['zip'],key='wrf_custom_existing_zip') if converted_zip is None else None
        if st.button('接入并运行已有自定义工程',key='wrf_custom_existing_start'):
            try:
                payload=converted_zip if converted_zip is not None else supplied.getvalue() if supplied is not None else None
                if not project_id or payload is None:raise ValueError('必须提供成功工程任务号及本次转换ZIP；不会回退登记案例')
                st.session_state['wrf_custom_existing_job']=start_existing(project_id,payload)
            except Exception as exc:st.error(str(exc))
        ident=st.session_state.get('wrf_custom_existing_job')
        if ident:
            @st.fragment(run_every=3)
            def show_existing():
                folder=directory(ident);record=status(ident)
                st.write('本次任务：'+ident)
                if record['state']=='succeeded':
                    st.success('已有工程接入、原生运行与出图完成；未评价预测精度。')
                    st.image(str(folder/'result.png'))
                    for ext in ('csv','png'):
                        st.download_button('下载本次结果'+ext.upper(),(folder/('result.'+ext)).read_bytes(),'wrf_result.'+ext,key='wrf_existing_'+ext)
                elif record['state']=='failed':st.error(record['error'])
                else:st.info(record['stage'])
                with st.expander('原工程、文件哈希及本次阶段记录'):
                    st.json(record)
                    if (folder/'stages.json').exists():st.json(json.loads((folder/'stages.json').read_text()))
            show_existing()
        return True
    uploads=st.file_uploader('原始GFS/FNL GRIB文件（2至9个时次；不是WRF输出）',accept_multiple_files=True,key='wrf_project_gribs')
    setting=st.file_uploader('区域、时段与计算设置JSON',type=['json'],key='wrf_project_settings')
    sample=dict(start='2016-10-06_18:00:00',end='2016-10-07_00:00:00',interval_seconds=21600,duration_seconds=1800,latitude=27.,longitude=-78.,dx=12000,nx=49,ny=49,levels=48,top_pressure=5000,time_step=60,physics_suite='CONUS')
    st.download_button('下载工程设置模板',json.dumps(sample,indent=2),'wrf_project_settings.json',key='wrf_project_template')
    st.caption('原始文件实际时次必须覆盖设置时段；区域、分辨率、层数和物理方案均保存在结果中。短时运行用于链路验收，未评估预报精度。')
    if st.button('建立并运行自定义WRF工程',key='wrf_project_start'):
        try:
            if not uploads or setting is None:raise ValueError('请上传原始GRIB及设置；不会回退登记案例')
            if setting.size>16384:raise ValueError('设置JSON超过16KiB')
            cfg=json.loads(setting.getvalue().decode('utf-8-sig'))
            st.session_state['wrf_project_job']=start([(p.name,p.getvalue()) for p in uploads],cfg,converted_zip)
        except Exception as exc:st.error(str(exc))
    ident=st.session_state.get('wrf_project_job')
    if ident:
        @st.fragment(run_every=3)
        def show():
            folder=directory(ident);record=status(ident)
            st.write('本次任务：'+ident)
            if record['state']=='succeeded':
                st.success('本次原始资料已完成原生链路；未验证预报精度。')
                st.image(str(folder/'result.png'))
                st.download_button('导出完整已有工程接入ZIP',export_project(ident),'wrf_existing_project.zip',key='wrf_project_export')
                st.download_button('下载本次数值CSV',(folder/'result.csv').read_bytes(),'wrf_result.csv',key='wrf_project_csv')
                st.download_button('下载本次图表PNG',(folder/'result.png').read_bytes(),'wrf_result.png',key='wrf_project_png')
            elif record['state']=='failed':st.error(record['error'])
            else:st.info(record['stage'])
            with st.expander('设置、输入哈希与实际阶段日志'):
                st.json(record)
                st.caption(str(folder))
                for name in ('vtable_policy.json','stages.json','prepare_launcher.log','run_launcher.log'):
                    if (folder/name).is_file():st.code((folder/name).read_text(encoding='utf-8',errors='replace')[-12000:])
        show()
    return True
