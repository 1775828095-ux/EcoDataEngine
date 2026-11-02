"""Bounded WRF/grid-parameter build and supplied CALPUFF7 project reconnect.

Optional explicit raw-raster geography; no automatic scientific configuration,
chemistry or accuracy claim.
"""
from pathlib import Path
from datetime import datetime
import io,json,re,zipfile,hashlib,subprocess,threading,uuid
import numpy as np
from ecodata.interface.calpuff7_results import control_values,read_discrete

ROOT=Path(__file__).resolve().parents[3]
RUNS=ROOT/'outputs/calpuff_projects'
COMMON={'CALPUFF.INP','provenance.json'}
REBUILD={'WRF.nc','CALMET.INP','geography.nc','geography_config.json','points_input.zip','point_config.json','receptors.csv','receptor_config.json'}
RAW_REBUILD=(REBUILD-{'geography.nc','geography_config.json'})|{'geography_raw.zip','geography_raw_config.json'}
EXISTING={'CALMET.DAT','CALMET.DAT.AUX'}

def unpack(raw,route,control_path=None,raw_geography_inputs=False):
    required=COMMON|(REBUILD if route=='rebuild' else EXISTING)
    if route not in {'rebuild','existing'}:raise ValueError('工程路线无效')
    if len(raw)>200*1024**2:raise ValueError('ZIP超过200MiB')
    import_report=None
    if route=='existing':
        from pathlib import PurePosixPath
        from ecodata.interface.wrf_external_project import read_archive
        original=read_archive(raw)
        candidates=[n for n,b in original.items() if b.startswith(b'CALPUFF.INP     7.0')]
        if control_path is not None:
            if control_path not in candidates:raise ValueError('指定文件不是支持的CALPUFF7控制文件')
            candidates=[control_path]
        if len(candidates)!=1:raise ValueError('需明确一个CALPUFF7.0格式控制文件；其他版本不自动改头，多个工程需选择')
        control=candidates[0];parent=PurePosixPath(control).parent
        values=control_values(original[control].decode('ascii'))
        from ecodata.interface.engineering_completeness import calpuff,require
        completeness=calpuff(original,control);require(completeness)
        if len(values.get('METDAT',[]))!=1 or len(values.get('AUXEXT',[]))!=1:raise ValueError('气象文件与AUX扩展引用不明确')
        met,aux=values['METDAT'][0],values['AUXEXT'][0]
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',met) or '..' in met or aux.upper()!='AUX':raise ValueError('气象引用需安全单层文件名及AUX，不改写外部路径')
        lookup={n.casefold():n for n in original}
        selected={}
        data={'CALPUFF.INP':original[control]}
        for filename in (met,met+'.'+aux):
            source=lookup.get(str(parent/filename).casefold())
            if source is None:raise ValueError('工程缺少实际引用的气象资料：'+str(parent/filename))
            data[filename.upper()]=original[source];selected[filename.upper()]=source
        provenance_path=lookup.get(str(parent/'provenance.json').casefold())
        if provenance_path is not None:data['provenance.json']=original[provenance_path]
        else:data['provenance.json']=json.dumps(dict(sources=['原上传工程未提供来源声明，平台不代为认证'],defaults=['原控制科学设置保留；不由平台补参数'],limitations=['来源、参数率定与预测精度未核实；受当前CALPUFF7入口限制'])).encode('utf8')
        import_report=dict(completeness=completeness,original_zip_sha256=hashlib.sha256(raw).hexdigest(),control_source=control,
            control_sha256=hashlib.sha256(original[control]).hexdigest(),control_bytes_preserved=True,
            file_mapping=selected,weather_bytes_preserved=True,scientific_parameters_changed=False,
            generated_provenance_declaration=provenance_path is None,
            ignored_files=[n for n in original if n not in {control,provenance_path,*selected.values()}],
            source_file_sha256={n:hashlib.sha256(original[n]).hexdigest() for n in {control,*selected.values()}})
    else:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=z.namelist()
            if raw_geography_inputs and set(names)==COMMON|(REBUILD-{'geography.nc','geography_config.json'}):required=set(names)
            if 'geography_raw.zip' in names:required=COMMON|RAW_REBUILD
            if len(names)!=len(required) or set(names)!=required or sum(i.file_size for i in z.infolist())>512*1024**2:raise ValueError('工程ZIP需且仅需：'+', '.join(sorted(required)))
            data={n:z.read(n) for n in names}
    provenance=json.loads(data['provenance.json'])
    if set(provenance)!={'sources','defaults','limitations'} or any(not isinstance(v,list) or not v or any(not isinstance(s,str) or not s.strip() for s in v) for v in provenance.values()):raise ValueError('provenance需非空sources/defaults/limitations文字列表')
    for name in ('CALPUFF.INP',*(['CALMET.INP'] if route=='rebuild' else [])):
        text=data[name].decode('ascii');v=control_values(text)
        # Only local single-level file references; native tools never receive paths.
        if any(any(c in x for c in ('/','\\',':')) or '..' in x for xs in v.values() for x in xs):raise ValueError('控制文件仅允许单层文件名，不接受外部路径：'+name)
        if name=='CALPUFF.INP':
            def one(k):
                if len(v.get(k,[]))!=1:raise ValueError('控制字段缺少或重复：'+k)
                return v[k][0]
            if not text.startswith('CALPUFF.INP     7.0') or one('LCFILES')!='F' or not re.fullmatch(r'[A-Za-z0-9_.-]+',one('METDAT')) or one('AUXEXT').upper()!='AUX':raise ValueError('当前需7.0控制头、LCFILES=F、安全METDAT及AUX')
            if one('LSAMP')!='F' or one('MSOURCE')!='1' or one('NRGRP')!='0' or one('MCHEM')!='0':raise ValueError('当前只验收无化学反应、分源输出的离散受体配置')
            for key in ('NPTDAT','NARDAT','NVOLDAT','NFLDAT','NRDDAT','NLNDAT','NPT2','NSPT1','NAR1','NAR2','NVL1','NVL2','NLN2','NRD1','NRD2','NFL2'):
                if int(one(key))!=0:raise ValueError('当前仅支持控制内常量点源：'+key)
            for key in ('PUFLST','CONDAT'):
                if not re.fullmatch(r'[A-Za-z0-9_.-]+',one(key)):raise ValueError('输出文件名无效')
            for key,low,high in [('NX',1,265),('NY',1,265),('NZ',1,100),('NREC',1,11000),('NPT1',1,200)]:
                if not low<=int(one(key))<=high:raise ValueError('当前入口规模超限：'+key)
            if len(v.get('SRCNAM',[]))!=int(one('NPT1')):raise ValueError('点源名称与数量不符')
    if import_report is not None:data['import_report.json']=json.dumps(import_report,ensure_ascii=False,indent=2).encode('utf8')
    return data,provenance

def directory(ident):
    if not re.fullmatch('[0-9a-f]{32}',ident):raise ValueError('任务号无效')
    return RUNS/ident

def status(ident):return json.loads((directory(ident)/'status.json').read_text(encoding='utf8'))

def convert_existing_table(payload,table,cfg,mode,control_path=None):
    """Bind raw CSV to the selected engineering control inside the platform."""
    if mode not in {'constant_points','receptors'} or cfg.get('mode')!=mode:raise ValueError('需明确且一致的点源或受体转换类别')
    data,_=unpack(payload,'existing',control_path)
    names=[cfg.get('table'),cfg.get('control_file')]
    if any(not isinstance(n,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',n) or '..' in n for n in names) or names[0]==names[1]:
        raise ValueError('转换配置需安全且不同的CSV和控制文件名')
    source=io.BytesIO()
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(names[0],table);z.writestr(names[1],data['CALPUFF.INP'])
    from ecodata.round1 import convert
    from ecodata.interface.calpuff_update_contract import validate_update
    files,report=convert('calpuff',source.getvalue(),cfg)
    contract=validate_update(data['CALPUFF.INP'],files['CALPUFF.INP'],mode)
    report.update(engineering_seed_sha256=hashlib.sha256(data['CALPUFF.INP']).hexdigest(),raw_table_sha256=hashlib.sha256(table).hexdigest(),update_contract=contract,
                  seed_policy='Selected existing project control bound inside platform; no external repacking or scientific defaults')
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
        z.writestr('engineering_conversion_report.json',json.dumps(report,ensure_ascii=False,indent=2))
    return output.getvalue(),report


def save(folder,record):
    from ecodata.interface.independent_cases import save as atomic_save
    atomic_save(folder,record)
    # Same atomic status format as the other project pages.

def start(payload,route,converted_zip=None,mode=None,control_path=None):
    from ecodata.interface import independent_cases as cases
    if converted_zip is not None and mode not in {'constant_points','receptors'}:raise ValueError('此入口仅接本次点源/受体转换控制，不会忽略其他类型')
    data,provenance=unpack(payload,route,control_path)
    with cases.LOCK:
        if cases.ACTIVE:raise ValueError('已有原生任务，需等完成；串行避免内存不足')
        ident=uuid.uuid4().hex;cases.ACTIVE=ident
    folder=directory(ident)
    try:
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'project_input.zip').write_bytes(payload)
        for name,content in data.items():(folder/name).write_bytes(content)
        if converted_zip is not None:(folder/'current_conversion.zip').write_bytes(converted_zip)
        record=dict(id=ident,state='running',stage='输入校验',route=route,provenance=provenance,input_sha256={n:hashlib.sha256(b).hexdigest() for n,b in data.items()},accuracy_evaluated=False,raw_dem_processing='geography_raw.zip' in data,stages=[])
        save(folder,record)
        threading.Thread(target=work,args=(folder,record,converted_zip,mode),daemon=True).start();return ident
    except Exception:
        with cases.LOCK:cases.ACTIVE=None
        raise

def execute(folder,record,converted_zip=None,mode=None):
    from ecodata.interface.independent_cases import SOURCE,linux
    from ecodata.round1 import convert
    def stage(label):record['stage']=label;save(folder,record)
    def native(cmd,cwd,logname,marker,allowed=(0,)):
        stage(logname)
        with (folder/logname).open('wb') as log:
            p=subprocess.run(cmd,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        text=(folder/logname).read_bytes().decode('latin1')
        record['stages'].append(dict(log=logname,returncode=p.returncode,marker_found=marker in text));save(folder,record)
        if p.returncode not in allowed or marker not in text:raise RuntimeError('原生阶段失败，保留日志：'+logname)
    def wsl(exe,*args,cwd=folder):return ['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--cd',linux(cwd),'--',exe,*args]
    def check_linux(exe,digest):
        p=subprocess.run(wsl('sha256sum',exe),capture_output=True,check=True)
        if p.stdout.decode().split()[0]!=digest:raise ValueError('原生程序哈希变化')
    if record['route']=='rebuild':
        if (folder/'geography_raw.zip').is_file():
            stage('平台原始DEM/土地覆盖栅格前处理与明确参考参数映射')
            from ecodata.interface.calmet_raw_geography import from_rasters
            generated,_=from_rasters((folder/'geography_raw.zip').read_bytes(),json.loads((folder/'geography_raw_config.json').read_text(encoding='utf8')),(folder/'CALMET.INP').read_bytes())
            for name,content in generated.items():(folder/name).write_bytes(content)
        stage('地理参数转换：'+('已在平台完成原始栅格前处理' if record.get('raw_dem_processing') else '使用已配置地理参数'))
        import xarray as xr
        from ecodata.models.adapters.calmet_geography import export_geography
        ds=xr.open_dataset(io.BytesIO((folder/'geography.nc').read_bytes()))
        files,report=export_geography(ds,json.loads((folder/'geography_config.json').read_text()))
        (folder/'geo.dat').write_bytes(files['GEO.DAT']);(folder/'geography_report.json').write_text(json.dumps(report,indent=2))
        # Supplied complete controls are scientific configuration, never guessed.
        met=control_values((folder/'CALMET.INP').read_text())
        expected={'GEODAT':'geo.dat','METDAT':'calmet.dat','METLST':'calmet.lst','M3DDAT':'wrf.m3d','NOOBS':'2','NM3D':'1','NUSTA':'0','NSSTA':'0','NOWSTA':'0'}
        if any(met.get(k)!=[v] for k,v in expected.items()):raise ValueError('当前WRF驱动配置不符；不更改用户科学设置')
        puff=control_values((folder/'CALPUFF.INP').read_text())
        for key in ('IBYR','IBMO','IBDY','IBHR','IEYR','IEMO','IEDY','IEHR','ABTZ','NX','NY','NZ','IUTMZN','UTMHEM','DATUM','DGRIDKM','XORIGKM','YORIGKM','ZFACE'):
            # Numerical formatting may differ; compare scalars/arrays by value.
            if len(met.get(key,[]))!=1 or len(puff.get(key,[]))!=1:raise ValueError('气象/扩散控制缺字段：'+key)
            try:equal=np.array_equal(np.array(met[key][0].split(','),dtype=float),np.array(puff[key][0].split(','),dtype=float))
            except ValueError:equal=met[key]==puff[key]
            if not equal:raise ValueError('气象与扩散设置不一致，不擅自修改：'+key)
        def utc(prefix):
            from datetime import timedelta
            z=puff['ABTZ'][0];m=re.fullmatch(r'UTC([+-])(\d{2})(\d{2})',z)
            if not m:raise ValueError('时区无效')
            t=datetime(*[int(puff[prefix+k][0]) for k in ('YR','MO','DY','HR')])
            return t-timedelta(minutes=(int(m[2])*60+int(m[3]))*(1 if m[1]=='+' else -1))
        # WRF file must supply exact hourly endpoints; CALWRF validates variables.
        from netCDF4 import Dataset
        with Dataset('memory',memory=(folder/'WRF.nc').read_bytes()) as d:
            times=[''.join(a.astype(str)) for a in np.asarray(d['Times'][:])]
        begin,end=utc('IB'),utc('IE');wanted=int((end-begin).total_seconds()/3600)
        from datetime import timedelta
        if not 1<=wanted<=24 or any((begin+timedelta(hours=i)).strftime('%Y-%m-%d_%H:%M:%S') not in times for i in range(wanted+1)):raise ValueError('WRF缺完整逐小时起止时刻；不补插')
        build=SOURCE.parent/'双任务推进_20261030/tools/calwrf_source_build'
        info=json.loads((build/'build.json').read_text());check_linux(linux(build/'calwrf'),info['exe_sha256'])
        (folder/'calwrf.inp').write_text('User WRF to prognostic meteorology\ncalwrf.lst\nwrf.m3d\n-1,-1,-1,-1,-1,-1\n'+begin.strftime('%Y%m%d%H')+'\n'+end.strftime('%Y%m%d%H')+'\n1\nWRF.nc\n',encoding='ascii')
        native(wsl(linux(build/'calwrf')),folder,'calwrf.log','CALWRF succeeded',allowed=(0,159))
        # STOP99999 is allowed only with downstream success and complete M3D.
        if not (folder/'wrf.m3d').is_file() or (folder/'wrf.m3d').stat().st_size<1000:raise ValueError('CALWRF未生成完整M3D')
        ev=json.loads((SOURCE/'evidence/Cambridge_CALMET_latest.json').read_text());exe=Path(ev['engine'])
        if hashlib.sha256(exe.read_bytes()).hexdigest()!=ev['engine_sha256']:raise ValueError('CALMET程序哈希变化')
        metdir=folder/'calmet';metdir.mkdir()
        import shutil
        for src,dest in [('CALMET.INP','calmet.inp'),('geo.dat','geo.dat'),('wrf.m3d','wrf.m3d')]:shutil.copyfile(folder/src,metdir/dest)
        native([str(exe),'calmet.inp'],metdir,'calmet.log','ENTERING TERMINATION PHASE')
        for name in ('calmet.dat','calmet.dat.aux'):shutil.copyfile(metdir/name,folder/name.upper())
        stage('平台点源与受体转换')
        cfg=json.loads((folder/'point_config.json').read_text());files,report=convert('calpuff',(folder/'points_input.zip').read_bytes(),cfg)
        # The source seed must be this project's complete control, not another grid.
        with zipfile.ZipFile(folder/'points_input.zip') as z:
            if z.read(cfg['control_file'])!=(folder/'CALPUFF.INP').read_bytes():raise ValueError('点源包的完整控制与本工程不同')
        (folder/'point_report.json').write_text(json.dumps(report,indent=2))
        cfg=json.loads((folder/'receptor_config.json').read_text());b=io.BytesIO()
        with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr(cfg['table'],(folder/'receptors.csv').read_bytes());z.writestr(cfg['control_file'],files['CALPUFF.INP'])
        files,report=convert('calpuff',b.getvalue(),cfg)
        (folder/'CALPUFF.INP').write_bytes(files['CALPUFF.INP']);(folder/'receptor_report.json').write_text(json.dumps(report,indent=2))
    if converted_zip is not None:
        from ecodata.interface.registered_conversion import member
        update=member(converted_zip,'CALPUFF.INP')
        from ecodata.interface.calpuff_update_contract import validate_update
        update_report=validate_update((folder/'CALPUFF.INP').read_bytes(),update,mode)
        (folder/'replacement_report.json').write_text(json.dumps(update_report,indent=2))
        # Keep both versions; rerun strict package validation on effective controls.
        (folder/'before_conversion.INP').write_bytes((folder/'CALPUFF.INP').read_bytes());(folder/'CALPUFF.INP').write_bytes(update)
    # Validate the effective input even after conversion, no silent fallback.
    b=io.BytesIO()
    effective=control_values((folder/'CALPUFF.INP').read_text())
    met=effective['METDAT'][0].upper();aux=effective['AUXEXT'][0].upper()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for n in COMMON|{met,met+'.'+aux}:z.writestr(n,(folder/n).read_bytes())
    unpack(b.getvalue(),'existing');(folder/'existing_project.zip').write_bytes(b.getvalue())
    engine=json.loads((SOURCE/'evidence/version_sources/build_calpuff721.json').read_text());check_linux(engine['exe'],engine['sha256'])
    (folder/'engine.json').write_text(json.dumps(engine,indent=2))
    native(wsl(engine['exe'],'CALPUFF.INP'),folder,'calpuff.log','TERMINATION PHASE')
    text=(folder/'CALPUFF.INP').read_text();v=control_values(text)
    concentration=folder/v['CONDAT'][0].upper();lst=folder/v['PUFLST'][0].upper()
    if not lst.is_file() or 'End of run' not in lst.read_text(encoding='latin1'):raise ValueError('缺原生完成日志')
    result=read_discrete(concentration.read_bytes(),text)
    names=result['species'];coords=result['coordinates'];stage('数值检查与出图')
    import csv
    with (folder/'concentrations.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['local_hour_ending','species','receptor','x_km','y_km','g_m3','ug_m3'])
        for s in names:
            for time,row in zip(result['times'],result['values_g_m3'][s]):
                for i,value in enumerate(row):w.writerow([time,s,i+1,coords[0,i],coords[1,i],float(value),float(value)*1e6])
    plot_result(folder,result)
    record.update(state='succeeded',stage='原生积分、读取与出图完成',periods=len(result['times']),receptors=coords.shape[1],species=names,maximum_ug_m3={s:float(result['values_g_m3'][s].max()*1e6) for s in names},finite_nonnegative=True,concentration_sha256=hashlib.sha256(concentration.read_bytes()).hexdigest())


def plot_result(folder,result):
    """Plot existing native values; shorten display dates without changing CSV."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    labels=[datetime.fromisoformat(t).strftime('%m-%d %H:%M') for t in result['times']]
    fig=Figure(figsize=(9,4));FigureCanvasAgg(fig);ax=fig.subplots()
    for s in result['species']:ax.plot(labels,result['values_g_m3'][s].max(axis=1)*1e6,'o-',label=s)
    ax.set(ylabel='Maximum concentration (ug/m3)',title='Native CALPUFF7 result; accuracy not assessed',xlabel='Local period ending');ax.set_ylim(bottom=0);ax.legend();fig.tight_layout();fig.savefig(folder/'result.png',dpi=150)


def work(folder,record,converted_zip=None,mode=None):
    from ecodata.interface import independent_cases as cases
    try:execute(folder,record,converted_zip,mode)
    except Exception as exc:record.update(state='failed',error=str(exc))
    finally:
        save(folder,record)
        with cases.LOCK:
            if cases.ACTIVE==record['id']:cases.ACTIVE=None

def render(converted_zip=None,mode=None):
    import streamlit as st
    st.header('CALPUFF7：WRF驱动新建 / 已有工程接入（试验）')
    labels=['已有CALMET气象与CALPUFF工程，接入并运行','从WRF与地理参数资料新建工程']
    selected=st.radio('工程资料',labels,key='calpuff_project_route');route='existing' if selected==labels[0] else 'rebuild'
    st.warning('限定常量点源、无化学反应、离散受体、CALPUFF7.2.1公开分支。新建起点为WRF气象与完整科学控制，可使用已配置地理参数，或明确上传原始DEM/土地覆盖及参考参数对照由平台前处理；不自动推断科学配置，不做数据融合或精度认证。')
    st.write('上传资料 → 检查来源与配置 → '+('平台地理转换 → CALWRF → CALMET → 平台点源/受体转换 → ' if route=='rebuild' else '复用上传CALMET，接入本次转换 → ')+'CALPUFF原生积分 → 数值检查 → 图表和下载')
    st.caption('新建ZIP需单层文件：'+', '.join(sorted(COMMON|REBUILD)) if route=='rebuild' else '已有工程ZIP可保留原目录与气象文件名；平台按CALPUFF控制的METDAT/AUXEXT寻找同目录资料。来源声明缺失会标记未知，不要求用户手工补写。')
    from ecodata.interface.independent_cases import SOURCE
    sample=SOURCE.parent/'双任务推进_20261030'/('Apex_WRF_grid_parameter_build.zip' if route=='rebuild' else 'Apex_existing_project.zip')
    if sample.is_file():
        st.download_button('下载已验证Apex情景资料（2018烟囱＋2021排放）',sample.read_bytes(),sample.name,key='calpuff_project_sample')
    uploaded=st.file_uploader('完整工程资料ZIP',type=['zip'],key='calpuff_project_upload')
    ready=False
    control_path=None
    if uploaded is not None:
        try:
            if route=='existing':
                from ecodata.interface.wrf_external_project import read_archive
                entries=read_archive(uploaded.getvalue())
                controls=[n for n,b in entries.items() if b.startswith(b'CALPUFF.INP     7.0')]
                if len(controls)>1:control_path=st.selectbox('选择本次CALPUFF工程控制文件',controls,index=None,key='calpuff_control_path')
                if len(controls)>1 and control_path is None:
                    st.info('压缩包有多个工程，请明确选择，不自动猜选。');return
            data,provenance=unpack(uploaded.getvalue(),route,control_path,raw_geography_inputs=route=='rebuild' and st.session_state.get('calpuff_raw_gis',False));ready=True
            for key,label in [('sources','上传资料的来源声明（需用户核实）'),('defaults','采用的参考参数与明确情景选择'),('limitations','适用范围与未验证项')]:
                st.markdown('**'+label+'**')
                for item in provenance[key]:st.write('• '+item)
            if 'import_report.json' in data:
                from ecodata.interface.engineering_completeness import render as render_completeness
                ready=render_completeness(json.loads(data['import_report.json'])['completeness'],'calpuff')
                with st.expander('实际文件引用与原文保留报告'):st.json(json.loads(data['import_report.json']))
                st.download_button('下载工程识别与文件映射报告',data['import_report.json'],'calpuff_import_report.json',key='calpuff_import_report')
        except Exception as exc:
            if hasattr(exc,'completeness'):
                from ecodata.interface.engineering_completeness import render as render_completeness
                render_completeness(exc.completeness,'calpuff')
            st.error(str(exc))
    raw_gis=False;gis_files=[];gis_cfg=None
    if route=='rebuild':
        raw_gis=st.checkbox('在平台内用原始DEM和土地覆盖生成地理输入（试验）',key='calpuff_raw_gis')
        if raw_gis:
            st.caption('原样上传GeoTIFF瓦片、MAKEGEO参数原文件和明确分类对照JSON。平台按CALMET目标网格处理：DEM平均、类别众数；缺测拒绝，不补值。参考地表参数不是当地实测；原包的geography.nc会被移除，不作为回退。上游WRF仍为已有气象输出，原点源/受体高程保留。')
            st.caption('本选项的基础ZIP无需geography.nc/geography_config.json；只需上述其余控制、WRF、点源/受体及来源文件。原基础包内若含旧地理输入，平台会丢弃旧地理输入后重新生成。')
            gis_files=st.file_uploader('原始GIS瓦片与参考参数文件',type=['tif','tiff','inp'],accept_multiple_files=True,key='calpuff_raw_gis_files')
            gis_cfg=st.file_uploader('原始GIS处理与分类对照JSON',type=['json'],key='calpuff_raw_gis_config')
            ready=ready and bool(gis_files) and gis_cfg is not None
    from ecodata.interface.input_result_binding import bind_result
    gis_identity=json.dumps([(f.name,hashlib.sha256(f.getvalue()).hexdigest()) for f in gis_files])
    bind_result(st.session_state,'calpuff_project_job','calpuff_project_binding',uploaded.getvalue() if uploaded else b'',route,converted_zip or b'',mode,control_path,raw_gis,gis_identity,gis_cfg.getvalue() if gis_cfg else b'')
    if st.button('检查资料并启动本次工程',disabled=not ready,key='calpuff_project_start'):
        st.session_state.pop('calpuff_project_job',None)
        try:
            payload=uploaded.getvalue()
            if raw_gis:
                if len({f.name for f in gis_files})!=len(gis_files):raise ValueError('GIS文件名重复')
                from ecodata.interface.calmet_raw_geography import attach_raw_geography
                payload,_=attach_raw_geography(payload,{f.name:f.getvalue() for f in gis_files},json.loads(gis_cfg.getvalue().decode('utf-8-sig')))
            st.session_state['calpuff_project_job']=start(payload,route,converted_zip,mode,control_path)
        except Exception as exc:st.error(str(exc))
    if route=='existing' and ready and converted_zip is None:
        with st.expander('在本工程内接入新的原始点源或受体表（可选）'):
            st.caption('平台将原始CSV与本工程控制关联后调用同一转换器，只替换所选组。无需用户在平台外重打包或改控制。')
            update_mode=st.selectbox('本工程新表类型',['receptors','constant_points'],key='calpuff_bound_mode')
            table=st.file_uploader('本工程新原始CSV',type=['csv'],key='calpuff_bound_table')
            config=st.file_uploader('新表字段单位与来源配置JSON',type=['json'],key='calpuff_bound_config')
            bind_result(st.session_state,'calpuff_project_job','calpuff_bound_binding',table.getvalue() if table else b'',config.getvalue() if config else b'',update_mode)
            bind_result(st.session_state,'calpuff_bound_conversion','calpuff_bound_conversion_binding',uploaded.getvalue(),control_path,table.getvalue() if table else b'',config.getvalue() if config else b'',update_mode)
            if st.button('转换新表并接入本工程运行',disabled=table is None or config is None,key='calpuff_bound_start'):
                st.session_state.pop('calpuff_project_job',None)
                try:
                    converted,report=convert_existing_table(uploaded.getvalue(),table.getvalue(),json.loads(config.getvalue().decode('utf-8-sig')),update_mode,control_path)
                    st.session_state['calpuff_bound_conversion']=(converted,report)
                    st.session_state['calpuff_project_job']=start(uploaded.getvalue(),route,converted,update_mode,control_path)
                except Exception as exc:st.error(str(exc))
            current=st.session_state.get('calpuff_bound_conversion')
            if current:
                st.download_button('下载本次点源或受体转换包',current[0],'CALPUFF_current_conversion.zip',key='calpuff_bound_zip')
                with st.expander('本次新表转换与工程更新范围'):st.json(current[1])
    ident=st.session_state.get('calpuff_project_job')
    if ident and ready:
        @st.fragment(run_every=3)
        def show():
            record=status(ident);folder=directory(ident);st.write('任务 '+ident+'：'+record['stage'])
            if record['state']=='succeeded':
                st.success('本次原生积分和读回完成；未评估预测精度。');st.image(str(folder/'result.png'))
                st.download_button('下载本次浓度CSV',(folder/'concentrations.csv').read_bytes(),'calpuff_concentrations.csv',key='calpuff_project_csv')
                st.download_button('下载本次CALPUFF结果PNG',(folder/'result.png').read_bytes(),'CALPUFF_result.png',key='calpuff_project_png')
                st.download_button('下载本次CALPUFF运行记录',(folder/'status.json').read_bytes(),'CALPUFF_status.json',key='calpuff_project_status')
                st.download_button('保存本次已有工程包',(folder/'existing_project.zip').read_bytes(),'CALPUFF_existing_project.zip',key='calpuff_project_export')
            elif record['state']=='failed':st.error(record.get('error'))
            with st.expander('来源、阶段和结果记录'):st.json(record)
            for log in folder.glob('*.log'):
                with st.expander(log.name):st.code(log.read_bytes().decode('latin1')[-12000:])
        show()
    return True
