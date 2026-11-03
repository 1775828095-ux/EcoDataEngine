"""CALMET/CALPUFF 5.8 formatted native-engineering adapter.

Recognises content and dependency closure, not a pinned case name/date/hash.
Uploaded programs/scripts and previous model outputs are never executed/reused.
Existing trusted surface writer and independent concentration reader are reused.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime,timedelta,timezone
import hashlib,io,json,os,re,subprocess,sys,zipfile
import numpy as np
import pandas as pd
import xarray as xr

ROUTE='calpuff_native_58_v1'
FIELDS=('wind_speed','wind_direction','ceiling','opaque_cover','temperature','relative_humidity','station_pressure','precip_code')
UNITS=('m/s','degree','hundred_ft','tenths','K','%','hPa','code')
MASK='retain source mask; CALMET SURF 2.0 encodes missing as 9999'
ENGINE_ROOT=Path('D:/EcoDataEngine/work/calpuff_official')
ENGINES={
 'CALMET':('CALMET_v5.8.5_verified/calmet.exe','083a426539d18414ada7b7022eed060a03ececa82aee4d231892776f707a335c'),
 'CALPUFF':('CALPUFF_v5.8.4/CALPUFF/calpuff.exe','cd8f0445b1bd6b2b3a216bcc470bd097694b240b0ac50dc43757cfbdd69786b7')}

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def values(raw):
    from ecodata.models.native_control_readers import control_values
    return control_values(raw.decode('ascii') if isinstance(raw,bytes) else raw)

def scalar(v,k,default=None):
    a=v.get(k,[])
    if len(a)>1:raise ValueError('控制参数重复：'+k)
    if not a:
        if default is not None:return str(default)
        raise ValueError('缺少控制参数：'+k)
    value=a[0].strip()
    return value[1:-1] if len(value)>1 and value[0]==value[-1] and value[0] in "'\"" else value

def integer(v,k,default=None):return int(scalar(v,k,default))

def safe_name(name):
    p=PurePosixPath(str(name).replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or ':' in str(p) or not p.parts:
        raise ValueError('原生文件引用不能是外部路径：'+str(name))
    return str(p)

def native_name(control,key):
    name=safe_name(scalar(control,key))
    return name.lower() if scalar(control,'LCFILES','T').upper()=='T' else name.upper()

def read_surface(raw):
    """Strict formatted SURF 2.0 / NONE reader, with explicit 9999 mask."""
    lines=[s.strip() for s in raw.decode('ascii').splitlines() if s.strip()]
    if len(lines)<6 or not re.match(r'^SURF\.DAT\s+2\.0\b',lines[0],re.I):raise ValueError('仅支持SURF.DAT 2.0格式')
    comments=int(lines[1]);i=2+comments
    if comments<0 or comments>100 or lines[i].upper()!='NONE':raise ValueError('SURF投影头不是已支持的NONE格式')
    header=list(map(int,lines[i+1].split()))
    if len(header)!=8:raise ValueError('SURF日期/时区/站点数头不完整')
    sy,sd,sh,ey,ed,eh,zone,n=header
    if not -12<=zone<=12 or not 1<=n<=10000:raise ValueError('SURF站点数或时区无效')
    tz=timezone(timedelta(hours=-zone))
    def stamp(y,d,h):
        if not 1900<=y<=2099 or not 1<=d<=366 or not 0<=h<=24:raise ValueError('SURF日期非法')
        t=datetime(y,1,1,tzinfo=tz)+timedelta(days=d-1,hours=h)
        if (datetime(y,1,1)+timedelta(days=d-1)).year!=y:raise ValueError('SURF年积日非法')
        return t
    start,end=stamp(sy,sd,sh),stamp(ey,ed,eh)
    i+=2;stations=[]
    while len(stations)<n:
        stations.extend(map(int,lines[i].split()));i+=1
    if len(stations)!=n or len(set(stations))!=n:raise ValueError('SURF站点重复或数量不符')
    clocks=[];data=[]
    while i<len(lines):
        parts=list(map(int,lines[i].split()));i+=1
        if len(parts)!=3:raise ValueError('SURF时标长度不正确')
        t=stamp(*parts);rows=[]
        for _ in stations:
            row=list(map(float,lines[i].split()));i+=1
            if len(row)!=8 or not np.isfinite(row).all():raise ValueError('SURF记录应为8个有限原生数值')
            rows.append(row)
        clocks.append(t);data.append(rows)
    expected=list(pd.date_range(start,end,freq='h').to_pydatetime())
    if clocks!=expected:raise ValueError('SURF缺小时、重复时间或头尾覆盖不一致')
    array=np.array(data,dtype='float64');array[array==9999]=np.nan
    return array,clocks,stations,zone

def dependency_plan(archive):
    """Select one distinct complete input closure; identical copied demos dedupe."""
    members={}
    for info in archive.infolist():
        if info.is_dir():continue
        name=safe_name(info.filename)
        if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('不接受压缩包符号链接')
        if name.casefold() in members:raise ValueError('工程成员大小写重复：'+name)
        members[name.casefold()]=name
    def find(control,reference):
        ref=safe_name(reference);key=str(PurePosixPath(control).parent/ref).casefold()
        if key not in members:raise ValueError('缺少工程输入：'+str(PurePosixPath(control).parent/ref))
        return members[key]
    candidates={'CALMET':[],'CALPUFF':[]};rejected=[]
    for name in members.values():
        if not name.lower().endswith('.inp'):continue
        raw=archive.read(name)
        try:v=values(raw)
        except (UnicodeError,ValueError):continue
        kind='CALMET' if all(k in v for k in ('GEODAT','SRFDAT','IPROG')) else 'CALPUFF' if all(k in v for k in ('CONDAT','NPT1','METDAT')) else None
        if not kind:continue
        try:
            refs=[]
            def add(k,default=None):
                ref=scalar(v,k,default);refs.append(dict(key=k,reference=ref,member=find(name,ref)))
            if kind=='CALMET':
                if integer(v,'IFORMS')!=2 or integer(v,'IRTYPE')!=1 or integer(v,'NOOBS')!=0:raise ValueError('当前路线要求IFORMS=2、IRTYPE=1、NOOBS=0')
                if integer(v,'IPROG') not in (0,4):raise ValueError('当前路线支持IPROG=0/4；其他预报输入需对应读取器')
                if integer(v,'IGFMET',0)!=0 or integer(v,'ICLOUD',0)!=0:raise ValueError('当前路线尚不支持外部猜测场或单独云场')
                for key in ('GEODAT','SRFDAT'):add(key)
                if integer(v,'NPSTA',0)>0:add('PRCDAT')
                for key,count in [('UPDAT',integer(v,'NUSTA')),('SEADAT',integer(v,'NOWSTA'))]:
                    if len(v.get(key,[]))!=count:raise ValueError(key+'文件数与控制文件不符')
                    for ref in v.get(key,[]):refs.append(dict(key=key,reference=ref,member=find(name,ref)))
                if integer(v,'IPROG')==4:
                    if integer(v,'NM3D',0)>0:
                        if len(v.get('M3DDAT',[]))!=integer(v,'NM3D'):raise ValueError('M3DDAT数目与NM3D不符')
                        for ref in v['M3DDAT']:refs.append(dict(key='M3DDAT',reference=ref,member=find(name,ref)))
                    else:add('MM4DAT','mm4.dat') # documented CALMET native default, not a generated field
            else:
                for key in ('NPTDAT','NARDAT','NVOLDAT','MRESTART','NHILL','NCTREC'):
                    if integer(v,key,0)!=0:raise ValueError('当前路线尚未支持启用的附加输入：'+key)
                if integer(v,'NMETDAT',1)!=1 or integer(v,'METRUN',0)!=0:raise ValueError('当前路线要求单个CALMET输入和METRUN=0')
                if integer(v,'MOZ',0)==1:add('OZDAT')
            items=[dict(member=name,sha256=hashlib.sha256(raw).hexdigest())]
            for r in refs:r['sha256']=hashlib.sha256(archive.read(r['member'])).hexdigest();items.append(dict(member=r['key'],sha256=r['sha256']))
            signature=hashlib.sha256(json.dumps([dict(member='CONTROL',sha256=items[0]['sha256']),*items[1:]],sort_keys=True).encode()).hexdigest()
            candidates[kind].append(dict(control=name,control_sha256=items[0]['sha256'],values=v,inputs=refs,signature=signature))
        except (ValueError,KeyError) as e:rejected.append(dict(control=name,reason=str(e)))
    selected={}
    for kind,choices in candidates.items():
        unique={}
        for c in sorted(choices,key=lambda c:c['control']):unique.setdefault(c['signature'],c)
        if len(unique)!=1:raise ValueError(f'{kind}需要一个明确完整工程，找到{len(unique)}种不同输入组合；'+json.dumps(rejected,ensure_ascii=False))
        selected[kind]=next(iter(unique.values()))
        selected[kind]['equivalent_control_copies']=[c['control'] for c in choices]
    met,puff=(selected[k]['values'] for k in ('CALMET','CALPUFF'))
    for key in ('IBYR','IBMO','IBDY','IBHR','IRLG','NX','NY','NZ','DGRIDKM','XORIGKM','YORIGKM','IUTMZN','UTMHEM','FEAST','FNORTH','PMAP','DATUM','ZFACE'):
        a=re.sub(r'\s+','',scalar(met,key));b=re.sub(r'\s+','',scalar(puff,key))
        try:equal=np.array_equal(np.array(a.split(','),float),np.array(b.split(','),float))
        except ValueError:equal=a==b
        if not equal:raise ValueError('CALMET/CALPUFF时间/网格不匹配：'+key)
    if float(scalar(met,'IBTZ'))!=float(scalar(puff,'XBTZ')):raise ValueError('两个模型时区不一致')
    for k in ('CALMET','CALPUFF'):
        v=selected[k]['values'];keys=('METDAT','METLST') if k=='CALMET' else ('METDAT','CONDAT','PUFLST','DFDAT','WFDAT','VISDAT')
        for key in keys:safe_name(scalar(v,key))
    return selected,rejected,list(members.values())

def contract(manifest):
    r=manifest.get('preparation',{})
    if r.get('route')!=ROUTE:return None
    return dict(schema='ecodata.converter.requirements.v1',model_id='calpuff',role='surface_meteorology',
        scope='CALMET formatted SURF 2.0 + uploaded complete CALPUFF 5.8 engineering; no accuracy claim',
        fields=[dict(name=f,unit=u,required=True) for f,u in zip(FIELDS,UNITS)],
        native_missing_policies={f:MASK for f in FIELDS},spatial_reference_mode='native_station_identifiers',
        station_binding_sha256=r['station_binding_sha256'],
        coordinates=dict(time=dict(datetime=True,interval_seconds=3600,dimension=True),
            station_id=dict(size=len(r['station_ids']),dimension=True)),
        dependencies=[dict(name=k,path=str(ENGINE_ROOT/p),sha256=h) for k,(p,h) in ENGINES.items()],
        metadata=dict(time_basis='UTC'),required_references=['station_id'])

def prepare(uploads,folder,progress=None):
    if len(uploads)!=1 or not str(uploads[0].name).lower().endswith('.zip'):return None
    folder=Path(folder);path=getattr(uploads[0],'path',None)
    if path is None:
        path=folder/'uploaded_engineering.zip';path.write_bytes(uploads[0].getvalue())
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        # Content signature, not a case filename or archive hash.
        controls=[n for n in names if n.lower().endswith('.inp')]
        if not any(b'SRFDAT' in archive.read(n).upper() and b'GEODAT' in archive.read(n).upper() for n in controls):return None
        if progress:progress('识别完整CALMET/CALPUFF原生工程，按控制文件检查实际依赖；不执行包内程序')
        if archive.testzip() is not None:raise ValueError('原始ZIP内部CRC校验失败')
        plan,rejected,names=dependency_plan(archive)
        met=plan['CALMET'];surface=next(x for x in met['inputs'] if x['key']=='SRFDAT')
        a,clocks,stations,zone=read_surface(archive.read(surface['member']))
        v=met['values'];station_table=[]
        for i in range(1,integer(v,'NSSTA')+1):
            row=scalar(v,'SS'+str(i));match=re.fullmatch(r"'[^']*'\s+(\d+)\s+([\d.+Ee-]+)\s+([\d.+Ee-]+)\s+([+-]?\d+)\s+([\d.+Ee-]+)",row)
            if not match:raise ValueError('无法解析控制文件SS站点定位：'+str(i))
            station_table.append(dict(id=int(match[1]),x_km=float(match[2]),y_km=float(match[3]),timezone_west_hours=int(match[4]),anemometer_height_m=float(match[5]),original=row))
        if stations!=[r['id'] for r in station_table]:raise ValueError('SURF站点顺序与CALMET控制站表不一致')
        binding=hashlib.sha256(json.dumps(dict(stations=station_table,control_sha256=met['control_sha256']),sort_keys=True).encode()).hexdigest()
        start=datetime(integer(v,'IBYR'),integer(v,'IBMO'),integer(v,'IBDY'),tzinfo=timezone(timedelta(hours=-zone)))+timedelta(hours=integer(v,'IBHR'))
        expected=[start+timedelta(hours=i) for i in range(integer(v,'IRLG'))]
        if float(scalar(v,'IBTZ'))!=zone or not all(t in clocks for t in expected):raise ValueError('SURF时间/时区没有完整覆盖模拟窗口')
        inputs=folder/'native_inputs';inputs.mkdir()
        dependencies=[]
        for kind,c in plan.items():
            target=inputs/kind;target.mkdir()
            lc=scalar(c['values'],'LCFILES','T').upper()=='T'
            control=target/('control.inp');control.write_bytes(archive.read(c['control']));c['control_path']=control.relative_to(folder).as_posix()
            records=[dict(path=c['control_path'],sha256=digest(control),role=kind+'_CONTROL',member=c['control'])]
            for r in c['inputs']:
                name=safe_name(r['reference']);name=name.lower() if lc else name.upper()
                dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.read(r['member']))
                r['path']=dest.relative_to(folder).as_posix();records.append(dict(path=r['path'],sha256=digest(dest),role=r['key'],member=r['member']))
            dependencies+=records
    product=folder/'surface_product.nc'
    utc=np.array([np.datetime64(t.astimezone(timezone.utc).replace(tzinfo=None),'ns') for t in clocks])
    ds=xr.Dataset({f:(('time','station_id'),a[:,:,i],dict(units=u,missing_policy=MASK,native_missing_value=9999)) for i,(f,u) in enumerate(zip(FIELDS,UNITS))},
        coords=dict(time=utc,station_id=stations),attrs=dict(time_basis='UTC',station_binding_sha256=binding,source_timezone_west_hours=zone))
    receipt=dict(route=ROUTE,source_archive=str(path),source_archive_sha256=digest(path),archive_crc='PASS',
        native_plan=plan,rejected_control_candidates=rejected,station_ids=stations,station_binding_sha256=binding,
        station_table=station_table,timezone_west_hours=zone,run_hour_ends=[t.isoformat() for t in expected],
        source_hours=len(clocks),source_records=len(clocks)*len(stations),
        missing_counts={f:int(np.isnan(a[:,:,i]).sum()) for i,f in enumerate(FIELDS)},
        tasks=[dict(role='surface_meteorology',field=f,unit=u,result=dict(status='UNCHANGED_COMPLETE_INPUT',
            missing_mask_preserved=True,scope='Native-formatted data normalisation; missing source values remain missing')) for f,u in zip(FIELDS,UNITS)],
        handoff_dependencies=dependencies,original_data_modified=False,uploaded_scripts_executed=False,
        native_model_executed=False,full_model_ready=False)
    raw=folder/'surface_source.nc';ds.to_netcdf(raw,engine='h5netcdf')
    dependencies.append(dict(path=raw.name,sha256=digest(raw)))
    # Every native source is evaluated before conversion. The native-engineering
    # hook may decode formats, but may not predeclare that fusion is unnecessary.
    from .calpuff_fusion_stage import process_surface
    try:ds,fusion,component_results=process_surface(ds,receipt,folder,progress)
    except Exception as exc:
        failed=folder/'fusion_stage/decision.json'
        decision=json.loads(failed.read_text(encoding='utf8')) if failed.is_file() else {}
        save(failed,dict(decision,status='BLOCK',error=str(exc),conversion_started=False))
        raise ValueError('平台融合阶段未通过：'+str(exc)+'；详见 '+str(failed)) from exc
    ds.to_netcdf(product,engine='h5netcdf')
    receipt['fusion_stage']=fusion
    receipt['fusion_decision']=dict(status='PLATFORM_ENGINE_RETURNED' if component_results else 'NOT_APPLICABLE_SAME_PRODUCT',
        algorithm_search_executed=bool(component_results),reason=fusion.get('reason',fusion.get('selection_rule')))
    if component_results:
        components=[dict(field=f,role='surface_vector_component',result=r,product=r['product']) for f,r in component_results.items()]
        for task in receipt['tasks']:
            if task['field'] in {'wind_speed','wind_direction'}:
                task['result']=dict(status='DERIVED_FROM_COMPONENT_SEARCH',component_tasks=components,
                    transform='speed=hypot(eastward,northward); direction=atan2(-eastward,-northward) meteorological degrees; calm direction=0',
                    original_observations_outside_fixed_support_retained=True)
    for p in (folder/'fusion_stage').rglob('*'):
        if p.is_file():dependencies.append(dict(path=p.relative_to(folder).as_posix(),sha256=digest(p)))
    used={x['member'] for x in dependencies if 'member' in x}
    save(folder/'bundle_inventory.json',dict(files=[dict(member=n,role='active_input_or_control' if n in used else
        'software_or_reference_not_executed' if n.lower().endswith(('.exe','.for','.bat','.zip')) else 'unreferenced_reference_or_prior_output_not_used') for n in names],
        policy='Only declared input dependencies enter a fresh native work directory; no archived model output is reused'))
    save(folder/'native_plan.json',receipt);receipt['handoff_dependencies'].append(dict(path='native_plan.json',sha256=digest(folder/'native_plan.json')))
    if progress:progress(f'本次{len(clocks)}小时×{len(stations)}站数据已规范化，保留原缺测；调用已有写出器，随后实际运行CALMET和CALPUFF')
    from ecodata.models.conversion_bridge import finish_preparation
    return finish_preparation(folder,'calpuff',[product]*len(FIELDS),receipt)

def convert(manifest_path,output_dir):
    manifest_path=Path(manifest_path).resolve();base=manifest_path.parent;out=Path(output_dir).resolve();out.mkdir(exist_ok=True)
    m=json.loads(manifest_path.read_text(encoding='utf8'));r=m['preparation'];reads=[]
    if r.get('route')!=ROUTE or m.get('model_id')!='calpuff':raise ValueError('错误的CALPUFF转换路线')
    fields=[];arrays=[]
    for item in m['products']:
        p=(base/item['path']).resolve();f=item['field']
        if not p.is_relative_to(base) or digest(p)!=item['sha256']:raise ValueError('本次处理产品越界或哈希变化')
        with xr.open_dataset(p) as d:arrays.append(d[[f]].load())
        fields.append(f);reads.append(dict(path=str(p),sha256=item['sha256'],field=f))
    if fields!=list(FIELDS):raise ValueError('本次8个地面变量不完整')
    ds=xr.merge(arrays,join='exact',compat='equals')
    if list(ds.station_id.values)!=r['station_ids'] or ds.attrs.get('station_binding_sha256')!=r['station_binding_sha256']:raise ValueError('当前产品站点绑定变化')
    for dep in r['handoff_dependencies']:
        p=(base/dep['path']).resolve()
        if not p.is_relative_to(base) or digest(p)!=dep['sha256']:raise ValueError('本次工程依赖哈希变化：'+dep['path'])
    if digest(r['source_archive'])!=r['source_archive_sha256']:raise ValueError('原始上传ZIP发生变化')
    # Every field below comes from THIS handoff, not the retained raw SURF file.
    frame=ds.to_dataframe().reset_index()
    frame['time']=[pd.Timestamp(t).tz_localize('UTC').isoformat() for t in frame.time]
    frame.to_csv(out/'current_surface.csv',index=False)
    cfg=dict(version='CALMET-5.8.5-SURF-2.0',timezone_west_hours=r['timezone_west_hours'],time_label='hour_end',
        station_ids=r['station_ids'],missing_policy='preserve',columns={f:f for f in FIELDS},units=dict(zip(FIELDS,UNITS)))
    bindings=json.loads(Path(__file__).with_name('bindings.json').read_text(encoding='utf8'))
    job=dict(action='surface',frame=str(out/'current_surface.csv'),config=cfg,output=str(out/'SURF.DAT'))
    worker=Path(__file__).with_name('calpuff_native_worker.py')
    def run_worker(job,label):
        p=out/(label+'_request.json');save(p,job)
        with (out/(label+'.log')).open('wb') as log:
            proc=subprocess.run([bindings['legacy_python'],str(worker),str(p)],stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=bindings['legacy_source'],PYTHONIOENCODING='utf8',PYTHONNOUSERSITE='1'),
                timeout=300,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        if proc.returncode:raise ValueError(label+'失败：'+(out/(label+'.log')).read_text(encoding='utf8',errors='replace')[-2400:])
    run_worker(job,'surface_export')
    actual,times,ids,zone=read_surface((out/'SURF.DAT').read_bytes())
    for i,f in enumerate(FIELDS):
        expected=ds[f].transpose('time','station_id').values
        if not np.array_equal(np.isnan(actual[:,:,i]),np.isnan(expected)) or not np.allclose(actual[:,:,i],expected,equal_nan=True,rtol=0,atol=0.000001):raise ValueError('原生SURF回读与本次产品不符：'+f)
    if ids!=r['station_ids'] or zone!=r['timezone_west_hours'] or not np.array_equal(ds.time.values,np.array([np.datetime64(t.astimezone(timezone.utc).replace(tzinfo=None),'ns') for t in times])):raise ValueError('原生SURF回读站点或时间不符')
    lineage=dict(manifest_sha256=digest(manifest_path),products_read=reads,native_surface_sha256=digest(out/'SURF.DAT'),
        source_archive_sha256=r['source_archive_sha256'],surface_readback_equal=True,source_missing_masks_preserved=True,
        scientific_settings_modified=False,uploaded_executables_used=False,old_model_outputs_reused=False)
    save(out/'lineage.json',lineage)
    # Fresh per-stage directories; only actual uploaded INPUT bytes are copied.
    import shutil
    stages=[]
    for kind in ('CALMET','CALPUFF'):
        c=r['native_plan'][kind];work=out/kind;work.mkdir(exist_ok=False)
        shutil.copyfile(base/c['control_path'],work/'control.inp')
        for dep in c['inputs']:
            if kind=='CALMET' and dep['key']=='SRFDAT':source=out/'SURF.DAT'
            else:source=base/dep['path']
            ref=safe_name(dep['reference']);ref=ref.lower() if scalar(c['values'],'LCFILES','T').upper()=='T' else ref.upper()
            dest=work/ref;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        if kind=='CALPUFF':
            met=r['native_plan']['CALMET']['values'];generated=out/'CALMET'/native_name(met,'METDAT')
            target=work/native_name(c['values'],'METDAT');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(generated,target)
            lineage['new_CALMET_consumed_by_CALPUFF']=dict(source=str(generated),target=str(target),sha256=digest(target))
        exe,pin=ENGINES[kind];exe=ENGINE_ROOT/exe
        if digest(exe)!=pin:raise ValueError('本机可信程序哈希不符：'+kind)
        command=[str(exe),'control.inp']
        with (work/'console.log').open('wb') as log:
            proc=subprocess.run(command,cwd=work,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=300,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        # Keep direct diagnostics in failed-download packages as well.
        shutil.copyfile(work/'console.log',out/(kind+'_console.log'))
        listing=work/native_name(c['values'],'METLST' if kind=='CALMET' else 'PUFLST')
        if listing.is_file():shutil.copyfile(listing,out/(kind+'_listing.log'))
        stages.append(dict(program=kind,command=command,executable_sha256=pin,returncode=proc.returncode))
        save(out/'native_run.json',dict(stages=stages,completed=False))
        if proc.returncode:raise ValueError(kind+'原生程序返回码'+str(proc.returncode)+'；详见'+str(work/'console.log'))
        if kind=='CALMET':
            console=(work/'console.log').read_text(encoding='latin1')
            if 'ENTERING TERMINATION PHASE' not in console or not (work/native_name(c['values'],'METDAT')).is_file():raise ValueError('CALMET未完整结束，保留本次日志')
        else:
            listing=work/native_name(c['values'],'PUFLST')
            text=listing.read_text(encoding='latin1')
            if 'LAST DAY/HOUR PROCESSED:' not in text or 'End of run -- Clock time:' not in text:raise ValueError('CALPUFF未完整结束，保留本次日志')
    conc=out/'CALPUFF'/native_name(r['native_plan']['CALPUFF']['values'],'CONDAT')
    run_worker(dict(action='results',source=str(conc),output=str(out),expected_hours=len(r['run_hour_ends']),
        expected_times=r['run_hour_ends'],expected_species=r['native_plan']['CALPUFF']['values']['CSPEC']), 'independent_concentration_read')
    numbers=json.loads((out/'numerical_check.json').read_text(encoding='utf8'))
    save(out/'lineage.json',lineage)
    listing=(out/'CALMET_listing.log').read_text(encoding='latin1').splitlines()
    warning_lines=[]
    for i,line in enumerate(listing):
        if 'warning' in line.lower() or 'reset' in line.lower():warning_lines.append('\n'.join(listing[max(0,i-1):i+5]))
    save(out/'model_warnings.json',dict(control_files_modified=False,model_generated_warnings=warning_lines,
        qualification='Model-internal resets are preserved in the native listing; unchanged control files do not mean no internal model adjustment'))
    save(out/'native_run.json',dict(stages=stages,completed=True,scientific_settings_modified=False,
        scope='CALMET 5.8.5 + CALPUFF 5.8.4; original uploaded controls, hypothetical demonstration emissions'))
    validation=dict(status='PASS',native_model_executed=True,full_model_ready=True,stages=stages,numerical=numbers,
        surface_readback_equal=True,source_missing_masks_preserved=True,accuracy_assessed=False,
        input_dependencies=r['handoff_dependencies'],lineage=lineage,model_warnings=warning_lines,
        scope='This complete 5.8 native-engineering case only; no universal format or accuracy claim')
    save(out/'validation.json',validation)
    files=[p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()]
    return dict(status='PASS',files=files,validation=validation)
