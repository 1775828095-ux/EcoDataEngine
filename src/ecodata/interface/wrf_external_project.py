"""Read-only external WRF project audit. Never executes uploaded code.

Native initial/boundary files are not the WPS companion fields needed to
replace a pressure conversion. This inspector deliberately reports the two
contracts separately; neither structural inspection proves native acceptance.
"""
from __future__ import annotations
import hashlib
import io
import re
import stat
import zipfile
from pathlib import PurePosixPath
from datetime import datetime
from netCDF4 import Dataset

MAX_BYTES=256*1024**2


def read_archive(payload):
    if not isinstance(payload,bytes) or len(payload)>MAX_BYTES:
        raise ValueError('工程ZIP超过256MiB或不是字节数据')
    files={}
    seen=set()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if len(z.infolist())>2000 or sum(i.file_size for i in z.infolist())>MAX_BYTES:
                raise ValueError('工程展开大小或文件数量超过检查限额')
            for info in z.infolist():
                name=info.filename
                p=PurePosixPath(name)
                if (info.orig_filename!=name or '\\' in name or p.is_absolute() or '..' in p.parts
                        or any(ord(c)<32 for c in name) or re.match(r'^[A-Za-z]:',name)
                        or stat.S_ISLNK(info.external_attr>>16)):
                    raise ValueError('工程ZIP包含不安全路径或链接：'+name)
                if info.is_dir():continue
                if name.casefold() in seen:raise ValueError('工程ZIP有重复文件名：'+name)
                seen.add(name.casefold())
                files[name]=z.read(info)
    except (zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:
        raise ValueError('不能读取工程ZIP') from exc
    if not files:raise ValueError('工程ZIP没有文件')
    return files


def _groups(raw):
    """Only inspect scalar keys needed here, never rewrite a Fortran namelist."""
    text=raw.decode('utf-8-sig')
    # Quoted strings may contain !. Preserve them while removing real comments.
    text=re.sub(r"'[^']*'|\"[^\"]*\"|![^\n]*",lambda m:'' if m[0].startswith('!') else m[0],text)
    result={}
    for match in re.finditer(r'(?is)&([a-z][a-z0-9_]*)\b(.*?)(?:^\s*/\s*$)',text,re.M):
        name,body=match[1].lower(),match[2]
        if name in result:raise ValueError('重复namelist组：'+name)
        keys=list(re.finditer(r'(?i)\b([a-z][a-z0-9_]*)\s*=',body))
        group={}
        for i,key in enumerate(keys):
            label=key[1].lower()
            if label in group:raise ValueError('重复namelist参数：'+name+'.'+label)
            group[label]=body[key.end():keys[i+1].start() if i+1<len(keys) else len(body)].strip().rstrip(',').strip()
        result[name]=group
    return result


def _integer(group,key):
    value=group.get(key,'').split(',')[0].strip()
    if not re.fullmatch(r'[+-]?\d+',value):raise ValueError('缺少或不能识别整数参数：'+key)
    return int(value)


def _stamp(group,prefix):
    return datetime(*[_integer(group,prefix+'_'+key) for key in ('year','month','day','hour','minute','second')])


def _string(group,key):
    value=group.get(key,'').split(',')[0].strip()
    if len(value)<2 or value[0] not in "'\"" or value[-1]!=value[0]:
        raise ValueError('缺少或不能识别字符串参数：'+key)
    return value[1:-1]


def _netcdf(raw):
    with Dataset('uploaded_project.nc',memory=raw) as ds:
        stamps=[]
        if 'Times' in ds.variables:
            for row in ds['Times'][:]:
                stamps.append(b''.join(row).decode('ascii').strip('\x00 '))
        boundary={}
        for key in ('thisbdytime','nextbdytime'):
            names=[n for n in ds.variables if '___'+key in n]
            if len(names)==1:
                boundary[key]=[b''.join(row).decode('ascii').strip('\x00 ') for row in ds[names[0]][:]]
        return dict(dimensions={n:len(d) for n,d in ds.dimensions.items()},times=stamps,
                    title=str(getattr(ds,'TITLE','')),map_projection=int(ds.MAP_PROJ) if hasattr(ds,'MAP_PROJ') else None,
                    dx=float(getattr(ds,'DX',0)),dy=float(getattr(ds,'DY',0)),
                    variables=list(ds.variables),boundary_intervals=boundary)


def inspect_project(payload,converted_zip=None,namelist_path=None):
    files=read_archive(payload)
    return inspect_project_files(files,converted_zip,namelist_path,hashlib.sha256(payload).hexdigest())


def inspect_project_files(files,converted_zip=None,namelist_path=None,archive_sha256=None):
    """Same native validator for an already bound, input-only resource mapping."""
    candidates=[n for n in files if PurePosixPath(n).name=='namelist.input']
    if namelist_path is not None:
        if namelist_path not in candidates:raise ValueError('指定的namelist.input不在工程内')
        candidates=[namelist_path]
    if len(candidates)!=1:
        raise ValueError('需要恰好一个namelist.input；多个运行目录请分别打包，不能猜选real或wrf目录')
    main=candidates[0]
    parent=PurePosixPath(main).parent
    def sibling(name):return str(parent/name)
    report=dict(schema='wrf-external-inspection-v1',native_model_executed=False,
                uploaded_project_sha256=archive_sha256,namelist=main,
                file_count=len(files),errors=[],warnings=[],native_inputs={},wps_companions={},
                replacement_checked=False,scientific_settings_changed=False,
                scope='Read-only structure and pressure assembly compatibility; no engine/physics-table compatibility or native run validation')
    errors=report['errors'];warnings=report['warnings']
    executable=[n for n in files if PurePosixPath(n).suffix.lower() in {'.exe','.sh','.py','.bat','.cmd','.ps1'}]
    report['ignored_executable_files']=executable
    if executable:warnings.append('上传脚本及程序仅列名，不导入、不执行；后续只能采用平台受控程序')
    group=_groups(files[main]);tc=group.get('time_control',{});domains=group.get('domains',{})
    try:
        count=_integer(domains,'max_dom')
        report['max_dom']=count
        if count!=1:errors.append('当前外部工程识别仅核对单网格；嵌套域不能按已支持处理')
        start,end=_stamp(tc,'start'),_stamp(tc,'end')
        if end<=start:raise ValueError('结束日期不晚于开始日期')
        report.update(start=start.isoformat(),end=end.isoformat())
    except ValueError as exc:
        errors.append(str(exc));start=end=None
    for name in ('wrfinput_d01','wrfbdy_d01'):
        key=sibling(name)
        if key not in files:
            errors.append('缺少与namelist.input同目录的'+name);continue
        try:
            meta=_netcdf(files[key]);meta['sha256']=hashlib.sha256(files[key]).hexdigest()
            report['native_inputs'][name]=meta
            if not meta['times']:errors.append(name+'缺少Times')
            elif start and meta['times'][0]!=start.strftime('%Y-%m-%d_%H:%M:%S'):
                errors.append(name+'起始时间与namelist不一致')
            stamps=[datetime.strptime(t,'%Y-%m-%d_%H:%M:%S') for t in meta['times']]
            if stamps!=sorted(set(stamps)):errors.append(name+'时间重复或倒序')
            if name=='wrfinput_d01' and len(stamps)!=1:errors.append('当前检查仅支持单时刻初始场')
            if name=='wrfbdy_d01':
                intervals=meta['boundary_intervals']
                this=intervals.get('thisbdytime',[]);nxt=intervals.get('nextbdytime',[])
                if not this or this!=meta['times'] or len(nxt)!=len(this):
                    errors.append('不能核验边界时段：缺少一致的thisbdytime/nextbdytime；不从Times行数猜测覆盖')
                else:
                    starts=[datetime.strptime(t,'%Y-%m-%d_%H:%M:%S') for t in this]
                    ends=[datetime.strptime(t,'%Y-%m-%d_%H:%M:%S') for t in nxt]
                    if any(b<=a for a,b in zip(starts,ends)) or ends[:-1]!=starts[1:]:
                        errors.append('边界时段存在缺口、重叠或无效间隔')
                    if end and ends[-1]<end:errors.append('边界时段不能覆盖namelist结束日期')
        except (ValueError,OSError,RuntimeError) as exc:errors.append(name+'读取失败：'+str(exc))
    if 'wrfinput_d01' in report['native_inputs']:
        dims=report['native_inputs']['wrfinput_d01']['dimensions']
        try:
            for key,dim in [('e_we','west_east'),('e_sn','south_north'),('e_vert','bottom_top')]:
                if dims.get(dim)!=_integer(domains,key)-1:errors.append('wrfinput维度与namelist不一致：'+key)
        except ValueError as exc:errors.append(str(exc))
    report['native_structure_candidate']=not errors and len(report['native_inputs'])==2
    # Locate WPS independently; never assume WRF initial files can be reverse-
    # converted into full ungrib records, nor silently supply a registered case.
    wpslists=[n for n in files if PurePosixPath(n).name=='namelist.wps']
    geos=[n for n in files if PurePosixPath(n).name=='geo_em.d01.nc']
    report['wps_companions'].update(namelists=wpslists,geogrid=geos)
    fields=[]
    for name,raw in files.items():
        if raw[:12]==b'\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x04':fields.append(name)
    report['wps_companions']['intermediate_files']=fields
    gaps=[]
    if len(wpslists)!=1:gaps.append('一个明确的namelist.wps')
    if len(geos)!=1:gaps.append('一个明确的geo_em.d01.nc')
    if not fields:gaps.append('与新转换时段对应的完整WPS中间场（湿度、地表、土壤等）')
    report['replacement_missing']=gaps
    if gaps:warnings.append('原生初始/边界文件不等于天气替换所需预处理工程：缺少'+'；'.join(gaps))
    if len(wpslists)==1 and len(geos)==1:
        try:
            wps=_groups(files[wpslists[0]])
            share=wps.get('share',{});geogrid=wps.get('geogrid',{})
            if _integer(share,'max_dom')!=1:raise ValueError('WPS不是单网格')
            ws=datetime.strptime(_string(share,'start_date'),'%Y-%m-%d_%H:%M:%S')
            we=datetime.strptime(_string(share,'end_date'),'%Y-%m-%d_%H:%M:%S')
            interval=_integer(share,'interval_seconds')
            if interval<=0 or we<=ws or (we-ws).total_seconds()%interval:
                raise ValueError('WPS时段与时间间隔不一致')
            if start and (ws!=start or we<end):raise ValueError('WPS时段未覆盖WRF时段或起始时间不同')
            geo=_netcdf(files[geos[0]])
            report['wps_companions']['geogrid_metadata']=geo
            for key,dim in [('e_we','west_east'),('e_sn','south_north')]:
                if geo['dimensions'].get(dim)!=_integer(geogrid,key)-1:
                    raise ValueError('geo_em维度与namelist.wps不一致：'+key)
                if 'wrfinput_d01' in report['native_inputs'] and geo['dimensions'].get(dim)!=report['native_inputs']['wrfinput_d01']['dimensions'].get(dim):
                    raise ValueError('geo_em与wrfinput网格不同：'+dim)
            if 'wrfinput_d01' in report['native_inputs']:
                native=report['native_inputs']['wrfinput_d01']
                if any(geo[k]!=native[k] for k in ('dx','dy','map_projection')):
                    raise ValueError('geo_em与wrfinput投影或分辨率不同')
            if converted_zip is not None:
                from datetime import timedelta
                from ecodata.interface.wrf_assembly import read_fields
                expected=[];stamp=ws
                while stamp<=we:
                    if len(expected)>=100:raise ValueError('WPS时次数超出本次只读检查限额')
                    expected.append(stamp.strftime('%Y-%m-%d_%H:%M:%S'))
                    stamp+=timedelta(seconds=interval)
                actual=[read_fields(files[n])[0].time for n in fields]
                if sorted(actual)!=expected:raise ValueError('完整WPS中间场时刻没有覆盖namelist.wps')
        except (ValueError,OSError,RuntimeError) as exc:errors.append('WPS配套检查：'+str(exc))
    if converted_zip is not None:
        from ecodata.interface.wrf_runner import read_update_zip
        from ecodata.interface.wrf_assembly import read_fields,assemble_pressure_update
        try:
            incoming=read_update_zip(converted_zip)
            original={}
            for name in fields:
                records=read_fields(files[name]);stamp=records[0].time
                if stamp in original:raise ValueError('同一时刻存在多个WPS中间文件，需明确前缀选择')
                original[stamp]=name
            checked=[];times=[]
            for name,raw in incoming.items():
                records=read_fields(raw);stamp=records[0].time
                if stamp in times:raise ValueError('转换包有重复时刻')
                times.append(stamp)
                if stamp not in original:raise ValueError('工程没有该时刻的完整WPS配套资料：'+stamp)
                _,check=assemble_pressure_update(files[original[stamp]],raw)
                checked.append(dict(source=original[stamp],incoming=name,assembly=check))
            report['pressure_assembly_checks']=checked
            if len(times)!=len(fields):raise ValueError('新转换未逐一覆盖全部原WPS时刻')
            report['replacement_checked']=not gaps and not errors
        except ValueError as exc:errors.append('替换兼容性：'+str(exc))
    report['file_manifest']={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()}
    warnings.append('未核验物理方案所需表、化学/同化/移动网格输入、全部WPS覆盖与程序版本；不是可运行性保证')
    from ecodata.interface.engineering_completeness import wrf
    report['completeness']=wrf(report,converted_zip is not None)
    return report


def validate_run_contract(payload,converted_zip,wrf_namelist,real_namelist,reference_wrf,reference_real):
    """Bounded existing-project execution: validate, never rewrite namelists.

Only the installed, reviewed single-domain CONUS science contract is accepted.
Different scientific settings are rejected rather than changed to the template.
"""
    report=inspect_project(payload,converted_zip,wrf_namelist)
    if not report['replacement_checked']:raise ValueError('外部工程替换检查未通过：'+'；'.join(report['errors']+report['replacement_missing']))
    files=read_archive(payload)
    if real_namelist not in files or real_namelist==wrf_namelist:
        raise ValueError('需明确选择单独的REAL配置；不从WRF配置推造或修改')
    allowed={'time_control':{'run_days','run_hours','run_minutes','run_seconds','interval_seconds','history_interval','frames_per_outfile'}|
             {prefix+'_'+key for prefix in ('start','end') for key in ('year','month','day','hour','minute','second')},
             'domains':{'e_we','e_sn','e_vert','dx','dy','time_step','p_top_requested','num_metgrid_levels','num_metgrid_soil_levels'}}
    parsed={}
    for kind,path,reference in [('wrf',wrf_namelist,reference_wrf),('real',real_namelist,reference_real)]:
        groups=_groups(files[path]);trusted=_groups(reference)
        if set(groups)!=set(trusted):raise ValueError(kind+'包含未支持或缺少的namelist组；不覆盖用户设置')
        for group,values in groups.items():
            if set(values)!=set(trusted[group]):raise ValueError(kind+'.'+group+'包含未支持或缺少的参数；不覆盖用户设置')
            for key,value in values.items():
                if key not in allowed.get(group,set()) and re.sub(r'\s+','',value).lower()!=re.sub(r'\s+','',trusted[group][key]).lower():
                    raise ValueError('当前受控运行不支持此科学或IO设置：'+kind+'.'+group+'.'+key+'；原值保留，不替换')
        tc=groups['time_control'];domain=groups['domains']
        begin,finish=_stamp(tc,'start'),_stamp(tc,'end')
        duration=int((finish-begin).total_seconds())
        declared=sum(_integer(tc,key)*factor for key,factor in [('run_days',86400),('run_hours',3600),('run_minutes',60),('run_seconds',1)])
        if not 0<duration<=86400 or declared!=duration:raise ValueError(kind+'运行长度与起止日期不一致或超过一天')
        if _integer(tc,'history_interval')!=5:raise ValueError('当前图表验收要求用户原配置history_interval=5；平台不改写')
        if any(not 25<=_integer(domain,k)<=101 for k in ('e_we','e_sn')) or not 30<=_integer(domain,'e_vert')<=65:
            raise ValueError('网格超过当前受控验收资源范围')
        if not 6000<=_integer(domain,'dx')<=50000 or _integer(domain,'dy')!=_integer(domain,'dx'):
            raise ValueError('当前受控验收仅支持6至50km方形网格')
        if not 1<=_integer(domain,'time_step')<=min(180,_integer(domain,'dx')/1000*6):raise ValueError('时间步超出当前验收范围')
        parsed[kind]=groups
    wrf=parsed['wrf'];real=parsed['real']
    for group in ('domains','physics','fdda','dynamics','bdy_control'):
        if wrf[group]!=real[group]:raise ValueError('REAL与WRF配置在'+group+'不一致；平台不猜测取舍')
    tc=wrf['time_control'];duration=int((_stamp(tc,'end')-_stamp(tc,'start')).total_seconds())
    if not 300<=duration<=21600 or duration%300:raise ValueError('当前外部工程积分限5分钟至6小时且整5分钟')
    wps=_groups(files[report['wps_companions']['namelists'][0]])
    share=wps['share'];met=wps.get('metgrid',{})
    if _stamp(real['time_control'],'start')!=datetime.strptime(_string(share,'start_date'),'%Y-%m-%d_%H:%M:%S') or _stamp(real['time_control'],'end')!=datetime.strptime(_string(share,'end_date'),'%Y-%m-%d_%H:%M:%S'):
        raise ValueError('REAL配置必须与完整WPS起止日期一致；平台不改写')
    prefix=_string(met,'fg_name')
    if not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,31}',prefix) or _string(met,'opt_metgrid_tbl_path') not in ('.','./'):
        raise ValueError('当前仅支持单一相对WPS前缀与本地受控METGRID.TBL；不改写外部路径')
    for meta in report['native_inputs'].values():
        if 'V4.8.0' not in meta['title']:raise ValueError('当前受控程序为WRF4.8.0；不自动迁移其他版本工程')
    report['execution_contract']=dict(version='WRF4.8.0',science='Reviewed single-domain CONUS settings; differing scientific options are rejected, not overwritten',
        wrf_namelist=wrf_namelist,real_namelist=real_namelist,namelist_bytes_preserved=True,
        runtime_tables='Installed controlled engine tables, not uploaded executables or scripts')
    cfg=dict(start=_stamp(tc,'start').strftime('%Y-%m-%d_%H:%M:%S'),duration_seconds=duration,
             external=True,metgrid_prefix=prefix,num_metgrid_levels=_integer(wrf['domains'],'num_metgrid_levels'),
             num_metgrid_soil_levels=_integer(wrf['domains'],'num_metgrid_soil_levels'))
    return files,report,cfg
