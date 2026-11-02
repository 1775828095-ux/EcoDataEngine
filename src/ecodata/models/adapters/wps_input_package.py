"""Strict WPS4.6 v5 input-package assembly; no time/grid/level interpolation."""
import hashlib,io,json,re,struct,zipfile
from pathlib import PurePosixPath
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.wps_fields import FIELDS


def _blocks(raw,markers=()):
    pos=0;records=[]
    while pos<len(raw):
        if pos+8>len(raw):raise SchemaError('WPS记录头被截断')
        n=struct.unpack_from('>i',raw,pos)[0]
        if n<0 or pos+n+8>len(raw) or struct.unpack_from('>i',raw,pos+n+4)[0]!=n:
            raise SchemaError('WPS Fortran记录长度或尾标记不匹配')
        records.append((raw[pos+4:pos+4+n],raw[pos:pos+8+n]));pos+=8+n
    if not records or len(records)%5:raise SchemaError('每个WPS场需5条完整记录')
    for i in range(0,len(records),5):
        (v,_),(h,_),(proj,_),(wind,_),(slab,_)=records[i:i+5]
        if v!=struct.pack('>i',5) or len(h)!=156 or len(wind)!=4:raise SchemaError('只接受WPS intermediate-v5头记录')
        try:
            date=h[:24].decode('ascii').strip();field=h[60:69].decode('ascii').strip();unit=h[69:94].decode('ascii').strip()
            t=pd.Timestamp(date.replace('_','T'))
        except (ValueError,UnicodeError) as exc:raise SchemaError('WPS日期或头文本无效') from exc
        if t.tzinfo is not None or t.minute or t.second or t.microsecond or t.nanosecond:raise SchemaError('WPS需UTC无偏移整点记录，不移动时间')
        level,nx,ny,iproj=struct.unpack('>fiii',h[140:])
        if nx<2 or ny<2 or nx*ny>10_000_000 or len(slab)!=nx*ny*4:raise SchemaError('WPS网格数量与数据长度不一致')
        if iproj not in {0,3} or len(proj)!={0:28,3:40}.get(iproj) or proj[:8]!=b'SWCORNER':raise SchemaError('当前组包支持IPROJ0/3、SWCORNER；其他投影未核对')
        p=np.array(struct.unpack('>'+('5f' if iproj==0 else '8f'),proj[8:]))
        # WPS regular-ll legitimately uses either -180..180 or 0..360 east
        # longitude (e.g. FNL starts at 230E). Preserve its original bytes.
        longitude_valid=(-180<=p[1]<=360) if iproj==0 else (-180<=p[1]<=180)
        if not np.isfinite(p).all() or not (-90<=p[0]<=90) or not longitude_valid or min(p[2:4])<=0 or p[-1]<=0:
            raise SchemaError('WPS坐标、格距或地球半径无效')
        if field not in FIELDS or unit!=FIELDS[field][0]:raise SchemaError(f'WPS字段{field}单位{unit}未匹配已支持规范；不改单位标签')
        kind=FIELDS[field][1]
        if not np.isfinite(level) or (kind=='pressure' and not 0<level<=110000) or (kind!='pressure' and level!=(201300 if field=='PMSL' else 200100)):
            raise SchemaError(f'{field}层标识无效')
        flag=struct.unpack('>i',wind)[0]
        if flag!=0:raise SchemaError('当前组包只支持已有earth-relative记录，不旋转网格风')
        values=np.frombuffer(slab,dtype='>f4')
        if not np.isfinite(values).all():raise SchemaError(f'{field}包含NaN/无穷值')
        if any((values==v).any() for v in markers):raise SchemaError(f'{field}存在源缺测标记')
        bounds={'TT':(100,400),'RH':(0,100),'PSFC':(1000,120000),'PMSL':(1000,120000),'SST':(250,330),'SKINTEMP':(100,400),
                'UU':(-500,500),'VV':(-500,500),'SNOW':(0,1e6),'SNOWH':(0,1000)}
        if kind=='soil':bounds[field]=(0,1) if field.startswith('SM') else (100,400)
        if field in bounds and (values.min()<bounds[field][0] or values.max()>bounds[field][1]):raise SchemaError(f'{field}超出既有工程数值范围')
        if field=='LANDSEA' and not np.isin(values,[0,1]).all():raise SchemaError('LANDSEA需0/1明确分类')
        yield dict(time=t,field=field,level=level,units=unit,grid=(nx,ny,iproj,proj),values=values,
                   raw=b''.join(r[1] for r in records[i:i+5]))


def export_package(content,config):
    if config.get('version')!='WPS-4.6-intermediate-v5-package' or config.get('time_basis')!='UTC':
        raise SchemaError('组包需version=WPS-4.6-intermediate-v5-package及time_basis=UTC')
    wanted=config.get('required_levels');markers=config.get('missing_values')
    if not isinstance(wanted,dict) or not wanted:raise SchemaError('需显式required_levels：每个必需字段及真实WPS层值列表')
    for field,levels in wanted.items():
        if field not in FIELDS or not isinstance(levels,list) or not levels or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in levels) or len(set(levels))!=len(levels):
            raise SchemaError(f'required_levels.{field}无效')
    if ('UU' in wanted)!=('VV' in wanted) or ('UU' in wanted and set(wanted['UU'])!=set(wanted['VV'])):
        raise SchemaError('UU/VV须共同提供且层一致')
    if not isinstance(markers,list) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in markers):raise SchemaError('需明确有限数值missing_values列表')
    try:times=pd.DatetimeIndex(config['expected_times_utc'])
    except (KeyError,TypeError,ValueError) as exc:raise SchemaError('expected_times_utc需完整UTC时刻列表') from exc
    if times.empty or times.tz is None or str(times.tz)!='UTC' or times.has_duplicates or not times.is_monotonic_increasing or any(t.minute or t.second or t.microsecond or t.nanosecond for t in times):
        raise SchemaError('expected_times_utc须递增唯一UTC整点并显式带Z/+00:00')
    interval=config.get('interval_seconds')
    if not isinstance(interval,int) or isinstance(interval,bool) or interval<=0 or (len(times)>1 and not np.all(np.diff(times.values)==np.timedelta64(interval,'s'))):
        raise SchemaError('时刻列表缺口或interval_seconds不一致；不补时刻')
    expected=set(times.tz_localize(None));prefix=config.get('prefix')
    if not isinstance(prefix,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,23}',prefix):raise SchemaError('prefix需安全ASCII字母数字下划线，最多24字符')
    try:archive=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('WPS组包输入需已有WPS文件ZIP') from exc
    grouped={};grid=None;inputs={}
    with archive:
        entries=archive.infolist();names=[e.filename for e in entries]
        if len(names)!=len(set(names)) or len(entries)>1000 or sum(e.file_size for e in entries)>128*1024**2:raise SchemaError('ZIP重复或超过128MB/1000项限制')
        for e in entries:
            p=PurePosixPath(e.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in e.filename or ':' in e.filename or e.is_dir():raise SchemaError('组包ZIP需安全常规文件名；不能包含目录/控制文件')
            raw=archive.read(e);inputs[e.filename]=hashlib.sha256(raw).hexdigest()
            for rec in _blocks(raw,markers):
                if rec['time'] not in expected:raise SchemaError(f'源文件{e.filename}有配置外时刻{rec["time"]}')
                if grid is None:grid=rec['grid']
                if grid!=rec['grid']:raise SchemaError(f'源文件{e.filename}网格/投影不一致；不重网格')
                key=(rec['field'],rec['level']);at=grouped.setdefault(rec['time'],{})
                if key in at:raise SchemaError(f'{rec["time"]}重复字段/层{key}；不静默覆盖')
                at[key]=rec
    if set(grouped)!=expected:raise SchemaError('输入WPS缺时刻；不能复制日场/快照填补')
    required={(k,float(v)) for k,vs in wanted.items() for v in vs}
    fieldsets=[];files={};counts={}
    for t in sorted(expected):
        recs=grouped[t];missing=required-set(recs)
        if missing:raise SchemaError(f'{t}缺字段/层{sorted(missing)}')
        fieldsets.append(set(recs))
        if fieldsets[-1]!=fieldsets[0]:raise SchemaError('各时刻字段/层集合不一致')
        winds={c:{level for field,level in recs if field==c} for c in ('UU','VV')}
        if winds['UU']!=winds['VV']:raise SchemaError('已有UU/VV层集合不一致')
        name=f'{prefix}_{t:%Y-%m-%d_%H}.wps';files[name]=b''.join(recs[k]['raw'] for k in sorted(recs));counts[name]=len(recs)
    scope=('WPS4.6 v5已有场检查与组包，保留每场原生五记录的逐字节内容。只接受IPROJ0/3同网格、earth-relative风；'
           '不插值、不补字段/层/日期，不生成geo_em/met_em/wrfinput，不认证物理方案完整输入。'
           '在WPS主机将PREFIX_YYYY-MM-DD_HH.wps改名PREFIX:YYYY-MM-DD_HH，合入已有metgrid fg_name；'
           '仍需目标域geo_em、METGRID.TBL和项目全部必要驱动。')
    files['README_WPS_package.txt']=scope.encode('utf8')
    return files,dict(target='WPS4.6 metgrid intermediate-v5',scope=scope,input_sha256=inputs,records_per_file=counts,
                      interpolation='none',grid_shape_y_x=[grid[1],grid[0]],iproj=grid[2],native_model_read_verified=False)
