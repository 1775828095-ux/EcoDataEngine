"""SWAT+62 HRU database attributes and optional processed topography tables.

Requires upstream HRU identities and existing referenced databases; does not
derive HRUs, watershed delineation, drainage topology or parameter defaults.
"""
import csv
import hashlib
import io
from pathlib import PurePosixPath
import re
import zipfile
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform

FIELDS = ('id','name','topo','hydro','soil','lu_mgt','soil_plant_init','surf_stor','snow','field')
MANDATORY = ('topo','hydro','soil','lu_mgt','soil_plant_init')
TOPO_FIELDS = ('name','slp','slp_len','lat_len','dist_cha','depos')


def _name(value, field, length=40, nullable=False):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,'+str(length-1)+'}',value) or value == 'sim' or (value=='null' and not nullable):
        raise SchemaError(f'{field}需1–{length}字符安全ASCII名称'+('或显式null' if nullable else '，不能为空/null'))
    return value


def _table(content, fields, configuration):
    mapping=configuration.get('fields')
    if not isinstance(mapping,dict) or set(fields)-set(mapping) or len({mapping.get(k) for k in fields})!=len(fields):
        raise SchemaError('所有字段必须有独立源列映射：'+','.join(fields))
    try:
        text=content.decode(configuration.get('encoding','utf-8-sig'))
        header=next(csv.reader(io.StringIO(text)))
        if len(header)!=len(set(header)):
            raise SchemaError('源CSV有重复字段')
        frame=pd.read_csv(io.StringIO(text),dtype=str,keep_default_na=False)
    except (ValueError,UnicodeError,StopIteration) as exc:
        raise SchemaError(f'CSV读取失败：{exc}') from exc
    if frame.empty or set(mapping[k] for k in fields)-set(frame.columns):
        raise SchemaError('源CSV为空或缺列：'+','.join(sorted(set(mapping[k] for k in fields)-set(frame.columns))))
    frame=frame[[mapping[k] for k in fields]].copy();frame.columns=fields
    if frame.isna().any().any() or frame.map(lambda v:not str(v).strip()).any().any():
        raise SchemaError('存在空字段；可选引用必须显式填null，不能自动使用默认')
    return frame


def export_topography_table(content, config):
    data=_table(content,TOPO_FIELDS,config)
    units=config.get('units')
    if not isinstance(units,dict) or set(TOPO_FIELDS[1:])-set(units):
        raise SchemaError('地形参数slp/slp_len/lat_len/dist_cha/depos的单位必须明确')
    markers=config.get('missing_values')
    if not isinstance(markers,list):raise SchemaError('地形需明确missing_values列表')
    for field in TOPO_FIELDS[1:]:
        values=pd.to_numeric(data[field],errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all():raise SchemaError(f'地形{field}含缺测/NaN/无效值')
        for marker in markers:
            try:
                if (values==float(marker)).any():raise SchemaError(f'地形{field}含缺测标记{marker}')
            except (ValueError,TypeError):pass
        if field in {'slp','depos'}:
            if units[field] not in {'1','m/m','%'}:
                raise SchemaError(f'地形{field}需明确无量纲或百分数，不能猜测')
            scale=.01 if units[field]=='%' else 1.
        else:
            scale,offset=unit_transform(units[field],'m')
            if offset:raise SchemaError('地形长度换算不能有偏移')
        data[field]=values*scale
        if not np.isfinite(data[field]).all() or (data[field]<0).any():
            raise SchemaError(f'地形{field}须非负有限值')
        if field in {'slp_len','lat_len'} and (data[field]<=0).any():
            raise SchemaError(f'地形{field}须正长度；不补造坡长')
    # dep_co is a coefficient, not documented as a fraction upper bounded by1.
    names=[_name(v,'topography.name',16) for v in data.name]
    if len(names)!=len(set(names)):raise SchemaError('地形名称重复')
    lines=['topography.hyd SWAT+62 processed parameters',' '.join(TOPO_FIELDS)]
    for _,row in data.sort_values('name').iterrows():lines.append(row['name']+' '+' '.join(repr(float(row[k])) for k in TOPO_FIELDS[1:]))
    return ('\n'.join(lines)+'\n').encode('ascii'),set(names)


def _reference_names(content, role):
    try:lines=content.decode('ascii').splitlines()[2:]
    except UnicodeError as exc:raise SchemaError(f'{role}数据库必须为目标版本ASCII') from exc
    names=[];i=0
    while i<len(lines):
        tokens=lines[i].split();i+=1
        if not tokens:continue
        minimum={'topo':6,'hydro':15,'soil':7,'lu_mgt':14,'soil_plant_init':7,'snow':9,'field':4}[role]
        if len(tokens)<minimum:raise SchemaError(f'{role}数据库记录字段不足，不接受仅名称的占位表')
        name=_name(tokens[0],role,20 if role=='soil' else (16 if role in {'topo','field','snow','hydro'} else 40))
        names.append(name)
        if role=='soil':
            if len(tokens)<7:
                raise SchemaError('soils.sol土壤记录头缺字段')
            try:n=int(tokens[1])
            except ValueError as exc:raise SchemaError('soils.sol层数无效') from exc
            if n<1 or i+n>len(lines):raise SchemaError('soils.sol层记录不完整')
            for layer in lines[i:i+n]:
                if len(layer.split())!=14:raise SchemaError('soils.sol每层需14项')
            i+=n
    if not names or len(names)!=len(set(names)):
        raise SchemaError(f'{role}数据库无记录或名称重复')
    return set(names)


def export_hru_bundle(content, config):
    if config.get('target_model_version')!='62.0.0':raise SchemaError('HRU属性仅核对SWAT+62.0.0')
    source=config.get('source')
    if not isinstance(source,dict) or not source.get('kind') or not source.get('description'):
        raise SchemaError('需明确source.kind/description；HRU属性必须来自工程划分或真实配置')
    try:
        archive=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('HRU输入为ZIP：源HRU.csv及已生成引用数据库') from exc
    with archive:
        entries=archive.infolist();names=[e.filename for e in entries]
        if len(names)!=len(set(names)) or sum(e.file_size for e in entries)>64*1024**2 or len(entries)>500:
            raise SchemaError('ZIP有重复成员或超过64MB/500成员限额')
        for name in names:
            path=PurePosixPath(name)
            if '\\' in name or path.is_absolute() or '..' in path.parts or ':' in name:
                raise SchemaError('ZIP成员路径不安全')
        table=config.get('table')
        if table not in names or archive.getinfo(table).is_dir():raise SchemaError('table未对应ZIP内真实HRU源CSV')
        frame=_table(archive.read(table),FIELDS,config)
        ids=pd.to_numeric(frame.id,errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(ids).all() or not np.array_equal(np.sort(ids),np.arange(1,len(frame)+1)):
            raise SchemaError('HRU id必须唯一连续整数1..N，不能虚构或重编号已有工程')
        frame['id']=ids.astype(int)
        for field in FIELDS[1:]:
            frame[field]=[_name(v,field,40,field in {'surf_stor','snow','field'}) for v in frame[field]]
        if frame.name.duplicated().any():raise SchemaError('HRU name重复')
        if (frame.surf_stor!='null').any():
            raise SchemaError('本属性转换尚未核对非null地表蓄水引用；请勿虚构湿地/蓄水对象')
        references=config.get('references')
        if not isinstance(references,dict):raise SchemaError('references必须指向ZIP内真实数据库，而非仅列名称')
        generated={};catalog={};checks={}
        topo=config.get('topography')
        if topo is not None:
            if not isinstance(topo,dict) or topo.get('table') not in names:raise SchemaError('topography.table未对应真实已处理地形参数CSV')
            generated['topography.hyd'],catalog['topo']=export_topography_table(archive.read(topo['table']),topo)
            checks['topo']={'source':topo['table'],'status':'generated from supplied processed parameters','names':sorted(catalog['topo'])}
        for role in ('topo','hydro','soil','lu_mgt','soil_plant_init','snow','field'):
            needed=set(frame[role])-{'null'}
            if role not in catalog and needed:
                filename=references.get(role)
                if filename not in names or archive.getinfo(filename).is_dir():
                    raise SchemaError(f'{role}缺ZIP内引用数据库文件；需要名称{sorted(needed)}')
                data=archive.read(filename);catalog[role]=_reference_names(data,role)
                checks[role]={'source':filename,'sha256':hashlib.sha256(data).hexdigest(),'status':'reference names checked only; parameter values not certified'}
            if needed-set(catalog.get(role,set())):
                raise SchemaError(f'{role}引用名称不存在：{sorted(needed-set(catalog.get(role,set())))}')
        lines=['hru-data.hru EcoDataEngine SWAT+62',' '.join(FIELDS)]
        for _,row in frame.sort_values('id').iterrows():lines.append(' '.join(str(row[k]) for k in FIELDS))
        generated['hru-data.hru']=('\n'.join(lines)+'\n').encode('ascii')
    scope=('仅HRU属性数据库hru-data.hru；可选从完整上游参数表写topography.hyd。'
           '不生成流域划分、HRU几何/面积/连接hru.con、河道/含水层拓扑、管理或初始参数，不提供file.cio。'
           '数据库引用名称已在用户所给文件中核对；其他数据库的数值、跨级依赖和完整工程有效性未认证。'
           '本格式无时间轴和坐标字段；HRU位置/面积及网络由上游工程提供。不允许非null surf_stor。')
    generated['README_HRU_attributes.txt']=scope.encode('utf8')
    return generated,{'target':'SWAT+62.0.0 hru_read/hru_databases_char','hru_records':len(frame),'reference_checks':checks,
                      'source':source,'generated_files':list(generated),'scope':scope,'topology_generation':'none','missing_policy':'reject'}
