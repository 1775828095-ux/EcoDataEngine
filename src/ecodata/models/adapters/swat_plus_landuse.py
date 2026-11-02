"""SWAT+62 vegetated nonurban landuse database references, never HRU derivation."""
import io,zipfile,hashlib
from pathlib import PurePosixPath
import numpy as np
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.swat_plus_hru import _table,_name

FIELDS=('name','cal_group','plnt_com','mgt','cn2','cons_prac','urban','urb_ro','ov_mann','tile','sep','vfs','grww','bmp')
ROLES={'plnt_com':'plant.ini','mgt':'management.sch','cn2':'cntable.lum','cons_prac':'cons_practice.lum','ov_mann':'ovn_table.lum'}

def _names(data,role):
    try:lines=[s.split() for s in data.decode('ascii').splitlines()[2:] if s.strip()]
    except UnicodeError as exc:raise SchemaError(f'{role}数据库须为SWAT+ ASCII') from exc
    if not lines:raise SchemaError(f'{role}数据库无真实记录')
    names=set();i=0
    while i<len(lines):
        row=lines[i];name=_name(row[0],role,40);i+=1
        if name in names:raise SchemaError(f'{role}数据库名称重复{name}')
        names.add(name)
        if role in {'plnt_com','mgt'}:
            if len(row)!=3:raise SchemaError(f'{role}记录需名称和两个计数/初始化参数，不接受仅名称占位')
            try:a,b=[int(x) for x in row[1:]]
            except ValueError as exc:raise SchemaError(f'{role}计数必须为整数') from exc
            if a<0 or b<0 or a+b>100000:raise SchemaError(f'{role}计数非法')
            count=a if role=='plnt_com' else a+b
            if i+count>len(lines):raise SchemaError(f'{role}数据库子记录数不足')
            if role=='plnt_com' and any(len(r)<8 for r in lines[i:i+count]):raise SchemaError('plant.ini植物子记录字段不足')
            if role=='mgt' and any(len(r)<7 for r in lines[i+b:i+count]):raise SchemaError('management.sch管理操作字段不足')
            i+=count
        else:
            n={'cn2':4,'cons_prac':2,'ov_mann':3}[role]
            if len(row)<n+1:raise SchemaError(f'{role}数据库数值字段不足')
            try:values=np.array([float(v) for v in row[1:n+1]])
            except ValueError as exc:raise SchemaError(f'{role}数据库数值无效') from exc
            if not np.isfinite(values).all() or (values<0).any():raise SchemaError(f'{role}含无效/负参数')
            if role=='cn2' and (values>100).any():raise SchemaError('cntable.lum CN值应在0–100')
            if role=='cons_prac' and values[0]>1:raise SchemaError('cons_practice.lum USLE P应在0–1')
            if role=='ov_mann' and not values[1]<=values[0]<=values[2]:raise SchemaError('ovn_table.lum需min<=mean<=max')
    return names

def export_landuse(content,config):
    if config.get('mode')!='landuse_database' or config.get('target_model_version')!='62.0.0' or config.get('profile')!='vegetated_nonurban':
        raise SchemaError('土地利用数据库需mode=landuse_database、target_model_version=62.0.0、profile=vegetated_nonurban')
    try:archive=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('土地利用输入须为ZIP内源CSV及真实引用数据库') from exc
    with archive:
        info=archive.infolist();names=[v.filename for v in info]
        if len(info)>500 or sum(v.file_size for v in info)>64*1024*1024 or len(set(names))!=len(names):raise SchemaError('ZIP重复名称或超过64MB/500文件限额')
        for n in names:
            p=PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n or ':' in n:raise SchemaError('ZIP存在不安全路径')
        table=config.get('table')
        if table not in names:raise SchemaError('table未指向ZIP内真实土地利用CSV')
        frame=_table(archive.read(table),FIELDS,config)
        for field in FIELDS:frame[field]=[_name(v,field,40,field!='name') for v in frame[field]]
        if frame.name.duplicated().any():raise SchemaError('土地利用名称重复')
        for field in ('urban','urb_ro','tile','sep','vfs','grww','bmp'):
            if (frame[field]!='null').any():raise SchemaError(f'{field}非null尚不支持；本轮仅植被非城市配置，不猜测城市/工程结构')
        for field in ('plnt_com','cn2','ov_mann'):
            if (frame[field]=='null').any():raise SchemaError(f'植被非城市配置需明确{field}，不能生成空配置')
        refs=config.get('references')
        if not isinstance(refs,dict):raise SchemaError('references需指向ZIP内已有参数数据库')
        checks={}
        for role in ROLES:
            needed=set(frame[role])-{'null'}
            if not needed:continue
            filename=refs.get(role)
            if filename not in names:raise SchemaError(f'{role}缺少实际数据库文件，需要{sorted(needed)}')
            data=archive.read(filename);available=_names(data,role)
            missing=needed-available
            if missing:raise SchemaError(f'{role}引用不存在：{sorted(missing)}')
            checks[role]={'file':filename,'sha256':hashlib.sha256(data).hexdigest(),'names':sorted(needed)}
        lines=['landuse.lum EcoDataEngine SWAT+62 vegetated nonurban',' '.join(FIELDS)]
        lines+=[' '.join(row) for row in frame.values.tolist()]
        scope=('仅SWAT+62.0.0植被非城市landuse.lum配置数据库；14项为名称/引用，不是遥感分类栅格转换。'
               '植物群落、管理表、CN/保育/糙率真实数据库由用户提供并保留；本模块核对直接名称和基本结构，'
               '不生成这些数据库，不认证全部内部依赖或项目参数。城市及tile/sep/vfs/grww/bmp非null暂不支持。'
               '不推导HRU、地类空间分配、作物或管理措施，不生成file.cio。本格式没有物理单位、时间轴或空间网格字段。')
        return {'landuse.lum':('\n'.join(lines)+'\n').encode('ascii'),'README_landuse_database.txt':scope.encode('utf8')},{
            'target':'SWAT+62.0.0 landuse_read / land_use_management','profile':'vegetated_nonurban','records':len(frame),
            'fields':list(FIELDS),'units':'names/references; no physical quantities in this file','reference_checks':checks,
            'scope':scope,'source':config.get('source'),'time_processing':'not applicable','spatial_processing':'none; no grid in this database',
            'complete_model_input':False,'native_program_read_verified':False}
