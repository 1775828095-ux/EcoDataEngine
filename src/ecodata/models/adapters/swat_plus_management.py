"""SWAT+62 fixed calendar fertilizer/manure schedules, explicit material refs.

The operation quantities come from users, never from a land-use class. This
profile excludes automatic decision rules, heat-unit triggers and rotations.
"""
import calendar,io,json,zipfile
from pathlib import PurePosixPath
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.swat_plus_hru import _table,_name
FIELDS=('schedule','order','operation','month','day','hu_sch','material','application','amount','amount_unit')


def _database(data,role,minimum):
    try:rows=[r.split() for r in data.decode('ascii').splitlines()[2:] if r.strip()]
    except UnicodeError as exc:raise SchemaError(f'{role}需目标ASCII数据库') from exc
    if not rows or any(len(r)<minimum for r in rows):raise SchemaError(f'{role}数据库缺完整字段，不能只给名称')
    widths={'fertilizer':16,'manure':25,'manure_om':64,'chem_app':40,'landuse':40}
    names=[_name(r[0],role,widths[role]) for r in rows]
    if len(names)!=len(set(names)):raise SchemaError(f'{role}名称重复')
    ranges={'fertilizer':(1,6),'manure_om':(1,8),'chem_app':(3,9)}
    if role in ranges:
        a,b=ranges[role]
        for r in rows:
            try:values=np.array([float(v.replace('D','E').replace('d','e')) for v in r[a:b]])
            except ValueError as exc:raise SchemaError(f'{role}数据库数值字段无效') from exc
            if not np.isfinite(values).all() or (values<0).any():raise SchemaError(f'{role}数据库含缺测/负参数')
            if role in {'fertilizer','manure_om'} and (values>1).any():raise SchemaError(f'{role}组成比例超出0..1')
    return {r[0]:r for r in rows}


def export_management(content,cfg):
    if cfg.get('target_model_version')!='62.0.0' or cfg.get('profile')!='calendar_fert_manure' or cfg.get('automatic_operations')!=[]:
        raise SchemaError('需SWAT+62.0.0、profile=calendar_fert_manure、automatic_operations=[]；自动/轮作另需开发')
    if cfg.get('units',{}).get('hu_sch')!='1':raise SchemaError('hu_sch单位必须明确1；本profile仅显式0的日历触发')
    source=cfg.get('source')
    if not isinstance(source,dict) or not all(source.get(k) for k in ('kind','description','license')):raise SchemaError('需source.kind/description/license')
    operations=cfg.get('operation_mapping')
    if not isinstance(operations,dict) or not operations or set(operations.values())-{'fert','manu'}:raise SchemaError('operation_mapping仅支持明确fert/manu；不猜测操作')
    try:z=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('输入需源日程CSV及材料/施用数据库ZIP') from exc
    with z:
        entries=z.infolist();names=[e.filename for e in entries]
        if len(names)!=len(set(names)) or len(entries)>500 or sum(e.file_size for e in entries)>64*1024**2:raise SchemaError('ZIP重复或超过限制')
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise SchemaError('ZIP路径不安全')
        if cfg.get('table') not in names:raise SchemaError('table缺源CSV')
        frame=_table(z.read(cfg['table']),FIELDS,cfg)
        for k in ('schedule','material','application'):frame[k]=[_name(v,k,40) for v in frame[k]]
        frame['operation']=[operations.get(v) for v in frame.operation]
        if frame.operation.isna().any():raise SchemaError('源操作缺operation_mapping')
        for k in ('order','month','day','hu_sch','amount'):
            values=pd.to_numeric(frame[k],errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(values).all():raise SchemaError(f'{k}含缺测/非数值')
            markers=cfg.get('missing_values')
            if not isinstance(markers,list):raise SchemaError('需明确missing_values列表')
            for m in markers:
                try:missing=float(m)
                except (TypeError,ValueError):continue
                if (values==missing).any():raise SchemaError(f'{k}含缺测标记')
            frame[k]=values
        if (frame.hu_sch!=0).any():raise SchemaError('hu_sch非0为热量触发，本profile不转换')
        if (frame.amount<0).any():raise SchemaError('施用量不能负数，不截零')
        for k in ('order','month','day'):
            if ((frame[k]%1)!=0).any():raise SchemaError(f'{k}需整数')
            frame[k]=frame[k].astype(int)
        for _,row in frame.iterrows():
            if not 1<=row.month<=12 or not 1<=row.day<=calendar.monthrange(2001,row.month)[1]:
                raise SchemaError('月日无效；固定日历profile不支持需特定年份的2月29日')
        for _,group in frame.groupby('schedule'):
            ordered=group.sort_values('order')
            if not np.array_equal(ordered.order,np.arange(1,len(group)+1)):raise SchemaError('每日程order须连续唯一1..N')
            dates=ordered.month*100+ordered.day
            if not dates.is_monotonic_increasing:raise SchemaError('日程月日与操作顺序不一致；跨年轮作另需开发')
        amounts=[]
        for _,r in frame.iterrows():
            scale,offset=unit_transform(r.amount_unit,'kg/ha')
            if offset or scale<=0:raise SchemaError('amount_unit需质量/面积正比例单位')
            amount=r.amount*scale
            if not np.isfinite(amount):raise SchemaError('施用量单位换算溢出')
            amounts.append(amount)
        frame['amount']=amounts
        refs=cfg.get('references')
        if not isinstance(refs,dict):raise SchemaError('references需真实材料和施用数据库')
        required={'chem_app':('chem_app.ops',9)}
        if (frame.operation=='fert').any():required['fertilizer']=('fertilizer.frt',6)
        if (frame.operation=='manu').any():required.update(manure=('manure_db.frt',8),manure_om=('manure_om.frt',9))
        catalogs={};files={}
        for role,(filename,minimum) in required.items():
            if refs.get(role) not in names:raise SchemaError(f'{role}引用文件缺失')
            data=z.read(refs[role]);catalogs[role]=_database(data,role,minimum);files[filename]=data
        for _,r in frame.iterrows():
            role='fertilizer' if r.operation=='fert' else 'manure'
            if r.material not in catalogs[role]:raise SchemaError(f'{role}材料引用{r.material}不存在')
            if r.application not in catalogs['chem_app']:raise SchemaError(f'施用方式引用{r.application}不存在')
            if role=='manure':
                record=catalogs[role][r.material]
                if record[1] not in catalogs['manure_om']:raise SchemaError('粪肥org_min引用缺manure_om属性')
                if any(v!='null' for v in record[2:7]):raise SchemaError('粪肥的pests/paths/hmets/salts/constit非null：当前尚未核查这些深层引用')
        lines=['management.sch EcoDataEngine SWAT+62 explicit calendar','name numb_ops numb_auto op_typ mon day hu_sch op_data1 op_data2 op_data3']
        for name,g in frame.groupby('schedule',sort=True):
            lines.append(f'{name} {len(g)} 0')
            for _,r in g.sort_values('order').iterrows():lines.append(f'{r.operation} {r.month} {r.day} 0 {r.material} {r.application} {float(r.amount)!r}')
        files['management.sch']=('\n'.join(lines)+'\n').encode('ascii')
        # Optional real landuse table: verify its direct schedule names, never
        # inject a management assignment or overwrite other parameter names.
        if cfg.get('landuse_file') is not None:
            if cfg['landuse_file'] not in names:raise SchemaError('landuse_file引用文件缺失')
            raw=z.read(cfg['landuse_file']);rows=_database(raw,'landuse',14)
            for row in rows.values():
                if row[3]!='null' and row[3] not in set(frame.schedule):raise SchemaError(f'landuse管理引用{row[3]}不存在')
            files['landuse.lum']=raw
    scope='SWAT+62 management.sch固定月日fert/manu日程及已提供材料/施用数据库；无自动规则、热量触发、轮作、灌溉或其他管理操作，不生成file.cio。只核对已使用材料、施用方式及粪肥有机属性名称，不认证整个数据库物理参数或完整工程。'
    files['README_management.txt']=scope.encode('utf8')
    return files,dict(scope=scope,schedules=sorted(set(frame.schedule)),operations=len(frame),amount_unit='kg/ha',time_processing='fixed month/day, original operation order; no interpolation',source=source,complete_model_input=False)
