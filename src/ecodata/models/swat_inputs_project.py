"""Attach converted SWAT+ database tables to an independent complete project.

No renaming inference, object creation or automatic CIO changes. Keep record
identities/order to protect cross-file numeric references; native run is required.
"""
import io,zipfile,hashlib
from decimal import Decimal,InvalidOperation
from ecodata.core.exceptions import SchemaError

TABLES={
 'wgn':{'weather-wgn.cli'},'soil':{'soils.sol'},
 'hru_attributes':{'hru-data.hru','topography.hyd'},
 'spatial_landuse':{'hru-data.hru','topography.hyd'},
 'landuse_database':{'landuse.lum'},'aquifer_parameters':{'aquifer.aqu'},
 'connections':{'hru.con','aquifer.con','rout_unit.con','chandeg.con'},
 'management_schedule':{'management.sch','chem_app.ops','fertilizer.frt','manure_db.frt','manure_om.frt','landuse.lum','lum.dtl'}}


def _identities(raw,name):
    try:lines=raw.decode('utf-8-sig').splitlines()
    except UnicodeError as exc:raise SchemaError('SWAT+参数表需ASCII/UTF8') from exc
    if len(lines)<3:raise SchemaError('参数表缺完整表头和记录')
    if name=='weather-wgn.cli':
        rows=[line.split() for line in lines[1:] if line.strip()]
        if len(rows)%14:raise SchemaError('WGN需每站表头及完整12个月')
        result=[]
        for i in range(0,len(rows),14):
            if len(rows[i])!=5:raise SchemaError('WGN站点身份表头需名称及四项数值')
            try:metadata=tuple(Decimal(v) for v in rows[i][1:])
            except InvalidOperation as exc:raise SchemaError('WGN站点身份数值无效') from exc
            if not all(v.is_finite() for v in metadata):raise SchemaError('WGN站点身份数值非有限')
            result.append((rows[i][0],*metadata))
        return result
    rows=[line.split() for line in lines[2:] if line.strip()]
    if name=='soils.sol':
        def identity(row):
            try:float(row[0]);return '<layer>'
            except ValueError:return row[0]
        return [identity(row) for row in rows]
    if name.endswith('.con') or name=='hru-data.hru':return [tuple(row[:2]) for row in rows]
    return [row[0] for row in rows]


def attach_inputs(files,payload,mode):
    from ecodata.interface.swat_runner import check_project
    if mode not in TABLES:raise SchemaError('未登记此参数接入类型')
    replacements={}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            entries=z.infolist()
            if len({e.filename for e in entries})!=len(entries):raise SchemaError('转换包重名')
            for e in entries:
                name=e.filename
                if '/' in name or '\\' in name or ':' in name or '..' in name:raise SchemaError('转换包必须为安全平铺文件')
                if name in TABLES[mode]:replacements[name]=z.read(e)
                elif not (name.startswith('README') or name.endswith('.json')):raise SchemaError('转换包出现未登记文件：'+name)
    except zipfile.BadZipFile as exc:raise SchemaError('转换包不是有效ZIP') from exc
    if not replacements:raise SchemaError('转换包缺本类别参数表')
    updates=[];result=dict(files)
    for name,raw in replacements.items():
        if name not in files:raise SchemaError('工程没有同名目标文件：'+name+'；不猜测映射、不新增未引用文件')
        if _identities(raw,name)!=_identities(files[name],name):raise SchemaError(name+'记录标识/顺序不一致；不能直接覆盖跨表引用，请先提供准确映射')
        result[name]=raw
        updates.append(dict(file=name,before_sha256=hashlib.sha256(files[name]).hexdigest(),after_sha256=hashlib.sha256(raw).hexdigest()))
    checked=check_project(result)
    if checked['errors']:raise SchemaError('接入后工程基本检查失败：'+'；'.join(checked['errors']))
    return result,dict(mode=mode,replacements=updates,scope='Same filenames and record identities/order only; direct CIO/weather checks passed; deep database references, physical suitability and native execution still require verification',warnings=checked['warnings'])
