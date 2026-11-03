"""CALPUFF5.8.4 group17 discrete receptors into an existing user control file.

The rest of the supplied engineering configuration is preserved, not generated.
"""
import io,re,zipfile,hashlib
import numpy as np
from pyproj import CRS
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.swat_plus_hru import _table

FIELDS=('id','x','y','elevation','height_agl')
TOKEN=re.compile(r'!(.*?)!',re.S)


def export_receptors(content,cfg):
    if cfg.get('version')!='CALPUFF-5.8.4-group17':raise SchemaError('当前仅核对CALPUFF5.8.4离散受体group17读取规范')
    if cfg.get('elevation_vertical_reference')!='MSL':raise SchemaError('需明确MSL地面海拔；height_agl是地面以上高度')
    source=cfg.get('source')
    if not isinstance(source,dict) or not all(source.get(k) for k in ('kind','description','license')):raise SchemaError('需明确source.kind/description/license')
    markers=cfg.get('missing_values');units=cfg.get('units')
    if not isinstance(markers,list) or not isinstance(units,dict) or set(units)!=set(FIELDS)-{'id'}:raise SchemaError('需明确missing_values及x/y/elevation/height_agl单位')
    try:archive=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('受体需ZIP含CSV和用户已有完整CALPUFF控制文件') from exc
    with archive:
        names=archive.namelist();info=archive.infolist()
        table=cfg.get('table');control=cfg.get('control_file')
        if not isinstance(table,str) or not isinstance(control,str) or table==control or set(names)!={table,control} or len(names)!=2 or any(not re.fullmatch(r'[A-Za-z0-9_.-]+',n) or '..' in n for n in names) or sum(i.file_size for i in info)>32*1024**2:raise SchemaError('ZIP仅允许安全单层CSV和实际控制文件两个成员，最多32MB')
        frame=_table(archive.read(table),FIELDS,cfg)
        try:text=archive.read(control).decode('ascii')
        except UnicodeError as exc:raise SchemaError('控制文件需目标版本ASCII') from exc
    if not 1<=len(frame)<=11000:raise SchemaError('受体需1..11000条（5.8.4默认编译上限）')
    if frame.id.duplicated().any() or any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',str(v)) for v in frame.id):raise SchemaError('外部受体id需唯一安全标识；native group17按行顺序记录')
    tokens=list(TOKEN.finditer(text));assignments={}
    nrec_token=None
    for token in tokens:
        m=re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*',token.group(1),re.S)
        if m:
            key=m.group(1).upper()
            if key in {'PMAP','IUTMZN','UTMHEM','DATUM','NX','NY','DGRIDKM','XORIGKM','YORIGKM','NREC'}:
                if key in assignments:raise SchemaError(f'控制文件活动变量{key}重复')
                assignments[key]=m.group(2).strip()
                if key=='NREC':nrec_token=token
    required={'PMAP','IUTMZN','UTMHEM','DATUM','NX','NY','DGRIDKM','XORIGKM','YORIGKM','NREC'}
    if set(assignments)!=required or nrec_token is None:raise SchemaError('控制文件缺明确投影/网格或NREC；不创建空模板')
    if assignments['PMAP']!='UTM' or assignments['UTMHEM'] not in {'N','S'}:raise SchemaError('当前受体profile仅支持明确UTM南/北半球控制网格')
    try:
        zone=int(assignments['IUTMZN']);nx=int(assignments['NX']);ny=int(assignments['NY']);old_count=int(assignments['NREC'])
        spacing=float(assignments['DGRIDKM']);x0=float(assignments['XORIGKM']);y0=float(assignments['YORIGKM'])
    except ValueError as exc:raise SchemaError('控制文件网格或NREC数值无效') from exc
    if not 1<=zone<=60 or not 1<=nx<=265 or not 1<=ny<=265 or not 0<=old_count<=11000 or not np.isfinite([spacing,x0,y0]).all() or spacing<=0:raise SchemaError('控制文件网格范围无效')
    datum=assignments['DATUM'];hem=assignments['UTMHEM']
    if datum=='WGS-84':target=CRS.from_epsg((32600 if hem=='N' else 32700)+zone)
    elif datum=='NAS-C' and hem=='N' and zone<=22:target=CRS.from_epsg(26700+zone)
    else:raise SchemaError('当前只核对WGS-84 UTM或NAS-C/NAD27北半球UTM，其他datum不猜测')
    try:source_crs=CRS.from_user_input(cfg.get('source_crs'))
    except Exception as exc:raise SchemaError('需明确受体source_crs') from exc
    if not source_crs.equals(target,ignore_axis_order=True):raise SchemaError('受体源CRS与控制网格不符；本profile不隐式做基准转换')
    after=[t for t in tokens if t.start()>nrec_token.start()]
    if not after or after[0].group(1).strip().upper()!='END':raise SchemaError('NREC子组缺紧接END终止符')
    receptor_tokens=after[1:]
    if len(receptor_tokens)!=old_count*2:raise SchemaError('已有受体子组数与NREC不符或group17后有不支持活动记录')
    for i in range(old_count):
        t,end=receptor_tokens[2*i:2*i+2]
        if not re.fullmatch(r'\s*X\s*=.*',t.group(1),re.I|re.S) or end.group(1).strip().upper()!='END':raise SchemaError('旧受体子组需X数组及END，不忽略未知记录')
    columns=[]
    for field in FIELDS[1:]:
        target_unit='km' if field in {'x','y'} else 'm';scale,offset=unit_transform(units[field],target_unit)
        if offset:raise SchemaError('坐标/高度单位转换不得带偏移')
        try:a=np.array([float(v) for v in frame[field]],dtype=float)
        except ValueError as exc:raise SchemaError(f'{field}需实数') from exc
        if not np.isfinite(a).all():raise SchemaError(f'{field}含缺测或非有限值')
        for marker in markers:
            try:numeric=float(marker)
            except (ValueError,TypeError):continue
            if np.any(a==numeric):raise SchemaError(f'{field}含缺测标记')
        columns.append(a*scale)
    values=np.column_stack(columns)
    if not np.isfinite(values).all() or np.any(values[:,3]<0):raise SchemaError('高度换算无效或height_agl为负')
    if np.any((values[:,0]<x0)|(values[:,0]>=x0+nx*spacing)|(values[:,1]<y0)|(values[:,1]>=y0+ny*spacing)):raise SchemaError('受体超出所给控制网格；不截取或移动坐标')
    tolerance=cfg.get('max_float32_error_m')
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not np.isfinite(tolerance) or tolerance<0:raise SchemaError('需事先给出max_float32_error_m')
    errors=np.abs(values-values.astype('f4').astype(float))*[1000,1000,1,1]
    if not np.isfinite(errors).all() or np.max(errors)>tolerance:raise SchemaError('原生float32坐标/高度舍入超过事先给定米制阈值')
    # Group17 is the last READCF input group; only its active count and records change.
    prefix=text[:nrec_token.start()]+f'! NREC = {len(values)} !'+text[nrec_token.end():after[0].end()]
    lines=[prefix.rstrip(),'','Discrete receptors supplied by EcoDataEngine (group17b)']
    for row in values:lines.extend(['! X = '+', '.join(format(v,'.17g') for v in row)+' !','! END !'])
    notes='CALPUFF5.8.4离散受体group17接入用户已有控制文件；X/Y=km、地面海拔MSL和地上高度=m。原控制文件在NREC之前保持，原受体替换；不生成排放/气象/其它控制参数，不检查其外部文件引用是否可用，未运行CALPUFF。'
    return {'CALPUFF.INP':('\n'.join(lines)+'\n').encode('ascii'),'README_receptors.txt':notes.encode('utf8')},dict(scope=notes,source=source,receptors=len(values),source_crs=target.to_string(),external_id_order=list(frame.id),max_float32_error_m=float(np.max(errors)),source_control_sha256=hashlib.sha256(text.encode('ascii')).hexdigest(),complete_model_input=False,model_run_verified=False)
