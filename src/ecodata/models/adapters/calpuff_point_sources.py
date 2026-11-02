"""CALPUFF 5.8.4/7.2.1 constant point sources into supplied group13.

Every modeled species has a user-supplied rate, including explicit zeros for
secondary species. No variable emissions, building downwash or missing defaults.
"""
import io,re,zipfile
import numpy as np
from pyproj import CRS
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.swat_plus_hru import _table
from ecodata.models.adapters.calpuff_receptors import TOKEN
FIELDS=('name','x','y','height','elevation','diameter','exit_velocity','exit_temperature','downwash','sigma_y','sigma_z','momentum_factor','platform_height')
TARGETS=dict(x='km',y='km',height='m',elevation='m',diameter='m',exit_velocity='m/s',exit_temperature='K',downwash='1',sigma_y='m',sigma_z='m',momentum_factor='1',platform_height='m')

def _real(value):
    text=format(value,'.17g')
    return text if '.' in text or 'e' in text.lower() else text+'.0'


def export_points(content,cfg):
    if cfg.get('version') not in {'CALPUFF-5.8.4-group13','CALPUFF-7.2.1-group13'} or cfg.get('emission_time_policy')!='constant_supplied':raise SchemaError('需CALPUFF-5.8.4-group13或CALPUFF-7.2.1-group13及constant_supplied；变排放另需开发')
    if cfg.get('elevation_vertical_reference')!='MSL':raise SchemaError('需明确MSL地面海拔，height为离地高度')
    source=cfg.get('source')
    if not isinstance(source,dict) or not all(source.get(k) for k in ('kind','description','license')):raise SchemaError('需source.kind/description/license')
    sigma_policy=cfg.get('initial_sigmas_policy','explicit')
    if sigma_policy not in {'explicit','preserve_source_omission'}:raise SchemaError('未知initial_sigmas_policy')
    required_fields=tuple(k for k in FIELDS if sigma_policy=='explicit' or k not in {'sigma_y','sigma_z'})
    required_units=set(TARGETS) if sigma_policy=='explicit' else set(TARGETS)-{'sigma_y','sigma_z'}
    units=cfg.get('units');markers=cfg.get('missing_values');species=cfg.get('species')
    if not isinstance(units,dict) or set(units)!=required_units or not isinstance(markers,list) or not isinstance(species,list) or not species:raise SchemaError('需完整几何字段单位、missing_values及物种映射')
    try:z=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:raise SchemaError('点源需CSV及已有控制文件ZIP') from exc
    with z:
        names=z.namelist();table=cfg.get('table');control=cfg.get('control_file')
        if len(names)!=2 or set(names)!={table,control} or table==control or any(not re.fullmatch('[A-Za-z0-9_.-]+',n) or '..' in n for n in names) or sum(e.file_size for e in z.infolist())>32*1024**2:raise SchemaError('ZIP需安全单层CSV和实际控制文件两项，最多32MB')
        try:text=z.read(control).decode('ascii')
        except UnicodeError as exc:raise SchemaError('控制文件需ASCII') from exc
        raw=z.read(table)
    tokens=list(TOKEN.finditer(text));assignments=[]
    for t in tokens:
        m=re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*',t.group(1),re.S)
        if m:assignments.append((m.group(1).upper(),m.group(2).strip(),t))
    needed={'PMAP','IUTMZN','UTMHEM','DATUM','NX','NY','DGRIDKM','XORIGKM','YORIGKM','NPT1','NPT2','IPTU','NSPT1','NAR1'}
    values={};positions={}
    for key,value,t in assignments:
        if key in needed:
            if key in values:raise SchemaError(f'控制变量{key}重复')
            values[key]=value;positions[key]=t
    if set(values)!=needed:raise SchemaError('控制文件缺完整网格或group13/14锚点，不造空控制')
    try:
        zone=int(values['IUTMZN']);nx=int(values['NX']);ny=int(values['NY']);old=int(values['NPT1'])
        x0=float(values['XORIGKM']);y0=float(values['YORIGKM']);spacing=float(values['DGRIDKM'])
        if int(values['NPT2'])!=0 or int(values['NSPT1'])!=0:raise SchemaError('当前只接常量点源；已有变排放/缩放不能丢弃')
    except ValueError as exc:raise SchemaError('控制网格/点源计数无效') from exc
    if values['PMAP']!='UTM' or not 1<=zone<=60 or not 1<=nx<=265 or not 1<=ny<=265 or not 0<=old<=200 or not np.isfinite([x0,y0,spacing]).all() or spacing<=0:raise SchemaError('仅明确UTM和有效控制网格')
    if values['DATUM']=='WGS-84' and values['UTMHEM'] in {'N','S'}:target=CRS.from_epsg((32600 if values['UTMHEM']=='N' else 32700)+zone)
    elif values['DATUM']=='NAS-C' and values['UTMHEM']=='N' and zone<=22:target=CRS.from_epsg(26700+zone)
    else:raise SchemaError('只支持WGS84 UTM或NAD27北半球UTM')
    try:sourcecrs=CRS.from_user_input(cfg['source_crs'])
    except Exception as exc:raise SchemaError('需明确source_crs') from exc
    if not sourcecrs.equals(target,ignore_axis_order=True):raise SchemaError('源CRS与控制网格冲突，不隐式做基准转换')
    model_species=[v for k,v,t in assignments if k=='CSPEC']
    if not 1<=len(model_species)<=20 or len(set(model_species))!=len(model_species):raise SchemaError('控制物种列表无效/重复或超过20个默认上限')
    mapping={}
    for s in species:
        if not isinstance(s,dict) or set(s)!={'name','column','unit'} or s['name'] in mapping:raise SchemaError('species每项需唯一name/column/unit')
        if s['unit'] not in {'g/s','kg/s','g/hour','kg/hour','g/h','kg/h'}:raise SchemaError('排放率需明确质量/秒或质量/小时，不猜测年度时长/分种')
        mapping[s['name']]=s
    if set(mapping)!=set(model_species):raise SchemaError('必须给每个控制物种的排放率；包括源明确给出的0，不能补零')
    last_cspec=max(t.end() for k,v,t in assignments if k=='CSPEC')
    remaining=[t for t in tokens if t.start()>=last_cspec]
    if not remaining or remaining[0].group(1).strip().upper()!='END':raise SchemaError('物种列表缺END')
    flags={}
    for t in remaining[1:]:
        if t.group(1).strip().upper()=='END':break
        m=re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*',t.group(1),re.S)
        if not m or m.group(1) not in model_species or m.group(1) in flags:raise SchemaError('控制物种选项不完整或重复')
        try:values_flag=[int(v.strip()) for v in m.group(2).split(',')]
        except ValueError as exc:raise SchemaError('控制物种选项须整数') from exc
        if len(values_flag)!=4 or values_flag[0]!=1 or values_flag[1] not in {0,1}:raise SchemaError('当前需已建模物种及明确发排放标志')
        flags[m.group(1)]=values_flag[1]
    if set(flags)!=set(model_species):raise SchemaError('控制物种选项缺项')
    fields=cfg.get('fields')
    if not isinstance(fields,dict) or set(fields)!=set(required_fields):raise SchemaError('点源需所有字段的明确映射')
    combined=dict(fields,**{f'__species_{i}':mapping[n]['column'] for i,n in enumerate(model_species)})
    frame=_table(raw,(*required_fields,*[f'__species_{i}' for i in range(len(model_species))]),dict(cfg,fields=combined))
    if sigma_policy=='preserve_source_omission':
        original_names=[v for k,v,t in assignments if k=='SRCNAM']
        if any(k=='SIGYZI' for k,v,t in assignments) or len(frame)!=old or frame.name.tolist()!=original_names:
            raise SchemaError('保留SIGYZI省略仅适用于原控制确实省略、源数量及名称顺序完全不变；不猜测新源扩散参数')
    if not 1<=len(frame)<=200 or frame.name.duplicated().any() or any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,12}',n) for n in frame.name):raise SchemaError('点源需1..200条，SRCNAM唯一安全ASCII最多12字符')
    arrays={}
    for field in frame.columns[1:]:
        try:a=np.asarray(frame[field],dtype=float)
        except ValueError as exc:raise SchemaError(f'{field}不是数值') from exc
        if not np.isfinite(a).all():raise SchemaError(f'{field}含缺测/NaN')
        for marker in markers:
            try:m=float(marker)
            except (ValueError,TypeError):continue
            if (a==m).any():raise SchemaError(f'{field}含缺测标记')
        if field in TARGETS:scale,offset=unit_transform(units[field],TARGETS[field])
        else:
            n=model_species[int(field.rsplit('_',1)[1])];scale,offset=unit_transform(mapping[n]['unit'],'g/s')
        a=a*scale+offset
        if not np.isfinite(a).all() or not np.isfinite(a.astype('f4')).all() or ((a!=0)&(a.astype('f4')==0)).any():raise SchemaError(f'{field}换算或float32溢出/下溢')
        arrays[field]=a
    for i,n in enumerate(model_species):
        if flags[n]==0 and (arrays[f'__species_{i}']!=0).any():raise SchemaError(f'{n}控制标为不排放，源值却非0；不静默忽略')
    for k in ('height','diameter','exit_temperature'):
        if (arrays[k]<=0).any():raise SchemaError(f'{k}必须正数')
    if any((arrays[k]<0).any() for k in ('exit_velocity',*(['sigma_y','sigma_z'] if sigma_policy=='explicit' else []),*[f'__species_{i}' for i in range(len(model_species))])):raise SchemaError('速度/初始扩散/排放不能负数，不截零')
    if (arrays['downwash']!=0).any() or (arrays['platform_height']!=0).any() or not np.isin(arrays['momentum_factor'],[0,1]).all():raise SchemaError('当前无下洗/平台profile：downwash/platform_height必须源显式0，momentum_factor必须0或1')
    if ((arrays['x']<x0)|(arrays['x']>=x0+nx*spacing)|(arrays['y']<y0)|(arrays['y']>=y0+ny*spacing)).any():raise SchemaError('点源超出控制网格')
    tolerance=cfg.get('max_float32_error_m')
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not np.isfinite(tolerance) or tolerance<0:raise SchemaError('需事先声明max_float32_error_m坐标舍入阈值')
    coordinate_error=max(float(np.max(np.abs(arrays[k]-arrays[k].astype('f4').astype(float))))*1000 for k in ('x','y'))
    if coordinate_error>tolerance:raise SchemaError('float32坐标误差超过声明米制阈值')
    # The authoritative active tokens, not explanatory headings, delimit the group.
    a,b=positions['NPT1'].start(),positions['NAR1'].start()
    region=[t for t in tokens if a<=t.start()<b]
    if a>=b:raise SchemaError('group13/14顺序无效')
    groups=[];current=[]
    for t in region:
        value=t.group(1).strip()
        if value.upper()=='END':groups.append(current);current=[]
        else:current.append(value.split('=',1)[0].strip().upper())
    if current or len(groups)!=old+1 or set(groups[0])!={'NPT1','NPT2','IPTU','NSPT1'} or any(set(g)-{'SRCNAM','X','SIGYZI','FMFAC','ZPLTFM'} or not {'SRCNAM','X'}<=set(g) or len(g)!=len(set(g)) for g in groups[1:]):raise SchemaError('已有group13子组/计数不符或有未知记录，不能丢弃')
    lines=[f'! NPT1 = {len(frame)} !','! NPT2 = 0 !','! IPTU = 1 !','! NSPT1 = 0 !','! END !']
    for i in range(len(frame)):
        x=[arrays[k][i] for k in ('x','y','height','elevation','diameter','exit_velocity','exit_temperature','downwash')]+[arrays[f'__species_{j}'][i] for j in range(len(model_species))]
        lines+=['! SRCNAM = '+frame.name.iloc[i]+' !','! X = '+', '.join(_real(v) for v in x)+' !']
        if sigma_policy=='explicit':lines+=['! SIGYZI = '+', '.join(_real(arrays[k][i]) for k in ('sigma_y','sigma_z'))+' !']
        lines+=['! FMFAC = '+_real(arrays['momentum_factor'][i])+' !','! ZPLTFM = '+_real(arrays['platform_height'][i])+' !','! END !']
    # Preserve every following explanatory heading too, not merely NAR1's
    # active assignment; the old constant subgroup ends at its last END.
    native=text[:a]+'\n'.join(lines)+'\n\n'+text[region[-1].end():]
    scope=cfg['version']+'常量点源，IPTU=1 g/s；用户已有控制中其它组保持。每物种/烟囱参数显式供给，无变排放、下洗、平台或自动分种；未检查所有外部引用，未运行CALPUFF。'
    return {'CALPUFF.INP':native.encode('ascii'),'README_point_sources.txt':scope.encode('utf8')},dict(scope=scope,source=source,initial_sigmas_policy=sigma_policy,point_sources=len(frame),species_order=model_species,source_crs=target.to_string(),emission_unit='g/s',time_processing='constant supplied rates; no time allocation',max_float32_coordinate_error_m=coordinate_error,float32_max_absolute_errors={k:float(np.max(np.abs(a-a.astype('f4').astype(float)))) for k,a in arrays.items()},complete_model_input=False,model_run_verified=False)
