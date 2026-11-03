"""CMAQ 5.5 ICON profile ASCII input; not CCTM I/O API initial conditions."""
import re
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError

VERSION='CMAQ-5.5-ICON-profile-CB6R3M_AE7_KMTBR'
UNITS={'ppmV','ug m-3','m-3','m2 m-3','Pa','m'}
FACTORS={('ppbV','ppmV'):0.001,('hPa','Pa'):100.0,('km','m'):1000.0}

def export_icon_profile(frame,config):
    if set(config)!={'version','layers','target_units','source_description'} or config.get('version')!=VERSION:
        raise SchemaError('仅支持CMAQ5.5 ICON/profile CB6R3M_AE7_KMTBR；需明确版本、层数、目标单位和来源')
    if type(config['layers']) is not int or config['layers']!=44:
        raise SchemaError('本专项仅支持已核对的44层剖面，不猜测或插值垂直层')
    target=config['target_units']
    if (not isinstance(target,dict) or not 4<=len(target)<=251
            or any(not re.fullmatch(r'[A-Z][A-Z0-9_]{0,15}',n) or u not in UNITS for n,u in target.items())
            or any(target.get(n)!=u for n,u in {'PRES':'Pa','ZH':'m','ZF':'m'}.items())):
        raise SchemaError('目标变量/单位须明确，至少包括PRES/Pa、ZH/m、ZF/m及一个物种')
    desc=config['source_description']
    if not isinstance(desc,str) or not desc or len(desc)>800 or not desc.isascii() or '\n' in desc or '\r' in desc:
        raise SchemaError('来源说明需单行ASCII，不能注入额外记录')
    if set(frame.columns)!={'variable','unit','layer','value'} or len(frame.columns)!=4:
        raise SchemaError('输入必须是variable,unit,layer,value长表；不能静默丢字段')
    data=frame.copy()
    if set(data.variable)!=set(target):raise SchemaError('变量与配置不一致或有字段丢失')
    try:
        data['layer']=pd.to_numeric(data.layer,errors='raise')
        data['value']=pd.to_numeric(data.value,errors='raise')
    except (ValueError,TypeError) as exc:raise SchemaError('层号/数值不是有效数值') from exc
    if not np.isfinite(data[['layer','value']].to_numpy(dtype=float)).all():
        raise SchemaError('存在缺失或非有限值，不填零')
    if (data.value<0).any():raise SchemaError('负值不允许静默截断')
    if set(data.layer)!=set(range(1,45)) or data.duplicated(['variable','layer']).any():
        raise SchemaError('层号需1..44完整整数且无重复')
    index=pd.MultiIndex.from_product([list(target),range(1,45)],names=['variable','layer'])
    arranged=data.set_index(['variable','layer']).reindex(index)
    if len(data)!=len(index) or arranged.isna().any().any():raise SchemaError('剖面有缺失记录或单位，禁止补造')
    arrays={};factors={}
    for name,unit in target.items():
        src=arranged.loc[name];declared=set(src.unit)
        if len(declared)!=1:raise SchemaError(f'{name}混合单位')
        actual=next(iter(declared))
        factor=1.0 if actual==unit else FACTORS.get((actual,unit))
        if factor is None:raise SchemaError(f'{name}不支持{actual}到{unit}；不猜测质量/摩尔转换')
        arr=src.value.to_numpy(dtype=float)*factor
        if not np.isfinite(arr).all():raise SchemaError('单位转换溢出')
        arrays[name]=arr;factors[name]=factor
    if (not (arrays['PRES']>0).all() or not (np.diff(arrays['PRES'])<0).all()
            or not (arrays['ZH']>0).all() or not (np.diff(arrays['ZH'])>0).all()
            or not (np.diff(arrays['ZF'])>0).all() or not (arrays['ZF']>arrays['ZH']).all()
            or not (arrays['ZH'][1:]>arrays['ZF'][:-1]).all()):
        raise SchemaError('压力应随层递减；层中心/顶面高度需递增且中心位于相邻层顶之间')
    lines=['# EcoData ICON profile candidate; no CCTM grid or timestamps', '# '+desc,
           'VNAME,UNITS,'+','.join(f'LAYER{i:02d}' for i in range(1,45))]
    # Fortran list-directed reading needs quoted character fields, especially units containing spaces.
    for name,unit in target.items():
        lines.append(f'"{name}","{unit}",'+','.join(repr(float(v)) for v in arrays[name]))
    report={'scope':'CMAQ 5.5 ICON ICTYPE=profile IC_PROFILE ASCII candidate',
            'rows':len(target),'chemical_species':len(target)-3,'layers':44,'values':len(data),
            'units':target,'conversion_factors':factors,'missing_policy':'reject; no fill or clipping',
            'time':'static profile; source annual 2016 average is not an hourly initial observation',
            'coordinates':'PRES/Pa, ZH and ZF/m; no geographic grid; ICON supplies target grid',
            'complete_model_input':False,'ICON_run_verified':False,
            'next_steps':'IC_PROFILE plus GRIDDESC/GRID_NAME and MET_CRO_3D_FIN with built ICON/profile; verify target chemical mechanism before CCTM use'}
    return {'IC_PROFILE_CMAQ55.csv':('\n'.join(lines)+'\n').encode('ascii')},report
