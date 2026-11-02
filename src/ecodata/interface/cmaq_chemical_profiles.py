"""Complete, explicit chemical profiles -> native ICON and BCON, without fill values."""
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import numpy as np
from netCDF4 import Dataset

TOOLS = {
    'icon': ('d09d326a1ee6dc05c52d49113170789822730b88f5cc691b49b8683e5b7d69e6', 'METCRO3D'),
    'bcon': ('75e6e42308c0a2ecf88b6c0355b5e82bc3de839b8668bf93e5d482d6f593e84a', 'METBDY3D'),
}
MET_NAMES = {'TFLAG', 'PRES', 'TA', 'RH', 'W_VEL', 'ZH', 'ZF', 'WVEL'}
REFERENCE_PROFILE_SHA256 = 'f390b4281f28b107e021d1072d72d4d9e262e06c7a7e904b8e924be0f22533f1'
REFERENCE_NATIVE_DEFAULTS = {'APNCOMI','APNCOMJ','APOCI','APOCJ','IEPOXP'}


def profile_contract(payload, required, accept_reference_defaults=False):
    """Separate explicit reference scenario; strict upload remains the default.

    No field is inserted. Only the exact audited EPA profile can opt into the
    five previously observed CCTM native minima. Arbitrary missing data fail.
    """
    if not accept_reference_defaults:
        return required, validate_profile(payload,required)
    if hashlib.sha256(payload).hexdigest()!=REFERENCE_PROFILE_SHA256:
        raise ValueError('原生默认模式只允许逐字节核验的官方参考剖面，不接受其他残缺资料')
    reduced={k:v for k,v in required.items() if k not in REFERENCE_NATIVE_DEFAULTS}
    report=validate_profile(payload,reduced)
    missing=set(required)-set(report['supplied_chemical_species'])
    if missing!=REFERENCE_NATIVE_DEFAULTS:
        raise ValueError('参考模式缺项与已审计五字段不一致')
    report['explicit_native_defaults_accepted']=True
    report['not_supplied_by_profile']=sorted(missing)
    report['native_default_value']=1e-30
    report['complete_input_supplied']=False
    return reduced,report


def required_species(reference):
    """Conservative contract: retain every chemical field supplied by this case."""
    with Dataset('reference', memory=Path(reference).read_bytes()) as d:
        return {name: v.units.strip() for name, v in d.variables.items() if name not in MET_NAMES}


def validate_profile(payload, required):
    if not isinstance(payload, bytes) or not payload or len(payload) > 8*1024**2:
        raise ValueError('完整剖面必须为非空CSV，且不超过8MiB')
    rows = list(csv.reader(line for line in payload.decode('utf-8-sig').splitlines() if line.strip() and not line.lstrip().startswith('#')))
    if not rows or len(rows[0]) < 4 or rows[0][:2] != ['VNAME', 'UNITS']:
        raise ValueError('完整剖面表头应为VNAME,UNITS,LAYER01,...，至少两层')
    layers = len(rows[0])-2
    if layers>256 or len(rows)>2050:
        raise ValueError('剖面超过256层或2048物种的入口限额')
    if len(set(rows[0][2:])) != layers:
        raise ValueError('剖面层名重复')
    species = {}; units = {}
    for row in rows[1:]:
        if len(row) != layers+2:
            raise ValueError('剖面行长度与层数不一致')
        name, unit = row[:2]
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,15}',name) or name in species or not unit:
            raise ValueError('物种名空缺、重复、过长或单位空缺：'+name)
        try:
            values = np.array([float(x.replace('D','E').replace('d','e')) for x in row[2:]],dtype='f8')
        except ValueError as exc:
            raise ValueError('剖面数值无法解析：'+name) from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('剖面包含缺测、非有限值或负值：'+name)
        species[name] = values; units[name] = unit.strip()
    for name,unit,increasing in [('ZH','m',True),('PRES','Pa',False)]:
        if name not in species or units[name] != unit:
            raise ValueError('缺少正确单位的垂直坐标：'+name)
        difference=np.diff(species[name])
        if not ((difference>0).all() if increasing else (difference<0).all()) or (name=='PRES' and (species[name]<=0).any()):
            raise ValueError('垂直坐标必须严格单调：'+name)
    missing=sorted(set(required)-set(species))
    mismatched={n:{'supplied':units[n],'required':u} for n,u in required.items() if n in species and units[n]!=u}
    if missing or mismatched:
        raise ValueError('完整化学剖面不满足当前案例的保守完整性要求；缺少物种='+','.join(missing)+'；单位不匹配='+json.dumps(mismatched,ensure_ascii=False))
    return dict(input_sha256=hashlib.sha256(payload).hexdigest(),layers=layers,
                supplied_chemical_species=sorted(set(species)-MET_NAMES),required_species=sorted(required),
                scientific_scope='Explicit horizontally uniform, time-independent background; native vertical interpolation, not local observations or original 3-D field reproduction')


def build_pair(folder, payload, required, metdir, grid_name, griddesc, accept_reference_defaults=False):
    """Execute only fixed installed tools after complete input validation."""
    required,report=profile_contract(payload,required,accept_reference_defaults)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    source=folder/'chemical_profile.csv';source.write_bytes(payload)
    def linux(path):
        p=Path(path).resolve();return '/mnt/'+p.drive[0].lower()+'/'+p.as_posix()[3:]
    report['tools']=[]
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for tool,(expected,metname) in TOOLS.items():
        exe=Path('D:/EcoDataEngine/work/cmaq55/native_build_20261021')/tool/(tool+'.exe')
        digest=hashlib.sha256(exe.read_bytes()).hexdigest()
        if digest!=expected:raise ValueError('化学预处理工具哈希不符：'+tool)
        metfiles=list(Path(metdir).glob(metname+'*.nc'))
        if len(metfiles)!=1:raise ValueError('目标气象文件不唯一：'+metname)
        with Dataset('met',memory=metfiles[0].read_bytes()) as met:
            target={key:met.getncattr(key) for key in ('NCOLS','NROWS','NLAYS','GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL','VGTYP','VGTOP','VGLVLS')}
        output=folder/(tool.upper()+'.nc')
        variables={'GRID_NAME':grid_name,'GRIDDESC':linux(griddesc),'IOAPI_ISPH':'20','IOAPI_LOG_WRITE':'F',
                   'PROMPTFLAG':'N','EXECUTION_ID':'COMPLETE_EXPLICIT_PROFILE_NO_COMPLEMENT_COPY',tool.upper()+'_TYPE':'profile',
                   'IC_PROFILE' if tool=='icon' else 'BC_PROFILE':linux(source),
                   'MET_CRO_3D_FIN' if tool=='icon' else 'MET_BDY_3D_FIN':linux(metfiles[0]),
                   'INIT_CONC_1' if tool=='icon' else 'BNDY_CONC_1':linux(output)+' -v'}
        (folder/(tool+'_environment.json')).write_text(json.dumps(variables,indent=2),encoding='utf-8')
        with (folder/(tool+'.log')).open('wb') as log:
            result=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','env',
                                   *[k+'='+v for k,v in variables.items()],linux(exe)],stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,timeout=180,env=os.environ.copy())
        success=result.returncode==0 and 'completed successfully' in (folder/(tool+'.log')).read_text(errors='replace')
        report['tools'].append(dict(tool=tool,returncode=result.returncode,success=success,executable_sha256=digest))
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if not success:raise ValueError('完整剖面原生预处理失败：'+tool)
        with Dataset('output',memory=output.read_bytes()) as d:
            if d.GDNAM.strip()!=grid_name:
                raise ValueError('原生输出网格名称不匹配')
            for key,value in target.items():
                if not np.allclose(d.getncattr(key),value,rtol=0,atol=1e-6):
                    raise ValueError('原生输出网格或垂直坐标不匹配：'+key)
            chemical=set(d.variables)-{'TFLAG'}
            if chemical != set(report['supplied_chemical_species']):
                raise ValueError('生成物种与完整上传剖面不一致，不采用残缺输出')
            for name in chemical:
                values=np.ma.asarray(d[name][:])
                if name in required and d[name].units.strip()!=required[name]:
                    raise ValueError('原生输出单位不满足机制：'+name)
                if np.ma.getmaskarray(values).any() or not np.isfinite(values).all() or (values<0).any():
                    raise ValueError('原生输出缺测、非有限或负值：'+name)
            report[tool]=dict(file=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                              species=len(chemical),layers=int(d.NLAYS),grid=d.GDNAM.strip())
    report['success']=True;report['cctm_validated']=False;report['full_raw_construction']=False
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
