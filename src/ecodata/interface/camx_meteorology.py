"""Bounded2016 raw GFS meteorology, explicit single-grid chemistry scenario.

Original prepared chemistry remains necessary. No independent emission
inventory processing, no extrapolation or conserved-column-mass claim.
"""
from pathlib import Path
import json,uuid,subprocess,struct,re,shutil
import numpy as np
from netCDF4 import Dataset
from pyproj import CRS,Transformer
from ecodata.interface.camx_vertical import layer_centers,interpolation_weights,interpolate

PINS={
 'gfsanl_4_20160609_1800_000.grb2':'04abacdf6cbb889f837ebc55c1ed38d7a6337a223524f252950190d1fc551da9',
 'gfsanl_4_20160610_0000_000.grb2':'97842841944fa7d02dd5bd9c2c15565a7ed36d8a7c718c90aec69a7a81c6f611',
 'gfsanl_4_20160610_0000_006.grb2':'2e16986ea4e11aa5aa6d86917e32a1eab6b5a2046b6801b566fd0748fa93ee2b'}
LAYER_MAPPING=[1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]


def validate_gfs_uploads(files):
    """Only the documented original case, never infer times from renamed files."""
    import hashlib
    if not isinstance(files,dict) or set(files)!=set(PINS):
        raise ValueError('需上传登记的三份原始GFS GRIB2，文件名/数量必须完整；当前不是任意区域入口')
    rows=[]
    for name,pin in PINS.items():
        raw=files[name]
        if not isinstance(raw,bytes) or not raw.startswith(b'GRIB') or hashlib.sha256(raw).hexdigest()!=pin:
            raise ValueError('原始GFS内容或哈希不匹配：'+name+'；不回退登记文件')
        rows.append(dict(name=name,sha256=pin,bytes=len(raw)))
    return rows


def build(folder,raw_gfs_files=None):
    from ecodata.interface.camx_runner import sha,linux,CASE
    raw=Path('D:/EcoDataEngine/cases/CAMx_20261026/raw_gfs_2016')
    sources=json.loads((raw/'sources.json').read_text())
    if raw_gfs_files is not None:
        validate_gfs_uploads(raw_gfs_files)
        target=folder/'uploaded_gfs';target.mkdir(exist_ok=False)
        sources=[dict(r) for r in sources]
        for r in sources:
            name=Path(r['file']).name
            (target/name).write_bytes(raw_gfs_files[name])
            r['file']=str(target/name)
        raw=target
    if [Path(r['file']).name for r in sources]!=list(PINS) or [r['forecast_lead_hours'] for r in sources]!=[0,0,6]:raise ValueError('登记2016GFS分析/六小时预报资料不完整或顺序错误')
    for r in sources:
        p=Path(r['file'])
        if p.resolve()!=(raw/p.name).resolve() or sha(p)!=PINS[p.name] or r['sha256']!=PINS[p.name]:raise ValueError('原始GFS来源文件或哈希不匹配')
        if not r['url'].startswith('https://www.ncei.noaa.gov/oa/prod-model/global-forecast-system/access/historical/analysis/'):raise ValueError('原始GFS来源地址不匹配')
    proj=CRS.from_proj4('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +R=6370000 +units=m +no_defs')
    lon,lat=Transformer.from_crs(proj,'EPSG:4326',always_xy=True).transform(288000,-234000)
    out=folder/'raw_meteorology'
    cfg=dict(id=uuid.uuid4().hex,linux_folder=linux(out),ref_lat=lat,ref_lon=lon,linux_grib_files=[linux(Path(r['file'])) for r in sources],raw_sources=sources)
    config=folder/'meteorology_configuration.json';config.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    worker=Path(__file__).with_name('camx_meteorology_worker.py')
    with (folder/'meteorology_launcher.log').open('wb') as log:
        proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','python3',linux(worker),linux(config)],stdout=log,stderr=subprocess.STDOUT,timeout=3000)
    if proc.returncode:raise ValueError('原始GFS→WRF失败，查看阶段日志；不会替换回原气象')
    stages=json.loads((out/'stages.json').read_text())
    if [r['stage'] for r in stages]!=['geogrid','ungrib','metgrid','real','wrf_12hours'] or not all(r['returncode']==0 and r['marker_seen'] for r in stages):raise ValueError('原始气象阶段未完整通过')
    wrf=out/'actual_wrfout.nc'
    with Dataset(wrf) as d:
        if int(d.BL_PBL_PHYSICS)!=2:raise ValueError('WRFCAMx MYJ与实际WRF物理方案不匹配')
        times=[b''.join(row).decode() for row in d['Times'][:]]
        if times[0]!='2016-06-09_18:00:00' or times[-1]!='2016-06-10_06:00:00' or len(times)!=13:raise ValueError('实际WRF时刻不完整')
    met=folder/'new_camx_meteorology';met.mkdir(exist_ok=False)
    source=CASE/'met/camx.2d.36km.20160610.nc'
    with Dataset(source) as d:
        if d['TFLAG'][0,0].tolist()!=[2016162,0] or d['snowage'].units.strip()!='hr':raise ValueError('原案例雪龄时间或单位不匹配')
        age=np.asarray(d['snowage'][0,0],dtype='<f4')
        if age.shape!=(91,158) or not np.isfinite(age).all() or age.min()<0:raise ValueError('原案例雪龄无效')
    def rec(b):return struct.pack('<i',len(b))+b+struct.pack('<i',len(b))
    (met/'snow_age_initial.bin').write_bytes(rec(struct.pack('<4i',16162,0,158,91))+rec(age.tobytes()))
    mc=dict(linux_folder=linux(met),linux_wrf=linux(wrf),layer_mapping=LAYER_MAPPING,snow_age_source=str(source),snow_age_source_sha256=sha(source))
    mp=folder/'wrfcamx_configuration.json';mp.write_text(json.dumps(mc,indent=2),encoding='utf8')
    with (folder/'wrfcamx_launcher.log').open('wb') as log:
        proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','python3',linux(Path(__file__).with_name('camx_wrfcamx_worker.py')),linux(mp)],stdout=log,stderr=subprocess.STDOUT,timeout=420)
    if proc.returncode:raise ValueError('本次WRFCAMx原生预处理失败')
    return met,dict(configuration=cfg,stages=stages,wrfcamx=mc,native_wrfcamx=json.loads((met/'native_result.json').read_text()),scope='原始GFS分析及六小时预报新建气象；原雪龄/化学/排放配套复用；新36km单网格20层场景，非原双网格复现或完整原始清单构建')


def apply(folder,met):
    from ecodata.interface.camx_runner import CASE,sha,linux
    with Dataset(met/'met3d.nc') as new,Dataset(CASE/'met/camx.3d.36km.20160610.nc') as old:
        if (int(new.NCOLS),int(new.NROWS),int(new.NLAYS))!=(158,91,20):raise ValueError('新CAMx气象维数不匹配')
        for key,want in [('XORIG',-2556000),('YORIG',-1872000),('XCELL',36000),('YCELL',36000)]:
            if abs(float(new.getncattr(key))-want)>.01:raise ValueError('新气象网格不匹配：'+key)
        for key in ['latitude','longitude']:
            if np.max(np.abs(np.asarray(new[key][:])-np.asarray(old[key][:])))>=.0001:raise ValueError('新气象地理位置不匹配：'+key)
        met_files=[]
        for p in met.glob('*.nc*'):
            with Dataset(p) as d:
                for name,v in d.variables.items():
                    if v.dtype.kind=='f':
                        supplied=np.ma.asarray(v[:])
                        if np.ma.getmaskarray(supplied).any():raise ValueError('新气象变量存在缺测：'+name)
                        x=supplied.compressed()
                        if not x.size or not np.isfinite(x).all():raise ValueError('新气象变量无效：'+name)
            met_files.append(dict(path=str(p),sha256=sha(p)))
        manifest=[]
        for name,ref,target,times in [('initial',CASE/'initial_3d5626e7fd8c/initial.nc','initial_new_vertical.nc',[(0,0)]),('boundary',CASE/'boundary_1e386ac9018b/boundary.nc','boundary_new_vertical.nc',[(0,0),(1,6)])]:
            current=folder/(name+'.CAMx7.40.nc');source=current if current.exists() else ref
            dest=folder/target;shutil.copyfile(source,dest);changes=[]
            with Dataset(dest,'r+') as d:
                names=[n for n,v in d.variables.items() if v.dimensions==('TSTEP','LAY','ROW','COL')]
                if len(names)!=48:raise ValueError('浓度原物种清单不完整')
                for t,hour in times:
                    if new['TFLAG'][hour,0].tolist()!=[2016162,hour*10000]:raise ValueError('新气象时间不匹配')
                    weights=interpolation_weights(layer_centers(old['z'][hour]),layer_centers(new['z'][hour]))
                    fields=[]
                    for n in names:
                        a=np.asarray(d[n][t]);b=interpolate(a,weights)
                        if a.min()<0 or b.min()<0:raise ValueError('浓度不能为负：'+n)
                        d[n][t]=b;fields.append(dict(name=n,units=d[n].units.strip(),max_absolute_change=float(np.max(np.abs(a-b)))))
                    changes.append(dict(time_index=t,meteorology_hour=hour,fields=fields))
            manifest.append(dict(category=name,source=str(source),source_sha256=sha(source),target=str(dest),target_sha256=sha(dest),changes=changes))
    text=(folder/'CAMx.in').read_text(encoding='ascii')
    replace={'Number_of_Grids':'1','Initial_Conditions':folder/'initial_new_vertical.nc','Boundary_Conditions':folder/'boundary_new_vertical.nc','Surface_Grid(1)':met/'landuse.nc','Met3D_Grid(1)':met/'met3d.nc','Met2D_Grid(1)':met/'met2d.nc','Vdiff_Grid(1)':met/'kv.nc.MYJ'}
    for key,value in replace.items():
        val=value if key=='Number_of_Grids' else "'"+linux(value)+"'"
        text,n=re.subn(r'(?m)^\s*'+re.escape(key)+r'\s*=.*$',' '+key+' = '+val+',',text)
        if n!=1:raise ValueError('控制参数不唯一：'+key)
    removed=[];lines=[]
    for line in text.splitlines():
        m=re.match(r'\s*([A-Za-z_][A-Za-z_0-9]*(?:\([^)]*\))?)\s*=',line);key=m[1] if m else ''
        if re.match(r'^(?:Surface_Grid|Met3D_Grid|Met2D_Grid|Vdiff_Grid|Emiss_Grid)\(2(?:,|\))',key) or key.startswith(('Nest_Meshing_Factor','Nest_Beg_I_Index','Nest_End_I_Index','Nest_Beg_J_Index','Nest_End_J_Index')) or key=='Nested_Grid_Restart':removed.append(key)
        else:lines.append(line)
    (folder/'CAMx.in').write_text('\n'.join(lines)+'\n',encoding='ascii')
    return dict(folder=str(met),files=met_files,inputs=manifest,removed_nested_settings=removed,vertical_policy='Linear concentration interpolation at AGL layer centres; no extrapolation/fill/new species. Not conservative column mass remapping.',scope='新单网格敏感性场景；不是原双网格结果复现或观测精度评估')
