"""Registered original GFS -> WPS/WRF/MCIP; no raw chemistry reconstruction."""
from pathlib import Path
import json,uuid,subprocess,re
import numpy as np
from netCDF4 import Dataset
from pyproj import CRS,Transformer


def build(folder,case):
    from ecodata.interface.cmaq_runner import sha,linux
    source=Path('D:/EcoDataEngine/cases/CMAQ_20261026/raw_gfs_2018')
    records=json.loads((source/'sources.json').read_text())
    expected={'gfsanl_4_20180630_1800_000.grb2':'f399666d6fea68b4c24fe43a7d4c0f285b393c59b1c7ce06eabbcd431a52aeb7',
              'gfsanl_4_20180701_0000_000.grb2':'8e26387b6edb4a837ef70e6b70957e91c475d532d58736b23be2d2fb552ef604',
              'gfsanl_4_20180701_0600_000.grb2':'772b3660f18776dc45c1d390af3600793b233fb7ba8aebaa8356842521be0b3d'}
    if [Path(r['file']).name for r in records]!=list(expected):raise ValueError('原始GFS三时刻资料不完整或顺序错误')
    for r in records:
        p=source/Path(r['file']).name
        if p.resolve()!=Path(r['file']).resolve() or not r['url'].startswith('https://www.ncei.noaa.gov/oa/prod-model/global-forecast-system/access/historical/analysis/'):raise ValueError('GFS来源不属于登记官方资料')
        pin=expected[p.name]
        if sha(p)!=r['sha256'] or (pin and sha(p)!=pin):raise ValueError('GFS原始文件哈希不匹配')
    proj=CRS.from_proj4('+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +R=6370000 +units=m +no_defs')
    lon,lat=Transformer.from_crs(proj,'EPSG:4326',always_xy=True).transform(1740000,174000)
    with Dataset(case/'inputs/met/mcipv5.4/METCRO3D_12NE3_20180701.nc') as d:levels=d.VGLVLS.tolist()
    out=folder/'raw_meteorology';cfg=dict(id=uuid.uuid4().hex,linux_folder=linux(out),ref_lat=lat,ref_lon=lon,eta_levels=levels,linux_grib_files=[linux(Path(r['file'])) for r in records],raw_sources=records)
    config=folder/'meteorology_configuration.json';config.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    worker=Path(__file__).with_name('cmaq_meteorology_worker.py')
    with (folder/'meteorology_launcher.log').open('wb') as log:
        proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','python3',linux(worker),linux(config)],stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    if proc.returncode:raise ValueError('原始GFS气象构建失败，查看meteorology_launcher.log及阶段日志')
    stages=json.loads((out/'stages.json').read_text())
    if [s['stage'] for s in stages]!=['geogrid','ungrib','metgrid','real','wrf_7hours','mcip'] or not all(s['returncode']==0 and s['marker_seen'] for s in stages):raise ValueError('原始气象阶段未完整通过')
    return out/'mcip',dict(configuration=cfg,stages=stages,scope='New original GFS meteorology; original chemical inputs reused, not full raw emission inventory construction')


def apply(folder,case,met):
    from ecodata.interface.cmaq_runner import linux,sha
    with Dataset(met/'METCRO3D.nc') as d,Dataset(case/'inputs/met/mcipv5.4/METCRO3D_12NE3_20180701.nc') as ref:
        for key in ['GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL','NCOLS','NROWS','NLAYS','VGTYP','VGTOP']:
            if abs(float(d.getncattr(key))-float(ref.getncattr(key)))>.001:raise ValueError('新气象网格/垂直参数不匹配：'+key)
        np.testing.assert_allclose(d.VGLVLS,ref.VGLVLS,rtol=0,atol=1e-6)
        if d['TFLAG'][0,0].tolist()!=[2018182,0] or d['TFLAG'][-1,0].tolist()!=[2018182,10000]:raise ValueError('新气象时刻不匹配')
    manifests=[]
    for p in met.glob('*.nc'):
        with Dataset(p) as d:
            for name,v in d.variables.items():
                if v.dtype.kind=='f':
                    values=np.ma.asarray(v[:]).compressed()
                    if not values.size or not np.isfinite(values).all():raise ValueError('新气象无有效值或存在非有限值：'+name)
        manifests.append(dict(path=str(p),sha256=sha(p)))
    text=(folder/'run.csh').read_text()
    text,n=re.subn(r'(?m)^setenv GRIDDESC .*$', 'setenv GRIDDESC '+linux(met/'GRIDDESC'),text)
    if n!=1:raise ValueError('GRIDDESC引用不唯一')
    text,n=re.subn(r'(?m)^set METpath\s*=.*$', 'set METpath = '+linux(met),text)
    if n!=1:raise ValueError('气象目录引用不唯一')
    for key,name in {'GRID_BDY_2D':'GRIDBDY2D','GRID_CRO_2D':'GRIDCRO2D','GRID_CRO_3D':'GRIDCRO3D','GRID_DOT_2D':'GRIDDOT2D','MET_CRO_2D':'METCRO2D','MET_CRO_3D':'METCRO3D','MET_DOT_3D':'METDOT3D','MET_BDY_3D':'METBDY3D','LUFRAC_CRO':'LUFRAC_CRO'}.items():
        text,n=re.subn(r'(?m)^\s*setenv '+key+r' .*$', '  setenv '+key+' '+linux(met/(name+'.nc')),text)
        if n!=1:raise ValueError('气象文件引用不唯一：'+key)
    (folder/'run.csh').write_text(text,encoding='utf8',newline='\n')
    return dict(folder=str(met),files=manifests,grid_and_time_verified=True)
