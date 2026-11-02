"""Explicit LISTOS WRF -> MCIP route; no raw GRIB or emissions claim."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
from netCDF4 import Dataset

PIN='337e309abbf70e3b2674c509dcc586976739005304c3ee09dec7f18c23b49a63'
PRODUCTS=('GRIDBDY2D','GRIDCRO2D','GRIDDOT2D','LUFRAC_CRO','METBDY3D','METCRO2D','METCRO3D','METDOT3D','SOI_CRO')
GRID_ATTRS=('NCOLS','NROWS','NLAYS','GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL','VGTYP','VGTOP','VGLVLS')

def validate_inputs(files):
    if not isinstance(files,(list,tuple)) or not 1<=len(files)<=3:raise ValueError('LISTOS路线需1..3份完整WRF文件')
    reports=[];times=set();total=0
    for raw in files:
        if not isinstance(raw,bytes) or not 1024<=len(raw)<=128*1024**2:raise ValueError('每份WRF需1KiB..128MiB')
        total+=len(raw)
        with Dataset('WRF',memory=raw) as d:
            if 'Times' not in d.variables or 'Time' not in d.dimensions:raise ValueError('不是含完整Times的WRF输出')
            if any(k not in d.variables for k in ('P','PB','T','PH','PHB','U','V','W','HGT','XLAT','XLONG','LU_INDEX')):raise ValueError('WRF缺气象必需变量，不补造字段')
            stamps=[''.join(v.astype(str)) for v in np.asarray(d['Times'][:])]
            times.update(stamps)
            if not 12000-1e-6<=float(d.DX)<=12000+1e-6 or int(d.MAP_PROJ)!=1:raise ValueError('当前同案例路线需要12km Lambert WRF')
            if max(len(d.dimensions[n]) for n in ('west_east','south_north'))>101:raise ValueError('当前页面仅验收裁剪后单轴不超过101格的小型资料')
            if len(d.dimensions['bottom_top'])!=35:raise ValueError('当前LISTOS案例需要35层，不能自动改层')
        reports.append(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),times=stamps))
    if total>300*1024**2:raise ValueError('WRF总量超过300MiB')
    needed={'2018-08-04_23:00:00','2018-08-05_00:00:00','2018-08-05_01:00:00'}
    if not needed<=times:raise ValueError('需2018-08-04 23时及次日00/01UTC完整记录；不补插时次')
    return reports

def build(folder,files,reference_met):
    report=dict(inputs=validate_inputs(files),full_raw_construction=False,scope='Same-case uploaded WRF output -> MCIP. Original fixed LISTOS grid/window, 35layers; no FNL->WRF rerun, no chemistry or emissions generation')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    from ecodata.interface.independent_cases import linux
    paths=[]
    for i,raw in enumerate(files):
        p=folder/f'WRF_{i:02d}.nc';p.write_bytes(raw);paths.append(p)
    exe=Path('D:/EcoDataEngine/work/cmaq55/native_build_20261021/mcip/mcip.exe')
    if hashlib.sha256(exe.read_bytes()).hexdigest()!=PIN:raise ValueError('MCIP程序哈希变化')
    cfg=dict(folder=linux(folder),exe=linux(exe),pin=PIN,wrf=[linux(p) for p in paths])
    (folder/'configuration.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    mm=', '.join('"'+linux(p)+'"' for p in paths)
    nml=f'''&FILENAMES
 file_gd="{linux(folder)}/GRIDDESC", file_mm={mm}, ioform=1
/
&USERDEFS
 lpv=0, lwout=0, luvbout=1,
 mcip_start="2018-08-05-00:00:00.0000", mcip_end="2018-08-05-01:00:00.0000", intvl=60,
 coordnam="LamCon_40N_97W", grdnam="2018_12Listos", wrf_lc_ref_lat=40.0,
 btrim=-1, lprt_col=0, lprt_row=0
/
&WINDOWDEFS
 x0=5, y0=5, ncolsin=25, nrowsin=25
/
'''
    (folder/'namelist.mcip').write_text(nml,encoding='utf8')
    worker=Path(__file__).with_name('cmaq_mcip_worker.py')
    with (folder/'launcher.log').open('wb') as log:
        p=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','python3',linux(worker),linux(folder/'configuration.json')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=210)
    if p.returncode or not (folder/'native_status.json').is_file():raise ValueError('MCIP阶段失败，查看raw_meteorology日志')
    report['native']=json.loads((folder/'native_status.json').read_text(encoding='utf8'))
    if not report['native']['normal_termination']:raise ValueError('MCIP缺少原生正常结束标记')
    report['products']={}
    for name in PRODUCTS:
        path=folder/(name+'.nc');refs=list(Path(reference_met).glob(name+'*.nc'))
        if len(refs)!=1:raise ValueError('案例气象参照不唯一：'+name)
        with Dataset('new',memory=path.read_bytes()) as d,Dataset('reference',memory=refs[0].read_bytes()) as r:
            for attr in GRID_ATTRS:
                if not np.allclose(d.getncattr(attr),r.getncattr(attr),rtol=0,atol=1e-6):raise ValueError('本次气象不匹配目标网格/层次：'+attr)
            # MCIP appends _CR/_DOT/_BDY for each staggered product.
            if d.GDNAM.strip()!=r.GDNAM.strip():raise ValueError('目标产品网格名称与登记工程不匹配')
            if name.startswith(('MET','SOI')):np.testing.assert_array_equal(d['TFLAG'][:,0],[[2018217,0],[2018217,10000]])
            ranges={}
            for n,v in d.variables.items():
                a=np.ma.asarray(v[:])
                if np.ma.getmaskarray(a).any() or not np.isfinite(a).all():raise ValueError('MCIP含缺测或非有限值：'+n)
                if v.dtype.kind=='f':
                    if not (a>-1e20).all():raise ValueError('MCIP含BADVAL：'+n)
                    ranges[n]=dict(minimum=float(a.min()),maximum=float(a.max()))
        report['products'][name]=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ranges=ranges)
    report['success']=True
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report
