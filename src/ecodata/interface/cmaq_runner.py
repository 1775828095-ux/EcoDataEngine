"""Registered CMAQ5.5 CB6R5 five-minute acceptance, prepared inputs only."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,threading,uuid,msvcrt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from ecodata.models.adapters.cmaq_gridded_emissions import export_gridded_emissions
from ecodata.interface.camx_runner import linux

BASE=Path('D:/EcoDataEngine')
CASE=BASE/'cases/CMAQ_20261024/official_12NE3'
RUNS=BASE/'outputs/cmaq_jobs'
EXE=BASE/'work/cmaq55/cctm_serial_20261021/CCTM/scripts/BLD_CCTM_v55_gcc_cb6r5_ae7_aq_m3dry/CCTM_v55.exe'
PIN='12b1c59bbd1989251d07226c40d33be9e3ed0fc0692211d43e3253e4e606f022'
LOCK=threading.Lock()
LEASE=None

def acquire_lease():
    """One native run across CLI and Streamlit processes on this host."""
    RUNS.mkdir(parents=True,exist_ok=True)
    stream=(RUNS/'native.lock').open('a+b')
    try:
        if stream.seek(0,2)==0:stream.write(b'0');stream.flush()
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        return stream
    except OSError:
        stream.close();raise ValueError('另一进程已有CMAQ任务运行中，请等待完成')

def release_lease():
    global LEASE
    if LEASE is not None:
        LEASE.close();LEASE=None
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def directory(ident):
    if not re.fullmatch('[0-9a-f]{32}',ident):raise ValueError('无效CMAQ任务编号')
    return RUNS/ident
def status(ident):return json.loads((directory(ident)/'result.json').read_text(encoding='utf-8'))
def save(folder,result):
    temp=folder/'result.tmp';temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(folder/'result.json')


def prepare_profile(folder,payload,mode,meteorology=None):
    """Exact same-case O3 sensitivity profile, native preprocessing, 224 complements."""
    from ecodata.interface.registered_conversion import member
    tool='bcon' if mode=='bcon_profile' else 'icon'
    name='BC_PROFILE_CMAQ55.csv' if tool=='bcon' else 'IC_PROFILE_CMAQ55.csv'
    raw=member(payload,name);reference=BASE/f'cases/CMAQ_20261026/matched_profiles/{tool}'/name
    if raw!=reference.read_bytes():raise ValueError('仅登记同案例35层O3平均剖面；旧44层/不同机制/不同值不能直接接入，不替换示例')
    (folder/name).write_bytes(raw)
    pins={'icon':'d09d326a1ee6dc05c52d49113170789822730b88f5cc691b49b8683e5b7d69e6','bcon':'75e6e42308c0a2ecf88b6c0355b5e82bc3de839b8668bf93e5d482d6f593e84a'}
    exe=BASE/f'work/cmaq55/native_build_20261021/{tool}/{tool}.exe'
    if sha(exe)!=pins[tool]:raise ValueError('剖面预处理程序哈希不匹配')
    pre=folder/'profile_preprocessing';pre.mkdir();native=pre/(tool.upper()+'.nc')
    env={'GRID_NAME':'2018_12NE3','GRIDDESC':linux(CASE/'inputs/GRIDDESC'),'IOAPI_ISPH':'20','IOAPI_LOG_WRITE':'F','PROMPTFLAG':'N','EXECUTION_ID':'CURRENT_ZIP_EXPLICIT_O3_PROFILE',tool.upper()+'_TYPE':'profile',('BC_PROFILE' if tool=='bcon' else 'IC_PROFILE'):linux(folder/name),('MET_BDY_3D_FIN' if tool=='bcon' else 'MET_CRO_3D_FIN'):linux(CASE/f'inputs/met/mcipv5.4/{"METBDY3D" if tool=="bcon" else "METCRO3D"}_12NE3_20180701.nc'),('BNDY_CONC_1' if tool=='bcon' else 'INIT_CONC_1'):linux(native)+' -v'}
    if meteorology is not None:
        env['GRIDDESC']=linux(meteorology/'GRIDDESC')
        env['MET_BDY_3D_FIN' if tool=='bcon' else 'MET_CRO_3D_FIN']=linux(meteorology/('METBDY3D.nc' if tool=='bcon' else 'METCRO3D.nc'))
    (pre/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
    with (pre/'console.log').open('wb') as log:
        proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','env',*[k+'='+v for k,v in env.items()],linux(exe)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if proc.returncode or 'completed successfully' not in (pre/'console.log').read_text(errors='replace'):raise ValueError('当前剖面原生预处理失败，查看profile_preprocessing日志')
    original=CASE/f'inputs/icbc/CCTM_{tool.upper()}_v54_cb6r5_ae7_aq_12NE3_20180701.nc';target=folder/f'complete_{tool}_from_profile.nc';shutil.copyfile(original,target)
    with Dataset(native) as src,Dataset(target,'r+') as dst,Dataset(original) as before:
        value=np.asarray(src['O3'][0]);assert value.shape==dst['O3'].shape[1:] and src['O3'].units.strip()==dst['O3'].units.strip()
        for t in range(dst['O3'].shape[0]):dst['O3'][t]=value
        for n,v in before.variables.items():
            if n not in ('O3','TFLAG'):np.testing.assert_array_equal(v[:],dst[n][:])
    return target,dict(input_sha256=hashlib.sha256(raw).hexdigest(),native_preprocessor=tool,native_executable_sha256=pins[tool],native_output=str(native),native_output_sha256=sha(native),converted_species=['O3'],preserved_other_species=224,scope='Explicit first-time same-case horizontal-mean O3 sensitivity profile; static profile over original timestamps; not independent observations or original spatial field equivalence')

def prepare(folder,route,converted_zip=None,mode=None,new_meteorology=False):
    meteorology=None;raw_report=None
    if new_meteorology:
        if route!='rebuild':raise ValueError('原始GFS气象新建仅用于新建路线')
        from ecodata.interface.cmaq_meteorology import build
        meteorology,raw_report=build(folder,CASE)
    source=CASE/'platform_no';cfg=json.loads((source/'config.json').read_text())
    original=next((CASE/'inputs/emis/merged_nobeis_norwc').glob('*.ncf'))
    if route=='existing':
        payload=(source/'A_official_finished_NO_decoded.nc').read_bytes()
    else:
        with Dataset(original) as data:
            no=np.asarray(data['NO'][:,0]);flags=np.asarray(data['TFLAG'][:,0]);g=cfg['grid']
            times=[pd.to_datetime(str(int(day)),format='%Y%j')+pd.Timedelta(hours=int(hour)//10000) for day,hour in flags]
            ds=xr.Dataset({'nitric_oxide':(('when','north','east'),no)},coords=dict(when=np.array(times,dtype='datetime64[ns]'),east=g['XORIG']+(np.arange(g['NCOLS'])+.5)*g['XCELL'],north=g['YORIG']+(np.arange(g['NROWS'])+.5)*g['YCELL']),attrs=dict(crs=cfg['source_crs']))
            ds.nitric_oxide.attrs['units']=data['NO'].units.strip();ds.east.attrs['units']=ds.north.attrs['units']='m'
            ds.to_netcdf(folder/'decoded_NO.nc',engine='scipy');ds.close()
        payload=(folder/'decoded_NO.nc').read_bytes()
    with xr.open_dataset(__import__('io').BytesIO(payload)) as data:outputs,report=export_gridded_emissions(data,cfg)
    (folder/'emissions.CMAQ55.nc').write_bytes(outputs['emissions.CMAQ55.nc'])
    current_report={}
    if converted_zip is not None:
        from ecodata.interface.registered_conversion import accept_netcdf
        if mode=='gridded_emissions':
            raw,current_report=accept_netcdf(converted_zip,'emissions.CMAQ55.nc',source/'converted/emissions.CMAQ55.nc')
            (folder/'emissions.CMAQ55.nc').write_bytes(raw)
        elif mode=='boundary_concentrations':
            raw,current_report=accept_netcdf(converted_zip,'boundary.CMAQ55.nc',CASE/'platform_boundary/reference.nc')
            (folder/'boundary.CMAQ55.nc').write_bytes(raw)
            original_boundary=CASE/'inputs/icbc/CCTM_BCON_v54_cb6r5_ae7_aq_12NE3_20180701.nc'
            shutil.copyfile(original_boundary,folder/'complete_boundary.nc')
            with Dataset(folder/'complete_boundary.nc','r+') as dst,Dataset('current',memory=raw) as src:
                dst.variables['O3'][:]=src.variables['O3'][:]
            current_report.update(converted_species=['O3'],complement_source=str(original_boundary),policy='All other original boundary species retained unchanged')
        elif mode in (None,'bcon_profile'):
            profile_target,current_report=prepare_profile(folder,converted_zip,mode,meteorology)
        else:raise ValueError('此转换类别尚未登记CCTM接入；不会用示例替代')
    remainder=folder/'official_except_NO.nc'
    if route=='existing':shutil.copy2(source/'official_gridded_except_NO.nc',remainder)
    else:
        with Dataset(original) as src,Dataset(remainder,'w',format='NETCDF3_CLASSIC') as dst:
            names=src.getncattr('VAR-LIST');species=[names[i:i+16].strip() for i in range(0,len(names),16)];keep=[i for i,s in enumerate(species) if s!='NO']
            if len(keep)!=len(species)-1:raise ValueError('NO物种表不唯一')
            for name,dim in src.dimensions.items():dst.createDimension(name,None if dim.isunlimited() else len(keep) if name=='VAR' else len(dim))
            dst.setncatts({k:src.getncattr(k) for k in src.ncattrs()});dst.NVARS=np.int32(len(keep));dst.setncattr('VAR-LIST',''.join(species[i].ljust(16) for i in keep))
            for name,var in src.variables.items():
                if name=='NO':continue
                new=dst.createVariable(name,var.dtype,var.dimensions,fill_value=var.getncattr('_FillValue') if '_FillValue' in var.ncattrs() else False)
                new.setncatts({k:var.getncattr(k) for k in var.ncattrs() if k!='_FillValue'});new[:]=var[:,keep,:] if name=='TFLAG' else var[:]
    with Dataset(original) as src,Dataset(remainder) as rem,Dataset(folder/'emissions.CMAQ55.nc') as converted:
        if 'NO' in rem.variables:raise ValueError('互补排放重复包含NO')
        np.testing.assert_array_equal(src['NO'][:],converted['NO'][:])
        for name,var in src.variables.items():
            if name not in ('NO','TFLAG'):np.testing.assert_array_equal(var[:],rem[name][:])
        np.testing.assert_array_equal(src['TFLAG'][:,0,:],converted['TFLAG'][:,0,:])
        count=int(src['NO'][:].size)
    script=(CASE/'native_platform_01/run.csh').read_text()
    replacements={linux(CASE/'native_platform_01'):linux(folder),linux(source/'converted/emissions.CMAQ55.nc'):linux(folder/'emissions.CMAQ55.nc'),linux(source/'official_gridded_except_NO.nc'):linux(remainder)}
    for old,new in replacements.items():
        if old not in script:raise ValueError('登记脚本路径不匹配：'+old)
        script=script.replace(old,new)
    if converted_zip is not None and mode=='boundary_concentrations':
        script,boundary_count=re.subn(r'(?m)^\s*setenv BNDY_CONC_1 .*$', '  setenv BNDY_CONC_1 '+linux(folder/'complete_boundary.nc'),script)
        if boundary_count!=1:raise ValueError('登记边界引用不唯一')
    if converted_zip is not None and mode in (None,'bcon_profile'):
        key='BNDY_CONC_1' if mode=='bcon_profile' else 'INIT_CONC_1'
        script,n=re.subn(r'(?m)^\s*setenv '+key+r' .*$', '  setenv '+key+' '+linux(profile_target),script)
        if n!=1:raise ValueError('登记剖面目标引用不唯一')
    if not re.search(r'set NSTEPS\s*=\s*000500',script):raise ValueError('运行时长不是登记的5分钟')
    if 'set CLOBBER_DATA = FALSE' not in script:raise ValueError('拒绝覆盖已有结果')
    from ecodata.interface.cmaq_external_project import instrument
    script=instrument(script,folder)
    (folder/'work').mkdir();(folder/'run.csh').write_text(script,encoding='utf-8',newline='\n')
    result=dict(conversion=report,current_conversion=current_report,conversion_mode=mode,NO_values_verified=count,source_sha256=sha(original),NO_sha256=sha(folder/'emissions.CMAQ55.nc'),complement_sha256=sha(remainder),retained_inputs='Official prepared MCIP meteorology, IC/BC, other gridded and point emissions, chemistry tables; not raw inventory reconstruction',original_header_and_emission_checks='IOAPI_CHECK_HEADERS=N, CTM_EMISCHK=N retained; not strict full-species acceptance')
    if meteorology is not None:
        from ecodata.interface.cmaq_meteorology import apply
        result['new_meteorology']=dict(build=raw_report,acceptance=apply(folder,CASE,meteorology))
        result['retained_inputs']='Original chemical ICBC/other gridded and point emissions/chemistry; new original GFS->WRF->MCIP replaces meteorology, not full raw inventory construction'
    (folder/'preparation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result

def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    path=next((folder/'output').glob('CCTM_CGRID_*.nc'))
    with Dataset(path) as data:
        if not np.all(data['TFLAG'][:]==np.array([2018182,500])):raise ValueError('最终场不是00:05')
        if data['O3'].shape!=(1,35,105,100) or data['O3'].units.strip()!='ppmV':raise ValueError('最终浓度结构或单位不匹配')
        values=np.asarray(data['O3'][0,0],dtype='float64')*1000
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('O3浓度无效')
        x=(float(data.XORIG)+(np.arange(100)+.5)*float(data.XCELL))/1000;y=(float(data.YORIG)+(np.arange(105)+.5)*float(data.YCELL))/1000
        xx,yy=np.meshgrid(x,y);pd.DataFrame(dict(x_km=xx.ravel(),y_km=yy.ravel(),O3_ppbv=values.ravel())).to_csv(folder/'O3_surface.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,7),layout='constrained');mesh=ax.pcolormesh(x,y,values,shading='nearest',cmap='viridis',vmin=0,vmax=110);ax.set_aspect('equal');ax.set(title='CMAQ 5.5 | 2018-07-01 00:05 UTC\nFirst model layer O3, instantaneous',xlabel='Lambert x (km)',ylabel='Lambert y (km)');fig.colorbar(mesh,ax=ax,label='O3 (ppbv)');fig.savefig(folder/'O3_surface.png',dpi=140);plt.close(fig)
        return dict(max_O3_ppbv=float(values.max()),shape=[35,105,100],species=int(data.NVARS),final_sha256=sha(path))

def execute(ident,route,converted_zip=None,mode=None,new_meteorology=False):
    folder=directory(ident);result=status(ident)
    try:
        if sha(EXE)!=PIN:raise ValueError('CMAQ可信程序哈希不匹配')
        result.update(prepare(folder,route,converted_zip,mode,new_meteorology));save(folder,result)
        with (folder/'launcher.log').open('wb') as console:
            proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','csh','-f',linux(folder/'run.csh')],stdout=console,stderr=subprocess.STDOUT,timeout=1800)
        logs=list((folder/'output/LOGS').glob('CTM_LOG*'))
        if not logs:raise ValueError('没有原生CCTM日志')
        text='\n'.join(p.read_text(errors='replace') for p in logs)
        if proc.returncode or 'PROGRAM COMPLETED SUCCESSFULLY' not in text or '2018182:000500' not in text or linux(folder/'emissions.CMAQ55.nc') not in text or 'PLATFORM_NO' not in text:raise ValueError('原生CCTM未完整完成5分钟或未读取本次NO流')
        if converted_zip is not None and mode in (None,'bcon_profile'):
            target=folder/f'complete_{"bcon" if mode=="bcon_profile" else "icon"}_from_profile.nc'
            if linux(target) not in text:raise ValueError('CCTM日志未证实读取当前剖面原生结果')
        if new_meteorology:
            for name in ('METCRO3D.nc','METCRO2D.nc','METBDY3D.nc'):
                if linux(folder/'raw_meteorology/mcip'/name) not in text:raise ValueError('CCTM未证实读取本次原始GFS气象：'+name)
        result.update(plot(folder));result.update(status='success',returncode=proc.returncode,engine_sha256=PIN)
    except Exception as exc:result.update(status='failed',error=str(exc))
    finally:
        try:save(folder,result)
        finally:release_lease();LOCK.release()

def start(route,converted_zip=None,mode=None,new_meteorology=False):
    global LEASE
    if route not in ('existing','rebuild'):raise ValueError('不支持的路线')
    if new_meteorology and route!='rebuild':raise ValueError('原始GFS气象新建仅用于新建路线')
    if not LOCK.acquire(blocking=False):raise ValueError('已有CMAQ任务运行中')
    ident=uuid.uuid4().hex
    try:
        LEASE=acquire_lease()
        folder=directory(ident);folder.mkdir(parents=True,exist_ok=False);save(folder,dict(id=ident,route=route,status='running',new_meteorology_requested=bool(new_meteorology),scope='Original GFS new meteorology plus prepared chemistry; not full raw inventory construction' if new_meteorology else 'Five-minute prepared official case; not full raw-data construction'))
        threading.Thread(target=execute,args=(ident,route,converted_zip,mode,new_meteorology),daemon=True).start();return ident
    except Exception:release_lease();LOCK.release();raise
