"""Official downstream acceptance of SST from unchanged platform GUI."""
from pathlib import Path
import datetime,hashlib,json,os,re,shutil,subprocess,sys,time
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);parser.add_argument('--job-id',required=True);parser.add_argument('--from-zero',action='store_true');args=parser.parse_args()
assert re.fullmatch('[0-9a-f]{32}',args.job_id)
R=Path(args.workspace).resolve();LOG=R/'native_logs';LOG.mkdir()
TAG='from_zero' if args.from_zero else 'existing'
BASE=Path('/opt/ecodata-wrf/cases/fnl20230331')
CASE=BASE/('sst_ui_'+args.job_id)
assert not CASE.exists();CASE.mkdir()
WPS=Path('/opt/ecodata-wrf/WPS-4.7.0');WRF=Path('/opt/ecodata-wrf/WRFV4.8.0')
RUN=CASE/'wps';REAL=CASE/'real';READ=CASE/'wrf_read'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):(LOG/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def link(src,dst):
    if dst.is_symlink():assert dst.resolve()==src.resolve()
    elif dst.exists():raise RuntimeError('Refuse overwrite '+str(dst))
    else:dst.symlink_to(src)
def write_once(p,text):
    if p.exists():assert p.read_text()==text,p
    else:p.write_text(text)
def command(tag,exe,cwd):
    start=time.monotonic();code=None;timeout=False
    with (LOG/(tag+'.log')).open('wb') as f:
        try:code=subprocess.run([str(exe)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,GFORTRAN_UNBUFFERED_ALL='1'),timeout=600).returncode
        except subprocess.TimeoutExpired:timeout=True
    record={'command':[str(exe)],'cwd':str(cwd),'returncode':code,'timeout':timeout,
        'wall_seconds':round(time.monotonic()-start,3),'wall_limit_seconds':600,
        'ended':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    save(tag+'.json',record);print(json.dumps(record),flush=True)
    assert code==0 and not timeout,(tag,'See log')
def export(tag,files):
    dest=R;rows=[]
    for p in files:
        assert p.is_file() and p.stat().st_size>0
        d=dest/p.name.replace(':','_');shutil.copyfile(p,d);assert sha(d)==sha(p)
        rows.append({'linux_file':str(p),'copy':str(d),'bytes':d.stat().st_size,'sha256':sha(d)})
    save(tag+'_outputs.json',rows)
def change(text,key,value,section='time_control'):
    value=str(value);pat=r'(?m)^(\s*'+re.escape(key)+r'\s*=)[^\n]*'
    text,n=re.subn(pat,lambda m:m[1]+' '+value+',',text);assert n<=1
    if not n:text=text.replace('&'+section,'&'+section+'\n '+key+' = '+value+',',1)
    return text
def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    for src,name in [(WPS/'geogrid.exe','geogrid.exe'),(WPS/'metgrid.exe','metgrid.exe'),
        (WPS/'geogrid/GEOGRID.TBL.ARW','GEOGRID.TBL'),(WPS/'metgrid/METGRID.TBL.ARW','METGRID.TBL')]:link(src,RUN/name)
    conf=(BASE/'reference_wps/namelist.wps').read_text()
    for before,after in [("2023-03-31_00:00:00","2023-03-31_12:00:00"),("2023-04-01_00:00:00","2023-03-31_18:00:00"),
                         ('dx = 30000','dx = 6000'),('dy = 30000','dy = 6000'),('ref_lat = 38.0','ref_lat = 30.0'),
                         ('ref_lon = -90.0','ref_lon = -60.0'),('stand_lon = -90.0','stand_lon = -60.0')]:
        assert before in conf;conf=conf.replace(before,after)
    conf=conf.replace(" fg_name = 'FILE',"," fg_name = 'FILE',\n constants_name = './ECODATA_SST_CONSTANT',")
    assert 'constants_name' in conf
    write_once(RUN/'namelist.wps',conf);shutil.copyfile(RUN/'namelist.wps',LOG/'namelist.wps')
    if args.from_zero:
        link(WPS/'ungrib.exe',RUN/'ungrib.exe');link(WPS/'ungrib/Variable_Tables/Vtable.GFS',RUN/'Vtable')
        originals=sorted((BASE/'severe_outbreak').glob('*.grib2'));assert len(originals)==5
        for i,p in enumerate(originals):link(p,RUN/('GRIBFILE.AA'+chr(65+i)))
    else:
        for name in ['FILE:2023-03-31_12','FILE:2023-03-31_18']:link(BASE/'reference_wps'/name,RUN/name)
    src=R/'ECODATA_SST_CONSTANT'
    assert src.is_file(),'Must use checked actual GUI download'
    link(src,RUN/'ECODATA_SST_CONSTANT')
    save('staging.json',{'domain':'100x100,6km,Lambert30N60W','times':['2023-03-31_12:00:00','2023-03-31_18:00:00'],
      'SST_source':str(src),'SST_sha256':sha(src),'official_complement':'same-date NOAA FNL fields, static NCAR lowres geography',
      'SST_temporal_method':'explicit constant snapshot only; no hourly source values fabricated',
      'platform_source_modified':False,'test_operator':'Codex; not human user acceptance'})
def geogrid():
    if not args.from_zero:
        link(Path('/opt/ecodata-wrf/cases/sst20230331_20261018/wps/geo_em.d01.nc'),RUN/'geo_em.d01.nc')
        export('geogrid',[RUN/'geo_em.d01.nc']);return
    command('geogrid',RUN/'geogrid.exe',RUN)
    assert 'Successful completion' in (RUN/'geogrid.log').read_text()
    shutil.copyfile(RUN/'geogrid.log',LOG/'geogrid_internal.log')
    export('geogrid',[RUN/'geo_em.d01.nc'])
def metgrid():
    command('metgrid',RUN/'metgrid.exe',RUN)
    assert 'Successful completion' in (RUN/'metgrid.log').read_text()
    shutil.copyfile(RUN/'metgrid.log',LOG/'metgrid_internal.log')
    products=sorted(RUN.glob('met_em.d01.*.nc'));assert len(products)==2
    export('metgrid',products)

def ungrib():
    if not args.from_zero:return
    command('ungrib',RUN/'ungrib.exe',RUN)
    assert 'Successful completion' in (RUN/'ungrib.log').read_text()
    for name in ['FILE:2023-03-31_12','FILE:2023-03-31_18']:assert sha(RUN/name)==sha(BASE/'reference_wps'/name)
def real():
    REAL.mkdir(exist_ok=True)
    for p in (BASE/'reference_real').iterdir():
        if p.is_symlink() and p.is_file() and not p.name.startswith('met_em.'):link(p.resolve(),REAL/p.name)
    for p in RUN.glob('met_em.d01.*.nc'):link(p,REAL/p.name)
    config=(BASE/'reference_real/namelist.input').read_text()
    values={'run_hours':6,'start_hour':12,'end_month':3,'end_day':31,'end_hour':18,
            'dx':6000,'dy':6000,'time_step':36}
    for key,value in values.items():config=change(config,key,value,'domains' if key in ['dx','dy','time_step'] else 'time_control')
    write_once(REAL/'namelist.input',config);shutil.copyfile(REAL/'namelist.input',LOG/'namelist.input.real')
    command('real',REAL/'real.exe',REAL)
    assert 'SUCCESS COMPLETE REAL_EM INIT' in (LOG/'real.log').read_text(errors='replace')
    export('real',[REAL/'wrfinput_d01',REAL/'wrfbdy_d01'])
def wrf_read():
    READ.mkdir(exist_ok=True)
    for p in REAL.iterdir():
        if p.name in ['wrfinput_d01','wrfbdy_d01'] or (p.is_symlink() and p.is_file() and not p.name.startswith('met_em.')):link(p.resolve(),READ/p.name)
    if not (READ/'wrf.exe').exists():link(WRF/'main/wrf.exe',READ/'wrf.exe')
    # Reuse tables computed by the identical CONUS microphysics setup in prior successful case.
    cached=BASE/'platform_user_20261018_wrf_read_diagnostic';cache=[]
    for name in ['qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat']:
        p=cached/name
        if p.exists():link(p,READ/name);cache.append({'path':str(p),'sha256':sha(p)})
    save('微物理缓存来源.json',{'files':cache,'reason':'same WRF4.8 binary and unchanged CONUS physics settings; no simulated fields reused'})
    config=(REAL/'namelist.input').read_text()
    values={'run_days':0,'run_hours':0,'run_minutes':0,'run_seconds':36,
            'start_minute':0,'start_second':0,'end_hour':12,'end_minute':0,'end_second':36,
            'history_interval':0,'history_interval_s':36,'frames_per_outfile':2,'debug_level':150}
    for key,value in values.items():config=change(config,key,value)
    write_once(READ/'namelist.input',config);shutil.copyfile(READ/'namelist.input',LOG/'namelist.input.wrf_read')
    command('wrf_read',READ/'wrf.exe',READ)
    assert 'SUCCESS COMPLETE WRF' in (LOG/'wrf_read.log').read_text(errors='replace')
    products=sorted(READ.glob('wrfout_d01_*'));assert products
    export('wrf_read',products)
if __name__=='__main__':
    stages=[]
    for name,func in [('prepare',prepare),('geogrid',geogrid),('ungrib',ungrib),('metgrid',metgrid),('real',real),('wrf_read',wrf_read)]:
        begin=time.monotonic();func();stages.append(dict(stage=name,success=True,seconds=time.monotonic()-begin))
    outputs=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in R.glob('wrfout*')]
    assert outputs
    (R/('native_'+TAG+'_outputs.json')).write_text(json.dumps(dict(stages=stages,outputs=outputs,simulation_seconds=36,reused_geometry=not args.from_zero,new_geogrid_executed=args.from_zero,scope='OISST constant snapshot; from_zero executes geogrid/ungrib from registered geography and raw FNL. Fixed domain/physics; not arbitrary new region'),indent=2))
