"""Build independent 2016 GFS analysis/forecast meteorology on exact CAMx36km grid, trusted local tools."""
from pathlib import Path
import sys,json,os,re,shutil,subprocess,time,hashlib
cfg=json.loads(Path(sys.argv[1]).read_text());OUT=Path(cfg['linux_folder']);OUT.mkdir(parents=True,exist_ok=False)
ROOT=Path('/opt/ecodata-wrf/cases')/('camx_raw2016_'+cfg['id']);ROOT.mkdir(exist_ok=False)
BASE=Path('/opt/ecodata-wrf/cases/fnl20230331');PREV=BASE/'user_zip9_20261018';WPS=Path('/opt/ecodata-wrf/WPS-4.7.0');results=[]
def link(a,b):assert a.is_file(),a;b.symlink_to(a.resolve())
def run(stage,exe,where,marker,env=None,timeout=1800):
    begin=time.time()
    logpath=where/(stage+'.console.log')
    with logpath.open('wb') as log:r=subprocess.run([str(exe)],cwd=where,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    text=logpath.read_text(errors='replace')
    result=dict(stage=stage,returncode=r.returncode,marker_seen=marker in text,elapsed_seconds=time.time()-begin,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),log=str(logpath))
    results.append(result);(OUT/'stages.json').write_text(json.dumps(results,indent=2));print(stage,r.returncode,result['marker_seen'],flush=True)
    assert r.returncode==0 and result['marker_seen'],result
def rewrite(text,values):
    for k,v in values.items():
        text,n=re.subn(r'(?m)^\s*'+k+r'\s*=.*$', ' '+k+' = '+str(v)+',',text)
        if n==0:
            if k in ['eta_levels','p_top_requested','num_metgrid_soil_levels']:text=text.replace('&domains','&domains\n '+k+' = '+str(v)+',',1)
            else:raise ValueError('Unknown original namelist field '+k)
        else:assert n==1,k
    return text
wps=ROOT/'wps';wps.mkdir()
for a,b in [(WPS/'geogrid.exe','geogrid.exe'),(WPS/'ungrib.exe','ungrib.exe'),(WPS/'metgrid.exe','metgrid.exe'),(WPS/'geogrid/GEOGRID.TBL.ARW','GEOGRID.TBL'),(WPS/'ungrib/Variable_Tables/Vtable.GFS','Vtable'),(WPS/'metgrid/METGRID.TBL.ARW','METGRID.TBL')]:link(a,wps/b)
base=(BASE/'reference_wps/namelist.wps').read_text();geog=re.search(r"geog_data_path\s*=\s*'([^']+)'",base).group(1)
namelist=f'''&share
 wrf_core='ARW', max_dom=1,
 start_date='2016-06-09_18:00:00',
 end_date='2016-06-10_06:00:00',
 interval_seconds=21600,
 io_form_geogrid=2,
/
&geogrid
 parent_id=1, parent_grid_ratio=1, i_parent_start=1, j_parent_start=1,
 e_we=171, e_sn=104,
 geog_data_res='lowres', dx=36000, dy=36000,
 map_proj='lambert', ref_lat={cfg['ref_lat']}, ref_lon={cfg['ref_lon']},
 truelat1=33.0, truelat2=45.0, stand_lon=-97.0,
 geog_data_path='{geog}',
 opt_geogrid_tbl_path='./',
/
&ungrib
 out_format='WPS', prefix='FILE',
/
&metgrid
 fg_name='FILE', io_form_metgrid=2, opt_metgrid_tbl_path='./',
/
'''
(wps/'namelist.wps').write_text(namelist)
for i,p in enumerate(cfg['linux_grib_files']):link(Path(p),wps/('GRIBFILE.AA'+chr(65+i)))
run('geogrid',wps/'geogrid.exe',wps,'Successful completion')
run('ungrib',wps/'ungrib.exe',wps,'Successful completion')
run('metgrid',wps/'metgrid.exe',wps,'Successful completion')
header=subprocess.check_output(['ncdump','-h',str(next(wps.glob('met_em.d01.*.nc')))],text=True)
nmet=int(re.search(r'num_metgrid_levels\s*=\s*(\d+)',header).group(1));nsoil=int(re.search(r'num_st_layers\s*=\s*(\d+)',header).group(1))
real=ROOT/'real';real.mkdir()
for p in (PREV/'real').iterdir():
    if p.is_symlink() and p.is_file() and not p.name.startswith('met_em.'):link(p,real/p.name)
for p in wps.glob('met_em.d01.*.nc'):link(p,real/p.name)
domains=dict(run_days=0,run_hours=12,run_minutes=0,run_seconds=0,start_year=2016,start_month=6,start_day=9,start_hour=18,end_year=2016,end_month=6,end_day=10,end_hour=6,e_we=171,e_sn=104,e_vert=48,dx=36000,dy=36000,time_step=90,num_metgrid_levels=nmet,num_metgrid_soil_levels=nsoil,p_top_requested=10000)
text=rewrite((PREV/'real/namelist.input').read_text(),domains);(real/'namelist.input').write_text(text)
run('real',real/'real.exe',real,'SUCCESS COMPLETE REAL_EM INIT')
wrf=ROOT/'wrf';wrf.mkdir()
for p in (PREV/'wrf').iterdir():
    if p.name in ['qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat']:shutil.copyfile(p,wrf/p.name)
    elif p.is_symlink() and p.is_file() and p.name not in ['wrfinput_d01','wrfbdy_d01']:link(p,wrf/p.name)
for name in ['wrfinput_d01','wrfbdy_d01']:link(real/name,wrf/name)
domains.update(run_hours=12,end_hour=6,start_minute=0,start_second=0,end_minute=0,end_second=0,history_interval=60,frames_per_outfile=13)
(wrf/'namelist.input').write_text(rewrite((PREV/'wrf/namelist.input').read_text(),domains))
(OUT/'explicit_scientific_configuration.json').write_text(json.dumps(dict(wps=namelist,wrf_namelist=(wrf/'namelist.input').read_text(),scope='New independent GFS-driven CAMx36km meteorology; inherited registered physical schemes, explicit36km/dt90/47layers; twelve hours with first six spinup. Not reproduction of official meteorology, not calibrated prediction.'),indent=2))
run('wrf_12hours',wrf/'wrf.exe',wrf,'SUCCESS COMPLETE WRF',timeout=2400)
results[-1]['cfl_exceeded_lines']=(wrf/'wrf_12hours.console.log').read_text(errors='replace').count('points exceeded v_cfl')
(OUT/'stages.json').write_text(json.dumps(results,indent=2))
assert results[-1]['cfl_exceeded_lines']==0,'CFL exceeded: do not use unstable meteorology'
paths=sorted(wrf.glob('wrfout*'));assert len(paths)==1,paths
shutil.copyfile(paths[0],OUT/'actual_wrfout.nc');shutil.copyfile(wps/'geo_em.d01.nc',OUT/'actual_geo_em.nc')
(OUT/'wrf_result.json').write_text(json.dumps(dict(root=str(ROOT),source_wrf=str(paths[0]),export=str(OUT/'actual_wrfout.nc'),stages=results,scope='GFS independent raw meteorology only; chemical fields remain official prepared inputs'),indent=2))

