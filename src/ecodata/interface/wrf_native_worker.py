"""Native acceptance of newly assembled inputs; preserve previous case directories."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
import sys
import argparse
import re

parser=argparse.ArgumentParser()
parser.add_argument('--workspace',required=True)
parser.add_argument('--job-id',required=True)
parser.add_argument('--from-zero',action='store_true')
args=parser.parse_args()
if not re.fullmatch(r'[0-9a-f]{32}',args.job_id):
    raise ValueError('Invalid managed job identifier')
HERE=Path(args.workspace).resolve()
assert HERE.is_dir()
BASE=Path('/opt/ecodata-wrf/cases/fnl20230331')
FROM_ZERO=args.from_zero
CUSTOM_TERRAIN=(HERE/'terrain/index').is_file()
CUSTOM_LANDUSE=(HERE/'landuse/index').is_file()
TAG='from_zero' if FROM_ZERO else 'existing'
TARGET=BASE/('platform_ui_'+args.job_id)
WPS=Path('/opt/ecodata-wrf/WPS-4.7.0')
PREVIOUS=BASE/'user_zip9_20261018'
assert not TARGET.exists(), 'Preserve existing acceptance directory; choose a new run id'
TARGET.mkdir()
results=[]
def link(source,dest):
    assert source.is_file(), source
    dest.symlink_to(source.resolve())
def execute(stage,executable,cwd,marker):
    start=time.time()
    with (cwd/'launcher.log').open('wb') as stream:
        process=subprocess.run([str(executable)],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    logs='\n'.join(p.read_text(errors='replace') for p in cwd.glob('*.log'))
    logs+='\n'+'\n'.join(p.read_text(errors='replace') for p in cwd.glob('rsl.*'))
    success=process.returncode==0 and marker in logs
    results.append(dict(stage=stage,returncode=process.returncode,seconds=round(time.time()-start,2),success=success))
    (HERE/f'native_{TAG}_status.json').write_text(json.dumps(results,indent=2))
    print(results[-1],flush=True)
    assert success, stage

wps=TARGET/'wps';wps.mkdir()
if FROM_ZERO or CUSTOM_TERRAIN or CUSTOM_LANDUSE:
    for source,name in [(WPS/'geogrid.exe','geogrid.exe'),(WPS/'ungrib.exe','ungrib.exe'),(WPS/'geogrid/GEOGRID.TBL.ARW','GEOGRID.TBL'),(WPS/'ungrib/Variable_Tables/Vtable.GFS','Vtable')]:
        link(source,wps/name)
    for custom_field,custom_directory in [('HGT_M','terrain'),('LANDUSEF','landuse')]:
        if not (HERE/custom_directory/'index').is_file():continue
        table=(wps/'GEOGRID.TBL').read_text();blocks=re.split(r'(?m)(^=+\s*$)',table);count=0
        for i,block in enumerate(blocks):
            if re.search(r'name\s*=\s*'+custom_field+r'\b',block):
                block=re.sub(r'(?m)^.*rel_path\s*=.*\n','',block)
                block+='\n abs_path = lowres:'+str(HERE/custom_directory)+'/\n abs_path = default:'+str(HERE/custom_directory)+'/\n'
                blocks[i]=block;count+=1
        assert count==1
        (wps/'GEOGRID.TBL').unlink();(wps/'GEOGRID.TBL').write_text(''.join(blocks))
    (wps/'namelist.wps').write_text((BASE/'reference_wps/namelist.wps').read_text())
    originals=sorted((BASE/'severe_outbreak').glob('*.grib2'))
    assert len(originals)==5
    for i,source in enumerate(originals):link(source,wps/('GRIBFILE.AA'+chr(65+i)))
    execute('geogrid',wps/'geogrid.exe',wps,'Successful completion')
    execute('ungrib',wps/'ungrib.exe',wps,'Successful completion')
    fresh=sorted(wps.glob('FILE:*'))
    assert len(fresh)==5
    for source in fresh:
        # Fresh preprocessing must reproduce the complementary source used in assembly.
        frozen=BASE/'reference_wps'/source.name
        assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(frozen.read_bytes()).digest()
link(WPS/'metgrid.exe',wps/'metgrid.exe')
link(WPS/'metgrid/METGRID.TBL.ARW',wps/'METGRID.TBL')
if not FROM_ZERO and not CUSTOM_TERRAIN and not CUSTOM_LANDUSE:link(BASE/'reference_wps/geo_em.d01.nc',wps/'geo_em.d01.nc')
config=(PREVIOUS/'wps/namelist.wps').read_text()
assert "MERGED" in config
(wps/'namelist.wps').write_text(config)
for source in sorted(HERE.glob('MERGED_*.wps')):
    link(source,wps/('MERGED:'+source.stem[len('MERGED_'):]))
assert len(list(wps.glob('MERGED:*')))==5
execute('metgrid',wps/'metgrid.exe',wps,'Successful completion')
real=TARGET/'real';real.mkdir()
for source in (PREVIOUS/'real').iterdir():
    if source.name=='namelist.input':shutil.copyfile(source,real/source.name)
    elif source.is_symlink() and source.is_file() and not source.name.startswith('met_em.'):
        link(source,real/source.name)
for source in wps.glob('met_em.d01.*.nc'):link(source,real/source.name)
execute('real',real/'real.exe',real,'SUCCESS COMPLETE REAL_EM INIT')
wrf=TARGET/'wrf';wrf.mkdir()
for source in (PREVIOUS/'wrf').iterdir():
    if source.name=='namelist.input':shutil.copyfile(source,wrf/source.name)
    elif source.name in ['qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat']:
        shutil.copyfile(source,wrf/source.name)
    elif source.is_symlink() and source.is_file() and source.name not in ['wrfinput_d01','wrfbdy_d01']:
        link(source,wrf/source.name)
for name in ['wrfinput_d01','wrfbdy_d01']:link(real/name,wrf/name)
config=(wrf/'namelist.input').read_text()
for key,value in {'run_days':0,'run_hours':0,'run_minutes':0,'run_seconds':180,'start_year':2023,'start_month':3,'start_day':31,'start_hour':0,'start_minute':0,'start_second':0,'end_year':2023,'end_month':3,'end_day':31,'end_hour':0,'end_minute':3,'end_second':0,'time_step':180,'max_dom':1}.items():
    matched=re.findall(r'(?m)^\s*'+key+r'\s*=\s*(\d+)\s*,',config)
    assert matched==[str(value)], 'Registered configuration changed: '+key
provenance=[]
for path in [wps/'namelist.wps',wps/'METGRID.TBL',real/'namelist.input',wrf/'namelist.input',wps/'metgrid.exe',real/'real.exe',wrf/'wrf.exe']:
    provenance.append(dict(path=str(path),resolved=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(HERE/'native_provenance.json').write_text(json.dumps(provenance,indent=2))
execute('wrf_180_seconds',wrf/'wrf.exe',wrf,'SUCCESS COMPLETE WRF')
outputs=[]
for source in wrf.glob('wrfout*'):
    dest=HERE/(TAG+'_'+source.name.replace(':','_'));shutil.copyfile(source,dest)
    outputs.append(dict(name=dest.name,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
assert outputs
(HERE/f'native_{TAG}_outputs.json').write_text(json.dumps(dict(stages=results,outputs=outputs,simulation_seconds=180,reused_geometry=not FROM_ZERO and not CUSTOM_TERRAIN,new_geogrid_executed=FROM_ZERO or CUSTOM_TERRAIN),indent=2))
