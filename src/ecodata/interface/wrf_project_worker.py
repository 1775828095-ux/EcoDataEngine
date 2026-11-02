"""Fixed native WRF worker for a bounded explicit domain/configuration."""
from pathlib import Path
import sys,json,re,shutil,subprocess,struct,time,hashlib,types
from datetime import datetime,timedelta
import numpy as np
from netCDF4 import Dataset
JOB=Path(sys.argv[1]);PHASE=sys.argv[2];CFG=json.loads((JOB/'settings.json').read_text())
ROOT=Path(__file__).resolve().parents[2]
for name,path in [('ecodata',ROOT/'ecodata'),('ecodata.core',ROOT/'ecodata/core')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
from ecodata.interface.wrf_assembly import read_fields,assemble_pressure_update
from ecodata.interface.wrf_source_crop import crop as stream_crop
BASE=Path('/opt/ecodata-wrf/cases/colorado20210313_45415d616ff0469faedc988eded94e90')
PINS={'wps/geogrid.exe':'aca54a60b5cd6c2c83899d5744b747979315908981f5fef58d0aced11774b744',
      'wps/ungrib.exe':'7fc3831efcc8f38c9e896aeed5c83382164445aeab0a4d20a301ab0fb0276ec7',
      'wps/metgrid.exe':'182c19b621c431f0434b8efd5f212713986f90a2a2829b01956ba0efe722bc87',
      'real/real.exe':'f5b5f98f7782c24b83fb76596e53b1351e3082a76e46cc9930b069a9cc8a2e37',
      'wrf/wrf.exe':'d0da833bee327679e695c2b6e31cc1f183a5f700297c75a5e85c4df8e07ea138'}
for name,expected in PINS.items():
    if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=expected:raise ValueError('Installed native engine hash changed: '+name)
(JOB/'engine_pins.json').write_text(json.dumps(PINS,indent=2))
# Source templates are fixed reviewed local platform assets, not uploaded scripts.
WORKSPACE=Path('/mnt/c/Users/ASUS/Documents/ChatGPT/测试重')
TEMPLATE=WORKSPACE/'完整融合案例_20261028/evidence/WRF_Colorado'
STAGES=json.loads((JOB/'stages.json').read_text()) if (JOB/'stages.json').exists() else []

def run(stage,exe,cwd,marker):
    begin=time.time()
    with (JOB/(stage+'.log')).open('wb') as log:p=subprocess.run([str(exe)],cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    text=(JOB/(stage+'.log')).read_text(errors='replace')+'\n'+'\n'.join(p.read_text(errors='replace') for p in cwd.glob('rsl.error.*'))
    rec=dict(stage=stage,returncode=p.returncode,seconds=time.time()-begin,marker=marker,marker_seen=marker in text)
    STAGES.append(rec);(JOB/'stages.json').write_text(json.dumps(STAGES,indent=2));print(rec,flush=True)
    if p.returncode or marker not in text:raise RuntimeError(stage+' failed')

def link(src,dst):
    assert src.is_file(),src;dst.symlink_to(src.resolve())

def set_value(text,key,value,quoted=False):
    replacement="'"+str(value)+"'" if quoted else str(value)
    text,count=re.subn(r'(?m)^(\s*'+key+r'\s*=).*$',lambda m:m[1]+' '+replacement+',',text)
    if count!=1:raise ValueError('Template setting not unique: '+key)
    return text

def crop(source,target):
    return stream_crop(source,target,CFG['latitude'],CFG['longitude'])


if PHASE=='external_prepare':
    for name in ('geogrid.exe','ungrib.exe','metgrid.exe','GEOGRID.TBL','METGRID.TBL'):link(BASE/'wps'/name,JOB/'wps'/name)
    STAGES.append(dict(stage='external_project_copy',returncode=0,marker_seen=True,scientific_settings_changed=False))
    (JOB/'stages.json').write_text(json.dumps(STAGES,indent=2))
elif PHASE=='reuse_prepare':
    record=json.loads((JOB/'reuse_project.json').read_text())
    source=Path(record['source'])
    for name,expected in record['pins'].items():
        src=source/name
        if hashlib.sha256(src.read_bytes()).hexdigest()!=expected:raise ValueError('Existing project changed: '+name)
        dest=JOB/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    for name in ('geogrid.exe','ungrib.exe','metgrid.exe','GEOGRID.TBL','METGRID.TBL'):link(BASE/'wps'/name,JOB/'wps'/name)
    STAGES.append(dict(stage='reuse_geo_and_complete_wps',returncode=0,marker_seen=True,source=str(source),sha256=record['pins']))
    (JOB/'stages.json').write_text(json.dumps(STAGES,indent=2))
elif PHASE=='prepare':
    wps=JOB/'wps';wps.mkdir()
    for name in ('geogrid.exe','ungrib.exe','metgrid.exe','GEOGRID.TBL','METGRID.TBL'):link(BASE/'wps'/name,wps/name)
    # WPS's own Vtable.GFS documents PRMSL before 2011-05-10 for GFS,
    # and before 2015-01-14 for GDAS/FNL. Keep an explicit mapping audit.
    original=(BASE/'wps/Vtable').read_text()
    legacy=datetime.strptime(CFG['start'],'%Y-%m-%d_%H:%M:%S') < datetime(2011,5,10,12)
    selected=original
    if legacy:
        old=' 130 | 102  |   0  |      | PMSL     | Pa      | Sea-level Pressure                      |  0  |  3  | 192 | 101 |'
        new='   2 | 102  |   0  |      | PMSL     | Pa      | Sea-level Pressure                      |  0  |  3  |  1  | 101 |'
        if original.count(old)!=1:raise ValueError('Reviewed PMSL mapping not found exactly once')
        selected=original.replace(old,new)
    (wps/'Vtable').write_text(selected)
    (JOB/'vtable_policy.json').write_text(json.dumps(dict(legacy_PRMSL=legacy,source=str((BASE/'wps/Vtable').resolve()),reason='WPS Vtable.GFS documented pre-MSLET compatibility; original meteorological values, no synthetic PMSL',sha256=hashlib.sha256(selected.encode()).hexdigest()),indent=2))
    start=CFG['start'];end=CFG['end']
    text=(TEMPLATE/'namelist.wps').read_text()
    text=text.replace('2021-03-13_12:00:00',start).replace('2021-03-13_15:00:00',end)
    for old,new in [('interval_seconds=10800','interval_seconds='+str(CFG['interval_seconds'])),('e_we=49','e_we='+str(CFG['nx'])),('e_sn=49','e_sn='+str(CFG['ny'])),('dx=12000','dx='+str(CFG['dx'])),('dy=12000','dy='+str(CFG['dx'])),('ref_lat=39.5','ref_lat='+str(CFG['latitude'])),('ref_lon=-105.2','ref_lon='+str(CFG['longitude'])),('stand_lon=-105.2','stand_lon='+str(CFG['longitude']))]:
        assert old in text;text=text.replace(old,new)
    (wps/'namelist.wps').write_text(text)
    for i,src in enumerate(sorted(JOB.glob('raw_*.grib'))):link(src,wps/('GRIBFILE.AA'+chr(65+i)))
    run('geogrid',wps/'geogrid.exe',wps,'Successful completion')
    run('ungrib',wps/'ungrib.exe',wps,'Successful completion')
    cropped=JOB/'original_wps';cropped.mkdir()
    for src in sorted(wps.glob('FILE:*')):crop(src,cropped/(src.name[5:]+'.wps'))
    groups=[read_fields(p.read_bytes()) for p in sorted(cropped.glob('*.wps'))]
    times=[g[0].time for g in groups]
    expected=[];t=datetime.strptime(start,'%Y-%m-%d_%H:%M:%S');last=datetime.strptime(end,'%Y-%m-%d_%H:%M:%S')
    while t<=last:expected.append(t.strftime('%Y-%m-%d_%H:%M:%S'));t+=timedelta(seconds=CFG['interval_seconds'])
    assert times==expected,('Raw GRIB time coverage mismatch',times,expected)
    f=groups[0][0];lat,lon,dy,dx,radius=f.grid
    levels=sorted({v.level for v in groups[0] if v.key[1]=='TT' and 0<v.level<200000})
    with Dataset(JOB/'raw_pressure.nc','w',format='NETCDF3_CLASSIC') as d:
        for name,n in [('time',len(times)),('pressure',len(levels)),('latitude',f.ny),('longitude',f.nx)]:d.createDimension(name,n)
        for name,values,units in [('time',[(datetime.strptime(t,'%Y-%m-%d_%H:%M:%S')-datetime(1970,1,1)).total_seconds() for t in times],'seconds since 1970-01-01'),('pressure',levels,'Pa'),('latitude',lat+np.arange(f.ny)*dy,'degrees_north'),('longitude',lon+np.arange(f.nx)*dx,'degrees_east')]:
            v=d.createVariable(name,'f8',(name,));v.units=units;v[:]=values
        for label,unit in [('TT','K'),('UU','m s-1'),('VV','m s-1'),('GHT','m')]:
            stacks=[]
            for group in groups:
                data={v.level:v for v in group if v.key[1]==label and 0<v.level<200000};assert set(data)==set(levels)
                assert all(v.wind==0 for v in data.values())
                stacks.append(np.stack([np.frombuffer(data[l].parts[4],dtype='>f4').reshape(f.ny,f.nx) for l in levels]))
            v=d.createVariable(label,'f4',('time','pressure','latitude','longitude'));v.units=unit;v[:]=np.stack(stacks)
        d.source='Original public GRIB -> native ungrib -> exact cell subset; no fusion, no generated meteorological values'
    config={'mode':'pressure_fields','version':'WPS-4.6-intermediate-v5-fields','time_basis':'UTC','crs':'EPSG:4326','coordinates':{k:k for k in ('time','pressure','latitude','longitude')},'coordinate_units':{'pressure':'Pa','latitude':'degree','longitude':'degree'},'fields':{k:{'variable':k} for k in ('TT','UU','VV','GHT')},'required_fields':['TT','UU','VV','GHT'],'wind_reference':'earth_relative','earth_radius_km':radius,'interval_seconds':CFG['interval_seconds'],'prefix':'NEWPROJECT'}
    (JOB/'config.json').write_text(json.dumps(config,indent=2))
elif PHASE=='run':
    wps=JOB/'wps';updates={}
    for p in (JOB/'converted').glob('*.wps'):
        stamp=read_fields(p.read_bytes())[0].time[:13]
        assert stamp not in updates;updates[stamp]=p
    sourcefiles=sorted((JOB/'original_wps').glob('*.wps'));assert len(updates)==len(sourcefiles)
    assembly=[]
    for p in sourcefiles:
        raw,report=assemble_pressure_update(p.read_bytes(),updates[p.stem].read_bytes())
        (wps/(CFG.get('metgrid_prefix','PLATFORM')+':'+p.stem)).write_bytes(raw);assembly.append(report)
    (JOB/'assembly.json').write_text(json.dumps(assembly,indent=2))
    run('metgrid',wps/'metgrid.exe',wps,'Successful completion')
    with Dataset(next(wps.glob('met_em.d01.*.nc'))) as d:
        metlevels=len(d.dimensions['num_metgrid_levels']);soillevels=len(d.dimensions['num_st_layers'])
    if CFG.get('external') and (metlevels!=CFG['num_metgrid_levels'] or soillevels!=CFG['num_metgrid_soil_levels']):
        raise ValueError('Metgrid vertical counts differ from uploaded namelists; not rewriting user parameters')
    def namelist(kind):
        text=(TEMPLATE/('namelist.input.'+kind)).read_text()
        start=datetime.strptime(CFG['start'],'%Y-%m-%d_%H:%M:%S');end=datetime.strptime(CFG['end'],'%Y-%m-%d_%H:%M:%S') if kind=='real' else start+timedelta(seconds=CFG['duration_seconds'])
        values={'run_days':0,'run_hours':0,'run_minutes':0,'run_seconds':int((end-start).total_seconds()),'interval_seconds':CFG['interval_seconds'],'e_we':CFG['nx'],'e_sn':CFG['ny'],'e_vert':CFG['levels'],'dx':CFG['dx'],'dy':CFG['dx'],'time_step':CFG['time_step'],'p_top_requested':CFG['top_pressure'],'num_metgrid_levels':metlevels,'num_metgrid_soil_levels':soillevels,'history_interval':5,'frames_per_outfile':100}
        for prefix,stamp in [('start',start),('end',end)]:
            for key in ('year','month','day','hour','minute','second'):values[prefix+'_'+key]=getattr(stamp,key)
        for key,value in values.items():text=set_value(text,key,value)
        return text
    runtime_assets={}
    for kind,marker in [('real','SUCCESS COMPLETE REAL_EM INIT'),('wrf','SUCCESS COMPLETE WRF')]:
        dest=JOB/kind;dest.mkdir()
        for p in (BASE/kind).iterdir():
            if p.is_symlink() and p.is_file() and not p.name.startswith(('met_em.','wrfinput','wrfbdy')):link(p,dest/p.name)
            elif p.name in ('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'):shutil.copyfile(p,dest/p.name)
        if CFG.get('external'):(dest/'namelist.input').write_bytes((JOB/('supplied_namelist.'+kind)).read_bytes())
        else:(dest/'namelist.input').write_text(namelist(kind))
        if kind=='real':
            for p in wps.glob('met_em.d01.*.nc'):link(p,dest/p.name)
        else:
            for name in ('wrfinput_d01','wrfbdy_d01'):link(JOB/'real'/name,dest/name)
        if CFG.get('external'):
            runtime_assets[kind]={p.name:dict(source=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.iterdir() if p.is_file() and not p.name.startswith(('met_em.','wrfinput','wrfbdy'))}
            (JOB/'controlled_runtime_assets.json').write_text(json.dumps(runtime_assets,indent=2))
        run(kind,dest/(kind+'.exe'),dest,marker)
    output=next((JOB/'wrf').glob('wrfout*'))
    with Dataset(output) as d:
        stamps=[b''.join(v).decode() for v in d['Times'][:]]
        expected=(datetime.strptime(CFG['start'],'%Y-%m-%d_%H:%M:%S')+timedelta(seconds=CFG['duration_seconds'])).strftime('%Y-%m-%d_%H:%M:%S');assert stamps[-1]==expected
        value=np.asarray(d['T2'][-1])-273.15;assert np.isfinite(value).all()
        np.savez(JOB/'plot_data.npz',value=value,x=np.asarray(d['XLONG'][-1]),y=np.asarray(d['XLAT'][-1]))
    result=dict(success=True,plot_title=('External uploaded WRF project | ' if CFG.get('external') else 'Configurable original GRIB project | ')+expected,plot_units='2m temperature (degC)',x_label='Longitude',y_label='Latitude',time=stamps,shape=list(value.shape),stages=STAGES,full_native_chain=True,accuracy_evaluated=False)
    (JOB/'native_result.json').write_text(json.dumps(result,indent=2))
else:raise ValueError('Unknown phase')
