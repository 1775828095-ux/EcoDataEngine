"""Pinned local case execution in WSL. Receives only managed directory paths."""
from pathlib import Path
import hashlib
import json
import os
import re
import resource
import shutil
import struct
import subprocess
import sys
import time
import numpy as np
from netCDF4 import Dataset

JOB = Path(sys.argv[1]).resolve()
SOURCE = Path(sys.argv[2]).resolve()
STATE = json.loads((JOB / 'status.json').read_text())
STAGES = []


def pin(path, sha=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha and digest != sha:
        raise ValueError('Source hash mismatch: '+str(path))
    return dict(path=str(path), sha256=digest)


def run(name, command, cwd, marker=None, timeout=1800):
    begin = time.time(); logfile = JOB / (name+'.log')
    with logfile.open('wb') as log:
        result = subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, env=dict(os.environ, OMP_NUM_THREADS='1'))
    text = logfile.read_text(errors='replace')
    text += '\n'.join(p.read_text(errors='replace') for p in Path(cwd).glob('rsl.error.*'))
    record = dict(stage=name, returncode=result.returncode, seconds=time.time()-begin,
                  marker=marker, marker_seen=marker is None or marker in text)
    STAGES.append(record)
    (JOB/'stages.json').write_text(json.dumps(STAGES, indent=2))
    print(json.dumps(record), flush=True)
    if result.returncode or not record['marker_seen']:
        raise RuntimeError(name+' failed; see '+str(logfile))


def to_linux(path):
    return '/mnt/'+path[0].lower()+'/'+path[3:].replace('\\', '/') if len(path)>2 and path[1]==':' else path


def link(source, dest):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.symlink_to(source)


def crop_wps(source, target):
    def record(f):
        h = f.read(4)
        if not h:
            return None
        n = struct.unpack('>i', h)[0]
        assert 0 < n < 100000000
        value = f.read(n); assert len(value)==n and f.read(4)==h
        return value
    def frame(v):
        n=struct.pack('>i',len(v)); return n+v+n
    with source.open('rb') as f, target.open('wb') as out:
        while True:
            version = record(f)
            if version is None:
                break
            header, projection, wind, slab = [record(f) for _ in range(4)]
            assert version==struct.pack('>i',5) and len(header)==156
            level,nx,ny,iproj=struct.unpack('>fiii',header[140:]); assert iproj==0 and projection[:8]==b'SWCORNER'
            lat,lon,dy,dx,radius=struct.unpack('>5f',projection[8:])
            iy=[j for j in range(ny) if 32<=lat+j*dy<=47]
            ix=[i for i in range(nx) if 243<=((lon+i*dx)%360)<=267]
            assert len(iy)>2 and len(ix)>2 and ix==list(range(ix[0],ix[-1]+1))
            h=bytearray(header); struct.pack_into('>ii',h,144,len(ix),len(iy))
            p=projection[:8]+struct.pack('>5f',lat+iy[0]*dy,lon+ix[0]*dx,dy,dx,radius)
            s=b''.join(slab[(j*nx+ix[0])*4:(j*nx+ix[-1]+1)*4] for j in iy)
            for v in (version,bytes(h),p,wind,s):
                out.write(frame(v))


def wrf():
    # This worker lives inside the package, permitting imports without a shell.
    # Load only the low-dependency binary assembler; the Windows application
    # package initializer imports unrelated pandas/pydantic pipeline modules.
    import types
    root = Path(__file__).resolve().parents[2]
    for name, path in [('ecodata',root/'ecodata'),('ecodata.core',root/'ecodata/core')]:
        package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
    from ecodata.interface.wrf_assembly import assemble_pressure_update,read_fields
    profile=SOURCE/'evidence/WRF_Colorado'
    base=Path(json.loads((profile/'case.json').read_text())['linux_case'])
    wps=JOB/'wps'; wps.mkdir()
    for name in ('geogrid.exe','ungrib.exe','metgrid.exe','GEOGRID.TBL','METGRID.TBL','Vtable'):
        link(base/'wps'/name,wps/name)
    # Keep proven domain and physics. Only private workspace/data paths change.
    shutil.copyfile(profile/'namelist.wps',wps/'namelist.wps')
    inputs=[]
    if STATE['route']=='rebuild':
        for i,p in enumerate(sorted((SOURCE/'data/WRF_Colorado_20210313').glob('*.grib2'))):
            link(p,wps/('GRIBFILE.AA'+chr(65+i))); inputs.append(pin(p))
        run('geogrid',[str(wps/'geogrid.exe')],wps,'Successful completion')
        run('ungrib',[str(wps/'ungrib.exe')],wps,'Successful completion')
        for src in wps.glob('FILE:*'):
            crop_wps(src,wps/(src.name[5:]+'.wps'))
    else:
        link(base/'wps/geo_em.d01.nc',wps/'geo_em.d01.nc')
        for src in (SOURCE/'platform_inputs/WRF_Colorado/cropped').glob('*.wps'):
            shutil.copyfile(src,wps/src.name); inputs.append(pin(src))
    assembly=[]
    for source in sorted(wps.glob('20*.wps')):
        stamp=source.stem
        updates=[p for p in JOB.glob('*.wps') if read_fields(p.read_bytes())[0].time[:13]==stamp]
        assert len(updates)==1,stamp
        merged,report=assemble_pressure_update(source.read_bytes(),updates[0].read_bytes())
        (wps/('PLATFORM:'+stamp)).write_bytes(merged); assembly.append(report)
    assert len(assembly)==2
    (JOB/'assembly.json').write_text(json.dumps(assembly,indent=2))
    run('metgrid',[str(wps/'metgrid.exe')],wps,'Successful completion')
    for name, marker in [('real','SUCCESS COMPLETE REAL_EM INIT'),('wrf','SUCCESS COMPLETE WRF')]:
        dest=JOB/name;dest.mkdir();old=base/name
        for p in old.iterdir():
            if p.is_symlink() and p.is_file() and not p.name.startswith(('met_em.','wrfinput','wrfbdy')):
                link(p,dest/p.name)
            elif p.name in ('qr_acr_qg_V4.dat','qr_acr_qsV2.dat','freezeH2O.dat'):
                shutil.copyfile(p,dest/p.name)
        shutil.copyfile(profile/('namelist.input.'+name),dest/'namelist.input')
        if name=='real':
            for p in wps.glob('met_em.d01.*.nc'):
                link(p,dest/p.name)
        else:
            for p in ('wrfinput_d01','wrfbdy_d01'):
                link(JOB/'real'/p,dest/p)
        run(name,[str(dest/(name+'.exe'))],dest,marker)
    output=next((JOB/'wrf').glob('wrfout*'))
    with Dataset(output) as data:
        a=np.asarray(data['T2'][-1])-273.15
        x=np.asarray(data['XLONG'][-1]);y=np.asarray(data['XLAT'][-1])
        assert np.isfinite(a).all()
        timestamps=[b''.join(row).decode() for row in data['Times'][:]]
        assert timestamps[-1]=='2021-03-13_15:00:00'
    np.savez(JOB/'plot_data.npz',value=a,x=x,y=y)
    return dict(success=True,plot_title='WRF Colorado | 2021-03-13 15 UTC | 3h integration',plot_units='2m temperature (degC)',x_label='Longitude',y_label='Latitude',shape=list(a.shape),time=timestamps,inputs=inputs)


def calpuff():
    engine=json.loads((SOURCE/'evidence/version_sources/build_calpuff721.json').read_text())
    exe=Path(engine['exe']);pin(exe,engine['sha256'])
    met=Path(to_linux(STATE['new_calmet']))
    for name in ('calmet_test.dat','calmet_test.dat.aux'):
        shutil.copyfile(met/name,JOB/name.upper())
    controls=list(JOB.glob('*.INP'));assert len(controls)==1
    control=controls[0].read_text(encoding='latin1')
    control=control.replace('..\\calmet\\calmet_test.dat','CALMET_TEST.DAT')
    (JOB/'CALPUFF_RUN.INP').write_text(control,encoding='latin1')
    run('calpuff',[str(exe),'CALPUFF_RUN.INP'],JOB,timeout=180)
    data=(JOB/'CALPUFF_PT1.CON').read_bytes();records=[];i=0
    while i<len(data):
        n=struct.unpack_from('<i',data,i)[0];assert n>0 and struct.unpack_from('<i',data,i+4+n)[0]==n
        records.append(data[i+4:i+4+n]);i+=n+8
    assert i==len(data) and records[0][:32].strip().startswith(b'CONC.DAT        7.0')
    comments=struct.unpack('<i',records[1])[0]
    assert records[comments+5]==b'NOX           1' and records[comments+6].strip()==b'g/m3'
    assert len(np.frombuffer(records[comments+7],dtype='<f4'))==30
    tail=records[-24:];values=[];hours=[]
    for k in range(0,24,8):
        stamp=struct.unpack('<8i',tail[k]);assert stamp[:2]==(2016,1) and stamp[6]-stamp[2]==1
        assert tail[k]==tail[k+4] and tail[k+7][:15]==b'NOX           1'
        a=np.frombuffer(tail[k+7][15:],dtype='<f4').astype('f8');assert len(a)==6 and np.isfinite(a).all() and (a>=0).all()
        hours.append(stamp[6]);values.append(a*1e6)
    a=np.stack(values);np.savez(JOB/'plot_data.npz',value=a,time=hours)
    return dict(success=True,plot_title='Cambridge | CALPUFF7.2.1 public fork | 2016-01-01',records=int(a.size),minimum=float(a.min()),maximum=float(a.max()),engine=engine,meteorology_recomputed=STATE['route']=='rebuild')


def camx():
    prior=json.loads((SOURCE/'evidence/CAMx_202011/latest600_run.json').read_text())
    engine=prior['engine'];exe=Path(engine['exe']);pin(exe,engine['sha256'])
    input_pins={key:pin(value['path'],value['sha256']) for key,value in prior['input_pins'].items()}
    (JOB/'outputs').mkdir()
    control=(Path(prior['folder'])/'CAMx.in').read_text()
    native=bytearray((SOURCE/'data/CAMx_202011/emisfile/area.d3.camx.2020307.bin').read_bytes())
    # Read this task's converted NO and overwrite only matching original NO slabs.
    converted=list(JOB.glob('emissions*.nc'));assert len(converted)==1
    with Dataset(converted[0]) as d: values=np.asarray(d['NO'][:,0])
    offset=0;recs=[]
    while offset<len(native):
        n=struct.unpack_from('>i',native,offset)[0];assert n>0 and struct.unpack_from('>i',native,offset+4+n)[0]==n
        recs.append((offset+4,n));offset+=n+8
    species=[native[recs[3][0]+i:recs[3][0]+i+40:4].decode().strip() for i in range(0,recs[3][1],40)]
    assert len(species)==31 and species[0]=='NO'
    t=0
    for j in range(4,len(recs),32):
        start,n=recs[j+1];assert n==44+157*145*4
        current=values[t];assert current.shape==(145,157)
        assert np.array_equal(np.frombuffer(native[start+44:start+n],dtype='>f4').reshape(145,157),current)
        native[start+44:start+n]=np.asarray(current,dtype='>f4').tobytes();t+=1
    assert t==len(values)
    inventory=JOB/'complete_inventory.bin';inventory.write_bytes(native)
    control=control.replace(str(Path(prior['folder'])/'outputs/CAMx.20201102'),str(JOB/'outputs/CAMx.20201102'))
    control=control.replace(prior['input_pins']['Emiss_Grid(1)']['path'],str(inventory))
    if STATE['route']=='rebuild':
        # Reconstruct a namelist from its explicit registered parameter list.
        parameters=[]
        for line in control.splitlines():
            if '=' in line:
                key,value=line.split('=',1);parameters.append((key.strip(),value.strip().rstrip(',')))
        assert len(parameters)==len({k for k,v in parameters})
        control=' &CAMx_Control\n'+''.join(f' {k} = {v},\n' for k,v in parameters)+' /\n'
    (JOB/'CAMx.in').write_text(control)
    resource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    run('camx',[str(exe)],JOB,'END SIMULATION',timeout=1200)
    diag=next((JOB/'outputs').glob('*.diag')).read_text(errors='replace')
    from camx_inventory_audit import audit_inventory
    coverage=audit_inventory(species,diag)
    (JOB/'inventory_audit.json').write_text(json.dumps(coverage,indent=2))
    mapped={kind:re.findall(rf'{kind} source species\s+\d+\s+(\S+)\s+mapped to model species\s+\d+\s+(\S+)',diag) for kind in ('Area','Point')}
    assert all(len(v)==30 and all((s,s) in v for s in ('TOLA','XYLA','ISP','TRP')) for v in mapped.values())
    p=next((JOB/'outputs').glob('*.avrg.grd01'));rec=[]
    with p.open('rb') as f:
        while h:=f.read(4):
            n=struct.unpack('>i',h)[0];v=f.read(n);assert f.read(4)==h;rec.append(v)
    assert len(rec)==11
    fields={}
    for v in rec[5:]:
        name=v[4:44:4].decode().strip();a=np.frombuffer(v[44:],dtype='>f4').astype('f8').reshape(145,157)
        assert np.isfinite(a).all() and (a>=0).all();fields[name]=a
    a=fields['O3'][1:-1,1:-1]*1000
    np.savez(JOB/'plot_data.npz',value=a,x=np.arange(1,156)*3+1.5,y=np.arange(1,144)*3+1.5)
    return dict(success=True,plot_title='CAMx6.00 CB05 | 2020-11-02 12-13 UTC hourly mean',plot_units='O3 (ppbv)',x_label='Grid-relative x (km)',y_label='Grid-relative y (km)',mapped_species_per_inventory=coverage['Area']['mapped_count'],inventory_species=len(species),unmapped=coverage['Area']['not_modeled'],full_inventory_adaptation=coverage['complete'],inventory_coverage=coverage,input_pins=input_pins)


def cmaq():
    prior=json.loads((SOURCE/'evidence/CMAQ_LISTOS/latest_run.json').read_text())
    engine=json.loads((SOURCE/'evidence/CMAQ_LISTOS/build.json').read_text())
    pin(engine['exe'],engine['exe_sha256'])
    previous=Path(prior['folder']);text=(previous/'run.csh').read_text()
    for name in ('work','output'):(JOB/name).mkdir()
    text=text.replace(str(previous),str(JOB))
    inventory=JOB/'complete_inventory.nc'
    shutil.copyfile(SOURCE/'platform_inputs/CMAQ_LISTOS/complete_inventory_with_platform_NO.nc',inventory)
    converted=list(JOB.glob('emissions*.nc'));assert len(converted)==1
    with Dataset(converted[0]) as src,Dataset(inventory,'r+') as dst:
        a=np.asarray(src['NO'][:]);old=np.asarray(dst['NO'][:]);assert np.array_equal(a,old)
        dst['NO'][:]=a
    text=text.replace(str(SOURCE/'platform_inputs/CMAQ_LISTOS/complete_inventory_with_platform_NO.nc'),str(inventory))
    if STATE.get('chemical_preprocessing'):
        for key,tool in [('INIT_CONC_1','icon'),('BNDY_CONC_1','bcon')]:
            newfile=JOB/'chemical_preprocessing'/(tool.upper()+'.nc')
            details=STATE['chemical_preprocessing'][tool]
            pin(newfile,details['sha256'])
            text,count=re.subn(r'^\s*setenv '+key+r' .*$', '  setenv '+key+' '+str(newfile),text,flags=re.M)
            assert count==1,'Chemical file reference must be unique: '+key
    for state_key,subdir,number in [('new_smoke','raw_emissions','003'),('new_smoke_non_egu','raw_non_egu','001'),('new_smoke_cmv','raw_cmv','004')]:
      if STATE.get(state_key):
        for key,label in [('STK_GRPS_'+number,'stack_groups'),('STK_EMIS_'+number,'inline')]:
            detail=STATE[state_key]['products'][label]
            newfile=JOB/subdir/('elevpoint/stack_groups.ncf' if label=='stack_groups' else 'merge/inline.ncf')
            pin(newfile,detail['sha256'])
            text,count=re.subn(r'^\s*setenv '+key+r' .*$', '  setenv '+key+' '+str(newfile),text,flags=re.M)
            assert count==1,'Point source reference must be unique: '+key
        text,count=re.subn(r'^\s*setenv STK_EM_SYM_DATE_'+number+r' .*$', '  setenv STK_EM_SYM_DATE_'+number+' F',text,flags=re.M)
        assert count==1
    if STATE.get('new_mcip'):
        for key,name in [('GRID_BDY_2D','GRIDBDY2D'),('GRID_CRO_2D','GRIDCRO2D'),('GRID_DOT_2D','GRIDDOT2D'),('MET_CRO_2D','METCRO2D'),('MET_CRO_3D','METCRO3D'),('MET_DOT_3D','METDOT3D'),('MET_BDY_3D','METBDY3D'),('LUFRAC_CRO','LUFRAC_CRO')]:
            newfile=JOB/'raw_meteorology'/(name+'.nc')
            pin(newfile,STATE['new_mcip']['products'][name]['sha256'])
            text,count=re.subn(r'^\s*setenv '+key+r' .*$', '  setenv '+key+' '+str(newfile),text,flags=re.M)
            assert count==1,'Meteorology reference must be unique: '+key
    # The registered run script already records module and representative-day choices.
    for key,value in prior['explicit_optional_module_choices'].items():
        assert re.search(r'^setenv '+key+' '+value+r'\s*$',text,re.M)
    assert str(previous) not in text
    (JOB/'run.csh').write_text(text)
    run('cctm',['csh',str(JOB/'run.csh')],JOB,'PROGRAM COMPLETED SUCCESSFULLY',timeout=1800)
    for state_key,subdir in [('new_smoke','raw_emissions'),('new_smoke_non_egu','raw_non_egu'),('new_smoke_cmv','raw_cmv')]:
      if STATE.get(state_key):
        logs=(JOB/'cctm.log').read_text(encoding='utf-8',errors='replace')+'\n'+'\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in (JOB/'output/LOGS').glob('*'))
        for name in ('elevpoint/stack_groups.ncf','merge/inline.ncf'):
            assert str(JOB/subdir/name) in logs,'No native read evidence for '+name
    if STATE.get('new_mcip'):
        logs=(JOB/'cctm.log').read_text(errors='replace')+'\n'+'\n'.join(p.read_text(errors='replace') for p in (JOB/'output/LOGS').glob('*'))
        for name in ('METCRO3D','METCRO2D','METBDY3D'):
            assert str(JOB/'raw_meteorology'/(name+'.nc')) in logs,'No native read evidence for '+name
    if STATE.get('chemical_preprocessing'):
        logs=(JOB/'cctm.log').read_text(errors='replace')+'\n'+'\n'.join(p.read_text(errors='replace') for p in (JOB/'output/LOGS').glob('*'))
        for tool in ('ICON','BCON'):
            assert str(JOB/'chemical_preprocessing'/(tool+'.nc')) in logs,'No native read evidence for '+tool
        if STATE.get('accept_reference_defaults'):
            expected={'APNCOMI','APNCOMJ','APOCI','APOCJ','IEPOXP'}
            defaults=set(re.findall(r'No IC found for species\s+(\w+)\s+in INIT_CONC_1; set to\s+1\.00E-30',logs))
            baseline={'HGIIAER','ISOPRXN','PCSOARXN','SESQRXN','SULRXN','TRPRXN'}
            assert defaults-baseline==expected,'Unexpected native defaults; refuse acceptance'
            (JOB/'native_defaults.json').write_text(json.dumps(dict(explicit_reference_scenario=True,additional_native_defaults=sorted(expected),all_native_defaults=sorted(defaults),complete_input_supplied=False),indent=2))
    output=next((JOB/'output').glob('CCTM_CONC*.nc'))
    with Dataset(output) as data:
        flags=np.asarray(data['TFLAG'][:,0]);assert flags[-1].tolist()==[2018217,10000]
        all_finite=all(not np.ma.getmaskarray(v[:]).any() and np.isfinite(v[:]).all() for k,v in data.variables.items() if k!='TFLAG');assert all_finite
        a=np.asarray(data['O3'][-1,0])*1000;assert (a>=0).all()
        x=(data.XORIG+(np.arange(a.shape[1])+.5)*data.XCELL)/1000
        y=(data.YORIG+(np.arange(a.shape[0])+.5)*data.YCELL)/1000
    np.savez(JOB/'plot_data.npz',value=a,x=x,y=y)
    return dict(success=True,plot_title='CMAQ5.5 CB6R3 | LISTOS 2018-08-05 01 UTC',plot_units='Surface O3 (ppbv)',x_label='Lambert x (km)',y_label='Lambert y (km)',time_flags=flags.tolist(),all_species_finite=bool(all_finite),explicit_module_choices=prior['explicit_optional_module_choices'],reused_mcip=not bool(STATE.get('new_mcip')),full_raw_construction=False)


result=globals()[STATE['model']]()
result['stages']=STAGES
(JOB/'native_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
