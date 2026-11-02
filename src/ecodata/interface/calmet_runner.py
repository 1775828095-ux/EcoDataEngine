"""Native execution of the registered four-hour CALMET/CALPUFF case."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import uuid
import numpy as np
import pandas as pd
from ecodata.interface.calmet_observations import decode_surface,decode_upper,decode_precipitation
from ecodata.interface.calpuff_results import read_concentrations
from ecodata.models.adapters.calmet_surface import export_surface
from ecodata.models.adapters.calmet_upper_air import export_upper_air
from ecodata.models.adapters.calmet_precipitation import export_precipitation

BASE=Path('D:/EcoDataEngine')
CASE=BASE/'cases/CALPUFF_20261021'
ENGINES=BASE/'work/calpuff_official'
RUNS=BASE/'outputs/calmet_jobs'
PINS={
 'CALMET_v5.8.5_verified/calmet.exe':'083a426539d18414ada7b7022eed060a03ececa82aee4d231892776f707a335c',
 'CALPUFF_v5.8.4/CALPUFF/calpuff.exe':'cd8f0445b1bd6b2b3a216bcc470bd097694b240b0ac50dc43757cfbdd69786b7',
 'MAKEGEO_v2.29/MAKEGEO/makegeo.exe':'89c07be5857659b3e3f49aa1de7e3ed2c3eaad5a083cc57978fe45182ba6f1c9'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def current_input(payload,mode):
    """Exact registered geometry/weather only; retain the actual uploaded bytes."""
    from ecodata.interface.registered_conversion import member
    if mode is None:
        f,c,_=decode_surface(CASE/'public_case_search/SMERGE_v5.57.zip')
        expected,_=export_surface(f,c);filename='SURF.DAT';target='surf.dat';reference=expected[filename]
    elif mode=='upper_air':
        raw=member(payload,'UP1.DAT')
        for station,target in [('PWM','uppwm.dat'),('CHH','upchh.dat')]:
            f,c,_=decode_upper(BASE/'downloads/READ62_v5.54.zip',station)
            expected,_=export_upper_air(f,c)
            if raw==expected['UP1.DAT']:return target,raw,dict(mode=mode,station=station,sha256=hashlib.sha256(raw).hexdigest())
        raise ValueError('高空输入不属于已核验PWM/CHH观测；不替换示例')
    elif mode=='precipitation':
        f,c,_=decode_precipitation(CASE/'public_case_search/PXTRACT_v4.25.zip')
        expected,_=export_precipitation(f,c);filename='PRECIP.DAT';target='precip.dat';reference=expected[filename]
    else:
        registered={
          'geography':('GEO.DAT','geo1km.dat',BASE/'cases/CALPUFF_20261025/platform_geo_2996325b/CALMET/geo1km.dat'),
          'constant_points':('CALPUFF.INP','CPUF.INP',BASE/'cases/CALPUFF_20261025/platform_points_221f6900/CPUF.INP'),
          'receptors':('CALPUFF.INP','CPUF.INP',BASE/'cases/CALPUFF_20261025/platform_real_receptors/CALPUFF.INP')}
        if mode not in registered:raise ValueError('未登记此输入类型')
        filename,target,path=registered[mode];reference=path.read_bytes()
    raw=member(payload,filename)
    if raw!=reference:raise ValueError('本次输入与已核验登记案例不一致；任意工程的兼容验证尚未实现，不会改用示例')
    return target,raw,dict(mode=mode,sha256=hashlib.sha256(raw).hexdigest(),scope='Current converted bytes; exact registered case only')


def execute(route,folder=None,converted_zip=None,mode=None):
    if route not in ('existing','rebuild'):raise ValueError('Unsupported route')
    folder=Path(folder) if folder else RUNS/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    result=dict(route=route,status='running',folder=str(folder),stages=[],scope='Registered 1990 four-hour case; observed weather, demonstration emissions')
    def save(): (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def native(key,control,work,marker):
        exe=ENGINES/key
        if digest(exe)!=PINS[key]:raise ValueError('Trusted executable hash mismatch: '+key)
        proc=subprocess.run([str(exe),control],cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        console=proc.stdout.decode('latin1');(work/'console.log').write_text(console,encoding='utf-8')
        result['stages'].append(dict(program=key,returncode=proc.returncode,sha256=digest(exe)))
        if proc.returncode or marker not in console:raise RuntimeError('Native stage failed: '+key)
        save()
    try:
        current=current_input(converted_zip,mode) if converted_zip is not None else None
        save();met=folder/'CALMET';met.mkdir()
        template=CASE/'complete_official_case/native_reference_585_explicit_mm4'
        for name in ['cmet.inp','geo1km.dat','upalbr.dat','upchh.dat','uppwm.dat','4007.dat','precip.dat','mm4.dat']:
            shutil.copy2(template/name,met/name)
        archive=CASE/'public_case_search'
        frame,config,sources=decode_surface(archive/'SMERGE_v5.57.zip')
        frame.to_csv(folder/'surface.csv',index=False)
        (folder/'surface_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        outputs,report=export_surface(frame,config);(met/'surf.dat').write_bytes(outputs['SURF.DAT'])
        result['surface']=dict(rows=len(frame),sources=sources,conversion=report)
        result['retained_inputs']=['MM4 prognostic meteorology','marine station 4007','official edited Albany upper-air']
        if route=='rebuild':
            for station,target in [('PWM','uppwm.dat'),('CHH','upchh.dat')]:
                f,c,p=decode_upper(BASE/'downloads/READ62_v5.54.zip',station)
                out,r=export_upper_air(f,c);(met/target).write_bytes(out['UP1.DAT'])
                f.to_csv(folder/(station+'.csv'),index=False);result[station]=dict(source=p,conversion=r)
            f,c,p=decode_precipitation(archive/'PXTRACT_v4.25.zip')
            out,r=export_precipitation(f,c);(met/'precip.dat').write_bytes(out['PRECIP.DAT'])
            result['precipitation']=dict(source=p,conversion=r)
            geo=folder/'MAKEGEO';geo.mkdir();src=ENGINES/'MAKEGEO_v2.29/MAKEGEO'
            for name in ['makegeo.inp','TERR1KM.DAT','LULC1KM.DAT']:shutil.copy2(src/name,geo/name)
            native('MAKEGEO_v2.29/MAKEGEO/makegeo.exe','makegeo.inp',geo,'')
            if digest(geo/'geo1km.dat')!=digest(template/'geo1km.dat'):raise ValueError('Geographic reconstruction differs from registered grid')
            shutil.copy2(geo/'geo1km.dat',met/'geo1km.dat')
            result['geography']='Rebuilt from pregridded terrain/land use, not raw DEM; native configured LU55 fill in 88 undercovered cells'
        else:result['retained_inputs']+=['geography','PWM/CHH upper-air','precipitation']
        if current and mode not in ('constant_points','receptors'):
            target,raw,report=current;(met/target).write_bytes(raw);result['current_conversion']=report
        native('CALMET_v5.8.5_verified/calmet.exe','cmet.inp',met,'ENTERING TERMINATION PHASE')
        if (met/'cmet.dat').stat().st_size<1000000:raise ValueError('Incomplete CALMET output')
        puff=folder/'CALPUFF';puff.mkdir();demo=ENGINES/'CALPUFF_v5.8.4/CALPUFF/DEMO'
        for name in ['CPUF.INP','OZONE.DAT']:shutil.copy2(demo/name,puff/name)
        if current and mode in ('constant_points','receptors'):
            target,raw,report=current;(puff/target).write_bytes(raw);result['current_conversion']=report
        shutil.copy2(met/'cmet.dat',puff/'cmet.dat')
        native('CALPUFF_v5.8.4/CALPUFF/calpuff.exe','CPUF.INP',puff,'')
        listing=(puff/'CPUF.LST').read_text(encoding='latin1')
        if 'LAST DAY/HOUR PROCESSED:' not in listing or 'End of run -- Clock time:' not in listing:raise ValueError('CALPUFF did not terminate normally')
        data=read_concentrations((puff/'CPUF.CON').read_bytes())
        if len(data['times'])!=4:raise ValueError('Expected four concentration hours')
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        values=data['values_g_m3']['SO2']*1e6
        fig,axes=plt.subplots(2,2,figsize=(10,9),layout='constrained')
        rows=[]
        for i,ax in enumerate(axes.flat):
            mesh=ax.pcolormesh(data['x_km'],data['y_km'],values[i],shading='nearest',vmin=0,vmax=float(values.max()),cmap='magma')
            ax.set(title=str(data['times'][i]),xlabel='UTM 19N easting (km), NAS-C',ylabel='Northing (km)');ax.set_aspect('equal')
            for j,y in enumerate(data['y_km']):
                for k,x in enumerate(data['x_km']):rows.append(dict(time=str(data['times'][i]),x_km=x,y_km=y,SO2_ug_m3=float(values[i,j,k])))
        fig.colorbar(mesh,ax=list(axes.flat),label='SO2 hourly concentration (ug/m3)')
        fig.suptitle('Observed weather / demonstration emissions | '+route)
        fig.savefig(folder/'SO2_four_hours.png',dpi=150);plt.close(fig)
        pd.DataFrame(rows).to_csv(folder/'SO2_grid.csv',index=False)
        if data['discrete_values_g_m3']:
            discrete=data['discrete_values_g_m3']['SO2']*1e6
            coords=data['discrete_coordinates_km_m'];discrete_rows=[]
            for i,stamp in enumerate(data['times']):
                for j in range(discrete.shape[1]):discrete_rows.append(dict(time=stamp,receptor_order=j+1,x_km=float(coords[0,j]),y_km=float(coords[1,j]),elevation_m=float(coords[2,j]),SO2_ug_m3=float(discrete[i,j])))
            pd.DataFrame(discrete_rows).to_csv(folder/'SO2_discrete.csv',index=False)
            fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
            for j in range(discrete.shape[1]):ax.plot(range(1,5),discrete[:,j],marker='o',label='Receptor '+str(j+1))
            ax.set(xlabel='Hourly period (05-08 local)',ylabel='SO2 (ug/m3)',title='Ground-level scenario at real EPA site positions');ax.legend();fig.savefig(folder/'SO2_discrete.png',dpi=140);plt.close(fig)
            result['discrete_receptors']=dict(count=discrete.shape[1],finite=bool(np.isfinite(discrete).all()),max_SO2_ug_m3=float(discrete.max()),scope='Actual native discrete output; ground-level scenario, not observed concentrations')
        result.update(status='success',times=[str(t) for t in data['times']],species=data['species'],SO2_max_ug_m3=float(values.max()),concentration_sha256=digest(puff/'CPUF.CON'))
        result['inputs']={str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.inp','.dat')}
        save();return result
    except Exception as exc:
        result.update(status='failed',error=str(exc));save();raise
