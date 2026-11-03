"""Controlled access to existing trusted writer/independent reader, no case data."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd

def main(job):
    if job['action']=='surface':
        from ecodata.models.adapters.calmet_surface import export_surface
        frame=pd.read_csv(job['frame']);files,report=export_surface(frame,job['config'])
        Path(job['output']).write_bytes(files['SURF.DAT'])
        print(json.dumps(report));return
    if job['action']!='results':raise ValueError('Unknown worker action')
    from ecodata.interface.calpuff_results import read_concentrations
    source=Path(job['source']);out=Path(job['output'])
    d=read_concentrations(source.read_bytes())
    if len(d['times'])!=job['expected_hours']:raise ValueError('实际浓度输出小时数与用户工程不符')
    if list(d['times'])!=job['expected_times']:raise ValueError('实际浓度输出时间与用户工程不符')
    if list(d['species'])!=job['expected_species']:raise ValueError('实际输出物种与用户工程不符')
    report=dict(hours=len(d['times']),times=d['times'],species=d['species'],species_checks={})
    rows=[]
    for species in d['species']:
        a=d['values_g_m3'][species]*1e6
        if not np.isfinite(a).all() or np.any(a<0):raise ValueError('实际浓度值非有限或为负数：'+species)
        report['species_checks'][species]=dict(finite=True,nonnegative=True,min_ug_m3=float(a.min()),max_ug_m3=float(a.max()),shape=list(a.shape))
        for i,t in enumerate(d['times']):
            for j,y in enumerate(d['y_km']):
                for k,x in enumerate(d['x_km']):rows.append(dict(time=t,species=species,x_km=float(x),y_km=float(y),concentration_ug_m3=float(a[i,j,k])))
    pd.DataFrame(rows).to_csv(out/'concentration_grid.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    species='SO2' if 'SO2' in d['species'] else d['species'][0];a=d['values_g_m3'][species]*1e6
    n=len(d['times']);cols=min(2,n);rowcount=(n+cols-1)//cols
    fig,axes=plt.subplots(rowcount,cols,figsize=(10,4.5*rowcount),squeeze=False,layout='constrained')
    for i,ax in enumerate(axes.flat):
        if i>=n:ax.set_visible(False);continue
        mesh=ax.pcolormesh(d['x_km'],d['y_km'],a[i],shading='nearest',vmin=0,vmax=max(float(a.max()),1e-12),cmap='magma')
        ax.set(title=str(d['times'][i]),xlabel='Native projected easting (km)',ylabel='Northing (km)');ax.set_aspect('equal')
    fig.colorbar(mesh,ax=[ax for ax in axes.flat if ax.get_visible()],label=species+' concentration (ug/m3)')
    fig.suptitle('Fresh native CALPUFF output | observed weather / hypothetical emissions')
    fig.savefig(out/'concentration_map.png',dpi=150);plt.close(fig)
    report['independent_reader']='existing ecodata.interface.calpuff_results.read_concentrations (Fortran records/CONC.DAT 2.1)'
    report['accuracy_assessed']=False
    (out/'numerical_check.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
    print(json.dumps(report,default=str))

if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text(encoding='utf8')))
