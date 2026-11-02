"""Run offline A->B comparisons with independent readers and predeclared thresholds.

From source root: python -m examples.ai_round1.run_round1
Optional --official-tools PATH points to top-level evidence/official to run
SMERGE and a CALMET attempt. No downloads and no model-run claims on absence.
"""
import argparse,io,json,sys,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from scipy.interpolate import RegularGridInterpolator
from examples.ai_round1.independent import surf,swat,wps,photolysis
from ecodata.round1 import convert

ROOT=Path(__file__).resolve().parent
THRESHOLDS=json.loads((ROOT/'thresholds.json').read_text(encoding='utf-8'))

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def metrics(actual,expected,atol):
    a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    assert a.shape==b.shape
    error=a-b
    return {'count':a.size,'max_absolute_error':float(np.abs(error).max()),'MAE':float(np.abs(error).mean()),
            'RMSE':float(np.sqrt((error**2).mean())),'bias':float(error.mean()),'threshold_atol':atol,
            'relative_error':'not used: values near zero occur','passed':bool(np.isfinite(error).all() and np.abs(error).max()<=atol)}

def export(model,path,config,out):
    files,report=convert(model,path.read_bytes(),json.loads(config.read_text(encoding='utf-8')))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():(out/name).write_bytes(data)
    return report

def calpuff():
    case=ROOT/'calpuff';r={}
    for label in ['reference','noaa','synthetic']:
        source={'reference':'A_reference_decoded.csv','noaa':'A_noaa_surface.csv','synthetic':'A_synthetic_complete.csv'}[label]
        report=export('calpuff',case/source,case/f'config_{label}.json',case/f'B_{label}')
        parsed=surf(case/f'B_{label}/SURF.DAT')
        r[label]={'format_parsed':True,'records':parsed['values'].shape[0]*parsed['values'].shape[1],
                  'missing_counts':report['missing_counts'],'time_label':'hour_end','complete_model_run':False}
        if label=='reference':
            expected=surf(case/'B2_official_smerge_SURF.DAT')
            mask=expected['values']==9999
            r[label].update(time_station_exact=parsed['times']==expected['times'] and parsed['stations']==expected['stations'],
                            missing_mask_exact=bool(np.array_equal(parsed['values']==9999,mask)),
                            value_metrics=metrics(parsed['values'][~mask],expected['values'][~mask],THRESHOLDS['calmet_value_atol']),
                            B2='Official unchanged SMERGE5.57 SURF.DAT; same decoded values/configuration, validates serialization/unit inversion only')
        elif label=='noaa':
            frame=pd.read_csv(case/source)
            # Independent NOAA quantities, applying temperature offset by hand.
            expected=frame.temperature.to_numpy()+273.15;actual=parsed['values'][:,0,4]
            valid=np.isfinite(expected)
            r[label]['temperature_conversion']=metrics(actual[valid],expected[valid],THRESHOLDS['calmet_value_atol'])
            r[label]['B2']=None
        else:
            r[label]['temperature_K']=metrics(parsed['values'][:,:,4],np.full(parsed['values'].shape[:2],293.15),THRESHOLDS['calmet_value_atol'])
    save(case/'comparison.json',r);return r

def swat_case():
    case=ROOT/'swat';report=export('swat',case/'A_daily_weather.csv',case/'config.json',case/'B')
    precipitation=swat(case/'B/weather.pcp','pcp');temperature=swat(case/'B/weather.tmp','tmp')
    # Derive expected values directly from raw NOAA source, not converted A or exporter.
    raw=pd.read_csv(case/'raw/noaa_gsod_2020.csv')
    p=raw.PRCP.to_numpy()*25.4;good=(raw.PRCP!=99.99)&raw.PRCP_ATTRIBUTES.isin(['D','F','G'])
    t=np.column_stack([(raw.MAX.to_numpy()-32)*5/9,(raw.MIN.to_numpy()-32)*5/9]);valid=np.column_stack([raw.MAX!=9999.9,raw.MIN!=9999.9])
    exported_p=precipitation['values'][:,0];exported_t=temperature['values']
    assert precipitation['dates']==temperature['dates']==pd.to_datetime(raw.DATE).to_list()
    mass_error=float(exported_p[good].sum()-p[good].sum());bound=.05*int(good.sum())+1e-9
    r={'format_parsed':True,'records':len(raw),'dates_exact':True,'leap_day_present':any(t.month==2 and t.day==29 for t in precipitation['dates']),
       'precipitation_mm':metrics(exported_p[good],p[good],THRESHOLDS['swat_atol_mm_or_degC']),
       'temperature_degC':metrics(exported_t[valid],t[valid],THRESHOLDS['swat_atol_mm_or_degC']),
       'missing_p_mask_exact':bool(np.array_equal(exported_p==-99,~good)),
       'precipitation_sum_error_mm':mass_error,'quantization_bound_mm':bound,'sum_within_bound':abs(mass_error)<=bound,
       'missing_policy':report['missing_policy'],'B2':None,'C2':None,'complete_model_run':False,
       'boundary':'Validates numeric conversion and fixed-format dates; source daily reporting window and complete watershed run remain unverified'}
    save(case/'comparison.json',r);return r

def wrf_case():
    case=ROOT/'wrf';report=export('wrf',case/'A_sst_subset.nc',case/'config.json',case/'B')
    exported=next((case/'B').glob('*.wps'));parsed=wps(exported)
    # netCDF4 independent backend reads source in memory: Windows C library cannot open non-ASCII filenames.
    with netCDF4.Dataset('independent_raw',memory=(case/'raw/oisst_20200101.nc').read_bytes()) as ds:
        lat=ds.variables['lat'][:];lon=ds.variables['lon'][:]
        ly=np.where((lat>=30)&(lat<=31))[0];lx=np.where((lon>=300)&(lon<=301))[0]
        expected=np.asarray(ds.variables['sst'][0,0,ly,lx],dtype=float)+273.15
        expected_grid=[float(lat[ly[0]]),float(lon[lx[0]]),.25,.25,6367.47]
    r={'format_parsed_by':'scipy.io.FortranFile (independent Fortran records)',
       'sst_K':metrics(parsed['values'],expected,THRESHOLDS['wps_sst_atol_K']),
       'grid_coordinates':metrics(parsed['projection'][:4],expected_grid[:4],THRESHOLDS['wps_grid_atol_degree']),
       'earth_radius_km':metrics(parsed['projection'][4:],expected_grid[4:],THRESHOLDS['wps_earth_radius_atol_km']),
       'field_unit_level_exact':parsed['field']=='SST' and parsed['unit']=='K' and parsed['level']==200100.,
       'UTC_time':parsed['date'],'B2':None,'C2':None,'complete_model_run':False,
       'boundary':'One SST supplemental slab; no pressure levels, winds, soil, geography, metgrid/real/WRF run'}
    # Coordinate-only tolerance was set independently; earth radius has its own float32 unit km.
    r['coordinate_max_error_degree']=float(np.abs(parsed['projection'][:4]-expected_grid[:4]).max())
    r['coordinates_passed']=r['coordinate_max_error_degree']<=THRESHOLDS['wps_grid_atol_degree']
    save(case/'comparison.json',r);return r

def camx_case():
    from ecodata.io.structures import read_structure,export_structure
    case=ROOT/'camx';raw=photolysis(case/'raw/tuv.do_CB6r5.20160610')
    ds=read_structure('A_photolysis_table.csv',(case/'A_photolysis_table.csv').read_bytes())
    ds.attrs.update(model_id='camx',reference_version='7.40',output_scope='standardized photolysis table only',native_model_input='false')
    ds.photolysis_native_value.attrs.update(units='native numeric convention retained; no unit conversion',physical_variable='photolysis_table')
    data,name,_=export_structure(ds,'NetCDF');out=case/'B';out.mkdir(exist_ok=True);(out/'prepared_photolysis.nc').write_bytes(data)
    with netCDF4.Dataset('read_cf',memory=data) as actual:
        value=actual.variables['photolysis_native_value'][:]
        context={n:np.array(actual.variables[n][:]) for n in ('block_id','ozone_column_native','albedo','terrain_native','height_km','zenith_index','reaction')}
    stride=raw['values'].shape[1]*raw['values'].shape[2]
    conditions=np.array(raw['conditions'])
    expected={'block_id':np.repeat(np.arange(len(raw['heights'])),stride),
        'ozone_column_native':np.repeat(conditions[:,0],stride),'albedo':np.repeat(conditions[:,1],stride),'terrain_native':np.repeat(conditions[:,2],stride),
        'height_km':np.repeat(raw['heights'],stride),'zenith_index':np.tile(np.arange(raw['values'].shape[2]),len(raw['heights'])*len(raw['reactions']))}
    context_checks={n:metrics(context[n],v,THRESHOLDS['camx_standardized_values_atol']) for n,v in expected.items()}
    reaction_expected=np.tile(np.repeat(raw['reactions'],raw['values'].shape[2]),len(raw['heights']))
    r={'format_parsed_by':'independent CAMx ASCII test reader and netCDF4',
       'native_source_shape':list(raw['values'].shape),'record_count':len(value),
       'preserved_numbers':metrics(value,raw['values'].ravel(),THRESHOLDS['camx_standardized_values_atol']),
       'context_preservation':context_checks,'reaction_labels_exact':bool(np.array_equal(context['reaction'].astype(str),reaction_expected)),
       'distinct_conditions':len(set(map(tuple,raw['conditions']))),'unique_heights':len(set(raw['heights'])),
       'B2':None,'C2':'official outputs URL exists but full archive not downloaded','complete_model_run':False,
       'boundary':'A is official native photolysis input; B is standardized CF table, not a CAMx native input; no upstream radiation reconstruction'}
    save(case/'comparison.json',r);return r

def cmaq_case():
    from ecodata.models import RequirementStore
    from ecodata.models.scenario import recognise_inputs,suggest_roles,execute_scenario
    from ecodata.models.adapters.ioapi_reader import CmaqIOAPIReader
    case=ROOT/'cmaq'
    inputs=recognise_inputs([(p.name,p.read_bytes()) for p in (case/'A_existing_case').iterdir()],{'pm25_station.csv':{'temporal_semantics':'mean'}})
    profile=RequirementStore().get('cmaq');roles=suggest_roles(profile,inputs)
    result=execute_scenario(profile,inputs,roles);out=case/'B';out.mkdir(exist_ok=True)
    for name,data in result.files.items():(out/name).write_bytes(data)
    with netCDF4.Dataset('actual_cf',memory=result.files['model_input.nc']) as nc:
        t=np.array(nc.variables['temperature'][:]);wind=np.array(nc.variables['wind'][:]);rh=np.array(nc.variables['humidity'][:]);pm=np.array(nc.variables['PM25'][:])
        lat=np.array(nc.variables['lat'][:]);lon=np.array(nc.variables['lon'][:]);time=netCDF4.num2date(nc.variables['time'][:],nc.variables['time'].units,only_use_cftime_datetimes=False)
        meanbounds=np.array(nc.variables['time_bnds'][:]);a=np.array(nc.variables['PM25_raw_fused'][:]);b=np.array(nc.variables['PM25_background'][:])
    with xr.open_dataset(io.BytesIO((case/'A_existing_case/geos_cf_meteorology.nc').read_bytes()),engine='h5netcdf') as met:
        iy=np.argsort(met.lat.values);ix=np.argsort(met.lon.values)
        centers=pd.to_datetime(time);indices=[list(pd.to_datetime(met.time.values)).index(v) for v in centers]
        points=np.array([[y,x] for y in lat for x in lon]);temperature=[]
        for i in indices:
            field=met.temperature.values[i][iy][:,ix]
            independent=RegularGridInterpolator((met.lat.values[iy],met.lon.values[ix]),field)(points).reshape(len(lat),len(lon))-273.15
            temperature.append(independent)
    r={'format_parsed_by':'netCDF4 for auxiliary CF; PseudoNetCDF for distinct old ICON reference',
       'temperature_K_to_degC_and_bilinear':metrics(t,temperature,THRESHOLDS['cmaq_affine_unit_atol']),
       'negative_clipping_identity':metrics(pm,np.maximum(a,0),0),'clipped_cells':int((a<0).sum()),
       'required_field_completeness':1.,'time_alignment_retained':1.,'meteorology_original_steps':len(met.time),'retained_steps':len(time),
       'B2':None,'C2':None,'complete_model_run':False,'native_model_input':False,
       'boundary':'NASA GEOS-CF/AirNow/USGS auxiliary case; observations constrain fusion so cannot be treated as independent C2. Old CMAQ5.0.2 ICON profile is format-only reference for different version/mechanism'}
    try:
        reader=CmaqIOAPIReader()
        if not reader.python.is_file():reader=CmaqIOAPIReader(sys.executable)
        reference=reader.inspect(case/'reference_format_only/ic_profile_CB05_CMAQ5.0.2.dat',format='icon_profile')
        save(case/'reference_format_only/pnc_read_report.json',reference)
        r['old_ICON_parsed']=True
        candidate=reader.inspect(out/'model_input.nc',format='ioapi')
        save(case/'candidate_ioapi_inspection.json',candidate)
        r['candidate_missing_ioapi_attributes']=candidate['missing_ioapi_attributes']
        r['candidate_TFLAG_present']=candidate['tflag_present']
    except Exception as exc:r.update(old_ICON_parsed=False,old_ICON_blocker=str(exc))
    save(case/'comparison.json',r);return r

def tools(official):
    case=ROOT/'calpuff';work=case/'C_preprocessor';work.mkdir(exist_ok=True)
    candidate=case/'B_reference/SURF.DAT';shutil.copy2(candidate,work/'candidate.dat')
    # SMERGE accepts previous formatted SURF.DAT with zero additional station files.
    control='''! NFF=0 !
! PREVDAT=candidate.dat !
! SURFDAT=accepted.dat !
! RUNLST=reader.lst !
! LCFILES=T !
!END!
! IBYR=1990 ! ! IBMO=1 ! ! IBDY=8 ! ! IBHR=1 !
! IEYR=1990 ! ! IEMO=1 ! ! IEDY=15 ! ! IEHR=0 !
! XBTZ=5 ! ! LPREV=T ! ! NBSTN=0 ! ! INFORM=2 !
! IOFORM=2 ! ! IOPACK=0 ! ! JDAT=2 ! ! IHUSWO=1 !
! LPCALC=F ! ! LVSRNOUT=F ! ! JTXT=1 !
!END!
'''
    (work/'reader.inp').write_text(control,encoding='ascii');exe=official/'smerge557/SMERGE/smerge.exe'
    r={'full_CALPUFF_run':False,'full_CALMET_run':False}
    try:
        completed=subprocess.run([str(exe.resolve()),'reader.inp'],cwd=work,input='',capture_output=True,text=True,timeout=45,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (work/'stdout.log').write_text(completed.stdout+'\nSTDERR\n'+completed.stderr,encoding='utf-8')
        r.update(SMERGE_returncode=completed.returncode,SMERGE_output_exists=(work/'accepted.dat').is_file())
        if (work/'accepted.dat').is_file():
            output=surf(work/'accepted.dat');expected=surf(candidate)
            r['SMERGE_read_and_processed']=True;r['values_after_official_processor']=metrics(output['values'],expected['values'],THRESHOLDS['smerge_rewrite_atol'])
            r['scope']='Real official SMERGE re-read/rewrite of B; does not establish full CALMET/CALPUFF simulation'
    except Exception as exc:r['SMERGE_blocker']=str(exc)
    calmet=official/'calmet585/CALMET_v5.8.5/CODE'
    attempt=case/'CALMET_attempt';attempt.mkdir(exist_ok=True)
    shutil.copy2(calmet/'CMET.INP',attempt/'CALMET.INP');shutil.copy2(candidate,attempt/'SURF.DAT')
    try:
        completed=subprocess.run([str((calmet/'calmet.exe').resolve())],cwd=attempt,input='',capture_output=True,text=True,timeout=45,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (attempt/'stdout.log').write_text(completed.stdout+'\nSTDERR\n'+completed.stderr,encoding='utf-8')
        r['CALMET_attempt_returncode']=completed.returncode
        r['CALMET_attempt_scope']='Official sample control references missing GEO1KM.DAT/upper-air files; no complete compatible case provided. Success requires outputs/log inspection, not returncode.'
    except Exception as exc:r['CALMET_blocker']=str(exc)
    save(case/'official_execution.json',r);return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--official-tools',type=Path);args=parser.parse_args()
    report={'thresholds':THRESHOLDS,'cases':{},'errors':{}}
    for name,function in [('calpuff',calpuff),('swat',swat_case),('wrf',wrf_case),('camx',camx_case),('cmaq',cmaq_case)]:
        try:
            report['cases'][name]=function();print(name,'converted and independently compared',flush=True)
        except Exception as exc:
            import traceback
            report['errors'][name]=traceback.format_exc();print(name,'FAILED',str(exc),flush=True)
    if args.official_tools:
        try:report['official_execution']=tools(args.official_tools)
        except Exception:
            import traceback
            report['errors']['official_execution']=traceback.format_exc()
    save(ROOT/'round1_results.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str),flush=True)
    return int(bool(report['errors']))

if __name__=='__main__':sys.exit(main())
