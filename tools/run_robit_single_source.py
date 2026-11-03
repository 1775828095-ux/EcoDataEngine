"""Frozen single-station Robit recovery study through the public platform entry."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.models import portrait_policy as policy
from ecodata.algorithms.composition_decision import Composition
from ecodata.models.adapters.swat_plus import SWATPlusAdapter

OUT=ROOT/'output/robit_single_source_20261026'
VAR='precipitation';METHODS=('linear','nearest','ffill','month_mean');SEEDS=(11,22,33)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')


def stores():
    ar.ROOT=OUT/'attributes';policy.ROOT=OUT/'models'


def plans():
    def step(method,cutoff=None):return dict(algorithm='reconstruction.daily_'+method,
        parameters={} if cutoff is None else dict(max_gap_days=cutoff))
    return [[step(m)] for m in METHODS]+[
        [step(a,c),step(b)] for a in METHODS for b in METHODS if a!=b for c in (1,3,7)]


def raw_weather():
    p=OUT/'source';times=pd.date_range('1990-01-01','2013-12-31')
    frame=pd.DataFrame(dict(time=times))
    for name,file in [(VAR,'robpcp.txt'),('solar_radiation','robsol.txt'),
                      ('relative_humidity','robrhum.txt'),('wind_speed','robwind.txt')]:
        assert (p/file).read_text().splitlines()[0].strip()=='19900101'
        frame[name]=pd.read_csv(p/file,skiprows=1,header=None)[0]
    temp=pd.read_csv(p/'robtmp.txt',skiprows=1,header=None)
    frame['air_temperature_max']=temp[0];frame['air_temperature_min']=temp[1]
    assert len(frame)==8766 and np.isfinite(frame.select_dtypes('number')).all().all()
    assert (frame.air_temperature_max>=frame.air_temperature_min).all()
    return frame


def dataset(frame):
    times=pd.DatetimeIndex(frame.time)
    ds=xr.Dataset({VAR:(("time","lat","lon"),frame[VAR].to_numpy(copy=True)[:,None,None]),
        'time_bounds':(('time','bounds'),np.stack([times.values,(times+pd.Timedelta(days=1)).values],axis=1))},
        coords=dict(time=times,lat=[11.6],lon=[37.36]),attrs=dict(crs='EPSG:4326',source_id='Robit_station',
        data_role='single_station_series',not_spatial_downscaling='true'))
    ds[VAR].attrs.update(units='mm/day',data_kind='continuous',physical_variable='precipitation_amount',
        physical_definition='Station daily accumulated precipitation over stated one-day bounds',
        spatial_semantics='point',temporal_semantics='accumulation',valid_min=0.)
    return ds


def mask_case(frame,first,kind,seed):
    part=frame[frame.time.between(str(first)+'-01-01',str(first+1)+'-12-31')].copy()
    ds=dataset(part);rng=np.random.default_rng(seed);n=len(part);mask=np.zeros(n,bool)
    length=dict(random=1,short=3,long=14)[kind]
    for start in rng.permutation(np.arange(1,n-length-1)):
        if mask[start-1:start+length+1].any():continue
        mask[start:start+length]=True
        if mask.sum()>=int(n*.1):break
    truth=ds[VAR].copy(deep=True).where(xr.DataArray(mask,dims='time',coords={'time':ds.time}))
    ds[VAR].values[mask]=np.nan
    return ds,truth,mask


def base_request(ds,branch_plans):
    return dict(task='temporal',variable=VAR,unit='mm/day',semantic_type='continuous',nonnegative=True,
        spatial_semantics='point',temporal_semantics='accumulation',target_lat=[11.6],target_lon=[37.36],
        target_times=[str(pd.Timestamp(t)) for t in ds.time.values],parameters={},
        model_profile=dict(id='swat_plus',role='weather',field=VAR),
        model_requirements=dict(hard=['nonnegative'],evidence='Daily precipitation cannot be negative'),
        scoring=dict(registry_version='robit-daily-state-v1'),
        portrait_policy=dict(version='robit-swatplus-input-v1',scenario='daily-recovery'),
        composition=dict(source_stages={'Robit_station':dict(plans=branch_plans,
            evidence='Frozen four real kernels, all unequal short/long assignments and 1/3/7-day cutoffs; no winner encoded')},
            post_stages=[],search=dict(survivors=2,population=2,generations=40,exact_check=False,backend='alns')))


def inputs_for(ds):
    return [api.SearchInput('masked_station.nc','primary',ds.to_netcdf()),
        api.SearchInput('prior_monthly_state.nc','state',(OUT/'training_state.nc').read_bytes())]


def prepare():
    stores();weather=raw_weather();train=weather[weather.time<'1996-01-01']
    state=xr.Dataset(dict(month_mean=('month',train.groupby(train.time.dt.month)[VAR].mean().to_numpy()),
        scale=float(train[VAR].std(ddof=0))),coords=dict(month=np.arange(1,13)),
        attrs=dict(training_start='1990-01-01',training_end='1995-12-31',split='training',
                   fit='Monthly arithmetic means and population standard deviation; training only'))
    state.to_netcdf(OUT/'training_state.nc',engine='h5netcdf')
    attrs=deepcopy(ar.load()['attributes'])
    for metric in ('missing_fraction','negative_fraction','expected_error'):
        attrs.append(dict(id='robit_'+metric,definition='Daily visible state '+metric,
            tables=['data','algorithm','model'],unit='dimensionless',role='scoring',scoring_enabled=True,
            hard_allowed=False,provider='state_metric',property=metric,activation='constant_one',
            evidence='expected_error is normalized visible-probe absolute mean bias, NOT SWAT output error',history=['Robit engineering study']))
    api.publish_attribute_registry('robit-daily-state-v1',attrs,evidence='Explicit daily recovery measurements; no new learned ranker')
    issues=[dict(attribute='robit_'+m,metric=m,acceptable=0.,bad=1.,importance=1.,
        evidence='Complete finite nonnegative daily input; normalized probe volume bias is a preparation proxy, not SWAT sensitivity')
        for m in ('missing_fraction','negative_fraction','expected_error')]
    api.publish_model_portrait('robit-swatplus-input-v1','swat_plus',[
        dict(id='daily-recovery',task='temporal',role='weather',field=VAR,issues=issues,coupled_constraints=[],
        evidence='SWAT+ daily precipitation preparation contract only; native model response remains unknown')],
        evidence='Reviewed swat_plus weather profile and explicit experimental input-quality priorities')
    protocol=dict(task='Single-source daily precipitation recovery; no independent-source fusion or spatial claim',
        fitting='1990-1995: monthly means and scale',development='1996-2001: freeze strongest fixed method/graph',
        validation='2002-2007: search only',test='2008-2013: after graph selection is frozen',
        cases=[dict(kind=k,development_start=1996+2*i,validation_start=2002+2*i,test_start=2008+2*i)
               for i,k in enumerate(('random','short','long'))],mask_seed=26026,masked_fraction_target=.1,
        wet_day_threshold_mm=1.,search_seeds=SEEDS,validation_execution_budget=8,
        visible_probe_graphs=40,random_total_graph_cost_control_budget=40,
        candidate_plans=plans(),candidate_count=len(plans()),
        scoring=dict(weights=dict(data=.5,model=.5,collaboration=0),
            evidence='Frozen equal data/input-contract priority, no empirically established SWAT sensitivity; collaboration bonus disabled'),
        cost='Visible-mask probes and actual branch-kernel calls reported in addition to target validation executions',
        target='SWAT+62 weather-only writer; no native model run, no outlet-flow assessment',
        uncertainty='Three paired time/scenario cases; seeds are optimizer repeats, not independent basins')
    save(OUT/'protocol.json',protocol);print('FROZEN',len(plans()),flush=True)


def service(ds,reference,request,folder,seed=11,limit=8):
    reference=reference.copy(deep=True);reference.attrs.update(split='search_validation',units='mm/day')
    return api.submit_search(inputs_for(ds),request,None,folder,mode='Profile',
        budget=dict(objective_limit=limit,wall_seconds=180,proposal_limit=400),seed=seed,
        validation=api.SearchValidation(reference,VAR))


def metrics(ds,truth,mask):
    p=ds[VAR].values[:,0,0][mask];y=truth.values[:,0,0][mask];e=p-y
    return dict(RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(abs(e).mean()),Bias=float(e.mean()),
        wet_day_error_rate=float(np.mean((p>=1)!=(y>=1))),n=len(y),sse=float(sum(e*e)))


def develop():
    stores();p=json.loads((OUT/'protocol.json').read_text());weather=raw_weather();rows=[]
    for i,case in enumerate(p['cases']):
        ds,truth,mask=mask_case(weather,case['development_start'],case['kind'],p['mask_seed'])
        req=base_request(ds,plans());req['composition']['search'].update(exact_check=True,backend='random',initialization='uniform_control')
        r=service(ds,truth,req,OUT/'development'/str(i),limit=40)
        if r['counts']['exact_usable_count']!=40:raise ValueError('Development failures: '+str([(x.get('error'),x['index']) for x in r['rows'] if x['status']!='PASS']))
        rows.extend(dict(case=i,index=x['index'],plan=x['plan'],**x['search_validation']) for x in r['rows'])
        print('DEVELOPMENT',i,r['final_validation_rmse'],flush=True)
    table=pd.DataFrame(rows);group=table.groupby('index').apply(lambda a:np.sqrt(np.average(a.RMSE**2,weights=a.denominator)),include_groups=False)
    engine=Composition([('primary',ds),('state',xr.open_dataset(OUT/'training_state.nc').load())],req,api.platform_registry())
    single=[idx for idx in group.index if len(engine.plan(engine.decode(int(idx)))[0]['choice'])==1]
    best=int(group.idxmin());best_single=int(group.loc[single].idxmin())
    save(OUT/'fixed_selection.json',dict(best_graph_index=best,best_single_index=best_single,
        best_graph_plan=engine.plan(engine.decode(best)),best_single_plan=engine.plan(engine.decode(best_single)),
        selection_partition='development only',scores={str(k):float(v) for k,v in group.items()}))


def run_searches():
    stores();p=json.loads((OUT/'protocol.json').read_text());weather=raw_weather();rows=[]
    for i,case in enumerate(p['cases']):
        ds,truth,mask=mask_case(weather,case['validation_start'],case['kind'],p['mask_seed'])
        for method in ('portrait_only','portrait_alns','uniform_alns','random','random_cost'):
            for seed in ((11,) if method=='portrait_only' else SEEDS):
                req=base_request(ds,plans());opts=req['composition']['search'];limit=8
                if method.startswith('portrait'):
                    req['composition']['parallel_scoring']=p['scoring']
                else:opts.update(initialization='uniform_control',backend='alns' if method=='uniform_alns' else 'random')
                if method=='portrait_only':opts.update(backend='basic',survivors=1,generations=0);limit=1
                if method=='random_cost':limit=40
                folder=OUT/'search'/f'{i}_{method}_{seed}'
                r=service(ds,truth,req,folder,seed,limit)
                if not r['best']:raise ValueError('No usable selection '+str(folder))
                rows.append(dict(case=i,kind=case['kind'],method=method,seed=seed,folder=str(folder),
                    index=r['best']['index'],plan=r['best']['plan'],key=r['recommended_key'],validation_RMSE=r['final_validation_rmse'],
                    target_graph_executions=r['counts']['evaluated'],probe_graph_executions=r.get('parallel_scoring',{}).get('probe_graph_executions',0),
                    seconds=r['total_service_seconds'],failed=r['counts']['failed_observed']))
                print('SEARCH',i,method,seed,round(r['final_validation_rmse'],5),r['counts']['evaluated'],flush=True)
    save(OUT/'frozen_validation_selections.json',rows)


def test_and_export():
    stores();protocol=json.loads((OUT/'protocol.json').read_text());weather=raw_weather()
    selections=json.loads((OUT/'frozen_validation_selections.json').read_text());fixed=json.loads((OUT/'fixed_selection.json').read_text())
    fixed_rows=[dict(method='fixed_'+m,seed=0,plan=[dict(branch='Robit_station',choice=[dict(algorithm='reconstruction.daily_'+m,parameters={})])]) for m in METHODS]
    fixed_rows.append(dict(method='development_best_fixed_combo',seed=0,plan=fixed['best_graph_plan']))
    results=[]
    for i,case in enumerate(protocol['cases']):
        ds,truth,mask=mask_case(weather,case['test_start'],case['kind'],protocol['mask_seed'])
        seen={}
        for selection in fixed_rows+[x for x in selections if x['case']==i]:
            chain=next(x['choice'] for x in selection['plan'] if x['branch']=='Robit_station')
            tag=hashlib.sha256(json.dumps(chain,sort_keys=True).encode()).hexdigest()[:16]
            if tag not in seen:
                folder=OUT/'test_execution_v2'/f'{i}_{tag}';req=base_request(ds,[chain])
                req['composition']['search'].update(backend='basic',generations=0,survivors=1)
                r=api.submit_search(inputs_for(ds),req,None,folder,mode='Profile',budget=dict(objective_limit=1,wall_seconds=60))
                if not r['best']:raise ValueError('Frozen test graph failed')
                product=xr.open_dataset(r['best']['product']).load()
                assert np.array_equal(product[VAR].values[:,0,0][~mask],ds[VAR].values[:,0,0][~mask])
                result=metrics(product,truth,mask)
                part=weather[weather.time.isin(pd.to_datetime(product.time.values))].copy();part[VAR]=product[VAR].values[:,0,0]
                config=dict(project_name='Robit_reconstruction',start=str(part.time.iloc[0].date()),end=str(part.time.iloc[-1].date()),
                    station=dict(name='Robit',latitude=11.6,longitude=37.36,elevation=1790),
                    weather=part.to_dict('list'),target_model_version='62.0.0',
                    additional_weather=['solar_radiation','relative_humidity','wind_speed'])
                export=SWATPlusAdapter().save_package(folder/'swatplus_weather',config)
                saved=pd.read_csv(export/'station.pcp',sep=r'\s+',skiprows=3,header=None)
                assert len(saved)==len(product.time)
                np.testing.assert_allclose(saved[2],part[VAR],atol=.00051,rtol=0)
                for column,filename,raw_file in [('solar_radiation','station.slr','robsol.txt'),('relative_humidity','station.hmd','robrhum.txt'),('wind_speed','station.wnd','robwind.txt')]:
                    actual=pd.read_csv(export/filename,sep=r'\s+',skiprows=3,header=None)[2]
                    np.testing.assert_allclose(actual,part[column],atol=.00051,rtol=0)
                actual=pd.read_csv(export/'station.tem',sep=r'\s+',skiprows=3,header=None)
                np.testing.assert_allclose(actual[[2,3]],part[['air_temperature_max','air_temperature_min']],atol=.00051,rtol=0)
                report=json.loads((export/'validation_report.json').read_text())
                assert report['native_model_input_verified'] is False and report['runnable_without_template'] is False
                handoff=folder/'weather_handoff';handoff.mkdir()
                for file in export.iterdir():
                    if file.name.startswith('station.') or file.name in {'pcp.cli','tmp.cli','slr.cli','hmd.cli','wnd.cli'}:
                        (handoff/file.name).write_bytes(file.read_bytes())
                (handoff/'README.md').write_text('仅五类逐日气象及其索引。不要覆盖队友工程file.cio、time.sim、拓扑或WGN。由队友保留工程WGN并映射站点。原生SWAT+运行未验收。',encoding='utf-8')
                seen[tag]=dict(metrics=result,product=r['best']['product'],weather_output=str(export),graph=chain,
                    input_observation_protection='PASS',weather_readback='PASS',native_SWAT_run=False)
            results.append(dict(case=i,kind=case['kind'],method=selection['method'],seed=selection['seed'],**seen[tag]))
    save(OUT/'test_results.json',results)
    rows=[]
    for method in sorted({x['method'] for x in results}):
        group=[x for x in results if x['method']==method];n=sum(x['metrics']['n'] for x in group)
        rows.append(dict(method=method,RMSE=float(np.sqrt(sum(x['metrics']['sse'] for x in group)/n)),
            MAE=sum(x['metrics']['MAE']*x['metrics']['n'] for x in group)/n,
            Bias=sum(x['metrics']['Bias']*x['metrics']['n'] for x in group)/n,
            cases=len(group),unique_time_cases=3))
    save(OUT/'test_summary.json',rows);print('TEST SUMMARY',json.dumps(rows),flush=True)


if __name__=='__main__':
    command=sys.argv[1]
    {'prepare':prepare,'develop':develop,'search':run_searches,'test':test_and_export}[command]()
