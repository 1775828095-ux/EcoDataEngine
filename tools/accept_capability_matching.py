"""Bounded development evidence and two-layer public clients, no matching policy.

Only six existing TRAIN fields; the old validation/test files are not opened.
Region is the split unit. Tiny holdout is descriptive, not a reliability claim.
"""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,xarray as xr
from ecodata.algorithms.platform_search import SearchInput,DatasetReference,EngineeringMetric,submit_search,platform_registry
from ecodata.interface.search_page import process_search_upload
from ecodata.algorithms.capability_scoring import CapabilityScoringService,digest
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.algorithms.raster_reconstruction_task import degradation
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.core.schemas import EcoDataBundle
from ecodata.algorithms.numerical_search import save

OUT=ROOT/'output/capability_matching_20261021'
MODEL='aod_consistency_dev_v1'
BASES=['raster_task.spatial.nearest','raster_task.spatial.bilinear','raster_task.spatial.xarray_cubic']
BUDGET=dict(objective_limit=5,proposal_limit=48,callback_limit=100,wall_seconds=20)
PROTOCOL=dict(version='two_class_development_v1',epsilon=1e-8,loss_unit='1',metric='RMSE',metric_kind='reference_error',
    seed=17,holdout_groups=['Sichuan'],fine_size=24,coarse_sizes=[12,8],bases=BASES,
    independent_unit='region; two dates/scales/implementations are not independent replicates',
    inference='developmental point prediction, NOT confidence interval or certified improvement',
    cost_policy='new actual paired collection at most 6 fields * 2 scales * 3 bases * 2 flows = 72 pipeline calls',
    probability='0.9 normalized old weights * exp(clip(relative_benefit,-1,1)) + 0.1 uniform; only optional stage',
    scope='LGHAP product-reference error, NOT independent environmental observation error')


def collect():
    if (OUT/'pairs.json').exists():return json.loads((OUT/'pairs.json').read_text(encoding='utf-8'))
    save(OUT/'protocol.json',PROTOCOL) # frozen before any new losses
    clock=time.perf_counter();records=[];failures=[];registry=platform_registry();pipe=UnifiedAlgorithmPipeline(registry)
    for path in sorted((ROOT/'output/lghap_reconstruction_20261020_v3/tiles/train').glob('*.nc')):
        with xr.open_dataset(path,engine='h5netcdf') as ds:fine=ds.isel(lat=slice(0,24),lon=slice(0,24)).load()
        fine.attrs['crs']='EPSG:4326';fine.aod.attrs.update(units='1',data_kind='continuous',spatial_semantics='cell_mean')
        region=path.stem.split('_')[0]
        for size in PROTOCOL['coarse_sizes']:
            coarse,operator=degradation(fine,size)
            context=AlgorithmContext((EcoDataBundle(coarse),),'aod','continuous',tuple(fine.lat.values),tuple(fine.lon.values),
                model_constraints=dict(unit='1',output_unit='1',nonnegative=True),parameters=dict(coarse_reference=coarse,degradation_operator=operator))
            features=SemanticFeatureExtractor.raster_features(coarse,'aod',24)
            reference=DatasetReference(fine.aod,'aod')
            for base in BASES:
                identifier=f'{path.stem}_{size}_{base}'
                paired=[]
                for tail in ['raster_task.nonnegative','raster_task.coarse_consistency']:
                    start=time.perf_counter()
                    try:
                        result=pipe.run(context,AlgorithmPlan(identifier+'_'+tail,(base,tail),0.,('Frozen development contrast',)))
                        folder=OUT/'paired_products'/identifier/tail;folder.mkdir(parents=True,exist_ok=True)
                        result.data.dataset.to_netcdf(folder/'product.nc',engine='h5netcdf')
                        with xr.open_dataset(folder/'product.nc',engine='h5netcdf') as ds:metrics=reference(ds.load())
                        common=dict(input_fingerprint=digest(coarse.aod.values.tolist()),queue_fingerprint=reference.protocol['queue_fingerprint'],
                            reference_fingerprint=reference.protocol['reference_sha256'],information_fingerprint=digest(['same coarse input','no labels in prediction']),
                            parameters_fingerprint=digest(dict(base=base,target=[fine.lat.values.tolist(),fine.lon.values.tolist()],operator='area weights',nonnegative=True)),
                            metric='RMSE',metric_kind='reference_error',loss_unit='1',status='PASS',actual_execution=True,quality_pass=result.report['quality']['status']!='BLOCK',
                            loss=metrics['RMSE'],seconds=time.perf_counter()-start,product=str(folder/'product.nc'),steps=[base,tail],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        save(folder/'evidence.json',dict(**common,metrics=metrics,quality=result.report));paired.append(common)
                    except Exception as exc:failures.append(dict(id=identifier,tail=tail,error=str(exc)));break
                if len(paired)==2:
                    records.append(dict(id=identifier,split='development',group=region,comparison_id=f'{path.stem}_{size}',
                        attribution='single_capability',changed_stages=[dict(before='raster_task.nonnegative',after='raster_task.coarse_consistency',additional_capability='coarse_scale_consistency')],
                        before=paired[0],after=paired[1],features=features,context=dict(task_family='spatial_raster',variable='aod',unit='1',metric='RMSE',metric_kind='reference_error',
                            capability='coarse_scale_consistency',implementation='raster_task.coarse_consistency',base_algorithm=base,support='cell_mean')))
    save(OUT/'pairs.json',records);save(OUT/'collection.json',dict(seconds=time.perf_counter()-clock,pairs=len(records),failures=failures,new_pipeline_calls=2*len(records)+len(failures)))
    return records


def run(version='v2'):
    records=collect()
    from ecodata.algorithms.capability_scoring import ROOT as store,paired_evidence
    if (store/MODEL/'report.json').exists():
        report=json.loads((store/MODEL/'report.json').read_text(encoding='utf-8'))
        accepted,_=paired_evidence(records,epsilon=PROTOCOL['epsilon'],loss_unit=PROTOCOL['loss_unit'])
        assert report['protocol']==PROTOCOL and report['training_records_fingerprint']==digest([r for r in accepted if r['group'] not in PROTOCOL['holdout_groups']])
    else:report=CapabilityScoringService.train(records,PROTOCOL,MODEL)
    if not (OUT/'model_report.json').exists():save(OUT/'model_report.json',report)
    # Use a TRAIN input, not a held-out field, for the integration check. The
    # separate Sichuan holdout in report.json evaluates empirical prediction.
    with xr.open_dataset(ROOT/'output/lghap_reconstruction_20261020_v3/tiles/train/BTH_20190101.nc',engine='h5netcdf') as ds:fine=ds.isel(lat=slice(0,24),lon=slice(0,24)).load()
    fine.attrs['crs']='EPSG:4326';fine.aod.attrs.update(units='1',data_kind='continuous',spatial_semantics='cell_mean')
    coarse,_=degradation(fine,12);path=OUT/'inputs/coarse.nc';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():coarse.to_netcdf(path,engine='h5netcdf')
    if not (OUT/'inputs/reference.nc').exists():fine.to_netcdf(OUT/'inputs/reference.nc',engine='h5netcdf')
    req=dict(task='spatial',variable='aod',unit='1',semantic_type='continuous',nonnegative=True,
        target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),requirements=dict(try_coarse_consistency=True),
        optional_scoring=dict(model_id=MODEL,enabled=True,acknowledge_limited_development=True))
    declarations=[dict(feature=k,value=v,unit='1',source_id='role:primary:aod',confirmed=True,status='CONFIRMED',evidence='independent fixed input inspection')
        for k,v in [('grid_change',True),('missing_count',0),('variable_class','continuous'),('spatial_support','cell_mean')]]
    inp=[SearchInput('coarse.nc','primary',path.read_bytes())];results={}
    for layer,client,enabled in [('declared','toolkit',True),('recognized','toolkit',True),('recognized','page',True),('recognized_off','toolkit',False)]:
        request=dict(req,optional_scoring=dict(req['optional_scoring'],enabled=enabled))
        if layer=='declared':request['feature_declarations']=declarations
        key=layer+'_'+client;folder=OUT/'runs'/(key+'_'+version)
        function=process_search_upload if client=='page' else submit_search
        result=function(inp,request,DatasetReference(fine.aod,'aod'),folder,budget=dict(BUDGET),mode='C-local',seed=11)
        results[key]=dict(counts=result['counts'],selected=result['best']['key'] if result['best'] else None,objective=result['best']['objective'] if result['best'] else None,
            seconds=result['total_service_seconds'],product=result['best']['product'] if result['best'] else None)
    save(OUT/('acceptance_'+version+'.json'),dict(results=results,scope='mechanism only; RF support/quality judged separately; no GOLEM efficacy experiment'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--version',default='v2');run(p.parse_args().version)
