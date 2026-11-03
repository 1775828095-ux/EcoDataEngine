"""File/page public clients on the frozen validation case, never test feedback."""
from copy import deepcopy
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecodata.algorithms import platform_search as api
from ecodata.interface.search_page import process_search_upload
OUT=ROOT/'output/scheduled_fusion_20261025';DELIVERY=OUT/'delivery'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert (OUT/'results.json').exists() and (OUT/'frozen_selections.json').exists()
    base=load(OUT/'cases/1510/config.json');req=base['request'];req['composition']['search'].update(backend='alns',exact_check=False)
    config=dict(inputs=base['inputs'],request=req,mode='Profile',seed=11,budget=load(OUT/'frozen_protocol.json')['budget'],search_validation=dict(path=str(OUT/'cases/1510/labels.nc')))
    save(DELIVERY/'request.json',config);save(DELIVERY/'page_request.json',req)
    exact=deepcopy(config);exact['request']['composition']['search']['exact_check']=True;exact['budget']=load(OUT/'frozen_protocol.json')['oracle_budget']
    exact['budget']['proposal_limit']=256
    save(DELIVERY/'request_exact_validation.json',exact)
    if (DELIVERY/'file_entry/summary.json').exists():file_result=load(DELIVERY/'file_entry/summary.json')
    else:file_result=api.submit_file_request(config,DELIVERY/'file_entry')
    inputs=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in base['inputs']]
    import xarray as xr
    with xr.open_dataset(config['search_validation']['path'],engine='h5netcdf') as ds:validation=api.SearchValidation(ds[req['variable']].load(),req['variable'])
    if (DELIVERY/'page_entry/summary.json').exists():page_result=load(DELIVERY/'page_entry/summary.json')
    else:page_result=process_search_upload(inputs,deepcopy(req),None,DELIVERY/'page_entry',mode='Profile',seed=11,budget=config['budget'],validation=validation)
    original=load(OUT/'search/1510/alns/11/summary.json')
    assert file_result['recommended_key']==page_result['recommended_key']==original['recommended_key']
    assert file_result['best']['numeric_fingerprint']==page_result['best']['numeric_fingerprint']==original['best']['numeric_fingerprint']
    assert file_result['search_validation_used_for_selection'] and not file_result['independent_test_used_for_selection']
    for report in (file_result,page_result):
        p=Path(report['best']['product']);assert hashlib.sha256(p.read_bytes()).hexdigest()==report['best']['product_sha256']
    controls=[]
    for _,start in load(OUT/'frozen_protocol.json')['validation_test_pairs']:
        case_config=load(OUT/'cases'/str(start)/'config.json')
        test_inputs=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in case_config['inputs']]
        for source in ('fine_slow','coarse_fast'):
            control=deepcopy(case_config['request']);composition=control['composition']
            composition['source_stages']['coarse_fast']=[dict(id='grid',options=[dict(algorithm='raster_task.spatial.bilinear')],evidence='Frozen bilinear single-coarse-source engineering control')]
            composition['post_stages']=[];composition['fusion_options']=[dict(algorithm='prepare.select_source',source=source,evidence='Supplementary source-only control; no test selection')]
            composition['search'].update(backend='none',generations=0,survivors=1)
            folder=OUT/'single_source_controls'/str(start)/source
            if (folder/'summary.json').exists():report=load(folder/'summary.json')
            else:report=api.submit_search(test_inputs,control,None,folder,mode='Profile',budget=dict(objective_limit=1,proposal_limit=1,wall_seconds=120))
            assert report['counts']['grammar_combinations']==1
            with xr.open_dataset(OUT/'cases'/str(start)/'labels.nc',engine='h5netcdf') as ds:truth=ds[control['variable']].load()
            with xr.open_dataset(report['best']['product'],engine='h5netcdf') as ds:metrics=api.DatasetReference(truth,control['variable'])(ds.load())
            controls.append(dict(case=start,source=source,metrics=metrics,key=report['recommended_key'],phase='supplementary test-only engineering baseline; no selection or fitting'))
    save(DELIVERY/'single_source_controls.json',controls)
    save(DELIVERY/'acceptance.json',dict(status='PASS',file_page_same_product=True,test_reference_supplied=False,
        validation_supplied=True,validation_rmse=file_result['final_validation_rmse'],optimizer=file_result['native_optimizer'],
        public_configuration=str(DELIVERY/'request.json'),scope='Public file and page handler; browser widgets not automated'))
    print(json.dumps(dict(status='PASS',key=file_result['recommended_key'],rmse=file_result['final_validation_rmse'])),flush=True)
if __name__=='__main__':main()
