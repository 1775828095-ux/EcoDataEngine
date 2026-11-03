"""Publish immutable tested portraits; exercise file and page public entrypoints."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.models import portrait_policy as policy
from ecodata.interface.search_page import process_search_upload
OUT=ROOT/'output/complementary_20261024/study_v3'
RELEASE=ROOT/'output/complementary_20261024/delivery'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def immutable_copy(source,target):
    raw=source.read_bytes();target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==raw,('Immutable version collision',str(target))
    else:target.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()

def main():
    results=load(OUT/'results.json');frozen=load(OUT/'frozen_protocol.json')
    for version in ('complementary-state-v1','complementary-state-v2'):
        immutable_copy(OUT/'attributes'/(version+'.json'),ar.ROOT/(version+'.json'));ar.load(version)
    for version in ('complementary-wrf-v1','complementary-wrf-v2'):
        immutable_copy(OUT/'models'/(version+'.json'),policy.ROOT/(version+'.json'));policy.load(version)
    response=ROOT/'data/algorithm_response_portraits/complementary-error-response-v1.json'
    sha=immutable_copy(OUT/'response_portrait.json',response)
    config=load(OUT/'cases/1600/request.json')
    req=config['request'];req['scoring']['registry_version']='complementary-state-v2';req['portrait_policy']['version']='complementary-wrf-v2'
    req['composition']['response_profile']=dict(path=str(response),sha256=sha)
    req['composition']['search']['backend']='alns';config['seed']=11
    config.pop('evaluation',None)
    save(RELEASE/'request_no_truth.json',config);save(RELEASE/'page_request.json',req)
    # No evaluator/reference passed by either client.
    file_result=api.submit_file_request(config,RELEASE/'file_entry')
    inputs=[api.SearchInput(Path(i['path']).name,i['role'],Path(i['path']).read_bytes()) for i in config['inputs']]
    page_result=process_search_upload(inputs,deepcopy(req),None,RELEASE/'page_entry',mode='Profile',seed=11,budget=config['budget'])
    original=load(OUT/'test/1600/alns/11/summary.json')
    assert file_result['recommended_key']==page_result['recommended_key']==original['recommended_key']
    assert file_result['best']['numeric_fingerprint']==page_result['best']['numeric_fingerprint']==original['best']['numeric_fingerprint']
    assert file_result['screening']['scored']==56 and file_result['native_optimizer']['package']=='alns'
    assert not file_result['reference_used_for_selection'] and not page_result['reference_used_for_selection']
    manifest=[]
    paths=[ROOT/'src/ecodata/algorithms'/name for name in ('composition_decision.py','composition_optimizers.py','composition_response.py','spatiotemporal_detail.py','portrait_state.py','portrait_rules.py','platform_search.py')]
    paths += [ROOT/'src/ecodata/models/portrait_policy.py',ROOT/'src/ecodata/interface/search_page.py',ROOT/'src/ecodata/temporal/semantic.py']
    for path in paths:
        if not path.exists():continue
        target=RELEASE/'source'/path.relative_to(ROOT);hash_value=immutable_copy(path,target)
        manifest.append(dict(path=str(path),sha256=hash_value))
    save(RELEASE/'acceptance.json',dict(status='PASS',screened=56,reference_supplied=False,file_page_same_product=True,
        response_sha256=sha,optimizer=file_result['native_optimizer'],mean_rmse=results['mean_rmse'],source_manifest=manifest,
        limitations=['Controlled degradation, not independent sensors','WRF input preference proxy, not WRF simulation','No proof of universal optimality or mutation accuracy gain']))
    print(json.dumps(dict(status='PASS',mean_rmse=results['mean_rmse']),ensure_ascii=False))

if __name__=='__main__':main()
