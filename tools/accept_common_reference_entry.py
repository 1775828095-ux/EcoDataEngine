from copy import deepcopy
import json,subprocess,sys
from pathlib import Path

from run_common_reference_study import ROOT,OUT,bundle
from run_scheduled_fusion_study import load,save,sha
from ecodata.algorithms import platform_search as api
from ecodata.interface.search_page import process_search_upload


def main():
    identifier='south_east_2000';inputs,req=bundle(identifier)
    req['composition']['parallel_scoring']=dict(weights=dict(data=.5,model=.5,collaboration=0),
        common_reference=dict(path=str(OUT/'common_reference.json'),sha256=sha(OUT/'common_reference.json')),
        evidence='Development-frozen common-reference secondary gate')
    req['composition']['search'].update(backend='none',generations=0)
    config=dict(inputs=load(OUT/'cases'/identifier/'config.json')['inputs'],request=req,mode='Profile',budget=dict(objective_limit=2,wall_seconds=180))
    save(OUT/'delivery/request_no_reference.json',config)
    results=[]
    for name in ('file','page'):
        folder=OUT/'delivery'/name
        result=load(folder/'summary.json') if (folder/'summary.json').exists() else (
            api.submit_file_request(config,folder) if name=='file' else process_search_upload(inputs,req,None,folder,mode='Profile',budget=config['budget']))
        results.append(result)
    folder=OUT/'delivery/cli'
    if not (folder/'summary.json').exists():
        done=subprocess.run([sys.executable,'-m','ecodata.algorithms.platform_search',str(OUT/'delivery/request_no_reference.json'),str(folder)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=240)
        (OUT/'delivery/cli.log').write_text(done.stdout+'\n'+done.stderr,encoding='utf-8');assert done.returncode==0,done.stderr
    results.append(load(folder/'summary.json'))
    assert len({r['recommended_key'] for r in results})==1
    assert len({r['best']['numeric_fingerprint'] for r in results})==1
    assert all(r['parallel_scoring']['probe_graph_executions']==76 for r in results)
    assert all(r['best']['matching']['common_gain']['status']=='MEASURED_VISIBLE_MASK' for r in results)
    validated=deepcopy(config);validated['request']['composition']['search'].update(backend='alns',population=64,generations=100)
    validated['budget']=dict(objective_limit=256,proposal_limit=4096,wall_seconds=1800)
    validated['search_validation']=dict(path=str(OUT/'cases'/identifier/'labels.nc'))
    save(OUT/'delivery/request_validated_search.json',validated)
    save(OUT/'delivery/acceptance.json',dict(status='PASS',file_page_cli_same_product=True,hidden_reference_supplied=False,
        reference_probe_executions=4,local_candidate_probes=72,common_reference_shared=True,
        scope='Actual public handlers and CLI; browser widgets not automated; declared engineering model preferences'))
    print(json.dumps(load(OUT/'delivery/acceptance.json')),flush=True)


if __name__=='__main__':main()
