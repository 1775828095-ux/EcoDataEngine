"""Exercise file/page handlers and an actual CLI, without hidden references."""
from copy import deepcopy
from pathlib import Path
import json,subprocess,sys

from run_parallel_portrait_study import OUT,ROOT,stores,bundle,load,save,sha
from ecodata.algorithms import platform_search as api
from ecodata.interface.search_page import process_search_upload


def main():
    stores();identifier='val_south_east_1600';inputs,req=bundle(identifier)
    calibration=load(OUT/'calibration.json')
    req['composition']['parallel_scoring']=dict(weights=calibration['weights']['full'],
        evidence='Frozen development calibration '+sha(OUT/'calibration.json'))
    req['composition']['search'].update(backend='none',generations=0)
    config=dict(inputs=load(OUT/'cases'/identifier/'config.json')['inputs'],request=req,
        mode='Profile',budget=dict(objective_limit=2,proposal_limit=1,wall_seconds=120))
    save(OUT/'delivery/request_no_reference.json',config)
    a=load(OUT/'delivery/file_entry/summary.json') if (OUT/'delivery/file_entry/summary.json').exists() else api.submit_file_request(config,OUT/'delivery/file_entry')
    b=load(OUT/'delivery/page_entry/summary.json') if (OUT/'delivery/page_entry/summary.json').exists() else process_search_upload(inputs,req,None,OUT/'delivery/page_entry',mode='Profile',budget=config['budget'])
    if not (OUT/'delivery/cli_entry/summary.json').exists():
        cli=subprocess.run([sys.executable,'-m','ecodata.algorithms.platform_search',
            str(OUT/'delivery/request_no_reference.json'),str(OUT/'delivery/cli_entry')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=180)
        (OUT/'delivery/cli.log').write_text(cli.stdout+'\n'+cli.stderr,encoding='utf-8')
        assert cli.returncode==0,cli.stderr
    c=load(OUT/'delivery/cli_entry/summary.json')
    assert a['recommended_key']==b['recommended_key']==c['recommended_key']
    assert a['best']['numeric_fingerprint']==b['best']['numeric_fingerprint']==c['best']['numeric_fingerprint']
    assert all(x['parallel_scoring']['unique_score_intervals']>1 for x in (a,b,c))
    assert all(x['parallel_scoring']['probe_graph_executions']==72 for x in (a,b,c))
    assert all(not x.get('search_validation_used_for_selection',False) for x in (a,b,c))
    # A ready validated search configuration uses the same public path.
    validated=deepcopy(config);validated['request']['composition']['search'].update(backend='alns',generations=100)
    validated['budget']=load(OUT/'protocol.json')['search_budget']
    validated['search_validation']=dict(path=str(OUT/'cases'/identifier/'labels.nc'))
    save(OUT/'delivery/request_validated_search.json',validated)
    if not (OUT/'delivery/cli_validated/summary.json').exists():
        validated_cli=subprocess.run([sys.executable,'-m','ecodata.algorithms.platform_search',
            str(OUT/'delivery/request_validated_search.json'),str(OUT/'delivery/cli_validated')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=180)
        (OUT/'delivery/cli_validated.log').write_text(validated_cli.stdout+'\n'+validated_cli.stderr,encoding='utf-8')
        assert validated_cli.returncode==0,validated_cli.stderr
    validated_result=load(OUT/'delivery/cli_validated/summary.json')
    formal=load(OUT/'search'/identifier/'full'/'11'/'summary.json')
    assert validated_result['recommended_key']==formal['recommended_key']
    assert validated_result['best']['numeric_fingerprint']==formal['best']['numeric_fingerprint']
    save(OUT/'delivery/acceptance.json',dict(status='PASS',file_page_cli_same_product=True,
        independent_reference_supplied=False,search_validation_supplied=False,visible_mask_probe_executions=72,
        recommended_key=a['recommended_key'],unique_score_intervals=a['parallel_scoring']['unique_score_intervals'],
        validated_cli_same_as_formal_search=True,validated_cli_actual_executions=validated_result['counts']['evaluated'],
        scope='Actual public handlers and CLI; browser widgets not automated; no WRF run claim'))
    print(json.dumps(load(OUT/'delivery/acceptance.json')),flush=True)


if __name__=='__main__':main()
