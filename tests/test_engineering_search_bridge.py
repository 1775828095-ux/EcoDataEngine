"""Real public search and native writeback, with synthetic, labelled fixtures."""
from pathlib import Path
from io import BytesIO
import json,zipfile
import numpy as np,pandas as pd,xarray as xr,pytest
from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.models.engineering.ports import canonical,InputPort
from ecodata.models.engineering.search_bridge import search_port,SearchNotReady
from test_unified_engineering_ports import upload,pack


@pytest.fixture(autouse=True)
def bounded_public_search(monkeypatch):
    import ecodata.algorithms.platform_search as service
    monkeypatch.setattr(service,'CURRENT_INPUT_BUDGET',dict(objective_limit=40,proposal_limit=150,total_call_limit=200,wall_seconds=15))


def inputs(n=9,days=4):
    xy=np.array([(i%3,i//3) for i in range(n)])
    d=canonical(np.tile(np.arange(n)+1.,(days,1)),pd.date_range('2020-01-01',periods=days),xy[:,0],xy[:,1],'precipitation','mm','accumulation')
    d.attrs['time_basis']='UTC';d.precipitation.attrs['interval_seconds']=86400
    arrays=[];origins=[]
    for i in range(2):
        a=d.copy(deep=True);a.precipitation.values[:]+=i;a.attrs['source_id']='real_input_identity_'+str(i)
        arrays.append(a);origins.append(dict(source_id=a.attrs['source_id']))
    return InputPort('rain.pcp','precipitation','precipitation','mm','accumulation','swat_weather',d,{}),arrays,origins


def test_public_mutation_without_sigma_executes_multiple_candidates_and_deploys(tmp_path):
    port,arrays,origins=inputs()
    out,receipt=search_port(port,arrays,origins,tmp_path/'run',budget=dict(objective_limit=40,proposal_limit=150,total_call_limit=200,wall_seconds=15))
    result=receipt['result']
    assert result['status']=='PASS' and result['optimizer']['backend']=='random'
    assert result['counts']['evaluated']>1 and result['evolution_used']
    assert result['deployment_trace'] and result['selected_graph']
    assert not result['self_validation_protocol']['test_used_for_selection']
    assert result['self_validation_protocol']['reference_scope']=='held visible grid estimates'
    assert not receipt['uncertainty_required'] and np.isfinite(out.precipitation).all()


def test_tiny_series_reports_validation_evidence_not_uncertainty(tmp_path):
    port,arrays,origins=inputs(n=1,days=2)
    with pytest.raises(SearchNotReady) as exc:search_port(port,arrays,origins,tmp_path/'run')
    assert exc.value.report['status']=='VALIDATION_EVIDENCE_INSUFFICIENT'
    assert (tmp_path/'run/search_receipt.json').exists()


def test_public_engineering_groups_real_sites_searches_and_writes_all_ports(tmp_path):
    files,name,_,_,_=pack('swat_plus','precipitation','mm','accumulation')
    # Nine actual coordinate records in a synthetic native project; no fake
    # uncertainty, preselected algorithm or reference values in a request.
    stations=[]
    for i in range(9):
        x,y=i%3,i//3
        stations.append(f'S{i} null r{i}.pcp temp.tem null null null')
        files[f'project/r{i}.pcp']=f'title\nNBYR TSTEP LAT LON ELEV\n1 0 {y} {x} 1\n2020 1 1\n2020 2 2\n'.encode()
    files['project/weather-sta.cli']=('title\nname wgn pcp tmp slr hmd wnd\n'+'\n'.join(stations)+'\n').encode()
    files['project/pcp.cli']=('title\nfilename\n'+'\n'.join(f'r{i}.pcp' for i in range(9))+'\n').encode()
    resources=[dict(resource=f'project/r{i}.pcp',role='model_input',variables={'precipitation':{'time_basis':'UTC'}},confirmed=True) for i in range(9)]
    files['semantics.json']=json.dumps(dict(schema='ecodata.engineering-semantics.v1',model_id='swat_plus',resources=resources)).encode()
    xy=np.array([(x,y) for y in range(3) for x in range(3)])
    for i in range(2):
        d=canonical(np.tile(np.arange(9)+3.+i,(2,1)),pd.date_range('2020-01-01',periods=2),xy[:,0],xy[:,1],'precipitation','mm','accumulation')
        d.precipitation.attrs['interval_seconds']=86400
        d.attrs.update(engineering_role='source',source_id='source_'+str(i),time_basis='UTC')
        files[f'source{i}.nc']=d.to_netcdf(engine='scipy')
    payload=BytesIO()
    with zipfile.ZipFile(payload,'w') as z:
        for n,v in files.items():z.writestr(n,v)
    result=prepare_uploads([upload('project.zip',payload.getvalue())],'swat_plus',tmp_path/'run')
    assert result['manifest']['status']=='PROJECT_UPDATED'
    tasks=result['manifest']['preparation']['tasks']
    assert len(tasks)==9 and len({t['shared_search_id'] for t in tasks})==1
    assert tasks[0]['execution']['result']['optimizer']['backend']=='random'
    with zipfile.ZipFile(result['archive']) as z:
        assert z.testzip() is None
        for n,v in files.items():
            if not n.startswith('project/r'):assert z.read(n)==v
        for i in range(9):
            rows=z.read(f'project/r{i}.pcp').decode().splitlines()[3:]
            assert len(rows)==2 and all(float(r.split()[2])>=0 for r in rows)
    from streamlit.testing.v1 import AppTest
    input_path=tmp_path/'request.zip';input_path.write_bytes(payload.getvalue())
    page=AppTest.from_file(Path(__file__).parents[1]/'src/ecodata/interface/app.py').run(timeout=30)
    page.text_input[0].set_value(str(input_path)).run();page.button[0].click().run(timeout=30)
    assert not page.exception and not page.error
    actual=page.session_state['workbench_result']['data']
    assert actual['manifest']['status']=='PROJECT_UPDATED'
    runs=actual['manifest']['preparation']['search_runs']
    assert len(runs)==1 and next(iter(runs.values()))['result']['optimizer']['backend']=='random'
