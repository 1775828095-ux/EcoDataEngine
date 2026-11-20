"""Mechanism fixtures, not hydrological accuracy evidence."""
from pathlib import Path
import hashlib,json,zipfile
import pytest
from ecodata.desktop.file_agent import FileTools,scan_index
from ecodata.desktop import discovery
from ecodata.models import input_context
from ecodata.interface.engineering_upload import LocalUpload


def test_control_bytes_are_real_and_not_eight_line_header(tmp_path):
    p=tmp_path/'case.zip'
    control=b'rev.60\nsimulation time.sim null\nclimate station.cli null\n'
    with zipfile.ZipFile(p,'w') as z:z.writestr('file.cio',control)
    tools=FileTools(p);r=tools.call('read_header',dict(path='file.cio'))
    row=next(iter(tools.rows.values()))
    assert row['original_probe_payload']==control
    assert 'original_probe_payload' not in r['result']['row']
    assert 'raw_base64' not in r['result']['document']


def test_text_pages_accumulate_without_replacing_control_start(tmp_path):
    p=tmp_path/'long.txt';raw=b'A'*16384+b'END';p.write_bytes(raw)
    tools=FileTools(p);tools.call('read_text',dict(path='',offset=0))
    row=next(iter(tools.rows.values()));assert 'original_probe_payload' not in row
    tools.call('read_text',dict(path='',offset=16384))
    assert next(iter(tools.rows.values()))['original_probe_payload']==raw


def test_saved_context_exact_content_and_model_not_filename(tmp_path,monkeypatch):
    store=tmp_path/'store';store.mkdir();monkeypatch.setattr(input_context,'STORE',store)
    p=tmp_path/'arbitrary.zip';p.write_bytes(b'actual input');sha=hashlib.sha256(p.read_bytes()).hexdigest()
    context=dict(schema='ecodata.saved-input-context.v1',source_sha256=sha,model_id='wrf',status='WORKING_ASSUMPTION_ONLY',
        documents={'assumption.json':dict(status='WORKING_ASSUMPTION_ONLY',rule='declared development assumption')})
    (store/(sha+'.json')).write_text(json.dumps(context))
    uploads,receipt=input_context.augment([LocalUpload(p.name,p)],'wrf')
    assert len(uploads)==2 and receipt['status']=='WORKING_ASSUMPTION_ONLY'
    assert input_context.find(p,'cmaq') is None
    renamed=tmp_path/'renamed.bin';renamed.write_bytes(p.read_bytes());assert input_context.find(renamed,'wrf')
    p.write_bytes(b'changed input');assert input_context.find(p,'wrf') is None


def test_declared_goal_survives_shallow_public_catalog(tmp_path):
    p=tmp_path/'input.csv';p.write_text('a\n1\n');scan,_=scan_index(p)
    summary,_=discovery.discover(scan,'swat_plus',task_goal='evaluate_gain')
    assert summary['task_goal']=='evaluate_gain'


def test_selected_header_loads_exact_companion_receipt(tmp_path):
    import xarray as xr
    p=tmp_path/'any.nc';xr.Dataset({'x':('n',[1.,2.])}).to_netcdf(p,engine='scipy')
    receipt=p.with_name(p.name+'.receipt.json');receipt.write_text(json.dumps(dict(url='https://example.org',sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
    tools=FileTools(tmp_path);tools.call('read_header',dict(path='any.nc'))
    assert any(name.endswith('any.nc.receipt.json') for name in tools.documents)


@pytest.mark.parametrize('quality_only',[False,True])
def test_real_project_delegates_to_public_port_checks_not_quality_pass(tmp_path,quality_only):
    import numpy as np
    import xarray as xr
    from ecodata.desktop import catalog_decision
    from ecodata.desktop.scan import Scanner
    for i in range(2):
        xr.Dataset({'temperature':(('time','lat','lon'),np.full((2,2,2),280.+i),
            dict(units='K',physical_variable='air_temperature',temporal_semantics='instantaneous',spatial_semantics='grid_estimate'))},
            coords=dict(time=np.array(['2020-01-01','2020-01-02'],dtype='datetime64[ns]'),lat=[1.,2.],lon=[1.,2.]),attrs=dict(crs='EPSG:4326')).to_netcdf(tmp_path/f'input{i}.nc',engine='scipy')
    scan,_=Scanner().scan(tmp_path);summary,_=discovery.discover(scan,'wrf',task_goal='evaluate_gain')
    originals,details=catalog_decision.local_fields(scan,summary)
    fields=[v for v in originals.values() if v[2]['name']=='temperature']
    pair,local=discovery.describe_pair(scan,'C1',*fields,'wrf',details)
    scan['native_model_probes']['wrf']['status']='project'  # Explicit functional fixture.
    decisions=[dict(candidate='C1',action='quality_screen',reason='check quality first')]
    if not quality_only:decisions.append(dict(candidate='C1',action='fusion',goal='evaluate_gain',reason='fixture'))
    checked,gated=discovery.check(scan,{'C1':local},dict(decisions=decisions),dict(summary,candidates=[pair]))
    row=next(r for r in checked if r['action']=='fusion')
    assert row['status']!='通过' and row['next_step']=='original_workbench'
    assert gated[('C1','fusion')]['route']=='original_engineering'
    quality=next(r for r in checked if r['action']=='quality_screen')
    assert quality['status']!='通过' and quality['next_step']=='runtime_quality_gate'
    assert not discovery.blocking_checks(checked)
