"""Synthetic workflow boundaries; not fusion accuracy evidence."""
from pathlib import Path
import io,json,tarfile,zipfile
import pytest
from ecodata.desktop.file_agent import FileTools
from ecodata.desktop.file_tools import inspect
from ecodata.desktop.discovery import blocking_checks
from ecodata.desktop.issues import checks


def test_prefix_header_uses_shared_compound_decoder_and_retains_full_size(tmp_path):
    header='STATION,DATE,REPORT_TYPE,LATITUDE,LONGITUDE,ELEVATION,TMP,WND\n'
    row='1,2000-01-01T00:00:00,FM-15,30,110,10,"+0200,1","180,1,N,0020,1"\n'
    payload=(header+row*10000).encode()
    p=tmp_path/'data.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('observations.csv',payload)
    tools=FileTools(p);folder=Path(tools._session.name)
    try:
        a=tools.call('read_header',dict(path='observations.csv'))
        b=tools.call('read_header',dict(path='observations.csv'))
        assert a==b and tools.events[-1]['cache_hit']
        fields={f['name']:f for f in a['result']['row']['fields']}
        assert fields['TMP.air_temperature']['unit']=='degC'
        assert fields['WND.wind_speed']['unit']=='m/s'
        assert fields['TMP']['noaa_compound']
        assert a['result']['row']['bytes']==len(payload)
        assert '16 KiB' in a['result']['sample_scope']
    finally:tools.close()
    assert not folder.exists()


def test_nested_tar_metadata_and_container_reused_and_removed(tmp_path):
    tar=io.BytesIO()
    with tarfile.open(fileobj=tar,mode='w:gz') as t:
        for name in ['a.csv','b.csv']:
            raw=b'time,value\n2000-01-01,1\n';entry=tarfile.TarInfo(name);entry.size=len(raw);t.addfile(entry,io.BytesIO(raw))
    p=tmp_path/'bundle.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('inputs.tgz',tar.getvalue())
    tools=FileTools(p);folder=Path(tools._session.name)
    try:
        for name in ['a.csv','b.csv']:
            assert tools.call('read_header',dict(path='inputs.tgz!/'+name))['ok']
        assert len(list(folder.glob('*.tgz')))==1
        assert len(list(folder.glob('*.tar-index.json')))==1
    finally:tools.close()
    assert not folder.exists()


def test_invalid_proposal_is_skipped_but_unknown_unit_still_blocks_its_operation():
    rows=[dict(candidate='C1',status='条件不足',block_code='coordinate_field',disposition='SKIPPED_NOT_PROCESSING_VARIABLE',reason='coordinate'),
          dict(candidate='C2',status='条件不足',block_code='unit_unknown',reason='unknown unit')]
    assert [x['candidate'] for x in blocking_checks(rows)]==['C2']
    assert checks(rows)[0]['status']=='已自动跳过'


def test_input_change_invalidates_header_cache(tmp_path):
    p=tmp_path/'a.csv';p.write_text('a\n1\n')
    tools=FileTools(p)
    try:
        assert tools.call('read_header',dict(path=''))['ok']
        p.write_text('a\n123456\n')
        with pytest.raises(ValueError,match='变化'):tools.call('read_header',dict(path=''))
    finally:tools.close()


def test_absent_flag_metadata_does_not_turn_scientific_field_into_quality_flag(tmp_path):
    import numpy as np
    import xarray as xr
    p=tmp_path/'grid.nc'
    ds=xr.Dataset(dict(temperature=(('lat','lon'),np.ones((2,2)))),coords=dict(lat=[30,31],lon=[110,111]))
    ds.temperature.attrs.update(units='K',standard_name='air_temperature')
    ds.to_netcdf(p)
    result=inspect(p,'read_header',dict(path=''))
    f=next(f for f in result['row']['fields'] if f['name']=='temperature')
    assert f['metadata_role']=='physical_candidate'


def test_record_unit_evidence_is_not_promoted_to_global_unit(tmp_path):
    p=tmp_path/'observations.csv'
    p.write_text('Parameter Code,Sample Measurement,Units of Measure\n44201,0.04,Parts per million\n')
    result=inspect(p,'read_header',dict(path=''))
    f=next(f for f in result['row']['fields'] if f['name']=='Sample Measurement')
    assert f['unit']=='' and f['unit_column']=='Units of Measure'
    assert f['sampled_unit_values']==['Parts per million']
    code=next(f for f in result['row']['fields'] if f['name']=='Parameter Code')
    assert not code.get('unit_column')


def test_session_cleanup_and_separate_timing_even_when_inspection_fails(tmp_path,monkeypatch):
    from ecodata.desktop import file_agent
    p=tmp_path/'a.csv';p.write_text('value\n1\n')
    paths=[]
    def complete(*args):
        paths.append(Path(args[-1]._session.name))
        return ({},{},{},{},{},{'file_agent_audit':{}})
    monkeypatch.setattr(file_agent,'_run',complete)
    result=file_agent.run({'path':p},{},{},False,None,None,None)
    assert result[-1]['file_agent_audit']['timing']['session_cache_removed']
    assert not paths[-1].exists()
    def fail(*args):
        paths.append(Path(args[-1]._session.name));raise ValueError('network failure')
    monkeypatch.setattr(file_agent,'_run',fail)
    with pytest.raises(ValueError,match='network failure'):
        file_agent.run({'path':p},{},{},False,None,None,None)
    assert not paths[-1].exists()
