"""Automatic repair boundaries; synthetic cases do not prove environmental accuracy."""
from pathlib import Path
import io,json,zipfile
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.core.recovery import recover,session,attach,search_inputs,recover_table


@pytest.fixture(autouse=True)
def no_unrequested_api(monkeypatch):monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)


def data(n=20):
    return xr.Dataset({'v':('time',np.arange(n,dtype=float)+280,
        dict(units='K',temporal_semantics='instantaneous',data_kind='continuous'))},
        coords=dict(time=pd.date_range('2020-01-01',periods=n,freq='h')),
        attrs=dict(source_id='one',crs='EPSG:4326',time_basis='UTC'))


def test_ds_evidence_is_applied_and_input_stays_readonly():
    ds=data();del ds.v.attrs['units'];before=ds.copy(deep=True)
    seen=[]
    def advisor(facts):
        seen.append(facts)
        return dict(use_fact_ids=[i for i,f in enumerate(facts) if f['key']=='units']),dict(kind='test_advisor')
    with session(advisor=advisor) as s:
        out,r=recover(ds,'A',docs=[dict(source='README.txt',text='v.units = K')])
    assert out.v.attrs['units']=='K' and seen and s.calls==1
    assert r['ds']['status']=='COMPLETED'
    xr.testing.assert_identical(ds,before)


def test_ds_cannot_invent_values_or_override_conflict():
    ds=data();del ds.v.attrs['units']
    with session(advisor=lambda facts:(dict(use_fact_ids=[999]),{})):
        out,r=recover(ds,'A',docs=[])
    assert 'units' not in out.v.attrs
    with session(advisor=lambda facts:(dict(use_fact_ids=[0]),{})):
        out,r=recover(ds,'A',docs=[dict(source='a',text='v.units = K'),dict(source='b',text='v.units = kg')])
    assert 'units' not in out.v.attrs
    assert r['metadata'][0]['action']=='UNRESOLVED_CONFLICT'


def test_ds_unavailable_still_uses_unambiguous_actual_evidence():
    ds=data();del ds.v.attrs['units']
    with session(advisor=lambda facts:(_ for _ in ()).throw(OSError('offline'))) as s:
        out,r=recover(ds,'A',docs=[dict(source='metadata.json',text=json.dumps(dict(variables=dict(v=dict(units='K')))))])
    assert out.v.attrs['units']=='K' and r['ds']['status']=='FAILED' and s.calls==1


@pytest.mark.parametrize('resolution',['ns','us'])
def test_local_gap_uses_existing_kernel_and_preserves_finite_values(tmp_path,resolution):
    ds=data().assign_coords(time=data().time.values.astype('datetime64['+resolution+']'))
    ds.v.values[5:7]=np.nan;before=ds.copy(deep=True)
    with session() as s:
        out,r=recover(ds,'A')
        result=attach({},tmp_path,s)
    np.testing.assert_allclose(out.v.values[5:7],[285.,286.])
    np.testing.assert_array_equal(out.v.values[np.isfinite(ds.v)],ds.v.values[np.isfinite(ds.v)])
    xr.testing.assert_identical(ds,before)
    assert out.v__recovered.values.sum()==2
    mask=np.load(tmp_path/r['repairs']['v']['mask_artifact'])
    assert mask['indices'].tolist()==[5,6]


def test_large_gap_edges_unknown_and_qa_are_not_made_up():
    for mode in ('large','edge','unknown','qa','irregular','duplicate'):
        ds=data()
        ds.v.values[4:12 if mode=='large' else 5]=np.nan
        if mode=='edge':ds.v.values[:1]=np.nan
        if mode=='unknown':ds.v.attrs['temporal_semantics']='UNKNOWN'
        if mode=='qa':ds.v.attrs['ancillary_variables']='qc'
        if mode=='irregular':ds=ds.assign_coords(time=ds.time.values+np.r_[np.zeros(10),np.ones(10)].astype('timedelta64[m]'))
        if mode=='duplicate':ds=ds.assign_coords(time=np.r_[ds.time.values[:1],ds.time.values[:-1]])
        with session():out,r=recover(ds,'A')
        if mode in {'large','unknown','qa','irregular','duplicate'}:assert not r['repairs']
        else:
            assert out.v.values[0]==281
            assert any(m.get('extrapolation') is True for m in r['repairs']['v']['methods'])


def test_equal_interval_amounts_are_estimates_not_disaggregated_totals():
    ds=data();ds.v.attrs.update(units='mm',temporal_semantics='accumulation',interval_seconds=3600)
    ds.v.values[5]=np.nan
    with session():out,r=recover(ds,'rain')
    assert out.v.values[5]==285 and r['repairs']['v']['filled']==1
    assert out.v.attrs['interval_seconds']==3600 and len(out.time)==len(ds.time)
    ds.v.attrs['interval_seconds']=86400
    with session():out,r=recover(ds,'rain')
    assert not r['repairs']


def test_legal_peer_fills_only_same_support_not_other_grid():
    ds=data();ds.v.values[5]=np.nan
    ds.v.attrs['temporal_semantics']='mean'  # No known interval; temporal fill ineligible.
    peer=data();peer.attrs['source_id']='two';peer.v.attrs['temporal_semantics']='mean'
    peer['time_bounds']=(('time','b'),np.column_stack([peer.time.values,peer.time.values+np.timedelta64(1,'h')]))
    peer.time.attrs['bounds']='time_bounds';ds['time_bounds']=peer.time_bounds;ds.time.attrs['bounds']='time_bounds'
    with session():out,r=recover(ds,'A',peers=[('B',peer)])
    assert out.v.values[5]==285 and r['repairs']['v']['methods'][0]['method']=='same_support_source_fill'
    peer=peer.assign_coords(time=peer.time+np.timedelta64(1,'h'))
    with session():out,r=recover(ds,'A',peers=[('B',peer)])
    assert np.isnan(out.v.values[5])


@pytest.mark.parametrize('role',['observations','reference','validation','test','target','paired_observations'])
def test_protected_roles_never_filled(role):
    ds=data();ds.v.values[5]=np.nan
    with session():out,r=recover(ds,'sealed',role=role)
    xr.testing.assert_identical(out,ds);assert r['status']=='PROTECTED_ROLE'


def test_search_never_turns_primary_gaps_into_evaluation_truth():
    from ecodata.algorithms.platform_search import SearchInput
    ds=data();ds.v.values[5]=np.nan;raw=bytes(ds.to_netcdf())
    with session():items=search_inputs([SearchInput('primary.nc','primary',raw),SearchInput('bg.nc','background',raw)])
    assert items[0].content==raw
    with xr.open_dataset(io.BytesIO(items[1].content)) as out:assert out.v.values[5]==285


def test_table_entities_do_not_borrow_each_others_values():
    t=pd.date_range('2020-01-01',periods=20,freq='h')
    f=pd.DataFrame(dict(t=list(t)*2,station=[1]*20+[2]*20,rain=np.r_[np.arange(20.),np.arange(20.)+100]))
    f.loc[[5,25],'rain']=np.nan
    cfg=dict(fields=dict(time='t',station='station',precipitation='rain'),quantity='one_hour_amount',
        accumulation_seconds=3600,source_units='mm',time_label='hour_end',timezone_west_hours=0)
    with session():out,r=recover_table(f,cfg,'A')
    assert out.loc[5,'rain']==5 and out.loc[25,'rain']==105 and len(r['groups'])==2
    assert f.rain.isna().sum()==2


def test_session_api_budget_and_no_key_in_receipt():
    def advisor(facts):return dict(use_fact_ids=list(range(len(facts)))),dict(usage=dict(total_tokens=1))
    with session(api_key='not-sent-test-key',advisor=advisor) as s:
        for i in range(3):
            ds=data();del ds.v.attrs['units']
            out,r=recover(ds,str(i),docs=[dict(source='README',text='v.units = K')])
            assert out.v.attrs['units']=='K'
        assert s.calls==1 and 'not-sent-test-key' not in json.dumps(s.report())


def test_scanned_evidence_is_source_scoped_and_reused_by_processing():
    from ecodata.core.recovery import scan_dataset
    a=data();del a.v.attrs['units'];b=a.copy(deep=True)
    scan=dict(rows=[dict(name='A.nc',format='NetCDF'),dict(name='B.nc',format='NetCDF')],
        engineering_docs=[('meta.json',json.dumps(dict(resource='A.nc',variables=dict(v=dict(units='K')))))])
    with session():
        aa,_=scan_dataset(scan,0,a);bb,_=scan_dataset(scan,1,b)
    assert aa.v.attrs['units']=='K' and 'units' not in bb.v.attrs


def test_unscoped_multi_source_description_cannot_donate_units():
    from ecodata.core.recovery import scan_dataset
    a=data();del a.v.attrs['units']
    scan=dict(rows=[dict(name='A.nc',format='NetCDF'),dict(name='B.nc',format='NetCDF')],
        engineering_docs=[('README.txt','v.units = K')])
    with session():out,_=scan_dataset(scan,1,a)
    assert 'units' not in out.v.attrs


def test_recovery_evidence_and_masks_are_in_data_only_download(tmp_path):
    from ecodata.desktop.service import bundle
    ds=data();ds.v.values[5]=np.nan
    result=dict(files=[],search=None,original_report=[],local=dict(artifacts=[],reason='diagnostics',native_ready=False))
    with session() as s:
        recover(ds,'background');attach(result,tmp_path,s)
    with zipfile.ZipFile(io.BytesIO(bundle(result))) as z:
        assert 'ecodata-automatic-recovery.json' in z.namelist()
        assert any(n.startswith('recovery-mask-') for n in z.namelist())
        assert z.testzip() is None


def test_large_missing_component_stops_only_it_and_keeps_recovery_receipt(tmp_path):
    from ecodata.algorithms.automatic_preparation import prepare_uploads
    d=xr.Dataset(dict(precipitation=('time',np.r_[np.ones(5),np.full(8,np.nan),np.ones(7)],
        dict(units='mm',temporal_semantics='accumulation',interval_seconds=86400))),
        coords=dict(time=pd.date_range('2018-01-01',periods=20),lat=30.,lon=110.,elevation=40.),attrs=dict(crs='EPSG:4326',time_basis='UTC'))
    raw=bytes(d.to_netcdf());b=io.BytesIO(raw);b.name='rain.nc'
    with pytest.raises(ValueError) as caught:prepare_uploads([b],'swat_plus',tmp_path)
    assert b.getvalue()==raw and caught.value.automatic_recovery['events']
    assert (tmp_path/'automatic_recovery_failures.jsonl').is_file()
    assert not (tmp_path/'EcoData-native-components.zip').exists()


def test_circular_and_categorical_data_are_not_scalar_interpolated():
    for changes in (dict(physical_variable='wind_from_direction'),dict(data_kind='categorical')):
        ds=data();ds.v.values[5]=np.nan;ds.v.attrs.update(changes)
        with session():out,r=recover(ds,'A')
        assert np.isnan(out.v.values[5]) and not r['repairs']


def test_bounded_ds_transport_has_one_attempt(monkeypatch):
    import urllib.request,urllib.error
    from ecodata.desktop import deepseek_transport as t
    calls=[]
    class Offline:
        def open(self,*args,**kwargs):
            calls.append(1)
            raise urllib.error.HTTPError('https://example.invalid',503,'offline',{},None)
    monkeypatch.setattr(t,'record_failure',lambda *args:None)
    logs=[]
    with pytest.raises(urllib.error.HTTPError):
        t.fetch(urllib.request.Request('https://example.invalid'),Offline(),'recovery','digest',logs.append,attempts=1,total_timeout=1)
    assert len(calls)==1 and any('1/1' in line for line in logs)


def test_front_gate_and_actual_pair_processing_share_repaired_metadata(tmp_path):
    from ecodata.desktop import discovery,catalog_decision,pair_processing
    from ecodata.desktop.scan import Scanner
    for i,step in enumerate((1.,.5)):
        axis=np.arange(1.,3.01,step)
        attrs=dict(physical_variable='air_temperature',temporal_semantics='instantaneous',spatial_semantics='grid_estimate',data_kind='continuous')
        if i:attrs['units']='K'
        d=xr.Dataset(dict(temperature=(('time','lat','lon'),np.full((2,len(axis),len(axis)),280.+i),attrs)),
            coords=dict(time=pd.date_range('2020-01-01',periods=2),lat=axis,lon=axis),attrs=dict(crs='EPSG:4326'))
        d.to_netcdf(tmp_path/f'source{i}.nc',engine='scipy')
    (tmp_path/'metadata.json').write_text(json.dumps(dict(resource='source0.nc',variables=dict(temperature=dict(units='K')))))
    scan,_=Scanner().scan(tmp_path)
    summary,_=discovery.discover(scan,'swat_plus')
    originals,details=catalog_decision.local_fields(scan,summary)
    refs=[v for v in originals.values() if v[2]['name']=='temperature']
    pair,local=discovery.describe_pair(scan,'C1',*refs,'swat_plus',details)
    with session():
        checked,gated=discovery.check(scan,{'C1':local},dict(decisions=[dict(candidate='C1',action='align_space')]),dict(summary,candidates=[pair]))
        assert ('C1','align_space') in gated,checked
        task=gated['C1','align_space']
        actual,_=pair_processing.fields(scan,task)
        assert all(d.temperature.attrs['units']=='K' for d in actual)


def test_existing_grid_gap_kernel_recovers_only_small_spatial_hole():
    y,x=np.meshgrid(np.arange(5.),np.arange(5.),indexing='ij')
    ds=xr.Dataset({'v':(('lat','lon'),280+x+y,dict(units='K',data_kind='continuous',spatial_semantics='grid_estimate'))},
        coords=dict(lat=np.arange(5.)+30,lon=np.arange(5.)+110),attrs=dict(crs='EPSG:4326'))
    ds.v.values[2,2]=np.nan;before=ds.copy(deep=True)
    with session():out,r=recover(ds,'grid')
    assert np.isfinite(out.v.values).all() and r['repairs']['v']['filled']==1
    assert r['repairs']['v']['methods'][0]['method']=='reconstruction.grid_gap_linear'
    xr.testing.assert_identical(ds,before)
    ds.v.values[1:4,1:4]=np.nan
    with session():out,r=recover(ds,'grid')
    assert not r['repairs'] and np.isnan(out.v.values).sum()==9


def test_source_configuration_is_evidence_but_target_units_are_not():
    from ecodata.core.recovery import configuration_document
    ds=data();del ds.v.attrs['units']
    declared=configuration_document(dict(fields=dict(output=dict(source='v',source_units='K'))),ds)
    with session():out,r=recover(ds,'A',docs=[declared])
    assert out.v.attrs['units']=='K'
    target_only=configuration_document(dict(fields=dict(output=dict(source='v',units='K'))),ds)
    with session():out,r=recover(ds,'A',docs=[target_only])
    assert 'units' not in out.v.attrs


@pytest.mark.parametrize('unit',['ns','us'])
def test_repaired_product_time_bounds_encoding_is_exact_and_shared(unit):
    from ecodata.core.recovery import serialize
    ds=data().assign_coords(time=data().time.values.astype('datetime64['+unit+']'))
    ds['time_bounds']=(('time','b'),np.column_stack([ds.time.values,ds.time.values+np.timedelta64(1,'h')]))
    ds.time.attrs['bounds']='time_bounds'
    raw=serialize(ds)
    with xr.open_dataset(io.BytesIO(raw)) as out:
        np.testing.assert_array_equal(out.time,ds.time)
        np.testing.assert_array_equal(out.time_bounds,ds.time_bounds)
    with xr.open_dataset(io.BytesIO(raw),decode_times=False) as out:
        assert out.time.attrs['units']==out.time_bounds.attrs.get('units',out.time.attrs['units'])
        np.testing.assert_array_equal(out.time_bounds.values[:,0],out.time.values)


def test_camx_public_conversion_repairs_local_csv_gap(tmp_path):
    """Extended, labelled synthetic format fixture; no environmental claim."""
    from ecodata.algorithms.automatic_preparation import prepare_uploads
    base=Path(__file__).parents[1]/'examples/ai_round1/camx/emissions'
    cfg=json.loads((base/'config_LATLON.json').read_text())
    old=pd.read_csv(base/'A_synthetic_LATLON_hourly.csv')
    grid=old[old.timestamp==old.timestamp.iloc[0]].copy()
    frames=[]
    for n in range(20):
        f=grid.copy();f.timestamp=(pd.Timestamp(old.timestamp.iloc[0])+pd.Timedelta(hours=n)).isoformat()
        f.nitric_oxide=f.nitric_oxide+n;frames.append(f)
    frame=pd.concat(frames,ignore_index=True);frame.loc[30,'nitric_oxide']=np.nan
    raw=frame.to_csv(index=False).encode()
    request=dict(schema='ecodata.model-task.v1',model_id='camx',operation='emissions',input='data.csv',configuration=cfg)
    metadata=dict(resource='data.csv',variables=dict(nitric_oxide=dict(units='mol/s',data_kind='continuous',temporal_semantics='mean',interval_seconds=3600,time_basis='interval_start_hourly')))
    def item(name,raw):b=io.BytesIO(raw);b.name=name;return b
    result=prepare_uploads([item('data.csv',raw),item('request.json',json.dumps(request).encode()),item('metadata.json',json.dumps(metadata).encode())],'camx',tmp_path)
    assert result['manifest']['status']=='COMPONENT_EXPORTED'
    assert sum(r['filled'] for e in result['automatic_recovery']['events'] for r in e.get('repairs',{}).values())==1
    with zipfile.ZipFile(result['archive']) as z:
        for info in result['manifest']['output_inventory']:
            if info['path'].endswith('.nc'):
                with xr.open_dataset(io.BytesIO(z.read(info['path'])),decode_times=False) as d:assert np.isfinite(d['NO']).all()
    assert frame.nitric_oxide.isna().sum()==1


def test_small_nonfinite_value_is_estimated_and_original_is_preserved():
    ds=data();ds.v.values[5]=np.inf
    with session():out,r=recover(ds,'A')
    assert out.v.values[5]==285 and out.v__recovered.values[5]==1
    assert np.isinf(ds.v.values[5])


def test_previously_recovered_primary_values_are_not_evaluation_truth():
    from ecodata.algorithms.platform_search import SearchInput,parse_inputs
    ds=data();ds.v.values[5]=np.nan
    with session():repaired,_=recover(ds,'A')
    assert repaired.v.values[5]==285
    parsed,_=parse_inputs([SearchInput('A.nc','primary',bytes(repaired.to_netcdf(engine='h5netcdf')))])
    assert np.isnan(parsed[0][1].v.values[5]) and np.isfinite(parsed[0][1].v.values).sum()==19


def test_ds_uncertainty_does_not_veto_explicit_labelled_working_declaration():
    from ecodata.core.recovery import descriptions
    ds=data();del ds.v.attrs['units']
    doc=dict(source='declared.json',text=json.dumps(dict(status='WORKING_ASSUMPTION_ONLY',variables=dict(v=dict(units='K')))))
    with session(advisor=lambda facts:(dict(use_fact_ids=[]),{})) as s:
        out,r=recover(ds,'A',docs=[doc])
        notes=descriptions(s.report())
    assert out.v.attrs['units']=='K' and r['metadata'][0]['ds_selected'] is False
    assert any(n['status']=='按工作假设补齐' for n in notes)


def test_persistent_no_data_domain_is_not_spatially_fabricated():
    v=np.ones((20,5,5))*280;v[:,0,0]=np.nan;v[5,2,2]=np.nan
    ds=xr.Dataset({'v':(('time','lat','lon'),v,dict(units='K',temporal_semantics='instantaneous',spatial_semantics='grid_estimate',data_kind='continuous'))},
        coords=dict(time=pd.date_range('2020-01-01',periods=20,freq='h'),lat=np.arange(5.)+30,lon=np.arange(5.)+110),attrs=dict(crs='EPSG:4326'))
    with session():out,r=recover(ds,'grid')
    assert np.isnan(out.v.values[:,0,0]).all() and out.v.values[5,2,2]==280


def test_unknown_unit_is_not_used_for_temporal_repair():
    ds=data();ds.v.attrs['units']='UNKNOWN';ds.v.values[5]=np.nan
    with session():out,r=recover(ds,'A')
    assert np.isnan(out.v.values[5]) and not r['repairs']
