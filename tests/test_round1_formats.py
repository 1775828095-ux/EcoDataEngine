"""Scope, invalid-data and scientific-semantic checks for new converters."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.calmet_surface import export_surface
from ecodata.models.adapters.swat_weather import export_weather
from ecodata.models.adapters.wps_sst import export_sst
from ecodata.round1 import convert
from examples.ai_round1.independent import surf,swat,wps

CASES=Path(__file__).resolve().parents[1]/'examples/ai_round1'

def cal():
    c=CASES/'calpuff'
    f=pd.read_csv(c/'A_synthetic_complete.csv')
    for name in f.select_dtypes('number').columns:f[name]=f[name].astype(float)
    return f,json.loads((c/'config_synthetic.json').read_text())

def test_calmet_known_units_time_zone_and_leap_boundary(tmp_path):
    frame,cfg=cal();frame.wind_speed=36;cfg['units']['wind_speed']='km/h'
    frame.temperature=68;cfg['units']['temperature']='degF'
    frame.station_pressure=101000;cfg['units']['station_pressure']='Pa'
    data,report=export_surface(frame,cfg);path=tmp_path/'SURF.DAT';path.write_bytes(data['SURF.DAT']);parsed=surf(path)
    assert parsed['times'][0].isoformat()=='2020-02-28T23:00:00'
    assert parsed['times'][1].isoformat()=='2020-02-29T00:00:00'
    np.testing.assert_allclose(parsed['values'][:,:,0],10,rtol=0,atol=1e-6)
    np.testing.assert_allclose(parsed['values'][:,:,4],293.15,rtol=0,atol=1e-6)
    np.testing.assert_allclose(parsed['values'][:,:,6],1010,rtol=0,atol=1e-6)
    assert not report['complete_model_input']

@pytest.mark.parametrize('problem',['gap','duplicate','fractional_station','missing_zone','bad_time','fractional_zone','wrong_version','unknown_station','bad_unit','missing_field','alias_collision','infinity','negative_wind','invalid_rh','sentinel_as_measurement','fractional_code'])
def test_calmet_rejects_invalid_semantics_without_silent_repair(problem):
    f,c=cal()
    if problem=='gap':f=f.drop(index=1)
    elif problem=='duplicate':f=pd.concat([f,f.iloc[[0]]])
    elif problem=='fractional_station':f.loc[0,'station_id']=12345.5
    elif problem=='missing_zone':f['time']=f.time.str.replace('+00:00','',regex=False)
    elif problem=='bad_time':f.loc[0,'time']='2021-02-29T04:00:00Z'
    elif problem=='fractional_zone':c['timezone_west_hours']=5.5
    elif problem=='wrong_version':c['version']='CALMET6'
    elif problem=='unknown_station':f.loc[0,'station_id']=2
    elif problem=='bad_unit':c['units']['station_pressure']='unknown'
    elif problem=='missing_field':f=f.drop(columns='station_pressure')
    elif problem=='alias_collision':c['columns']['temperature']='wind_speed'
    elif problem=='infinity':f.loc[0,'temperature']=np.inf
    elif problem=='negative_wind':f.loc[0,'wind_speed']=-1
    elif problem=='invalid_rh':f.loc[0,'relative_humidity']=101
    elif problem=='sentinel_as_measurement':f.loc[0,'temperature']=9999
    elif problem=='fractional_code':f.loc[0,'precip_code']=.5
    with pytest.raises(SchemaError):export_surface(f,c)

def test_calmet_missing_requires_explicit_policy_and_mask_is_preserved(tmp_path):
    f,c=cal();f.loc[0,'temperature']=np.nan
    with pytest.raises(SchemaError):export_surface(f,c)
    c['missing_policy']='preserve';files,report=export_surface(f,c)
    path=tmp_path/'SURF.DAT';path.write_bytes(files['SURF.DAT'])
    assert surf(path)['values'][0,0,4]==9999 and report['missing_counts']['temperature']==1

def test_calmet_multiple_stations_follow_configured_order(tmp_path):
    f,c=cal();other=f.copy();other.station_id=12346;other.wind_speed=7
    c['station_ids']=[12346,12345];files,_=export_surface(pd.concat([f,other]),c)
    path=tmp_path/'SURF.DAT';path.write_bytes(files['SURF.DAT']);parsed=surf(path)
    np.testing.assert_array_equal(parsed['values'][:,:,0],np.tile([7,3],(4,1)))

@pytest.mark.parametrize('encoding',['utf-8-sig','gb18030'])
def test_csv_encoding_and_chinese_column_mapping(encoding):
    f,c=cal();f=f.rename(columns={'temperature':'气温'});c['columns']['temperature']='气温';c['encoding']=encoding
    files,_=convert('calpuff',f.to_csv(index=False).encode(encoding),c)
    assert files['SURF.DAT'].startswith(b'SURF.DAT')

def sw():
    f=pd.DataFrame({'date':['2020-02-28','2020-02-29','2020-03-01'],'precipitation':[1.,0.,2.],'tmax':[68.,50.,59.],'tmin':[32.,14.,41.]})
    c=json.loads((CASES/'swat/config.json').read_text());c['missing_policy']='reject'
    return f,c

def test_swat_units_and_fixed_positions_with_leap_day(tmp_path):
    f,c=sw();files,_=export_weather(f,c)
    for name,b in files.items():(tmp_path/name).write_bytes(b)
    parsed=swat(tmp_path/'weather.pcp','pcp');temps=swat(tmp_path/'weather.tmp','tmp')
    np.testing.assert_allclose(parsed['values'][:,0],[25.4,0,50.8],rtol=0,atol=1e-12)
    np.testing.assert_allclose(temps['values'],[[20,0],[10,-10],[15,5]],rtol=0,atol=1e-12)
    assert parsed['dates'][1].day==29

@pytest.mark.parametrize('problem',['gap','duplicate','swat_plus','unknown_crs','latitude','negative_precip','temperature_order','precip_overflow','unknown_unit','missing','invalid_date'])
def test_swat_rejects_unrepresentable_or_mixed_branch_inputs(problem):
    f,c=sw()
    if problem=='gap':f=f.drop(index=1)
    elif problem=='duplicate':f.loc[1,'date']=f.date.iloc[0]
    elif problem=='swat_plus':c['version']='SWAT+'
    elif problem=='unknown_crs':c['station']['crs']='EPSG:3857'
    elif problem=='latitude':c['station']['latitude']=91
    elif problem=='negative_precip':f.loc[0,'precipitation']=-1
    elif problem=='temperature_order':f.loc[0,'tmax']=20
    elif problem=='precip_overflow':f.loc[0,'precipitation']=100
    elif problem=='unknown_unit':c['units']['tmax']='kelvinish'
    elif problem=='missing':f.loc[0,'precipitation']=np.nan
    elif problem=='invalid_date':f.loc[0,'date']='2021-02-29'
    with pytest.raises(SchemaError):export_weather(f,c)

def test_swat_generator_is_explicit_and_not_zero_fill(tmp_path):
    f,c=sw();f.loc[0,'precipitation']=np.nan;c['missing_policy']='swat_generator'
    files,report=export_weather(f,c);path=tmp_path/'weather.pcp';path.write_bytes(files['weather.pcp'])
    assert swat(path,'pcp')['values'][0,0]==-99 and report['missing_counts']['precipitation']==1


def test_swat_hourly_to_daily_requires_explicit_aggregation(tmp_path):
    times=pd.date_range('2020-01-01',periods=48,freq='h',tz='UTC')
    frame=pd.DataFrame({
        'time':times,
        'precipitation':np.concatenate([np.ones(24),np.ones(24)*2]),
        'tmax':np.concatenate([np.linspace(1,10,24),np.linspace(2,20,24)]),
        'tmin':np.concatenate([np.linspace(-10,0,24),np.linspace(-5,5,24)]),
    })
    cfg=sw()[1]
    cfg['units']={'precipitation':'mm','tmax':'degC','tmin':'degC'}
    cfg.update({'source_frequency':'hourly','source_time_zone':'UTC','time_column':'time',
                'aggregation':{'precipitation':'sum','tmax':'max','tmin':'min'}})
    files,report=export_weather(frame,cfg)
    for name,content in files.items():(tmp_path/name).write_bytes(content)
    assert report['aggregation_status']=='AGGREGATED'
    parsed=swat(tmp_path/'weather.pcp','pcp')
    np.testing.assert_allclose(parsed['values'][:,0],[24,48],rtol=0,atol=1e-12)
    temps=swat(tmp_path/'weather.tmp','tmp')
    np.testing.assert_allclose(temps['values'],[[10,-10],[20,-5]],rtol=0,atol=1e-12)


def test_swat_hourly_missing_hour_is_rejected():
    times=pd.date_range('2020-01-01',periods=25,freq='h',tz='UTC').delete(12)
    frame=pd.DataFrame({'time':times,'precipitation':1.,'tmax':10.,'tmin':0.})
    cfg=sw()[1]
    cfg.update({'source_frequency':'hourly','source_time_zone':'UTC','time_column':'time',
                'aggregation':{'precipitation':'sum','tmax':'max','tmin':'min'}})
    with pytest.raises(SchemaError,match='缺小时'):
        export_weather(frame,cfg)

def wf():
    c=json.loads((CASES/'wrf/config.json').read_text())
    ds=xr.Dataset({'sst':(('time','lat','lon'),np.array([[[1,2],[3,4]]],dtype=float))},coords={'time':pd.to_datetime(['2020-01-01T12:00:00']),'lat':[30.,31.],'lon':[300.,301.]})
    ds.sst.attrs['units']='Celsius'
    return ds,c

def test_wps_descending_grid_is_reordered_without_transposition(tmp_path):
    ds,c=wf();ds=ds.sortby('lat',ascending=False).sortby('lon',ascending=False)
    files,_=export_sst(ds,c);path=tmp_path/'sample.wps';path.write_bytes(next(iter(files.values())));parsed=wps(path)
    np.testing.assert_allclose(parsed['values'],np.array([[1,2],[3,4]])+273.15,rtol=0,atol=2e-5)
    np.testing.assert_allclose(parsed['projection'][:4],[30,300,1,1],rtol=0,atol=1e-6)

@pytest.mark.parametrize('problem',['projection','unit','missing','layer','duplicates','irregular','bad_time','missing_day','nonmonotonic','no_time_basis'])
def test_wps_rejects_silent_loss_and_bad_grid(problem):
    ds,c=wf()
    if problem=='projection':c['crs']='EPSG:3857'
    elif problem=='unit':ds.sst.attrs['units']='unknown'
    elif problem=='missing':ds.sst.values[0,0,0]=np.nan
    elif problem=='layer':ds['sst']=ds.sst.expand_dims(layer=[1,2])
    elif problem=='duplicates':ds=ds.assign_coords(lat=[30.,30.])
    elif problem=='irregular':ds=ds.reindex(lat=[30.,31.,33.],fill_value=5)
    elif problem=='bad_time':ds=ds.assign_coords(time=pd.to_datetime(['2020-01-01T12:30:00']))
    elif problem=='missing_day':
        values=np.stack([ds.sst.values[0],ds.sst.values[0]+1])
        ds=xr.Dataset({'sst':(('time','lat','lon'),values)},coords={'time':pd.to_datetime(['2020-01-01T00:00:00','2020-01-03T00:00:00']),'lat':[30.,31.],'lon':[300.,301.]})
        ds.sst.attrs['units']='Celsius'
    elif problem=='nonmonotonic':ds=ds.reindex(lat=[31.,30.,32.],fill_value=5)
    elif problem=='no_time_basis':c.pop('time_basis')
    with pytest.raises(SchemaError):export_sst(ds,c)

def test_all_independent_case_comparisons_meet_predeclared_thresholds():
    report=json.loads((CASES/'round1_results.json').read_text(encoding='utf-8'))
    assert not report['errors'] and set(report['cases'])=={'calpuff','swat','wrf','camx','cmaq'}
    comparisons=[]
    def walk(value):
        if isinstance(value,dict):
            if 'threshold_atol' in value:comparisons.append(value['passed'])
            for v in value.values():walk(v)
    walk(report['cases'])
    assert comparisons and all(comparisons)
    assert report['cases']['calpuff']['reference']['missing_mask_exact']
    assert report['cases']['swat']['dates_exact']
    assert report['cases']['camx']['reaction_labels_exact']
    assert report['cases']['camx']['distinct_conditions']==75
    assert report['cases']['camx']['unique_heights']==11
    assert report['cases']['camx']['record_count']==189750
