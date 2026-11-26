"""Shared registered codec at the final table gate, independent of model."""
import csv
import pytest
from ecodata.desktop.release_gate import table_review
from ecodata.desktop.sample_values import field_values


@pytest.mark.parametrize('name,column,unit,cells',[
    ('TMP.air_temperature','TMP','degC',['+0200,1','+9999,9','+0250,2']),
    ('WND.wind_speed','WND','m/s',['090,1,N,0030,1','999,9,9,9999,9','090,1,N,0010,2']),
    ('SLP.air_pressure_at_mean_sea_level','SLP','hPa',['10123,1','99999,9','10000,2'])])
def test_runtime_uses_the_same_quality_aware_codec(tmp_path,name,column,unit,cells):
    path=tmp_path/'observations.csv'
    with path.open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['STATION','DATE',column])
        for i,cell in enumerate(cells):writer.writerow(['station',f'2020-01-01T0{i}:00:00',cell])
    field=dict(name=name,source_column=column,unit=unit,noaa_component=True)
    row=dict(name=path.name,format='CSV',ref=(path,'',None,None),fields=[field],bytes=path.stat().st_size)
    report=table_review(row,field,tmp_path/'gate')
    assert report['records']==3 and report['missing']==2
    assert report['missing']==sum(v is None for v in field_values(cells,field))


def test_unknown_compound_values_do_not_become_valid_numbers(tmp_path):
    path=tmp_path/'unknown.csv';path.write_text('value\n"+0200,1"\n',encoding='utf8')
    field=dict(name='value',unit='degC')
    row=dict(name=path.name,format='CSV',ref=(path,'',None,None),fields=[field],bytes=path.stat().st_size)
    with pytest.raises(ValueError,match='没有有效数值'):table_review(row,field,tmp_path/'gate')


@pytest.mark.parametrize('name',['station','STATION','site_no','station_id'])
def test_numeric_station_identity_survives_shared_recognition(name):
    from ecodata.recognition import recognise_file
    from ecodata.io.structures import table_frame
    raw=f'{name},value\n072520094823,280\n72520094823,281\n'.encode()
    data,report=recognise_file('arbitrary.csv',raw)
    assert table_frame(data)[name].tolist()==['072520094823','72520094823']
    assert report.semantic_types[name]=='string_identifier'


def test_broad_quantity_alias_cannot_mix_speed_with_vector_component():
    from ecodata.desktop.station_grid import task_for
    component=dict(name='u10',dimensions=['TSTEP','LAY','ROW','COL'],unit='m/s',long_name='10 m wind component')
    speed=dict(name='WND.wind_speed',standard_name='wind_speed',unit='m/s',noaa_component=True)
    columns=[dict(name=n) for n in ['STATION','DATE','LATITUDE','LONGITUDE']]+[speed]
    scan=dict(rows=[dict(format='NetCDF'),dict(format='CSV',fields=columns)])
    pair=((0,'D1.F1',component),(1,'D2.F5',speed))
    with pytest.raises(ValueError,match='风分量不能作为标量风速'):
        task_for(scan,pair,'fusion',dict(status='equivalent',quantity='wind_velocity'))


def test_comparison_advice_does_not_erase_explicit_gain_evaluation(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from ecodata.desktop import discovery,catalog_decision
    from ecodata.desktop.scan import Scanner
    times=pd.date_range('2020-01-01',periods=24,freq='h')
    xr.Dataset({'temperature':(('time','lat','lon'),np.full((24,2,2),280.),dict(units='K',standard_name='air_temperature'))},
        coords=dict(time=times,lat=[1.,2.],lon=[1.,2.]),attrs=dict(crs='EPSG:4326')).to_netcdf(tmp_path/'grid.nc')
    pd.DataFrame(dict(station_id='s',time_utc=times.astype(str)+'Z',latitude=1.5,longitude=1.5,**{'air_temperature (K)':282.})).to_csv(tmp_path/'station.csv',index=False)
    scan,_=Scanner().scan(tmp_path);summary,_=discovery.discover(scan,'data_only',task_goal='evaluate_gain')
    originals,details=catalog_decision.local_fields(scan,summary)
    fields=[v for v in originals.values() if v[2]['name'] in {'temperature','air_temperature'}]
    assert len(fields)==2
    pair,local=discovery.describe_pair(scan,'C1',*fields,'data_only',details)
    reply=dict(decisions=[dict(candidate='C1',action='compare_station_grid',reason='alignment advice')])
    checked,gated=discovery.check(scan,{'C1':local},reply,dict(summary,candidates=[pair]))
    assert ('C1','fusion') in gated and gated['C1','fusion']['route']=='station_grid',checked
    assert any(r['action']=='fusion' and r['status']=='通过' for r in checked)
    _,comparison=discovery.check(scan,{'C1':local},reply,dict(summary,task_goal='grid_comparison',candidates=[pair]))
    assert ('C1','fusion') not in comparison


@pytest.mark.parametrize('station',['72520094823','072520094823'])
def test_numeric_station_product_binds_back_to_actual_native_indices(tmp_path,station):
    # Synthetic adapter regression, not environmental accuracy evidence.
    import numpy as np
    import pandas as pd
    import xarray as xr
    from ecodata.desktop import station_grid,local_search
    clock=pd.date_range('2020-01-01',periods=24,freq='h')
    source=tmp_path/'grid.nc'
    xr.Dataset({'temperature':(('time','lat','lon'),np.full((24,2,2),280.),dict(units='K'))},
        coords=dict(time=clock,lat=[30.,31.],lon=[110.,111.])).to_netcdf(source)
    scan=dict(rows=[dict(ref=(source,'',None,None),containers=[],name=source.name,format='NetCDF',
        bytes=source.stat().st_size,fields=[dict(name='temperature',dimensions=['time','lat','lon'])]),dict(format='CSV')])
    task=dict(route='station_grid',grid_index=0,table_index=1,grid_name='temperature',
        table_name='temperature',table_header='temperature',quantity='air_temperature',unit='K',obs_unit='K',
        recipes=[[station_grid.EXACT,station_grid.MATCH,station_grid.BIAS]])
    data=pd.DataFrame(dict(station_id=station,time_utc=clock.astype(str)+'Z',longitude=110.5,latitude=30.5,temperature=282.))
    frame=station_grid.load(scan,task,table_data=data)[station_grid.EXACT];frame['unit']='K'
    request=dict(task='temporal',adapter='paired_station_calibration',variable='air_temperature',unit='K')
    result=local_search.run_paired(scan,'C1:fusion',task,frame,request,tmp_path/'run',None,{},source_indices=(0,1))
    update,=result['native_updates']
    assert update['task']=='C1:fusion' and update['variable']=='temperature' and update['points']
    assert result['original_report'][0]['mutation_search']['station']==station
    for point in update['points']:
        assert len(point['index'])==3 and point['expected']==280.
