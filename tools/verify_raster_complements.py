"""Development-only real-data interface acceptance, not an effect experiment."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
from copy import deepcopy
import hashlib,json
import numpy as np
import xarray as xr
from scipy.signal import convolve2d
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.capabilities import describe_algorithm
from ecodata.algorithms.numerical_search import save

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/complementary_library_20261029'
OLD=ROOT/'output/halfdate_reconstruction_20261028'
IDS=['bias.xsdba_scaling','spatial.denoise_tv','spatial.denoise_bilateral','spatial.wavelet_lowpass','fusion.wavelet_detail_transfer']
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def sources(spec):return [api.SearchInput(Path(p['path']).name,p['role'],Path(p['path']).read_bytes()) for p in spec]


def run():
    OUT.mkdir(parents=True,exist_ok=True);registry=api.platform_registry()
    save(OUT/'algorithm_profiles.json',[dict(spec=registry.get(k).spec.as_dict(),capability=describe_algorithm(registry.get(k).spec).as_dict(),
        empirical_gain='UNKNOWN',ranking_bonus=None) for k in IDS])
    cfg=load(OLD/'cases/patch0_alternating/config_development.json')
    with xr.open_dataset(OLD/'data/patch0_all_reference.nc') as ds:training=ds.ndvi.sel(time=slice('2013','2014')).load()
    good=np.isfinite(training.values).sum(0)>=2
    window=None
    for size in [16,12,8]:
        candidates=np.argwhere(convolve2d(good.astype(int),np.ones((size,size),int),mode='valid')==size*size)
        if len(candidates):
            centre=(np.asarray(good.shape)-size)/2
            y,x=candidates[np.argmin(((candidates-centre)**2).sum(1))];window=(int(y),int(x),size);break
    if window is None:raise ValueError('Insufficient training-only support; do not fill training truth')
    y,x,size=window;training=training.isel(lat=slice(y,y+size),lon=slice(x,x+size))
    with xr.open_dataset(cfg['inputs'][0]['path']) as ds:fine=ds.isel(lat=slice(y,y+size),lon=slice(x,x+size)).load()
    with xr.open_dataset(cfg['inputs'][1]['path']) as ds:coarse=ds.load()
    with xr.open_dataset(OLD/'data/patch0_coarse.nc') as ds:historical=ds.sel(time=slice('2013','2014')).load()
    folder=OUT/'real_2015';folder.mkdir(exist_ok=True)
    fine.to_netcdf(folder/'fine_visible_2015.nc',engine='h5netcdf')
    coarse.to_netcdf(folder/'coarse_2015.nc',engine='h5netcdf')
    historical.to_netcdf(folder/'coarse_prior_2013_2014.nc',engine='h5netcdf')
    prior_request=deepcopy(cfg['request']);prior_request.pop('protect_observed_source',None)
    prior_request.update(target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist(),target_times=[str(t) for t in training.time.values])
    prior_request['parameters'].update(raster_target_times=prior_request['target_times'])
    prior_graph=dict(records={'s0':dict(operation='source',parents=[],parameters=dict(source='source0')),
        't':dict(operation='temporal.raster_linear',parents=['s0'],parameters={}),
        'g':dict(operation='reconstruction.grid_gap_nearest',parents=['t'],parameters={}),
        'u':dict(operation='spatial.bilinear',parents=['g'],parameters={})},key='u')
    prior_input=[dict(path=str(folder/'coarse_prior_2013_2014.nc'),role='primary')]
    save(folder/'prior_recipe.json',dict(inputs=prior_input,request=prior_request,graph=prior_graph))
    prior=api.execute_frozen_graph(sources(prior_input),prior_request,prior_graph,folder/'prior_interpolation')
    with xr.open_dataset(prior['product']) as ds:hist=ds.ndvi.load()
    training.attrs=dict(training.attrs,split='train');hist.attrs=dict(hist.attrs,split='train')
    training.to_dataset().to_netcdf(folder/'training_reference.nc',engine='h5netcdf')
    hist.to_dataset().to_netcdf(folder/'historical_simulation.nc',engine='h5netcdf')
    request=deepcopy(cfg['request']);request.update(allow_training=True,target_lat=fine.lat.values.tolist(),target_lon=fine.lon.values.tolist())
    request['parameters'].update(calibration_source_id=coarse.attrs['source_id'])
    inputs=[dict(path=str(folder/'fine_visible_2015.nc'),role='primary'),dict(path=str(folder/'coarse_2015.nc'),role='background'),
        dict(path=str(folder/'training_reference.nc'),role='training_reference'),dict(path=str(folder/'historical_simulation.nc'),role='historical_simulation')]
    records={f's{i}':dict(operation='source',parents=[],parameters=dict(source=f'source{i}')) for i in [0,1]}
    records.update(ft=dict(operation='temporal.raster_linear',parents=['s0'],parameters={}),
        fg=dict(operation='reconstruction.grid_gap_nearest',parents=['ft'],parameters={}),
        tv=dict(operation='spatial.denoise_tv',parents=['fg'],parameters=dict(noise_multiplier=1.)),
        ct=dict(operation='temporal.raster_linear',parents=['s1'],parameters={}),
        cg=dict(operation='reconstruction.grid_gap_nearest',parents=['ct'],parameters={}),
        up=dict(operation='spatial.bilinear',parents=['cg'],parameters={}),
        cal=dict(operation='bias.xsdba_scaling',parents=['up'],parameters={}),
        bilateral=dict(operation='spatial.denoise_bilateral',parents=['cal'],parameters=dict(noise_multiplier=1.,sigma_spatial_pixels=2.)),
        low=dict(operation='spatial.wavelet_lowpass',parents=['bilateral'],parameters=dict(wavelet_level=1)),
        fuse=dict(operation='fusion.wavelet_detail_transfer',parents=['low','tv'],parameters=dict(wavelet_level=1,detail_index=1,detail_strength=1.)),
        physical=dict(operation='constraint.declared_range_projection',parents=['fuse'],parameters={}))
    graph=dict(records=records,key='physical');save(folder/'mechanism_config.json',dict(inputs=inputs,request=request,graph=graph,
        purpose='Fixed wiring solely for interface acceptance; not an optimizer pool or proposed winning scientific route'))
    result=api.execute_frozen_graph(sources(inputs),request,graph,folder/'deployment')
    with xr.open_dataset(result['product']) as opened:
        product=opened.load();assert np.isfinite(product.ndvi).all()
        known=np.isfinite(fine.ndvi.values);restored=product.ndvi.sel(time=fine.time).values
        np.testing.assert_array_equal(restored[known],fine.ndvi.values[known])
    save(OUT/'real_data_acceptance.json',dict(status='PASS',development_year=2015,training_years=[2013,2014],
         window=dict(lat_index=y,lon_index=x,size=size,selected_on='Prior valid-pair counts and geometry only'),
         new_algorithms_executed=IDS,reference_used_for_selection=False,formal_test_labels_read=False,
         prior_registered_calls=prior['cost']['used'],deployment_registered_calls=result['cost']['used'],
         output=str(Path(result['product'])),shape=list(product.ndvi.shape),all_output_finite=True,visible_observations_unchanged=True,
         scientific_advantage='NOT_EVALUATED',purpose='Real-input public API callability and connection acceptance only'))
    save(OUT/'input_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.nc')})
    print('REAL DEVELOPMENT ACCEPTANCE PASS',size,list(product.ndvi.shape),result['cost']['used'],flush=True)


if __name__=='__main__':run()
