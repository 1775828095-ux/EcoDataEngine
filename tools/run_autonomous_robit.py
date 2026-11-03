"""Public-entry autonomous Robit observation, no algorithm whitelist or plans."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import run_robit_single_source as old
from ecodata.algorithms import platform_search as api
from ecodata.models.adapters.swat_plus import SWATPlusAdapter

OUT=ROOT/'output/autonomous_decision_20261027'


def main():
    weather=old.raw_weather();protocol=json.loads((old.OUT/'protocol.json').read_text())
    parts=[]
    for case in protocol['cases']:
        ds,_,_=old.mask_case(weather,case['validation_start'],case['kind'],protocol['mask_seed']);parts.append(ds)
    source=xr.concat(parts,dim='time').sortby('time')
    path=OUT/'input.nc';source.to_netcdf(path,engine='h5netcdf')
    req=dict(task='temporal',variable='precipitation',unit='mm/day',semantic_type='continuous',nonnegative=True,
        spatial_semantics='point',temporal_semantics='accumulation',target_lat=[11.6],target_lon=[37.36],
        parameters=dict(target_time_frequency='1D'),model_profile=dict(id='swat_plus',role='weather',field='precipitation'))
    config=dict(inputs=[dict(path=str(path),role='primary'),dict(path=str(old.OUT/'training_state.nc'),role='state')],
        request=req,mode='Autonomous',budget=dict(objective_limit=80,proposal_limit=3000,wall_seconds=180),seed=11)
    (OUT/'file_request.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    run_id=sys.argv[1] if len(sys.argv)>1 else 'run'
    result=api.submit_file_request(config,OUT/run_id)
    print(json.dumps({k:result.get(k) for k in ['status','registry_algorithms','available_registry_algorithms','attempts','executed_candidates','failed_candidates','kernel_calls','unique_working_states','stop_reason','certification','selected_key','total_seconds']},indent=2),flush=True)
    if not result.get('product'):raise ValueError('No product; preserved failures must be inspected')
    with xr.open_dataset(result['product']) as file:product=file.load()
    values=product.precipitation.values[:,0,0];visible=source.precipitation.values[:,0,0]
    np.testing.assert_array_equal(values[np.isfinite(visible)],visible[np.isfinite(visible)])
    assert np.isfinite(values).all() and (values>=0).all()
    frame=weather[weather.time.isin(pd.to_datetime(product.time.values))].copy();frame['precipitation']=values
    original=weather.set_index('time').precipitation.reindex(frame.time).to_numpy();missing=~np.isfinite(visible)
    error=values[missing]-original[missing]
    diagnostic=dict(label='Previously used years, exploratory readback only; never a new blind test',
        RMSE=float(np.sqrt(np.mean(error*error))),MAE=float(abs(error).mean()),n=int(missing.sum()),
        selected_before_reading_reference=True)
    output=OUT/('weather_output_'+run_id)
    SWATPlusAdapter().save_package(output,dict(project_name='Robit_autonomous',start='2002-01-01',end='2007-12-31',
        station=dict(name='Robit',latitude=11.6,longitude=37.36,elevation=1790),weather=frame.to_dict('list'),
        target_model_version='62.0.0',additional_weather=['solar_radiation','relative_humidity','wind_speed']))
    reread=pd.read_csv(output/'station.pcp',sep=r'\s+',skiprows=3,header=None)
    np.testing.assert_allclose(reread[2],values,rtol=0,atol=.00051)
    handoff=OUT/('weather_handoff_'+run_id);handoff.mkdir()
    for file in output.iterdir():
        if file.name.startswith('station.') or file.name in ['pcp.cli','tmp.cli','slr.cli','hmd.cli','wnd.cli']:
            (handoff/file.name).write_bytes(file.read_bytes())
    (handoff/'README.md').write_text('2002—2007年单源降水自主重建输出。只接入天气，保留队友file.cio、time.sim、WGN及拓扑；由队友映射站点。无原生SWAT运行，不是新盲测。',encoding='utf-8')
    diagnostic.update(weather_readback='PASS',raw_observations_preserved=True,native_SWAT_run=False,
        frozen_request_sha256=hashlib.sha256((OUT/'file_request.json').read_bytes()).hexdigest())
    (OUT/('diagnostic_'+run_id+'.json')).write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    print('Weather handoff and raw-observation protection PASS',diagnostic,flush=True)


if __name__=='__main__':main()
