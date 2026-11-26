"""Bind authorized current observations to existing residual/ratio state kernels.

This is a state adapter, not an algorithm selector or a learned estimator.
Retrospective concurrent analysis is explicit; no forecast-latency claim is made.
"""
import hashlib
import numpy as np
import pandas as pd
from pyproj import CRS
from ecodata.algorithms.quality_aware import fit_ratio_frames
from ecodata.fusion.point_weights import distance_weights,weighted_values
from ecodata.fusion.residual import residual_difference


def conditioning_pairs(prior,background,observations,variable,policy):
    if not isinstance(policy,dict) or policy.get('mode')!='contemporaneous_analysis' or not policy.get('evidence'):
        raise ValueError('Observation update requires explicit concurrent-analysis permission and evidence; forecasts are not implied')
    if prior.attrs.get('calibration_source_id')!=background.attrs.get('source_id'):
        raise ValueError('Observation update prior/background identity mismatch')
    field=background[variable]
    for name in ['affine_slope','affine_intercept']:
        if name not in prior or prior[name].ndim or not np.isfinite(prior[name]):raise ValueError('Finite scalar historical affine state required')
    if prior.attrs.get('unit')!=field.attrs.get('units'):raise ValueError('Historical state unit conflict')
    if not prior.attrs.get('training_end'):raise ValueError('Historical fitting end required')
    if field.attrs.get('spatial_semantics')!='point':raise ValueError('Observation-state target requires declared point support; use existing spatial adapters for grids')
    if field.dims!=('time','lat','lon') or field.shape[1:]!=(1,1):
        raise ValueError('Observation-state adapter currently binds one explicit target point series')
    if not background.attrs.get('crs') or not CRS.from_user_input(background.attrs['crs']).equals(CRS.from_epsg(4326)):
        raise ValueError('Observation-state distances require explicit EPSG:4326')
    frame=observations.copy()
    required={'station_id','time','value','background','unit','background_unit','background_product',
        'variable','longitude','latitude','split','qc_flag','temporal_semantics','crs'}
    if not required<=set(frame):raise ValueError('Conditioning observations missing '+str(sorted(required-set(frame))))
    if any(not CRS.from_user_input(c).equals(CRS.from_epsg(4326)) for c in frame.crs.unique()):raise ValueError('Conditioning CRS conflict')
    if frame.empty:raise ValueError('Empty conditioning resource is not an observation update')
    if not frame.split.eq('train').all():raise ValueError('State updates cannot read validation/test labels')
    if frame.station_id.isna().any():raise ValueError('Conditioning station identity required')
    frame['station_id']=frame.station_id.astype(str)
    target=str(prior.attrs.get('station',''))
    if not target:raise ValueError('Target identity required for observation exclusion')
    excluded={target,*map(str,policy.get('excluded_station_ids',[]))}
    if frame.station_id.isin(excluded).any():raise ValueError('Held target station is forbidden in conditioning observations')
    xy=np.array([float(background.lon[0]),float(background.lat[0])])
    if not np.isfinite(xy).all() or abs(xy[0])>180 or abs(xy[1])>90:raise ValueError('Invalid target longitude/latitude')
    points=frame[['longitude','latitude']].to_numpy(float)
    if (not np.isfinite(points).all() or (abs(points[:,0])>180).any() or (abs(points[:,1])>90).any()
            or np.isclose(points,xy,rtol=0,atol=1e-9).all(axis=1).any()):
        raise ValueError('Invalid conditioning coordinates or target-location alias')
    if frame.groupby('station_id')[['longitude','latitude']].nunique().gt(1).any().any():
        raise ValueError('Moving conditioning station requires a trajectory contract')
    if frame.drop_duplicates('station_id').duplicated(['longitude','latitude']).any():
        raise ValueError('Duplicate conditioning locations cannot inflate observation support')
    unit=field.attrs.get('units');physical=field.attrs.get('physical_variable')
    if not unit or not physical or not frame.unit.eq(unit).all() or not frame.background_unit.eq(unit).all():
        raise ValueError('Conditioning observation/background unit conflict')
    if not frame.variable.eq(physical).all():raise ValueError('Conditioning physical variable conflict')
    product=background.attrs.get('background_product')
    if not product or not frame.background_product.eq(product).all():raise ValueError('Collocated conditioning background must belong to the declared background product')
    semantics=field.attrs.get('temporal_semantics')
    if not frame.temporal_semantics.eq(semantics).all():raise ValueError('Conditioning temporal semantics conflict')
    frame['time']=pd.to_datetime(frame.time,errors='raise')
    clock=pd.DatetimeIndex(background.time.values).as_unit('ns')
    if not len(clock):raise ValueError('No application state epochs')
    if pd.Timestamp(prior.attrs['training_end'])>=clock[0]:raise ValueError('Historical fitting must precede application')
    if clock.hasnans or clock.has_duplicates or not clock.is_monotonic_increasing:raise ValueError('Unique increasing target state epochs required')
    if pd.DatetimeIndex(frame.time).tz!=clock.tz:raise ValueError('Conditioning timezone mismatch; normalize explicitly')
    if frame.time.isna().any() or frame.duplicated(['station_id','time']).any():raise ValueError('Invalid or duplicate station/time conditioning records')
    # Only the exact supported observation period can update its target epoch.
    frame=frame.loc[frame.time.isin(clock)].copy()
    if semantics!='instantaneous':
        if semantics not in {'mean','accumulation'} or 'time_bounds' not in background:
            raise ValueError('Explicit supported interval bounds required for observation update')
        if not {'interval_start','interval_end'}<=set(frame):raise ValueError('Conditioning intervals required')
        expected=pd.DataFrame(background.time_bounds.values,index=clock,columns=['interval_start','interval_end'])
        if pd.Timestamp(prior.attrs['training_end'])>pd.Timestamp(expected.interval_start.iloc[0]):
            raise ValueError('Historical fitting overlaps application interval')
        for name in ['interval_start','interval_end']:
            actual=pd.to_datetime(frame[name],errors='raise').to_numpy(dtype='datetime64[ns]')
            if not np.array_equal(actual,expected.loc[frame.time,name].to_numpy(dtype='datetime64[ns]')):
                raise ValueError('Conditioning and target intervals differ; no adjacent-period borrowing')
    frame.loc[~frame.qc_flag.eq('VALID'),['value','background']]=np.nan
    pairs=frame[['value','background']].to_numpy(float)
    if np.isinf(pairs).any():raise ValueError('Infinite conditioning values are not valid observations')
    if policy.get('nonnegative') and (pairs<0).any():raise ValueError('Negative conditioning values violate task requirements')
    return frame,clock,xy


def update_state(prior,background,observations,variable,policy):
    frame,clock,xy=conditioning_pairs(prior,background,observations,variable,policy)
    target=str(prior.attrs['station'])
    state=prior.copy(deep=True)
    identifiers=sorted(frame.station_id.unique())
    counts=np.zeros(len(clock),int);offset=np.full(len(clock),float(prior.affine_intercept))
    ratio=np.full(len(clock),float(prior.mean_ratio) if 'mean_ratio' in prior else np.nan)
    ratio_updated=np.zeros(len(clock),bool);distances=[]
    if identifiers:
        meta=frame.drop_duplicates('station_id').set_index('station_id').loc[identifiers]
        obs=frame.pivot(index='time',columns='station_id',values='value').reindex(index=clock,columns=identifiers).to_numpy(float)
        bg=frame.pivot(index='time',columns='station_id',values='background').reindex(index=clock,columns=identifiers).to_numpy(float)
        valid=np.isfinite(obs)&np.isfinite(bg);counts=valid.sum(axis=1)
        weights=distance_weights(meta[['longitude','latitude']].to_numpy(float),xy[None,:])
        delta=weighted_values(residual_difference(obs,float(prior.affine_slope)*bg),weights)[:,0]
        offset=np.where(np.isfinite(delta),delta,offset)
        current,okay=fit_ratio_frames(bg,obs);ratio_updated=okay[:,0]
        ratio=np.where(ratio_updated,current[:,0],ratio)
        distances=(weights[0]**(-.5)/1000).tolist()
    state['affine_intercept']=('time',offset)
    if np.isfinite(ratio).all():state['mean_ratio']=('time',ratio)
    state=state.assign_coords(time=clock)
    state['conditioning_count']=('time',counts)
    state['ratio_updated']=('time',ratio_updated)
    state.attrs.update(state_update_mode='contemporaneous_analysis',target_station=target,
        update_rule='Exact-period other-location residual IDW / existing ratio-of-sums; no target labels; prior fallback when unsupported',
        observation_state_version='observation_state_v1',conditioning_stations='|'.join(identifiers),
        prior_training_end=prior.attrs['training_end'],forecast_latency_verified='false')
    report=dict(mode='contemporaneous_analysis',target_station=target,excluded_station_ids=sorted({target,*map(str,policy.get('excluded_station_ids',[]))}),conditioning_stations=identifiers,
        residual_updated_epochs=int((counts>0).sum()),ratio_updated_epochs=int(ratio_updated.sum()),
        prior_fallback_epochs=int((counts==0).sum()),affine_prior_fallback_epochs=int((counts==0).sum()),
        ratio_prior_fallback_epochs=int((~ratio_updated).sum()),conditioning_counts=counts.tolist(),distance_km=distances,
        source_observation_sha256=hashlib.sha256(pd.util.hash_pandas_object(observations,index=True).values.tobytes()).hexdigest(),
        coefficients=dict(time=[str(t) for t in clock],affine_slope=float(prior.affine_slope),
            affine_intercept=offset.tolist(),mean_ratio=ratio.tolist() if np.isfinite(ratio).all() else None,ratio_updated=ratio_updated.tolist()),
        rule=state.attrs['update_rule'],reference_used=False,forecast_latency_verified=False,
        warning='Other sites are not assumed spatially representative; actual validation and unchanged review gate decide')
    return state,report
