"""Native-format decoding/eligibility bridge to the EXISTING platform search.

Neither a substitute fusion algorithm nor a case-specific external experiment.
Every source/time/support decision is retained before candidate evaluation.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import xarray as xr
from .calmet_mm4_reader import decode_mm4,wind_at_height,ORIGIN

def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def process_surface(surface,receipt,folder,progress=None):
    from pyproj import CRS,Transformer
    from scipy.spatial import Delaunay
    from ecodata.algorithms.platform_search import SearchInput,process_current_input
    from .calpuff_native_bundle import FIELDS,scalar,integer,digest
    folder=Path(folder);stage=folder/'fusion_stage';stage.mkdir()
    met=receipt['native_plan']['CALMET'];v=met['values'];sources=met['inputs']
    report=dict(schema='ecodata.native-fusion-stage.v1',status='CHECKING',algorithm_search_executed=False,
        original_engineering_modified=False,raw_sources=[dict(role=r['key'],member=r['member'],sha256=r['sha256']) for r in sources],
        excluded=[],searches=[],accuracy_evaluated=False)
    save(stage/'decision.json',report)
    model_inputs=[x for x in sources if x['key'] in {'MM4DAT','M3DDAT'}]
    if not model_inputs:
        report.update(status='NOT_APPLICABLE',reason='本工程没有提供另一份可比较的同物理量气象场；原生站点资料保留')
        save(stage/'decision.json',report);return surface,report,{}
    if len(model_inputs)!=1 or integer(v,'IPROG')!=4:
        raise ValueError('融合阶段尚未支持此多文件/非MM4预报资料，未跳到转换')
    if scalar(v,'PMAP').upper()!='UTM' or scalar(v,'DATUM')!='NAS-C':
        raise ValueError('融合阶段缺少当前投影/基准的可信定位规则；未把工程坐标直接当经纬度')
    mm4=model_inputs[0];source=folder/mm4['path']
    if digest(source)!=mm4['sha256']:raise ValueError('本次MM4资料哈希变化')
    d=decode_mm4(source.read_bytes())
    # Native mapping is documented by CALMET BLOCK DATA (NWS-84) and the
    # uploaded UTM/NAS-C control, not guessed from a case location/name.
    zone=integer(v,'IUTMZN');south=scalar(v,'UTMHEM').upper()=='S'
    station_crs=CRS.from_epsg((26700 if not south else 0)+zone) if not south else None
    if station_crs is None:raise ValueError('NAS-C南半球定位未支持')
    station_tx=Transformer.from_crs(station_crs,4326,always_xy=True)
    sx=np.array([s['x_km'] for s in receipt['station_table']])*1000
    sy=np.array([s['y_km'] for s in receipt['station_table']])*1000
    lon,lat=station_tx.transform(sx,sy)
    field_tx=Transformer.from_crs(d['source_geographic_crs'],4326,always_xy=True)
    flon,flat=field_tx.transform(d['geometry'][:,1],d['geometry'][:,0])
    # Selection support is fixed by geometry/time/height BEFORE looking at
    # candidate scores. No outside-area validation stations are silently sampled.
    hull=Delaunay(np.column_stack([flon,flat]))
    inside=hull.find_simplex(np.column_stack([lon,lat]))>=0
    heights=np.array([s['anemometer_height_m'] for s in receipt['station_table']],float)
    report['coordinate_alignment']=dict(station_source=station_crs.to_string(),field_source=d['source_geographic_crs'],
        target='EPSG:4326',station_operation=station_tx.description,field_operation=field_tx.description,
        station_accuracy_m=station_tx.accuracy,field_accuracy_m=field_tx.accuracy,
        limitation='Library coordinate transform accuracy is not independent positioning validation; NWS-84 spherical datum has no survey accuracy claim')
    requested=pd.DatetimeIndex([pd.Timestamp(t).tz_convert('UTC').tz_localize(None) for t in receipt['run_hour_ends']])
    tids=pd.DatetimeIndex(d['times']).get_indexer(requested)
    oids=pd.DatetimeIndex(surface.time.values).get_indexer(requested)
    if (tids<0).any() or (oids<0).any():raise ValueError('融合资料没有精确覆盖同一组模拟小时；不补时间')
    report['source_reader']=dict(origin=ORIGIN,path=str(source),sha256=digest(source),shape=d['shape'])
    report['fixed_comparison_support']=dict(hours=[str(t) for t in requested],
        included_station_ids=[int(i) for i in surface.station_id.values[inside]],
        excluded_station_ids=[int(i) for i in surface.station_id.values[~inside]],
        reason='MM4原生网格凸包内且有明确观测高度；对所有候选固定同一验证范围，范围外保留观测，不外推')
    for i in np.where(~inside)[0]:report['excluded'].append(dict(station_id=int(surface.station_id.values[i]),reason='outside actual MM4 source support; retained original observations'))
    if inside.sum()<3:raise ValueError('共同空间范围内不足3个站点，无法执行平台现有留站验证；融合未跳过，条件不足')
    if len(np.unique(heights[inside]))!=1:raise ValueError('站点风速观测高度不同，需要分高度处理入口，未混合不同高度')
    height=float(heights[inside][0]);wind=wind_at_height(d,height)[tids]
    report['vertical_alignment']=dict(target_anemometer_height_m=height,method='bounded linear interpolation of eastward/northward components in height above native terrain',
        surface_level_encoding='MM4 first mandatory surface record is at native terrain elevation',underground_levels_excluded=True,extrapolation=False,
        qualification='Model-derived estimate at declared station height; not a measured 10m wind field or terrain-scale accuracy claim')
    speed=surface.wind_speed.values[oids][:,inside];direction=surface.wind_direction.values[oids][:,inside]*np.pi/180
    observed=[-speed*np.sin(direction),-speed*np.cos(direction)]
    if not np.isfinite(observed).all():raise ValueError('共同范围风观测仍有缺失，当前固定验证集合不能完整评分')
    selected={};results={}
    for component,unit,fieldvalues,obs in [('eastward_wind','m/s',wind[:,:,0],observed[0]),('northward_wind','m/s',wind[:,:,1],observed[1])]:
        work=stage/component;work.mkdir()
        attrs=dict(units=unit,physical_variable=component,physical_definition=f'{component} at {height:g} m above local ground; model estimates compared with station observations',
            temporal_semantics='instantaneous',data_kind='continuous')
        model=xr.Dataset({component:(('time','location'),fieldvalues,dict(attrs,spatial_semantics='grid_estimate'))},
            coords=dict(time=requested.values,lon=('location',flon),lat=('location',flat)),attrs=dict(crs='EPSG:4326',source_id='MM4:'+mm4['sha256']))
        points=xr.Dataset({component:(('time','location'),obs,dict(attrs,spatial_semantics='point'))},
            coords=dict(time=requested.values,lon=('location',lon[inside]),lat=('location',lat[inside])),attrs=dict(crs='EPSG:4326',source_id='SURF:'+receipt['source_archive_sha256']))
        target=xr.Dataset(coords=dict(time=requested.values,lon=('location',lon[inside]),lat=('location',lat[inside])),attrs=dict(crs='EPSG:4326'))
        inputs=[]
        for role,name,ds in [('primary','station_observations',points),('background','model_field',model),('target','target_stations',target)]:
            for axis in ('lat','lon'):ds[axis].attrs['units']='degrees_north' if axis=='lat' else 'degrees_east'
            p=work/(name+'.nc');ds.to_netcdf(p,engine='h5netcdf');inputs.append(SearchInput(p.name,role,p.read_bytes()))
        request=dict(task='multisource',processing_goal='spatial_fusion',variable=component,unit=unit,semantic_type='continuous',
            physical_variable=component,temporal_semantics='instantaneous',allow_training=True,mutation_control='random',soft_guidance=False,
            observation_crs='EPSG:4326',nonnegative=False)
        if progress:progress(f'平台算法比较：{component}，{inside.sum()}个实际站点与{len(flon)}个MM4网格，固定同一留站评分范围')
        result=process_current_input(inputs,request,work/'search',progress=progress)
        if not result.get('product'):raise ValueError('平台融合算法没有给出合格结果：'+component+'；'+str(result.get('reason',result.get('status'))))
        from .netcdf_io import load_dataset
        ds=load_dataset(result['product'])
        value=ds[component].values.reshape(len(requested),int(inside.sum()))
        if not np.isfinite(value).all():raise ValueError('平台选择结果不完整：'+component)
        comparison=result.get('selection_comparison',{})
        if not comparison:raise ValueError('缺少单源/多源统一评分记录，不能继续转换')
        selected[component]=value;results[component]=result
        report['searches'].append(dict(component=component,status=result.get('status'),counts=result.get('counts'),
            selection_comparison=comparison,post_selection_assessment=result.get('post_selection_assessment'),
            product=str(result['product']),product_sha256=digest(result['product']),selected_graph=result.get('selected_graph')))
        save(stage/'decision.json',report)
    updated=surface.copy(deep=True)
    u=selected['eastward_wind'];vwind=selected['northward_wind']
    new_speed=np.hypot(u,vwind);new_direction=np.mod(np.degrees(np.arctan2(-u,-vwind)),360)
    new_direction[new_speed==0]=0
    idx=np.where(inside)[0]
    updated.wind_speed.values[np.ix_(oids,idx)]=new_speed
    updated.wind_direction.values[np.ix_(oids,idx)]=new_direction
    report.update(status='COMPLETE',algorithm_search_executed=True,
        selection_rule='Use the existing platform selected graph; never force multi-source to win',
        max_wind_speed_change_m_s=float(np.max(abs(new_speed-speed))),
        updated_source_records=int(len(oids)*len(idx)),untouched_records=int(surface.wind_speed.size-len(oids)*len(idx)),
        other_physical_fields_retained=list(set(FIELDS)-{'wind_speed','wind_direction'}),
        limitations=['Only common source support is processed; no outside-area extrapolation',
            'Three shared stations are insufficient for a separate outer spatial holdout; existing platform reports the limitation',
            'Retrospective observation cross-validation does not establish forecast skill',
            'No resolution increase or predictive-accuracy improvement is claimed'])
    save(stage/'decision.json',report)
    return updated,report,results
