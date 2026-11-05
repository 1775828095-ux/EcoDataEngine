"""Native ports -> existing task validation/random search -> native-shaped values.

No local candidate list, uncertainty requirement, evaluator or mutation driver.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr


class SearchNotReady(ValueError):
    def __init__(self,report):
        self.report=report
        super().__init__(report.get('reason',report['status']))


def search_port(port,arrays,sources,folder,*,budget=None,seed=11,progress=None):
    from ecodata.algorithms.platform_search import SearchInput,process_current_input
    folder=Path(folder);folder.mkdir()
    inputs=[];shape=port.data[port.quantity].shape
    paired='station_id' not in port.data.coords
    sites=shape[1] if len(shape)==2 else int(np.prod(shape[1:]))
    if not paired:
        raise SearchNotReady(dict(status='SEARCH_SUPPORT_UNCONFIRMED',reason='站点ID端口缺少可验证的地理坐标，不能用站号制造空间位置'))
    temporal=sites<3
    for index,(source,ds) in enumerate(sorted(zip(sources,arrays),key=lambda pair:pair[0]['source_id'])):
        a=ds[port.quantity]
        if temporal and sites!=1:
            raise SearchNotReady(dict(status='VALIDATION_EVIDENCE_INSUFFICIENT',reason='两个非规则位置不足以做空间留出；未将它们伪造成规则网格'))
        if temporal:
            out=xr.Dataset({port.quantity:(('time','lat','lon'),a.values.reshape(len(ds.time),1,1),dict(a.attrs))},
                coords=dict(time=ds.time.values,lat=[float(ds.lat.values[0])],lon=[float(ds.lon.values[0])]),attrs=dict(ds.attrs))
            for name,field in ds.data_vars.items():
                if name!=port.quantity and field.shape==a.shape:
                    out[name]=(('time','lat','lon'),field.values.reshape(len(ds.time),1,1),dict(field.attrs))
        else:out=ds.copy(deep=True)
        out.attrs.update(source_id=source['source_id'],crs=ds.attrs['crs'])
        if port.temporal=='accumulation':
            seconds=a.attrs.get('interval_seconds')
            if seconds!=86400:raise SearchNotReady(dict(status='SEARCH_INTERVAL_UNSUPPORTED',reason='当前累计量公共搜索需要已核实的逐日区间'))
            dates=pd.DatetimeIndex(out.time.values)
            out['time_bounds']=(('time','bounds'),np.column_stack([dates.values,(dates+pd.Timedelta(days=1)).values]))
            out.time.attrs['bounds']='time_bounds'
            out.time.encoding['units']='seconds since 1970-01-01'
            out.time_bounds.encoding['units']='seconds since 1970-01-01'
        inputs.append(SearchInput(f'source_{index}.nc','primary' if index==0 else 'background',out.to_netcdf(engine='scipy')))
    request=dict(task='multisource',variable=port.quantity,unit=port.unit,semantic_type='continuous',
        temporal_semantics=port.temporal,mutation_control='random',soft_guidance=False,
        nonnegative=port.quantity in {'precipitation','relative_humidity','wind_speed','solar_radiation','leaf_area_index'},
        parameters=dict(require_complete_output=True))
    if not temporal:request['processing_goal']='spatial_fusion'
    else:
        request['application_contract']=dict(id='native-input-port',version='1',field=port.quantity,unit=port.unit,
            semantic_type='continuous',valid_range=None,issues=[],
            evidence='Actual native port binding '+port.resource+' '+str(port.binding.get('sha256',''))+'; model soft preferences and unprovided physical bounds remain unknown')
    try:result=process_current_input(inputs,request,folder/'search',budget=budget,seed=seed,progress=progress)
    except ValueError as exc:
        message=str(exc)
        if not any(token in message for token in ('No visible validation','VALIDATION_EVIDENCE_INSUFFICIENT','At least three spatial locations','No executable spatial estimator')):raise
        result=dict(status='VALIDATION_EVIDENCE_INSUFFICIENT',reason=message,product=None)
    receipt=dict(route='platform_search.process_current_input',request=request,result=result,
        primary_policy='stable source identity order; reference proxy is not independent truth',
        uncertainty_required=False,test_used_for_selection=False)
    (folder/'search_receipt.json').write_text(json.dumps(receipt,ensure_ascii=True,indent=2,default=str),encoding='utf8')
    if not result.get('product'):raise SearchNotReady(dict(status=result['status'],reason=result.get('reason','搜索未产生合格产物'),receipt=str(folder/'search_receipt.json')))
    with Path(result['product']).open('rb') as stream,xr.open_dataset(stream) as opened:output=opened.load()
    values=output[port.quantity].values.reshape(shape)
    delivered=port.data.copy(deep=True);delivered[port.quantity].values[:]=values
    return delivered,receipt


def port_groups(work):
    """Joint validation over actual geographic sites, retaining native bindings."""
    groups={}
    for port,sources in work:
        attrs=port.data[port.quantity].attrs
        key=(port.quantity,port.unit,port.temporal,port.kind,port.data.attrs.get('time_basis'),
            tuple((k,str(attrs.get(k))) for k in ('physical_definition','height_m','vertical_reference','vector_reference')),
            tuple(sorted(s['sha256'] for s in sources)))
        groups.setdefault(key,[]).append((port,sources))
    return list(groups.values())


def combine_ports(members):
    from dataclasses import replace
    if len(members)==1:return members[0][0],None
    first=members[0][0];times=pd.DatetimeIndex(first.data.time.values)
    for port,_ in members:times=times.intersection(pd.DatetimeIndex(port.data.time.values))
    if times.empty:raise SearchNotReady(dict(status='NO_COMMON_ENGINEERING_DATES',reason='同变量工程端口无共同日期，未删记录或猜日期'))
    xy=[];lookup={};mapping=[];columns=[]
    for port,_ in members:
        if not {'lat','lon'}<=set(port.data.coords):
            raise SearchNotReady(dict(status='SEARCH_SUPPORT_UNCONFIRMED',reason='工程端口缺实际地理坐标，未忽略其余端口或用站号代替坐标'))
        indices=[];values=port.data[port.quantity].sel(time=times).values.reshape(len(times),-1)
        for i,point in enumerate(zip(port.data.lon.values.ravel(),port.data.lat.values.ravel())):
            if point not in lookup:
                lookup[point]=len(xy);xy.append(point);columns.append(values[:,i])
            indices.append(lookup[point])
        mapping.append(indices)
    from .ports import canonical
    d=canonical(np.column_stack(columns),times,np.asarray(xy)[:,0],np.asarray(xy)[:,1],first.quantity,first.unit,first.temporal)
    d.attrs.update(first.data.attrs);d[first.quantity].attrs=dict(first.data[first.quantity].attrs)
    return replace(first,data=d),mapping
