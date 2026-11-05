"""Product adaptation using evidenced intervals, xarray and the shared unit engine.

No fusion weights, model-day guesses, or numerical missing-value imputation.
"""
import hashlib,json
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.engines.units import UnitEngine


def acquisition_receipts(files):
    result=[]
    for name in sorted(files.files):
        if not name.lower().endswith('.json'):continue
        with files.files[name][0]() as stream:
            try:doc=json.load(stream)
            except (ValueError,UnicodeError):continue
        if isinstance(doc,dict) and isinstance(doc.get('sha256'),str) and isinstance(doc.get('url'),str):
            result.append(dict(resource=name,receipt_sha256=files.identity(name)['sha256'],**doc))
    return result


def power_time_basis(source,receipts):
    """Only a hash-bound original request establishes the missing API choice."""
    ds=source['dataset'];matches=[]
    for receipt in receipts:
        if receipt['sha256']!=source['sha256']:continue
        url=urlsplit(receipt['url']);q=parse_qs(url.query)
        if url.scheme!='https' or url.hostname!='power.larc.nasa.gov' or url.username or url.port not in (None,443):continue
        if url.path not in {'/api/temporal/daily/regional','/api/temporal/daily/point'}:continue
        if source['field'] not in ','.join(q.get('parameters',[])).split(','):continue
        basis=q.get('time-standard',[])
        if len(basis)!=1 or basis[0] not in {'UTC','LST'}:continue
        try:
            start=pd.to_datetime(q['start'][0],format='%Y%m%d');end=pd.to_datetime(q['end'][0],format='%Y%m%d')
        except (KeyError,ValueError):continue
        if start!=pd.Timestamp(ds.time.values[0]) or end!=pd.Timestamp(ds.time.values[-1]):continue
        matches.append(dict(time_basis=basis[0],receipt=receipt['resource'],receipt_sha256=receipt['receipt_sha256'],url=receipt['url']))
    if len({m['time_basis'] for m in matches})>1:raise ValueError('同一文件的原始下载凭据时间基准冲突')
    return matches[0] if matches else None


def _longitude(ds):
    """Relabel equivalent angular coordinates; carry data and bounds together."""
    lon=ds.lon;wrapped=((lon+180)%360)-180
    if len(np.unique(wrapped))!=wrapped.size:raise ValueError('经度等价转换后重复，不能静默合并网格')
    out=ds.assign_coords(lon=wrapped.assign_attrs(lon.attrs)).sortby('lon')
    bound=lon.attrs.get('bounds')
    if bound in out:
        # Shift both cell edges by the centre's integral number of revolutions.
        order=np.argsort(wrapped.values)
        shift=xr.DataArray((lon.values-wrapped.values)[order],dims='lon',coords={'lon':out.lon})
        out[bound]=out[bound]-shift
    return out


def adapt_products(sources,audit,folder,receipts):
    """Return actual amount datasets plus a trace; incomplete inputs stay blocked."""
    folder=Path(folder);folder.mkdir()
    groups={};ordinary=[];reports=[]
    for s in sources:
        if s.get('product') in {'NASA_POWER','NOAA_CMORPH'}:groups.setdefault((s['source_id'],s['quantity']),[]).append(s)
        else:ordinary.append(s)
    for index,((source_id,quantity),members) in enumerate(sorted(groups.items())):
        pieces=[];issues=[];operations=[]
        for s in members:
            row=next(r for r in audit if r['resource']==s['resource'])
            f=next(f for f in row['fields'] if f['field']==s['field'])
            blockers=list(f['blockers']);basis=f['time_basis']
            if s['product']=='NASA_POWER':
                receipt=power_time_basis(s,receipts)
                if receipt:
                    if basis not in {'UNKNOWN',receipt['time_basis']}:raise ValueError('POWER文件声明与原始请求时间基准冲突')
                    basis=receipt['time_basis'];operations.append(dict(operation='hash_bound_acquisition_receipt',**receipt))
                    blockers=[b for b in blockers if b['code']!='SOURCE_DAY_UNCONFIRMED']
                    f['initial_time_basis']=f['time_basis'];f['time_basis']=basis;f['blockers']=blockers
            row['blockers']=blockers
            field=s['dataset'][s['field']]
            duration=f.get('interval_seconds')
            # Unknown day origin does not prevent an evidenced 24h rate integral;
            # it still prevents cross-source matching and engineering writeback.
            if not duration:
                issues.extend(blockers);continue
            severe=[b for b in blockers if b['code']!='SOURCE_DAY_UNCONFIRMED']
            if severe:issues.extend(severe);continue
            keep=[s['field']]
            sigmas=[n for n in str(field.attrs.get('ancillary_variables','')).split()
                if n in s['dataset'] and s['dataset'][n].attrs.get('uncertainty_type')=='standard_deviation']
            if len(sigmas)>1:raise ValueError('同一产品字段标准差不唯一')
            keep.extend(sigmas)
            for axis in ('lat','lon'):
                bound=s['dataset'][axis].attrs.get('bounds')
                if bound in s['dataset']:keep.append(bound)
            # Canonical interval bounds are derived alongside raw boundary evidence.
            d=s['dataset'][keep].copy(deep=True).rename({s['field']:quantity})
            d=_longitude(d)
            starts=pd.DatetimeIndex(d.time.values)
            d['time_bounds']=(('time','nv'),np.column_stack([starts.values,(starts+pd.Timedelta(seconds=duration)).values]))
            d.time.attrs['bounds']='time_bounds'
            d.attrs=dict(source_id=source_id,time_basis=basis,product=s['product'])
            if s['dataset'].attrs.get('crs'):d.attrs['crs']=s['dataset'].attrs['crs']
            d[quantity].attrs=dict(field.attrs,units=field.attrs['units'],physical_variable=quantity)
            if sigmas:
                if sigmas[0]!='uncertainty':d=d.rename({sigmas[0]:'uncertainty'})
                d[quantity].attrs['ancillary_variables']='uncertainty'
            pieces.append((s,d,duration));issues.extend(blockers)
        report=dict(source_id=source_id,quantity=quantity,resources=[dict(resource=s['resource'],sha256=s['sha256']) for s in members],
            operations=operations,blockers=issues,product=None,raw_modified=False,fusion_executed=False)
        if len(pieces)!=len(members):
            report['status']='BLOCKED_PRODUCT_CONTRACT';reports.append(report);ordinary.extend(members);continue
        attrs=[p[1].attrs for p in pieces]
        if len({a['time_basis'] for a in attrs})!=1:raise ValueError('同一产品分片时间基准不一致')
        # join=exact forbids accidental spatial outer joins or grid resizing.
        merged=xr.concat([p[1] for p in pieces],dim='time',data_vars='minimal',coords='minimal',compat='equals',join='exact',combine_attrs='override').sortby('time')
        dates=pd.DatetimeIndex(merged.time.values)
        if dates.has_duplicates:raise ValueError('同产品时间分片存在重复日期；不平均或覆盖冲突')
        durations=(merged.time_bounds.values[:,1]-merged.time_bounds.values[:,0])/np.timedelta64(1,'s')
        merged[quantity]=UnitEngine().integrate_rate(merged[quantity],durations,'mm')
        if 'uncertainty' in merged:
            merged['uncertainty']=UnitEngine().integrate_rate(merged.uncertainty,durations,'mm')
            merged.uncertainty.attrs['uncertainty_type']='standard_deviation'
        merged[quantity].attrs.update(standard_name='lwe_thickness_of_precipitation_amount',physical_definition=quantity,
            temporal_semantics='accumulation',interval_seconds=86400,data_kind='continuous',spatial_semantics='cell_mean')
        merged.attrs=attrs[0]
        product=folder/f'product_{index}.nc'
        merged.to_netcdf(product,engine='h5netcdf')
        with product.open('rb') as stream,xr.open_dataset(stream) as reopened:
            np.testing.assert_allclose(reopened[quantity],merged[quantity],equal_nan=True,rtol=0,atol=0)
        report.update(status='ADAPTED_AWAITING_ADMISSION',product=str(product),time_basis=merged.attrs['time_basis'],
            time_records=len(dates),units='mm',interval_seconds=86400,
            operations=operations+[dict(operation='coordinate_equivalence',backend='xarray.assign_coords/sortby',resampling=False),
                dict(operation='strict_time_concatenation',backend='xarray.concat join=exact',members=len(members)),
                dict(operation='rate_interval_integral',details=json.loads(merged[quantity].attrs['interval_integration']))],
            support='original grid-cell support retained; station sampling is a separate scientific operation')
        combined=dict(members[0],dataset=merged,field=quantity,
            resource=';'.join(s['resource'] for s in members),sha256=hashlib.sha256(''.join(s['sha256'] for s in members).encode()).hexdigest(),
            members=report['resources'],blockers=issues)
        ordinary.append(combined);reports.append(report)
        for s in members:
            row=next(r for r in audit if r['resource']==s['resource'])
            row['adaptation']=dict(product=product.name,status=report['status'],units='mm',time_basis=merged.attrs['time_basis'])
    (folder/'adaptation.json').write_text(json.dumps(reports,ensure_ascii=True,indent=2,default=str),encoding='utf8')
    return ordinary,reports


def native_cell_values(ds,field,port):
    """Extract native cells for virtual weather forcing, not point observations.

    xarray's nearest coordinate lookup retains one native cell's value/support;
    there is no interpolation, subgrid detail, or covariance interpolation.
    """
    if any(ds[a].ndim!=1 for a in ('lat','lon')):raise ValueError('天气网格提取仅支持明确的一维经纬度坐标')
    query_lon=((np.asarray(port.data.lon)+180)%360)-180
    query_lat=np.asarray(port.data.lat)
    for axis,query in [('lon',query_lon),('lat',query_lat)]:
        bound=ds[axis].attrs.get('bounds')
        limits=ds[bound].values if bound in ds else ds[axis].values
        if np.any(query<np.nanmin(limits)) or np.any(query>np.nanmax(limits)):
            raise ValueError('天气站超出原始网格可确认支持域；不外推：'+axis)
    selected=ds[field].sel(lon=xr.DataArray(query_lon,dims='point'),lat=xr.DataArray(query_lat,dims='point'),method='nearest')
    selected=selected.transpose('time','point').reindex(time=port.data.time.values)
    evidence=dict(operator='xarray.sel nearest native cell',resampling=False,
        meaning='原网格单元驱动估计；不是新增站点观测或空间精细化',
        cell_lon=np.asarray(selected.lon).tolist(),cell_lat=np.asarray(selected.lat).tolist(),
        original_resolution_retained=True)
    return selected.values,evidence
