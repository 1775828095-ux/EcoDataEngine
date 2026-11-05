"""One scientific input-port planner/processor shared by all model drivers.

Resource semantics are data metadata, never requested algorithms or graph winners.
Unknown meanings are not inferred from filenames or numerical ranges.
"""
from contextlib import ExitStack
from pathlib import Path,PurePosixPath
import hashlib,json,shutil,zipfile
from dataclasses import asdict
from uuid import uuid4
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.models.project_binding import ProjectFiles,bind_project
from ecodata.models.scientific_schema import identity
from .ports import netcdf_ports,weather_ports,surface_ports

SCHEMA='ecodata.engineering-semantics.v1'
ATTRS={'physical_variable','standard_name','temporal_semantics','spatial_semantics','data_kind','physical_definition','height_m','vertical_reference','vector_reference','interval_seconds','time_basis'}


def metadata_for(files,model):
    result={}
    for name in sorted(files.files):
        if not name.endswith('.json'):continue
        with files.files[name][0]() as s:
            try:doc=json.load(s)
            except (ValueError,UnicodeError):continue
        if not isinstance(doc,dict) or doc.get('schema')!=SCHEMA:continue
        if doc.get('model_id')!=model:raise ValueError('工程语义清单与目标模型不符')
        if set(doc)-{'schema','model_id','resources'}:raise ValueError('工程语义清单只允许科学元数据，不能指定算法/操作')
        for entry in doc.get('resources',[]):
            if set(entry)-{'resource','role','variables','confirmed'} or entry.get('confirmed') is not True:
                raise ValueError('资源科学声明需确认，且不能含算法、目标映射或执行命令')
            target,_=files.resolve(name,entry['resource'])
            if target in result:raise ValueError('重复资源科学声明：'+target)
            if entry.get('role') not in {'source','model_input'}:raise ValueError('资源角色必须明确source或model_input')
            for attrs in entry.get('variables',{}).values():
                if set(attrs)-ATTRS:raise ValueError('变量声明只允许物理/时间语义，不允许注入数值、单位覆盖或算法')
            result[target]=entry
    return result


def _apply_semantics(ds,variables):
    ds=ds.copy(deep=False)
    for name,attrs in variables.items():
        if name not in ds:raise ValueError('语义清单引用不存在的字段：'+name)
        for k,v in attrs.items():
            actual=ds[name].attrs.get(k)
            if actual is not None and actual!=v:raise ValueError('声明与文件元数据冲突：'+name+'.'+k)
        ds[name].attrs=dict(ds[name].attrs,**attrs)
    return ds


def source_fields(files,metadata,audit=None):
    sources=[]
    for name in sorted(files.files):
        if not files.prefix(name,8).startswith((b'CDF',b'\x89HDF')):continue
        entry=metadata.get(name,{})
        with files.files[name][0]() as s,xr.open_dataset(s) as opened:
            native=any(k in opened.attrs for k in ('IOAPI_VERSION','CAMx_NAME','WEST-EAST_GRID_DIMENSION'))
            role=entry.get('role',opened.attrs.get('engineering_role'))
            if role in (None,'source') and not native:
                from .precipitation_products import inspect_product
                interpreted=_apply_semantics(opened,entry.get('variables',{}))
                product=inspect_product(interpreted)
                if product is not None:
                    ds=interpreted.load();digest=files.identity(name)['sha256']
                    for field in product['fields']:
                        sources.append(dict(resource=name,field=field['field'],quantity=field['quantity'],
                            dataset=ds,source_id=product['source_id'],sha256=digest,
                            product=product['product'],blockers=field['blockers']))
                    if audit is not None:audit.append(dict(resource=name,sha256=digest,
                        status='IDENTIFIED_PRODUCT',used=False,**product))
                    continue
            if role is None and not native and opened.attrs.get('source_id') and any(
                    v.attrs.get('standard_name') or v.attrs.get('physical_variable') for v in opened.data_vars.values()):
                role='source'
            if role!='source':
                if audit is not None and role is None and not native:audit.append(dict(resource=name,sha256=files.identity(name)['sha256'],
                    status='UNRECOGNIZED_SOURCE',used=False,blockers=[dict(code='SOURCE_ROLE_UNCONFIRMED',
                    message='未声明来源角色，也未匹配已核查的产品元数据；未根据文件名制造来源身份')]))
                continue
            if native:raise ValueError('原生工程文件不能同时声明为独立新来源')
            ds=_apply_semantics(opened.load(),entry.get('variables',{}))
        source_id=ds.attrs.get('source_id')
        if not source_id:raise ValueError('独立来源需明确source_id，不能用文件名制造独立性')
        ancillary={x for a in ds.data_vars.values() for x in str(a.attrs.get('ancillary_variables','')).split()}
        if any(n in ds and any(k in ds[n].attrs for k in ('flag_values','flag_masks')) for n in ancillary):
            raise ValueError('来源含尚未绑定处理规则的QA标记，不能静默忽略后融合：'+name)
        for var in ds.data_vars:
            if var in ancillary:continue
            quantity,basis=identity(var,ds[var].attrs)
            if basis not in {'standard_name','physical_variable'}:continue
            if not ds[var].attrs.get('units') or not ds[var].attrs.get('temporal_semantics'):
                raise ValueError('来源变量缺少单位或时间语义：'+name+':'+var)
            sources.append(dict(resource=name,field=var,quantity=quantity,dataset=ds,source_id=source_id,sha256=files.identity(name)['sha256']))
        if audit is not None:audit.append(dict(resource=name,source_id=source_id,sha256=files.identity(name)['sha256'],
            status='DECLARED_SOURCE',used=False,fields=[dict(field=s['field'],quantity=s['quantity']) for s in sources if s['resource']==name]))
    return sources


def _source_on_port(source,port):
    from ecodata.engines.units import UnitEngine
    from ecodata.algorithms.spatial_fusion_task import sample
    ds=source['dataset'];field=source['field'];a=ds[field]
    basis=port.data.attrs.get('time_basis','UNKNOWN')
    if basis=='UNKNOWN' or a.attrs.get('time_basis',ds.attrs.get('time_basis'))!=basis:
        raise ValueError('时间基准未知或不一致；不按无时区时间标签猜UTC/当地时间')
    if a.attrs.get('physical_definition',port.quantity)!=port.data[port.quantity].attrs.get('physical_definition',port.quantity):
        raise ValueError('同名变量的物理定义不同，不能直接融合')
    for key in ['height_m','vertical_reference','vector_reference']:
        wanted=port.data[port.quantity].attrs.get(key)
        if wanted is not None and a.attrs.get(key)!=wanted:raise ValueError('物理支持未知或不一致：'+key)
    interval=port.data[port.quantity].attrs.get('interval_seconds')
    if interval is not None and a.attrs.get('interval_seconds')!=interval:
        raise ValueError('区间长度未知或不一致；不能只按日标签融合')
    if a.attrs['temporal_semantics']!=port.temporal:raise ValueError('源/端口时间统计含义不一致，不仅比较时间标签')
    if a.attrs.get('data_kind','continuous')!=port.data[port.quantity].attrs.get('data_kind','continuous'):
        raise ValueError('源/端口数据类型不一致')
    converted=UnitEngine().convert(a,port.unit)
    output=port.data.copy(deep=True);output[port.quantity].values[:]=np.nan
    if source.get('product') and port.kind=='swat_weather' and a.attrs.get('spatial_semantics')=='cell_mean':
        from .precipitation_adaptation import native_cell_values
        grid=ds.copy(deep=False);grid[field]=converted
        values,evidence=native_cell_values(grid,field,port)
        output[port.quantity].values[:]=values
        sigma=[n for n in str(a.attrs.get('ancillary_variables','')).split() if n in ds and ds[n].attrs.get('uncertainty_type')=='standard_deviation']
        if len(sigma)>1:raise ValueError('同来源标准差字段不唯一')
        if sigma:
            uncertainty=UnitEngine().convert(ds[sigma[0]],port.unit)
            grid=ds.copy(deep=False);grid[sigma[0]]=uncertainty
            sampled,_=native_cell_values(grid,sigma[0],port)
            output['uncertainty']=(output[port.quantity].dims,sampled,dict(units=port.unit,uncertainty_type='standard_deviation'))
        output.attrs.update(source_id=source['source_id'],native_cell_extraction=json.dumps(evidence,ensure_ascii=False))
        return output
    if 'station_id' in output.coords:
        if not {'time','station_id'}<=set(ds.coords):raise ValueError('无地理坐标的原生站点端口需要明确station_id/time')
        if list(map(str,ds.station_id.values))!=list(map(str,output.station_id.values)):
            raise ValueError('站点身份/顺序不一致，不能只按数组大小绑定')
        if not np.array_equal(pd.DatetimeIndex(ds.time.values),pd.DatetimeIndex(output.time.values)):
            raise ValueError('站点时间轴不匹配；未声明区间处理时不猜聚合')
        values=converted.transpose('time','station_id').values
        if values.shape!=output[port.quantity].shape:raise ValueError('站点端口形状不匹配')
        output[port.quantity].values[:]=values
        def align(v):return v.transpose('time','station_id').values
    else:
        if ds.attrs.get('crs')!='EPSG:4326':raise ValueError('网格来源须提供实际EPSG:4326经纬度；不猜投影')
        spatial=ds[[field]].rename({field:port.quantity});spatial[port.quantity]=converted
        # Existing sampler accepts rectilinear grids or paired coordinates.
        if spatial.lat.ndim==2 and spatial.lon.dims==spatial.lat.dims:
            dims=spatial.lat.dims;spatial=spatial.stack(point=dims).reset_index('point',drop=True)
        values=sample(spatial,port.quantity,pd.DatetimeIndex(output.time.values),np.column_stack([output.lon.values,output.lat.values]))
        output[port.quantity].values[:]=values
        def align(v):
            copy=spatial[[port.quantity]].copy();copy[port.quantity]=v.stack(point=v.dims[-2:]).reset_index('point',drop=True) if v.ndim==3 and v.dims[-2:]==ds.lat.dims else v
            return sample(copy,port.quantity,pd.DatetimeIndex(output.time.values),np.column_stack([output.lon.values,output.lat.values]))
    output[port.quantity].attrs=dict(output[port.quantity].attrs,physical_variable=port.quantity,units=port.unit)
    sigma=[n for n in str(a.attrs.get('ancillary_variables','')).split() if n in ds and ds[n].attrs.get('uncertainty_type')=='standard_deviation']
    if len(sigma)>1:raise ValueError('同来源标准差字段不唯一')
    if sigma:
        if 'station_id' not in output.coords:
            # Standard deviations cannot be interpolated like measurements:
            # propagation would require a covariance/support contract.
            if not (np.array_equal(ds.time.values,output.time.values)
                    and np.array_equal(np.asarray(ds.lon).ravel(),output.lon.values)
                    and np.array_equal(np.asarray(ds.lat).ravel(),output.lat.values)):
                raise ValueError('不确定性仅支持完全一致的时空支持；未声明协方差不能直接插值标准差')
        u=ds[sigma[0]]
        scale,offset=UnitEngine().affine(u.attrs.get('units',''),port.unit)
        output['uncertainty']=(output[port.quantity].dims,align(u*abs(scale)),dict(units=port.unit,uncertainty_type='standard_deviation'))
    output.attrs.update(source_id=source['source_id'])
    return output


def process_port(port,sources,folder,progress=None):
    from ecodata.evaluation.quality_gates import RawInputQualityGate,ResultQualityGate
    folder.mkdir()
    original_port=port
    if any(s.get('product') for s in sources):
        from dataclasses import replace
        common=pd.DatetimeIndex(port.data.time.values)
        for source in sources:common=common.intersection(pd.DatetimeIndex(source['dataset'].time.values))
        if common.empty:raise ValueError('新来源与工程没有共同日期，不能写回空更新')
        port=replace(port,data=port.data.sel(time=common))
    arrays=[_source_on_port(s,port) for s in sources]
    ids=[s['source_id'] for s in sources]
    if len(ids)!=len(set(ids)) or len({s['sha256'] for s in sources})!=len(sources):
        raise ValueError('同来源/同文件别名不能作为独立多源重复加权')
    kind=port.data[port.quantity].attrs.get('data_kind','continuous')
    if kind!='continuous' and len(arrays)>1:raise ValueError('分类/方向数据不能当连续标量直接融合')
    source_quality=[RawInputQualityGate().validate(d,variable=port.quantity,
        requires_spatial='lat' in d.coords,expected_unit=port.unit) for d in arrays]
    (folder/'source_quality.json').write_text(json.dumps(source_quality,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    if any(q['status']=='BLOCK' for q in source_quality):raise ValueError('公共原始来源质量门拒绝处理')
    input_quality=dict(source_quality[0],per_source=source_quality,
        support_binding='native_station_identifiers' if 'station_id' in port.data.coords else 'explicit_geographic_coordinates')
    semantic_type='categorical' if kind=='categorical' else 'continuous'
    from .search_bridge import search_port
    output,execution=search_port(port,arrays,sources,folder/'decision',progress=progress)
    raw=RawInputQualityGate().validate(output,variable=port.quantity,requires_spatial=False,expected_unit=port.unit)
    quality=ResultQualityGate().validate(port.data,output,variable=port.quantity,semantic_type=semantic_type,expected_output_unit=port.unit)
    if raw['status']=='BLOCK' or quality['status']=='BLOCK':raise ValueError('公共双层质量门拒绝端口产品')
    field=output[port.quantity].values
    if not np.isfinite(field).any():raise ValueError('来源在目标端口没有有效覆盖，不能交付空更新')
    report=dict(port=dict(resource=port.resource,variable=port.variable,quantity=port.quantity,unit=port.unit,
        temporal=port.temporal,kind=port.kind,binding=port.binding),sources=[{k:v for k,v in s.items() if k!='dataset'} for s in sources],
        execution=execution,input_quality=input_quality,final_raw_quality=raw,output_quality=quality,
        updated_cells=int(np.isfinite(field).sum()),total_cells=int(field.size),accuracy_assessed=False)
    report['source_alignment']=[json.loads(d.attrs['native_cell_extraction']) for d in arrays if d.attrs.get('native_cell_extraction')]
    output.to_netcdf(folder/'product.nc',engine='h5netcdf')
    (folder/'decision.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    if original_port is not port:
        complete=np.full(original_port.data[port.quantity].shape,np.nan)
        positions=pd.DatetimeIndex(original_port.data.time.values).get_indexer(port.data.time.values)
        complete[positions]=field
        report['native_update_scope']=dict(common_dates=len(positions),original_dates=original_port.data.sizes['time'],
            outside_common_dates='unchanged original records, including native missing markers')
        (folder/'decision.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        return complete,report
    return field,report


def prepare_engineering_inputs(uploads,model,folder,progress=None):
    """The public shared planner; drivers do not select scientific algorithms."""
    folder=Path(folder)
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack)
        for name in files.files:
            if not name.endswith('.json'):continue
            with files.files[name][0]() as s:
                try:decl=json.load(s)
                except (ValueError,UnicodeError):continue
            if isinstance(decl,dict) and decl.get('schema')=='ecodata.model-task.v1':return None
        metadata=metadata_for(files,model);source_audit=[]
        sources=source_fields(files,metadata,source_audit)
        has_swat_project=model=='swat_plus' and any(PurePosixPath(n).name=='file.cio' for n in files.files)
        if not sources and not has_swat_project:return None
        if progress:progress('按实际工程输入端口匹配变量、单位、时间与空间；调用公共处理机制')
        if model in {'wrf','camx','cmaq'}:
            binding=bind_project(uploads,model)
            if binding:
                metadata.setdefault(binding['control']['path'],{})['_active_resources']=[b['target'] for b in binding['bindings']]
            elif not any(m.get('role')=='model_input' for m in metadata.values()):
                return None
            ports,unknown=netcdf_ports(files,model,metadata)
        elif model=='swat_plus':
            if not any(PurePosixPath(n).name=='file.cio' for n in files.files):
                from ecodata.models.swat_editor_project import prepare_editor_project
                from ecodata.interface.engineering_upload import LocalUpload
                boot=folder/'engineering_bootstrap';boot.mkdir()
                exported=prepare_editor_project(uploads,model,boot,progress)
                if exported is None:return None
                if exported['manifest']['status']=='CONVERSION_FAILED':return exported
                post=folder/'engineering_post_export';post.mkdir()
                return prepare_engineering_inputs([LocalUpload('exported_project.zip',Path(exported['archive']))],model,post,progress)
            try:ports,unknown=weather_ports(files,metadata,requested_quantities={s['quantity'] for s in sources} if sources else None)
            except ValueError as exc:
                if sources:raise
                from ecodata.models.project_inventory import prepare_project_inventory
                return prepare_project_inventory(uploads,model,folder,source_audit=source_audit,
                    weather_audit=dict(ports=0,issues=[dict(code='WEATHER_PORT_UNREADABLE',reason=str(exc))]))
        elif model=='calpuff':
            if not any(n.lower().endswith('.inp') for n in files.files) and not any(m.get('role')=='model_input' for m in metadata.values()):return None
            ports,unknown=surface_ports(files,metadata)
        else:return None
        # Product recognition is distinct from scientific admission. Return a
        # downloadable audit instead of silently dropping unadmitted resources.
        adaptation=[]
        if any(s.get('product') for s in sources):
            from .precipitation_adaptation import adapt_products,acquisition_receipts
            sources,adaptation=adapt_products(sources,source_audit,folder/'engineering_adaptation',acquisition_receipts(files))
        blocked=[]
        for item in source_audit:
            reasons=list(item.get('blockers',[]))
            if not item.get('adaptation'):
                for field in item.get('fields',[]):reasons.extend(field.get('blockers',[]))
            relevant=[p for p in ports if any((s['resource']==item['resource'] or any(m['resource']==item['resource'] for m in s.get('members',[]))) and s['quantity']==p.quantity for s in sources)]
            if item['status']=='IDENTIFIED_PRODUCT' and relevant and any(p.data.attrs.get('time_basis')=='model_calendar_day' for p in relevant):
                reasons.append(dict(code='MODEL_DAY_UNCONFIRMED',message='工程日记录未声明与来源UTC/LST日的对应关系；需核实模型日边界，不能按相同日期标签直接融合'))
            source_basis=item.get('adaptation',{}).get('time_basis')
            if source_basis in {'UTC','LST'} and any(p.data.attrs.get('time_basis') in {'UTC','LST'} and p.data.attrs['time_basis']!=source_basis for p in relevant):
                reasons.append(dict(code='DAY_BOUNDARIES_INCOMPATIBLE',message='来源与工程的统计日界不同；需要更高时间频率资料重划区间，不能把日累计量平移标签或按比例拆分'))
            if source_basis=='LST' and any(p.data.attrs.get('time_basis')=='LST' for p in relevant):
                reasons.append(dict(code='LOCAL_DAY_SUPPORT_UNCONFIRMED',message='LST缩写不足以确认同一日界：POWER采用随经度变化的地方太阳时，需双方明确的UTC区间边界，不能猜成当地标准时'))
            if not item.get('adaptation') and item['status']=='IDENTIFIED_PRODUCT' and relevant and any('lon' in p.data.coords and p.data.lon.min()<0 for p in relevant) and any(
                    f.get('coordinates',{}).get('lon',{}).get('min',-180)>=0 and
                    f.get('coordinates',{}).get('lon',{}).get('max',0)>180 for f in item.get('fields',[])):
                item['longitude_binding_required']='原始0–360经度与目标负经度需在保持原网格支持下对应；尚未执行重采样'
            item['blockers']=list({r['code']:r for r in reasons}.values())
            if reasons:blocked.append(item['resource'])
        if blocked or not sources:
            from ecodata.models.project_inventory import prepare_project_inventory
            return prepare_project_inventory(uploads,model,folder,source_audit=source_audit,
                weather_audit=dict(ports=len(ports),issues=unknown,missing_policy='preserve native missing markers; no imputation; no WGN execution'),
                adaptation=adaptation,attachments=folder/'engineering_adaptation' if adaptation else None)
        stage=folder/'engineering_inputs';stage.mkdir()
        work=[];unmatched=[]
        for source in sources:
            matches=[p for p in ports if p.quantity==source['quantity']]
            if not matches:unmatched.append(dict(source=source['resource'],field=source['field'],quantity=source['quantity']))
        for port in ports:
            matches=[s for s in sources if s['quantity']==port.quantity]
            if matches:work.append((port,matches))
        plan=dict(schema='ecodata.engineering-input-plan.v1',model_id=model,matched_ports=len(work),
            unknown=unknown,unmatched_sources=unmatched,source_modified=False,algorithm_policy='public task validation and random mutation; no local fusion shortlist')
        from .general import EngineeringProject
        context=EngineeringProject(model_name=model,state_variables=[dict(resource=p.resource,name=p.variable,
            quantity=p.quantity,unit=p.unit,temporal=p.temporal,kind=p.kind,binding=p.binding) for p,_ in work],
            metadata=dict(schema='ecodata.engineering-input-context.v1',unknown=unknown,
                scope='verified input ports; not full simulator configuration or native run'),
            provenance=dict(source_modified=False,resource_roles='active control edges or explicitly confirmed scientific input roles'))
        plan['engineering_context']=asdict(context)
        (stage/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        if unmatched or not work:raise ValueError('来源没有可验收的工程端口，缺字段语义/工程角色或所需支持；见engineering_inputs/plan.json')
        from .search_bridge import SearchNotReady,port_groups,combine_ports
        products=[];reports=[]
        for i,members in enumerate(port_groups(work)):
            port=members[0][0]
            try:
                port,mapping=combine_ports(members)
                values,report=process_port(port,members[0][1],stage/str(i),progress=progress)
            except SearchNotReady as exc:
                failure=dict(port=port.resource,variable=port.quantity,**exc.report)
                (stage/'search_blocked.json').write_text(json.dumps(failure,ensure_ascii=True,indent=2),encoding='utf8')
                for item in source_audit:item['blockers']=[dict(code=exc.report['status'],message=str(exc))]
                from ecodata.models.project_inventory import prepare_project_inventory
                diagnostic=folder/'engineering_adaptation'/'search_diagnostics'
                diagnostic.mkdir(parents=True,exist_ok=True)
                for evidence in stage.rglob('*'):
                    if evidence.is_file() and evidence.suffix in {'.json','.jsonl'}:
                        target=diagnostic/evidence.relative_to(stage);target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(evidence,target)
                return prepare_project_inventory(uploads,model,folder,source_audit=source_audit,
                    weather_audit=dict(ports=len(ports),issues=unknown,
                        completed_search_groups=len(reports),blocked_search=failure,
                        candidate_evidence='engineering_adaptation/search_diagnostics; no native writeback performed'),adaptation=adaptation,
                    attachments=folder/'engineering_adaptation')
            report['shared_search_id']=str(stage/str(i))
            for index,(native,_) in enumerate(members):
                if mapping is None:local=values
                else:
                    local=np.full(native.data[native.quantity].shape,np.nan)
                    rows=pd.DatetimeIndex(native.data.time.values).get_indexer(port.data.time.values)
                    local[rows]=values[:,mapping[index]].reshape(len(rows),-1)
                products.append((native,local));reports.append(dict(report,native_resource=native.resource))
        return write_ports(files,model,products,reports,stage,adaptation=adaptation,source_audit=source_audit,
            adaptation_folder=folder/'engineering_adaptation' if adaptation else None)


def write_ports(files,model,products,reports,stage,*,adaptation=None,source_audit=None,adaptation_folder=None):
    from existing_air_converters.native_netcdf_update import update_variable
    from ecodata.models.adapters.calmet_surface import export_surface
    from existing_air_converters.calpuff_native_bundle import read_surface,FIELDS,UNITS
    replacements={};grouped={}
    for port,values in products:grouped.setdefault(port.resource,[]).append((port,values))
    for i,(name,items) in enumerate(grouped.items()):
        with files.files[name][0]() as src:raw=src.read()
        if items[0][0].kind=='netcdf':
            current=stage/f'template_{i}.nc';current.write_bytes(raw)
            for j,(port,values) in enumerate(items):
                with xr.open_dataset(current,decode_times=False) as d:
                    old=d[port.variable].values;attrs=dict(d[port.variable].attrs)
                mask=np.isfinite(values.reshape(old.shape))
                values=np.where(mask,values.reshape(old.shape),old)
                from ecodata.models.native_source_inspection import check_native_values
                check_native_values(attrs,values[mask])
                dest=stage/f'native_{i}_{j}.nc';update_variable(current,dest,port.variable,values,allowed_mask=mask);current=dest
            replacements[name]=current.read_bytes()
        elif items[0][0].kind=='swat_weather':
            lines=raw.decode('utf-8-sig').splitlines(keepends=True)
            positions=[i for i in range(3,len(lines)) if lines[i].strip()]
            rows=[lines[i].split() for i in positions];changed=set()
            for port,values in items:
                for n,value in enumerate(values[:,0]):
                    if not np.isfinite(value):continue
                    if value<=-97 or (port.binding['kind']!='tmp' and value<0):
                        raise ValueError('处理产品不能引入负天气值或新的原生缺测标记')
                    column=port.binding['column']
                    if float(rows[n][column])!=value:
                        rows[n][column]=format(float(value),'.10g');changed.add(n)
            for n in changed:
                index=positions[n];ending='\r\n' if lines[index].endswith('\r\n') else '\n' if lines[index].endswith('\n') else ''
                lines[index]=' '.join(rows[n])+ending
            payload=(''.join(lines)).encode('utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else 'utf8')
            from ecodata.models.swat_weather_project import _records
            _records(payload,items[0][0].binding['kind'],allow_native_missing=True);replacements[name]=payload
        else:
            values,times,stations,zone=read_surface(raw)
            for port,field in items:
                k=port.binding['index'];values[:,:,k]=np.where(np.isfinite(field),field,values[:,:,k])
            frame=pd.DataFrame([dict(time=t.isoformat(),station_id=s,**dict(zip(FIELDS,values[i,j]))) for i,t in enumerate(times) for j,s in enumerate(stations)])
            cfg=dict(version='CALMET-5.8.5-SURF-2.0',timezone_west_hours=zone,time_label='hour_end',station_ids=stations,
                missing_policy='preserve',columns={f:f for f in FIELDS},units=dict(zip(FIELDS,UNITS)))
            generated,_=export_surface(frame,cfg);actual,_,_,_=read_surface(generated['SURF.DAT'])
            np.testing.assert_allclose(actual,values,equal_nan=True,atol=1e-6,rtol=0);replacements[name]=generated['SURF.DAT']
    changes=[dict(path=n,before=files.identity(n)['sha256'],after=hashlib.sha256(raw).hexdigest()) for n,raw in replacements.items()]
    manifest=dict(protocol='ecodata.converter.v1',model_id=model,status='PROJECT_UPDATED',native_ready=False,
        complete_model_input=False,native_writeback_verified=True,native_model_executed=False,
        preparation=dict(route='unified_engineering_input_ports_v1',tasks=reports),changes=changes,
        note='公共端口驱动处理及原生更新；没有独立真值时不宣称精度最优，不等于全部模型求解通过')
    if adaptation:manifest['preparation']['adaptation']=adaptation
    runs={r['shared_search_id']:r['execution'] for r in reports if r.get('shared_search_id')}
    if runs:manifest['preparation'].update(search_runs=runs,cost_accounting='Each shared_search_id is one actual search; do not sum repeated per-port receipts')
    if source_audit:
        usage=[dict(s,used=True) for s in source_audit]
        (stage/'source_usage.json').write_text(json.dumps(usage,ensure_ascii=True,indent=2,default=str),encoding='utf8')
    handoff=stage/'handoff.json';handoff.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    archive=stage/'EcoData-processed-project.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in sorted(files.files):
            if name in replacements:z.writestr(name,replacements[name])
            else:
                with files.files[name][0]() as src,z.open(name,'w',force_zip64=True) as dest:shutil.copyfileobj(src,dest,8*1024**2)
        z.write(handoff,'_ecodata/'+uuid4().hex+'/handoff.json')
        for p in stage.rglob('*.json'):z.write(p,'_ecodata/decision/'+p.relative_to(stage).as_posix())
        if adaptation_folder:
            base=Path(adaptation_folder).resolve()
            if not base.is_relative_to(stage.parent.resolve()):raise ValueError('适配证据目录越界')
            for p in sorted(base.rglob('*')):
                if p.is_file():z.write(p,'_ecodata/adaptation/'+p.relative_to(base).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('最终工程包CRC失败')
        for name,raw in replacements.items():
            if z.read(name)!=raw:raise ValueError('原生产品包读回不一致')
        for name in files.files:
            if name in replacements:continue
            digest=hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
            if digest.hexdigest()!=files.identity(name)['sha256']:
                raise ValueError('非目标工程成员发生变化：'+name)
    return dict(manifest=manifest,archive=str(archive))
