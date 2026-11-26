"""Native resources bridge the existing EngineeringProject and component writers.

No algorithm selection lives here. Binding is by actual resource identity and
native support, or by the existing model writer's reference/identity checks.
"""
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import shutil
import zipfile
from uuid import uuid4

import numpy as np
import xarray as xr

from .general import EngineeringProject
from ..project_binding import ProjectFiles


def _json_value(value):
    if isinstance(value,dict):return {k:_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_value(v) for v in value]
    if isinstance(value,np.ndarray):return _json_value(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_json_value(value),sort_keys=True,default=str).encode()).hexdigest()


def support_signature(ds, variable, *, legacy=False):
    """Hash native axes, grid/time declarations and dimension order, not values."""
    from ..scientific_schema import dataset_facts
    _,coordinates=dataset_facts(ds)
    coordinates |= set(ds.variables)&{'TFLAG','ETFLAG','Times','XLAT','XLONG','X','Y','longitude','latitude','lon','lat'}
    axes={}
    for name in sorted(coordinates-{variable}):
        a=ds[name];values=np.asarray(a.values)
        raw=json.dumps(values.tolist(),default=str).encode() if values.dtype.kind=='O' else values.tobytes()
        axes[name]=dict(dims=list(a.dims),shape=list(a.shape),dtype=str(a.dtype),attrs=dict(a.attrs),sha256=hashlib.sha256(raw).hexdigest())
    attrs={k:v for k,v in ds.attrs.items() if not k.startswith('native_') and k not in {'history','engineering_role'}}
    value=dict(dims=list(ds[variable].dims),shape=list(ds[variable].shape),axes=axes,attributes=attrs)
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest() if legacy else _digest(value)


def extract_native_field(uploads, model_id, resource, variable):
    """Public producer: actual native field + support/provenance for processing.

The caller supplies a scientific field, never an algorithm or winner. Converted
or fused products must retain this provenance and the actual native support.
"""
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack)
        if resource not in files.files:raise ValueError('工程资源不存在：'+resource)
        resource_id=files.identity(resource)
        with files.netcdf(resource,decode_times=False) as ds:
            from ..native_source_inspection import native_container_evidence
            native_container_evidence(model_id,ds.attrs,ds.variables)
            if variable not in ds or not ds[variable].attrs.get('units'):
                raise ValueError('原生字段不存在或单位未知')
            signature=support_signature(ds,variable)
            from ..scientific_schema import dataset_facts
            _,coords=dataset_facts(ds)
            coords |= set(ds)&{'TFLAG','ETFLAG','Times','XLAT','XLONG','X','Y','longitude','latitude','lon','lat'}
            product=ds[[variable,*sorted(coords-{variable})]].load()
            # netCDF4's attribute writer can reinterpret non-native-endian
            # arrays as host bytes. Preserve numeric values before serialization.
            for attrs in [product.attrs,*[a.attrs for a in product.variables.values()]]:
                for key,value in list(attrs.items()):
                    if isinstance(value,np.ndarray):
                        attrs[key]=value.astype(value.dtype.newbyteorder('='))
            product.attrs.update(native_reference_sha256=resource_id['sha256'],native_support_sha256=signature,
                                 native_model=model_id,native_variable=variable,native_support_version=2,engineering_role='processed_product')
            product[variable].attrs['native_variable']=variable
    return product


def prepare_indexed_products(uploads,model_id,folder,updates,progress=None):
    """Bind delivered point predictions back to their byte-identical native parent.

    Indices and expected background values must come from the actual upstream
    collocation. This adapter does no fitting, selection or spatial propagation.
    Independent native projects are returned separately, never silently merged.
    """
    from ..selective_resources import resource_scope
    from ..component_service import prepare_components,capabilities
    from ..project_binding import bind_project
    from ..final_check import attach_final_check
    from ecodata.engines.units import UnitEngine
    if not any(row['task']=='update_project' for row in capabilities(model_id)):
        raise ValueError('该模型未登记NetCDF原生字段写回契约')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    outputs=[];active=uploads
    with resource_scope(active,progress) as intake:
        with ExitStack() as stack:
            files=ProjectFiles(active,stack)
            binding=bind_project(active,model_id)
            if binding is None:raise ValueError('没有可绑定的实际工程；保留处理产品，不假造原生工程')
            groups={}
            for row in updates:groups.setdefault((row['source_sha256'],row['variable']),[]).append(row)
            for number,((digest,variable),rows) in enumerate(groups.items()):
                if any(r.get('purpose')=='selected_product' for r in rows):
                    rows=[r for r in rows if r.get('purpose')=='selected_product']
                hits=[r['path'] for r in binding['resources'] if r['sha256']==digest]
                if len(hits)!=1:raise ValueError('处理产品未唯一绑定实际工程输入资源')
                product=extract_native_field(active,model_id,hits[0],variable)
                occupied={}
                for row in rows:
                    if row['delivery_status'] not in {'BASELINE_RETAINED','ENHANCEMENT_RECOMMENDED'}:
                        raise ValueError('未冻结或未验收的候选不能原生写回')
                    scale,offset=UnitEngine().affine(row['unit'],product[variable].attrs['units'])
                    for point in row['points']:
                        index=tuple(point['index']);value=point['value']*scale+offset
                        if len(index)!=product[variable].ndim or any(i<0 or i>=n for i,n in zip(index,product[variable].shape)):
                            raise ValueError('上游索引超出原生支持域')
                        expected=point['expected']*scale+offset
                        if not np.isfinite(value) or not np.isclose(product[variable].values[index],expected,rtol=1e-6,atol=1e-6):
                            raise ValueError('上游实际背景与原生父资源不一致')
                        if index in occupied and occupied[index]!=value:raise ValueError('多个处理分支对同一原生位置输出冲突')
                        occupied[index]=value
                for index,value in occupied.items():product[variable].values[index]=value
                name=f'_ecodata_processed_{number}.nc';path=folder/name
                product.to_netcdf(path,engine='h5netcdf')
                files.add(name,lambda p=path:p.open('rb'),dict(upload='verified_pipeline_product',bytes=path.stat().st_size))
                stage=folder/f'project_{number}';stage.mkdir()
                if progress:progress('将冻结的实际处理产品交给公共原生写回器：'+variable)
                result=prepare_components(active,model_id,stage,progress=progress,
                    task=dict(type='model_component',operation='update_project',input=name,configuration={}))
                if not result or not result['manifest'].get('native_writeback_verified'):
                    raise ValueError('原生组件未完成工程写回；处理产品保留')
                result['manifest']['processed_product_binding']=dict(source_sha256=digest,variable=variable,
                    assigned_positions=len(occupied),tasks=[r['task'] for r in rows],
                    delivery_status=[r['delivery_status'] for r in rows],purposes=[r.get('purpose') for r in rows],selection_or_fitting_performed=False)
                attach_final_check(result,model_id,stage/'final_check')
                outputs.append(result)
    cleanup=dict(status=intake['cleanup'],selective_resources=intake)
    for result in outputs:
        result['manifest']['temporary_cleanup']=cleanup
        with zipfile.ZipFile(result['archive'],'a',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('_ecodata/indexed_product_binding.json',json.dumps(result['manifest']['processed_product_binding'],ensure_ascii=False,indent=2))
            archive.writestr('_ecodata/indexed_product_cleanup.json',json.dumps(cleanup,ensure_ascii=False,indent=2))
    return outputs


def bind_native_product(files,model_id,raw,configuration):
    """Resolve this product to a unique uploaded resource; no host paths/cases."""
    with xr.open_dataset(io.BytesIO(raw),decode_times=False) as opened:product=opened.load()
    if product.attrs.get('native_model')!=model_id:raise ValueError('产品目标模型与本次工程不符')
    digest=product.attrs.get('native_reference_sha256')
    signature=product.attrs.get('native_support_sha256')
    try:version=int(product.attrs.get('native_support_version',1))
    except (ValueError,TypeError) as exc:raise ValueError('原生支持域指纹版本未知') from exc
    if version not in (1,2):raise ValueError('不支持此原生支持域指纹版本')
    variable=product.attrs.get('native_variable')
    if not digest or not signature or not variable:raise ValueError('缺原生来源/网格时间/字段契约，不能自动绑定')
    hits=[n for n in files.files if files.prefix(n,8).startswith((b'CDF',b'\x89HDF')) and files.identity(n)['sha256']==digest]
    if len(hits)!=1:raise ValueError('原生来源须唯一匹配，实际匹配数量：'+str(len(hits)))
    target=hits[0];source_variable=configuration.get('source_variable',variable)
    if source_variable not in product:raise ValueError('处理产品缺少绑定字段')
    with files.netcdf(target,decode_times=False) as native:
        from ..native_source_inspection import native_container_evidence
        evidence=native_container_evidence(model_id,native.attrs,native.variables)
        native_signature=support_signature(native,variable)
        expected=support_signature(native,variable,legacy=True) if version==1 else native_signature
        if signature!=expected or support_signature(product,source_variable)!=native_signature:
            raise ValueError('原生网格/时间/坐标契约已变化，拒绝按形状写回')
        for key in ('standard_name','physical_variable','cell_methods','grid_mapping'):
            if product[source_variable].attrs.get(key)!=native[variable].attrs.get(key):
                raise ValueError('原生字段统计/物理语义声明已改变或未知：'+key)
        fields=[dict(name=variable,unit=native[variable].attrs.get('units'),dimensions=list(native[variable].dims),
                     shape=list(native[variable].shape),attributes=dict(native[variable].attrs))]
        domain=dict(resource=target,dimensions=dict(native.sizes),native_support_sha256=signature)
        attrs=dict(native.attrs)
    context=EngineeringProject(model_name=model_id,domains=[domain],state_variables=fields,
        metadata=dict(schema='ecodata.native-context.v1',resources=[files.identity(target)],native_attributes=attrs,native_type_evidence=evidence),
        existing_mappings=[dict(source_id='sha256:'+hashlib.sha256(raw).hexdigest(),target_id=target,
                               source_variable=source_variable,target_variable=variable,confirmed=True,
                               mapping_method='native content identity and support contract')],
        provenance=dict(source_modified=False,binding='exact content identity and native support',support_fingerprint_version=2,input_fingerprint_version=version))
    result=dict(configuration,target=target,target_sha256=digest,variable=variable,source_variable=source_variable)
    result.setdefault('preserve_nonfinite',True)
    return result,context


def bind_control_context(uploads,model_id,configuration,context):
    """Consume the existing active-input binding, when a supported control exists."""
    from ..project_binding import bind_project
    binding=bind_project(uploads,model_id)
    context.metadata['control_binding_status']='UNKNOWN'
    if binding is None:return
    edges=[r for r in binding['bindings'] if r.get('target')==configuration['target']]
    if not edges:raise ValueError('目标资源没有已验证的工程输入端口；未支持的可选端口须补契约')
    context.metadata.update(control_binding_status='BOUND',control=binding['control'],active_input_ports=edges,
                            project_binding=binding)
    for row in context.existing_mappings:row['active_input_ports']=edges
    # Verify explicit requirements, without inventing omitted model defaults.
    if model_id=='wrf' and len(edges)==1 and edges[0]['parameter']=='initial_state':
        index=edges[0]['indexes'][0]-1
        def at_domain(value):
            if isinstance(value,list):return value[index] if index<len(value) else None
            return value if index==0 else None
        native=context.metadata['native_attributes'];parameters=binding['parameters']
        checks=[];unknown=[]
        for key,attr in [('e_we','WEST-EAST_GRID_DIMENSION'),('e_sn','SOUTH-NORTH_GRID_DIMENSION')]:
            wanted=at_domain(parameters.get('domains',{}).get(key));actual=native.get(attr)
            if wanted is None or actual is None:unknown.append(key);continue
            if wanted!=actual:raise ValueError('工程控制与原生网格声明冲突：'+key)
            checks.append(dict(parameter=key,expected=wanted,actual=actual))
        stamp=native.get('START_DATE')
        if stamp:
            stamp=datetime.strptime(str(stamp),'%Y-%m-%d_%H:%M:%S')
            for key in ('year','month','day','hour','minute','second'):
                wanted=at_domain(parameters.get('time_control',{}).get('start_'+key))
                if wanted is None:unknown.append('start_'+key);continue
                if wanted!=getattr(stamp,key):raise ValueError('工程控制与原生起始时间冲突：start_'+key)
                checks.append(dict(parameter='start_'+key,expected=wanted,actual=getattr(stamp,key)))
        else:unknown.append('native START_DATE')
        context.metadata['control_requirement_checks']=dict(checked=checks,unknown=unknown,time_basis='native model calendar; no timezone assumed')


def _swat_update(files,operation,generated,configuration):
    roots=[PurePosixPath(n).parent for n in files.files if PurePosixPath(n).name=='file.cio']
    if not roots:return None
    if len(roots)!=1:raise ValueError('多个SWAT+工程根，不能猜选写回目标')
    root=roots[0]
    inert={'.exe','.dll','.bat','.cmd','.ps1','.py','.sh','.com','.scr','.lnk'}
    native={PurePosixPath(n).relative_to(root).as_posix():None for n in files.files
            if PurePosixPath(n).is_relative_to(root) and PurePosixPath(n).suffix.lower() not in inert
            and not PurePosixPath(n).relative_to(root).as_posix().startswith('_ecodata/')}
    for name in native:
        with files.files[(root/name).as_posix()][0]() as stream:native[name]=stream.read()
    if operation=='weather':
        from ..swat_weather_project import attach_weather,_layout,_table,_weather_metadata
        station=configuration.get('station',{}).get('name')
        _,_,climate,_=_layout(native)
        _,header,rows=_table(native[climate[1]])
        selected=[r for r in rows if r[header.index('name')]==station]
        if len(selected)!=1:raise ValueError('转换天气站点ID不能唯一绑定工程站点')
        old_name=selected[0][header.index('pcp')]
        if old_name not in native:raise ValueError('缺少可验证的工程站点天气坐标，不猜空间绑定')
        old_meta=_weather_metadata(native[old_name]);new_meta=_weather_metadata(generated['station.pcp'])
        if any(old_meta[k]!=new_meta[k] for k in ('latitude','longitude','elevation_m')):
            raise ValueError('转换天气与工程原站点坐标/高程不一致')
        blob=io.BytesIO()
        with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as z:
            for name,payload in native.items():z.writestr(name,payload)
        # Existing native checks retain the period and other station references.
        payload,report=attach_weather(blob.getvalue(),generated,station)
        with zipfile.ZipFile(io.BytesIO(payload)) as z:updated={n:z.read(n) for n in z.namelist()}
    else:
        from ..swat_inputs_project import attach_inputs,TABLES
        mode='landuse_database' if operation=='landuse' else operation
        if mode not in TABLES:raise ValueError('此SWAT+组件尚无完整工程写回契约：'+operation)
        blob=io.BytesIO()
        with zipfile.ZipFile(blob,'w') as z:
            for name,payload in generated.items():z.writestr(name,payload)
        updated,report=attach_inputs(native,blob.getvalue(),mode)
    replacements={(root/name).as_posix():value for name,value in updated.items() if native.get(name)!=value}
    context=EngineeringProject(model_name='swat_plus',
                               state_variables=[dict(name=k,unit=v,role='forcing') for k,v in report.get('unit_contract',{}).items()],
                               metadata=dict(schema='ecodata.native-context.v1',root=str(root),validation=report),
                               provenance=dict(source_modified=False,writer='existing SWAT+ attachment'))
    changes=report.get('changes',report.get('replacements',[]))
    context.existing_mappings=[dict(source_id='sha256:'+r.get('sha256',r.get('after_sha256','')),
        target_id=(root/r.get('target',r.get('file',''))).as_posix(),confirmed=True,
        mapping_method='existing native station/record identity and reference validation') for r in changes]
    return replacements,context


def _calpuff_update(files,operation,generated):
    if operation!='surface_meteorology':return None
    from ..project_inventory import _calpuff
    from existing_air_converters.calpuff_native_bundle import read_surface,native_name
    plans=_calpuff(files)
    if not plans:return None
    payload=generated.get('SURF.DAT')
    if payload is None:raise ValueError('地面气象writer未产生SURF.DAT')
    _,times,stations,zone=read_surface(payload)
    candidates={e['target'] for p in plans for e in p['bindings'] if e['parameter']=='SRFDAT'}
    eligible=[]
    for name in candidates:
        with files.files[name][0]() as stream:old=stream.read()
        _,old_times,old_stations,old_zone=read_surface(old)
        if list(times)==list(old_times) and stations==old_stations and zone==old_zone:eligible.append(name)
    if len(eligible)!=1:raise ValueError('CALMET时间/站点/时区匹配目标不唯一：'+str(len(eligible)))
    target=eligible[0]
    context=EngineeringProject(model_name='calpuff',state_variables=[dict(name='surface_meteorology',resource=target,role='forcing')],
        domains=[dict(stations=stations,native_record_start=times[0].isoformat(),native_record_end=times[-1].isoformat(),time_label='hour_end')],
        metadata=dict(schema='ecodata.native-context.v1',stations=stations,timezone_west_hours=zone),
        provenance=dict(source_modified=False,writer='existing SURF2 reader and surface writer'))
    context.existing_mappings=[dict(source_id='sha256:'+hashlib.sha256(payload).hexdigest(),target_id=target,
        confirmed=True,mapping_method='active SRFDAT role, station/time/zone contract')]
    # Updating CALMET forcing does not regenerate its downstream CALPUFF METDAT.
    context.metadata['runtime_dependencies']=[dict(control=p['control']['path'],kind=p['kind'],
        declared_metdat=(PurePosixPath(p['control']['path']).parent/native_name(p['parameters'],'METDAT')).as_posix(),
        role='generated_output' if p['kind']=='CALMET' else 'meteorological_input',
        status='NOT_REGENERATED') for p in plans if p.get('active_dependency_plan')]
    return {target:payload},context


def finish_project(uploads,model_id,folder,operation,generated,configuration,validation,*,context=None):
    """Preserve uploaded layout and stream a copied project with verified edits."""
    folder=Path(folder)
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack)
        if context is not None:
            resources=context.metadata.get('resources',[])
            if context.model_name!=model_id or len(resources)!=1 or len(context.state_variables)!=1:
                raise ValueError('统一工程上下文的模型/目标资源/字段不明确')
            resource=resources[0];field=context.state_variables[0]
            if configuration['target']!=resource['path'] or configuration['variable']!=field['name']:
                raise ValueError('写回参数与统一工程上下文不一致')
            if files.identity(resource['path'])['sha256']!=resource['sha256']:
                raise ValueError('绑定后工程资源发生变化')
            control=context.metadata.get('control')
            if control and files.identity(control['path'])['sha256']!=control['sha256']:
                raise ValueError('绑定后工程控制文件发生变化')
            replacements={resource['path']:generated['native_updated.nc']}
        elif model_id=='swat_plus':
            result=_swat_update(files,operation,generated,configuration)
            if result is None:return None
            replacements,context=result
        elif model_id=='calpuff':
            result=_calpuff_update(files,operation,generated)
            if result is None:return None
            replacements,context=result
        else:return None
        # EngineeringProject is the existing shared structure, no parallel model.
        record=dict(schema='ecodata.native-context.v1',model_name=context.model_name,
                    domains=context.domains,state_variables=context.state_variables,existing_mappings=context.existing_mappings,
                    metadata=context.metadata,provenance=context.provenance,source_resource=validation.get('source_resource'))
        changes=[dict(path=n,before=files.identity(n)['sha256'] if n in files.files else None,
                      after=hashlib.sha256(payload).hexdigest()) for n,payload in sorted(replacements.items())]
        record['changes']=changes
        context_path=folder/'engineering_context.json';context_path.write_text(json.dumps(_json_value(record),ensure_ascii=False,indent=2,default=str),encoding='utf8')
        audit_root='_ecodata/'+uuid4().hex
        archive=folder/'EcoData-updated-project.zip'
        inventory=[]
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            included=set(files.files)
            binding=context.metadata.get('project_binding')
            if binding:
                included={r['path'] for r in binding['resources']}|{binding['control']['path']}
                included.update(r['path'] for r in binding.get('equivalent_controls',[]))
            for name in sorted(included|set(replacements)):
                if name in replacements:z.writestr(name,replacements[name])
                else:
                    with files.files[name][0]() as source,z.open(name,'w',force_zip64=True) as dest:
                        shutil.copyfileobj(source,dest,8*1024**2)
                inventory.append(name)
            z.write(context_path,audit_root+'/engineering_context.json')
        with zipfile.ZipFile(archive) as z:
            if z.testzip():raise ValueError('更新工程CRC错误')
            # Read back every source member; unchanged bytes verified by hashes.
            for name in inventory:
                with z.open(name) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                expected=hashlib.sha256(replacements[name]).hexdigest() if name in replacements else files.identity(name)['sha256']
                if digest!=expected:raise ValueError('工程副本读回不一致：'+name)
        manifest=dict(protocol='ecodata.converter.v1',model_id=model_id,status='PROJECT_UPDATED',native_ready=False,
                      native_writeback_verified=True,native_model_executed=False,complete_model_input=False,
                      preparation=dict(route='shared_native_context_v1',tasks=[dict(operation=operation,validation=validation)]),
                      context=context_path.name,context_archive_path=audit_root+'/engineering_context.json',changes=changes,archive=str(archive),
                      note='上传资源布局副本及原生写回；非原生模型运行或全部工程闭包认证')
        (folder/'handoff.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        with zipfile.ZipFile(archive,'a') as z:z.write(folder/'handoff.json',audit_root+'/handoff.json')
        with zipfile.ZipFile(archive) as z:
            if z.read(audit_root+'/handoff.json')!=(folder/'handoff.json').read_bytes():raise ValueError('工程交接记录回读失败')
    return dict(manifest=manifest,archive=str(archive))
