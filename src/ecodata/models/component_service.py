"""Portable model task contracts over the existing native component writers.

The catalog selects scientific operations, never a case, machine or algorithm
winner. All source paths resolve inside the uploaded resource set.
"""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import io
import json
import zipfile

import pandas as pd
import xarray as xr

from ecodata.models.project_binding import ProjectFiles, safe_name


# (model, task) -> existing implementation and its documented data interface.
CATALOG = {
    ('swat_plus','weather'): ('swat_plus_upload','export_uploaded_weather','bytes'),
    ('swat_plus','soil'): ('swat_plus_soil','export_soils','bytes'),
    ('swat_plus','landuse'): ('swat_plus_landuse','export_landuse','bytes'),
    ('swat_plus','wgn'): ('swat_plus_wgn','export_wgn','bytes'),
    ('swat_plus','hru_attributes'): ('swat_plus_hru','export_hru_bundle','bytes'),
    ('swat_plus','aquifer_parameters'): ('swat_plus_aquifer','export_aquifer_bundle','bytes'),
    ('swat_plus','connections'): ('swat_plus_connections','export_connections','bytes'),
    ('swat_plus','spatial_landuse'): ('swat_plus_spatial_landuse','export_spatial_landuse','bytes'),
    ('swat_plus','management_schedule'): ('swat_plus_management','export_management','bytes'),
    ('camx','emissions'): ('camx_emissions','export_gridded_emissions','table'),
    ('camx','initial_conditions'): ('camx_initial','export_initial','table'),
    ('camx','boundary_conditions'): ('camx_boundary','export_boundary','table'),
    ('cmaq','emissions'): ('cmaq_gridded_emissions','export_gridded_emissions','dataset'),
    ('cmaq','boundary_conditions'): ('cmaq_boundary','export_boundary','dataset'),
    ('cmaq','bcon_profile'): ('cmaq_bcon_profile','export_bcon_profile','bytes'),
    ('calpuff','surface_meteorology'): ('calmet_surface','export_surface','table'),
    ('calpuff','precipitation'): ('calmet_precipitation','export_precipitation','table'),
    ('calpuff','upper_air'): ('calmet_upper_air','export_upper_air','table'),
    ('calpuff','geography'): ('calmet_geography','export_geography','dataset'),
    ('calpuff','receptors'): ('calpuff_receptors','export_receptors','bytes'),
    ('calpuff','point_sources'): ('calpuff_point_sources','export_points','bytes'),
    ('camx','update_field'): ('native_netcdf_update','update_variable','native_update'),
    ('cmaq','update_field'): ('native_netcdf_update','update_variable','native_update'),
    ('wrf','update_project'): ('native_netcdf_update','update_variable','native_update'),
    ('camx','update_project'): ('native_netcdf_update','update_variable','native_update'),
    ('cmaq','update_project'): ('native_netcdf_update','update_variable','native_update'),
}

TASK_MODES = {'weather':None,'landuse':'landuse_database','surface_meteorology':None,
              'emissions':'gridded_emissions','initial_conditions':'initial_concentrations',
              'boundary_conditions':'boundary_concentrations','point_sources':'constant_points'}


def capabilities(model):
    return [dict(task=task,backend=module+'.'+function,input_kind=kind)
            for (name,task),(module,function,kind) in sorted(CATALOG.items()) if name==model]


def generic_requirements(model):
    return dict(schema='ecodata.converter.requirements.v1',model_id=model,
        role='component',fields=[],coordinates={},dependencies=[],
        component_dispatch=True,components=capabilities(model),
        scope='Uploaded task/resource contracts; no default station, period or engineering case')


def unsupported_conversion(manifest_path, output_dir):
    """Normalised data are not permission to attach an unrelated old project."""
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    raise ValueError('数据已准备，但尚缺本次原生任务/工程契约；请声明任务及上传的目标资源。支持：'+
                     ', '.join(row['task'] for row in capabilities(manifest['model_id'])))


def _auto_weather(files):
    """Unambiguous complete daily weather can be exported without a sidecar."""
    candidates=[]
    needed={'precipitation','air_temperature_max','air_temperature_min'}
    for name in sorted(files.files):
        if not files.prefix(name,8).startswith((b'CDF',b'\x89HDF')):continue
        with files.files[name][0]() as stream, xr.open_dataset(stream) as ds:
            if not needed <= set(ds.data_vars):continue
            if ds.attrs.get('crs')!='EPSG:4326':raise ValueError('SWAT+天气必须明确站点坐标基准')
            coordinates={}
            for key,aliases in [('latitude',('lat','latitude')),('longitude',('lon','longitude')),('elevation',('elevation','elevation_m'))]:
                found=[n for n in aliases if n in ds.coords and ds[n].size==1]
                if len(found)==1:coordinates[key]=float(ds[found[0]].values.reshape(-1)[0])
                elif key in ds.attrs:coordinates[key]=float(ds.attrs[key])
            if len(coordinates)!=3:continue  # Missing station metadata, not guessed.
            times=pd.DatetimeIndex(ds.time.values)
            if len(times)<2 or not (times[1:]-times[:-1]==pd.Timedelta(days=1)).all():
                raise ValueError('SWAT+天气需明确逐日序列，小时处理需先声明聚合任务')
            station=str(ds.attrs.get('station_id','station_'+files.identity(name)['sha256'][:10]))
            configuration=dict(project_name='uploaded_weather',start=str(times[0].date()),end=str(times[-1].date()),
                               station=dict(name=station,**coordinates))
            additional=[n for n in ('solar_radiation','relative_humidity','wind_speed') if n in ds]
            if additional:
                if ds.attrs.get('target_model_version')!='62.0.0':
                    raise ValueError('扩展SWAT+天气字段需明确target_model_version=62.0.0')
                configuration.update(additional_weather=additional,target_model_version='62.0.0')
            candidates.append(dict(operation='weather',input=name,configuration=configuration))
    if len(candidates)>1:raise ValueError('多个完整天气来源需先完成融合/选择，不能猜选写出来源')
    return candidates[0] if candidates else None


def prepare_components(uploads, model_id, folder, *, task=None, progress=None):
    """Public page/toolkit route, accepting goals and scientific metadata only.

An optional JSON sidecar uses ecodata.model-task.v1. It cannot select a callable,
execute scripts or access a host pathname. Algorithms stay in the public service.
"""
    folder=Path(folder)
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack)
        declarations=[]
        for name in sorted(files.files):
            if not name.lower().endswith('.json'):continue
            with files.files[name][0]() as stream:
                try:document=json.load(stream)
                except (ValueError,UnicodeError):continue
            if isinstance(document,dict) and document.get('schema')=='ecodata.model-task.v1':
                declarations.append((name,document))
        if task and task.get('type')=='model_component':
            declarations.append(('',dict(task,schema='ecodata.model-task.v1',model_id=model_id)))
        if len(declarations)>1:raise ValueError('多个原生任务声明，不能自动选择')
        if declarations:
            declaration_name,request=declarations[0]
            if request.get('model_id')!=model_id:raise ValueError('原生任务目标模型与页面不一致')
            if {'algorithm','callable','python','command','winner'} & set(request):
                raise ValueError('任务声明不能指定算法赢家或执行程序')
        else:
            products=[]
            if model_id in {'wrf','camx','cmaq'}:
                for name in sorted(files.files):
                    if not files.prefix(name,8).startswith((b'CDF',b'\x89HDF')):continue
                    with files.files[name][0]() as stream, xr.open_dataset(stream,decode_times=False) as ds:
                        if ds.attrs.get('engineering_role')=='processed_product' and ds.attrs.get('native_reference_sha256'):
                            products.append(name)
            if len(products)>1:raise ValueError('多个工程处理产品，须明确本次目标资源；不猜选')
            request=dict(operation='update_project',input=products[0],configuration={}) if products else (_auto_weather(files) if model_id=='swat_plus' else None)
            if request is None:return None
            declaration_name=''
        operation=request.get('operation');key=(model_id,operation)
        if key not in CATALOG:
            raise ValueError('未实现此原生组件任务：'+str(key)+'；支持：'+', '.join(r['task'] for r in capabilities(model_id)))
        source_name=request.get('input')
        if not isinstance(source_name,str):raise ValueError('原生任务需要明确输入资源身份')
        if declaration_name:
            source_name,_=files.resolve(declaration_name,source_name)
        elif source_name not in files.files:raise ValueError('输入资源不在本次上传资料内')
        source_identity=files.identity(source_name)
        if request.get('input_sha256') and source_identity['sha256']!=request['input_sha256']:
            raise ValueError('原生任务输入指纹不一致')
        with files.files[source_name][0]() as stream:raw=stream.read()
        module,function,kind=CATALOG[key]
        configuration=request.get('configuration')
        if not isinstance(configuration,dict):raise ValueError('缺少本次任务的单位、时间、网格等科学配置')
        # External process dependencies are not accepted as arbitrary task data.
        if {'python','command','executable','legacy_source','legacy_python'} & set(configuration):
            raise ValueError('原生任务配置不能启动外部程序')
        destination=folder/'components';destination.mkdir(exist_ok=False)
        context=None
        if kind=='native_update':
            if operation=='update_project':
                from ecodata.models.engineering.native import bind_native_product,bind_control_context
                if progress:progress('匹配本次工程来源、网格时间与字段语义')
                configuration,context=bind_native_product(files,model_id,raw,configuration)
                bind_control_context(uploads,model_id,configuration,context)
            if progress:progress('执行原生字段写回、质量门和独立回读')
            generated,validation=_update_field(files,'' if context is not None else declaration_name,raw,configuration,destination)
        else:
            from ecodata.round1 import convert
            expected_mode=TASK_MODES.get(operation,operation)
            if configuration.get('mode',expected_mode)!=expected_mode:
                raise ValueError('任务语义与转换配置mode冲突')
            configuration=dict(configuration)
            if expected_mode is not None:configuration['mode']=expected_mode
            if progress:progress('调用已有原生组件写出器：'+model_id+' / '+operation)
            station_index=None
            if request.get('weather_station_index'):
                index_name,_=files.resolve(declaration_name or 'request.json',request['weather_station_index'])
                with files.files[index_name][0]() as stream:station_index=stream.read()
            generated,validation=convert(model_id,raw,configuration,weather_station_index=station_index)
        inventory=[]
        for name,payload in generated.items():
            name=safe_name(name);path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
            if not isinstance(payload,bytes) or not payload:raise ValueError('写出器未返回非空真实文件')
            path.write_bytes(payload)
            if path.read_bytes()!=payload:raise ValueError('写出产品回读不一致')
            inventory.append(dict(path=path.relative_to(folder).as_posix(),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
        from ecodata.models.engineering.native import finish_project
        validation=dict(validation,source_resource=source_identity)
        if progress:progress('核对工程引用并生成经过校验的工程副本')
        attached=finish_project(uploads,model_id,folder,operation,generated,configuration,validation,context=context)
        if attached is not None:return attached
        manifest=dict(protocol='ecodata.converter.v1',model_id=model_id,status='COMPONENT_EXPORTED',native_ready=False,
            component_format_generated=True,native_model_executed=False,complete_model_input=False,
            preparation=dict(route='model_component_service_v1',tasks=[dict(operation=operation,backend=module+'.'+function,
                source=source_identity,configuration=configuration,validation=validation)]),
            output_inventory=inventory,note='真实模型组件文件；不等于全部工程齐备或原生模型已读取运行')
        path=folder/'handoff.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        archive=folder/'EcoData-native-components.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            z.write(path,path.name)
            for row in inventory:z.write(folder/row['path'],row['path'])
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('组件包CRC不一致')
        return dict(manifest=manifest,archive=str(archive))


def _update_field(files, declaration_name, raw, configuration, destination):
    """Existing lossless native patcher, bound to THIS uploaded target fingerprint."""
    import numpy as np
    from netCDF4 import Dataset
    from ecodata.engines.units import UnitEngine
    from existing_air_converters.native_netcdf_update import update_variable
    required={'target','target_sha256','variable'}
    if not required <= set(configuration):raise ValueError('字段写回缺目标资源、内容指纹或变量')
    target_name,_=files.resolve(declaration_name or 'request.json',configuration['target'])
    target=files.identity(target_name)
    if target['sha256']!=configuration['target_sha256']:raise ValueError('目标原生文件指纹不一致')
    with xr.open_dataset(io.BytesIO(raw),decode_times=False) as opened:source=opened.load()
    if source.attrs.get('native_reference_sha256')!=target['sha256']:
        raise ValueError('产品缺少与目标网格/时间绑定的native_reference_sha256，不能只按形状猜测')
    variable=configuration['variable'];source_variable=configuration.get('source_variable',variable)
    from ecodata.evaluation.quality_gates import RawInputQualityGate,ResultQualityGate
    raw_quality=RawInputQualityGate().validate(source,variable=source_variable,
                                             intended_operation='conversion',requires_spatial=False)
    (destination/'raw_quality.json').write_text(json.dumps(raw_quality,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    if raw_quality['status']=='BLOCK':raise ValueError('原生写回输入质量门阻止处理，见raw_quality.json')
    template=destination/'template.nc'
    with files.files[target_name][0]() as stream,template.open('wb') as dest:
        import shutil
        shutil.copyfileobj(stream,dest,8*1024**2)
    with Dataset(template) as native:
        if variable not in native.variables or source_variable not in source:raise ValueError('指定目标字段不存在')
        v=native[variable]
        if source[source_variable].dims!=v.dimensions or source[source_variable].shape!=v.shape:
            raise ValueError('产品与原生变量的维度顺序/形状不一致')
        unit=str(getattr(v,'units','')).strip()
        if not unit:raise ValueError('原生变量单位未知')
        converted=UnitEngine().convert(source[source_variable],unit).values
        original=v[:] if configuration.get('mask_variable') or configuration.get('preserve_nonfinite') else None
        native_attrs={k:v.getncattr(k) for k in v.ncattrs()}
    mask=None
    if configuration.get('mask_variable'):
        a=source[configuration['mask_variable']]
        if a.dims!=source[source_variable].dims or a.dtype.kind!='b':raise ValueError('写回掩码须与字段同形的布尔数组')
        mask=a.values
        converted=np.where(mask,converted,original)
    elif configuration.get('preserve_nonfinite'):
        mask=np.isfinite(converted)
        converted=np.where(mask,converted,original)
    proposed=converted[mask] if mask is not None else converted
    from ecodata.models.native_source_inspection import check_native_values
    check_native_values(native_attrs,proposed)
    output=destination/'native_updated.nc'
    report=update_variable(template,output,variable,converted,allowed_mask=mask)
    if report['template_sha256']!=target['sha256']:raise ValueError('读取过程中原生目标已变化，拒绝交付')
    with xr.open_dataset(output,decode_times=False,mask_and_scale=False) as actual:
        expected=np.where(mask,converted,original) if mask is not None else converted
        # Patcher independently verifies all non-target fields and unchanged mask.
        np.testing.assert_allclose(actual[variable].values,expected,rtol=1e-6,atol=1e-6,equal_nan=True)
        normalized=xr.decode_cf(actual[[variable]],decode_times=False).copy()
        normalized[variable].attrs['units']=unit
        result_quality=ResultQualityGate().validate(source,normalized,variable=variable,expected_output_unit=unit)
        (destination/'result_quality.json').write_text(json.dumps(result_quality,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        if result_quality['status']=='BLOCK':raise ValueError('原生写回结果质量门未通过，见result_quality.json')
    payload=output.read_bytes();output.unlink();template.unlink()
    return {'native_updated.nc':payload},dict(native_update=report,target=target,variable=variable,
        raw_quality=raw_quality,result_quality=result_quality,
        source_grid_time_binding=source.attrs['native_reference_sha256'],native_model_executed=False,
        complete_model_input=False,scope='单个原生字段写回，非完整模型运行或跨变量物理平衡认证')
