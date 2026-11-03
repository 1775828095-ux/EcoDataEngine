"""Platform-owned WRF bundle ingestion; preserves daily OISST time and masks.

This route exports an SST supplement, not an updated runnable WRF experiment.
No user scripts, reference simulation output, or registered old case is consumed.
"""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import zipfile
import numpy as np
import xarray as xr
import re


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _oisst(raw):
    with xr.open_dataset(io.BytesIO(raw)) as opened:
        title = str(opened.attrs.get('title', ''))
        if 'NOAA/NCEI' not in title or 'Optimum Interpolation' not in title or 'sst' not in opened:
            return None
        if not {'time', 'lat', 'lon'} <= set(opened.coords):
            raise ValueError('OISST 文件缺少 time/lat/lon 坐标')
        extra=set(opened.data_vars)-{'sst','anom','err','ice'}
        if extra:raise ValueError('OISST存在尚未识别用途的字段：'+','.join(sorted(extra)))
        ds = opened[['sst']].load()
        ds.attrs['oisst_auxiliary_fields_not_temperature_sources']=json.dumps(sorted(set(opened.data_vars)-{'sst'}))
        for dim in tuple(ds.sst.dims):
            if dim not in {'time','lat','lon'}:
                if dim != 'zlev' or ds.sizes[dim] != 1:
                    raise ValueError('OISST 出现未支持的额外维度：' + dim)
                ds = ds.isel({dim:0}, drop=True)
        if ds.sst.attrs.get('units') not in {'Celsius','degC','degrees_Celsius','K'}:
            raise ValueError('OISST 温度单位无法确定')
        if ds.lat.attrs.get('units') != 'degrees_north' or ds.lon.attrs.get('units') != 'degrees_east':
            raise ValueError('OISST 经纬度单位与产品定义不符')
        # Longitude convention conversion and ordering do not resample values.
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby('lon').sortby('lat')
        return ds


def _sst_update_enabled(raw):
    """Read only the active namelist's SST-update switch; WRF's default is off."""
    text = raw.decode('latin1')
    text = '\n'.join(line.split('!', 1)[0] for line in text.splitlines())
    physics = re.search(r'(?is)&physics\b(.*?)(?:/|&end\b)', text)
    if physics is None:
        return False
    matches = re.findall(r'(?im)^\s*sst_update\s*=\s*([^\n]+)', physics.group(1))
    if not matches:
        return False
    for assignment in matches:
        # A following namelist assignment may share a line after a comma.
        assignment = re.split(r',\s*[A-Za-z][A-Za-z0-9_]*\s*=', assignment, maxsplit=1)[0]
        values = [value.strip().strip(',').strip() for value in assignment.split(',') if value.strip(' ,')]
        if not values or any(value not in {'0', '1'} for value in values):
            raise ValueError('namelist.input 中 sst_update 不是可识别的 0/1 设置')
        if any(value == '1' for value in values):
            return True
    return False


def prepare(uploads, folder, progress=None):
    from ecodata.interface.engineering_upload import MAX_FILE_BYTES, MAX_TOTAL_BYTES
    folder = Path(folder)
    inventory, sources, archives = [], [], []
    native = []
    namelist_settings = []
    catalog_files = []
    dependencies = []
    total = 0
    for upload in uploads or []:
        if not str(upload.name).lower().endswith('.zip'):
            return None
        raw = upload.getvalue()
        archives.append(dict(name=upload.name, sha256=sha(raw), bytes=len(raw)))
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            seen = set()
            for item in z.infolist():
                if item.is_dir():
                    continue
                name = PurePosixPath(item.filename.replace('\\','/'))
                if name.is_absolute() or '..' in name.parts or ':' in name.parts[0] or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('不安全的压缩包成员：'+item.filename)
                if str(name).casefold() in seen:
                    raise ValueError('压缩包存在重复路径：'+str(name))
                seen.add(str(name).casefold())
                total += item.file_size
                if item.file_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                    raise ValueError('压缩包成员或解压总量超过10 GB限制')
                base = name.name.lower()
                record = dict(file=str(name), size=item.file_size)
                if base.startswith('wrfout_') or 'reference_outputs' in name.parts:
                    from ecodata.models.source_catalog import header
                    facts=header(z.read(item))
                    if not facts or 'Times' not in facts['variables'] or 'WRF' not in str(facts['attributes'].get('TITLE','')):
                        record['role']='unrecognised_netcdf'
                    else:
                        record.update(role='reference_simulation_output_not_input',
                            evidence='原生WRF输出头与Times已核对；不会作为观测真值或新输入')
                elif base.startswith(('wrfinput_d','wrfbdy_d')):
                    from netCDF4 import Dataset
                    payload=z.read(item)
                    with Dataset('memory',memory=payload) as d:
                        if 'REAL_EM' not in str(getattr(d,'TITLE','')):
                            raise ValueError('原生工程文件类型与内容不一致：'+str(name))
                        record.update(role='native_engineering_dependency',sha256=sha(payload),
                            title=d.TITLE,dimensions={k:len(v) for k,v in d.dimensions.items()})
                        if base.startswith('wrfinput'):
                            if not {'XLAT','XLONG','LANDMASK','Times'} <= set(d.variables):
                                raise ValueError('WRF工程缺少网格、海陆掩码或时间字段：'+str(name))
                            record['extent']=[float(d['XLAT'][:].min()),float(d['XLAT'][:].max()),float(d['XLONG'][:].min()),float(d['XLONG'][:].max())]
                            record['contains_initial_sst']='SST' in d.variables
                        native.append(record)
                    catalog_files.append((str(name),payload))
                elif base.endswith(('.nc','.nc4')):
                    payload=z.read(item)
                    catalog_files.append((str(name),payload))
                    ds=_oisst(payload)
                    if ds is None:
                        record.update(role='unrecognised_netcdf', sha256=sha(payload))
                    else:
                        record.update(role='sst_analysis',sha256=sha(payload),
                            auxiliary_fields_not_temperature_sources=json.loads(ds.attrs['oisst_auxiliary_fields_not_temperature_sources']),
                            processing='sst selected; anomaly, analysis error and sea-ice fraction recorded as auxiliary quantities, never mixed as temperatures')
                        sources.append((record,ds))
                elif base.endswith(('.tar.gz','.tgz')):
                    # Compression is identified structurally, never decoded as ISD text.
                    import tarfile
                    with tarfile.open(fileobj=io.BytesIO(z.read(item)), mode='r:gz') as tar:
                        members=[]
                        for member in tar:
                            if not member.isfile():
                                if member.isdir(): continue
                                raise ValueError('嵌套资料包含链接或特殊成员')
                            p=PurePosixPath(member.name)
                            if p.is_absolute() or '..' in p.parts or ':' in p.parts[0]:
                                raise ValueError('嵌套资料包路径不安全')
                            total += member.size
                            if member.size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                                raise ValueError('嵌套资料展开后超过10 GB限制')
                            stream=tar.extractfile(member); magic=stream.read(8)
                            if magic[:4]!=b'GRIB' or magic[7] not in (1,2):
                                raise ValueError('嵌套资料不是可识别的GRIB气象场：'+member.name)
                            payload=magic+stream.read()
                            dep=folder/'dependencies'/('forcing_'+str(len(dependencies))+'.grib')
                            dep.parent.mkdir(parents=True,exist_ok=True);dep.write_bytes(payload)
                            dependency=dict(path=dep.relative_to(folder).as_posix(),sha256=sha(payload),
                                            role='raw_atmospheric_forcing',source_member=member.name)
                            dependencies.append(dependency)
                            members.append(dict(file=member.name,size=member.size,format='GRIB'+str(magic[7]),sha256=dependency['sha256']))
                    record.update(role='raw_atmospheric_forcing_dependency',members=members,
                                  processing='retained as dependency; not treated as SST or discarded')
                elif base=='namelist.input':
                    payload=z.read(item)
                    sst_update=_sst_update_enabled(payload)
                    record.update(role='engineering_configuration',sha256=sha(payload),sst_update_enabled=sst_update)
                    namelist_settings.append(dict(file=str(name),sst_update_enabled=sst_update))
                    dep=folder/'dependencies'/'namelist.input'
                    dep.parent.mkdir(parents=True,exist_ok=True)
                    if dep.exists():raise ValueError('存在多个namelist.input，不能自动选择')
                    dep.write_bytes(payload)
                    catalog_files.append((str(name),dep.read_bytes()))
                    dependencies.append(dict(path=dep.relative_to(folder).as_posix(),sha256=record['sha256'],role='engineering_configuration',source_member=str(name)))
                elif 'wrf_runtime_tables' in name.parts:
                    record['role']='runtime_lookup_table_not_fusion_input'
                elif name.suffix.lower() in {'.json','.md','.txt'} or 'license' in base or 'readme' in base:
                    record['role']='documentation_not_processing_instruction'
                else:
                    record['role']='unrecognised'
                if 'sha256' not in record:
                    with z.open(item) as stream:record['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
                inventory.append(record)
    if not sources and native:
        folder.mkdir(parents=True,exist_ok=True)
        update_requested=any(item['sst_update_enabled'] for item in namelist_settings)
        initial_sst_present=any(item.get('contains_initial_sst',False) for item in native)
        disposition=dict(route='wrf_daily_sst_bundle_v1',status='SST_SUPPLEMENT_NOT_REQUESTED',
            sst_update_enabled=update_requested,existing_initial_sst_present=initial_sst_present,
            reason='工程未启用 sst_update；WPS 专项 SST 更新入口不适用，交由常规资料识别继续处理')
        save(folder/'bundle_inventory.json',dict(archives=archives,files=inventory,route_decision=disposition))
        if update_requested:
            raise ValueError('工程已启用 sst_update=1，但没有识别到可用的独立时变SST资料。现有wrfinput中的初始SST不能代替wrflowinp更新资料。')
        return None
    if not sources:
        return None
    folder.mkdir(parents=True,exist_ok=True)
    unknown=[r['file'] for r in inventory if r['role'] in {'unrecognised','unrecognised_netcdf'}]
    save(folder/'bundle_inventory.json',dict(archives=archives,files=inventory))
    if unknown:
        raise ValueError('以下资料用途不能确定，未静默跳过：'+'、'.join(unknown))
    from ecodata.models.source_catalog import build_catalog, write_catalog, processing_plan, single_source_action
    catalog=build_catalog(catalog_files,'wrf')
    catalogued={r['file'] for r in catalog['files']}
    roles={'reference_simulation_output_not_input':'reference_output',
           'runtime_lookup_table_not_fusion_input':'engineering_dependency',
           'raw_atmospheric_forcing_dependency':'engineering_dependency',
           'documentation_not_processing_instruction':'documentation'}
    for item in inventory:
        if item['file'] not in catalogued:
            catalog['files'].append(dict(file=item['file'],sha256=item['sha256'],bytes=item['size'],
                role=roles.get(item['role'],'unknown'),evidence=[item.get('evidence',item['role'])],
                members=item.get('members',[])))
    write_catalog(folder,catalog)
    if catalog['issues']:
        raise ValueError('工程资料识别未通过：'+'；'.join(i['file']+'：'+i['message'] for i in catalog['issues']))
    if progress: progress('核对同产品重复资料与原始日期；保留海陆及缺测掩码')
    # A derived subset is eligible only if every supplied value matches originals.
    # Identity comes from product metadata + exact coordinate/value comparison.
    one_time=[s for s in sources if s[1].sizes['time']==1]
    combined=[s for s in sources if s[1].sizes['time']>1]
    relations=[]
    if one_time and combined:
        originals=xr.concat([ds for _,ds in one_time],dim='time').sortby('time')
        if len(combined)!=1:
            raise ValueError('存在多个多日OISST产品，尚不能自动确定目标范围')
        selected_record, selected=combined[0]
        try:
            check=originals.sst.sel(time=selected.time,lat=selected.lat,lon=selected.lon)
        except KeyError as exc:
            raise ValueError('区域产品的坐标/日期不能在原始OISST中一一对应') from exc
        if not np.array_equal(check.values,selected.sst.values,equal_nan=True):
            raise ValueError('区域裁剪版与原始OISST重叠值冲突；不能当作同一份资料去重')
        relations.append(dict(operation='verified_same_product_subset',derived=selected_record['file'],
            originals=[r['file'] for r,_ in one_time],comparison='exact decoded values, coordinates and missing mask'))
    elif combined:
        if len(combined)!=1: raise ValueError('多个SST覆盖范围不明确，不能自动选取')
        selected_record,selected=combined[0]
    else:
        selected_record=one_time[0][0]
        selected=xr.concat([ds for _,ds in one_time],dim='time').sortby('time')
    times=selected.time.values
    if len(np.unique(times))!=len(times) or (len(times)>1 and not np.all(np.diff(times)==np.timedelta64(1,'D'))):
        raise ValueError('OISST日期重复或不连续；不自动补时刻')
    # Unit conversion is owned by the existing platform, not an experiment script.
    from ecodata.engines.units import UnitEngine
    sst=UnitEngine().convert(selected.sst,'K').transpose('time','lat','lon')
    finite=np.isfinite(sst.values)
    if not finite.any() or np.isinf(sst.values).any() or np.any((sst.values[finite]<250)|(sst.values[finite]>330)):
        raise ValueError('SST有效温度范围或缺失数据检查失败')
    product=selected[['sst']].copy();product['sst']=sst
    product.attrs.update(crs='EPSG:4326',time_basis='UTC',
        metadata_evidence='NOAA/NCEI OISST v2.1 product profile; original geographic axes, daily analysis at source timestamps',
        processing_scope='daily SST supplementary input only; no temporal interpolation, no WRF project overwrite')
    product.sst.attrs.update(physical_variable='sea_surface_temperature',
        temporal_semantics='daily_analysis',spatial_semantics='grid_estimate',
        missing_policy='retain source mask; WPS encodes missing as -1e30')
    product.sst.encoding={}
    dest=folder/'sst_product.nc';product.to_netcdf(dest,engine='h5netcdf')
    receipt=dict(route='wrf_daily_sst_bundle_v1',inventory=inventory,archives=archives,source_catalog=catalog,
        source_relationships=relations,original_data_modified=False,handoff_dependencies=dependencies,
        fusion_decision=dict(status='PENDING_PLATFORM_ASSESSMENT',algorithm_search_executed=False),
        tasks=[dict(role='sea_surface_temperature_supplement',field='sst',unit='K',
                    result=dict(status='SAME_PRODUCT_VERIFIED_UNIT_CONVERSION',product=str(dest),
                    valid_values=int(finite.sum()),missing_values=int((~finite).sum())))],
        full_model_ready=False,native_model_executed=False,
        scope='Daily SST WPS intermediate supplement. Does not replace wrfinput or supply subdaily forcing. Original engineering retained read-only.')
    save(folder/'preparation.json',receipt)
    # The common catalog separates project state from new scientific sources.
    # Selecting a reader does not make an engineering initial state an independent
    # observation, even when its field is also called SST.
    decision=processing_plan(catalog,[selected_record['file']]+[r['file'] for r in native],relations)
    from ecodata.algorithms.platform_search import SearchInput, process_current_input
    candidates=[SearchInput(dest.name,'primary',dest.read_bytes())]
    attempt=dict(entry_point='ecodata.models.source_catalog.single_source_action',
        inputs=[dict(filename=c.filename,role=c.role,sha256=sha(c.content)) for c in candidates],
        source_decision=decision,original_data_modified=False,status='STARTED',
        candidate_search_started=False,engine_called=False)
    save(folder/'automatic_fusion_attempt.json',attempt)
    from ecodata.models.preparation_bridge import build_request, source_report
    context=dict(route=receipt['route'])
    request, target=build_request('wrf','sea_surface_temperature_supplement','sst',len(candidates),preparation=context)
    plan=source_report([(selected_record['file'],product)],'sst',target)
    plan['source_decision']=decision
    save(folder/'automatic_plan.json',plan)
    if progress: progress('平台按资料用途与目标要求判断是否需要融合')
    try:
        action=single_source_action(product,'sst',request,target['converter_contract'])
        if action['status']=='PRESERVE_SINGLE_SOURCE':
            # A unit-normalised product still goes through the real converter
            # preflight and independent native readback; missing cells stay missing.
            result=dict(status=action['status'],product=str(dest),reason=action['reason'])
            attempt.update(status='SINGLE_SOURCE_PRESERVED',action=action)
        else:
            from ecodata.models.alignment_bridge import align_inputs
            candidates,alignment=align_inputs(candidates,request,folder/'alignment')
            request=dict(request,observation_crs=alignment['observation_crs'])
            attempt.update(alignment=alignment,engine_called=True,
                entry_point='ecodata.algorithms.platform_search.process_current_input')
            save(folder/'automatic_fusion_attempt.json',attempt)
            result=process_current_input(candidates,request,folder/'automatic_fusion_search',progress=progress)
            attempt.update(status='ENGINE_RETURNED',result=result,
                candidate_search_started=bool(result.get('counts',{}).get('evaluated',0)))
            if not result.get('product'):
                raise ValueError(str(result.get('reason',result.get('status','未生成处理产物'))))
            dest=Path(result['product'])
        save(folder/'automatic_fusion_attempt.json',attempt)
        receipt['tasks'][0]['result']=dict(receipt['tasks'][0]['result'],**result)
        receipt['fusion_decision']=dict(status=action['status'],source_decision=decision,
            algorithm_search_executed=attempt['candidate_search_started'],engine_called=attempt['engine_called'],
            reason=action['reason'])
        save(folder/'preparation.json',receipt)
    except Exception as exc:
        import traceback
        attempt.update(status='PLATFORM_PREPARATION_REJECTED',error=str(exc),traceback=traceback.format_exc())
        save(folder/'automatic_fusion_attempt.json',attempt)
        receipt['fusion_decision']=dict(status='PLATFORM_PREPARATION_REJECTED',reason=str(exc),
            algorithm_search_executed='NOT_CONFIRMED' if attempt.get('engine_called') else False,
            evidence='automatic_fusion_attempt.json')
        save(folder/'preparation.json',receipt)
        raise ValueError('平台自动处理检查未通过，未继续转换：'+str(exc)) from exc
    if progress: progress('海温资料已准备，生成并独立读回WPS原生格式')
    from ecodata.models.conversion_bridge import finish_preparation
    return finish_preparation(folder,'wrf',[dest],receipt)


def contract(manifest):
    if manifest.get('preparation',{}).get('route')!='wrf_daily_sst_bundle_v1': return None
    return dict(schema='ecodata.converter.requirements.v1',model_id='wrf',
        scope='WPS v5 daily SST supplement with explicit missing mask',role='sea_surface_temperature_supplement',
        fields=[dict(name='sst',unit='K',required=True,data_type='continuous',allow_missing=True)],
        coordinates=dict(time=dict(datetime=True,interval_seconds=86400,dimension=True),
                         lat=dict(dimension=True),lon=dict(dimension=True)),
        dependencies=[],metadata=dict(time_basis='UTC'),reference_fields=[],required_references=[],
        crs='EPSG:4326',allow_missing_fields=['sst'],
        deferred_checks=['Native independent WPS record readback and source mask comparison'])
