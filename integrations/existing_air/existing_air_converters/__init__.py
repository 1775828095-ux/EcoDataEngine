"""Existing four model converters behind the teammate's unchanged entry point.

The isolated worker imports the OLD platform, avoiding package-name collisions.
Only products named by this invocation's handoff are supplied to its writers.
No native model is started by this interface installation.
"""
from pathlib import Path
import hashlib
import io
import json
import os
import subprocess
import sys
import xarray as xr


def digest(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _convert(model, manifest_path, output_dir):
    manifest_path=Path(manifest_path).resolve();out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('protocol')!='ecodata.converter.v1' or manifest.get('model_id')!=model:
        raise ValueError('交接协议或模型标识不符：'+model)
    bindings=json.loads(Path(__file__).with_name('bindings.json').read_text(encoding='utf8'))
    binding=bindings['models'][model]
    config_path=Path(binding['configuration'])
    if not config_path.is_file() or digest(config_path)!=binding['configuration_sha256']:
        raise ValueError('已有转换配置缺失或已变化：'+str(config_path))
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    tasks=manifest.get('preparation',{}).get('tasks',[]);products=manifest.get('products',[])
    if not products or len(products)!=len(tasks):raise ValueError('缺少本次处理产品或无法对应处理字段')
    arrays=[];reads=[]
    for item,task in zip(products,tasks):
        if item.get('field',task.get('field'))!=task.get('field') or item.get('role',task.get('role'))!=task.get('role'):
            raise ValueError('交接产品与任务的物理字段/用途不一致')
        relative=Path(item['path']);p=(manifest_path.parent/relative).resolve()
        if relative.is_absolute() or not p.is_relative_to(manifest_path.parent) or not p.is_file():raise ValueError('产品路径缺失或越界')
        if digest(p)!=item['sha256']:raise ValueError('本次产品SHA256不符：'+str(relative))
        ds=xr.open_dataset(io.BytesIO(p.read_bytes())).load();field=task.get('field')
        if field not in ds.data_vars:raise ValueError('处理字段不在产品中：'+str(field))
        # Never consume other, possibly unprocessed fields retained in product.nc.
        arrays.append(ds[[field]])
        reads.append(dict(path=str(p),sha256=item['sha256'],field=field,role=task.get('role'),unit=ds[field].attrs.get('units')))
    merged=xr.merge(arrays,join='exact',compat='equals',combine_attrs='drop_conflicts')
    # Only contract metadata is shared across products. Source provenance stays
    # in lineage instead of preventing valid multi-source products from merging.
    for key in ('crs','source_crs','time_basis','source_timezone','wind_reference','grid','elevation_vertical_reference'):
        values=[array.attrs[key] for array in arrays if key in array.attrs]
        if values and any(value!=values[0] for value in values):raise ValueError('处理产品元信息冲突：'+key)
        if values:merged.attrs[key]=values[0]
    # Explicit administrator aliases only; no fuzzy guessing of quantities.
    renames={a:b for a,b in binding.get('aliases',{}).items() if a in merged and b not in merged}
    if renames:merged=merged.rename(renames)
    selected=set(task['field'] for task in tasks)
    allowed=set(binding['accepted_fields'])
    if selected-allowed:raise ValueError('现有专项转换器尚不接收这些处理字段：'+str(sorted(selected-allowed))+'；不会伪造变量或回退旧数据')
    source=out/'current_products.nc';merged.to_netcdf(source,engine='h5netcdf')
    receipt=dict(model=model,manifest=str(manifest_path),manifest_sha256=digest(manifest_path),
                 products_read=reads,source_sha256=digest(source),configuration=str(config_path),
                 configuration_sha256=binding['configuration_sha256'],aliases_applied=renames,
                 native_model_executed=False)
    (out/'lineage.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    job=dict(model=model,source=str(source),output=str(out),binding=binding,configuration=config,lineage=receipt)
    job_path=out/'worker_request.json';job_path.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf8')
    runtime=Path(bindings['legacy_python']);legacy=Path(bindings['legacy_source'])
    if not runtime.is_file() or not legacy.is_dir():raise ValueError('本机已有转换运行环境缺失')
    env=dict(os.environ,PYTHONPATH=str(legacy),PYTHONIOENCODING='utf-8',PYTHONNOUSERSITE='1')
    command=[str(runtime),str(Path(__file__).with_name('worker.py')),str(job_path)]
    with (out/'conversion.log').open('wb') as log:
        proc=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
    if proc.returncode:
        text=(out/'conversion.log').read_text(encoding='utf8',errors='replace')
        raise ValueError(model+'已有转换代码拒绝本次资料：'+text[-3000:])
    result=json.loads((out/'converter_result.json').read_text(encoding='utf8'))
    # The public bridge checks files too; retain a local output contract guard.
    for name in result.get('files',[]):
        p=(out/name).resolve()
        if not p.is_relative_to(out) or not p.is_file() or not p.stat().st_size:raise ValueError('转换产物缺失、为空或越界')
    return result


def wrf(manifest_path,output_dir):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if manifest.get('preparation',{}).get('route')=='wrf_daily_sst_bundle_v1':
        from .wrf_sst_convert import convert
        return convert(manifest_path,output_dir)
    return _convert('wrf',manifest_path,output_dir)
def camx(manifest_path,output_dir):return _convert('camx',manifest_path,output_dir)
def cmaq(manifest_path,output_dir):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if manifest.get('preparation',{}).get('route')=='cmaq_reviewed_lai_v1':
        from .cmaq_reviewed import convert
        return convert(manifest_path,output_dir)
    return _convert('cmaq',manifest_path,output_dir)
def calpuff(manifest_path,output_dir):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if manifest.get('preparation',{}).get('route')=='calpuff_native_58_v1':
        from .calpuff_native_bundle import convert
        return convert(manifest_path,output_dir)
    return _convert('calpuff',manifest_path,output_dir)


from .contracts import requirements as _requirements
wrf.requirements = lambda: _requirements('wrf')
from .wrf_bundle_pipeline import prepare as _prepare_wrf_bundle, contract as _wrf_bundle_contract
wrf.prepare_uploads = _prepare_wrf_bundle
wrf.requirements_for_manifest = _wrf_bundle_contract
camx.requirements = lambda: _requirements('camx')
cmaq.requirements = lambda: _requirements('cmaq')
from .cmaq_reviewed import prepare as _prepare_cmaq_reviewed, contract as _cmaq_reviewed_contract
cmaq.prepare_uploads = _prepare_cmaq_reviewed
cmaq.requirements_for_manifest = _cmaq_reviewed_contract
calpuff.requirements = lambda: _requirements('calpuff')
from .calpuff_native_bundle import prepare as _prepare_calpuff_native, contract as _calpuff_native_contract
calpuff.prepare_uploads = _prepare_calpuff_native
calpuff.requirements_for_manifest = _calpuff_native_contract

def _source_inspector(model):
    def inspect(files, records):
        from ecodata.models.native_source_inspection import inspect_sources
        return inspect_sources(model, files, records)
    return inspect
for _model, _converter in [('wrf',wrf),('camx',camx),('cmaq',cmaq),('calpuff',calpuff)]:
    _converter.inspect_sources = _source_inspector(_model)

def _prefix_inspector(model):
    def inspect(prefixes):
        if model=='cmaq':
            from .cmaq_reviewed import probe
            reviewed=probe(prefixes)
            if reviewed is not None:return reviewed
        from ecodata.models.native_source_inspection import probe_sources
        return probe_sources(model,prefixes)
    return inspect
for _model, _converter in [('wrf',wrf),('camx',camx),('cmaq',cmaq),('calpuff',calpuff)]:
    _converter.probe_sources=_prefix_inspector(_model)
