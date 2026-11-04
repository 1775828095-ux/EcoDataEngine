"""Thin, fail-closed bridge to the existing SWAT+ weather writer/attachment.

Only the administrator-bound Hubbard W2/HQ daily case is accepted. No spatial
averaging, metadata repair, date changes or scientific parameter changes.
"""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import zipfile

import numpy as np
import pandas as pd
import xarray as xr


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _binding():
    return json.loads(Path(__file__).with_name('binding.json').read_text(encoding='utf8'))


def _convert_hubbard_legacy(manifest_path, output_dir):
    from ecodata.models.adapters.swat_plus import SWATPlusWriter
    from .weather_attachment import attach_weather, _read_zip, _check_refs
    from .project_dependencies import weather_dependencies
    manifest_path=Path(manifest_path).resolve();out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('protocol')!='ecodata.converter.v1' or manifest.get('model_id')!='swat_plus':
        raise ValueError('转换协议或目标模型不匹配')
    cfg=_binding();project=Path(cfg['project_zip'])
    if not project.is_file():raise ValueError('缺少绑定的Hubbard原工程：'+str(project))
    if sha(project)!=cfg['project_sha256']:raise ValueError('原工程SHA256变化，拒绝使用未知工程')
    datasets=[];reads=[]
    tasks=manifest.get('preparation',{}).get('tasks',[])
    products=manifest.get('products',[])
    if len(tasks)!=len(products) or not products:raise ValueError('产品与处理任务无法逐项对应')
    quantities={};dates=None
    needed={'precipitation':'mm/day','air_temperature_max':'degC','air_temperature_min':'degC'}
    for product,task in zip(products,tasks):
        rel=Path(product['path']);p=(manifest_path.parent/rel).resolve()
        if rel.is_absolute() or not p.is_relative_to(manifest_path.parent) or not p.is_file():
            raise ValueError('处理产品路径缺失或越界')
        if sha(p)!=product['sha256']:raise ValueError('本次处理产品SHA256不匹配：'+str(rel))
        ds=xr.open_dataset(io.BytesIO(p.read_bytes())).load();datasets.append(ds)
        reads.append(dict(path=str(p),relative_path=product['path'],sha256=sha(p),field=task.get('field'),read_verified=True))
        (out/'attempt.json').write_text(json.dumps(dict(manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),products_read=reads),ensure_ascii=False,indent=2),encoding='utf8')
        name=task.get('field')
        if name not in needed:raise ValueError('本转换器尚未支持该字段，不静默忽略：'+str(name))
        if name in quantities or name not in ds:raise ValueError('处理字段重复或产品缺少字段：'+name)
        if ds[name].attrs.get('units')!=needed[name]:raise ValueError('单位不匹配：'+name+'，需要'+needed[name])
        if ds.attrs.get('crs')!='EPSG:4326':raise ValueError('产品缺少明确EPSG:4326坐标系，不推测')
        for aliases,expected in [(('lon','longitude'),cfg['longitude']),(('lat','latitude'),cfg['latitude'])]:
            found=[n for n in aliases if n in ds.coords]
            if len(found)!=1 or ds[found[0]].size!=1:raise ValueError('需要明确的单站经纬度；不自动平均网格')
            if not np.isclose(float(ds[found[0]].values.reshape(-1)[0]),expected,atol=1e-6,rtol=0):
                raise ValueError('资料位置与绑定的Hubbard站点不匹配；不能使用其他流域工程')
        value=ds[name]
        if 'time' not in value.dims or any(value.sizes[d]!=1 for d in value.dims if d!='time'):
            raise ValueError('需要单站逐日时间序列，不自动空间聚合')
        clock=pd.DatetimeIndex(ds.time.values)
        expected=pd.date_range(cfg['start'],cfg['end'],freq='D')
        if not clock.equals(expected):raise ValueError('日期未连续覆盖绑定工程模拟及预热期；不补值、不改日期')
        if dates is not None and not clock.equals(dates):raise ValueError('产品日期不一致')
        dates=clock
        values=np.asarray(value.transpose('time',*[d for d in value.dims if d!='time']).values).reshape(len(clock))
        if not np.isfinite(values).all():raise ValueError('处理产品仍有缺失/非有限值：'+name)
        quantities[name]=values
        reads[-1]['unit']=needed[name]
    missing=sorted(set(needed)-set(quantities))
    if missing:raise ValueError('缺少关键气象变量：'+', '.join(missing))
    if (quantities['precipitation']<0).any() or (quantities['air_temperature_max']<quantities['air_temperature_min']).any():
        raise ValueError('降水为负或最高温低于最低温')
    frame=pd.DataFrame(dict(time=dates,**quantities))
    weather=SWATPlusWriter().write(dict(project_name='Hubbard_current_platform_products',start=cfg['start'],end=cfg['end'],
        target_model_version='62.0.0',station=dict(name='HQ',latitude=cfg['latitude'],longitude=cfg['longitude'],elevation=cfg['elevation']),
        weather=frame.to_dict('records')))
    packed,attachment=attach_weather(project.read_bytes(),{k:v for k,v in weather.items() if k in {'station.pcp','station.tem'}},cfg['station'])
    files=_read_zip(packed);_check_refs(files)
    deps=weather_dependencies(files,cfg['station'])
    if deps['errors']:raise ValueError('工程天气依赖未通过：'+'; '.join(deps['errors']))
    # Independent numeric readback, separate from the writer's internal checks.
    for suffix,names in [('pcp',['precipitation']),('tmp',['air_temperature_max','air_temperature_min'])]:
        path=next(c['target'] for c in attachment['changes'] if c['variable']==suffix)
        lines=files[path].decode().splitlines()[3:]
        rows=np.array([[float(t) for t in line.split()] for line in lines if line.strip()])
        if len(rows)!=len(frame):raise ValueError('独立文本读回记录数不符')
        for col,name in enumerate(names,2):
            if not np.allclose(rows[:,col],quantities[name],atol=0.00051,rtol=0):raise ValueError('独立文本读回数值不符：'+name)
    run=out/'project';run.mkdir(exist_ok=False)
    for name,raw in files.items():(run/name).write_bytes(raw)
    native={'status':'NOT_RUN'}
    if cfg.get('run_native'):
        exe=Path(cfg['engine'])
        if not exe.is_file() or sha(exe)!=cfg['engine_sha256']:raise ValueError('绑定的原生模型不存在或校验失败')
        # Old outputs are not allowed to count as current success.
        if (run/'success.fin').exists():raise ValueError('工程输入混入旧success.fin，不用旧结果证明本次运行')
        command=[str(exe)]
        with (out/'native.log').open('wb') as log:
            proc=subprocess.run(command,cwd=run,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        native=dict(command=command,returncode=proc.returncode,engine_sha256=sha(exe),version='SWAT+ rev.62',status='PASS' if proc.returncode==0 and (run/'success.fin').is_file() else 'FAILED')
        if native['status']!='PASS':raise ValueError('原生模型未完成；保留本次native.log')
    if sha(project)!=cfg['project_sha256']:raise ValueError('原工程意外变化')
    validation=dict(products_read=reads,project_source=str(project),project_sha256=cfg['project_sha256'],
                    project_original_unchanged=True,attachment=attachment,independent_numeric_readback=True,
                    optional_weather='辐射、湿度、风速沿用原工程WGN，不冒充本次观测',native=native,
                    accuracy_improvement_evaluated=False)
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf8')
    with zipfile.ZipFile(out/'SWATplus-current-project.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in run.iterdir():
            if p.is_file():z.write(p,p.name)
    resultfiles=['SWATplus-current-project.zip','validation.json','attempt.json']
    if (out/'native.log').exists():resultfiles.append('native.log')
    return dict(status='PASS',files=resultfiles,validation=validation)


def convert(manifest_path, output_dir):
    from ecodata.models.component_service import unsupported_conversion
    return unsupported_conversion(manifest_path, output_dir)


def _requirements():
    from ecodata.models.component_service import generic_requirements
    return generic_requirements('swat_plus')

convert.requirements = _requirements

def _inspect_sources(files, records):
    from ecodata.models.native_source_inspection import inspect_sources
    return inspect_sources('swat_plus', files, records)
convert.inspect_sources = _inspect_sources

def _probe_sources(prefixes):
    from ecodata.models.native_source_inspection import probe_sources
    return probe_sources('swat_plus',prefixes)
convert.probe_sources=_probe_sources
