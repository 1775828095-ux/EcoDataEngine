"""Invoke the installed legacy WPS writer in its isolated scientific runtime."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys


def convert(manifest_path,output_dir):
    manifest_path=Path(manifest_path).resolve();out=Path(output_dir).resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('model_id')!='wrf' or manifest.get('preparation',{}).get('route')!='wrf_daily_sst_bundle_v1':
        raise ValueError('WRF SST交接类型错误')
    products=manifest.get('products',[])
    if len(products)!=1 or products[0].get('field')!='sst': raise ValueError('需要本次SST处理产品')
    relative=Path(products[0]['path']);p=(manifest_path.parent/relative).resolve()
    if relative.is_absolute() or not p.is_relative_to(manifest_path.parent) or not p.is_file():
        raise ValueError('SST产品缺失或路径越界')
    with p.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest!=products[0]['sha256']: raise ValueError('本次SST产品哈希不符')
    cfg=json.loads(Path(__file__).with_name('bindings.json').read_text(encoding='utf8'))
    dependencies=[]
    for item in manifest.get('preparation',{}).get('handoff_dependencies',[]):
        rel=Path(item['path']);source=(manifest_path.parent/rel).resolve()
        if rel.is_absolute() or not source.is_relative_to(manifest_path.parent) or not source.is_file():raise ValueError('本次原始资料依赖缺失或越界')
        with source.open('rb') as stream:current=hashlib.file_digest(stream,'sha256').hexdigest()
        if current!=item['sha256']:raise ValueError('原始驱动资料哈希不符')
        dependencies.append(dict(item,path=str(source)))
    job=dict(product=str(p),output=str(out),product_sha256=digest,manifest=str(manifest_path),dependencies=dependencies)
    request=out/'sst_request.json';request.write_text(json.dumps(job),encoding='utf8')
    env=dict(os.environ,PYTHONPATH=cfg['legacy_source'],PYTHONIOENCODING='utf8',PYTHONNOUSERSITE='1')
    command=[cfg['legacy_python'],str(Path(__file__).with_name('wrf_sst_worker.py')),str(request)]
    with (out/'conversion.log').open('wb') as log:
        result=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,
            timeout=600,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
    if result.returncode:
        raise ValueError('SST转换失败：'+(out/'conversion.log').read_text(encoding='utf8',errors='replace')[-2000:])
    result=json.loads((out/'converter_result.json').read_text(encoding='utf8'))
    result['validation']['input_manifest']=str(manifest_path)
    return result
