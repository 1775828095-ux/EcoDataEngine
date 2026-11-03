"""Controlled adapter worker: imports only the existing platform code tree.

No model is run here. All writes, copied engineering files and logs stay under
the current converter output directory. No old converted result is an input.
"""
from pathlib import Path
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def archive(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
    return stream.getvalue()


def main(request):
    import xarray as xr
    from ecodata.round1 import convert
    job=json.loads(Path(request).read_text(encoding='utf8'))
    model=job['model'];cfg=job['configuration'];binding=job['binding'];out=Path(job['output']).resolve()
    seed=Path(binding['engineering'])
    if not seed.exists():raise ValueError('绑定的真实工程缺失：'+str(seed))
    if seed.is_file() and digest(seed)!=binding['engineering_sha256']:raise ValueError('绑定工程已变化，需维护者重新核对')
    ds=xr.open_dataset(job['source']).load()
    native=out/'native';native.mkdir(exist_ok=False)
    project=out/'project';project.mkdir(exist_ok=False)
    details={};complete=False
    if model=='calpuff':
        from ecodata.interface.calpuff_project import convert_existing_table,unpack
        # CSV values are only this invocation's product values and coordinates.
        frame=ds.to_dataframe().reset_index()
        table=frame.to_csv(index=False).encode('utf8')
        payload,details=convert_existing_table(seed.read_bytes(),table,cfg,cfg['mode'])
        with zipfile.ZipFile(io.BytesIO(payload)) as z:files={n:z.read(n) for n in z.namelist()}
        original,_=unpack(seed.read_bytes(),'existing')
        original['CALPUFF.INP']=files['CALPUFF.INP']
        for name,raw in original.items():
            destination=(project/name).resolve()
            if not destination.is_relative_to(project):raise ValueError('工程文件路径越界')
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
        complete=True
    else:
        if model=='camx':
            # Require actual units to agree with the bound writer configuration.
            for item in cfg['species']:
                variable=item['column']
                if variable not in ds:raise ValueError('本次产品缺少排放物种：'+variable)
                if ds[variable].attrs.get('units')!=item['unit']:raise ValueError('排放产品单位与原配置不一致：'+variable)
            content=ds.to_dataframe().reset_index().to_csv(index=False).encode('utf8')
        else:content=Path(job['source']).read_bytes()
        files,details=convert(model,content,cfg)
        if model=='wrf':
            from ecodata.interface.wrf_assembly import read_fields,assemble_pressure_update
            merged=[]
            for name,raw in files.items():
                if not name.endswith('.wps'):continue
                stamp=read_fields(raw)[0].time[:13];source=seed/('FILE_'+stamp)
                if not source.is_file():raise ValueError('绑定FNL工程缺少本时刻完整配套气象场：'+stamp)
                assembled,receipt=assemble_pressure_update(source.read_bytes(),raw)
                (project/('FILE_'+stamp)).write_bytes(assembled)
                merged.append(dict(time=stamp,original_sha256=digest(source),assembly=receipt))
            geo=seed/'geo_em.d01.nc'
            if not geo.is_file():raise ValueError('已有工程缺少geo_em.d01.nc')
            shutil.copyfile(geo,project/geo.name)
            details['existing_wps_assembly']=merged
            details['downstream_required']='完整WPS中间场已合并；后续仍须使用原工程配置执行metgrid/REAL/WRF。本接口不把旧wrfinput当新预处理结果。'
        elif model=='camx':
            from ecodata.interface.camx_external_project import inspect,emission_subset,PATTERN
            engineering=inspect(str(seed));target=binding['target']
            reference=Path(next(i['path'] for i in engineering['inputs'] if i['key']==target))
            details['engineering_update']=emission_subset(archive(files),reference,engineering,project/'current_emissions.nc')
            mapping={};inputdir=project/'inputs';inputdir.mkdir()
            for index,item in enumerate(engineering['inputs']):
                source=Path(item['path'])
                if item['key']==target:dest=project/'current_emissions.nc'
                else:
                    dest=inputdir/(str(index)+'_'+source.name);shutil.copyfile(source,dest)
                    if digest(dest)!=item['sha256']:raise ValueError('工程副本字节不符：'+item['key'])
                mapping[item['key']]="'"+dest.relative_to(project).as_posix()+"'"
            mapping['Root_Output_Name']="'output/current'"
            for item in engineering['inactive_references']:mapping[item['key']]="'inactive/"+item['key']+"'"
            rendered=[]
            for line in seed.read_text(encoding='utf-8-sig').splitlines(keepends=True):
                match=PATTERN.fullmatch(line.rstrip('\r\n'))
                if match:
                    key=re.sub(r'\s','',match[1])
                    if key in mapping:line=line[:match.start(2)]+mapping[key]+line[match.end(2):]
                rendered.append(line)
            (project/'CAMx.in').write_text(''.join(rendered),encoding='utf8')
            (project/'output').mkdir();details['engineering_control']=str(seed);complete=True
        elif model=='cmaq':
            from ecodata.interface.cmaq_external_project import inspect_manifest
            from ecodata.interface.registered_conversion import compatible_netcdf
            manifest=json.loads(seed.read_text(encoding='utf8'))
            engineering=inspect_manifest(manifest,binding.get('preserve_original_vertical_headers',False))
            target=binding['target'];raw=files['emissions.CMAQ55.nc']
            from cmaq_subset import assemble_species
            reference=Path(manifest['inputs'][target]['path'])
            assembled=out/'assembled_emissions.nc'
            details['species_update']=assemble_species(raw, reference, assembled,
                selected=list(cfg['fields']), preserve_vertical=False)
            raw=assembled.read_bytes()
            details['engineering_update']=compatible_netcdf(raw,reference)
            inputdir=project/'inputs';inputdir.mkdir();environment=dict(manifest['environment'])
            for index,item in enumerate(engineering['files']):
                name=str(index)+'_'+Path(item['path']).name;dest=inputdir/name
                if item['key']==target:dest.write_bytes(raw)
                else:
                    shutil.copyfile(item['path'],dest)
                    if digest(dest)!=item['sha256']:raise ValueError('工程复制内容不符：'+item['key'])
                environment[item['key']]='inputs/'+name
            for key in engineering['outputs']:environment[key]='output/'+key+'.nc'
            for key in engineering['unavailable_inputs']:environment[key]='unavailable/'+key
            environment.update(OUTDIR='output',LOGDIR='output/LOGS',WORKDIR='.')
            (project/'output/LOGS').mkdir(parents=True)
            (project/'effective_environment.json').write_text(json.dumps(environment,indent=2),encoding='utf8')
            details['launcher_policy']='Use the existing controlled CMAQ launcher to resolve these project-relative paths; no uploaded script is executed.'
            complete=True
    for name,raw in files.items():
        p=(native/name).resolve()
        if not p.is_relative_to(native) or not raw:raise ValueError('旧writer输出路径越界或内容为空')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    validation=dict(lineage=job['lineage'],existing_conversion=details,
                    engineering_source=str(seed),engineering_copied=True,complete_engineering_inputs=complete,
                    native_model_executed=False,accuracy_evaluated=False,
                    qualification='现有原生格式writer及工程连接代码的接口调用；未进行本轮独立读取或模型验收，不声称模拟已跑通')
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    with zipfile.ZipFile(out/'model-inputs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in (native,project):
            for p in folder.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(out).as_posix())
    result=dict(status='PASS',files=['model-inputs.zip','lineage.json','validation.json','conversion.log'],validation=validation)
    (out/'converter_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf8')


if __name__=='__main__':main(sys.argv[1])
