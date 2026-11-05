"""Recognize official WRFDA inputs without misrouting them as GRIB forcing.

WRFDA observation operators/variational analysis require a native backend.
An inventory is never represented as an assimilation or ordinary WRF update.
"""
from pathlib import Path, PurePosixPath
from contextlib import ExitStack
import json
import zipfile
import f90nml
from .project_binding import ProjectFiles


def prepare_wrfda_inputs(uploads,folder,progress=None):
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack,posix_members=True);controls=[]
        for name in files.files:
            if not PurePosixPath(name).name.startswith('namelist.input'):continue
            with files.files[name][0]() as s:
                try:nml=f90nml.reads(s.read().decode('utf-8-sig'))
                except (ValueError,UnicodeError):continue
            if 'wrfvar18' in nml and 'wrfvar1' in nml:
                controls.append(dict(resource=name,analysis_date=nml['wrfvar18'].get('analysis_date'),
                    var4d=nml['wrfvar1'].get('var4d',False),sha256=files.identity(name)['sha256']))
        if not controls:return None
        from .wrfda_backend import configuration,execute
        cfg=configuration()
        if cfg:
            from .wrfda_3dvar import plan,validate
            folder=Path(folder)
            request,binding=plan(files,controls,folder/'wrfda_request')
            native=folder/'wrfda_analysis'
            execution=execute(request,native,cfg,progress=progress)
            validation=validate(request,native,execution)
            manifest=dict(protocol='ecodata.converter.v1',model_id='wrf',status='COMPONENT_EXPORTED',native_ready=False,
                native_writeback_verified=True,native_model_executed=False,assimilation_executed=True,
                wrf_forecast_executed=False,products=[dict(path='wrfda_analysis/wrfvar_output',
                    sha256=__import__('hashlib').sha256((native/'wrfvar_output').read_bytes()).hexdigest(),
                    kind='official WRFDA native analysis')],
                preparation=dict(route='wrfda_3dvar_v1',bindings=binding,execution=execution,validation=validation),
                note='已执行官方WRFDA 3DVAR并核验真实分析场；没有更新侧边界或运行后续WRF预报，未证明独立精度提高')
            (folder/'handoff.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
            archive=folder/'EcoData-WRFDA-analysis.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                z.write(folder/'handoff.json','handoff.json')
                for directory in [native,folder/'wrfda_request']:
                    for p in directory.iterdir():
                        if p.is_file() and p.name not in {'fg','be.dat','ob.ascii'}:z.write(p,p.relative_to(folder).as_posix())
            return dict(manifest=manifest,archive=str(archive))
        resources=[]
        for name in sorted(files.files):
            base=PurePosixPath(name).name
            role='background_error' if base=='be.dat' else 'background' if base.startswith('wrfinput') else 'boundary' if base.startswith('wrfbdy') else 'observation' if base.startswith(('obs_gts','ob.ascii')) else None
            if role:resources.append(dict(resource=name,role=role,**files.identity(name)))
    blocker=dict(code='WRFDA_BACKEND_NOT_CONNECTED',message='已识别WRFDA同化工程；当前平台未接入可调用的WRFDA同化后端、观测算子和原生运行依赖。普通WRF的GRIB/SST处理不能代替3DVAR/4DVAR。')
    inventory=dict(schema='ecodata.wrfda-inputs.v1',controls=controls,resources=resources,
        source_modified=False,assimilation_executed=False,native_model_executed=False,
        official_reference='https://github.com/wrf-model/WRF/tree/master/var',required_information=[blocker])
    assessment=dict(sources=[],required_information=[blocker],weather={},
        stages=dict(engineering_inspected=True,fusion_executed=False,native_writeback=False,native_model_executed=False))
    manifest=dict(protocol='ecodata.converter.v1',model_id='wrf',status='ENGINEERING_INSPECTED',native_ready=False,
        products=[],native_writeback_verified=False,native_model_executed=False,
        preparation=dict(route='wrfda_inputs_v1',inventory='project_inventory.json',source_admission=assessment),
        note=blocker['message'])
    folder=Path(folder)
    for name,value in [('project_inventory.json',inventory),('required_information.json',[blocker]),('source_admission.json',assessment),('handoff.json',manifest)]:
        (folder/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    (folder/'processing_status.txt').write_text(blocker['message']+'\n未融合、未同化、未写回、未运行模型。原始资料不变。',encoding='utf8')
    archive=folder/'EcoData-WRFDA-inspection.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['project_inventory.json','required_information.json','source_admission.json','handoff.json','processing_status.txt']:z.write(folder/name,name)
    return dict(manifest=manifest,archive=str(archive))
