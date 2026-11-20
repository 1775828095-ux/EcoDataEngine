"""Evidence-based post-conversion gate; AI cannot promote a local model trial."""
from pathlib import Path,PurePosixPath
from tempfile import TemporaryDirectory
from datetime import date,datetime
import hashlib,json,shutil,time,zipfile
from uuid import uuid4


def _digest(raw):return hashlib.sha256(raw).hexdigest()


def _swat_inputs(archive,root):
    """Existing CIO bindings plus actual downstream filename edges, not old outputs."""
    from ecodata.models.project_inventory import inspect_projects
    from ecodata.interface.engineering_upload import LocalUpload
    from ecodata.interface.swat_runner import FORBIDDEN_INPUT_SUFFIXES
    path=Path(archive.filename)
    inventory=inspect_projects([LocalUpload(path.name,path)],'swat_plus')
    available={PurePosixPath(i.filename).name:i for i in archive.infolist()
        if not i.is_dir() and PurePosixPath(i.filename).parent==root}
    pending={'file.cio'}
    for project in (inventory or {}).get('projects',[]):
        for binding in project['bindings']:
            if binding.get('status')=='BOUND' and PurePosixPath(binding['target']).parent==root:
                pending.add(PurePosixPath(binding['target']).name)
    files={}
    while pending:
        name=min(pending);pending.remove(name)
        if name in files:continue
        if PurePosixPath(name).suffix.lower() in FORBIDDEN_INPUT_SUFFIXES:
            raise ValueError('原生输入依赖不能包含上传的程序或运行库：'+name)
        raw=archive.read(available[name]);files[name]=raw
        try:text=raw.decode('utf-8-sig')
        except UnicodeError:continue
        # Native filenames are literal references. Foreign-key names/numbers do
        # not become files unless that exact uploaded resource actually exists.
        for token in text.split():
            reference=token.strip('"\'')
            if reference in available and reference not in files:pending.add(reference)
    return files,dict(strategy='public project_inventory CIO bindings + literal dependency closure',
        included=len(files),loaded_bytes=sum(len(v) for v in files.values()),
        resources=[dict(name=n,bytes=len(raw),sha256=_digest(raw)) for n,raw in files.items()],
        control_bindings=[dict(parameter=b['parameter'],target=PurePosixPath(b['target']).name)
            for p in (inventory or {}).get('projects',[]) for b in p['bindings'] if b.get('status')=='BOUND'],
        excluded=[dict(name=n,bytes=i.file_size,reason='not reached by native input dependencies')
            for n,i in available.items() if n not in files])


def _swat_trial(files,validation,manifest,folder,timeout):
    from ecodata.interface import swat_runner as runner
    from ecodata.core.exceptions import SchemaError
    version=validation.get('declared_version')
    engine={'60.5.7':'official60','61.0.2':'official61','62.0.0':'gnu62','62':'gnu62'}.get(version)
    if not engine:return dict(status='NOT_RUN',reason='没有登记与工程声明版本匹配的运行程序',declared_version=version)
    try:program=runner.engine_path(engine)
    except (ValueError,SchemaError) as exc:return dict(status='NOT_RUN',reason=str(exc),engine_id=engine)
    dates=validation['dates']
    start=date.fromordinal(date(dates['yrc_start'],1,1).toordinal()+max(1,dates['day_start'])-1)
    end=(date.fromordinal(date(dates['yrc_end'],1,1).toordinal()+dates['day_end']-1)
         if dates['day_end'] else date(dates['yrc_end'],12,31))
    exercises_update=False
    for task in manifest.get('preparation',{}).get('tasks',[]):
        label=task.get('execution',{}).get('result',{}).get('self_validation_protocol',{}).get('search_start')
        if label:
            candidate=datetime.fromisoformat(label).date()
            if start<=candidate<=end:start=candidate;exercises_update=True;break
    # The existing version-aware setter changes only the disposable trial copy.
    trial=runner.set_dates(files,start,start,engine_id=engine)
    time_name=validation['time_file']
    result=dict(engine_id=engine,program_sha256=_digest(program.read_bytes()),declared_version=version,
        trial_start=start.isoformat(),trial_end=start.isoformat(),full_simulation_verified=False,
        updated_date_exercised=exercises_update,
        date_control=dict(file=time_name,original_sha256=_digest(files[time_name]),trial_sha256=_digest(trial[time_name])),
        timeout_seconds=timeout)
    with TemporaryDirectory(prefix='native-read-',dir=folder) as temporary:
        ident=runner.start_job(trial,Path(temporary),timeout=timeout,engine_id=engine)
        deadline=time.monotonic()+timeout+10
        while runner.job_status(ident)['state']=='running' and time.monotonic()<deadline:time.sleep(.1)
        state=runner.job_status(ident)
        if state['state']=='running':
            runner.stop_job(ident)
            stop_deadline=time.monotonic()+5
            while runner.job_status(ident)['state']=='running' and time.monotonic()<stop_deadline:time.sleep(.1)
            state=runner.job_status(ident)
            if state['state']=='running':raise RuntimeError('原生程序未在停止期限内结束，未确认试读')
        directory=Path(state['directory'])
        result.update(status='PASS' if state['state']=='succeeded' else 'FAIL',
            state=state['state'],returncode=state.get('returncode'),fresh_success_marker=state.get('success_file',False),
            model_read_confirmed=state['state']=='succeeded',reason=state.get('error'))
        for name in ('runner.log','runner_manifest.json','runner_status.json','success.fin'):
            path=directory/name
            if path.is_file():shutil.copy2(path,folder/name)
        log=folder/'runner.log'
        result['log_tail']=(log.read_bytes()[-6000:].decode('utf8',errors='replace').replace(str(directory),'<trial>') if log.exists() else '')
        result['generated_files']=[dict(name=p.name,bytes=p.stat().st_size) for p in directory.iterdir()
            if p.is_file() and p.name not in files][:80]
    result['temporary_directory_removed']=not Path(temporary).exists()
    return result


def check_model_output(result,model_id,folder,*,run_native=True,timeout=30):
    """Check the produced artifact, not a guessed source package or machine default."""
    from ecodata.models.project_binding import safe_name
    from ecodata.models.requirements import RequirementStore
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    archive=Path(result.get('archive') or result['original_archive'])
    profile=RequirementStore().get(model_id)
    report=dict(schema='ecodata.final-model-check.v1',model_id=model_id,
        model_requirements=dict(reference_version=profile.get('reference_version'),official_sources=profile.get('official_sources',[]),
            required_variables=profile.get('required_variables',[]),input_formats=profile.get('input_formats',[])),
        format=dict(status='UNKNOWN',errors=[]),project=dict(status='UNKNOWN'),
        native=dict(status='NOT_RUN',reason='尚无该模型已绑定的最终试读适配器'),
        scientific_definitions=dict(status='UNCONFIRMED',assumptions=[]),ai_review=dict(status='PENDING'))
    try:
        with archive.open('rb') as stream:report['archive_sha256_before_receipts']=hashlib.file_digest(stream,'sha256').hexdigest()
        with zipfile.ZipFile(archive) as z:
            if z.testzip():raise ValueError('输出归档CRC检查失败')
            names=[i.filename for i in z.infolist() if not i.is_dir()]
            if len(names)!=len(set(names)):raise ValueError('输出归档含重复成员')
            for name in names:safe_name(name,posix=True)
            report['format']['zip_readback']=True
            manifest=result.get('manifest',{})
            for change in manifest.get('changes',[]):
                if _digest(z.read(change['path']))!=change['after']:raise ValueError('写回文件与转换记录指纹不同：'+change['path'])
            for info in z.infolist():
                # Bounded metadata inspection, not a data-size/upload restriction.
                if not info.filename.endswith('.json') or '_ecodata' in PurePosixPath(info.filename).parts or info.file_size>65536:continue
                try:declaration=json.loads(z.read(info))
                except (ValueError,UnicodeError):continue
                if isinstance(declaration,dict) and declaration.get('status')=='WORKING_ASSUMPTION_ONLY' and isinstance(declaration.get('rule'),str):
                    report['scientific_definitions']['assumptions'].append(dict(declaration,source_document=info.filename))
            if report['scientific_definitions']['assumptions']:report['scientific_definitions']['status']='WORKING_ASSUMPTIONS'
            if model_id=='swat_plus':
                tasks=manifest.get('preparation',{}).get('tasks',[])
                if manifest.get('status')=='COMPONENT_EXPORTED' and tasks and all(t.get('operation') in {'weather','precipitation'} for t in tasks):
                    from ecodata.models.swat_weather_project import _records
                    checked={}
                    for item in manifest.get('output_inventory',[]):
                        name=item['path'];raw=z.read(name)
                        if _digest(raw)!=item['sha256']:raise ValueError('组件指纹不一致：'+name)
                        kind={'.pcp':'pcp','.tem':'tmp','.slr':'slr','.hmd':'hmd','.wnd':'wnd'}.get(PurePosixPath(name).suffix)
                        if kind:
                            dates=_records(raw,kind,allow_native_missing=False)
                            checked[name]=dict(status='PASS',records=len(dates),kind=kind)
                    if not checked:raise ValueError('未发现已交付的实际天气组件')
                    report['format'].update(status='PASS',scope='delivered_weather_components_only',weather_files=checked)
                    report['project']=dict(status='NOT_REQUESTED',reason='本次仅导出天气组件，不要求完整流域工程')
                    report['native']=dict(status='NOT_RUN',model_read_confirmed=False,reason='未请求完整模型运行；组件格式通过不代表完整工程可运行')
                    (folder/'final_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
                    return report
                from ecodata.interface.swat_runner import check_project
                from ecodata.models.swat_weather_project import _records,_table,_layout
                roots=[PurePosixPath(n).parent for n in names if PurePosixPath(n).name=='file.cio']
                if len(roots)!=1:raise ValueError('最终SWAT+产品需要唯一实际file.cio工程')
                project_root=roots[0]
                files,dependency_scope=_swat_inputs(z,project_root)
                report['input_dependency_scope']=dependency_scope
                validation=check_project(files)
                report['project']=dict(status='PASS' if not validation['errors'] else 'FAIL',validation=validation)
                _,_,climate,_=_layout(files);_,header,stations=_table(files[climate[1]])
                checked={}
                for station in stations:
                    for kind in ('pcp','tmp','slr','hmd','wnd'):
                        name=station[header.index(kind)]
                        if name.lower() in {'null','sim','0'}:continue
                        if name in checked:continue
                        try:
                            dates=_records(files[name],kind,allow_native_missing=True)
                            checked[name]=dict(status='PASS',kind=kind,records=len(dates),start=str(dates[0]),end=str(dates[-1]))
                        except (ValueError,KeyError,IndexError) as exc:
                            checked[name]=dict(status='FAIL',kind=kind,error=str(exc))
                errors=[f"{n}: {r['error']}" for n,r in checked.items() if r['status']=='FAIL']
                report['format'].update(status='FAIL' if errors else 'PASS',errors=errors,weather_files=checked,
                    scope='原生天气表头/字段/有限值/缺测标记/日期连续性及写回SHA；工程深层要求另列')
                if validation['errors'] or errors:
                    report['native']=dict(status='NOT_RUN',reason='确定性检查未通过，未启动模型')
                elif run_native:
                    try:report['native']=_swat_trial(files,validation,manifest,folder,timeout)
                    except Exception as exc:report['native']=dict(status='ERROR',reason=str(exc),model_read_confirmed=False)
                else:report['native']=dict(status='NOT_RUN',reason='本次调用明确关闭原生试读')
            elif model_id in {'wrf','cmaq','camx','calpuff'}:
                from .final_native_check import check_other_model
                report.update(check_other_model(z,model_id,folder,run_native=run_native,timeout=timeout))
                if manifest.get('native_model_executed'):
                    report['native']['previous_native_execution']=True
            else:
                report['format']['reason']='归档与写回指纹通过；该模型全量原生格式最终检查尚未接入，不标PASS'
                if manifest.get('native_model_executed'):
                    report['native']=dict(status='NOT_RECHECKED',previous_native_execution=True,
                        reason='原流程已有原生运行记录，本关卡尚未接入该模型的最终复核；未覆盖原有证据')
    except Exception as exc:
        report['format'].update(status='FAIL',errors=[*report['format']['errors'],str(exc)])
        report['native']=dict(status='NOT_RUN',reason='最终检查异常，未确认模型实际读入')
    (folder/'final_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    return report


def attach_final_check(result,model_id,folder):
    folder=Path(folder)
    archive=Path(result.get('archive') or result['original_archive']).resolve()
    if not archive.is_relative_to(folder.resolve().parent):
        raise ValueError('最终检查不能改写本次输出目录之外的归档')
    report=check_model_output(result,model_id,folder)
    report['evidence_prefix']='_ecodata/final_check/'+uuid4().hex
    (folder/'final_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    result['manifest']['final_check']=report
    result['final_check_folder']=str(folder)
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        for name in ('final_check.json','runner.log','native_listing.log','runner_manifest.json','runner_status.json','success.fin'):
            path=folder/name
            if path.is_file():z.write(path,report['evidence_prefix']+'/'+name)
    return report


def complete_final_check(result,review,*,mock=False,error=None,request_sha256=None):
    """Append a final receipt; preserve local checks even if AI disagrees."""
    report=result.get('manifest',{}).get('final_check')
    if not report:return
    report['ai_review']=dict(status='FAILED' if error else 'SIMULATED' if mock else 'COMPLETED',review=review,error=error,
        request_sha256=request_sha256)
    folder=Path(result['final_check_folder'])
    archive=Path(result.get('archive') or result['original_archive'])
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        name=report['evidence_prefix']+'/complete.json'
        revision=1
        while name in z.namelist():
            revision+=1;name=report['evidence_prefix']+f'/complete.{revision}.json'
        report['review_revision']=revision;report['final_receipt']=name
        complete=folder/'complete.json'
        complete.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        z.write(complete,name)
