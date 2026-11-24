"""Post-writeback checks over actual public bindings, never uploaded executables."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib,json,shutil,subprocess,time


def _runtime(model):
    if model in {'camx','cmaq'}:
        from ecodata.interface import camx_runner,cmaq_runner
        runner={'camx':camx_runner,'cmaq':cmaq_runner}[model]
        return dict(engine=str(runner.EXE),sha256=runner.PIN,transport='WSL EcoData-WRF-Ubuntu2204')
    if model=='calpuff':
        from existing_air_converters.calpuff_native_bundle import ENGINES,ENGINE_ROOT
        exe,pin=ENGINES['CALPUFF']
        return dict(engine=str(ENGINE_ROOT/exe),sha256=pin,transport='Windows',supported_control='5.8')
    return dict(engine=None,transport='WRF forecast',
        reason='现有WRF运行器仅绑定登记示范工程；WRFDA同化程序不能代替任意WRF预报试读，尚无可匹配的通用预报运行契约')


def _probe(model):
    runtime=_runtime(model);path=runtime.get('engine')
    if not path:return dict(runtime,status='NOT_RUN',reason=runtime['reason'],model_read_confirmed=False)
    if not Path(path).is_file():
        return dict(runtime,status='NOT_RUN',reason='本机未安装已登记且指纹固定的匹配程序：'+path,model_read_confirmed=False)
    with Path(path).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=runtime['sha256']:
        return dict(runtime,status='NOT_RUN',reason='登记程序SHA256不匹配，未执行',model_read_confirmed=False)
    return dict(runtime,status='AVAILABLE',program_sha256=actual,model_read_confirmed=False)


def _camx_trial(archive,binding,folder,runtime,timeout):
    """Relocate only parsed file references; run fixed host program, no script."""
    import f90nml,io,re
    from ecodata.interface.camx_runner import linux
    from ecodata.interface.camx_external_project import inspect
    with TemporaryDirectory(prefix='camx_trial_',dir=folder) as work:
        work=Path(work);nml=f90nml.read(io.StringIO(archive.read(binding['control']['path']).decode('utf-8-sig')))
        group=nml['camx_control']
        for i,row in enumerate(binding['bindings']):
            target=work/f'input_{i:03}'
            with archive.open(row['target']) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest)
            key=row['parameter'];indexes=row['indexes'];value=group[key]
            if not indexes:group[key]=linux(target)
            else:
                for j in indexes[:-1]:value=value[j]
                value[indexes[-1]]=linux(target)
        group['root_output_name']=linux(work/'current')
        nml.end_comma=True
        nml.write(work/'CAMx.in')
        original=archive.read(binding['control']['path']).decode('utf-8-sig')
        spelling={m[1].lower():m[1] for m in re.finditer(r'(?m)^\s*(\w+)(?:\([^\n]*?\))?\s*=',original)}
        text=(work/'CAMx.in').read_text()
        text=re.sub(r'(?m)^(\s*)(\w+)(?=\s*[=(])',lambda m:m[1]+spelling.get(m[2].lower(),m[2]),text)
        (work/'CAMx.in').write_text(text,encoding='ascii')
        # The existing external contract imposes the actual supported time/grid
        # bounds. It is not permission to shrink or replace scientific settings.
        inspect(str(work/'CAMx.in'))
        command=['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--cd',linux(work),'--',
            'timeout','--signal=TERM','--kill-after=2',str(timeout),linux(runtime['engine'])]
        try:
            with (folder/'runner.log').open('wb') as log:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=timeout+10)
            text=(folder/'runner.log').read_text(errors='replace')
            ok=proc.returncode==0 and 'END SIMULATION' in text and 'Diagnostic Initialization Test Complete' not in text
            return dict(runtime,status='PASS' if ok else 'FAIL',returncode=proc.returncode,model_read_confirmed=ok,
                full_simulation_verified=False,log_tail=text[-1800:],reason=None if ok else '未获得本次CAMx成功积分终止标记')
        except subprocess.TimeoutExpired:return dict(runtime,status='FAIL',reason='有界试读超时，未确认原生读入',model_read_confirmed=False)


def _headers(files,resources,folder,model):
    """Stream only active dependencies; inspect headers/time, not all data arrays."""
    from netCDF4 import Dataset
    import numpy as np
    checked={};errors=[]
    with TemporaryDirectory(prefix='native_headers_',dir=folder) as work:
        for n,row in enumerate(resources):
            name=row['path'];prefix=files.prefix(name,8)
            if not (prefix.startswith(b'CDF') or prefix.startswith(b'\x89HDF')):continue
            path=Path(work)/str(n)
            with files.files[name][0]() as source,path.open('wb') as target:shutil.copyfileobj(source,target)
            try:
                with Dataset(path) as ds:
                    dims={k:len(v) for k,v in ds.dimensions.items()}
                    meta=dict(status='PASS',dimensions=dims,variables=list(ds.variables),
                        units={k:str(getattr(v,'units','UNKNOWN')) for k,v in ds.variables.items()})
                    if model in {'camx','cmaq'}:
                        if 'TFLAG' not in ds.variables:raise ValueError('缺少原生TFLAG时间标记')
                        flag=np.asarray(ds['TFLAG'][:])
                        if flag.ndim!=3 or flag.shape[-1]!=2 or not flag.size:raise ValueError('TFLAG维度或记录无效')
                        if not np.all(flag==flag[:,:1,:]):raise ValueError('同一时刻变量TFLAG不一致')
                        stamps=[tuple(map(int,a)) for a in flag[:,0,:]]
                        if stamps!=sorted(set(stamps)):raise ValueError('TFLAG时间重复或倒序')
                        from datetime import datetime
                        for day,clock in stamps:
                            date=datetime.strptime(str(day),'%Y%j')
                            if int(date.strftime('%Y%j'))!=day or not 0<=clock//10000<24 or not 0<=clock//100%100<60 or not 0<=clock%100<60:raise ValueError('TFLAG日期/时刻无效')
                        meta.update(first_time=stamps[0],last_time=stamps[-1])
                    checked[name]=meta
            except Exception as exc:
                checked[name]=dict(status='FAIL',error=str(exc));errors.append(name+': '+str(exc))
    return checked,errors


def _cmaq_trial(archive,binding,folder,runtime,timeout):
    """Reuse the reviewed CCTM runtime contract; imported C-shell is never run."""
    from ecodata.interface.cmaq_external_project import inspect_manifest
    from ecodata.interface.cmaq_runner import linux
    candidates=[]
    for info in archive.infolist():
        if not info.filename.endswith('.json') or info.file_size>1024**2:continue
        try:value=json.loads(archive.read(info))
        except (ValueError,UnicodeError):continue
        if isinstance(value,dict) and value.get('schema')=='ecodata.cmaq.local-project.v1':candidates.append(value)
    if len(candidates)!=1:raise ValueError('需唯一完整ecodata.cmaq.local-project.v1运行清单；静态气象绑定不足以证明化学机制和全部输入可运行')
    manifest=candidates[0]
    with TemporaryDirectory(prefix='cmaq_trial_',dir=folder) as work:
        work=Path(work);manifest['inputs']={}
        env=manifest['environment']=dict(manifest['environment'])
        seen=set()
        for i,row in enumerate(binding['bindings']):
            key=row['parameter']
            if key in seen:raise ValueError('多运行日输入不能直接作为单次试读，需明确单日运行契约')
            seen.add(key);dest=work/f'input_{i:03}'
            with archive.open(row['target']) as source,dest.open('wb') as target:shutil.copyfileobj(source,target)
            env[key]=linux(dest);manifest['inputs'][key]=dict(path=str(dest),sha256=row['sha256'])
        validation=inspect_manifest(manifest)
        for key in validation['outputs']:env[key]=linux(work/key)
        for key in ('OUTDIR','LOGDIR','WORKDIR','INPDIR'):
            if key in env:env[key]=linux(work)
        command=['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--cd',linux(work),'--',
            'timeout','--signal=TERM','--kill-after=2',str(timeout),'env','-i',
            *[k+'='+v for k,v in env.items()],'OMP_NUM_THREADS=1',linux(runtime['engine'])]
        try:
            with (folder/'runner.log').open('wb') as log:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=timeout+10)
        except subprocess.TimeoutExpired:
            return dict(runtime,status='FAIL',reason='有界CCTM试读超时，未确认读入',model_read_confirmed=False)
        text='\n'.join(p.read_text(errors='replace') for p in work.rglob('CTM_LOG*'))
        (folder/'native_listing.log').write_text(text,encoding='utf8')
        ok=proc.returncode==0 and 'PROGRAM COMPLETED SUCCESSFULLY' in text
        return dict(runtime,status='PASS' if ok else 'FAIL',returncode=proc.returncode,model_read_confirmed=ok,run_contract_verified=True,
            full_simulation_verified=False,log_tail=text[-1800:],reason=None if ok else '未获得本次CCTM成功终止证据')


def _calpuff(archive,folder,run_native,timeout):
    from existing_air_converters.calpuff_native_bundle import dependency_plan,read_surface,native_name,scalar
    from ecodata.interface.swat_runner import FORBIDDEN_INPUT_SUFFIXES
    plan,rejected,_=dependency_plan(archive)
    resources=[];checked={};errors=[]
    for kind,p in plan.items():
        for row in p['inputs']:
            if Path(row['reference']).suffix.lower() in FORBIDDEN_INPUT_SUFFIXES:
                raise ValueError('原生数据依赖不能携带上传运行库或程序：'+row['reference'])
            resources.append(dict(name=row['member'],sha256=row['sha256'],parameter=row['key']))
            if kind=='CALMET' and row['key']=='SRFDAT':
                try:
                    data,times,ids,zone=read_surface(archive.read(row['member']))
                    checked[row['member']]=dict(status='PASS',records=len(times),stations=ids,timezone_west_hours=zone)
                except Exception as exc:errors.append(str(exc))
    native=_probe('calpuff')
    if not run_native:native.update(status='NOT_RUN',reason='本次调用明确关闭原生试读')
    elif errors:native.update(status='NOT_RUN',reason='原生格式检查失败')
    elif native['status']=='AVAILABLE':
        p=plan['CALPUFF'];header=archive.read(p['control']).splitlines()[0].decode('ascii')
        if not ('5.8' in header):native.update(status='NOT_RUN',reason='控制格式未明确匹配登记的CALPUFF5.8程序：'+header)
        else:
            with TemporaryDirectory(prefix='calpuff_trial_',dir=folder) as work:
                work=Path(work);(work/'control.inp').write_bytes(archive.read(p['control']))
                for row in p['inputs']:
                    ref=row['reference'];ref=ref.lower() if scalar(p['values'],'LCFILES','T').upper()=='T' else ref.upper()
                    dest=work/ref;dest.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(row['member']) as source,dest.open('wb') as target:shutil.copyfileobj(source,target)
                try:
                    with (folder/'runner.log').open('wb') as log:
                        proc=subprocess.run([native['engine'],'control.inp'],cwd=work,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=timeout)
                    listing=work/native_name(p['values'],'PUFLST')
                    text=listing.read_text(encoding='latin1') if listing.exists() else ''
                    ok=proc.returncode==0 and 'TERMINATION PHASE' in text
                    if listing.exists():shutil.copyfile(listing,folder/'native_listing.log')
                    native.update(status='PASS' if ok else 'FAIL',returncode=proc.returncode,model_read_confirmed=ok,
                        reason=None if ok else '未获得本次运行的成功终止证据',full_simulation_verified=False)
                except subprocess.TimeoutExpired:native.update(status='FAIL',reason='有界试读超时；已终止程序，未确认读入',model_read_confirmed=False)
            native['temporary_directory_removed']=not work.exists()
    return dict(format=dict(status='FAIL' if errors else 'PASS',errors=errors,native_files=checked,
                    scope='既有CALMET/CALPUFF5.8依赖计划、控制兼容及SURF回读；二进制气象内部需原生验证'),
        project=dict(status='PASS',validation=dict(errors=[],rejected_controls=rejected)),native=native,
        input_dependency_scope=dict(resources=resources,control_bindings=[dict(kind=k,control=v['control']) for k,v in plan.items()]))


def check_other_model(archive,model,folder,*,run_native=True,timeout=30):
    """Only produced resources enter this gate; source manifests are not proof."""
    from ecodata.interface.engineering_upload import LocalUpload
    from ecodata.models.project_binding import ProjectFiles,bind_project
    folder=Path(folder);started=time.monotonic();runtime=_probe(model)
    result=dict(format=dict(status='UNKNOWN',errors=[]),project=dict(status='UNKNOWN'),native=runtime)
    try:
        if model=='calpuff':
            checked=_calpuff(archive,folder,run_native,timeout)
            checked['elapsed_seconds']=time.monotonic()-started
            return checked
        uploads=[LocalUpload(Path(archive.filename).name,Path(archive.filename))]
        binding=bind_project(uploads,model)
        if binding is None:raise ValueError('输出包中没有可绑定的实际模型控制与依赖；不能将仅有报告的包当完整工程')
        unresolved=binding.get('unresolved',[])
        result['project']=dict(status='UNKNOWN' if unresolved else 'PASS',validation=dict(errors=[],unresolved=unresolved),binding=binding,
            scope='实际激活依赖绑定；不等于所有版本/可选模块完整性认证')
        if model=='cmaq':
            result['project'].update(status='UNKNOWN',reason='核心气象/网格可绑定；完整化学运行清单仍须现有CCTM运行契约验收')
        result['input_dependency_scope']=dict(resources=binding['resources'],control_bindings=binding['bindings'])
        with ExitStack() as stack:
            files=ProjectFiles(uploads,stack)
            checked,errors=_headers(files,binding['resources'],folder,model)
            if model=='wrf':
                from ecodata.interface.wrf_external_project import inspect_project_files
                names={binding['control']['path'],*(r['path'] for r in binding['resources'])}
                raw={n:archive.read(n) for n in names}
                validation=inspect_project_files(raw,namelist_path=binding['control']['path'])
                errors.extend(validation['errors']);result['project']['validation']=validation
                result['project']['status']='FAIL' if validation['errors'] else 'PASS'
            result['format']=dict(status='FAIL' if errors else 'PASS' if checked else 'UNKNOWN',errors=errors,native_files=checked,
                scope='实际依赖NetCDF回读、维度/字段/单位和原生时间标记；非NetCDF二进制及化学/物理完整性不据此认证')
        if not run_native:runtime.update(status='NOT_RUN',reason='本次调用明确关闭原生试读')
        elif errors or unresolved:runtime.update(status='NOT_RUN',reason='格式或依赖检查未通过，未启动模型')
        elif runtime['status']=='AVAILABLE' and model in {'camx','cmaq'}:
            try:result['native']={'camx':_camx_trial,'cmaq':_cmaq_trial}[model](archive,binding,folder,runtime,timeout)
            except Exception as exc:runtime.update(status='NOT_RUN',reason='程序已匹配，但原生运行契约不满足：'+str(exc))
            if result['native'].get('run_contract_verified'):result['project'].update(status='PASS',reason='现有完整原生运行契约已验收')
        elif runtime['status']=='AVAILABLE':
            # Existing external runners require a separate complete local runtime
            # manifest (mechanism/physics tables); mere executable equality is insufficient.
            runtime.update(status='NOT_RUN',reason='已找到指纹匹配程序，但当前包尚未通过现有外部运行器的完整运行清单（版本/化学机制/运行表）验收；不能直接执行上传脚本')
    except Exception as exc:
        result['project']=dict(status='FAIL',validation=dict(errors=[str(exc)]))
        result['format']['reason']='工程绑定未通过，未将归档可读冒充原生格式通过'
        result['native'].update(status='NOT_RUN',reason='工程检查未通过：'+str(exc),model_read_confirmed=False)
    result['elapsed_seconds']=time.monotonic()-started
    return result
