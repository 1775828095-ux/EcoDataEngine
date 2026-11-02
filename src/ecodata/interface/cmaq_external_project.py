"""Capture only named model configuration from the reviewed platform launcher.

No user shell script is executed. Full external project execution is separate
from producing this auditable, resolved environment contract.
"""
from pathlib import Path
import json,re,hashlib,shutil,threading,uuid,subprocess


def launcher_contract(script):
    active='\n'.join(line for line in script.splitlines() if not line.lstrip().startswith('#'))
    keys=sorted(set(re.findall(r'\bsetenv\s+([A-Za-z_][A-Za-z0-9_]*)\b',active)))
    forbidden={'PATH','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','HOME','SHELL','BASH_ENV','ENV'}
    keys=[k for k in keys if k not in forbidden]
    outputs=sorted(set(re.findall(r'\bsetenv\s+([A-Za-z_][A-Za-z0-9_]*)\s+[^\n]*?\$\{?OUTDIR\b',active)))
    return dict(keys=keys,output_keys=outputs)


def instrument(script,folder):
    """Platform-owned observation immediately before CCTM, no science edits."""
    from ecodata.interface.camx_runner import linux
    contract=launcher_contract(script)
    (folder/'environment_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
    helper=folder/'capture_model_environment.py'
    helper.write_text('import os,json\nfrom pathlib import Path\np=Path(__file__).parent\nc=json.loads((p/"environment_contract.json").read_text())\n(p/"resolved_model_environment.json").write_text(json.dumps({k:os.environ[k] for k in c["keys"] if k in os.environ},indent=2))\n',encoding='ascii')
    needle='  ( /usr/bin/time -p $BLD/$EXEC ) |& tee buff_${EXECUTION_ID}.txt'
    if script.count(needle)!=1:raise ValueError('受控CMAQ启动点不唯一，不猜测插入位置')
    return script.replace(needle,'  /usr/bin/python3 "'+linux(helper)+'"\n'+needle)


def windows_path(value):
    m=re.fullmatch(r'/mnt/([a-z])/(.+)',value)
    if not m:raise ValueError('当前仅接本机Windows挂载的工程文件，不执行远程路径：'+value)
    return Path(m[1].upper()+':/'+m[2])


def griddesc_contract(raw,grid_name):
    """Read explicit two-record I/O API projection/grid entries, no defaults."""
    lines=[line.strip() for line in raw.decode('ascii').splitlines() if line.strip() and not line.lstrip().startswith(('!','#'))]
    projections={};grids={}
    number=lambda value:float(value.replace('D','E').replace('d','e'))
    for index,line in enumerate(lines[:-1]):
        match=re.fullmatch(r"'([^']+)'",line)
        if not match or not match[1].strip():continue
        following=lines[index+1];grid=re.fullmatch(r"'([^']+)'\s+(.+)",following)
        if grid:
            values=grid[2].split()
            if len(values)!=7:raise ValueError('GRIDDESC网格记录字段需完整七项')
            grids[match[1]]=(grid[1],list(map(number,values)))
        else:
            values=following.split()
            if len(values)==6:projections[match[1]]=list(map(number,values))
    if grid_name not in grids or grids[grid_name][0] not in projections:raise ValueError('GRIDDESC没有明确匹配GRID_NAME与投影')
    projection,values=grids[grid_name];p=projections[projection]
    return dict(zip(['GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL','NCOLS','NROWS','NTHIK'],p+values))


def physical_input_contract(env,inputs,preserve_original_vertical_headers=False):
    """Check active model inputs against the actual engineering grid and clock."""
    from netCDF4 import Dataset
    import numpy as np
    from datetime import datetime,timedelta
    grid=griddesc_contract(Path(inputs['GRIDDESC']['path']).read_bytes(),env['GRID_NAME'])
    def seconds(s):return int(s[:-4])*3600+int(s[-4:-2])*60+int(s[-2:])
    begin=datetime.strptime(env['CTM_STDATE'],'%Y%j')+timedelta(seconds=seconds(env['CTM_STTIME']))
    end=begin+timedelta(seconds=seconds(env['CTM_RUNLEN']))
    required=['INIT_CONC_1','BNDY_CONC_1','MET_CRO_3D','MET_CRO_2D','MET_BDY_3D']
    try:
        n_gr=int(env['N_EMIS_GR']);n_pt=int(env['N_EMIS_PT'])
        if not 0<=n_gr<=99 or not 0<=n_pt<=99:raise ValueError()
    except (KeyError,ValueError):raise ValueError('排放流数量需明确，当前每类至多99条')
    required += [f'GR_EMIS_{i:03}' for i in range(1,n_gr+1)]
    required += [f'{kind}_{i:03}' for i in range(1,n_pt+1) for kind in ['STK_GRPS','STK_EMIS']]
    if set(required)-set(inputs):raise ValueError('活动模块的必需输入缺失：'+str(sorted(set(required)-set(inputs))))
    checked=[];vertical=None;differences=[]
    for key in ['MET_CRO_3D','MET_CRO_2D','MET_BDY_3D','INIT_CONC_1','BNDY_CONC_1',*[f'GR_EMIS_{i:03}' for i in range(1,n_gr+1)]]:
        path=Path(inputs[key]['path'])
        # Current Windows NetCDF library cannot reliably open Unicode paths.
        # Read bytes when needed; never rename the user's file outside platform.
        kw={} if str(path).isascii() else {'memory':path.read_bytes()}
        with Dataset(str(path) if not kw else 'input',**kw) as data:
            for name in ['GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL','NCOLS','NROWS']:
                if name not in data.ncattrs() or not np.array_equal(data.getncattr(name),grid[name]):raise ValueError('工程物理网格引用不一致：'+key+'.'+name)
            if key=='MET_CRO_3D':vertical={n:np.asarray(data.getncattr(n)) for n in ['NLAYS','VGTYP','VGTOP','VGLVLS']}
            if key in ['MET_BDY_3D','INIT_CONC_1','BNDY_CONC_1']:
                for name,value in vertical.items():
                    if name not in data.ncattrs():raise ValueError('工程垂直定义缺失：'+key+'.'+name)
                    if not np.array_equal(data.getncattr(name),value):
                        if name!='VGTYP' or not preserve_original_vertical_headers:raise ValueError('工程垂直定义不一致：'+key+'.'+name+'；不会修改原数据。若仅复验原工程，可明确选择保留类型码差异运行；不认证物理等价')
                        differences.append(dict(key=key,attribute=name,meteorology_value=value.tolist(),input_value=np.asarray(data.getncattr(name)).tolist(),policy='Explicitly preserve original differing type codes, not physical-equivalence certification'))
            flags=np.asarray(data['TFLAG'][:])
            if flags.ndim!=3 or flags.shape[2]!=2 or not np.array_equal(flags,np.broadcast_to(flags[:,0:1,:],flags.shape)):raise ValueError('输入各物种时标不一致：'+key)
            times=[datetime.strptime(str(int(day)),'%Y%j')+timedelta(seconds=seconds(str(int(hour)).zfill(6))) for day,hour in flags[:,0]]
            if key=='INIT_CONC_1':
                if begin not in times:raise ValueError('初始场没有模拟起始时刻，不改日期：'+key)
            elif int(getattr(data,'TSTEP',0))==0:
                if key not in {f'GR_EMIS_{i:03}' for i in range(1,n_gr+1)}:raise ValueError('气象或边界不能作为静态排放处理：'+key)
            elif not times or min(times)>begin or max(times)<end or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('气象/边界/排放未覆盖所选运行时段：'+key)
            checked.append(dict(key=key,timestamps=[t.isoformat() for t in times],grid_matches=True))
    return dict(grid=grid,vertical={k:v.tolist() for k,v in vertical.items()},begin=begin.isoformat(),end=end.isoformat(),checked=checked,active_input_keys=required,vertical_header_differences=differences,
                preserve_original_vertical_headers=bool(preserve_original_vertical_headers),physical_equivalence_certified=False)


def export_manifest(ident):
    """Export exact resolved configuration and dependency hashes, not helper data."""
    from ecodata.interface import cmaq_runner as runner
    folder=runner.directory(ident)
    if runner.status(ident)['status']!='success':raise ValueError('仅导出已成功且记录了生效环境的本机任务')
    env=json.loads((folder/'resolved_model_environment.json').read_text())
    contract=json.loads((folder/'environment_contract.json').read_text())
    # These are previous-day INPUTS despite their $OUTDIR location. Preserve
    # available bytes or declare missing; never turn them into output files.
    outputs=set(contract['output_keys'])-{'BEIS_SOILINP','MEGAN_SOILINP','BDSNPINP','ISAM_PREVDAY','LOGDIR'}
    dirs={'OUTDIR','LOGDIR','WORKDIR','INPDIR'}
    inputs={};unavailable={}
    for key,value in env.items():
        if key in outputs or key in dirs or not value.startswith('/'):continue
        path=windows_path(value) if value!='/ ' and value!='/' else None
        if path is not None and path.is_file():inputs[key]=dict(path=str(path),sha256=runner.sha(path),bytes=path.stat().st_size)
        else:unavailable[key]=value
    return dict(schema='ecodata.cmaq.local-project.v1',engine_sha256=runner.PIN,
                mechanism='cb6r5_ae7_aq_m3dry',source_job=ident,environment=env,
                inputs=inputs,unavailable_inputs=unavailable,
                qualification='Local file references, not a portable full-data archive; unavailable optional inputs are not filled. Native use would fail.',
                scientific_settings_modified=False)


def inspect_manifest(manifest,preserve_original_vertical_headers=False):
    from ecodata.interface import cmaq_runner as runner
    if manifest.get('schema')!='ecodata.cmaq.local-project.v1' or manifest.get('engine_sha256')!=runner.PIN or manifest.get('mechanism')!='cb6r5_ae7_aq_m3dry':
        raise ValueError('当前仅支持受控CMAQ5.5 CB6R5/AE7/m3dry工程清单')
    contract=launcher_contract((runner.CASE/'native_platform_01/run.csh').read_text())
    env=manifest['environment']
    if not isinstance(env,dict) or set(env)-set(contract['keys']):raise ValueError('含非模型配置或不支持的环境键；不接受命令、动态库或解释器配置')
    if any(not isinstance(v,str) or '\0' in v or len(v)>32768 or '$' in v or '\n' in v for v in env.values()):raise ValueError('环境值必须为已解析的单行原文')
    from ecodata.interface.engineering_completeness import cmaq,require
    completeness=cmaq(manifest);require(completeness)
    required={'CTM_STDATE','CTM_STTIME','CTM_RUNLEN','CTM_TSTEP','GRID_NAME','INIT_CONC_1','BNDY_CONC_1','GRIDDESC','S_CGRID','MET_CRO_3D','gc_matrix_nml','ae_matrix_nml','nr_matrix_nml','tr_matrix_nml','CSQY_DATA','DESID_CTRL_NML','DESID_CHEM_CTRL_NML'}
    if required-set(env):raise ValueError('模型生效配置不完整：'+str(sorted(required-set(env))))
    def seconds(s):
        if not re.fullmatch(r'\d{6}',s) or int(s[2:4])>59 or int(s[4:])>59:raise ValueError('时长格式无效')
        return int(s[:2])*3600+int(s[2:4])*60+int(s[4:])
    if not 60<=seconds(env['CTM_RUNLEN'])<=3600 or seconds(env['CTM_TSTEP'])<60:raise ValueError('当前受控短跑时长为1至60分钟，不更改用户时长')
    outputs=set(contract['output_keys'])-{'BEIS_SOILINP','MEGAN_SOILINP','BDSNPINP','ISAM_PREVDAY','LOGDIR'}
    inputs=manifest.get('inputs',{});unavailable=manifest.get('unavailable_inputs',{})
    if set(inputs)&set(unavailable) or (set(inputs)|set(unavailable))-set(env):raise ValueError('依赖映射重复或未在环境中引用')
    for key,value in unavailable.items():
        if value!=env[key] or not value.startswith('/') or key in outputs|{'OUTDIR','LOGDIR','WORKDIR','INPDIR'}:raise ValueError('缺失项必须是原环境中的输入路径，不得改变标量配置或输出')
    files=[]
    for key,item in inputs.items():
        if key in outputs or key in {'OUTDIR','LOGDIR','WORKDIR','INPDIR'}:raise ValueError('输出或目录不能伪装成输入')
        path=Path(item['path'])
        if not path.is_absolute() or not path.is_file() or runner.sha(path)!=item['sha256']:raise ValueError('输入缺失或哈希变化：'+key)
        if str(path.resolve()).casefold()!=str(windows_path(env[key]).resolve()).casefold():raise ValueError('环境引用与实际路径不一致：'+key)
        files.append(dict(key=key,path=str(path),sha256=item['sha256'],bytes=path.stat().st_size))
    dirs={'OUTDIR','LOGDIR','WORKDIR','INPDIR'}
    for key,value in env.items():
        if value.startswith('/') and key not in outputs|dirs|set(inputs)|set(unavailable):raise ValueError('工程引用没有明确依赖声明：'+key)
    for key in ('INIT_CONC_1','BNDY_CONC_1','GRIDDESC','MET_CRO_3D','gc_matrix_nml','ae_matrix_nml','nr_matrix_nml','tr_matrix_nml','CSQY_DATA','DESID_CTRL_NML','DESID_CHEM_CTRL_NML'):
        if key not in inputs:raise ValueError('必需工程依赖不可缺失：'+key)
    from netCDF4 import Dataset
    with Dataset('met',memory=Path(inputs['MET_CRO_3D']['path']).read_bytes()) as data:
        dims={k:len(v) for k,v in data.dimensions.items()}
        if dims.get('COL',9999)>300 or dims.get('ROW',9999)>300 or dims.get('LAY',9999)>50:raise ValueError('当前工程网格超出受控短跑范围')
    physical=physical_input_contract(env,inputs,preserve_original_vertical_headers)
    return dict(files=files,unavailable_inputs=unavailable,outputs=sorted(outputs&set(env)),physical_contract=physical,completeness=completeness,
                settings_preserved=True,native_executed=False,scope='Declared resolved local project; no shell scripts executed, no missing data filled')


def start_manifest(manifest,converted_zip=None,target_key=None,member_name=None,preserve_original_vertical_headers=False):
    from ecodata.interface import cmaq_runner as runner
    report=inspect_manifest(manifest,preserve_original_vertical_headers)
    if converted_zip is not None:
        if target_key not in manifest['inputs'] or not re.fullmatch(r'GR_EMIS_\d{3}',target_key or ''):raise ValueError('当前只允许明确选择完整网格排放流进行替换')
        from ecodata.interface.registered_conversion import member,compatible_netcdf
        raw=member(converted_zip,member_name)
        report['replacement']=compatible_netcdf(raw,Path(manifest['inputs'][target_key]['path']))
    if not runner.LOCK.acquire(blocking=False):raise ValueError('已有CMAQ任务运行中')
    ident=uuid.uuid4().hex
    try:
        runner.LEASE=runner.acquire_lease()
        folder=runner.directory(ident);folder.mkdir(parents=True,exist_ok=False)
        (folder/'uploaded_project.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
        if converted_zip is not None:(folder/'current_conversion.zip').write_bytes(converted_zip)
        runner.save(folder,dict(id=ident,route='external_local',status='running',scope=report['scope'],inspection=report,accuracy_evaluated=False))
        threading.Thread(target=_work,args=(ident,manifest,report,converted_zip,target_key,member_name),daemon=True).start()
        return ident
    except Exception:runner.release_lease();runner.LOCK.release();raise


def _work(ident,manifest,report,converted_zip,target_key,member_name):
    from ecodata.interface import cmaq_runner as runner
    folder=runner.directory(ident);state=runner.status(ident)
    try:
        if runner.sha(runner.EXE)!=runner.PIN:raise ValueError('原生引擎哈希变化')
        env=dict(manifest['environment']);(folder/'inputs').mkdir();(folder/'output/LOGS').mkdir(parents=True)
        mappings=[]
        for index,item in enumerate(report['files']):
            source=Path(item['path']);dest=folder/'inputs'/f'{index:03}_{source.name}'
            if runner.sha(source)!=item['sha256']:raise ValueError('运行前输入变化')
            shutil.copyfile(source,dest)
            if runner.sha(dest)!=item['sha256']:raise ValueError('工程副本字节不一致')
            env[item['key']]=runner.linux(dest);mappings.append(dict(key=item['key'],source=str(source),copy=str(dest),sha256=item['sha256']))
        for key in report['unavailable_inputs']:env[key]=runner.linux(folder/'unavailable'/key)
        for key in report['outputs']:env[key]=runner.linux(folder/'output'/(key+'.nc'))+(' -v' if manifest['environment'][key].endswith(' -v') else '')
        env.update(OUTDIR=runner.linux(folder/'output'),LOGDIR=runner.linux(folder/'output/LOGS'),WORKDIR=runner.linux(folder))
        if converted_zip is not None:
            from ecodata.interface.registered_conversion import member
            replacement=folder/'current_emissions.nc';replacement.write_bytes(member(converted_zip,member_name));env[target_key]=runner.linux(replacement)
        (folder/'effective_environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
        (folder/'file_mapping.json').write_text(json.dumps(mappings,indent=2),encoding='utf8')
        # Direct fixed executable; neither imported scripts nor their commands run.
        worker=folder/'native_worker.py'
        worker.write_text('import os,json,resource,subprocess,sys\nfrom pathlib import Path\np=Path(__file__).parent\nresource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))\ne=dict(os.environ,**json.loads((p/"effective_environment.json").read_text()),OMP_NUM_THREADS="1")\nsys.exit(subprocess.run([sys.argv[1]],cwd=p,env=e,stdin=subprocess.DEVNULL).returncode)\n',encoding='ascii')
        with (folder/'launcher.log').open('wb') as log:
            proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--cd',runner.linux(folder),'--','python3',runner.linux(worker),runner.linux(runner.EXE)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        logs=list(folder.glob('**/CTM_LOG*'))
        text='\n'.join(p.read_text(errors='replace') for p in logs)
        if proc.returncode or 'PROGRAM COMPLETED SUCCESSFULLY' not in text:raise ValueError('CCTM原生运行未完成，保留当前日志，不回退登记案例')
        for key in ('INIT_CONC_1','BNDY_CONC_1','MET_CRO_3D'):
            if env[key] not in text:raise ValueError('原生日志未确认本次工程输入：'+key)
        if converted_zip is not None and env[target_key] not in text:raise ValueError('原生日志未确认读取本次替换排放')
        result=plot_result(folder)
        from datetime import datetime,timedelta
        def seconds(s):return int(s[:-4])*3600+int(s[-4:-2])*60+int(s[-2:])
        expected=datetime.strptime(env['CTM_STDATE'],'%Y%j')+timedelta(seconds=seconds(env['CTM_STTIME'])+seconds(env['CTM_RUNLEN']))
        if result['final_time']!=[int(expected.strftime('%Y%j')),int(expected.strftime('%H%M%S'))]:raise ValueError('原生输出最终时刻与原工程设置不一致')
        state.update(result,status='success',returncode=proc.returncode,native_read_verified=True,engine_sha256=runner.PIN)
        state['original_inputs_unchanged']=all(runner.sha(Path(i['path']))==i['sha256'] for i in report['files'])
        if not state['original_inputs_unchanged']:raise ValueError('源输入在运行期间发生变化')
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        try:runner.save(folder,state)
        finally:runner.release_lease();runner.LOCK.release()


def plot_result(folder):
    import numpy as np
    import pandas as pd
    from netCDF4 import Dataset
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with Dataset(folder/'output/S_CGRID.nc') as data:
        variable=data['O3'];units=variable.units.strip()
        if units!='ppmV':raise ValueError('当前O3结果单位未支持')
        values=np.ma.asarray(variable[-1,0])
        if np.ma.getmaskarray(values).any() or not np.isfinite(values).all() or (values<0).any():raise ValueError('输出存在缺测或无效浓度')
        values=np.asarray(values,dtype='float64')*1000;ny,nx=values.shape
        x=(float(data.XORIG)+(np.arange(nx)+.5)*float(data.XCELL))/1000
        y=(float(data.YORIG)+(np.arange(ny)+.5)*float(data.YCELL))/1000
        stamp=np.asarray(data['TFLAG'][-1,0]).astype(int).tolist()
    xx,yy=np.meshgrid(x,y);pd.DataFrame(dict(x_km=xx.ravel(),y_km=yy.ravel(),O3_ppbv=values.ravel())).to_csv(folder/'O3_surface.csv',index=False)
    fig=Figure(figsize=(7,6));FigureCanvasAgg(fig);ax=fig.subplots();mesh=ax.pcolormesh(x,y,values,shading='nearest');fig.colorbar(mesh,ax=ax,label='O3 (ppbv)');ax.set(title=f'CMAQ native output {stamp}; accuracy not assessed',xlabel='Projected x (km)',ylabel='Projected y (km)');fig.tight_layout();fig.savefig(folder/'O3_surface.png',dpi=140)
    return dict(shape=[ny,nx],final_time=stamp,max_O3_ppbv=float(values.max()),result_sha256=hashlib.sha256((folder/'output/S_CGRID.nc').read_bytes()).hexdigest())


def render(converted_zip=None,mode=None):
    import streamlit as st
    from ecodata.interface import cmaq_runner as runner
    st.header('CMAQ：已有完整工程接入')
    st.warning('受控CMAQ5.5 CB6R5/AE7/m3dry，1–60分钟。支持含明确配置的工程ZIP，或本机配置清单。不执行用户脚本、不补物种、不改科学选项；不能自动解释任意作者脚本。')
    st.write('生效配置清单 → 引用与哈希检查 → 字节相同的工程副本 → 可选完整排放流替换 → CCTM原生运行 → 实际读取核验 → 图表与下载')
    with st.expander('从已成功的平台任务导出生效配置'):
        source=st.text_input('已记录生效环境的CMAQ任务号',key='cmaq_export_env_job')
        if st.button('生成工程生效配置清单',key='cmaq_export_env'):
            try:st.session_state['cmaq_export_env_bytes']=json.dumps(export_manifest(source),ensure_ascii=False,indent=2).encode('utf8')
            except Exception as exc:st.error(str(exc))
        if st.session_state.get('cmaq_export_env_bytes'):
            st.download_button('下载CMAQ已有工程生效配置',st.session_state['cmaq_export_env_bytes'],'CMAQ_existing_project.json',key='cmaq_env_download')
    package=st.radio('已有工程提交方式',['完整工程ZIP','本机配置JSON'],key='cmaq_external_upload_format')
    is_zip=package=='完整工程ZIP'
    if is_zip:
        st.caption('ZIP内需有一份cmaq_project.json（schema: ecodata.cmaq.project-zip.v1）；environment和inputs中对应文件使用相对此JSON的包内路径，其他科学配置需明确。缺文件会列出，不使用本机同名文件代替。当前压缩包至多1GiB，解压后至多8GiB；上传还受服务设置限制。')
    uploaded=st.file_uploader('完整工程ZIP（包含配置与实际输入文件）' if is_zip else '已有工程生效配置JSON（引用本机原文件及哈希）',type=['zip'] if is_zip else ['json'],key='cmaq_external_zip' if is_zip else 'cmaq_external_env',max_upload_size=1024 if is_zip else 200)
    preserve_headers=st.checkbox('仅复验原工程：保留已记录的垂直坐标类型码差异运行（不认证物理等价）',key='cmaq_external_preserve_headers')
    if preserve_headers:st.warning('仅允许保留VGTYP类型码差异；层数、顶压、层界、网格和时段仍必须一致。原文件不改头，原科学设置不变；能运行不代表两种垂直坐标物理等价。')
    from ecodata.interface.engineering_conversion import render as render_conversion
    converted_zip,mode,conversion_ready=render_conversion('cmaq',converted_zip,mode)
    manifest=None;target=None;ready=False
    if uploaded is not None:
        try:
            if is_zip:
                from ecodata.interface.cmaq_project_zip import unpack
                manifest=unpack(uploaded.getvalue(),Path(__file__).resolve().parents[3]/'outputs/cmaq_uploaded_projects')
            else:
                if len(uploaded.getvalue())>2*1024**2:raise ValueError('工程清单超过2MiB')
                manifest=json.loads(uploaded.getvalue().decode('utf-8-sig'))
            report=inspect_manifest(manifest,preserve_headers);ready=True
            from ecodata.interface.engineering_completeness import render as render_completeness
            ready=render_completeness(report['completeness'],'cmaq')
            with st.expander('实际依赖、缺失项及配置保留检查'):st.json(report)
            if report['unavailable_inputs']:st.info('未提供的输入按本次工程开关分类；无法确认用途的缺失项阻止运行，不一律视为可选。')
            if converted_zip is not None:
                if mode!='gridded_emissions':raise ValueError('此新入口当前仅接完整网格排放流；其他类别不自动接入')
                target=st.selectbox('明确替换哪一个完整排放流',[k for k in manifest['inputs'] if re.fullmatch(r'GR_EMIS_\d{3}',k)],index=None,key='cmaq_external_target')
                ready=target is not None
        except Exception as exc:
            if hasattr(exc,'completeness'):
                from ecodata.interface.engineering_completeness import render as render_completeness
                render_completeness(exc.completeness,'cmaq')
            elif manifest is not None:
                from ecodata.interface.engineering_completeness import cmaq,from_validation,render as render_completeness
                render_completeness(from_validation('CMAQ',dict(errors=[str(exc)]),cmaq(manifest)['findings']),'cmaq')
            st.error(str(exc));ready=False
    from ecodata.interface.input_result_binding import bind_result
    bind_result(st.session_state,'cmaq_external_job','cmaq_external_input_binding',uploaded.getvalue() if uploaded else b'',converted_zip or b'',target,preserve_headers)
    ready=ready and conversion_ready
    if st.button('检查并运行本次CMAQ工程副本',disabled=not ready,key='cmaq_external_start'):
        st.session_state.pop('cmaq_external_job',None)
        try:st.session_state['cmaq_external_job']=start_manifest(manifest,converted_zip,target,'emissions.CMAQ55.nc' if converted_zip is not None else None,preserve_original_vertical_headers=preserve_headers)
        except Exception as exc:st.error(str(exc))
    ident=st.session_state.get('cmaq_external_job')
    if ident and ready:
        @st.fragment(run_every=5)
        def view():
            record=runner.status(ident);folder=runner.directory(ident);st.write('本次CMAQ工程任务：'+ident)
            if record['status']=='running':st.info('正在复制原文件并执行CCTM；科学配置未改动。')
            elif record['status']=='failed':st.error(record.get('error'))
            else:
                st.success('本次CCTM积分、输入读取核验与出图完成；未评价预测精度。');st.image(str(folder/'O3_surface.png'))
                for label,name in [('下载本次CMAQ网格CSV','O3_surface.csv'),('下载本次CMAQ结果PNG','O3_surface.png'),('下载本次CMAQ运行记录','result.json')]:
                    st.download_button(label,(folder/name).read_bytes(),name,key='cmaq_external_'+name)
            with st.expander('本次状态与日志位置'):st.json(record);st.code(str(folder))
        view()
    return True
