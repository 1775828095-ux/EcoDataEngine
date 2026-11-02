"""Local complete CAMx namelist import; never execute uploaded programs."""
from pathlib import Path
from datetime import datetime
import re,json,shutil,threading,uuid,subprocess

FILES={'Chemistry_Parameters','Photolysis_Rates','Ozone_Column','Initial_Conditions','Boundary_Conditions','Master_Grid_Restart','Nested_Grid_Restart'}
PATTERN=re.compile(r'\s*([A-Za-z][A-Za-z_0-9]*(?:\([\d, ]+\))?)\s*=\s*(.*?)\s*,\s*(?:!.*)?$')


def inspect(control_path):
    from ecodata.interface import camx_runner as runner
    path=Path(control_path)
    if not path.is_absolute() or not path.is_file():raise ValueError('请选择本机完整工程原控制文件的绝对路径')
    raw=path.read_bytes()
    if len(raw)>1024**2:raise ValueError('控制文件超过1MiB')
    text=raw.decode('utf-8-sig');rows=[];values={}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('!') or line.strip().lower() in ('&camx_control','/'):continue
        m=PATTERN.fullmatch(line)
        if not m:raise ValueError('当前需一行一个完整赋值，不猜测解析：'+line[:100])
        key,value=m.groups();key=re.sub(r'\s','',key)
        if '$' in value or '\0' in value:raise ValueError('控制须为已解析原值，不执行变量或脚本')
        if key in values and not key.startswith('Output_Species_Names('):raise ValueError('控制赋值重复：'+key)
        values[key]=value;rows.append((key,value))
    from ecodata.interface.engineering_completeness import camx,require
    completeness=camx(path,values);require(completeness)
    def one(key):
        if key not in values:raise ValueError('缺必要控制：'+key)
        return values[key]
    def stamp(key):
        parts=one(key).split(',')
        if len(parts)!=4:raise ValueError('模拟日期格式无效')
        a=list(map(int,parts));return datetime(*a[:3],a[3]//100,a[3]%100)
    minutes=(stamp('End_Date_Hour')-stamp('Start_Date_Hour')).total_seconds()/60
    if not 1<=minutes<=60:raise ValueError('当前限定1至60分钟，不修改用户模拟时段')
    if one('Restart').lower()!='.false.' or one('Diagnostic_Error_Check').lower()!='.false.':raise ValueError('当前需非重启、实际积分配置；不更改科学选项')
    if not 1<=int(one('Number_of_Grids'))<=2 or not 1<=int(one('Number_of_Layers'))<=35 or any(not 1<=int(one(k))<=200 for k in ('Master_Grid_Columns','Master_Grid_Rows')):raise ValueError('超出当前受控网格范围')
    inputs=[];ignored=[]
    from ecodata.interface.cmaq_external_project import windows_path
    for key,value in rows:
        is_file=key in FILES or re.fullmatch(r'(?:Point_Sources|Surface_Grid|Met3D_Grid|Met2D_Grid|Vdiff_Grid|Emiss_Grid)\([\d,]+\)',key)
        if not is_file:
            if '/' in value and key!='Root_Output_Name':raise ValueError('尚未识别的外部引用：'+key)
            continue
        if not re.fullmatch(r"'[^'\r\n]+'",value):raise ValueError('文件引用需明确单引号原路径：'+key)
        original=value[1:-1]
        if key in {'Master_Grid_Restart','Nested_Grid_Restart'}:ignored.append(dict(key=key,reason='Restart=false: inactive original restart reference',original=original));continue
        source=windows_path(original) if original.startswith('/mnt/') else Path(original) if Path(original).is_absolute() else (path.parent/original).resolve()
        if not source.is_file():raise ValueError('工程实际引用的文件缺失：'+key+' '+str(source))
        inputs.append(dict(key=key,path=str(source),sha256=runner.sha(source),bytes=source.stat().st_size))
    present={r['key'] for r in inputs}
    for key in ('Chemistry_Parameters','Photolysis_Rates','Ozone_Column','Initial_Conditions','Boundary_Conditions','Surface_Grid(1)','Met3D_Grid(1)','Met2D_Grid(1)','Vdiff_Grid(1)'):
        if key not in present:raise ValueError('缺少完整工程依赖：'+key)
    chemistry=next(i for i in inputs if i['key']=='Chemistry_Parameters')
    header=Path(chemistry['path']).open(encoding='ascii').readline().strip()
    if not re.fullmatch(r'CAMx Version\s*\|VERSION7\.4',header):raise ValueError('化学参数不是当前VERSION7.4，不改版本头强接引擎')
    return dict(control=str(path),control_sha256=runner.sha(path),inputs=inputs,inactive_references=ignored,completeness=completeness,
                minutes=minutes,grids=int(values['Number_of_Grids']),start=stamp('Start_Date_Hour').isoformat(),end=stamp('End_Date_Hour').isoformat(),
                original_assignments=rows,scientific_settings_preserved=True,native_executed=False,
                duplicate_output_species_indexes=[k for k,v in rows if sum(k==a for a,b in rows)>1],
                scope='Pinned CAMx7.40 local resolved namelist, copied dependencies; version/physics compatibility ultimately requires native acceptance')


def start(control_path,converted_zip=None,target_key=None,member_name=None):
    from ecodata.interface import camx_runner as runner
    report=inspect(control_path)
    if converted_zip is not None:
        if not (target_key in {'Initial_Conditions','Boundary_Conditions'} or re.fullmatch(r'Emiss_Grid\([\d,]+\)',target_key or '')):raise ValueError('只允许明确的完整排放或初边界流替换')
        target=next((i for i in report['inputs'] if i['key']==target_key),None)
        if target is None:raise ValueError('替换目标未在原控制中引用')
        from ecodata.interface.registered_conversion import member,compatible_netcdf
        if target_key.startswith('Emiss_Grid('):
            report['replacement']=emission_subset(converted_zip,Path(target['path']),report)
        else:report['replacement']=compatible_netcdf(member(converted_zip,member_name),Path(target['path']))
    if not runner.LOCK.acquire(blocking=False):raise ValueError('已有CAMx任务运行中')
    ident=uuid.uuid4().hex
    try:
        folder=runner.directory(ident);folder.mkdir(parents=True,exist_ok=False)
        (folder/'original_control.INP').write_bytes(Path(control_path).read_bytes())
        if converted_zip is not None:(folder/'current_conversion.zip').write_bytes(converted_zip)
        runner.save(folder,dict(id=ident,route='external_local',status='running',inspection=report,accuracy_evaluated=False))
        threading.Thread(target=work,args=(ident,report,converted_zip,target_key,member_name),daemon=True).start();return ident
    except Exception:runner.LOCK.release();raise


def work(ident,report,converted_zip,target_key,member_name):
    from ecodata.interface import camx_runner as runner
    folder=runner.directory(ident);state=runner.status(ident)
    try:
        if runner.sha(runner.EXE)!=runner.PIN:raise ValueError('引擎哈希变化')
        if runner.sha(Path(report['control']))!=report['control_sha256']:raise ValueError('原控制文件发生变化')
        (folder/'inputs').mkdir();(folder/'outputs').mkdir();mapping={};records=[]
        for index,item in enumerate(report['inputs']):
            source=Path(item['path']);dest=folder/'inputs'/f'{index:03}_{source.name}'
            if runner.sha(source)!=item['sha256']:raise ValueError('原输入在检查后发生变化')
            shutil.copyfile(source,dest)
            if runner.sha(dest)!=item['sha256']:raise ValueError('工程副本字节不符')
            mapping[item['key']]="'"+runner.linux(dest)+"'";records.append(dict(**item,copy=str(dest)))
        if converted_zip is not None:
            from ecodata.interface.registered_conversion import member
            dest=folder/'current_input.nc'
            if target_key.startswith('Emiss_Grid('):
                source=Path(next(i for i in report['inputs'] if i['key']==target_key)['path'])
                applied=emission_subset(converted_zip,source,report,dest)
                (folder/'species_replacement.json').write_text(json.dumps(applied,indent=2),encoding='utf8')
            else:dest.write_bytes(member(converted_zip,member_name))
            mapping[target_key]="'"+runner.linux(dest)+"'"
        for row in report['inactive_references']:mapping[row['key']]="'"+runner.linux(folder/'inactive'/row['key'])+"'"
        mapping['Root_Output_Name']="'"+runner.linux(folder/'outputs/CAMx.current')+"'"
        lines=[]
        for line in (folder/'original_control.INP').read_text(encoding='utf-8-sig').splitlines(keepends=True):
            m=PATTERN.fullmatch(line.rstrip('\r\n'))
            if m:
                key=re.sub(r'\s','',m[1])
                if key in mapping:line=line[:m.start(2)]+mapping[key]+line[m.end(2):]
            lines.append(line)
        (folder/'CAMx.in').write_bytes(''.join(lines).encode('ascii'))
        (folder/'file_mapping.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        worker=folder/'worker.py'
        worker.write_text('import os,subprocess,resource,sys\nfrom pathlib import Path\np=Path(__file__).parent\nresource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))\nenv=dict(os.environ,OMP_NUM_THREADS="1",OMP_STACKSIZE="128M")\nwith (p/"console.log").open("wb") as log:r=subprocess.run([sys.argv[1]],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)\nsys.exit(r.returncode)\n',encoding='ascii')
        proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','python3',runner.linux(worker),runner.linux(runner.EXE)],capture_output=True,timeout=960)
        (folder/'launcher.log').write_bytes(proc.stdout+proc.stderr)
        text=(folder/'console.log').read_text(errors='replace')
        if proc.returncode or 'END SIMULATION' not in text or 'Diagnostic Initialization Test Complete' in text:raise ValueError('CAMx未完成实际积分，保留原生日志')
        ending=datetime.fromisoformat(report['end']).strftime('Date/time: %y/%m/%d %H:%M:%S')
        if ending not in text:raise ValueError('原生结束时刻与用户原控制不符')
        log=(folder/'outputs/CAMx.current.out').read_text(errors='replace')
        for key in ('Initial_Conditions','Boundary_Conditions','Met3D_Grid(1)'):
            if mapping[key].strip("'") not in log:raise ValueError('日志未确认读取本次工程输入：'+key)
        if converted_zip is not None and mapping[target_key].strip("'") not in log:raise ValueError('日志未确认读取本次替换资料')
        state.update(plot(folder,report),status='success',native_read_verified=True,engine_sha256=runner.PIN)
        state['original_inputs_unchanged']=all(runner.sha(Path(i['path']))==i['sha256'] for i in report['inputs']) and runner.sha(Path(report['control']))==report['control_sha256']
        if not state['original_inputs_unchanged']:raise ValueError('源资料在运行期间变化')
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:runner.save(folder,state);runner.LOCK.release()


def plot(folder,report):
    import numpy as np,pandas as pd
    from netCDF4 import Dataset
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(6*report['grids'],5));FigureCanvasAgg(fig);axes=np.atleast_1d(fig.subplots(1,report['grids']));results=[]
    for grid,ax in enumerate(axes,1):
        output=folder/f'outputs/CAMx.current.avrg.grd{grid:02}.nc'
        with Dataset(output) as data:
            if data['O3'].units.strip()!='ppmv':raise ValueError('O3单位不匹配')
            a=np.ma.asarray(data['O3'][-1,0])
            if np.ma.getmaskarray(a).any() or not np.isfinite(a).all() or (a<0).any():raise ValueError('结果缺失或无效')
            a=np.asarray(a,dtype='float64')*1000;ny,nx=a.shape
            x=(float(data.XORIG)+(np.arange(nx)+.5)*float(data.XCELL))/1000;y=(float(data.YORIG)+(np.arange(ny)+.5)*float(data.YCELL))/1000
            stamp=np.asarray(data['TFLAG'][-1,0]).astype(int).tolist()
        xx,yy=np.meshgrid(x,y);pd.DataFrame(dict(x_km=xx.ravel(),y_km=yy.ravel(),O3_ppbv=a.ravel())).to_csv(folder/f'O3_grid{grid}.csv',index=False)
        mesh=ax.pcolormesh(x,y,a,shading='nearest');fig.colorbar(mesh,ax=ax,label='O3 ppbv');ax.set(title=f'Grid {grid}: native mean O3',xlabel='Projected x (km)',ylabel='Projected y (km)')
        results.append(dict(grid=grid,shape=[ny,nx],time=stamp,max_O3_ppbv=float(a.max())))
    fig.suptitle('CAMx existing project; accuracy not assessed');fig.tight_layout();fig.savefig(folder/'result.png',dpi=140)
    return dict(grids=results)


def emission_subset(payload,reference,engineering,destination=None):
    """Update supplied ground emissions, preserve all other original species."""
    import numpy as np
    from netCDF4 import Dataset
    from ecodata.interface.registered_conversion import member
    from ecodata.interface import camx_runner as runner
    conversion=json.loads(member(payload,'conversion_report.json'))
    chemistry=Path(next(i for i in engineering['inputs'] if i['key']=='Chemistry_Parameters')['path'])
    canonical=runner.CASE/'inputs/CAMx7.40.chemparam.CB6r5_CF3_COMPLX'
    if conversion.get('target')!='CAMx-7.40' or conversion.get('mechanism')!='CB6r5_CF3_COMPLX' or runner.sha(chemistry)!=runner.sha(canonical):raise ValueError('当前部分排放接入需已核验CB6r5_CF3_COMPLX机制；不翻译物种')
    raw=member(payload,'emissions.CAMx7.40.nc');fields=[]
    with Dataset('current',memory=raw) as src,Dataset('reference',memory=reference.read_bytes()) as dst:
        for key in ('NCOLS','NROWS','NLAYS','SDATE','STIME','TSTEP','NSTEPS','XORIG','YORIG','XCELL','YCELL','XCENT','YCENT','P_ALP','P_BET','P_GAM','CPROJ','GDTYP','IUTM','ISTAG','ITZON'):
            if key not in src.ncattrs() or key not in dst.ncattrs() or not np.array_equal(src.getncattr(key),dst.getncattr(key)):raise ValueError('排放物理网格或时段不一致：'+key)
        if int(src.NLAYS)!=1 or int(dst.NLAYS)!=1:raise ValueError('当前部分排放只支持明确的地面单层，不推断垂直分配')
        fields=[n for n,v in src.variables.items() if v.dimensions==('TSTEP','LAY','ROW','COL')]
        if not fields or set(fields)!=set(conversion.get('species',[])):raise ValueError('转换报告与实际排放物种不一致')
        names=lambda d:[d.getncattr('VAR-LIST')[i:i+16].strip() for i in range(0,len(d.getncattr('VAR-LIST')),16)]
        sn,dn=names(src),names(dst)
        if len(sn)!=len(set(sn)) or len(dn)!=len(set(dn)):raise ValueError('物种索引重复')
        for n in fields:
            if n not in dst.variables or n not in dn or n not in sn:raise ValueError('原工程未包含该物种，不自动扩机制：'+n)
            a,b=src[n],dst[n]
            if a.dimensions!=b.dimensions or a.shape!=b.shape or a.dtype!=b.dtype or a.units.strip()!=b.units.strip():raise ValueError('排放结构、精度或单位不同：'+n)
            values=np.ma.asarray(a[:])
            if np.ma.getmaskarray(values).any() or not np.isfinite(values).all() or (values<0).any():raise ValueError('排放存在缺失、负值或非有限值：'+n)
            np.testing.assert_array_equal(src['TFLAG'][:,sn.index(n)],dst['TFLAG'][:,dn.index(n)],err_msg='物种逐时标不匹配：'+n)
            if 'ETFLAG' in src.variables and 'ETFLAG' in dst.variables:np.testing.assert_array_equal(src['ETFLAG'][:,sn.index(n)],dst['ETFLAG'][:,dn.index(n)])
        preserved=[n for n in dst.variables if n not in fields]
    if destination is not None:
        shutil.copyfile(reference,destination)
        with Dataset('current',memory=raw) as src,Dataset(destination,'r+') as dst:
            for n in fields:dst[n][:]=src[n][:]
        with Dataset('reference',memory=reference.read_bytes()) as before,Dataset(destination) as after,Dataset('current',memory=raw) as src:
            for n in preserved:np.testing.assert_array_equal(before[n][:],after[n][:],err_msg='未替换变量被更改：'+n)
            for n in fields:np.testing.assert_array_equal(src[n][:],after[n][:])
    return dict(replaced_species=fields,preserved_variables=preserved,reference_sha256=runner.sha(reference),input_sha256=__import__('hashlib').sha256(raw).hexdigest(),no_species_filled=True,no_interpolation=True,
                scope='Ground hourly same-grid, same-mechanism explicit species update; all complement variables retained from actual engineering input, not fixtures; source coordinate helper variables not copied')


def render(converted_zip=None,mode=None):
    import streamlit as st
    from ecodata.interface import camx_runner as runner
    st.header('CAMx：已有完整工程接入')
    st.warning('固定CAMx7.40引擎，非重启、1–60分钟、至多两层网格。接入工程ZIP或本机原控制和配套文件；不执行用户脚本，不改模拟设置。陌生版本、机制及可选模块须另行验收。')
    st.write('原控制路径 → 依赖检查 → 文件原字节复制与路径映射 → 可选输入更新 → 原生积分 → 读取核验 → 出图与下载')
    st.caption('网格排放可更新包中明确提供的物种，保留原工程其余物种；要求同机制、同网格、逐时标与单位一致。初边界目前只接受结构完整的数据流。')
    package=st.radio('CAMx工程提交方式',['完整工程ZIP','本机控制文件'],key='camx_external_format')
    path='';uploaded=None;zip_report=None
    if package=='完整工程ZIP':
        st.caption('ZIP中含唯一camx_project.json，声明原控制文件及每个活动输入的包内路径。平台只重新定位文件引用，原控制与原始数据均保留。最多2GiB，解压最多8GiB。')
        uploaded=st.file_uploader('CAMx完整工程ZIP',type=['zip'],key='camx_external_zip',max_upload_size=2048)
        if uploaded:
            try:
                from ecodata.interface.camx_project_zip import unpack
                path,zip_report=unpack(uploaded.getvalue(),Path(__file__).resolve().parents[3]/'outputs/camx_uploaded_projects')
            except Exception as exc:
                if hasattr(exc,'completeness'):
                    from ecodata.interface.engineering_completeness import render as render_completeness
                    render_completeness(exc.completeness,'camx')
                st.error(str(exc))
    else:
        path=st.text_input('本机完整工程原CAMx控制文件绝对路径',key='camx_external_control')
        if st.button('将此本机工程打包为完整ZIP',disabled=not path,key='camx_external_pack'):
            try:
                from ecodata.interface.camx_project_zip import pack
                p=Path(__file__).resolve().parents[3]/'outputs/camx_project_exports'/(__import__('hashlib').sha256(path.encode()).hexdigest()+'.zip')
                st.session_state['camx_exported_project']=str(pack(path,p))
            except Exception as exc:st.error(str(exc))
        if st.session_state.get('camx_exported_project'):
            st.download_button('下载CAMx完整工程ZIP',Path(st.session_state['camx_exported_project']).read_bytes(),'CAMx_existing_project.zip',key='camx_external_pack_download')
    from ecodata.interface.engineering_conversion import render as render_conversion
    converted_zip,mode,conversion_ready=render_conversion('camx',converted_zip,mode)
    report=None;target=None;member_name=None
    if st.button('检查该CAMx工程原资料',disabled=not path,key='camx_external_inspect'):
        try:st.session_state['camx_external_report']=inspect(path)
        except Exception as exc:
            st.session_state.pop('camx_external_report',None)
            if hasattr(exc,'completeness'):
                from ecodata.interface.engineering_completeness import render as render_completeness
                render_completeness(exc.completeness,'camx')
            st.error(str(exc))
    report=zip_report or st.session_state.get('camx_external_report')
    if report and (not path or report['control']!=str(Path(path))):report=None
    ready=report is not None
    if report:
        from ecodata.interface.engineering_completeness import render as render_completeness
        ready=render_completeness(report['completeness'],'camx')
        with st.expander('实际依赖、原设置及重复输出索引'):st.json(report)
        st.download_button('下载CAMx工程依赖检查报告',json.dumps(report,ensure_ascii=False,indent=2).encode('utf8'),'CAMx_import_report.json',key='camx_external_report_download')
        if converted_zip is not None:
            kinds={'gridded_emissions':('emissions.CAMx7.40.nc',lambda k:k.startswith('Emiss_Grid(')),
                   'initial_concentrations':('initial.CAMx7.40.nc',lambda k:k=='Initial_Conditions'),
                   'boundary_concentrations':('boundary.CAMx7.40.nc',lambda k:k=='Boundary_Conditions')}
            if mode not in kinds:st.error('此入口仅支持完整排放或初边界流转换；不回退原数据');ready=False
            else:
                member_name,choose=kinds[mode]
                target=st.selectbox('明确替换原工程的哪一个完整输入流',[i['key'] for i in report['inputs'] if choose(i['key'])],index=None,key='camx_external_target');ready=target is not None
    from ecodata.interface.input_result_binding import bind_result
    bind_result(st.session_state,'camx_external_job','camx_external_input_binding_v2',path,uploaded.getvalue() if uploaded else b'',converted_zip or b'',target,conversion_ready)
    ready=ready and conversion_ready
    if st.button('在副本运行本次CAMx工程',disabled=not ready,key='camx_external_start'):
        st.session_state.pop('camx_external_job',None)
        try:st.session_state['camx_external_job']=start(path,converted_zip,target,member_name)
        except Exception as exc:st.error(str(exc))
    ident=st.session_state.get('camx_external_job')
    if ident and ready:
        @st.fragment(run_every=5)
        def view():
            state=runner.status(ident);folder=runner.directory(ident);st.write('本次CAMx工程任务：'+ident)
            if state['status']=='running':st.info('正在复制原输入并执行固定原生引擎。')
            elif state['status']=='failed':st.error(state.get('error'))
            else:
                st.success('本次CAMx实际积分、读取核验及出图完成；未评价预测精度。');st.image(str(folder/'result.png'))
                st.download_button('下载本次CAMx结果PNG',(folder/'result.png').read_bytes(),'CAMx_result.png',key='camx_external_png')
                for row in state['grids']:
                    n=row['grid'];st.download_button('下载本次CAMx网格'+str(n)+' CSV',(folder/f'O3_grid{n}.csv').read_bytes(),f'CAMx_grid{n}.csv',key='camx_external_csv_'+str(n))
            with st.expander('本次记录与日志位置'):st.json(state);st.code(str(folder))
        view()
    return True
