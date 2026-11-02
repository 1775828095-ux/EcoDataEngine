"""Bounded five-minute CAMx case: prepared inputs, no raw inventory creation."""
from pathlib import Path
import hashlib,json,os,re,subprocess,threading,uuid
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from ecodata.models.adapters.camx_photolysis import export_photolysis

BASE=Path('D:/EcoDataEngine')
CASE=BASE/'cases/CAMx740_20261020/official'
RUNS=BASE/'outputs/camx_jobs'
EXE=BASE/'work/camx740/src.v7.40/CAMx.v7.40.noMPI.NCF4.gfortran'
PIN='6a37a63b5f2b65ca520a83e40b29861ef66407812fd3c6619231cfd586a67d89'
LOCK=threading.Lock()

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def linux(p):
    p=Path(p).resolve();return '/mnt/'+p.drive[0].lower()+'/'+p.as_posix()[3:]
def directory(ident):
    if not re.fullmatch('[0-9a-f]{32}',ident):raise ValueError('无效任务编号')
    return RUNS/ident
def status(ident):return json.loads((directory(ident)/'result.json').read_text(encoding='utf-8'))
def save(folder,result):
    temp=folder/'result.tmp';temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(folder/'result.json')

def decode_table(path):
    rows=[];condition=None;block=-1
    for line in path.read_text(encoding='ascii').splitlines()[1:]:
        if 'O3 Column' in line:
            match=re.fullmatch(r'\s*O3 Column\s*=\s*([\d.Ee+-]+)\s+Albedo\s*=\s*([\d.Ee+-]+)\s+Terrain\s*=\s*([\d.Ee+-]+)\s*',line)
            if not match:raise ValueError('光解表条件头无效')
            condition=tuple(map(float,match.groups()))
        elif 'km above surface' in line:height=float(line.split()[0]);block+=1
        elif 'CBX' in line:
            values,label=line.split('CBX',1);reaction=label.split(':',1)[1].strip()
            for i in range(10):rows.append((block,*condition,height,reaction,i,float(values[1+i*12:1+(i+1)*12])))
    return pd.DataFrame(rows,columns=['block_id','ozone_column_native','albedo','terrain_native','height_km','reaction','zenith_index','photolysis_native_value'])

def assemble(folder,route,converted_zip=None,mode=None):
    template=CASE/'native_platform_photolysis_5min_20261021/CAMx.in'
    text=template.read_text(encoding='ascii')
    fixture=BASE/'work/SWATplus_UI_20261020/platform/examples/ai_round1/camx'
    cfg=json.loads((fixture/'config_photolysis.json').read_text())
    frame=pd.read_csv(fixture/'A_photolysis_table.csv') if route=='existing' else decode_table(CASE/'inputs/tuv.do_CB6r5.20160610')
    outputs,report=export_photolysis(frame,cfg)
    generated=next(data for name,data in outputs.items() if name.endswith('.txt') and name.startswith('photolysis.'))
    (folder/'photolysis.txt').write_bytes(generated)
    reference=decode_table(CASE/'inputs/tuv.do_CB6r5.20160610')
    check=decode_table(folder/'photolysis.txt')
    pd.testing.assert_frame_equal(reference,check,check_exact=True)
    current_input = None
    current_report = {}
    if converted_zip is not None:
        from ecodata.interface.registered_conversion import member, accept_netcdf
        if mode is None:
            generated = member(converted_zip, 'photolysis.CAMx7.40.CB6r5.txt')
            (folder/'photolysis.txt').write_bytes(generated)
            pd.testing.assert_frame_equal(reference,decode_table(folder/'photolysis.txt'),check_exact=True)
            current_report = dict(input_sha256=sha(folder/'photolysis.txt'),scope='Current converted photolysis, exact registered benchmark')
        else:
            registered = {
                'gridded_emissions': ('emissions.CAMx7.40.nc',CASE/'native_complx_emission_probe/B/emissions.CAMx7.40.nc','Emiss_Grid(1,1)'),
                'initial_concentrations': ('initial.CAMx7.40.nc',CASE/'initial_3d5626e7fd8c/initial.nc','Initial_Conditions'),
                'boundary_concentrations': ('boundary.CAMx7.40.nc',CASE/'boundary_1e386ac9018b/boundary.nc','Boundary_Conditions'),
            }
            if mode not in registered:raise ValueError('此转换类别未登记原生接入规则')
            filename, reference_file, control_key = registered[mode]
            raw,current_report=accept_netcdf(converted_zip,filename,reference_file)
            current_input=folder/filename;current_input.write_bytes(raw)
            if mode=='gridded_emissions':
                import shutil
                original=CASE/'emiss/camx_area.area.POA_OP.20160610.36km.nc'
                merged=folder/'complete_emissions.nc';shutil.copyfile(original,merged)
                with Dataset(merged,'r+') as dst,Dataset('current',memory=raw) as src:
                    fields=[n for n,v in src.variables.items() if v.dimensions==('TSTEP','LAY','ROW','COL')]
                    if set(fields)!={'NO','PEC'}:raise ValueError('登记排放仅允许NO/PEC；其余物种保留原输入')
                    for name in fields:dst.variables[name][:]=src.variables[name][:]
                current_input=merged
                current_report.update(converted_species=fields,preserved_species=52,complement_source=str(original))
    # Preserve the supplied control's ordered assignments, including its repeated
    # Output_Species_Names(22). Do not silently change the case chemistry.
    assignments=[];input_files={}
    for line in text.splitlines():
        match=re.match(r'\s*([A-Za-z][A-Za-z_0-9]*(?:\([^)]*\))?)\s*=\s*(.*?)\s*,\s*(?:!.*)?$',line)
        if not match:continue
        key,value=match.groups()
        if key=='Root_Output_Name':value="'"+linux(folder/'outputs/CAMx.20160610')+"'"
        elif key=='Photolysis_Rates':value="'"+linux(folder/'photolysis.txt')+"'"
        elif current_input is not None and key==control_key:value="'"+linux(current_input)+"'"
        elif value.startswith("'../"):
            source=(template.parent/value.strip("'")).resolve()
            if source.is_file():input_files[str(source)]=dict(sha256=sha(source),bytes=source.stat().st_size)
            elif key not in ('Master_Grid_Restart','Nested_Grid_Restart'):raise ValueError('配套资料缺失：'+str(source))
            value="'"+linux(source)+"'"
        assignments.append([key,value])
    if len(assignments)<70:raise ValueError('登记控制参数不完整')
    (folder/'engineering_manifest.json').write_text(json.dumps(dict(assignments=assignments,prepared_inputs=input_files),indent=2),encoding='utf-8')
    if route=='rebuild':
        text=' &CAMx_Control\n'+''.join(f' {key} = {value},\n' for key,value in assignments)+' /\n'
    else:
        ordered=iter(assignments);lines=[]
        for line in text.splitlines():
            if re.match(r'\s*([A-Za-z][A-Za-z_0-9]*(?:\([^)]*\))?)\s*=',line):
                key,value=next(ordered);line=f' {key} = {value},'
            lines.append(line)
        text='\n'.join(lines)+'\n'
    (folder/'CAMx.in').write_text(text,encoding='ascii');(folder/'outputs').mkdir()
    return dict(conversion=report,current_conversion=current_report,conversion_mode=mode,prepared_inputs=input_files,photolysis_rows=len(frame),photolysis_sha256=sha(folder/'photolysis.txt'),control_sha256=sha(folder/'CAMx.in'))

def plot_outputs(folder,grid_count=2):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,grid_count,figsize=(6*grid_count,5),layout='constrained');summaries=[]
    axes=np.atleast_1d(axes)
    for grid,ax in enumerate(axes,1):
        path=folder/f'outputs/CAMx.20160610.avrg.grd{grid:02}.nc'
        with Dataset('output',memory=path.read_bytes()) as data:
            if int(data.TSTEP)!=500 or data['O3'].shape[0]!=1:raise ValueError('输出时段不匹配')
            if not np.all(data['TFLAG'][:]==np.array([2016162,0])):raise ValueError('输出时间标签不匹配')
            finite_fields=[]
            for name,field in data.variables.items():
                if field.dtype.kind=='f':
                    supplied=np.ma.asarray(field[:])
                    if np.ma.getmaskarray(supplied).any():raise ValueError('CAMx输出变量存在缺测：'+name)
                    valid=supplied.compressed()
                    if not valid.size or not np.isfinite(valid).all():raise ValueError('CAMx输出变量无有效值或存在非有限值：'+name)
                    finite_fields.append(name)
            v=data['O3'];units=str(v.units).strip()
            if units!='ppmv':raise ValueError('O3原生单位不符')
            values=v[0,0].filled(np.nan) if np.ma.isMaskedArray(v[0,0]) else v[0,0]
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('浓度场无效')
            values=values.astype('float64')*1000
            ny,nx=values.shape;x=(float(data.XORIG)+(np.arange(nx)+.5)*float(data.XCELL))/1000;y=(float(data.YORIG)+(np.arange(ny)+.5)*float(data.YCELL))/1000
            mesh=ax.pcolormesh(x,y,values,shading='nearest',vmin=0,vmax=100,cmap='viridis');ax.set_aspect('equal')
            ax.set(title=f'Grid {grid}: O3, 00:00-00:05 UTC mean',xlabel='Lambert x (km)',ylabel='Lambert y (km)')
            xx,yy=np.meshgrid(x,y);pd.DataFrame(dict(x_km=xx.ravel(),y_km=yy.ravel(),O3_ppbv=values.ravel())).to_csv(folder/f'O3_grid{grid}.csv',index=False)
            summaries.append(dict(grid=grid,shape=[ny,nx],max_O3_ppbv=float(values.max()),sha256=sha(path),finite_output_fields=finite_fields))
    fig.colorbar(mesh,ax=list(axes),label='O3 (ppbv)');fig.suptitle('CAMx 7.40 | 2016-06-10 | five-minute integration'+(' | NEW single-grid meteorology' if grid_count==1 else ' | registered two-grid case'))
    fig.savefig(folder/'O3_two_grids.png',dpi=140);plt.close(fig);return summaries

def execute(ident,route,converted_zip=None,mode=None,new_meteorology=False,meteorology_project=None,raw_gfs_files=None):
    folder=directory(ident);result=status(ident)
    try:
        if sha(EXE)!=PIN:raise ValueError('CAMx可信程序哈希不匹配')
        result.update(assemble(folder,route,converted_zip,mode));save(folder,result)
        single_grid=bool(new_meteorology or meteorology_project)
        if single_grid:
            from ecodata.interface.camx_meteorology import build,apply
            if new_meteorology:
                met,built=build(folder,raw_gfs_files) if raw_gfs_files is not None else build(folder);result['raw_meteorology']=built
            else:
                project=status(meteorology_project);met=directory(meteorology_project)/'new_camx_meteorology'
                if project['status']!='success' or not project.get('new_meteorology_requested'):raise ValueError('只能接入已成功的本机原始气象新建工程')
                pins=project.get('new_meteorology',{}).get('files',[])
                if not pins:raise ValueError('已有新气象工程缺文件哈希记录')
                for row in pins:
                    p=Path(row['path'])
                    if p.parent.resolve()!=met.resolve() or sha(p)!=row['sha256']:raise ValueError('已有新气象工程文件路径或哈希不匹配')
                result['reused_meteorology_project']=meteorology_project
            result['new_meteorology']=apply(folder,met)
            result['scope']='Original2016 GFS -> WPS/WRF12h/WRFCAMx -> NEW36km single-grid CAMx5min; original chemical/emission/snow-age companions reused; not full raw inventory or original benchmark reproduction'
            save(folder,result)
        worker='''import os,sys,subprocess,resource,json,time
from pathlib import Path
folder=Path(sys.argv[1]);resource.setrlimit(resource.RLIMIT_STACK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
env=dict(os.environ,OMP_NUM_THREADS='1',OMP_STACKSIZE='128M')
start=time.time()
with (folder/'console.log').open('wb') as log:
 p=subprocess.run([sys.argv[2]],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
(folder/'native.json').write_text(json.dumps(dict(returncode=p.returncode,seconds=time.time()-start)))
sys.exit(p.returncode)
'''
        (folder/'worker.py').write_text(worker,encoding='ascii')
        proc=subprocess.run(['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','python3',linux(folder/'worker.py'),linux(folder),linux(EXE)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=660)
        (folder/'launcher.log').write_bytes(proc.stdout)
        console=(folder/'console.log').read_text(errors='replace')
        if proc.returncode or 'END SIMULATION' not in console or 'Date/time: 16/06/10 00:05:00' not in console or 'Diagnostic Initialization Test Complete' in console:raise ValueError('原生CAMx未完成5分钟积分')
        if single_grid:
            native_log=(folder/'outputs/CAMx.20160610.out').read_text(errors='replace')
            for p in [folder/'initial_new_vertical.nc',folder/'boundary_new_vertical.nc',met/'met3d.nc',met/'met2d.nc',met/'kv.nc.MYJ']:
                if linux(p) not in native_log:raise ValueError('原生读取日志未确认本次新输入：'+str(p))
            result['native_read_log']=str(folder/'outputs/CAMx.20160610.out')
        result.update(status='success',grids=plot_outputs(folder,1 if single_grid else 2),native=json.loads((folder/'native.json').read_text()),engine_sha256=PIN)
    except Exception as exc:result.update(status='failed',error=str(exc))
    finally:save(folder,result);LOCK.release()

def start(route,converted_zip=None,mode=None,new_meteorology=False,meteorology_project=None,raw_gfs_files=None):
    if route not in ('existing','rebuild'):raise ValueError('不支持的路线')
    if new_meteorology and route!='rebuild':raise ValueError('原始GFS新建气象只允许新建路线')
    if raw_gfs_files is not None:
        if not new_meteorology or route!='rebuild':raise ValueError('上传原始GFS仅适用于新建气象路线')
        from ecodata.interface.camx_meteorology import validate_gfs_uploads
        upload_report=validate_gfs_uploads(raw_gfs_files)
    if meteorology_project:
        if route!='existing' or new_meteorology:raise ValueError('已有新气象工程仅允许已有路线且不能同时重建')
        directory(meteorology_project)
    if not LOCK.acquire(blocking=False):raise ValueError('已有CAMx任务运行中，请等待完成')
    ident=uuid.uuid4().hex
    try:
        folder=directory(ident);folder.mkdir(parents=True,exist_ok=False)
        save(folder,dict(id=ident,route=route,status='running',new_meteorology_requested=new_meteorology,meteorology_project=meteorology_project,scope='Original GFS meteorology requested/reused; chemical companions reused; not full raw inventory construction' if new_meteorology or meteorology_project else 'Prepared official case, not raw emissions/meteorology reconstruction; five minutes only'))
        if raw_gfs_files is not None:
            record=status(ident);record['uploaded_gfs']=upload_report;save(folder,record)
        threading.Thread(target=execute,args=(ident,route,converted_zip,mode,new_meteorology,meteorology_project,raw_gfs_files),daemon=True).start();return ident
    except Exception:LOCK.release();raise
