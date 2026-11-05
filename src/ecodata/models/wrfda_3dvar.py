"""Bind real WRFDA inputs, execute the official solver and validate its analysis."""
from pathlib import Path,PurePosixPath
import hashlib,json,re
import f90nml
import numpy as np
import pandas as pd
import xarray as xr
from .wrf_meteorology import wrf_times


RADIANCE_FLAGS={'use_amsuaobs','use_amsubobs','use_msuobs','use_hirs2obs','use_hirs3obs','use_hirs4obs',
 'use_mhsobs','use_airsobs','use_ssmisobs','use_amsreobs','use_iasiobs','use_atmsobs','use_crisobs',
 'use_ssmiobs','use_seviriobs','use_ahiobs','use_gmiobs','use_mwriobs','use_mwhsobs','use_mwhs2obs','use_mwtsobs','use_mwts2obs'}


def _date(value):return pd.Timestamp(str(value).replace('_','T')).tz_localize(None)


def plan(files,controls,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    candidates=[c for c in controls if not c['var4d']]
    candidates={c['sha256']:c for c in candidates}
    if len(candidates)!=1:raise ValueError('需要唯一、明确的3DVAR配置；不自动改用4DVAR或挑选冲突配置')
    control=next(iter(candidates.values()))
    with files.files[control['resource']][0]() as s:raw=s.read()
    nml=f90nml.reads(raw.decode('utf-8-sig'));date=_date(nml['wrfvar18']['analysis_date'])
    lo=_date(nml['wrfvar21']['time_window_min']);hi=_date(nml['wrfvar22']['time_window_max'])
    if not lo<=date<=hi:raise ValueError('WRFDA分析时刻不在声明的观测窗口内')
    declarations=[]
    for name in files.files:
        if not name.endswith('.json'):continue
        with files.files[name][0]() as s:
            try:value=json.load(s)
            except (ValueError,UnicodeError):continue
        if isinstance(value,dict) and value.get('schema')=='ecodata.wrfda-task.v1':declarations.append(value)
    if len(declarations)>1:raise ValueError('存在多个WRFDA任务范围声明')
    requested=declarations[0] if declarations else {}
    if requested and (requested.get('mode')!='3dvar' or requested.get('observation_scope')!='conventional' or requested.get('confirmed') is not True):
        raise ValueError('WRFDA任务范围需明确确认为常规观测3DVAR')
    active=[k for k in RADIANCE_FLAGS if nml.get('wrfvar4',{}).get(k,False)]
    changes=[]
    if active and not requested:raise ValueError('配置启用了卫星辐射同化；当前常规3DVAR后端不会擅自关闭其科学选项')
    if requested:
        for k in active:
            changes.append(dict(group='wrfvar4',field=k,original=True,executed=False,basis='explicitly confirmed conventional-only task'))
            nml['wrfvar4'][k]=False
        if active:
            for k,v in [('rtminit_nsensor',0),('use_varbc',False)]:
                changes.append(dict(group='wrfvar14',field=k,original=nml.get('wrfvar14',{}).get(k),executed=v,basis='inactive radiance subsystem for confirmed conventional-only scope'))
                nml.setdefault('wrfvar14',{})[k]=v
    backgrounds=[];observations=[];errors=[]
    for name in sorted(files.files):
        base=PurePosixPath(name).name
        if base=='be.dat':errors.append(name)
        if base.startswith('wrfinput'):
            with files.files[name][0]() as s,xr.open_dataset(s,decode_times=False) as ds:
                if not {'Times','T','U','V','QVAPOR','P','PB','PH','PHB','XLAT','XLONG'}<=set(ds.variables):continue
                if list(wrf_times(ds.Times.values))==[date]:
                    dimensions={k:int(v) for k,v in ds.sizes.items()}
                    backgrounds.append((name,dimensions,str(ds.attrs.get('TITLE','')),ds.attrs.get('NUM_LAND_CAT')))
        if base.lower().endswith('.3dvar') or base=='ob.ascii':
            with files.files[name][0]() as s:payload=s.read()
            if not payload.startswith(b'TOTAL =') or b'INFO_FMT' not in payload[:5000] or b'EACH_FMT' not in payload[:5000]:continue
            rows=payload.decode('ascii',errors='replace').splitlines()
            # OBSPROC INFO_FMT has a 12-character platform field, including
            # multiword types such as "FM-36 TEMP S"; date is A19 after 1X.
            dates=[_date(line[13:32]) for line in rows if line.startswith('FM-')]
            if dates and min(dates)>=lo and max(dates)<=hi:
                declared=int(re.search(rb'TOTAL\s*=\s*(\d+)',payload[:100]).group(1))
                if declared!=len(dates):raise ValueError('OBSPROC记录数与文件头声明不符：'+name)
                observations.append(dict(resource=name,records=declared,sha256=hashlib.sha256(payload).hexdigest(),preferred=base.lower().endswith('.3dvar')))
    def unique(names):
        by_hash={files.identity(n)['sha256']:n for n in names}
        if len(by_hash)!=1:raise ValueError('WRFDA输入缺失或不唯一，未按文件名猜选：'+str(names))
        return next(iter(by_hash.values()))
    foreground=unique([n for n,_,_,_ in backgrounds]);error=unique(errors)
    chosen=[r for r in observations if r['preferred']] or observations
    by_hash={r['sha256']:r for r in chosen}
    if len(by_hash)!=1:raise ValueError('需要唯一匹配分析窗口的已处理3DVAR观测；未把4DVAR分时观测或原LITTLE_R混入')
    observation=next(iter(by_hash.values()))
    dimensions=next(d for n,d,_,_ in backgrounds if n==foreground)
    title=next(t for n,_,t,_ in backgrounds if n==foreground)
    land_categories=next(v for n,_,_,v in backgrounds if n==foreground)
    if land_categories is not None:
        stated=nml.get('physics',{}).get('num_land_cat')
        if stated is not None and int(stated)!=int(land_categories):raise ValueError('配置土地类别数与背景原生声明冲突，未重标类别')
        if stated is None:
            nml.setdefault('physics',{})['num_land_cat']=int(land_categories)
            changes.append(dict(group='physics',field='num_land_cat',original=None,executed=int(land_categories),basis='background NUM_LAND_CAT; prevent backend-version default from relabeling native categories'))
    if re.search(r'\bV[123]\.',title) and not nml.get('time_control',{}).get('force_use_old_data',False):
        nml.setdefault('time_control',{})['force_use_old_data']=True
        changes.append(dict(group='time_control',field='force_use_old_data',original=None,executed=True,
            basis='official WRF v4 input-reader compatibility for verified pre-v4 TITLE; original arrays untouched'))
    for parameter,dimension in [('e_we','west_east_stag'),('e_sn','south_north_stag'),('e_vert','bottom_top_stag')]:
        declared=nml.get('domains',{}).get(parameter)
        if isinstance(declared,list):declared=declared[0]
        if declared is not None and dimension in dimensions and int(declared)!=dimensions[dimension]:raise ValueError('WRFDA配置网格与背景不一致：'+parameter)
    request={};bindings=[]
    for port,name in [('fg',foreground),('be.dat',error),('ob.ascii',observation['resource'])]:
        target=folder/port
        with files.files[name][0]() as s,target.open('wb') as out:__import__('shutil').copyfileobj(s,out)
        request[port]=target;bindings.append(dict(port=port,resource=name,**files.identity(name)))
    target=folder/'namelist.input';target.write_bytes((str(nml)+'\n').encode() if changes else raw);request['namelist.input']=target
    report=dict(schema='ecodata.wrfda-bindings.v1',analysis_date=str(date),time_window=[str(lo),str(hi)],
        control=control,bindings=bindings,observation_records_declared=observation['records'],configuration_changes=changes,
        source_modified=False,profile='explicitly confirmed conventional-only 3DVAR' if requested else 'author-declared 3DVAR with recorded native I/O compatibility',
        science_note='header report count is not the post-QC assimilation count; no independent truth or accuracy gain is claimed')
    (folder/'bindings.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return request,report


def validate(request,folder,execution):
    folder=Path(folder);output=folder/'wrfvar_output'
    if execution['returncode']!=0 or execution['timed_out'] or not output.is_file():raise ValueError('官方WRFDA未成功生成分析场，诊断目录：'+str(folder))
    log=(folder/'wrfda.log').read_text(errors='replace')
    # The official serial solver emits the same completion marker as its MPI build.
    if not re.search(r'(?:WRF-Var|WRFDA).*completed successfully',log,re.I):raise ValueError('缺少原生同化完成标记，未只凭输出文件存在宣布成功')
    changes={}
    with xr.open_dataset(request['fg'],decode_times=False) as bg,xr.open_dataset(output,decode_times=False) as analysis:
        if list(wrf_times(bg.Times.values))!=list(wrf_times(analysis.Times.values)) or dict(bg.sizes)!=dict(analysis.sizes):raise ValueError('分析场时刻或网格与背景不一致')
        for field in ['T','U','V','QVAPOR','P','PB','PH','PHB']:
            a=np.asarray(analysis[field].values)
            if not np.isfinite(a).all():raise ValueError('分析场含非有限值：'+field)
            delta=a-np.asarray(bg[field].values)
            changes[field]=dict(changed_cells=int(np.count_nonzero(delta)),max_abs_increment=float(np.max(np.abs(delta))),rms_increment=float(np.sqrt(np.mean(delta.astype('f8')**2))))
        if not np.all(analysis.P.values+analysis.PB.values>0) or not np.all(analysis.T.values+300>0):raise ValueError('分析场压力或位温违反基本物理边界')
        if np.min(analysis.QVAPOR.values)<-1e-8:raise ValueError('分析场水汽混合比为负')
        for coordinate in ['XLAT','XLONG']:
            if not np.array_equal(bg[coordinate].values,analysis[coordinate].values):raise ValueError('同化意外改变背景水平网格')
    if not any(changes[k]['changed_cells'] for k in ['T','U','V','QVAPOR']):raise ValueError('未检出状态增量，不能作为观测已产生同化作用的验收证据')
    cost=[]
    for line in (folder/'cost_fn').read_text().splitlines():
        row=line.split()
        if len(row)>5 and row[0].isdigit():cost.append([float(x) for x in row[:6]])
    if len(cost)<2 or not np.isfinite(cost).all() or cost[0][5]<=0 or cost[-1][3]>cost[0][3]:
        raise ValueError('原生代价记录缺失、观测项无贡献或求解未降低目标，保留诊断但不验收成功')
    report=dict(schema='ecodata.wrfda-analysis-check.v1',status='PASS',native_completion_marker=True,
        variables=changes,objective=dict(initial_J=cost[0][3],final_J=cost[-1][3],initial_Jo=cost[0][5],final_Jo=cost[-1][5],inner_iterations=int(cost[-1][2])),
        wrf_forecast_executed=False,accuracy_improvement='NOT_ESTABLISHED',
        note='nonzero increments demonstrate a numerical analysis, not independent accuracy improvement')
    (folder/'analysis_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
