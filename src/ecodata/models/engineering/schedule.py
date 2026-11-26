"""Connect supported control templates to the existing native input executor.

Scientific declarations come from the format parser, never case names. Runtime
outputs are dependency barriers, not fabricated inputs or model execution.
"""
from graphlib import TopologicalSorter
import hashlib,json,re


def dependency_order(runs,bindings):
    starts={tuple(r['start']):i for i,r in enumerate(runs)}
    if len(starts)!=len(runs):raise ValueError('运行起点重复，不能自动选配置')
    dependencies={i:set() for i in range(len(runs))}
    for b in bindings:
        if b['status']!='GENERATED_DEPENDENCY':continue
        i=b['run_index'];producer=starts.get(tuple(b['producer_required']['run_start']))
        if producer is None or producer>=i:raise ValueError('重启依赖没有明确的前序生产者')
        dependencies[i].add(producer)
    return list(TopologicalSorter(dependencies).static_order()),dependencies


def bind_templates(files):
    from ..camx_templates import inspect_template
    choices=[];identities={}
    for name in sorted(files.files):
        if not name.lower().endswith(('.job','.csh','.sh')):continue
        text=files.prefix(name,1024*1024).decode('utf-8-sig',errors='replace')
        if not re.search(r'(?im)^\s*&CAMx_Control\b',text):continue
        result=inspect_template(files,name,text,identities)
        if result is not None:choices.append(result)
    if not choices:return None
    signatures={json.dumps([r['parameters'] for r in c['runs']],sort_keys=True,default=str) for c in choices}
    if len(signatures)!=1:raise ValueError('多个科学配置不同的工程模板，需明确配置，不能猜选')
    return _binding(files,choices[0],identities,[c['control'] for c in choices])


def bind_plain_schedule(files,controls):
    """Actual same-directory sequential controls, not a choice of projects."""
    from pathlib import PurePosixPath
    import f90nml,io
    from ..camx_templates import bind_rendered_controls
    if len({str(PurePosixPath(n).parent) for n in controls})!=1:
        raise ValueError('多个工程控制目录彼此独立，不能猜选运行计划')
    rendered=[]
    for name in controls:
        with files.files[name][0]() as stream:text=stream.read().decode('utf-8-sig')
        group=f90nml.read(io.StringIO(text))['camx_control']
        start=group.get('start_date_hour')
        if not isinstance(start,list) or len(start)!=4:raise ValueError('多控制运行日期不明确')
        rendered.append((tuple(start),name,text))
    rendered.sort()
    identities={}
    selected=bind_rendered_controls(files,[(n,t) for _,n,t in rendered],identities)
    binding=_binding(files,selected,identities,[files.identity(n) for _,n,_ in rendered])
    if any(not binding['runtime_dependencies'][str(i)] for i in binding['execution_order'][1:]):
        raise ValueError('多个工程控制没有明确的运行依赖，不能自动合成运行计划')
    return binding


def _binding(files,selected,identities,controls):
    if selected['unknown'] or any(b['status'] not in {'BOUND','GENERATED_DEPENDENCY'} for b in selected['bindings']):
        raise ValueError('模板仍有缺失/未知依赖：'+str(selected['unknown']))
    order,deps=dependency_order(selected['runs'],selected['bindings'])
    targets=sorted({b['target'] for b in selected['bindings'] if b['status']=='BOUND'})
    return dict(schema='ecodata.project-binding.v1',model_id='camx',status='BOUND',
        control=selected['control'],equivalent_controls=controls,
        bindings=selected['bindings'],resources=[identities[n] for n in targets],runs=selected['runs'],
        execution_order=order,runtime_dependencies={str(k):sorted(v) for k,v in deps.items()},
        inactive=[],parameters=selected['runs'][0]['parameters'],source_modified=False,
        native_model_executed=False,scope='多时段输入准备；生成依赖仍需模型运行，不宣称重启产物已存在')


def materialize_controls(binding,mapping,folder):
    """Reuse f90nml; only relocate paths, preserving scientific settings."""
    from pathlib import Path
    import f90nml,copy
    folder=Path(folder);output_roots={i:f'outputs/run_{i}/CAMx' for i in range(len(binding['runs']))}
    starts={tuple(r['start']):i for i,r in enumerate(binding['runs'])}
    receipts=[]
    for i in binding['execution_order']:
        params=copy.deepcopy(binding['runs'][i]['parameters'])
        for b in binding['bindings']:
            if b['run_index']!=i:continue
            if b['status']=='BOUND':value=mapping[b['target']]
            else:
                producer=starts[tuple(b['producer_required']['run_start'])]
                suffix=b['original'][len(b['producer_required']['output_root']):]
                value=output_roots[producer]+suffix
            if not b['indexes']:params[b['parameter']]=value
            else:
                field=params[b['parameter']]
                for index in b['indexes'][:-1]:field=field[index]
                field[b['indexes'][-1]]=value
        params['root_output_name']=output_roots[i]
        (folder/output_roots[i]).parent.mkdir(parents=True,exist_ok=True)
        p=folder/f'CAMx_run_{i}.in';nml=f90nml.Namelist(dict(camx_control=params));nml.write(p)
        if f90nml.read(p)['camx_control'].todict()!=nml['camx_control'].todict():raise ValueError('派生控制回读不一致')
        receipts.append(dict(run=i,control=p.name,start=binding['runs'][i]['start'],
            depends_on=binding['runtime_dependencies'][str(i)],model_executed=False,
            output_is_not_present=True,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return receipts
