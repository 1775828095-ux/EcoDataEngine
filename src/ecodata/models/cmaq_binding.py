"""CMAQ I/O API input ports from the shared non-executing control analysis."""
import hashlib,json,re
from .control_flow import ShellDeclarations

PORTS=re.compile(r'^(?:MET_(?:CRO|DOT|BDY)_\dD|GRID_(?:CRO|DOT|BDY)_\dD|LUFRAC_CRO|INIT_CONC_\d+|BNDY_CONC_\d+|GR_EMIS_\d+|STK_(?:GRPS|EMIS)_\d+|OCEAN_\d+|BEIS_NORM_EMIS|E2C_(?:SOIL|CHEM|LU))$')
CORE={'MET_CRO_2D','GRID_CRO_2D'}

def bind_cmaq(files):
    candidates=[];rejected=[];identities={}
    for control in sorted(n for n in files.files if n.lower().endswith('.csh')):
        text=files.prefix(control,65536).decode('utf8',errors='replace')
        if not re.search(r'(?m)^\s*setenv\s+MET_CRO_2D\b',text):continue
        try:
            analysis=ShellDeclarations(files).analyze(control)
            rows=[];missing=[]
            for day_index,frame in enumerate(analysis['frames']):
                for key,value in frame['values'].items():
                    if not PORTS.fullmatch(key):continue
                    stamp=frame['values'].get('CTM_STDATE');stamp=stamp.text if stamp and not stamp.unknown else None
                    row=dict(parameter=key,indexes=[day_index],original=value.text,control=control,line=frame['line'],day=stamp)
                    try:
                        reference=value.text
                        if value.unknown:
                            # Unknown machine roots may relocate by an actual,
                            # unique >=2-segment suffix; interior unknowns cannot.
                            if not re.fullmatch(r'\$\{\w+\}/[^$`]+',reference):raise ValueError('输入值仍有内部UNKNOWN：'+str(sorted(value.unknown)))
                            reference='/'+reference
                        target,reason=files.resolve(control,reference)
                        if target not in identities:identities[target]=files.identity(target)
                        rows.append(dict(row,target=target,reason=reason,sha256=identities[target]['sha256'],status='BOUND',
                            unknown_root=sorted(value.unknown),activation='SUPPORTED_DECLARATION_AT_MODEL_CALL'))
                    except ValueError as e:missing.append(dict(row,status='UNKNOWN_OR_MISSING',reason=str(e)))
            if not rows:continue
            bad=[r for r in missing if r['parameter'] in CORE]
            if bad or not CORE<=set(r['parameter'] for r in rows):
                rejected.append(dict(control=control,reason='核心气象/网格声明不能覆盖各运行日',missing=bad));continue
            parameters={k:dict(text=v.text,unknown=sorted(v.unknown)) for k,v in analysis['frames'][0]['values'].items()}
            signature=hashlib.sha256(json.dumps(dict(rows=[(r['parameter'],r['day'],r['sha256']) for r in rows],parameters=parameters),sort_keys=True).encode()).hexdigest()
            candidates.append(dict(control=files.identity(control),bindings=rows,unresolved=missing,
                parameters=parameters,analysis=dict(trace=analysis['trace'],includes=analysis['includes'],steps=analysis['steps'],
                commands_executed=False),signature=signature))
        except ValueError as e:rejected.append(dict(control=control,reason=str(e)))
    if not candidates:
        if rejected:raise ValueError('CMAQ没有可验收的核心输入绑定：'+json.dumps(rejected,ensure_ascii=False))
        return None
    if len({r['signature'] for r in candidates})!=1:raise ValueError('多个不同CMAQ运行配置均可绑定，不能自动猜选：'+str([r['control']['path'] for r in candidates]))
    selected=candidates[0]
    return dict(schema='ecodata.project-binding.v1',model_id='cmaq',status='BOUND',**{k:v for k,v in selected.items() if k!='signature'},
        equivalent_controls=[c['control']['path'] for c in candidates],rejected_controls=rejected,
        resources=[identities[n] for n in sorted({r['target'] for r in selected['bindings']})],
        inactive=[],unbound_files=sorted(set(files.files)-{r['target'] for r in selected['bindings']}),
        source_modified=False,payload_hashes_verified=True,native_schema_verified=False,native_model_executed=False,
        parser=dict(name='bounded shared C-shell declaration analysis',version='1'),
        scope='输入准备绑定；source/编译环境及未提供的运行依赖保持UNKNOWN，不声称脚本可执行或完整CCTM运行')
