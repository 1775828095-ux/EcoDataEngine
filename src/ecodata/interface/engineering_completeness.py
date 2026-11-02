"""Read-only project-dependent requirements. Never infer scientific input values.

This is a preflight report, not certification that every model input is valid.
Native validators remain authoritative and must run after this dependency gate.
"""
from pathlib import Path, PurePosixPath
import json
import re

LABELS = {'present': '已找到', 'missing': '缺失', 'conflict': '冲突',
          'unknown': '无法判断', 'not_required': '本次不需要'}


def finding(item, status, basis, action='', required=True):
    return dict(item=item, status=status, required=required, basis=basis,
                action=action, blocks_run=required and status in {'missing','conflict','unknown'})


def report(model, items, limitations=()):
    return dict(schema='ecodata.engineering-completeness.v1', model=model,
                findings=items, can_proceed=not any(i['blocks_run'] for i in items),
                limitations=list(limitations), scientific_data_modified=False,
                native_read_verified=False, qualification='仅通过已列出的预检范围；不代表完整兼容、原生读取或预测精度验证')


class IncompleteProject(ValueError):
    def __init__(self,result):
        self.completeness=result
        super().__init__('工程资料预检未通过：'+'；'.join(i['item']+'：'+LABELS[i['status']]+'（'+i['basis']+'）' for i in result['findings'] if i['blocks_run']))


def require(result):
    blocked=[i for i in result['findings'] if i['blocks_run']]
    if blocked:
        raise IncompleteProject(result)


def render(result, key):
    import streamlit as st
    st.markdown('#### 本次工程资料是否齐全')
    st.caption(result['qualification'])
    rows=[{'资料或检查项':i['item'],'状态':LABELS[i['status']],
           '是否阻止运行':'是' if i['blocks_run'] else '否',
           '判断依据':i['basis'],'需要补充或处理':i['action']} for i in result['findings']]
    if rows: st.dataframe(rows, hide_index=True, use_container_width=True)
    if not result['can_proceed']: st.error('必需资料存在缺失、冲突或无法判断的项目，请按清单处理后重新检查。')
    for note in result['limitations']: st.warning('检查边界：'+note)
    st.download_button('下载本次资料检查清单',json.dumps(result,ensure_ascii=False,indent=2).encode('utf8'),
                       key+'_completeness.json',key=key+'_completeness_download')
    return result['can_proceed']


def from_validation(model, validation, extra=()):
    """Preserve validator wording; do not guess missing/conflict from prose."""
    items=list(extra)
    items.extend(finding('模型检查 '+str(n+1),'unknown',message,'按原检查信息补充或修正工程；不修改科学设置')
                 for n,message in enumerate(validation.get('errors',[])))
    if not validation.get('errors'):
        items.append(finding('模型已实现的结构检查','present','本次实际调用模型检查器；仅覆盖下列检查边界'))
    return report(model,items,validation.get('warnings',[]))


def swat(files, validation):
    items=[]
    # Reuse the version-aware validator for conditional WGN/index exceptions.
    # Only classify direct missing references here when not explicitly optional.
    if 'file.cio' not in files:
        items.append(finding('file.cio','missing','SWAT+工程入口','补充完整TxtInOut工程，天气转换包不是完整工程'))
    else:
        items.append(finding('file.cio','present','已读取SWAT+工程入口'))
    for key in ('time_file','weather_dependencies'):
        if key in validation:
            items.append(finding(key,'present','已调用SWAT+版本/日期/天气站依赖检查；详情保留在原报告',required=False))
    return from_validation('SWAT+',validation,items)


def wrf(validation, updating=False):
    items=[]
    for key in ('wrfinput_d01','wrfbdy_d01'):
        present=key in validation.get('native_inputs',{})
        items.append(finding(key,'present' if present else 'unknown',
                             '按选定namelist.input的同目录引用读取实际NetCDF元数据',
                             '' if present else '补充可读取的对应文件；具体缺失或解析原因见模型检查'))
    for gap in validation.get('replacement_missing',[]):
        items.append(finding(gap,'missing' if updating else 'not_required',
                             '本次替换气象场需要WPS配套资料' if updating else '尚未请求新气象替换；原生工程结构检查不要求重新提供WPS',
                             '补充当前工程配套预处理资料' if updating else '',required=updating))
    if updating and not validation.get('replacement_checked'):
        items.append(finding('新气象替换兼容性','unknown','本次替换未完成全部检查','补齐替换资料并重新检查'))
    return from_validation('WRF',validation,items)


def cmaq(manifest):
    env=manifest.get('environment',{});inputs=manifest.get('inputs',{});items=[]
    if not isinstance(env,dict) or not isinstance(inputs,dict):
        return report('CMAQ',[finding('工程环境和输入映射','unknown','无法读取environment/inputs','补充有效工程配置')])
    required=set('INIT_CONC_1 BNDY_CONC_1 GRIDDESC MET_CRO_3D MET_CRO_2D MET_BDY_3D gc_matrix_nml ae_matrix_nml nr_matrix_nml tr_matrix_nml CSQY_DATA DESID_CTRL_NML DESID_CHEM_CTRL_NML'.split())
    for key in ('CTM_STDATE','CTM_STTIME','CTM_RUNLEN','CTM_TSTEP','GRID_NAME'):
        items.append(finding(key,'present' if env.get(key) else 'missing','本次时间和目标网格配置','补充原工程的明确配置' if not env.get(key) else ''))
    for count,prefixes in [('N_EMIS_GR',['GR_EMIS']),('N_EMIS_PT',['STK_GRPS','STK_EMIS'])]:
        try:
            n=int(env[count])
            if not 0<=n<=99:raise ValueError()
            required.update(f'{p}_{i:03}' for p in prefixes for i in range(1,n+1))
            items.append(finding(count,'present',f'原配置{count}={n}；按数量逐条检查输入流'))
        except (KeyError,ValueError,TypeError):
            items.append(finding(count,'unknown','缺少明确的0..99排放流数量','提供原工程排放流数量；平台不猜测'))
    # These predicates apply to the pinned CB6r5/AE7/m3dry entry only.
    disabled={}
    truth=lambda value:str(value).upper() in {'Y','YES','T','TRUE','.TRUE.'}
    false=lambda value:str(value).upper() in {'N','NO','F','FALSE','.FALSE.'}
    for flag,keys in [('CTM_DDM3D',['INIT_SENS_1','SEN_INPUT']),('VERTEXT',['VERTEXT_COORD_PATH'])]:
        for key in keys:
            if truth(env.get(flag)):required.add(key)
            elif false(env.get(flag)):disabled[key]=flag+'='+env[flag]
    if truth(env.get('NEW_START')):disabled['BEIS_SOILINP']='NEW_START='+env['NEW_START']+'；原生BEIS只在续算且未忽略土壤输入时读取'
    elif truth(env.get('CTM_BIOGEMIS_BE')) and false(env.get('NEW_START')):required.add('BEIS_SOILINP')
    if manifest.get('mechanism')=='cb6r5_ae7_aq_m3dry':
        disabled['STAGECTRL_NML']='本入口明确为m3dry构建，未启用STAGE沉降模块'
        disabled['GRID_CRO_3D']='当前受控CMAQ5.5常规网格构建不读取此历史引用；不能推广到其他构建'
    disabled['NMLpath']='启动目录提示；实际物种/控制文件按其独立输入映射检查，不把目录当科学文件'
    for key in sorted(required | set(inputs) | set(manifest.get('unavailable_inputs',{}))):
        if key in disabled and key not in required:
            items.append(finding(key,'not_required',disabled[key],required=False));continue
        item=inputs.get(key)
        if not isinstance(item,dict) or not item.get('path'):
            status='missing' if key in required else 'unknown'
            items.append(finding(key,status,'当前配置的必需输入' if key in required else '原工程声明不可用，但尚无规则证明本次无需读取',
                                 '补充对应原文件及工程内绑定；不沿用无关案例数据'));continue
        path=Path(item['path'])
        exists=path.is_file() and path.stat().st_size>0
        items.append(finding(key,'present' if exists else 'missing',str(path),'' if exists else '补充工程所引用的非空文件'))
    return report('CMAQ',items,['后续原有检查继续核对哈希、工程绑定、时间、物理网格和垂直定义；此清单本身不认证文件内容。',
                              '尚未穷举所有可选模块和点源文件内部物种/时间规则；未识别的缺失输入阻止运行。'])


def camx(path, values):
    """Parsed original assignments, not executable namelists or filename guessing."""
    from ecodata.interface.camx_external_project import FILES
    from ecodata.interface.cmaq_external_project import windows_path
    items=[];required={'Chemistry_Parameters','Photolysis_Rates','Ozone_Column','Initial_Conditions','Boundary_Conditions'}
    try:
        grids=int(values['Number_of_Grids'])
        if not 1<=grids<=2:raise ValueError()
        required.update(f'{kind}({g})' for g in range(1,grids+1) for kind in ['Surface_Grid','Met3D_Grid','Met2D_Grid','Vdiff_Grid'])
    except (KeyError,ValueError):
        grids=0;items.append(finding('Number_of_Grids','unknown','无法确定当前网格数量','提供支持范围内的明确网格数量'))
    refs={k:v for k,v in values.items() if k in FILES or re.fullmatch(r'(?:Point_Sources|Surface_Grid|Met3D_Grid|Met2D_Grid|Vdiff_Grid|Emiss_Grid)\([\d,]+\)',k)}
    for flag,prefix in [('Gridded_Emissions','Emiss_Grid'),('Point_Emissions','Point_Sources')]:
        if values.get(flag,'').lower()=='.true.':
            if prefix=='Emiss_Grid':
                for g in range(1,grids+1):
                    if not any(k.startswith(f'Emiss_Grid({g},') for k in refs):
                        items.append(finding(f'Emiss_Grid({g},*)','missing',flag+'=.true.，该网格未声明排放流','补充该网格的原排放流引用和文件'))
            elif not any(k.startswith('Point_Sources(') for k in refs):
                items.append(finding('Point_Sources(*)','missing',flag+'=.true.','补充点源文件引用和文件'))
        elif values.get(flag,'').lower()!='.false.':
            items.append(finding(flag,'unknown','排放启用状态未明确','提供原控制中的明确开关；不推测'))
    for key in sorted(required | set(refs)):
        value=refs.get(key)
        if key in {'Master_Grid_Restart','Nested_Grid_Restart'} and values.get('Restart','').lower()=='.false.':
            items.append(finding(key,'not_required','Restart=.false.；保留原文，不读取旧重启引用',required=False));continue
        if value is None:
            items.append(finding(key,'missing','当前网格和原生运行必需项','补充原控制引用及对应文件'));continue
        if not re.fullmatch(r"'[^'\r\n]+'",value):
            items.append(finding(key,'unknown','文件路径不是已解析的单引号路径','补充明确原路径；不执行变量'));continue
        try:
            original=value[1:-1];p=windows_path(original) if original.startswith('/mnt/') else Path(original)
            if not p.is_absolute():p=Path(path).parent/p
            exists=p.is_file() and p.stat().st_size>0
            items.append(finding(key,'present' if exists else 'missing',str(p),'' if exists else '补充原控制实际引用的非空文件'))
        except (OSError,ValueError) as exc:items.append(finding(key,'unknown',str(exc),'补充可识别的文件引用'))
    return report('CAMx',items,['依赖检查覆盖当前1至2网格、显式排放开关与引用；未穷举所有可选模块。',
                              '气象/点源内部变量、单位、时段和化学一致性尚不能由此清单全部判定，仍须后续兼容检查及原生读取。'])


def calpuff(files, control):
    from ecodata.interface.calpuff7_results import control_values
    values=control_values(files[control].decode('ascii'));items=[]
    lookup={n.casefold():n for n in files};parent=PurePosixPath(control).parent
    met=values.get('METDAT',[]);aux=values.get('AUXEXT',[])
    if len(met)!=1 or len(aux)!=1:
        items.append(finding('METDAT/AUXEXT','unknown','原控制缺少唯一气象引用','补充唯一明确的原控制引用'))
    else:
        for filename in (met[0],met[0]+'.'+aux[0]):
            key=str(parent/filename);found=lookup.get(key.casefold())
            items.append(finding(key,'present' if found and files[found] else 'missing','CALPUFF控制文件实际METDAT/AUXEXT引用',
                                 '' if found and files[found] else '补充原工程气象文件，不补零也不回退其他工程'))
    for key in ('NPTDAT','NARDAT','NVOLDAT','NFLDAT','NRDDAT','NLNDAT','NPT2','NSPT1','NAR1','NAR2','NVL1','NVL2','NLN2','NRD1','NRD2','NFL2'):
        if values.get(key)==['0']:
            items.append(finding(key,'not_required','原控制'+key+'=0；对应外部源未启用',required=False))
        else:items.append(finding(key,'unknown','当前仅支持控制内常量点源；原控制启用了其他源类型，或未明确提供唯一数量',
                                  '当前入口不能自动接入此源类型；保留数据并补充支持规则，不忽略该源'))
    return report('CALMET/CALPUFF',items,['已有CALMET气象文件时，不重复要求原始地形/气象观测；尚未在本检查器解析CALMET二进制内部覆盖。',
                                        '点源、受体、网格和控制字段仍由原有控制检查及原生程序判断；此清单不保证科学精度。'])
