"""Five native engineering inspection hooks sharing one catalog protocol.

Native text readers are reused verbatim from the existing platform. These are
read-only structural checks, not a new native-model execution or a certification
that every possible project/version is supported.
"""
from pathlib import PurePosixPath
import json
import re
from ecodata.models.native_control_readers import _groups, _integer, _stamp, weather_dependencies, control_values, PATTERN


def check_native_values(attrs, proposed):
    """Shared final native-domain acceptance, independent of processing route."""
    import numpy as np
    if 'flag_masks' in attrs:raise ValueError('位掩码字段尚无自动原生写回契约')
    if 'flag_values' in attrs and not np.isin(proposed,np.asarray(attrs['flag_values'])).all():
        raise ValueError('写回引入原生flag_values未允许的类别')
    bounds=np.asarray(attrs['valid_range']) if 'valid_range' in attrs else None
    low=attrs.get('valid_min',bounds[0] if bounds is not None else None)
    high=attrs.get('valid_max',bounds[-1] if bounds is not None else None)
    if low is not None and (proposed<low).any() or high is not None and (proposed>high).any():
        raise ValueError('写回超出原生字段声明的有效范围')


def native_container_evidence(model, attributes, variables):
    """Header evidence only; never certify a complete simulator input schema."""
    families=[]
    title=str(attributes.get('TITLE',''))
    if ('REAL_EM' in title or 'OUTPUT FROM WRF' in title) and 'Times' in variables:families.append('wrf')
    if 'CAMx_NAME' in attributes and 'TFLAG' in variables:families.append('camx')
    if 'IOAPI_VERSION' in attributes and 'TFLAG' in variables and 'CAMx_NAME' not in attributes:families.append('cmaq')
    if families!=[model]:raise ValueError('原生文件模型类型未知、冲突或与目标不符：'+str(families))
    return dict(model_id=model,evidence='actual native header and time-control variables',native_schema_verified=False)


def probe_sources(model, prefixes):
    """Reject unsupported native controls before materialising large data."""
    issues=[];controls=[]
    if model=='cmaq':
        for name,raw in prefixes:
            if PurePosixPath(name).suffix.lower()!='.csh':continue
            text=raw.decode('utf8',errors='replace')
            if ('CMAQ' in text or 'CTM_' in text) and ('setenv ' in text or 'set ' in text):
                controls.append(name)
                issues.append(dict(file=name,code='UNSUPPORTED_NATIVE_CONTROL_READER',
                    message='已识别CMAQ原生C-shell控制脚本，但当前转换插件只接收已解析的工程清单，尚不能静态解析此脚本及其变量引用。不会执行包内脚本或换用绑定的其他工程。'))
    return dict(model_id=model,controls=controls,issues=issues,
        native_model_executed=False,source_modified=False)


def inspect_sources(model, files, records):
    payloads=dict(files); claims={}; issues=[]
    unread={row['file'] for row in records if row.get('payload_read') is False}
    partial_inventory=any(row.get('partial_inventory') for row in records)
    def claim(name,role,reason,**extra):
        if name in claims and role=='data_candidate':
            return  # A resolved engineering reference is stronger than format.
        if name in claims and claims[name]['role']=='data_candidate':
            del claims[name]
        if name in claims and claims[name]['role']!=role:
            issues.append(dict(file=name,message='工程用途冲突：'+claims[name]['role']+' / '+role))
            return
        claims[name]=dict(file=name,role=role,evidence=[reason],**extra)
    def reference(control,value,key):
        # A portable upload can resolve only actual relative paths. Never read
        # an arbitrary local file named by an uploaded JSON or shell script.
        value=str(value).replace('\\','/')
        p=PurePosixPath(value)
        if p.is_absolute() or '..' in p.parts or ':' in value:
            issues.append(dict(file=control,message='工程引用不是包内安全相对路径：'+key+'='+value));return
        target=(PurePosixPath(control).parent/p).as_posix()
        if target not in payloads:
            issues.append(dict(file=control,message=('工程引用尚未按需核对：' if partial_inventory else '工程实际引用缺失：')+key+' → '+target));return
        claim(target,'engineering_dependency','工程控制明确引用：'+control+' / '+key)
    for row in records:
        name=row['file']
        if not name and row.get('partial_inventory'):
            continue
        base=PurePosixPath(name).name.lower()
        facts=row.get('facts') or {}; attrs=facts.get('attributes',{}); variables=facts.get('variables',{})
        if model=='wrf' and ('REAL_EM' in str(attrs.get('TITLE','')) or 'OUTPUT FROM WRF' in str(attrs.get('TITLE',''))):
            if base.startswith('wrfout'):
                claim(name,'reference_output','原生WRF模拟结果；不作为新的独立观测')
            elif base.startswith(('wrfinput_d','wrfbdy_d')) and 'Times' in variables:
                claim(name,'engineering_dependency','原生WRF初始/边界状态；同名SST不自动成为独立新来源',
                      native_dimensions=facts['dimensions'],native_projection={k:attrs[k] for k in ('MAP_PROJ','TRUELAT1','TRUELAT2','STAND_LON','DX','DY') if k in attrs})
        if model=='cmaq' and 'IOAPI_VERSION' in attrs and 'TFLAG' in variables:
            # The same IOAPI encoding serves emissions, meteorology and output.
            # File type alone cannot decide its scientific role.
            claim(name,'data_candidate','IOAPI头与TFLAG已读取；须继续按变量及工程引用判定用途',native_dimensions=facts['dimensions'])
        if model=='wrf' and base=='namelist.input':
            try:
                groups=_groups(payloads[name]);tc=groups['time_control'];domains=groups['domains']
                count=_integer(domains,'max_dom')
                if not 1<=count<=99:raise ValueError('max_dom超出可检查范围')
                dates={prefix:{component:_integer(tc,prefix+'_'+component)
                               for component in ('year','month','day','hour','minute','second')
                               if prefix+'_'+component in tc} for prefix in ('start','end')}
                unresolved=[prefix+'_'+component for prefix in ('start','end')
                            for component in ('year','month','day','hour','minute','second')
                            if component not in dates[prefix]]
                claim(name,'engineering_configuration','使用既有WRF namelist只读解析器；仅记录显式日期分量，不补写模型默认值',
                    native_requirements=dict(date_components=dates,unresolved_date_defaults=unresolved,
                                             interval_seconds=_integer(tc,'interval_seconds'),domains=count))
                for number in range(1,count+1):reference(name,f'wrfinput_d{number:02d}','initial_state')
                reference(name,'wrfbdy_d01','boundary_state')
            except (ValueError,KeyError,UnicodeError) as exc:issues.append(dict(file=name,message='WRF控制读取失败：'+str(exc)))
        elif model=='swat_plus' and base=='file.cio':
            if name in unread:
                issues.append(dict(file=name,message='工程控制尚未完整读取，不能判断simulation/climate组是否存在'))
                continue
            try:
                members={PurePosixPath(n).name:n for n in payloads if PurePosixPath(n).parent==PurePosixPath(name).parent}
                local={base:payloads[n] for base,n in members.items()}
                groups={}
                for line in payloads[name].decode('utf-8-sig').splitlines()[1:]:
                    tokens=line.split('!',1)[0].split()
                    if tokens:
                        if tokens[0] in groups:raise ValueError('file.cio重复组：'+tokens[0])
                        groups[tokens[0]]=tokens[1:]
                if not {'simulation','climate'}<=set(groups):raise ValueError('缺simulation/climate组，不能只凭文件名确认SWAT工程')
                claim(name,'engineering_configuration','原生file.cio分组与文件引用；天气站关系复用已有检查器')
                for group,values in groups.items():
                    for value in values:
                        if value.lower() not in {'null','sim','0'}:reference(name,value,group)
                check=weather_dependencies(local,unread_files={base for base,n in members.items() if n in unread})
                claims[name]['weather_dependencies']=check
                issues.extend(dict(file=name,message=m) for m in check['errors'])
                for climate_file in groups.get('climate',[])[2:]:
                    if climate_file not in local:continue
                    # SWAT climate list: title, filename header, daily files.
                    for line in local[climate_file].decode('utf-8-sig').splitlines()[2:]:
                        tokens=line.split()
                        if len(tokens)==1 and tokens[0].lower() not in {'null','sim'}:
                            reference(name,tokens[0],climate_file)
            except (ValueError,UnicodeError) as exc:issues.append(dict(file=name,message='SWAT控制读取失败：'+str(exc)))
        elif model=='calpuff' and payloads[name].startswith(b'CALPUFF.INP     7.0'):
            try:
                values=control_values(payloads[name].decode('ascii'))
                claim(name,'engineering_configuration','CALPUFF7控制头与既有control_values解析器',native_requirements={k:values.get(k) for k in ('NX','NY','NZ','NREC','NPT1')})
                met=values.get('METDAT',[]);aux=values.get('AUXEXT',[])
                if len(met)!=1 or len(aux)!=1:raise ValueError('METDAT/AUXEXT缺少或重复')
                reference(name,met[0],'METDAT');reference(name,met[0]+'.'+aux[0],'AUXEXT')
            except (ValueError,UnicodeError) as exc:issues.append(dict(file=name,message='CALPUFF控制读取失败：'+str(exc)))
        elif model=='calpuff' and base in {'calpuff.inp', 'cpuf.inp'}:
            message = ('CALPUFF 控制文件尚未完整读取。' if name in unread else
                       '控制文件头不符合当前 CALPUFF 7.0 工程入口。')
            issues.append(dict(file=name, message=message))
        elif model=='camx' and b'&camx_control' in payloads[name][:4096].lower():
            try:
                assignments={}
                for line in payloads[name].decode('utf-8-sig').splitlines():
                    if not line.strip() or line.lstrip().startswith('!') or line.strip().lower() in {'&camx_control','/'}:continue
                    match=PATTERN.fullmatch(line)
                    if not match:raise ValueError('不能解析控制行：'+line[:100])
                    key,value=match.groups();key=re.sub(r'\s','',key)
                    if key in assignments:raise ValueError('重复控制项：'+key)
                    assignments[key]=value
                claim(name,'engineering_configuration','复用既有CAMx完整赋值解析规则，不执行控制脚本')
                file_keys={'Chemistry_Parameters','Photolysis_Rates','Ozone_Column','Initial_Conditions','Boundary_Conditions'}
                for key,value in assignments.items():
                    if key in file_keys or re.fullmatch(r'(?:Point_Sources|Surface_Grid|Met3D_Grid|Met2D_Grid|Vdiff_Grid|Emiss_Grid)\([\d,]+\)',key):
                        if not re.fullmatch(r"'[^'\r\n]+'",value):raise ValueError('文件引用未明确：'+key)
                        reference(name,value[1:-1],key)
            except (ValueError,UnicodeError) as exc:issues.append(dict(file=name,message='CAMx控制读取失败：'+str(exc)))
        elif model=='cmaq' and PurePosixPath(name).suffix.lower()=='.json':
            try: document=json.loads(payloads[name])
            except (ValueError,UnicodeError):continue
            if not isinstance(document,dict) or document.get('schema')!='ecodata.cmaq.local-project.v1':continue
            claim(name,'engineering_configuration','已识别既有CMAQ清单协议；不执行启动脚本')
            for key,item in document.get('inputs',{}).items():
                if not isinstance(item,dict) or not isinstance(item.get('path'),str):
                    issues.append(dict(file=name,message='CMAQ输入声明缺路径：'+key));continue
                reference(name,item['path'],key)
            for key in document.get('unavailable_inputs',{}):issues.append(dict(file=name,message='CMAQ声明缺少输入：'+key))
    controls=[r['file'] for r in claims.values() if r['role']=='engineering_configuration']
    if len(controls)>1:
        issues.append(dict(file='工程',message='识别到多个工程控制文件，不能自动猜选：'+'、'.join(controls)))
    return dict(claims=list(claims.values()),issues=issues,
        scope='原生头/控制/包内引用只读识别；不是完整工程可运行性或版本兼容认证')
