"""Read-only project dependency inspection over the existing uploaded resources.

Native parsing and dependency binding are deliberately separate from native
execution. Dynamic scripts are evidence of unresolved inputs, never executed.
"""
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
import json
import re
import shlex
import zipfile

from .project_binding import ProjectFiles


def _text(files, name):
    with files.files[name][0]() as stream:
        return stream.read().decode('utf-8-sig')


def _bind(files, control, key, reference):
    row=dict(control=control,parameter=key,original=reference)
    try:
        target,reason=files.resolve(control,reference)
        row.update(status='BOUND',target=target,reason=reason,sha256=files.identity(target)['sha256'])
    except ValueError as exc:
        row.update(status='UNRESOLVED',reason=str(exc))
    return row


def _swat(files):
    # Positional groups follow the official readcio_read, not the label text.
    groups=('simulation basin climate connect channel reservoir routing_unit hru exco recall dr aquifer '
            'herd water_rights link hydrology structural hru_parm_db ops lum chg init soils decision_table regions').split()
    projects=[]
    for control in sorted(n for n in files.files if PurePosixPath(n).name=='file.cio'):
        lines=[r for r in _text(files,control).splitlines() if r.strip()]
        version=re.search(r'(?:\brev[.]?\s*|\bfor\s+SWAT\+\s+)(60|61|62)(?:[.\s]|$)',lines[0],re.I) if lines else None
        row=dict(control=files.identity(control),bindings=[],unknown=[],parser='existing SWAT weather dependencies + official positional CIO')
        projects.append(row)
        if not version or len(lines)<26:
            row['unknown'].append('缺少已支持的rev.60/61/62声明或完整25组CIO记录');continue
        sections={g:lines[i+1].split('!',1)[0].split()[1:] for i,g in enumerate(groups)}
        if len(sections['basin']) != (3 if version[1]=='62' else 2) or len(sections['climate'])!=9:
            row['unknown'].append('CIO位置/字段数量与声明版本不符');continue
        for group,refs in sections.items():
            for i,ref in enumerate(refs):
                if ref.lower() not in {'null','sim','0'}:
                    row['bindings'].append(_bind(files,control,group+'.'+str(i),ref))
        # Weather index entries are actual downstream file dependencies.
        for i in range(3,8):
            index=sections['climate'][i]
            if index.lower() in {'null','sim','0'}:continue
            try:index_name,_=files.resolve(control,index)
            except ValueError:continue
            # Native weather directory is a path relative to the run directory.
            path_rows=[r.split(None,1) for r in lines[26:31]]
            weather_path=path_rows[i-3][1].strip() if len(path_rows)>i-3 and len(path_rows[i-3])==2 else 'null'
            for line in _text(files,index_name).splitlines()[2:]:
                if not line.strip():continue
                ref=line.strip()
                if weather_path.lower() not in {'null','0','sim'}:ref=weather_path.rstrip('/\\')+'/'+ref
                row['bindings'].append(_bind(files,control,'weather_index:'+index,ref))
        # Reuse existing station/WGN/object linkage validation with lazy reads.
        from .swat_project_dependencies import weather_dependencies
        wanted={control,*[r['target'] for r in row['bindings'] if r['status']=='BOUND']}
        root=PurePosixPath(control).parent
        local={PurePosixPath(n).relative_to(root).as_posix():_text(files,n).encode()
               for n in wanted if PurePosixPath(n).is_relative_to(root)
               and (n==control or n.endswith(('.cli','.con')))}
        try:
            row['weather_dependencies']=weather_dependencies(local)
            row['unknown'].extend(row['weather_dependencies']['errors'])
            row['unknown'].extend(row['weather_dependencies']['warnings'])
        except (ValueError,UnicodeError) as exc:
            row['unknown'].append('既有天气关联读取失败：'+str(exc))
        row['unknown'].append('尚未验证全部数据库深层引用与原生版本兼容；不等于完整工程通过')
    return projects


def _cmaq(files):
    projects=[]
    for control in sorted(files.files):
        if not control.lower().endswith('.csh'):continue
        text=_text(files,control)
        # Official CCTM scripts provide I/O API file assignments. These are
        # declarations only: loops, sourcing and command substitution stay UNKNOWN.
        if not re.search(r'(?im)^\s*setenv\s+(?:INIT_CONC_1|BNDY_CONC_1|MET_CRO_2D|GRID_CRO_2D)\b',text):continue
        row=dict(control=files.identity(control),bindings=[],unknown=[],parser='CMAQ I/O API declarations (non-executing)')
        for number,line in enumerate(text.splitlines(),1):
            match=re.match(r'\s*setenv\s+(\w+)\s+(.+)',line)
            if not match:continue
            key,value=match.groups()
            if not re.match(r'^(?:INIT_CONC|BNDY_CONC|MET_|GRID_|OCEAN_|LUFRAC_|EMIS_|GR_EMIS|STK_)',key):continue
            try:tokens=shlex.split(value,comments=True)
            except ValueError:tokens=[]
            if len(tokens)!=1:
                row['bindings'].append(dict(control=control,parameter=key,original=value,
                    status='UNKNOWN',activation='UNKNOWN',line=number,reason='不是可静态读取的单值声明'));continue
            value=tokens[0]
            # Literal candidates can be located, but their active branch is not
            # established. Never silently substitute host environment variables.
            item=_bind(files,control,key,value) if not re.search(r'[$`*?;]',value) else dict(control=control,parameter=key,original=value,status='UNKNOWN',reason='依赖运行时变量、循环或命令；未执行上传脚本')
            item['line']=number;item['activation']='UNKNOWN';row['bindings'].append(item)
        row['unknown'].append('C-shell条件、循环、source及日期展开尚无可靠静态求值器；声明不等于实际激活输入')
        projects.append(row)
    return projects


def _calpuff(files):
    from ecodata.models.native_control_readers import control_values
    projects=[]
    for control in sorted(n for n in files.files if n.lower().endswith('.inp')):
        try:values=control_values(_text(files,control))
        except (UnicodeError,ValueError):continue
        kind='CALMET' if {'GEODAT','SRFDAT','IPROG'}<=set(values) else 'CALPUFF' if {'METDAT','CONDAT','NPT1'}<=set(values) else None
        if kind:projects.append(dict(control=files.identity(control),kind=kind,parameters=values,bindings=[],unknown=['解析控制语法；激活依赖由现有CALMET/CALPUFF驱动验收，非任意版本原生认证'],parser='existing native_control_readers.control_values'))
    if projects:
        # Reuse the existing active-input planner instead of maintaining a
        # second IPROG/count/file-role table. No extraction or process launch.
        from types import SimpleNamespace
        from existing_air_converters.calpuff_native_bundle import dependency_plan
        class ArchiveView:
            def infolist(self):
                return [SimpleNamespace(filename=n,external_attr=0,is_dir=lambda:False) for n in files.files]
            def read(self,name):
                with files.files[name][0]() as stream:return stream.read()
        try:
            selected,rejected,_=dependency_plan(ArchiveView())
            for row in projects:
                plan=selected[row['kind']]
                if row['control']['path'] not in plan['equivalent_control_copies']:continue
                row['bindings']=[dict(control=row['control']['path'],parameter=r['key'],original=r['reference'],
                                      target=r['member'],sha256=r['sha256'],status='BOUND') for r in plan['inputs']]
                row['active_dependency_plan']='existing CALMET/CALPUFF validated closure'
        except (ValueError,KeyError) as exc:
            for row in projects:row['unknown'].append(str(exc))
    return projects


def inspect_projects(uploads,model_id):
    reader={'swat_plus':_swat,'cmaq':_cmaq,'calpuff':_calpuff}.get(model_id)
    if reader is None:return None
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack);projects=reader(files)
    if not projects:return None
    return dict(schema='ecodata.engineering-inventory.v1',model_id=model_id,status='PARTIAL',
                projects=projects,source_modified=False,native_schema_verified=False,native_model_executed=False,
                scope='可追溯的工程引用盘点；缺失/未知显式保留，不声称任意完整工程已理解')


def prepare_project_inventory(uploads,model_id,folder,*,source_audit=None,weather_audit=None,adaptation=None,attachments=None):
    inventory=inspect_projects(uploads,model_id)
    if inventory is None:
        if not weather_audit or not weather_audit.get('ports'):return None
        inventory=dict(schema='ecodata.engineering-inventory.v1',model_id=model_id,status='PARTIAL',projects=[],
            source_modified=False,native_schema_verified=False,native_model_executed=False,
            scope='已识别声明的资源端口，但未识别完整原生工程控制；仅返回搜索缺项，不回退到其他处理分支')
    folder=Path(folder);path=folder/'project_inventory.json'
    path.write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')
    assessment=dict(schema='ecodata.engineering-admission.v1',sources=source_audit or [],weather=weather_audit or {},
        stages=dict(engineering_inspected=True,fusion_executed=False,native_writeback=False,native_model_executed=False),
        external_resources_identified=sum(s['status'] in {'IDENTIFIED_PRODUCT','DECLARED_SOURCE'} for s in source_audit or []),
        source_families=sorted({s['source_id'] for s in source_audit or [] if s.get('source_id')}),
        adaptation=adaptation or [],accuracy_assessed=False,
        note='来源身份不等于误差独立性。适配产物不等于融合；未更新工程、未运行模型；不能据此判断融合精度。')
    required={}
    for source in assessment['sources']:
        for reason in source.get('blockers',[]):
            row=required.setdefault(reason['code'],dict(code=reason['code'],message=reason['message'],resources=[]))
            row['resources'].append(source['resource'])
    assessment['required_information']=list(required.values())
    admission=folder/'source_admission.json'
    admission.write_text(json.dumps(assessment,ensure_ascii=True,indent=2,default=str),encoding='utf8')
    manifest=dict(protocol='ecodata.converter.v1',model_id=model_id,status='ENGINEERING_INSPECTED',native_ready=False,
                  products=[],preparation=dict(route='shared_project_inventory_v1',tasks=[],inventory=path.name,
                    source_admission=assessment),
                  note='工程已解析盘点，仍有未验收依赖；未执行融合或原生模型')
    manifest.update(native_writeback_verified=False,native_model_executed=False)
    handoff=folder/'handoff.json';handoff.write_text(json.dumps(manifest,ensure_ascii=True,indent=2,default=str),encoding='utf8')
    summary=['工程已盘点；未执行融合、工程写回或原生模型。',
        '外部资源识别数量：'+str(assessment['external_resources_identified']),
        '同产品不同日期是时段分片，不代表独立来源；原始文件未改写。']
    if assessment['weather'].get('blocked_search'):
        summary[0]='工程核验完成；尚未生成完整融合交付、未写回或运行原生模型。候选执行证据见search_diagnostics。'
    if not assessment['sources']:summary.append('本次未提交可识别的外部数值来源，仅盘点原工程。')
    for item in assessment['sources']:
        summary.append('\n'+item['resource']+'：'+item['status']+'；未使用')
        summary.extend('- '+r['message'] for r in item.get('blockers',[]))
        if item.get('longitude_binding_required'):summary.append('- '+item['longitude_binding_required'])
    for item in assessment['weather'].get('issues',[]):summary.append('\n天气端口：'+item.get('reason',''))
    for item in assessment['adaptation']:
        summary.append('\n自动适配：'+item['source_id']+'：'+item['status']+'；不是融合结果')
    summary.append('\n待补信息（集中汇总）：')
    summary.extend('- '+r['code']+'：'+r['message'] for r in assessment['required_information'])
    readme=folder/'processing_status.txt';readme.write_text('\n'.join(summary),encoding='utf8')
    missing=folder/'required_information.json'
    missing.write_text(json.dumps(assessment['required_information'],ensure_ascii=False,indent=2),encoding='utf8')
    archive=folder/'EcoData-project-inventory.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for item in (path,handoff,admission,readme,missing):z.write(item,item.name)
        if attachments:
            base=Path(attachments).resolve()
            if not base.is_relative_to(folder.resolve()):raise ValueError('适配证据目录越界')
            for item in sorted(base.rglob('*')):
                if item.is_file():z.write(item,'adaptation/'+item.relative_to(base).as_posix())
    return dict(manifest=manifest,archive=str(archive))
