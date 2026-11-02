"""Portable CAMx engineering: explicit file bindings, unchanged science."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,re,shutil,stat,zipfile
from ecodata.interface.cmaq_project_zip import relative

SCHEMA='ecodata.camx.project-zip.v1'


def pack(control_path,destination):
    from ecodata.interface.camx_external_project import inspect
    report=inspect(control_path)
    manifest=dict(schema=SCHEMA,control='original_control.INP',inputs={})
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(report['control'],'original_control.INP')
        for n,item in enumerate(report['inputs']):
            name=f'inputs/{n:03}_{Path(item["path"]).name}'
            manifest['inputs'][item['key']]={'path':name,'sha256':item['sha256']}
            z.write(item['path'],name)
        z.writestr('camx_project.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    return destination


def unpack(payload,root):
    from ecodata.interface.camx_external_project import PATTERN,FILES,inspect
    from ecodata.interface.camx_runner import linux,sha
    if len(payload)>2*1024**3:raise ValueError('工程ZIP超过2GiB上传上限')
    digest=hashlib.sha256(payload).hexdigest();root=Path(root)/digest
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        entries={};seen=set();total=0
        for info in z.infolist():
            name=info.filename.rstrip('/') if info.is_dir() else info.filename
            relative(name)
            if name.casefold() in seen:raise ValueError('工程包路径重名：'+name)
            seen.add(name.casefold())
            if stat.S_ISLNK(info.external_attr>>16) or info.flag_bits&1:raise ValueError('不接受链接或加密工程文件')
            if not info.is_dir():entries[name]=info;total+=info.file_size
        if len(entries)>20000 or total>8*1024**3:raise ValueError('工程包文件数量或解压大小超限')
        configs=[n for n in entries if PurePosixPath(n).name=='camx_project.json']
        if len(configs)!=1:raise ValueError('需且只能包含一份camx_project.json，明确控制文件及输入对应关系')
        cfg=configs[0]
        if entries[cfg].file_size>2*1024**2:raise ValueError('工程配置超过2MiB')
        m=json.loads(z.read(cfg).decode('utf-8-sig'));parent=PurePosixPath(cfg).parent
        if m.get('schema')!=SCHEMA or not isinstance(m.get('inputs'),dict):raise ValueError('CAMx工程配置schema或inputs无效')
        control=str(parent/relative(m.get('control')))
        if control not in entries:raise ValueError('工程缺少原控制文件：'+control)
        if entries[control].file_size>1024**2:raise ValueError('控制文件超过1MiB')
        raw=z.read(control);text=raw.decode('utf-8-sig');keys=set();inactive=set();rows=[]
        for line in text.splitlines(keepends=True):
            item=PATTERN.fullmatch(line.rstrip('\r\n'))
            if item:
                key=re.sub(r'\s','',item[1])
                if key in {'Master_Grid_Restart','Nested_Grid_Restart'}:inactive.add(key)
                elif key in FILES or re.fullmatch(r'(?:Point_Sources|Surface_Grid|Met3D_Grid|Met2D_Grid|Vdiff_Grid|Emiss_Grid)\([\d,]+\)',key):keys.add(key)
            rows.append((line,item))
        if keys!=set(m['inputs']):raise ValueError('包内输入映射必须与原控制活动引用逐项对应：'+str(sorted(keys^set(m['inputs']))))
        members={};missing=[]
        for key,item in m['inputs'].items():
            if not isinstance(item,dict):raise ValueError('输入声明格式错误：'+key)
            name=str(parent/relative(item.get('path')));members[key]=name
            if name not in entries:missing.append(key+' → '+name)
        if missing:raise ValueError('工程包缺少文件，请补齐：\n'+'\n'.join(missing))
        root.mkdir(parents=True,exist_ok=True);original=root/'original_project.zip'
        if original.exists() and sha(original)!=digest:raise ValueError('保存的原工程包已变化')
        if not original.exists():original.write_bytes(payload)
        files=root/'files';files.mkdir(exist_ok=True)
        for name,info in entries.items():
            dest=files.joinpath(*PurePosixPath(name).parts)
            if dest.exists():
                with z.open(info) as f:h=hashlib.file_digest(f,'sha256').hexdigest()
                if sha(dest)!=h:raise ValueError('工程缓存文件被修改：'+name)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
        mapping={}
        for key,name in members.items():
            dest=files.joinpath(*PurePosixPath(name).parts)
            h=sha(dest)
            if m['inputs'][key].get('sha256',h)!=h:raise ValueError('工程输入校验值不符：'+key)
            mapping[key]=linux(dest.resolve())
        mapping.update({k:linux((root/'inactive'/k).resolve()) for k in inactive})
        mapping['Root_Output_Name']=linux((root/'output/CAMx.current').resolve())
        rendered=[]
        for line,item in rows:
            if item:
                key=re.sub(r'\s','',item[1])
                if key in mapping:line=line[:item.start(2)]+"'"+mapping[key]+"'"+line[item.end(2):]
            rendered.append(line)
        result=root/'resolved_control.INP';result.write_bytes(''.join(rendered).encode('utf8'))
        report=inspect(str(result.resolve()))
        report.update(uploaded_archive_sha256=digest,original_control_sha256=hashlib.sha256(raw).hexdigest(),
                      path_mapping=mapping,only_paths_relocated=True)
        (root/'import_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        return str(result.resolve()),report
