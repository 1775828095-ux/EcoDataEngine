"""Portable declared CMAQ projects; never executes archive scripts."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,re,shutil,stat,zipfile

SCHEMA='ecodata.cmaq.project-zip.v1'

def relative(name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or '\0' in name:
        raise ValueError('工程包路径必须是包内相对路径：'+str(name))
    p=PurePosixPath(name)
    if p.is_absolute() or any(x in ('','.','..') for x in name.split('/')):
        raise ValueError('工程包路径越界或不明确：'+name)
    if any(x.endswith((' ','.')) or re.match(r'(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)',x) for x in p.parts):
        raise ValueError('工程包路径不兼容Windows：'+name)
    return p

def unpack(payload,root):
    """Materialize immutable uploaded bytes and resolve declared input paths."""
    if len(payload)>1024*1024**2:raise ValueError('工程ZIP超过1GiB上传上限')
    digest=hashlib.sha256(payload).hexdigest();root=Path(root)/digest
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        entries={};seen=set();total=0
        for i in z.infolist():
            name=i.filename.rstrip('/') if i.is_dir() else i.filename
            relative(name)
            if name.casefold() in seen:raise ValueError('工程包路径重名：'+name)
            seen.add(name.casefold())
            if stat.S_ISLNK(i.external_attr>>16) or i.flag_bits&1:raise ValueError('不接受链接或加密工程文件')
            if not i.is_dir():entries[name]=i;total+=i.file_size
        if len(entries)>20000 or total>8*1024**3:raise ValueError('工程包文件数量或解压大小超限')
        candidates=[n for n in entries if PurePosixPath(n).name=='cmaq_project.json']
        if len(candidates)!=1:raise ValueError('工程包需且只能包含一份cmaq_project.json，明确配置与输入文件对应关系；不会猜测或执行运行脚本')
        m=candidates[0]
        if entries[m].file_size>2*1024**2:raise ValueError('工程配置超过2MiB')
        manifest=json.loads(z.read(m).decode('utf-8-sig'))
        if manifest.get('schema')!=SCHEMA:raise ValueError('工程配置schema应为'+SCHEMA)
        parent=PurePosixPath(m).parent
        inputs=manifest.get('inputs');env=manifest.get('environment')
        if not isinstance(inputs,dict) or not isinstance(env,dict):raise ValueError('工程配置需包含environment和inputs对象')
        unavailable=manifest.get('unavailable_inputs',{})
        if not isinstance(unavailable,dict):raise ValueError('未提供输入声明需为对象')
        for key,value in unavailable.items():
            if key in inputs or env.get(key)!=value or not isinstance(value,str):raise ValueError('未提供输入声明与配置不一致：'+key)
        resolved={};missing=[]
        for key,item in inputs.items():
            if not isinstance(item,dict):raise ValueError('输入声明格式错误：'+key)
            rel=relative(item.get('path'))
            if env.get(key)!=str(rel):raise ValueError('环境配置与包内输入路径不一致：'+key)
            member=str(parent/rel)
            if member not in entries:missing.append(key+' → '+member)
            resolved[key]=member
        if missing:raise ValueError('工程包缺少文件，请补齐：\n'+'\n'.join(missing))
        root.mkdir(parents=True,exist_ok=True)
        raw=root/'original_project.zip'
        if not raw.exists():raw.write_bytes(payload)
        elif hashlib.sha256(raw.read_bytes()).hexdigest()!=digest:raise ValueError('保存的原工程包发生变化')
        files=root/'files';files.mkdir(exist_ok=True)
        for name,info in entries.items():
            dest=files.joinpath(*PurePosixPath(name).parts)
            if dest.exists():
                # Re-read the member to check cached bytes, not only its name.
                with z.open(info) as stream:
                    h=hashlib.file_digest(stream,'sha256').hexdigest()
                if hashlib.sha256(dest.read_bytes()).hexdigest()!=h:raise ValueError('工程缓存文件被修改：'+name)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as stream,dest.open('wb') as out:shutil.copyfileobj(stream,out)
        from ecodata.interface.camx_runner import linux
        local=dict(manifest,schema='ecodata.cmaq.local-project.v1',environment=dict(env),inputs={},unavailable_inputs={})
        for key,name in resolved.items():
            dest=files.joinpath(*PurePosixPath(name).parts);h=hashlib.sha256(dest.read_bytes()).hexdigest()
            expected=inputs[key].get('sha256')
            if expected is not None and expected!=h:raise ValueError('工程输入校验值不一致：'+key)
            local['inputs'][key]={'path':str(dest.resolve()),'sha256':h,'bytes':dest.stat().st_size}
            local['environment'][key]=linux(dest.resolve())
        # Preserve declared absent optional references, never fall back to the
        # author's computer. Existing inspect_manifest rejects required gaps;
        # existing UI displays optional gaps before allowing a run.
        for key in unavailable:
            value=linux((root/'unavailable'/key).resolve())
            local['environment'][key]=value;local['unavailable_inputs'][key]=value
        local['uploaded_archive_sha256']=digest
        (root/'resolved_project.json').write_text(json.dumps(local,ensure_ascii=False,indent=2),encoding='utf-8')
        return local
