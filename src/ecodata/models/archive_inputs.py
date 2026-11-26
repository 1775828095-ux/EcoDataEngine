"""Bounded nested-container ingestion; scientific metadata stays unchanged.

ZIP/TAR use Python's readers, 7z uses py7zr's streaming WriterFactory. Internal
TAR links become verified byte copies, never filesystem links or host reads.
"""
from pathlib import Path, PurePosixPath
from contextlib import ExitStack, contextmanager
import hashlib
import json
import posixpath
import tarfile
import zipfile

from ecodata.interface.engineering_upload import LocalUpload, MAX_TOTAL_BYTES
from ecodata.models.project_binding import safe_name as _safe_name


def safe_name(name):
    # Virtual names are written only as ZIP members; stage paths are integers.
    return _safe_name(name,posix=True)


def _kind(name):
    name=str(name).lower()
    if name.endswith('.zip'):return 'zip'
    if name.endswith(('.tar','.tar.gz','.tgz','.tar.bz2','.tar.xz')):return 'tar'
    if name.endswith('.7z'):return '7z'


def normalize_archives(uploads, folder):
    """Retain existing flat ZIP contracts; expand only nested/mixed containers."""
    with ExitStack() as stack:
        nested=False
        for upload in uploads:
            kind=_kind(upload.name)
            if kind in {'tar','7z'}:nested=True
            elif kind=='zip':
                stream=Path(upload.path) if getattr(upload,'path',None) else __import__('io').BytesIO(upload.getvalue())
                archive=stack.enter_context(zipfile.ZipFile(stream))
                nested |= any(_kind(i.filename) for i in archive.infolist() if not i.is_dir())
        if not nested:return uploads,None
    folder=Path(folder);stage=folder/'archive_intake';stage.mkdir()
    reader=_Reader(stage)
    for upload in uploads:
        if getattr(upload,'path',None):path=Path(upload.path)
        else:
            path=stage/f'upload-{len(reader.origins)}';path.write_bytes(upload.getvalue())
        reader.visit(path,upload.name,'',0)
    bundle=folder/'expanded_inputs.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,(path,_) in sorted(reader.files.items()):z.write(path,name)
    audit=dict(schema='ecodata.nested-inputs.v1',source_modified=False,
               decoded_bytes=reader.total,resources=reader.origins,
               bundle_sha256=hashlib.file_digest(bundle.open('rb'),'sha256').hexdigest(),
               policy='preserve raw bytes; identical same-path resources collapse; conflicting resources fail; no uploaded code executed')
    (folder/'archive_intake.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    return [LocalUpload(bundle.name,bundle)],audit


@contextmanager
def normalized_archive_scope(uploads, folder, *, keep=None, progress=None):
    """Release only this call's intake copies; products and original inputs survive."""
    import shutil
    folder=Path(folder).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    names=('archive_intake','expanded_inputs.zip')
    existed={name:(folder/name).exists() or (folder/name).is_symlink() for name in names}
    protected=[Path(u.path).resolve() for u in uploads if getattr(u,'path',None)]
    keep=[] if keep is None else keep
    report=dict(schema='ecodata.temporary-cleanup.v1',status='PENDING',removed=[],retained=[],failures=[],released_bytes=0,
        scope='Only newly created intake copies of this invocation; no historical workspace sweep')
    audit=None
    try:
        # Public readers share a virtual resource index for nested ZIP/TAR.
        # Never build an expanded ZIP merely to let ProjectFiles read it again.
        nested=False
        supported=True
        for upload in uploads:
            typ=_kind(upload.name)
            supported &= typ!='7z'
            nested |= typ=='tar'
            if typ=='zip':
                source=Path(upload.path) if getattr(upload,'path',None) else __import__('io').BytesIO(upload.getvalue())
                with zipfile.ZipFile(source) as archive:
                    kinds=[_kind(i.filename) for i in archive.infolist() if not i.is_dir()]
                    nested |= any(kinds)
                    supported &= '7z' not in kinds
        if nested and supported:
            from .selective_resources import resource_scope
            with resource_scope(uploads,progress) as audit:
                yield uploads,audit,report
            (folder/'archive_intake.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
        else:
            normalized,audit=normalize_archives(uploads,folder)
            yield normalized,audit,report
    finally:
        if audit and audit.get('schema')=='ecodata.selective-resources.v1':
            path=Path(audit['temporary_directory'])
            audit['temporary_directory_absent']=not path.exists()
            audit['cleanup']='COMPLETED' if not path.exists() else 'PARTIAL'
            report['selective_resources']=audit
            if path.exists():report['failures'].append(dict(path=str(path),reason=audit.get('cleanup_error','按需资源清理未完成')))
            report['released_bytes']+=audit['container_bytes']+audit['staged_bytes'] if not path.exists() else 0
            (folder/'archive_intake.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
        protected.extend(Path(p).resolve() for p in keep)
        for name in names:
            path=folder/name
            if not path.exists() and not path.is_symlink():continue
            resolved=path.resolve()
            if (existed[name] or path.is_symlink() or not resolved.is_relative_to(folder)
                    or any(resolved==p or p.is_relative_to(resolved) for p in protected)):
                report['retained'].append(dict(path=str(path),reason='preexisting, symlink, input or retained product'))
                continue
            try:
                size=(sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
                      if path.is_dir() else path.stat().st_size)
                if path.is_dir():shutil.rmtree(path)
                else:path.unlink()
                report['removed'].append(dict(path=str(path),bytes=size));report['released_bytes']+=size
            except OSError as exc:
                report['failures'].append(dict(path=str(path),reason=str(exc)))
        report['status']='PARTIAL' if report['failures'] else 'COMPLETED'
        try:
            (folder/'temporary_cleanup.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        except OSError as exc:
            if progress:progress('临时文件清理回执写入失败：'+str(exc))
        if progress:
            progress(f"临时副本清理：{report['status']}，释放 {report['released_bytes']} 字节；原始数据、产品和记录保留")


class _Reader:
    def __init__(self,stage):
        self.stage=stage;self.files={};self.origins=[];self.total=0;self.count=0

    def path(self):
        self.count+=1
        if self.count>100000:raise ValueError('压缩资料成员过多，未继续展开')
        return self.stage/str(self.count)

    def copy(self,stream,path,expected):
        if expected<0 or self.total+expected>MAX_TOTAL_BYTES:raise ValueError('压缩资料展开超过10 GiB接收上限')
        size=0
        with path.open('wb') as out:
            while block:=stream.read(1024*1024):
                size+=len(block);self.total+=len(block)
                if size>expected or self.total>MAX_TOTAL_BYTES:raise ValueError('压缩资料实际展开大小超过声明或接收上限')
                out.write(block)
        if size!=expected:raise ValueError('压缩成员长度与声明不符')

    def add(self,path,name,origin):
        name=safe_name(name);digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        key=name.casefold();same=next((n for n in self.files if n.casefold()==key),None)
        if same is not None and (same!=name or self.files[same][1]!=digest):
            raise ValueError('嵌套工程成员冲突，未选择版本或覆盖：'+name)
        if same is None:self.files[name]=(path,digest)
        self.origins.append(dict(resource=name,origin=origin,sha256=digest,bytes=path.stat().st_size,duplicate=same is not None))

    def visit(self,path,name,prefix,depth,origin=None,project=False):
        if depth>5:raise ValueError('嵌套压缩层数超过5层，未继续展开')
        name=safe_name(name);origin=origin or name;kind=_kind(name)
        target=safe_name(posixpath.join(prefix,name))
        if kind is None or project:
            self.add(path,target,origin);return
        base=posixpath.dirname(target)
        if kind=='zip':
            with zipfile.ZipFile(path) as archive:
                project=any(PurePosixPath(i.filename).name.lower() in {'file.cio','namelist.input','namelist.wps','calmet.inp','calpuff.inp'} for i in archive.infolist())
                for item in archive.infolist():
                    safe_name(item.filename)
                    if item.is_dir():continue
                    if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('ZIP含符号链接，未按普通数据读取')
                    dest=self.path()
                    with archive.open(item) as stream:self.copy(stream,dest,item.file_size)
                    self.visit(dest,item.filename,base,depth+1,origin+'!/'+item.filename,project=project)
        elif kind=='tar':
            with tarfile.open(path,'r:*') as archive:
                entries=archive.getmembers()
                # The existing WRF converter owns opaque GRIB forcing bundles.
                # Keep that contract instead of turning them into unknown files.
                grib=bool(entries) and all(i.isfile() or i.isdir() for i in entries)
                regular=[i for i in entries if i.isfile()]
                for item in regular:
                    with archive.extractfile(item) as stream:magic=stream.read(8)
                    grib &= len(magic)==8 and magic[:4]==b'GRIB' and magic[7] in (1,2)
                    if not grib:break
                if grib and regular:
                    for item in entries:safe_name(item.name)
                    self.add(path,target,origin);return
                members={}
                for item in entries:
                    key=safe_name(item.name)
                    if key in members:raise ValueError('TAR成员重名：'+key)
                    members[key]=item
                def read_member(key,chain):
                    if key in chain:raise ValueError('TAR内部链接循环：'+key)
                    if key not in members:raise ValueError('TAR内部链接目标不存在：'+key)
                    item=members[key]
                    if item.issym() or item.islnk():
                        if item.linkname.startswith(('/','\\')):
                            from types import SimpleNamespace
                            from ecodata.models.project_binding import ProjectFiles
                            # Reuse the existing >=2-component unique suffix relocation.
                            # No archived absolute path is ever opened on the host.
                            ref,_=ProjectFiles.resolve(SimpleNamespace(files=members),key,item.linkname)
                        else:ref=posixpath.normpath(posixpath.join(posixpath.dirname(key),item.linkname) if item.issym() else item.linkname)
                        safe_name(ref)
                        return read_member(ref,chain|{key})
                    if not item.isfile():raise ValueError('TAR含不支持的特殊成员：'+key)
                    return archive.extractfile(item),item.size
                for key,item in members.items():
                    if item.isdir():continue
                    stream,size=read_member(key,set());dest=self.path()
                    with stream:self.copy(stream,dest,size)
                    self.visit(dest,key,base,depth+1,origin+'!/'+key)
        else:
            import py7zr
            from py7zr.io import Py7zIO,WriterFactory
            owner=self;paths={}
            class Writer(Py7zIO):
                def __init__(self,path,expected):self.stream=path.open('w+b');self.expected=expected;self.written=0
                def write(self,data):
                    self.written+=len(data);owner.total+=len(data)
                    if self.written>self.expected or owner.total>MAX_TOTAL_BYTES:raise ValueError('7z实际展开大小超过声明或接收上限')
                    return self.stream.write(data)
                def read(self,size=None):return self.stream.read(size)
                def seek(self,offset,whence=0):return self.stream.seek(offset,whence)
                def flush(self):self.stream.flush()
                def size(self):return self.written
                def close(self):self.stream.close()
            class Factory(WriterFactory):
                def create(self,filename):
                    key=safe_name(filename)
                    if key not in sizes:raise ValueError('7z产生未声明的成员')
                    dest=owner.path();paths[key]=dest
                    writer=Writer(dest,sizes[key]);writers.append(writer);return writer
            with py7zr.SevenZipFile(path,'r') as archive:
                sizes={};writers=[]
                for item in archive.list():
                    key=safe_name(item.filename)
                    if item.is_directory:continue
                    if item.is_symlink or not item.is_file:raise ValueError('7z含链接或特殊成员')
                    if key in sizes:raise ValueError('7z成员重名')
                    sizes[key]=item.uncompressed
                if self.total+sum(sizes.values())>MAX_TOTAL_BYTES:raise ValueError('7z展开超过10 GiB接收上限')
                try:archive.extractall(factory=Factory())
                finally:
                    for writer in writers:writer.close()
                project=any(PurePosixPath(n).name.lower() in {'file.cio','namelist.input','namelist.wps','calmet.inp','calpuff.inp'} for n in sizes)
                for key,size in sizes.items():
                    if key not in paths or paths[key].stat().st_size!=size:raise ValueError('7z成员长度不符')
                    self.visit(paths[key],key,base,depth+1,origin+'!/'+key,project=project)
