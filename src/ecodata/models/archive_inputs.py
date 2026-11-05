"""Bounded nested-container ingestion; scientific metadata stays unchanged.

ZIP/TAR use Python's readers, 7z uses py7zr's streaming WriterFactory. Internal
TAR links become verified byte copies, never filesystem links or host reads.
"""
from pathlib import Path, PurePosixPath
from contextlib import ExitStack
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
                        if item.linkname.startswith(('/','\\')):raise ValueError('TAR链接目标越界')
                        ref=posixpath.normpath(posixpath.join(posixpath.dirname(key),item.linkname) if item.issym() else item.linkname)
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
