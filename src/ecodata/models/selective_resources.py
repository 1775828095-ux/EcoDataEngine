"""Scoped virtual ZIP/TAR resources; only requested leaves are staged.

Compressed TAR must be scanned sequentially to find its headers. That scan
does not write leaf payloads. Seekable compressed containers and requested
members are owned by the existing temporary-directory lifecycle.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
import io
import hashlib
import shutil
import tarfile
import zipfile

_active=ContextVar('ecodata_resource_index',default=None)


def signature(uploads):
    return tuple((u.name,str(getattr(u,'path','')),id(u) if not getattr(u,'path',None) else None) for u in uploads)


def current_index(uploads):
    value=_active.get()
    return value[1] if value is not None and value[0]==signature(uploads) else None


def kind(name):
    name=name.lower()
    if name.endswith('.zip'):return 'zip'
    if name.endswith(('.tgz','.tar','.tar.gz','.tar.bz2','.tar.xz')):return 'tar'


@contextmanager
def resource_scope(uploads,progress=None):
    from ecodata.core.temporary import TemporaryDirectory
    from .project_binding import ProjectFiles,safe_name
    temporary=TemporaryDirectory(prefix='ecodata-selected-resources-')
    directory=temporary.name
    with ExitStack() as stack:
        root=Path(directory)
        index=ProjectFiles([],stack,posix_members=True)
        index._prefixes={}
        index._prefix_readers={}
        audit=dict(schema='ecodata.selective-resources.v1',indexed_members=0,selected_members=[],
                   staged_bytes=0,container_bytes=0,prefix_cache_bytes=0,
                   policy='header index; selected member staging; 1 MiB stream blocks; 8 MiB prefix cache',
                   temporary_directory=str(root),cleanup='PENDING')
        def cleanup():
            try:temporary.cleanup()
            except OSError as exc:
                audit['cleanup']='PARTIAL';audit['cleanup_error']=str(exc)
                if progress:progress('按需资源清理未完成，已保留重试记录：'+str(exc))
            else:audit['cleanup']='COMPLETED'
            audit['temporary_directory_absent']=not root.exists()
        stack.callback(cleanup)
        serial=0

        def stage(opener,size,container=False):
            nonlocal serial
            serial+=1
            path=root/str(serial)
            if size<0 or shutil.disk_usage(root).free<size+64*1024**2:
                raise ValueError('磁盘空间不足以暂存实际所需资源；未展开无关数据')
            count=0
            with opener() as src,path.open('wb') as dst:
                while block:=src.read(1024*1024):
                    count+=len(block)
                    if count>size:raise ValueError('成员实际大小超过声明')
                    dst.write(block)
            if count!=size:raise ValueError('成员长度不符')
            audit['container_bytes' if container else 'staged_bytes']+=count
            return path

        def add(name,opener,size,origin,prefix=None):
            name=safe_name(name,posix=True)
            if name in index.files:
                previous,locator=index.files[name]
                if size!=locator['bytes']:raise ValueError('嵌套工程成员冲突：'+name)
                with previous() as a,opener() as b:
                    if hashlib.file_digest(a,'sha256').digest()!=hashlib.file_digest(b,'sha256').digest():
                        raise ValueError('嵌套工程成员冲突：'+name)
                audit.setdefault('identical_duplicates',[]).append(dict(path=name,origin=origin))
                return
            cached=None
            def selected():
                nonlocal cached
                if cached is None:
                    cached=stage(opener,size)
                    audit['selected_members'].append(dict(path=name,bytes=size,origin=origin))
                    if progress:progress('按需读取工程资源：'+name+f'（{size} 字节，分块复制）')
                return cached.open('rb')
            index.add(name,selected,dict(upload=origin,member=name,bytes=size))
            index._prefix_readers[name]=opener
            if prefix is not None:index._prefixes[name]=prefix
            audit['indexed_members']+=1
            if audit['indexed_members']>100000:raise ValueError('归档成员超过100000，未继续索引')

        def visit(opener,size,name,base='',depth=0,origin=''):
            if depth>5:raise ValueError('归档嵌套超过5层')
            name=safe_name(name,posix=True)
            target=str(PurePosixPath(base)/name)
            typ=kind(name)
            if not typ:
                add(target,opener,size,origin);return
            if progress:progress('索引嵌套归档目录（不展开叶文件）：'+target)
            path=stage(opener,size,container=True)
            parent=str(PurePosixPath(target).parent)
            if typ=='zip':
                archive=stack.enter_context(zipfile.ZipFile(path))
                for item in archive.infolist():
                    safe_name(item.filename,posix=True)
                    if item.is_dir():continue
                    if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('ZIP符号链接不适用')
                    visit(lambda a=archive,i=item:a.open(i),item.file_size,item.filename,parent,depth+1,origin+'!/'+item.filename)
            else:
                # One forward pass caches header prefixes; no full leaf extraction.
                members={}
                with tarfile.open(path,'r|*') as archive:
                    for item in archive:
                        key=safe_name(item.name,posix=True)
                        if key in members:raise ValueError('TAR成员重复：'+key)
                        if item.isdir():continue
                        if len(members)+audit['indexed_members']>=100000:raise ValueError('归档成员超过100000，未继续索引')
                        if not item.isfile():raise ValueError('TAR链接或特殊成员尚不能按需验收：'+key)
                        length=min(4096,item.size,max(0,8*1024**2-audit['prefix_cache_bytes']))
                        with archive.extractfile(item) as stream:prefix=stream.read(length)
                        audit['prefix_cache_bytes']+=len(prefix)
                        members[key]=(item,prefix)
                archive=stack.enter_context(tarfile.open(path,'r:*'))
                for key,(item,prefix) in members.items():
                    opening=lambda a=archive,i=item:a.extractfile(i)
                    if kind(key):visit(opening,item.size,key,parent,depth+1,origin+'!/'+key)
                    else:add(str(PurePosixPath(parent)/key),opening,item.size,origin+'!/'+key,prefix)

        for upload in uploads:
            path=getattr(upload,'path',None)
            if path:
                path=Path(path)
                # Top-level ZIP remains seekable in-place: do not copy the input.
                if kind(upload.name)=='zip':
                    archive=stack.enter_context(zipfile.ZipFile(path))
                    for item in archive.infolist():
                        safe_name(item.filename,posix=True)
                        if item.is_dir():continue
                        if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('ZIP符号链接不适用')
                        visit(lambda a=archive,i=item:a.open(i),item.file_size,item.filename,origin=upload.name+'!/'+item.filename)
                else:visit(lambda p=path:p.open('rb'),path.stat().st_size,upload.name,origin=upload.name)
            else:
                raw=upload.getvalue()
                visit(lambda b=raw:io.BytesIO(b),len(raw),upload.name,origin=upload.name)
        token=_active.set((signature(uploads),index))
        try:yield audit
        finally:_active.reset(token)
    audit['cleanup']='COMPLETED'
    audit['temporary_directory_absent']=not root.exists()
