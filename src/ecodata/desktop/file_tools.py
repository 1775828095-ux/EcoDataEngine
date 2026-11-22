"""Read-only, selected-member inspection in a disposable worker process."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import csv
import base64
import json
import sqlite3
import sys
import tarfile
import zipfile

from .scan import FORMATS, archive_stream, decode, kind, materialize_member, safe, scientific, tabular


@contextmanager
def selected(root, locator, *, stream_table=False, max_leaf_bytes=None):
    root = Path(root).resolve(strict=True)
    parts = locator.split('!/') if locator else []
    if root.is_dir():
        relative = parts.pop(0) if parts else ''
        if relative and not safe(relative):
            raise ValueError('路径越出授权目录')
        current = (root / relative).resolve(strict=True)
        if not current.is_relative_to(root):
            raise ValueError('路径越出授权目录')
    else:
        current = root
    original = current
    containers = []
    reference = [original, '', None, None]
    with ExitStack() as stack:
        for index, member in enumerate(parts):
            if not safe(member):
                raise ValueError('不安全的归档成员路径')
            format_kind = kind(parts[index - 1]) if index else kind(original)
            if format_kind == 'zip':
                with zipfile.ZipFile(current) as archive:
                    matches = [x for x in archive.infolist() if x.filename == member and not x.is_dir()]
                    if len(matches) != 1:
                        raise ValueError('归档成员不存在或重复')
                    size, offset = matches[0].file_size, matches[0].header_offset
            elif format_kind == '7z':
                import py7zr
                with py7zr.SevenZipFile(current) as archive:
                    matches = [x for x in archive.list() if x.filename == member and not x.is_directory]
                    if len(matches) != 1:
                        raise ValueError('归档成员不存在或重复')
                    size, offset = matches[0].uncompressed, None
            elif format_kind == 'tar':
                with tarfile.open(current) as archive:
                    matches = [x for x in archive if x.name == member and x.isfile()]
                    if len(matches) != 1:
                        raise ValueError('归档成员不存在或重复')
                    size, offset = matches[0].size, matches[0].offset_data
            else:
                raise ValueError('只有归档路径可以包含 !/')
            reference = [original, format_kind, member, offset]
            if index == len(parts) - 1 and max_leaf_bytes is not None and size > max_leaf_bytes:
                raise ValueError("工程说明超出本地引用读取预算")
            if (stream_table and index == len(parts) - 1 and format_kind == 'zip'
                    and Path(member).suffix.lower() in {'.csv', '.tsv'}):
                break
            current = stack.enter_context(materialize_member((current, format_kind, member, offset), size))
            # Preserve the suffix used by format-specific readers, including nested archives.
            if index < len(parts) - 1:
                containers.append((reference.copy(), size))
        yield current, reference, containers


def entries(path, locator):
    format_kind = kind(locator or path.name)
    if path.is_dir():
        return [dict(name=p.name, bytes=p.stat().st_size if p.is_file() else None,
                     directory=p.is_dir()) for p in sorted(path.iterdir()) if not p.is_symlink()]
    if format_kind == 'zip':
        with zipfile.ZipFile(path) as archive:
            return [dict(name=x.filename, bytes=x.file_size, directory=x.is_dir())
                    for x in archive.infolist() if safe(x.filename.rstrip('/'))]
    if format_kind == '7z':
        import py7zr
        with py7zr.SevenZipFile(path) as archive:
            return [dict(name=x.filename, bytes=x.uncompressed, directory=x.is_directory)
                    for x in archive.list() if safe(x.filename.rstrip('/'))]
    if format_kind == 'tar':
        with tarfile.open(path) as archive:
            return [dict(name=x.name, bytes=x.size, directory=x.isdir())
                    for x in archive if (x.isfile() or x.isdir()) and safe(x.name.rstrip('/'))]
    return [dict(name=path.name, bytes=path.stat().st_size, directory=False)]


def inspect(root, operation, arguments):
    locator = arguments.get('path', '')
    if not isinstance(locator, str):
        raise ValueError('path必须为字符串')
    if operation == 'read_text' and Path(locator or root).suffix.lower() in {'.csv', '.tsv'}:
        raise ValueError('请用 read_header 读取列名，再按已审核字段申请数值样本')
    with selected(root, locator, stream_table=operation == 'read_header') as (path, reference, containers):
        if operation == 'list_directory':
            start = arguments.get('offset', 0)
            if type(start) is not int or start < 0:
                raise ValueError('offset必须为非负整数')
            rows = entries(path, locator)
            for row in rows:
                if path.is_dir():
                    row['path'] = '/'.join(x for x in (locator, row['name']) if x)
                elif kind(locator or path.name):
                    row['path'] = '!/'.join(x for x in (locator, row['name']) if x)
                else:
                    row['path'] = locator
            return dict(entries=rows[start:start + 150], total=len(rows),
                        next_offset=start + 150 if start + 150 < len(rows) else None,
                        arrays_read=False)
        display = (Path(root).name + '!/' + locator) if Path(root).is_file() and locator else locator or path.name
        row = dict(name=display, bytes=path.stat().st_size, ref=reference, containers=containers,
                   format=FORMATS.get(Path(locator or path.name).suffix.lower(), '其他'),
                   fields=[], resolution={}, basis=[], pending=[], evidence=[], roles=[])
        if operation == 'read_text':
            offset = arguments.get('offset', 0)
            if type(offset) is not int or offset < 0:
                raise ValueError('offset必须为非负字节位置')
            suffix = Path(locator or path.name).suffix.lower()
            if suffix in {'.docx', '.pdf'}:
                from .document_evidence import extract
                evidence = extract(path, suffix, offset)
                return dict(**evidence, row=row, document=dict(name=display,
                            text=evidence['text'], partial=True), evidence_path=locator, arrays_read=False)
            with path.open('rb') as stream:
                stream.seek(offset)
                raw = stream.read(16385)
            if b'\x00' in raw:
                raise ValueError('此文件为二进制，请使用read_header')
            text = decode(raw[:16384])
            result = dict(text=text, next_offset=offset + 16384 if len(raw) > 16384 else None,
                        row=row, document=dict(name=display, text=text, offset=offset,total_bytes=path.stat().st_size,
                            raw_base64=base64.b64encode(raw[:16384]).decode('ascii'),partial=offset > 0 or len(raw) > 16384),
                        evidence_path=locator, arrays_read=False)
            if suffix in {'.md', '.txt'}:
                from .document_evidence import text_passages
                result['evidence'] = text_passages(text, markdown=suffix == '.md')
            return result
        if operation != 'read_header':
            raise ValueError('未知只读工具')
        suffix = Path(locator or path.name).suffix.lower()
        if suffix in {'.sqlite', '.sqlite3', '.db', '.gpkg'}:
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as database:
                tables = database.execute("select name,sql from sqlite_master where type='table' order by name limit 150").fetchall()
            return dict(tables=tables, row=row, arrays_read=False)
        if suffix in {'.csv', '.tsv', '.txt', '.pcp', '.tmp', '.cli', '.cio'}:
            if suffix in {'.csv', '.tsv'} and reference[1] == 'zip':
                with zipfile.ZipFile(path) as archive:
                    row['bytes'] = archive.getinfo(reference[2]).file_size
                with archive_stream(path, *reference[1:]) as stream:
                    raw = stream.read(16384)
            else:
                with path.open('rb') as stream:
                    raw = stream.read(16384)
            lines = decode(raw).splitlines()[:8]
            if suffix in {'.csv', '.tsv'} and lines:
                delimiter = '\t' if suffix == '.tsv' else ','
                columns = next(csv.reader([lines[0]], delimiter=delimiter))
                from io import StringIO
                header = StringIO()
                csv.writer(header, delimiter=delimiter).writerow(columns)
                tabular(row, header.getvalue().encode('utf-8'), False)
            if suffix in {'.csv', '.tsv'}:
                lines = lines[:1]
            answer=dict(header_lines=lines, row=row, arrays_read=False,
                        sample_scope='first eight lines, at most 16 KiB; not a full-series audit')
            if suffix in {'.csv', '.tsv'}:
                answer['sample_scope'] = 'column header only; observation samples are requested separately'
            if suffix in {'.cio','.cli'} and len(raw)==path.stat().st_size:
                answer['document']=dict(name=display,text=decode(raw),offset=0,total_bytes=len(raw),
                    raw_base64=base64.b64encode(raw).decode('ascii'),partial=False)
            return answer
        if suffix not in {'.nc', '.nc4', '.netcdf', '.h5', '.hdf5', '.hdf', '.tif', '.tiff', '.xlsx', '.parquet'}:
            raise ValueError('该格式尚无表头读取器，可用read_text核对文本')
        scientific(row, path)
        row.pop('original_probe_payload', None)
        companions=[]
        try:
            with selected(root,locator+'.receipt.json') as (receipt,_,__):
                if receipt.stat().st_size<=65536:
                    companions.append(dict(name=display+'.receipt.json',text=receipt.read_text(encoding='utf-8-sig'),partial=False))
        except (ValueError,OSError,KeyError):pass
        return dict(row=row, companion_documents=companions,evidence_path=locator,
                    scope='selected file metadata via existing format reader')


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        result = inspect(request['root'], request['operation'], request['arguments'])
        print(json.dumps(dict(ok=True, result=result), ensure_ascii=True, default=str))
    except Exception as error:
        print(json.dumps(dict(ok=False, error=str(error)), ensure_ascii=True))
