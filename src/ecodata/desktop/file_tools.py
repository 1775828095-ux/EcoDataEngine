"""Read-only, selected-member inspection in a disposable worker process."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from ecodata.core.temporary import TemporaryDirectory
import csv
import base64
import json
import hashlib
import shutil
import sqlite3
import sys
import tarfile
import zipfile

from .scan import FORMATS, archive_stream, decode, kind, materialize_member, safe, scientific, tabular

OBSERVATION_TEXT = {".csv", ".tsv", ".pcp", ".tmp", ".hmd", ".slr", ".wnd", ".rdb", ".isd"}


def sendable_inspection(result, locator):
    """Construct the tool reply from structure and filtered engineering text."""
    from .document_content import PLACEHOLDER, classify, numeric_row, scalar_numeric, table_row
    from .document_evidence import text_passages
    from .engineering_excerpt import INSPECTION_NUMBER, PRIVATE, redact_private
    from .sampling_plan import recommendation_metadata
    from urllib.parse import parse_qs, urlsplit

    def excerpt(name, raw, *, engineering_passage=False):
        document = classify(name, raw)
        text = document.text
        suffix = Path(name).suffix.casefold()
        # Short controls may contain a single setting, insufficient for the
        # general document classifier to identify the entire fragment.
        if text is None and document.mode == "unknown" and suffix in {".cio", ".cli", ".ini", ".cfg", ".nml"}:
            text = "\n".join(line for line in raw.splitlines() if "=" in line or ":" in line)
        if text is None and document.mode == "unknown" and engineering_passage:
            text = raw
        if text is None:
            return PLACEHOLDER
        lines = []
        for line in text.splitlines():
            cells = table_row(line)
            data = numeric_row(cells) or any(
                delimiter in line and any(scalar_numeric(cell) for cell in cells)
                for delimiter in (",", "|", "\t", ";"))
            lines.append(PLACEHOLDER if data else INSPECTION_NUMBER.sub("[数值]", PRIVATE.sub(redact_private, line)))
        return "\n".join(lines)

    def companion(doc):
        clean = dict(doc, text=excerpt(doc["name"], doc["text"]))
        if doc["name"].casefold().endswith(".receipt.json"):
            try:
                url = json.loads(doc["text"]).get("url")
                values = parse_qs(urlsplit(url).query).get("time-standard", [])
                if values in (["UTC"], ["LST"]):
                    clean["declared_time_standard"] = values[0]
            except (AttributeError, TypeError, ValueError):
                pass
        return clean

    clean = dict(result)
    if "row" in clean:
        local_keys = {"ref", "containers", "attribute_member"}
        clean["row"] = recommendation_metadata(
            {key: value for key, value in result["row"].items() if key not in local_keys})
    name = result.get("row", {}).get("name", locator) or "说明.txt"
    if "text" in result:
        if Path(name).suffix.casefold() in {".pdf", ".docx"} and "evidence" in result:
            clean["evidence"] = [
                dict(location=INSPECTION_NUMBER.sub("[编号]", PRIVATE.sub(redact_private, item["location"])),
                     excerpt=excerpt(name, item["excerpt"], engineering_passage=True))
                for item in result["evidence"]]
            clean["text"] = "\n".join(item["excerpt"] for item in clean["evidence"])
        else:
            clean["text"] = excerpt(name, result["text"])
        if "evidence" in result and Path(name).suffix.casefold() not in {".pdf", ".docx"}:
            clean["evidence"] = text_passages(clean["text"], markdown=Path(name).suffix.casefold() == ".md")
    if "document" in result:
        doc = result["document"]
        clean["document"] = {**{key: value for key, value in doc.items() if key not in {"raw_base64", "text"}},
                             "text": clean["text"] if "text" in clean else excerpt(doc["name"], doc["text"])}
    if "companion_documents" in result:
        clean["companion_documents"] = [companion(doc) for doc in result["companion_documents"]]
    return clean


def tar_members(path,cache_folder=None):
    cache=None
    if cache_folder:
        key=hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
        cache=Path(cache_folder)/(key+'.tar-index.json')
        if cache.exists():return json.loads(cache.read_text(encoding='utf8'))
    with tarfile.open(path) as archive:
        rows=[dict(name=x.name,bytes=x.size,directory=x.isdir(),offset=x.offset_data)
            for x in archive if (x.isfile() or x.isdir()) and safe(x.name.rstrip('/'))]
    if cache:
        cache.parent.mkdir(parents=True,exist_ok=True)
        cache.write_text(json.dumps(rows),encoding='utf8')
    return rows


@contextmanager
def selected(root, locator, cache_folder=None, *, prefix_bytes=None, inspection=None, stream_table=False, max_leaf_bytes=None):
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
                matches=[x for x in tar_members(current,cache_folder) if x['name']==member and not x['directory']]
                if len(matches)!=1:raise ValueError('归档成员不存在或重复')
                size,offset=matches[0]['bytes'],matches[0]['offset']
            else:
                raise ValueError('只有归档路径可以包含 !/')
            reference = [original, format_kind, member, offset]
            if index == len(parts) - 1 and max_leaf_bytes is not None and size > max_leaf_bytes:
                raise ValueError("工程说明超出本地引用读取预算")
            if (stream_table and index == len(parts) - 1 and format_kind == 'zip'
                    and Path(member).suffix.lower() in {'.csv', '.tsv'}):
                break
            if index==len(parts)-1 and prefix_bytes is not None and format_kind in {'zip','tar'}:
                from .scan import archive_stream
                folder=stack.enter_context(TemporaryDirectory(prefix='ecodata-header-prefix-'))
                destination=Path(folder)/('prefix'+Path(member).suffix)
                with archive_stream(current,format_kind,member,offset) as source,destination.open('wb') as output:
                    output.write(source.read(prefix_bytes))
                current=destination
                if inspection is not None:inspection['source_bytes']=size
            elif cache_folder is not None and index < len(parts)-1:
                cache=Path(cache_folder);cache.mkdir(parents=True,exist_ok=True)
                identity=hashlib.sha256((str(root)+'!/'+'!/'.join(parts[:index+1])).encode()).hexdigest()
                cached=cache/(identity+Path(member).suffix)
                if not cached.exists():
                    with materialize_member((current,format_kind,member,offset),size) as staged:
                        shutil.copyfile(staged,cached)
                if cached.stat().st_size!=size:raise ValueError('Session container cache size mismatch')
                current=cached
            else:
                current = stack.enter_context(materialize_member((current, format_kind, member, offset), size))
            # Preserve the suffix used by format-specific readers, including nested archives.
            if index < len(parts) - 1:
                containers.append((reference.copy(), size))
        yield current, reference, containers


def entries(path, locator, cache_folder=None):
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
        return [{k:v for k,v in row.items() if k!='offset'} for row in tar_members(path,cache_folder)]
    return [dict(name=path.name, bytes=path.stat().st_size, directory=False)]


def inspect(root, operation, arguments, cache_folder=None):
    locator = arguments.get('path', '')
    if not isinstance(locator, str):
        raise ValueError('path必须为字符串')
    if operation == 'read_text' and Path(locator or root).suffix.lower() in OBSERVATION_TEXT:
        raise ValueError('请用 read_header 读取列名，再按已审核字段申请数值样本')
    inspection={}
    with selected(root, locator, cache_folder, stream_table=operation == 'read_header', inspection=inspection) as (path, reference, containers):
        if operation == 'list_directory':
            from .file_cards import describe

            start = arguments.get('offset', 0)
            if type(start) is not int or start < 0:
                raise ValueError('offset必须为非负整数')
            rows = entries(path, locator, cache_folder)
            for row in rows:
                if path.is_dir():
                    row['path'] = '/'.join(x for x in (locator, row['name']) if x)
                elif kind(locator or path.name):
                    row['path'] = '!/'.join(x for x in (locator, row['name']) if x)
                else:
                    row['path'] = locator
                if not row["directory"]:
                    row["local_format"] = describe(row["name"], row["bytes"])["format"]
            return dict(entries=rows[start:start + 150], total=len(rows),
                        next_offset=start + 150 if start + 150 < len(rows) else None,
                        arrays_read=False)
        display = (Path(root).name + '!/' + locator) if Path(root).is_file() and locator else locator or path.name
        source_bytes=inspection.get('source_bytes',path.stat().st_size)
        row = dict(name=display, bytes=source_bytes, ref=reference, containers=containers,
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
        if suffix in OBSERVATION_TEXT | {'.txt', '.cli', '.cio'}:
            if suffix in {'.csv', '.tsv'} and reference[1] == 'zip':
                with zipfile.ZipFile(path) as archive:
                    row['bytes'] = archive.getinfo(reference[2]).file_size
                with archive_stream(path, *reference[1:]) as stream:
                    raw = stream.read(16384)
            else:
                with path.open('rb') as stream:
                    raw = stream.read(16384)
            from .document_content import DATE, scalar_numeric, table_row

            lines = decode(raw).splitlines()[:8]
            first = table_row(lines[0]) if lines else []
            header = bool(first and not any(scalar_numeric(cell) or DATE.fullmatch(cell) for cell in first))
            if suffix in {'.csv', '.tsv'} and header:
                delimiter = '\t' if suffix == '.tsv' else ','
                columns = next(csv.reader([lines[0]], delimiter=delimiter))
                from io import StringIO
                header = StringIO()
                csv.writer(header, delimiter=delimiter).writerow(columns)
                tabular(row, header.getvalue().encode('utf-8'), False)
            if suffix in {'.csv', '.tsv'} and header:
                header_lines = lines[:1]
            else:
                from .scan import swat_result_fields

                recognized = swat_result_fields(row, lines)
                header_lines = [" ".join(field["name"] for field in row["fields"])] if recognized else []
                if not recognized and header and len(first) >= 2 and any(
                        token.casefold() in {"time", "date", "year", "day", "jday", "station_id", "日期", "时间"} for token in first):
                    header_lines = [lines[0]]
            answer = dict(header_lines=header_lines, row=row, arrays_read=False,
                          sample_scope='仅返回可识别的列名、类型和单位；观测记录须经抽样后另行提供')
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
        result = inspect(request['root'], request['operation'], request['arguments'],request.get('cache_folder'))
        print(json.dumps(dict(ok=True, result=result), ensure_ascii=True, default=str))
    except Exception as error:
        print(json.dumps(dict(ok=False, error=str(error)), ensure_ascii=True))
