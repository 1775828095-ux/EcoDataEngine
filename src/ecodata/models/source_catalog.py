"""Shared, evidence-based source catalog before the existing fusion engine.

Reader probing follows Intake's data/reader separation; model hooks follow
HydroMT's source/driver/adapter separation. No external project code is run.
CF evidence uses the already installed cf-xarray and pyproj libraries.
"""
from pathlib import PurePosixPath
import hashlib
import io
import json


ROLES = {'data_candidate', 'engineering_configuration', 'engineering_dependency',
         'reference_output', 'software_reference', 'documentation', 'source_metadata', 'unknown'}


def partition_uploads(uploads, folder):
    """Quarantine evidenced source snapshots before opening engineering inputs.

    A download declaration, pinned repository commit, actual SHA256 and source
    tree must agree. Mixed runtime/data archives stay subject to strict intake.
    Reference ZIPs are never extracted, imported or executed (including links).
    """
    from pathlib import Path
    from urllib.parse import urlparse
    import re,zipfile
    declarations=[]
    for item in uploads:
        if not str(item.name).lower().endswith('.json'):continue
        try:
            value=json.loads(item.getvalue())
        except (ValueError,UnicodeError):continue
        records=value if isinstance(value,list) else value.get('files',[]) if isinstance(value,dict) else []
        for record in records:
            if not isinstance(record,dict):continue
            try:url=urlparse(str(record.get('url','')))
            except ValueError:continue
            match=re.fullmatch(r'/[^/]+/([^/]+)/zip/([a-fA-F0-9]{40})',url.path) if url.hostname=='codeload.github.com' else None
            if not match and url.hostname=='github.com':
                match=re.fullmatch(r'/[^/]+/([^/]+)/archive/([a-fA-F0-9]{40})\.zip',url.path)
            if url.scheme!='https' or not match or not re.fullmatch('[a-fA-F0-9]{64}',str(record.get('sha256',''))):continue
            declarations.append(dict(record,repository=match[1],commit=match[2],declaration=item.name))
    active=[];references=[]
    for item in uploads:
        candidates=[r for r in declarations if r.get('bytes')==getattr(item,'path',Path()).stat().st_size] if getattr(item,'path',None) else []
        raw=None
        if str(item.name).lower().endswith('.zip') and declarations:
            if getattr(item,'path',None):source=Path(item.path)
            else:
                raw=item.getvalue();source=io.BytesIO(raw)
                candidates=[r for r in declarations if r.get('bytes')==len(raw)]
            if candidates:
                if raw is None:
                    with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                else:digest=hashlib.sha256(raw).hexdigest()
                candidates=[r for r in candidates if r['sha256'].lower()==digest]
                with zipfile.ZipFile(source) as archive:
                    members=[i for i in archive.infolist() if not i.is_dir()]
                    names=[PurePosixPath(i.filename.replace('\\','/')) for i in members]
                    safe=all(p.parts and not p.is_absolute() and '..' not in p.parts and ':' not in p.parts[0] and '\x00' not in str(p) for p in names)
                    roots={p.parts[0] for p in names}
                    docs={p.name.lower() for p in names if len(p.parts)==2}
                    code=sum(p.suffix.lower() in {'.f90','.f','.for','.py','.c','.cpp','.rs','.js','.ts','.java'} for p in names)
                    runtime=any(p.suffix.lower() in {'.nc','.nc4','.ncf','.h5','.hdf','.tif','.tiff','.sqlite','.db','.pcp','.tem','.cli','.inp','.camx'}
                                or p.name.lower() in {'file.cio','namelist.input'} for p in names)
                    for info,p in zip(members,names):
                        if (info.external_attr>>16)&0o170000!=0o120000:
                            with archive.open(info) as stream:prefix=stream.read(8)
                            runtime |= prefix.startswith((b'CDF',b'\x89HDF\r\n\x1a\n',b'II*\x00',b'MM\x00*',b'SURF.DAT',b'GEO.DAT'))
                        if p.suffix.lower()=='.csv' and (info.external_attr>>16)&0o170000!=0o120000:
                            import csv
                            with archive.open(info) as stream:line=stream.readline(8192)
                            if len(line)==8192 and not line.endswith(b'\n'):runtime=True
                            columns=set(next(csv.reader([line.decode('utf-8-sig',errors='replace')]),[]))
                            runtime |= {'STATION','DATE','TMP'}<=columns or {'station','time_utc','temperature_C'}<=columns
                    witness=next((r for r in candidates if roots=={r['repository']+'-'+r['commit']}),None)
                    if safe and code and not runtime and witness and any(n.startswith('readme') for n in docs) and any(n.startswith(('license','copying')) for n in docs):
                        references.append(dict(file=item.name,role='software_reference',sha256=digest,bytes=witness['bytes'],
                            url=witness['url'],commit=witness['commit'],evidence=witness['declaration'],
                            source_files=code,symbolic_links=sum((i.external_attr>>16)&0o170000==0o120000 for i in members),
                            extracted=False,executed=False,processing_input=False))
                        continue
        active.append(item)
    audit=dict(schema='ecodata.input-roles.v1',references=references,active_inputs=[u.name for u in active],
        source_modified=False,policy='verified source-only repository snapshots remain opaque references; all other archives retain strict safety checks')
    if references:(Path(folder)/'input_roles.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    return active,audit


def reconcile_records(frame, identity, payload):
    """One canonical scientific record, retaining every byte-source reference.

    Only equal payloads for the same provider/element/station/time can merge.
    Conflicting values, QA or report types are never averaged or ranked here.
    """
    import pandas as pd
    if frame[[*identity,*payload]].isna().any().any():raise ValueError('观测身份或科学字段未知，不能归并为已确认观测')
    result=[];groups=[];collapsed=0
    ordered=frame.sort_values([*identity,'source_sha256','source_member','source_row'])
    for key,rows in ordered.groupby(identity,sort=True,dropna=False):
        if any(rows[c].nunique(dropna=False)>1 for c in payload):
            raise ValueError('同一观测身份存在冲突的数值/QA/报告，未平均或选择：'+str(key))
        row=rows.iloc[0].copy()
        origins=rows[['source_member','source_sha256','source_row']].to_dict('records')
        row['source_provenance']=origins
        row['record_id']=hashlib.sha256(json.dumps(list(key),default=str).encode()).hexdigest()
        result.append(row)
        if len(rows)>1:
            collapsed+=len(rows)-1;groups.append(dict(record_id=row['record_id'],origins=origins))
    output=pd.DataFrame(result).reset_index(drop=True) if result else frame.assign(source_provenance=pd.Series(dtype=object),record_id=pd.Series(dtype=str))
    return output,dict(identity=list(identity),payload=list(payload),input_rows=len(frame),output_rows=len(output),
                       collapsed_rows=collapsed,alias_groups=groups,policy='exact scientific record reconciliation, not value averaging')


def header(raw):
    """Inspect native NetCDF headers without loading numerical fields."""
    from netCDF4 import Dataset
    if isinstance(raw,memoryview):raw=raw.tobytes()
    if not (raw.startswith(b'CDF') or raw.startswith(b'\x89HDF\r\n\x1a\n')):
        return None
    with Dataset('source_header.nc', memory=raw) as ds:
        def plain(value):
            return value.tolist() if hasattr(value, 'tolist') else value
        return dict(reader='netCDF4', dimensions={k: len(v) for k,v in ds.dimensions.items()},
            attributes={k: plain(ds.getncattr(k)) for k in ds.ncattrs()},
            variables={k: dict(dimensions=list(v.dimensions),
                              dtype=str(v.dtype),
                              attributes={a: plain(v.getncattr(a)) for a in v.ncattrs()})
                       for k,v in ds.variables.items()})


def cf_evidence(facts):
    """Variable-specific mapping evidence: never apply the first CRS globally."""
    from pyproj import CRS
    evidence = {}
    variables = facts['variables']
    for name,item in variables.items():
        attrs = item['attributes']
        mapping = attrs.get('grid_mapping')
        if not mapping:
            continue
        # CF's expanded syntax may name several mappings; do not guess one.
        tokens = str(mapping).split()
        row = dict(grid_mapping=mapping, dimensions=item['dimensions'])
        if len(tokens) != 1 or ':' in str(mapping):
            row['issue'] = '多个/扩展坐标映射需按变量解释，未猜选坐标系'
        elif tokens[0] not in variables:
            row['issue'] = '变量引用的grid_mapping不存在：'+tokens[0]
        else:
            try:
                row['crs'] = CRS.from_cf(variables[tokens[0]]['attributes']).to_wkt()
            except (ValueError, KeyError) as exc:
                row['issue'] = str(exc)
        evidence[name] = row
    return evidence


def build_catalog(files, model_id, converter=None):
    """Uniform catalog; native claims come only from the installed converter."""
    from ecodata.models.converter_contract import load_converter
    converter = converter or load_converter(model_id)
    records, seen = [], set()
    for filename,raw in files:
        name = str(filename).replace('\\','/')
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or ':' in p.parts[0] or name.casefold() in seen:
            raise ValueError('资料路径不安全或重复：'+name)
        seen.add(name.casefold())
        facts = header(raw)
        row = dict(file=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                   role='unknown', evidence=[], reader=None)
        if facts:
            row.update(reader=facts['reader'], facts=facts, cf_mappings=cf_evidence(facts))
            from ecodata.models.scientific_schema import describe_variables
            row['scientific_variables'] = describe_variables(facts)
        records.append(row)
    hook = getattr(converter, 'inspect_sources', None)
    inspection = hook(files, records) if hook else dict(claims=[], issues=[],
        scope='转换器未提供原生工程识别插件；仅识别普通数据')
    by_name = {r['file']:r for r in records}
    claimed = set()
    for claim in inspection.get('claims', []):
        name = claim['file']
        if name not in by_name or name in claimed or claim.get('role') not in ROLES:
            raise ValueError('工程识别插件返回重复、越界或未知角色：'+name)
        claimed.add(name)
        by_name[name].update({k:v for k,v in claim.items() if k not in {'sha256','bytes','facts','cf_mappings'}})
    for row in records:
        if row['role'] != 'unknown':
            continue
        name = row['file']; suffix = PurePosixPath(name).suffix.lower()
        if row['reader'] or suffix in {'.csv','.tsv','.rdb','.xlsx','.parquet','.tif','.tiff','.h5','.isd','.gz'}:
            row.update(role='data_candidate', evidence=['文件结构可由平台已有数据读取器检查；变量用途仍须匹配模型契约'])
        elif suffix in {'.md'} or PurePosixPath(name).name.lower().startswith(('readme','license')):
            row.update(role='documentation', evidence=['说明文档只保存来源，不作为执行指令或科学输入'])
        elif suffix=='.json':
            try:
                document=json.loads(dict(files)[name])
                allowed={'project_name','start','end','station','target_model_version'}
                if (isinstance(document,dict) and {'start','end','station'}<=set(document)
                        and not set(document)-allowed and isinstance(document['station'],dict)
                        and {'latitude','longitude'}<=set(document['station'])):
                    from datetime import datetime
                    from math import isfinite
                    start,end=datetime.fromisoformat(document['start']),datetime.fromisoformat(document['end'])
                    lat,lon=float(document['station']['latitude']),float(document['station']['longitude'])
                    if not (start<=end and isfinite(lat) and isfinite(lon) and -90<=lat<=90 and -180<=lon<=180):
                        raise ValueError('站点/日期元信息无效')
                    row.update(role='source_metadata',declared_metadata=document,
                        evidence=['已有站点/日期元信息结构；只记录声明，不覆盖数据坐标、工程版本或转换器参数'])
            except (ValueError,TypeError,KeyError,UnicodeError):
                pass  # Unrecognised JSON remains explicit UNKNOWN below.
    issues = list(inspection.get('issues', []))
    for row in records:
        for variable,mapping in row.get('cf_mappings',{}).items():
            if mapping.get('issue'):
                issues.append(dict(file=row['file'],message=variable+'：'+mapping['issue']))
        if row['role'] == 'unknown':
            issues.append(dict(file=row['file'], message='不能确定该文件用途，不会丢弃或混入融合'))
    from ecodata.models.scientific_schema import relationships
    return dict(schema='ecodata.source-catalog.v1', model_id=model_id, files=records,
        scientific_relationships=relationships(records),
        issues=issues, native_inspection=inspection.get('scope'),
        status='BLOCK' if issues else 'IDENTIFIED', native_run_validated=False)


def processing_plan(catalog, candidate_names, relationships=()):
    """Separate scientific source eligibility from mere file readability."""
    requested = set(candidate_names)
    known = {r['file']:r for r in catalog['files']}
    if requested - set(known):
        raise ValueError('处理来源没有资料清单证据：'+str(sorted(requested-set(known))))
    eligible, excluded = [], []
    for name in candidate_names:
        row=known[name]
        if row['role']=='data_candidate':
            eligible.append(name)
        else:
            excluded.append(dict(file=name,role=row['role'],reason='工程状态与参考结果不是自动融合的独立新来源'))
    if not eligible:
        raise ValueError('未识别到可处理的新资料；工程文件本身不等于新观测')
    return dict(schema='ecodata.processing-decision.v1', sources=eligible,
        excluded=excluded, relationships=list(relationships),
        decision='ASSESS_SINGLE_SOURCE' if len(eligible)==1 else 'ALIGN_AND_SEARCH',
        algorithm_search_executed=False,
        reason='按文件内容、工程引用与物理用途判断，不按模型名或压缩包名强制融合')


def single_source_action(dataset, variable, request, contract):
    from ecodata.models.alignment_bridge import can_keep_unchanged
    if can_keep_unchanged(dataset,variable,request,contract):
        return dict(status='PRESERVE_SINGLE_SOURCE',algorithm_search_executed=False,
            reason='只有一个独立来源且已满足目标；保留原有信息和明确允许的缺测掩码，仍须转换前校验')
    return dict(status='ALIGN_AND_SEARCH',algorithm_search_executed=False,
        reason='单源尚不满足目标，交给共用对齐与既有处理引擎；不承诺缺信息也能重建')


def write_catalog(folder, catalog):
    (folder/'source_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2,default=str),encoding='utf8')
