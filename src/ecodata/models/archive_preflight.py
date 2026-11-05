"""Bounded archive probing before any eager expansion of multi-GB inputs.

Only trusted converter hooks interpret these prefixes. No uploaded code is run.
Passing this probe does not certify complete contents, CRC, or native readiness.
"""
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
import io
import json
import hashlib
import zipfile


def inspect_upload_archives(uploads, converter, folder, *, native_required=True):
    reports=[];issues=[];native_issues=[];total=0;scientific=[];unread=[]
    from ecodata.models.archive_inputs import deployment_limit
    limit=deployment_limit()
    probe=getattr(converter,'probe_sources',None)
    for upload in uploads:
        if not str(upload.name).lower().endswith('.zip'):continue
        with ExitStack() as stack:
            # LocalUpload exposes a read-only path: never load a 2.5GB ZIP just
            # to obtain its directory and small control-file prefixes.
            path=getattr(upload,'path',None)
            source=Path(path) if path is not None else io.BytesIO(upload.getvalue())
            archive=stack.enter_context(zipfile.ZipFile(source))
            members=[];prefixes=[];seen=set()
            for item in archive.infolist():
                if item.is_dir():continue
                name=item.filename.replace('\\','/');p=PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts or ':' in p.parts[0] or '\x00' in name or (item.external_attr>>16)&0o170000==0o120000:
                    issues.append(dict(file=name,message='压缩包路径不安全'));continue
                if name.casefold() in seen:
                    issues.append(dict(file=name,message='压缩包路径重复'));continue
                seen.add(name.casefold());total+=item.file_size
                members.append(dict(file=name,bytes=item.file_size))
                if total>limit:
                    issues.append(dict(file=upload.name,message=f'ZIP声明展开总量{total}字节超过部署接收配额{limit}字节'))
                    break
                with archive.open(item) as stream:
                    prefix=stream.read(65536)
                    if probe:prefixes.append((name,prefix))
                    # Most classic NetCDF headers fit in 64 KiB even when the
                    # arrays occupy GB. A header read is NOT a payload/CRC test.
                    if prefix.startswith((b'CDF',b'\x89HDF\r\n\x1a\n')):
                        from ecodata.models.source_catalog import header, cf_evidence
                        from ecodata.models.scientific_schema import describe_variables
                        raw=prefix
                        if prefix.startswith(b'\x89HDF') and item.file_size<=8*1024*1024:
                            raw+=stream.read()
                        try:
                            try:facts=header(raw)
                            except (ValueError,OSError,RuntimeError):
                                if not raw.startswith(b'CDF'):raise
                                raw+=stream.read(max(0,1024*1024-len(raw)))
                                facts=header(raw)
                        except (ValueError,OSError,RuntimeError) as exc:
                            unread.append(dict(file=name,archive=upload.name,
                                reason='有限内存检查未取得完整头；不认定资料无效：'+str(exc)))
                            continue
                        scientific.append(dict(file=name,archive=upload.name,bytes=item.file_size,
                            inspected_bytes=len(raw),complete_payload_read=len(raw)==item.file_size,
                            inspected_bytes_sha256=hashlib.sha256(raw).hexdigest(),
                            role='not_classified',facts=facts,cf_mappings=cf_evidence(facts),
                            scientific_variables=describe_variables(facts)))
            result=probe(prefixes) if probe else dict(issues=[])
            native_issues.extend(result.get('issues',[]))
            reports.append(dict(file=upload.name,members=members,native_probe=result))
    if native_required:issues.extend(native_issues)
    record=dict(schema='ecodata.archive-preflight.v1',status='BLOCK' if issues else 'CONTINUE',
        native_contract_issues=native_issues,native_contract_required=native_required,
        archives=reports,unpacked_bytes=total,limit_bytes=limit,issues=issues,full_CRC_verified=False,
        scope='目录/控制前缀与有限科学头检查；不代替完整读取、科学检查或模型运行')
    if reports:
        from ecodata.models.scientific_schema import relationships
        audit=dict(schema='ecodata.archive-scientific-inventory.v1',files=scientific,
            relationships=relationships(scientific),deferred_header_reads=unread,
            source_modified=False,model_executed=False,fusion_executed=False,
            full_payloads_verified=False,
            scope='通用变量/单位/辅助字段/时间支持盘点；至多1MiB经典NetCDF头或最多8MiB小型HDF；不是完整工程验收')
        (folder/'scientific_inventory.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    if reports:
        (folder/'archive_preflight.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    if issues:
        raise ValueError('工程读取前检查未通过：'+'；'.join(i['file']+'：'+i['message'] for i in issues))
    return record
