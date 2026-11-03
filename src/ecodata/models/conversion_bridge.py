"""Versioned handoff, converter-owned requirements, provenance and download."""
from pathlib import Path
import hashlib
import json
import zipfile
from .converter_contract import (load_converter, converter_requirements,
                                 describe_product, preflight, digest)
from . import converter_contract as _contracts


def entry_points(**kwargs):
    # Retain the original bridge's public discovery seam for existing tests.
    return _contracts.entry_points(**kwargs)


def finish_preparation(folder, model_id, products, receipt):
    folder=Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    tasks=receipt.get('tasks', [])
    if len(products) != len(tasks):
        raise ValueError('本次处理产品与任务数量不一致，不能按猜测交接')
    from .fusion_assessment import assess_preparation
    assessment=assess_preparation(folder,receipt,products)
    assessment_path=folder/'fusion_assessment.json'
    assessment_path.write_text(json.dumps(assessment,ensure_ascii=False,indent=2),encoding='utf8')
    receipt=dict(receipt,fusion_assessment=assessment)
    decision=dict(receipt.get('fusion_decision',{}))
    decision['processing_status']=decision.get('status','UNKNOWN')
    if decision['processing_status'] not in {'PRESERVE_SINGLE_SOURCE','NOT_APPLICABLE_SAME_PRODUCT'}:
        decision['status']=assessment['outcome']
    decision['accuracy_improvement']='NOT_ESTABLISHED'
    receipt['fusion_decision']=decision
    (folder/'preparation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    manifest=dict(protocol='ecodata.converter.v1', contract_version=2, model_id=model_id,
        status='AWAITING_CONVERTER', native_ready=False,
        products=[describe_product(folder, product, task) for product, task in zip(products, tasks)],
        preparation=receipt)
    path=folder/'handoff.json'; outputs=[]; reports=[assessment_path]
    def save():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    save()
    try:
        converter=load_converter(model_id, finder=entry_points)
        if converter:
            route_provider=getattr(converter, 'requirements_for_manifest', None)
            contract=route_provider(manifest) if route_provider else None
            if contract is None:
                contract=converter_requirements(model_id, converter=converter)
            if contract and (contract.get('model_id')!=model_id or contract.get('schema')!='ecodata.converter.requirements.v1'):
                raise ValueError('本次转换契约与所选模型不符')
            if contract:
                manifest['requirements']=contract
                save()
                report=preflight(path, contract)
                report_path=folder/'requirements_check.json'
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf8')
                reports.append(report_path)
                manifest['requirements_check']=report
                if not report['passed']:
                    raise ValueError('工程接入缺项或冲突：'+'；'.join(item['message'] for item in report['missing_or_conflicting']))
            save()  # The converter must read the manifest for THIS preparation.
            converter_input_hash=digest(path)
            snapshot=folder/'handoff_before_conversion.json'
            snapshot.write_bytes(path.read_bytes()); reports.append(snapshot)
            target=folder/'converted'; target.mkdir(exist_ok=True)
            result=converter(path, target)
            if digest(path) != converter_input_hash:
                raise ValueError('转换器改写了交接清单，拒绝交付')
            for item in manifest['products']:
                if digest(folder/item['path']) != item['sha256']:
                    raise ValueError('转换器改写了融合产品，拒绝交付')
            if result.get('status') != 'PASS' or not result.get('validation') or not result.get('files'):
                raise ValueError('转换器未提供合格文件及校验记录')
            inventory=[]
            for name in result['files']:
                relative=Path(name); item=(target/relative).resolve()
                if relative.is_absolute() or not item.is_relative_to(target) or not item.is_file() or not item.stat().st_size:
                    raise ValueError('转换器返回的文件不存在、为空或超出输出目录')
                outputs.append(item)
                inventory.append(dict(path=item.relative_to(folder).as_posix(), size=item.stat().st_size, sha256=digest(item)))
            manifest.update(status='CONVERTED', native_ready=True, conversion=result,
                            converter_input_manifest_sha256=converter_input_hash, output_inventory=inventory)
    except Exception as exc:
        outputs=[]
        manifest.update(status='CONVERSION_FAILED', native_ready=False, conversion_error=str(exc))
    save()
    archive=folder/'EcoData-output.zip'
    members=[path, *map(Path, products), *reports, *outputs]
    for dependency in receipt.get('handoff_dependencies', []):
        relative=Path(dependency['path']); item=(folder/relative).resolve()
        if relative.is_absolute() or not item.is_relative_to(folder) or not item.is_file() or digest(item)!=dependency['sha256']:
            raise ValueError('本次上传工程/原始驱动依赖缺失、越界或哈希变化')
        members.append(item)
    inventory=folder/'bundle_inventory.json'
    if inventory.is_file():members.append(inventory)
    preparation=folder/'preparation.json'
    if preparation.is_file():
        members.append(preparation)
    for report_name in ('source_catalog.json','automatic_plan.json','automatic_fusion_attempt.json','archive_preflight.json',
                        'scientific_inventory.json','field_matching.json'):
        report_path=folder/report_name
        if report_path.is_file():members.append(report_path)
    # Preserve failure diagnostics too, explicitly not accepted native outputs.
    target=folder/'converted'
    if manifest['status']=='CONVERSION_FAILED' and target.is_dir():
        members.extend(p for p in target.iterdir() if p.is_file() and p.suffix in {'.json','.log'})
    unique={item.resolve().relative_to(folder).as_posix():item for item in members}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name,item in unique.items():
            bundle.write(item,name)
    # Verify the final downloadable archive independently of writer return values.
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('最终下载包 CRC 校验失败')
        for name,item in unique.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest(item):
                raise ValueError('最终下载包内容与当前产物不一致：'+name)
    return dict(manifest=manifest,archive=str(archive))
