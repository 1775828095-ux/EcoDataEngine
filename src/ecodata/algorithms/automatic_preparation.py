"""Upload + reviewed model contract -> existing public numerical processing."""
from pathlib import Path
import json


def prepare_uploads(uploads, model_id, folder, progress=None, *, task=None, request_builder=None):
    """Shared input roles apply to page, toolkit and research callers alike."""
    from ecodata.models.source_catalog import partition_uploads
    import hashlib,zipfile
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    active,roles=partition_uploads(uploads,folder)
    from ecodata.models.archive_inputs import normalize_archives
    active,intake=normalize_archives(active,folder)
    result=_prepare_uploads(active,model_id,folder,progress,task=task,request_builder=request_builder)
    if intake:
        with zipfile.ZipFile(result['archive'],'a',zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder/'archive_intake.json','_ecodata/archive_intake.json')
        result['archive_intake']=intake
    if roles['references']:
        raw=json.dumps(roles,ensure_ascii=False,indent=2).encode('utf8')
        name='_ecodata/input_roles_'+hashlib.sha256(raw).hexdigest()[:16]+'.json'
        with zipfile.ZipFile(result['archive'],'a',zipfile.ZIP_DEFLATED) as archive:
            if name in archive.namelist():raise ValueError('输入角色核验路径冲突，未覆盖已有产物')
            archive.writestr(name,raw)
        result['input_roles']=roles
    return result


def _prepare_uploads(uploads, model_id, folder, progress=None, *, task=None, request_builder=None):
    # Lazy scientific imports keep the input screen responsive.
    import numpy as np
    from ecodata.recognition import recognise_file
    from ecodata.interface.engineering_upload import scan_model_uploads
    from ecodata.models.scientific_schema import match_fields
    from ecodata.engines.units import UnitEngine
    from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate
    from ecodata.models.conversion_bridge import finish_preparation
    from ecodata.models.converter_contract import preparation_table, coordinate_references

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    from ecodata.models.converter_contract import load_converter
    converter = load_converter(model_id)
    from ecodata.models.archive_preflight import inspect_upload_archives
    archive_probe = inspect_upload_archives(uploads,converter,folder,native_required=False)
    if model_id=='wrf':
        from ecodata.models.wrfda_inputs import prepare_wrfda_inputs
        wrfda=prepare_wrfda_inputs(uploads,folder,progress=progress)
        if wrfda is not None:return wrfda
    if task is None:
        from ecodata.models.engineering.preparation import prepare_engineering_inputs
        prepared = prepare_engineering_inputs(uploads, model_id, folder, progress=progress, request_builder=request_builder)
        if prepared is not None:
            return prepared
    from ecodata.models.component_service import prepare_components
    component = prepare_components(uploads, model_id, folder, progress=progress, task=task)
    if component is not None:
        return component
    from ecodata.models.task_dispatch import dispatch_preparation
    dispatched = dispatch_preparation(uploads, model_id, folder, progress=progress, task=task, request_builder=request_builder)
    if dispatched is not None:
        return dispatched
    from ecodata.models.swat_editor_project import prepare_editor_project
    edited = prepare_editor_project(uploads, model_id, folder, progress=progress)
    if edited is not None:
        return edited
    # A collection of inner project ZIPs is not the project they contain. Do
    # not let a legacy converter substitute a machine-local default case.
    for archive in archive_probe['archives']:
        members = [m['file'].lower() for m in archive['members']]
        if any(n.endswith('.zip') for n in members) and all(
                n.endswith(('.zip', '.csv', '.txt', '.md', '.json')) for n in members):
            raise ValueError('尚未展开嵌套工程ZIP：请分别上传内层工程ZIP及其观测文件；'
                             '不会使用本机默认工程替代。来源：'+str(archive['file']))
    # Native-format claims still require native checks; data-only tasks above
    # cannot execute unknown scripts or claim a complete native engineering run.
    from ecodata.models.project_inventory import prepare_project_inventory
    # The old converter probe rejects all C-shell syntax. The public bounded
    # binder validates supported declarations before that legacy fallback.
    if model_id in {'cmaq','camx'}:
        from ecodata.models.project_binding import prepare_project_binding
        bound=prepare_project_binding(uploads,model_id,folder,progress=progress)
        if bound is not None:return bound
    try:
        inspect_upload_archives(uploads,converter,folder,native_required=True)
    except ValueError:
        # A read-only dependency report is allowed even when native delivery
        # cannot be certified. It never becomes CONVERTED/native_ready.
        inspected = prepare_project_inventory(uploads, model_id, folder)
        if inspected is not None:
            return inspected
        raise
    from ecodata.models.project_binding import prepare_project_binding
    bound = prepare_project_binding(uploads, model_id, folder, progress=progress)
    if bound is not None:
        return bound
    prepare_bundle = getattr(converter, 'prepare_uploads', None)
    if prepare_bundle:
        prepared = prepare_bundle(uploads, folder, progress=progress)
        if prepared is not None:
            return prepared
    inspected = prepare_project_inventory(uploads, model_id, folder)
    if inspected is not None:
        return inspected
    from ecodata.algorithms.profile_matching import model_table
    from ecodata.algorithms.platform_search import SearchInput, process_current_input
    table = preparation_table(model_id, model_table(model_id))
    contract = table.get("converter_contract")
    scan = scan_model_uploads(uploads, include_all=True)
    if scan.ignored:
        raise ValueError('存在无法读取或不安全的上传资料，未跳过：'+'；'.join(i['file']+'：'+i['reason'] for i in scan.ignored))
    if not scan.files:
        raise ValueError("未找到可读取的数据，请上传 NetCDF、GeoTIFF、CSV 等数据文件或数据 ZIP。")
    from ecodata.models.source_catalog import build_catalog, write_catalog, single_source_action
    catalog=build_catalog(scan.files,model_id,converter)
    write_catalog(folder,catalog)
    if catalog['issues']:
        raise ValueError('资料识别发现缺项：'+'；'.join(i['file']+'：'+i['message'] for i in catalog['issues']))
    source_roles={r['file']:r['role'] for r in catalog['files']}
    if any(role=='engineering_configuration' for role in source_roles.values()):
        # Recognising a project is not permission to substitute the configured
        # watershed/grid. Until a writer supports uploaded-project binding,
        # accept only an exact copy of its existing, pinned dependency.
        import hashlib
        supplied={hashlib.sha256(u.getvalue()).hexdigest() for u in uploads}
        supplied.update(r['sha256'] for r in catalog['files'])
        pinned={d.get('sha256') for d in (contract or {}).get('dependencies',[]) if d.get('name')=='engineering'}-{None}
        if not supplied.intersection(pinned):
            raise ValueError('已识别用户上传的工程，但它不是当前转换器绑定的工程；该写出器尚未支持自动绑定此工程。不会换用其他流域或网格。详见source_catalog.json。')
    groups, inventory, unmatched = {}, [], []
    if progress:
        progress("识别变量、单位、时间与空间关系")
    for filename, content in scan.files:
        if source_roles[filename]!='data_candidate':
            inventory.append(dict(filename=filename,role=source_roles[filename],processing='保留工程/参考资料证据，不作为独立融合来源'))
            continue
        ds, info = recognise_file(filename, content)
        inventory.append(dict(filename=filename, source_sha256=__import__("hashlib").sha256(content).hexdigest(), descriptor=info.as_dict()))
        ds = coordinate_references(ds, contract)
        role_mappings = []
        match_reports = []
        for role in table['roles']:
            if role['record_only']:
                continue
            mapping, reports = match_fields(ds, role['fields'])
            role_mappings.append((role, mapping))
            match_reports.extend(dict(role=role['role'], **r) for r in reports)
        inventory[-1]['field_matching'] = match_reports
        (folder/'field_matching.json').write_text(json.dumps(inventory, ensure_ascii=False,
            indent=2, default=str), encoding='utf8')
        invalid_units = [candidate for report in match_reports for candidate in report['rejected']]
        if invalid_units:
            raise ValueError(filename+' 已识别物理量但单位检查未通过：'+'；'.join(
                c['source']+'：'+c['reason'] for c in invalid_units)+'；详见field_matching.json')
        if any(r['status']=='AMBIGUOUS' for r in match_reports):
            raise ValueError(filename+' 存在多个同义字段，未猜选；详见field_matching.json')
        matched = False
        for name in info.variables:
            if name not in ds.data_vars or ds[name].dtype.kind not in "iuf":
                continue
            choices = []
            for role, mapping in role_mappings:
                for field in role["fields"]:
                    if mapping.get(field['name']) == name:
                        choices.append((role, field))
            if not choices:
                continue
            declared = ds.attrs.get("model_role")
            if declared:
                choices = [(r, f) for r, f in choices if r["role"] == declared]
            if len(choices) != 1:
                raise ValueError(f"{filename} 的 {name} 对应多个模型用途。请在数据元信息中注明 model_role，避免把观测误作边界条件。")
            role, field = choices[0]
            if field.get("data_type", "continuous") != "continuous":
                raise ValueError(f"{name} 是分类变量，当前自动重建入口不能用连续插值处理；需保留分类处理契约。")
            if (not contract or contract.get("spatial_gate", True)) and (not info.crs or not info.x_coordinate or not info.y_coordinate):
                missing = []
                if not info.crs:
                    missing.append("坐标系未声明，不能仅凭经纬度猜测")
                if not info.x_coordinate or not info.y_coordinate:
                    location = info.location_metadata or {}
                    if location.get("complete"):
                        missing.append("已读取资料属性中的经纬度，但未明确这些位置适用于哪些数据；不能自动套用到所有变量")
                    else:
                        missing.append("缺少可确定的数据坐标")
                raise ValueError(f"{filename}：" + "；".join(missing))
            converted = UnitEngine().convert(ds[name], field["unit"])
            selected = ds.copy()
            selected[name] = converted
            # Preserve time bounds and source evidence, never manufacture them.
            if name != field["name"]:
                selected = selected.rename({name: field["name"]})
            key = (role["role"], field["name"])
            groups.setdefault(key, []).append((filename, selected, field))
            matched = True
        if not matched:
            unmatched.append(filename)
    if not groups:
        raise ValueError("数据中未识别到绑定转换场景的输入变量。需要：" + "、".join(f["name"] for r in table["roles"] for f in r["fields"]))
    if unmatched:
        raise ValueError("部分文件的物理用途尚不明确，未自动丢弃或融合：" + "、".join(unmatched))
    if contract:
        from ecodata.models.preparation_bridge import inventory_report
        plan=inventory_report(model_id,groups)
        (folder/'automatic_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        if plan['missing']:
            raise ValueError('自动检查发现缺项：'+'；'.join(i['message'] for i in plan['missing']))
    products, tasks = [], []
    for index, ((role, variable), sources) in enumerate(groups.items()):
        task_folder = folder / f"field-{index + 1}"
        task_folder.mkdir()
        field = sources[0][2]
        # Stable source order; no primary choice depends on browser upload order.
        sources.sort(key=lambda s: (str(s[1].attrs.get("source_id", "")), s[0]))
        request = dict(task="multisource" if len(sources) > 1 else "spatial",
                       variable=variable, unit=field["unit"], semantic_type="continuous",
                       model_profile=dict(id=model_id, role=role, field=variable),
                       soft_guidance=False, mutation_control="random", allow_training=True)
        if len(sources) > 1:
            request["processing_goal"] = "spatial_fusion"
        if contract:
            from ecodata.models.preparation_bridge import build_request, source_report
            request, target = build_request(model_id, role, variable, len(sources))
            report = source_report([(name, ds) for name,ds,_ in sources], variable, target)
            (task_folder/'automatic_plan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        search_inputs = []
        for source_index, (filename, ds, _) in enumerate(sources):
            local = task_folder / f"source-{source_index + 1}.nc"
            checks = RawInputQualityGate().validate(ds, variable=variable, requires_spatial=not contract or contract.get("spatial_gate", True), expected_unit=field["unit"])
            if not checks["passed"]:
                raise ValueError(f"{filename} 未通过输入质量检查：" + "; ".join(c["message"] for c in checks["checks"] if c["status"] == "BLOCK"))
            ds.to_netcdf(local, engine="h5netcdf")
            search_inputs.append(SearchInput(local.name, "primary" if source_index == 0 else "background", local.read_bytes()))
        if progress:
            progress(f"自动处理 {variable} · 检查质量并搜索可执行组合")
        single = sources[0][1]
        from ecodata.models.alignment_bridge import can_keep_unchanged
        action=single_source_action(single,variable,request,contract) if len(sources)==1 else dict(status='ALIGN_AND_SEARCH',algorithm_search_executed=False)
        if action['status']=='PRESERVE_SINGLE_SOURCE':
            # Complete inputs need no invented fusion or degradation.
            product = task_folder / "product.nc"
            single.to_netcdf(product, engine="h5netcdf")
            result = dict(status="UNCHANGED_COMPLETE_INPUT", product=str(product),
                          reason="单源数据完整，保留观测及原有分辨率", test_used_for_selection=False)
            result["quality"] = checks
            output_checks = ResultQualityGate().validate(single, single, variable=variable, expected_output_unit=field["unit"])
            if not output_checks["passed"]:
                raise ValueError(f"{variable} 未通过输出质量检查")
            result["output_quality"] = output_checks
        else:
            if contract and not contract.get("spatial_gate", True):
                raise ValueError("原生索引网格/专用坐标的多源或缺失重建需要明确的空间处理契约；不会把行列编号当物理距离融合。")
            from ecodata.models.alignment_bridge import align_inputs
            search_inputs, alignment = align_inputs(search_inputs, request, task_folder/'alignment')
            request=dict(request,observation_crs=alignment['observation_crs'])
            if request_builder is not None:
                request = request_builder(request)
            result = process_current_input(search_inputs, request, task_folder / "search", progress=progress)
            result['alignment']=alignment
            action['algorithm_search_executed']=bool(result.get('counts',{}).get('evaluated',0))
            if not result.get("product"):
                raise ValueError(f"{variable} 暂未生成合格结果：{result.get('reason', result.get('status'))}")
        products.append(Path(result["product"]))
        result['source_decision']=action
        tasks.append(dict(role=role, field=variable, unit=field["unit"], result=result))
    receipt = dict(inventory=inventory, source_catalog=catalog, tasks=tasks, ignored=scan.ignored,
                   duplicates=scan.duplicates, full_model_ready=False,
                   scope="已识别字段的数据准备；完整工程依赖及原生格式由转换器校验")
    receipt['fusion_decision']=dict(
        status='PRESERVE_SINGLE_SOURCE' if all(t['result']['source_decision']['status']=='PRESERVE_SINGLE_SOURCE' for t in tasks) else 'PLATFORM_ENGINE_RETURNED',
        algorithm_search_executed=any(t['result']['source_decision'].get('algorithm_search_executed') for t in tasks))
    (folder / "preparation.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if progress:
        progress("数据处理完成，检查模型格式转化模块")
    return finish_preparation(folder, model_id, products, receipt)
