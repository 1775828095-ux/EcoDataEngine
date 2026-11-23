"""Automatic first-stage discovery and a local gate for model suggestions."""
from collections import defaultdict
import json
import re

from ecodata.models.scientific_schema import compare_units, identity
from ecodata.core.exceptions import EcoDataError
from .scan import assert_current, kind
from .engineering_excerpt import associate_documents
from .service import field_type, prepare
from ecodata.models.task_dispatch import OBSERVATION_COLUMNS

PHYSICAL = {
    "precipitation": {"cmorph","lwe_precipitation_rate","precipitation_corrected","precip","precipitation","prectotcorr","rain","rainfall","precipitation_flux","precipitation_amount","lwe_thickness_of_precipitation_amount"},
    "air_temperature": {"airtemperature","temperature","t2m","t2","tas"},
    "no2": {"no2","nitrogendioxide","molefractionofnitrogendioxideinair"},
    "leaf_area_index": {"lai","lai_500m","leaf_area_index","leaf_area_index_from_px_lsm"},
    "pm25": {"pm25","massconcentrationofpm25ambientaerosolinair"},
}
def norm(value):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value.casefold())

def physical(field):
    names={norm(field[k]) for k in ("name","standard_name") if field.get(k)}
    for key,aliases in PHYSICAL.items():
        if names & {norm(a) for a in aliases}: return key
    return ""

def resolution(row):
    axes=row["resolution"]
    x=next((axes[k] for k in ("lon","longitude","x") if k in axes),None)
    y=next((axes[k] for k in ("lat","latitude","y") if k in axes),None)
    interval=axes.get("time_seconds")
    if interval is None and row.get("declared_time_resolution")=="P1D" and row.get("time_cell_seconds") in {86399,86400}:
        interval=86400.0
    return interval,(x,y) if x and y else axes.get("pixel")

# Public model input roles, from the teammate's preparation profiles.
MODELS={"data_only":"仅数据处理（融合/对齐，不生成模型工程）","swat_plus":"SWAT+","wrf":"WRF","cmaq":"CMAQ","camx":"CAMx","calpuff":"CALPUFF"}
MODEL_USES={
    "swat_plus":{"precipitation":"气象驱动：降水","air_temperature":"气象驱动：气温"},
    "wrf":{"air_temperature":"气象分析或初始及边界资料：气温，具体类别待核验"},
    "cmaq":{"leaf_area_index":"地表植被参数：比较卫星与 MCIP 叶面积指数的支持范围和时间口径","pm25":"PM2.5 背景场或观测约束，按工程角色确认","no2":"化学浓度候选，核对机制及初始或边界输入"},
    "camx":{"no2":"化学浓度候选，核对机制及初始或边界输入"},
    "calpuff":{"air_temperature":"气象预处理候选，需核对 CALMET 输入"},
}
PUBLIC_UNITS={"1","m2 m-2","m^2/m^2","mm/day","mm","mm d-1","kg m-2 s-1","K","degC","°C","ppb","ppm","ug/m3","µg/m³"}
# Public I/O API field descriptions verified against the variable's own header.
IOAPI_DESCRIPTIONS={
    ("MSFX2","map-scale factor squared"):("map_scale_factor_squared","地图投影比例因子的平方"),
    ("HT","terrain elevation"):("terrain_elevation","地形高程"),
    ("DLUSE","dominant land use category nlcd40"):("dominant_land_use_category","主要土地利用类别（NLCD40）"),
    ("LWMASK","land-water mask (1=land, 0=water)"):("land_water_mask","陆水掩膜（1 为陆地，0 为水面）"),
    ("PURB","urban percent of cell based on land"):("urban_land_percentage","网格陆地面积中的城市用地百分比"),
}
GRID_ROLES={"boundary":"边界网格（PERIM）","interior":"内部网格（ROW × COL）"}


def field_identity(row, field):
    if field.get("observation_contract"):
        return OBSERVATION_COLUMNS[field["name"]][0], "declared_observation_schema"
    quantity, basis = identity(field["name"],field)
    if quantity:
        # Only normalize explicit aliases of the same rate. Amount and flux
        # still need their own quantity and temporal checks.
        if basis in {"standard_name","physical_variable"} and norm(quantity) in {
            "lweprecipitationrate","precipitationcorrected","prectotcorr"
        }:
            quantity="precipitation"
        return quantity, basis
    known=IOAPI_DESCRIPTIONS.get((field["name"],field.get("var_desc","").casefold()))
    if row.get("ioapi_grid_role") and known:
        return known[0], "verified_ioapi_var_desc"
    return None, None


def pair_units(ar, a, br, b, inferred_quantity=None):
    qa, ba = field_identity(ar,a)
    qb, bb = field_identity(br,b)
    trusted={"standard_name","physical_variable","long_name","verified_ioapi_var_desc","declared_observation_schema"}
    if qa==qb and qa and {ba,bb}<=(trusted|{"variable_name"}) and {ba,bb}&trusted:
        return compare_units(qa,a["unit"],qb,b["unit"])
    if qa and qb and qa!=qb and {ba,bb}<=trusted:
        return dict(compare_units(qa,a["unit"],qb,b["unit"]),status="quantity_conflict")
    if inferred_quantity:
        return dict(compare_units(inferred_quantity,a["unit"],inferred_quantity,b["unit"]),
                    identity_basis="deepseek")
    # An explicitly trusted product audit may independently establish identity later.
    return compare_units(None,a["unit"],None,b["unit"])


def public_unit(value):
    if value in PUBLIC_UNITS:
        return value
    from ecodata.core.exceptions import SchemaError
    from ecodata.engines.units import UnitEngine, normalize_unit
    value = normalize_unit(value)
    if not re.fullmatch(r"[A-Za-z0-9°µμ%*/^(). +_-]{1,40}", value):
        return None
    try:
        return f"{UnitEngine.parse(value):~}"
    except (SchemaError, ValueError, TypeError):
        return None


def public_unit_report(report):
    safe = dict(report,source_unit=public_unit(report["source_unit"]),
                target_unit=public_unit(report["target_unit"]))
    if safe["quantity"] and not re.fullmatch(r"[a-z][a-z0-9_]{0,90}",safe["quantity"]):
        safe["quantity"]=None
    return safe
ROLE_LABELS={"observations":"站点观测","background":"背景场","target":"目标网格","forcing":"气象驱动","validation":"独立验证","comparison":"对照来源"}

def spacing_comparison(a,b):
    if a is None or b is None:
        return "unknown"
    if isinstance(a,tuple) and isinstance(b,tuple):
        if any(not x["unit"] or x["unit"]!=y["unit"] for x,y in zip(a,b)):
            return "not_comparable"
    elif isinstance(a,dict) and isinstance(b,dict):
        if not a.get("crs") or a["crs"]!=b.get("crs"):
            return "not_comparable"
    elif type(a)!=type(b):
        return "not_comparable"
    from math import isclose
    if isinstance(a,tuple) and isinstance(b,tuple):
        return "same" if all(isclose(x["step"],y["step"],rel_tol=1e-7) for x,y in zip(a,b)) else "different"
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return "same" if isclose(a,b,rel_tol=1e-7) else "different"
    return "same" if a==b else "different"

def comparable_calendar(a,b):
    first=a[3] if len(a)>3 else "gregorian"
    second=b[3] if len(b)>3 else "gregorian"
    return first==second or (
        {first,second}<={"gregorian","proleptic_gregorian"}
        and min(a[0],b[0])>=0
    )

def overlap(a,b):
    a,b=a.get("time_coverage"),b.get("time_coverage")
    if not a or not b or not comparable_calendar(a,b):
        return "unknown"
    intersects=max(a[0],b[0])<min(a[1],b[1]) if all(x[2]=="cell_bounds_endpoints" for x in (a,b)) else max(a[0],b[0])<=min(a[1],b[1])
    if any(x[2]=="sampled_table_records" for x in (a,b)):
        return "sampled_ranges_overlap" if intersects else "unknown"
    return "ranges_overlap" if intersects else "ranges_disjoint"

def time_class(seconds):
    if not seconds:
        return "unknown"
    if seconds < 3600:
        return "sub_hourly"
    if seconds < 86400:
        return "hourly_or_subdaily"
    if seconds == 86400:
        return "daily"
    return "multi_day_or_coarser"

def relative_spacing(a,b):
    relation=spacing_comparison(a,b)
    if relation!="different":
        return relation
    if isinstance(a,tuple):
        aa,bb=[tuple(v["step"] for v in pair) for pair in (a,b)]
    elif isinstance(a,dict):
        aa,bb=a["step"],b["step"]
    else:
        aa,bb=(a,),(b,)
    if all(x<=y for x,y in zip(aa,bb)):
        return "A_finer"
    if all(x>=y for x,y in zip(aa,bb)):
        return "B_finer"
    return "mixed_by_axis"

def model_field_matches(row, model_id, model_roles):
    """Run the original scientific matcher on metadata-only scalar placeholders."""
    import numpy as np
    import xarray as xr
    from ecodata.models.scientific_schema import match_fields

    variables={}
    for field in row["fields"]:
        if field["axis"] or field.get("raw_companion") or field_type(field["type"]) not in {"integer","number"}:
            continue
        if not field.get("unit"):
            continue
        attrs={k:field[k] for k in ("standard_name","long_name","physical_variable") if field.get(k)}
        attrs["units"]=field["unit"]
        variables[field["name"]]=xr.DataArray(np.float64(0),attrs=attrs)
    if not variables:
        return {}
    metadata=xr.Dataset(variables)
    matches=defaultdict(list)
    for role in model_roles:
        matched,_=match_fields(metadata,[f for f in role["fields"] if isinstance(f,dict) and f.get("name") and f.get("unit")])
        for target,source in matched.items():
            matches[source].append(f"{role['role']} / {target}")
    return dict(matches)


def field_references(field,fid):
    if not field.get("variable_references"):
        return {}
    source=fid.partition(".")[0]
    def described(edge):
        record={key:value for key,value in edge.items() if key not in {"field_index","target_references"}}
        if index:=edge.get("field_index"):
            record["field_id"]=f"{source}.F{index}"
        if children:=edge.get("target_references"):
            record["target_references"]=[described(child) for child in children]
        return record
    return {"variable_references":[described(edge) for edge in field["variable_references"]]}


def describe_field(row,fid,field,model_id,*,native_role=None,model_matches=()):
    family=physical(field)
    roles=[ROLE_LABELS[r] for r in row["roles"] if r in ROLE_LABELS]
    t,g=resolution(row)
    grid={"x":g[0],"y":g[1]} if isinstance(g,tuple) else g
    metadata=IOAPI_DESCRIPTIONS.get((field["name"],field.get("var_desc","").casefold())) if row.get("ioapi_grid_role") else None
    role=GRID_ROLES.get(row.get("ioapi_grid_role"))
    native_labels={"engineering_dependency":"原版识别为工程依赖","reference_output":"原版识别为模型参考输出",
                   "engineering_configuration":"原版识别为工程配置"}
    native_use=native_labels.get(native_role)
    return dict(id=fid, header=field.get("original_header",field["name"]), variable_name=field["name"],
                type=field_type(field["type"]), unit=field["unit"] or None,
                physical_family=field_identity(row,field)[0] or family or None,
                standard_name=field.get("standard_name") or None, physical_variable=field.get("physical_variable") or None,
                long_name=field.get("long_name") or None, var_desc=field.get("var_desc") or None,
                temporal_semantics=field.get("temporal_semantics") or None,
                **({"time_cell_methods":field["time_cell_methods"],"time_cell_methods_basis":"CF cell_methods"}
                   if field.get("time_cell_methods") else {}),
                source_column=field.get("source_column"),
                dimensions=field.get("dimensions"), shape=field.get("shape"), sheet=field.get("sheet"), axis=field["axis"],
                raw_companion=field.get("raw_companion",False),
                **({"metadata_role":field["metadata_role"]} if field.get("metadata_role") else {}),
                **({"soil_depth":field["soil_depth"]} if field.get("soil_depth") else {}),
                **field_references(field,fid),
                physical_meaning=metadata[1] if metadata else ("叶面积与地面积之比" if family=="leaf_area_index" else None),
                description_source="I/O API var_desc" if metadata else None,
                grid_role=role,
                engineering_use=roles or ([native_use] if native_use else None) or (["MCIP 工程叶面积指数，用于与遥感 LAI 对照"]
                    if family=="leaf_area_index" and row.get("time_spacing_basis")=="ioapi_timestep_and_tflag"
                    else ["遥感合成叶面积指数，用于与工程 LAI 对照"]
                    if family=="leaf_area_index" and row.get("time_coverage",(None,None,None))[2]=="dataset_declared_interval"
                    else ["待结合关联工程文档核验用途"]),
                engineering_evidence=("按工程关联定向核验 NetCDF 字段和元数据" if row.get("engineering_probe")
                                      else "工程资料有文件引用" if row["evidence"] else
                                      "原版模型入口识别到工程角色" if native_use else "未找到工程引用"),
                candidate_model_input=("；".join(model_matches) if model_matches else
                    MODEL_USES.get(model_id,{}).get(family,"目标模型输入类别待确认")),
                model_input_match_basis="原版模型画像与字段单位匹配" if model_matches else None,
                time_resolution_class=row.get("declared_output_period") or time_class(t),
                **({"declared_output_period":row["declared_output_period"]} if row.get("declared_output_period") else {}),
                sampling_interval_seconds=t, time_cell_duration_seconds=row.get("time_cell_seconds"), declared_time_resolution=row.get("declared_time_resolution"), grid_interval=grid,
                time_spacing_evidence=(
                    "P1D_declaration_and_single_cell_bounds" if t and not row["resolution"].get("time_seconds")
                    else row.get("time_spacing_basis", "table_sample" if row["format"] in {"CSV","TSV"} else "coordinate_sample")
                    if t or row.get("declared_output_period") else "unknown"),
                time_coverage_evidence=row.get("time_coverage",(None,None,"unknown"))[2],
                grid_spacing_evidence=row.get("grid_spacing_basis","coordinate_or_raster_metadata") if g else "unknown",
                declared_coverage_duration_seconds=(row["time_coverage"][1]-row["time_coverage"][0]
                    if row.get("time_coverage",(None,None,None))[2]=="dataset_declared_interval" else None))

TASK_GOALS={
    "grid_comparison":"逐格比较",
    "gap_fill":"用其他来源补缺",
    "bias_correction":"偏差订正",
    "evaluate_gain":"评估融合能否改善结果",
}

def discover(scan, model_id=None, task_goal="auto"):
    if scan.get("shallow"):
        return dict(stage="decision",catalog_first=True,
                    target_model=dict(id=model_id,name=MODELS[model_id]) if model_id in MODELS else None,
                    sources=[],candidates=[],engineering_documents=[],engineering_inventory=[],
                    inspection_mode='on_demand_tools' if scan.get('on_demand') else 'legacy_shallow',
                    initial_directory=scan.get('initial_directory'),
                    **({'task_goal':task_goal} if task_goal in TASK_GOALS else {})), {}
    from .native_dispatch import inspect
    native=inspect(scan,model_id)
    native_roles={claim["file"]:claim["role"] for claim in native.get("claims",[])}
    model_roles=[]
    if model_id in MODELS and model_id!='data_only':
        from ecodata.algorithms.profile_matching import model_table
        model_roles=[role for role in model_table(model_id)["roles"] if not role["record_only"]]
    from .research_dispatch import discover as research_tasks
    scan.setdefault("research_tasks",{})[model_id]=research_tasks(scan,model_id)
    scan["candidate_native_roles"]=native_roles
    scan["candidate_field_context"]={}
    sources=[]
    for index,row in enumerate(scan["rows"],1):
        if not row["fields"]:
            continue
        source=f"D{index}"
        matches=model_field_matches(row,model_id,model_roles) if model_roles else {}
        scan["candidate_field_context"][index-1]=matches
        fields=[describe_field(
            row,f"{source}.F{offset}",field,model_id,native_role=native_roles.get(row["name"]),
            model_matches=matches.get(field["name"],[]))
            for offset,field in enumerate(row["fields"],1)]
        sources.append(dict(id=source,format=row["format"],fields=fields,
                            dimensions=row.get("dimensions"),roles=row["roles"],
                            engineering_reference=bool(row["evidence"]),
                            **({"coordinate_reference":row["coordinate_reference"]} if row.get("coordinate_reference") else {}),
                            time_coverage=coarse_time_coverage(row),
                            **({"declared_output_print_period":row["declared_output_print_period"]}
                               if row.get("declared_output_print_period") else {})))
        if row.get("single_average_annual_window"):
            sources[-1]["single_average_annual_window"]=True
    docs,document_refs=associate_documents(scan,{int(source["id"][1:]):source["id"] for source in sources})
    for source in sources:
        source["engineering_documents"]=document_refs.get(source["id"],[])
    assert_current(scan)
    return dict(
        stage="decision", catalog_first=True,
        **({"task_goal":task_goal} if task_goal in TASK_GOALS else {}),
        target_model=dict(id=model_id,name=MODELS[model_id]) if model_id in MODELS else None,
        saved_input_context=scan.get('saved_input_context'),
        sources=sources,candidates=[],engineering_documents=docs,
        project_context_document_ids=[d["id"] for d in docs if d["scope"]=="project_context"],
        engineering_inventory=[dict(id=f"M{i}",kind=row["format"],nested_archive=bool(kind(row["name"])),
                                    nested_members_read=row.get("nested_members_read",False))
                               for i,row in enumerate(scan["rows"],1)
                               if kind(row["name"]) or not row["fields"]]), {}


def coarse_time_coverage(row):
    from datetime import datetime,timezone
    span=row.get("time_coverage")
    if not span:
        return None
    calendar=span[3] if len(span)>3 else "gregorian"
    try:
        if calendar in {"gregorian","proleptic_gregorian"}:
            dates=[datetime.fromtimestamp(value,timezone.utc) for value in span[:2]]
        else:
            from netCDF4 import num2date
            dates=num2date(span[:2],"seconds since 1970-01-01",calendar=calendar)
        months=[f"{date.year:04d}-{date.month:02d}" for date in dates]
    except (ValueError,OverflowError,OSError,ImportError):
        return dict(basis=span[2],calendar=calendar)
    return dict(start_month=months[0],end_month=months[1],basis=span[2],calendar=calendar)


def describe_pair(scan,cid,left,right,model_id,details):
    ai,af,a=left
    bi,bf,b=right
    ar,br=scan["rows"][ai],scan["rows"][bi]
    t1,g1=resolution(ar)
    t2,g2=resolution(br)
    time_spacing=spacing_comparison(t1,t2)
    grid_spacing=spacing_comparison(g1,g2)
    from .station_grid import eligible
    actions=["fusion","convert_units","convert_format","preprocess"]
    if eligible(ar,a,br) or eligible(br,b,ar):
        actions.extend(("compare_station_grid","align_time","align_space"))
    if time_spacing=="different" and "align_time" not in actions:
        actions.append("align_time")
    if "align_space" not in actions and (grid_spacing=="different" or (g1 and g2 and grid_spacing in {"unknown","not_comparable"})):
        actions.append("align_space")
    unit_check=pair_units(ar,a,br,b)
    soil={}
    if a.get("soil_depth") or b.get("soil_depth"):
        from .soil_depth import compare
        soil["soil_depth_comparison"]=compare(a.get("soil_depth"),b.get("soil_depth"))
    pair=dict(id=cid,fields=[af,bf],possible_actions=actions,target_model=MODELS.get(model_id),
              pair_details=[dict(details[af],source="A"),dict(details[bf],source="B")],
              resolution_comparison=dict(time_spacing=time_spacing,grid_spacing=grid_spacing,
                                         time_coverage=overlap(ar,br)),
              relative_spacing=dict(time=relative_spacing(t1,t2),grid=relative_spacing(g1,g2)),
              match_basis="deepseek_catalog",roles=[ar["roles"],br["roles"]],
              engineering_reference=bool(ar["evidence"] or br["evidence"]),
              unit_check=public_unit_report(unit_check),units_equal=unit_check["status"]=="equivalent",**soil)
    return pair,(left,right,actions)

def local_overview(scan,candidates):
    return dict(members=len(scan["rows"]), warnings=scan["warnings"],
                cmaq_grid_checks=scan.get("cmaq_grid_checks",[]),
                original_model_probes=scan.get("native_model_probes",{}),
                local_research_tools=[dict(file=task["file"],label=task["label"])
                    for tasks in scan.get("research_tasks",{}).values() for task in tasks],
                candidate_scan_limit=scan.get("candidate_scan_limit",False),
                local_field_inventory=[dict(file=r["name"],format=r["format"],total_fields=len(r["fields"]),
                    resolution=r["resolution"],pending=r["pending"],roles=r["roles"],
                    fields=[dict(name=f.get("original_header",f["name"]),type=f["type"],unit=f["unit"],
                                **({"soil_depth":f["soil_depth"]} if f.get("soil_depth") else {}),
                                **({"variable_references":f["variable_references"]} if f.get("variable_references") else {}))
                            for f in r["fields"]])
                    for r in scan["rows"] if r["fields"] or r["pending"]],
                candidates=[dict(id=k,actions=actions,fields=[
                    dict(file=scan["rows"][i]["name"],field=f["name"],unit=f["unit"],
                         resolution=scan["rows"][i]["resolution"],engineering=scan["rows"][i]["evidence"],
                         native_role=scan.get("candidate_native_roles",{}).get(scan["rows"][i]["name"]),
                         model_input_matches=scan.get("candidate_field_context",{}).get(i,{}).get(f["name"],[]))
                    for i,_,f in (left,right)]) for k,(left,right,actions) in candidates.items()],
                members_needing_inspection=[r["name"] for r in scan["rows"] if r["pending"] and not r["fields"]][:60])

def audit_precipitation(scan,row,field,raw,dataset):
    from hashlib import sha256
    from urllib.parse import parse_qs,urlsplit
    from ecodata.models.engineering.precipitation_products import inspect_product

    audit=inspect_product(dataset)
    if not audit:
        return None
    detail=next((item for item in audit["fields"] if item["field"]==field["name"]),None)
    if not detail:
        return None
    if audit["product"]=="NASA_POWER" and detail["time_basis"]=="UNKNOWN":
        for name,content in scan.get("engineering_docs",()):
            if name!=row["name"]+".receipt.json":
                continue
            try:
                receipt=json.loads(content)
                url=urlsplit(receipt["url"])
                params=parse_qs(url.query)
                basis=params.get("time-standard",[])
                if (url.scheme=="https" and url.hostname=="power.larc.nasa.gov"
                    and url.path.startswith("/api/temporal/daily/")
                    and basis in (["UTC"],["LST"])
                    and receipt["sha256"]==sha256(raw).hexdigest()
                    and field["name"] in params.get("parameters",[""])[0].split(",")):
                    detail=dict(detail,time_basis=basis[0],time_basis_evidence="sha256_verified_download_receipt",
                                blockers=[item for item in detail["blockers"] if item["code"]!="SOURCE_DAY_UNCONFIRMED"])
            except (ValueError,TypeError,KeyError):
                pass
    return dict(product=audit["product"],**detail)

def check(scan,candidates,reply,approved_summary,*,station_matches=None):
    from .scan import read_selected
    from ecodata.recognition import recognise_file
    assert_current(scan)
    allowed={c.get("id"):c for c in approved_summary.get("candidates",[]) if isinstance(c,dict) and isinstance(c.get("id"),str)}
    results=[]
    prepared={}
    datasets={}
    audits={}
    loaded_bytes=0

    def read_for_check(index,field):
        nonlocal loaded_bytes
        row=scan["rows"][index]
        if index not in datasets:
            if len(datasets)>=8 or loaded_bytes+row["bytes"]>64*1024**2:
                raise ValueError("已达自动深读文件或字节上限")
            loaded_bytes+=row["bytes"]
            raw=read_selected(row)
            dataset,_=recognise_file(row["name"],raw,record_read_source=False)
            datasets[index]=dataset
            if physical(field)=="precipitation":
                audits[index,field["name"]]=audit_precipitation(scan,row,field,raw,dataset)
        return datasets[index]
    target=approved_summary.get("target_model")
    model_id=target.get("id") if isinstance(target,dict) else None
    task_goal=approved_summary.get("task_goal","auto")
    from .reference_search import plan as reference_plan
    proposals=list(reply.get("decisions",[]))
    if task_goal=='evaluate_gain':
        # A QA suggestion is a prerequisite, not a replacement for the explicit
        # user task. No algorithm, score or winner is introduced here.
        for cid in sorted(allowed.keys() & candidates.keys()):
            related=[row for row in proposals if row.get('candidate')==cid]
            if related and any(row.get('action')=='quality_screen' for row in related) and not any(
                row.get('action') in {'fusion','insufficient','pending_goal','reject','direct'} for row in related):
                proposals.append(dict(candidate=cid,action='insufficient',goal=task_goal,
                    reason='显式用户目标仍需比较处理价值；质量筛查建议交给公共质量门，不能替代工程任务核验。'))
    proposed_quantities={}
    for proposal in proposals:
        if quantity:=proposal.get("physical_quantity"):
            proposed_quantities.setdefault(proposal["candidate"],set()).add(quantity)
    blocked={p["candidate"] for p in proposals if p["action"]=="reject"}
    proposed_plans={proposal["candidate"]:proposal["preprocessing_plan"] for proposal in proposals
                    if proposal.get("preprocessing_plan") and proposal["candidate"] not in blocked}
    if proposed_plans:
        from .statistical_preprocessing import prepare as prepare_statistics,references
        approved_fields={field["id"] for source in approved_summary.get("sources",[]) for field in source["fields"]}
        for cid,plan in proposed_plans.items():
            if cid not in candidates or cid not in allowed:
                continue
            result=dict(candidate=cid,action="preprocess",status="条件不足",block_code="preprocessing_evidence_pending",
                        next_step="recommendation_only")
            results.append(result)
            try:
                if not references(plan)<=approved_fields:
                    raise ValueError("换算依据引用了未审核的字段")
                task=prepare_statistics(scan,plan,allowed[cid]["fields"],approved_summary.get("engineering_documents",[]))
                prepared[(cid,"preprocess")]=task
                result.update(status="通过",reason="已核实对象、完整统计窗口和换算依据，可执行时间聚合与单位换算组合。")
                result.pop("next_step")
            except (EcoDataError,ValueError,OSError,KeyError,TypeError,RuntimeError) as error:
                result["reason"]=str(error)
    work=[]
    for proposal in proposals:
        cid,action=proposal["candidate"],proposal["action"]
        if action=="preprocess":
            if cid not in proposed_plans:
                results.append(dict(candidate=cid,action=action,status="条件不足",block_code="preprocessing_plan_missing",
                                    next_step="recommendation_only",reason="需补齐具体的统计窗口、面积引用和换算步骤"))
            continue
        if action=="reject":
            results.append(dict(candidate=cid,action=action,status="未核验",
                                reason="DeepSeek 明确反对融合"))
            continue
        if cid in blocked and action=="fusion":
            continue
        if cid not in allowed or cid not in candidates:
            continue
        if action=='quality_screen':
            results.append(dict(candidate=cid,action=action,status='待执行质量核验',next_step='runtime_quality_gate',
                reason='质量筛查建议不是融合目标。若同一候选进入公共工程处理，将由原始输入和输出质量门实际检查；额外QA规则缺少实现或证据时仍拒绝执行。'))
            continue
        goal=task_goal if isinstance(task_goal,str) and task_goal in TASK_GOALS else proposal.get("goal","unknown")
        ready_reference = reference_plan(scan,candidates[cid],model_id)

        if action=="pending_goal" and task_goal in TASK_GOALS:
            # A declared user goal is separate from missing validation evidence.
            # Continue the same unit/time/support checks as an insufficient proposal.
            action="insufficient"
        if action=="pending_goal" and not ready_reference:
            results.append(dict(candidate=cid,action=action,status="待确认用途",
                                reason="暂缺融合用途依据，继续本地核验候选的时间和空间对齐条件。"))
        if action=="direct":
            results.append(dict(candidate=cid,action=action,status="无需新增处理",
                                reason="DeepSeek 建议按当前任务目标分别使用。"))
            continue
        allowed_actions={"align_space","align_time","fusion","compare_station_grid","convert_units","convert_format"}
        if action=="pending_goal" or (action=="insufficient" and goal not in TASK_GOALS):
            allowed_actions={"align_space","align_time","compare_station_grid","convert_units","convert_format"}
            if ready_reference:
                allowed_actions.add("fusion")
        if action in {"insufficient","pending_goal"}:
            approved_options=allowed[cid].get("possible_actions",[])
            if isinstance(approved_options,list):
                work.extend((cid,option,action) for option in candidates[cid][2]
                            if option in approved_options and option in allowed_actions and option!="convert_format"
                            and (option!="fusion" or cid not in blocked))
        elif action in allowed_actions:
            work.append((cid,action,action))
            if action=="fusion" and "compare_station_grid" in candidates[cid][2]:
                work.append((cid,"compare_station_grid",action))
        else:
            results.append(dict(candidate=cid,action=action,status="与任务目标不符",
                                reason="逐格比较只需核验对齐。评估融合需明确补缺、订正或对比收益的目标。"))
    seen=set()
    station_matches={} if station_matches is None else station_matches
    for cid,action,online_action in work:
        if (cid,action) in seen:
            continue
        seen.add((cid,action))
        if (cid,"preprocess") in prepared and action in {"align_time","convert_units"}:
            continue
        row=dict(candidate=cid,action=action,online_action=online_action,status="条件不足",block_code="scope")
        results.append(row)
        if (cid,"preprocess") in prepared and action=="fusion":
            row.update(block_code="preprocessing_first",next_step="recommendation_only",status="先执行已核实前处理",
                       reason="可先执行统计窗口统一和换算，再依据对象空间支持与可用留出样本评估融合。")
            continue
        approved=allowed.get(cid,{})
        actions=approved.get("possible_actions",[])
        delegated=(isinstance(actions,list) and action in {'fusion','align_space','align_time'} and task_goal in TASK_GOALS and
            scan.get('native_model_probes',{}).get(model_id,{}).get('status')=='project' and 'fusion' in actions)
        if cid not in candidates or not isinstance(actions,list) or (action not in actions and not delegated):
            row["reason"]="建议不在本地扫描且经审核的候选范围内"
            continue
        left,right,detected=candidates[cid]
        if approved.get("fields")!=[left[1],right[1]] or (action not in detected and not delegated):
            row["reason"]="建议的字段或操作与本地候选不符"
            continue
        ai,_,a=left
        bi,_,b=right
        ar,br=scan["rows"][ai],scan["rows"][bi]
        row["block_code"]="verification_pending"
        if a["axis"] or b["axis"]:
            row.update(block_code="coordinate_field",reason="坐标轴可用于对齐，不能作为待融合的观测变量")
            continue
        guesses=proposed_quantities.get(cid,set())
        inferred=next(iter(guesses)) if len(guesses)==1 else None
        unit_check=pair_units(ar,a,br,b,inferred)
        _,basis_a=field_identity(ar,a)
        _,basis_b=field_identity(br,b)
        same_quantity=bool(unit_check["quantity"])
        cf_daily_pair = (action in {"align_space", "fusion"} and not delegated
                         and unit_check.get("quantity") == "precipitation"
                         and ar["format"] == br["format"] == "NetCDF")
        row["unit_check"]=unit_check
        row["physical_identity"]=dict(left_basis=basis_a,right_basis=basis_b,match=same_quantity,
                                      mapping_basis=unit_check.get("identity_basis","local_metadata"))
        if (action != "convert_format" and not cf_daily_pair
                and unit_check["status"] not in {"equivalent", "convertible", "physical_identity_unconfirmed"}):
            row["block_code"]={"quantity_conflict":"quantity_conflict","unit_not_convertible":"unit_conflict"}.get(
                unit_check["status"],"unit_unknown")
            row["reason"]=("两份元数据明确声明了不同物理量" if row["block_code"]=="quantity_conflict"
                           else "物理量已核对，但单位缺失或无法换算")
            continue
        if action=="fusion" and {ar.get("ioapi_grid_role"),br.get("ioapi_grid_role")}=={"boundary","interior"}:
            row["block_code"]="different_grid_roles"
            row["reason"]="I/O API 边界和内部字段承担不同模型角色。只有经工程确认来源及角色后才能拿来比较或融合。"
            continue
        try:
            if delegated:
                prepared[(cid,action)]=dict(action=action,route='original_engineering',fields=[(ai,a['name']),(bi,b['name'])])
                row.update(status='待工程核验',next_step='original_workbench',reason='已识别实际工程控制与同物理量候选；交由公共工程输入端口服务核验日界、依赖、拟合隔离及真实数值效果。此状态不是融合或质量PASS。')
                continue
            if action in {"convert_units","convert_format"}:
                from .pair_processing import conversion_task
                if action=="convert_units" and unit_check["status"] not in {"convertible","equivalent"}:
                    row["block_code"]="quantity_unknown"
                    raise ValueError("还需确定可共同换算的物理量和单位")
                if action=="convert_units" and unit_check["status"]=="equivalent":
                    from .table_unit_segments import unit_column
                    explicitly_segmented = any(
                        source["format"] in {"CSV", "TSV", "Excel"} and
                        unit_column([column.get("original_header") or column["name"] for column in source["fields"]
                                     if column.get("sheet") == field.get("sheet")], field) is not None
                        for source, field in ((ar, a), (br, b)))
                    if not explicitly_segmented:
                        results.pop()
                        continue
                task=conversion_task(scan,[(ai,a),(bi,b)],action,unit_check)
                prepared[(cid,action)]=task
                row.update(status="通过",reason="可独立转换字段并回读核对，供后续配对和融合使用。")
                continue
            if "compare_station_grid" in detected and action in {"compare_station_grid","align_time","align_space","fusion"}:
                from .station_grid import BIAS, EXACT, FORMAT_STEP, MATCH, NEAREST, UNIT_STEP, load, plan, task_for
                if unit_check["status"] not in {"equivalent","convertible"}:
                    row["block_code"]="quantity_unknown"
                    raise ValueError("请识别共同物理量并检查单位后进行站点配对")
                task=task_for(scan,(left,right),action,unit_check)
                if action == "fusion":
                    from .recipe_trials import FILL

                    task["fields"] = [(ai, a["name"]), (bi, b["name"])]
                    task["recipe_steps"] = [[UNIT_STEP, FORMAT_STEP, method, MATCH, FILL, BIAS]
                                            for method in (EXACT, NEAREST)]
                    prepared[(cid,action)] = task
                    row.update(status="通过", reason="按表格索引留出站点或月份，在实际网格位置和时间上试跑订正与补位。")
                    continue
                evidence_key=tuple(item[1] for item in (left,right)),unit_check["quantity"]
                if evidence_key not in station_matches:
                    station_matches[evidence_key]=load(scan,task)
                task["recipe_steps"]=plan(station_matches[evidence_key],action)
                prepared[(cid,action)]=task
                row.update(status="通过",reason="原始站点与网格时空配对通过。可运行已登记的组合。")
                continue
            cf_error = None
            if cf_daily_pair:
                from .independent_alignment import check_cf_headers

                selections = [(ai, a["name"]), (bi, b["name"])]
                try:
                    check_cf_headers(scan, selections)
                except (EcoDataError, OSError, ValueError, KeyError) as error:
                    cf_error = error
                else:
                    prepared[(cid, action)] = dict(action=action, cf_daily=True,
                        route="independent_grid_search" if action == "fusion" else "independent_alignment",
                        fields=selections)
                    row.update(status="通过", next_step="alignment_then_fusion",
                               reason="已核对 UTC 区间语义、单位和共同网格范围。完整日对齐，不完整日留待补位试跑。")
                    continue
            if action == "fusion" and unit_check["status"] in {"equivalent", "convertible"}:
                from . import indexed_table_recipe, recipe_trials, windowed_grid_recipe

                formats = {ar["format"], br["format"]}
                task = dict(action=action, fields=[(ai, a["name"]), (bi, b["name"])], field_metadata=[a, b],
                            quantity=unit_check["quantity"], unit=unit_check["target_unit"])
                steps = []
                if formats <= indexed_table_recipe.FORMATS:
                    task["route"] = "indexed_table_recipe"
                    steps = [[indexed_table_recipe.LINEAR]]
                elif formats <= windowed_grid_recipe.FORMATS and not approved_summary.get("task_requirements", {}).get("target_grid"):
                    try:
                        steps = windowed_grid_recipe.available(scan, task)
                        task["route"] = "windowed_grid_recipe"
                    except (EcoDataError, OSError, ValueError, KeyError):
                        # Let the registered engineering adapters handle other grid supports.
                        pass
                if steps:
                    task["recipe_steps"] = [["units.harmonize", "format.normalize", *align, recipe_trials.FILL, recipe_trials.BLEND]
                                             for align in steps]
                    prepared[(cid, action)] = task
                    row.update(status="通过", next_step="alignment_then_fusion", algorithm_candidates=list(dict.fromkeys(
                               operation for recipe in steps for operation in recipe)),
                               reason="字段物理量和单位可对应。留出组的原始数据将执行完整方案并与基线比较。")
                    continue
            if physical(a)==physical(b)=="precipitation":
                read_for_check(ai,a)
                read_for_check(bi,b)
                products=[audits.get((index,field["name"])) for index,field in ((ai,a),(bi,b))]
                if all(products) and all(p["value_semantics"]=="daily_mean_precipitation_rate" for p in products):
                    unit_check=compare_units("precipitation",a["unit"],"precipitation",b["unit"])
                    row["unit_check"]=unit_check
                    if unit_check["status"] not in {"equivalent","convertible"}:
                        raise ValueError("已核实降水率物理量，但两源单位无法统一")
                    row["product_checks"]=[{k:p.get(k) for k in ("product","field","unit","time_basis","interval_seconds","time_basis_evidence","blockers")} for p in products]
                    row["coverage_check"]=overlap(ar,br)
                    blockers=[item["message"] for p in products for item in p["blockers"]]
                    if blockers:
                        raise ValueError("已确认两字段都是日降水率；"+"；".join(dict.fromkeys(blockers)))
                    if any(p["time_basis"]!="UTC" for p in products):
                        raise ValueError("已确认两字段都是日降水率；需核对两产品统计日边界的一致性")
                    if action in {"fusion","align_space"}:
                        row["physical_identity"]=dict(match=True,left_basis="original_product_audit",right_basis="original_product_audit")
                        row["target_model"]=MODELS.get(model_id)
                        if row["coverage_check"]=="ranges_disjoint":
                            row["block_code"]="disjoint_time"
                            raise ValueError("两份日降水资料的时间覆盖不重叠")
                        if action=="align_space":
                            prepared[(cid,action)]=dict(action=action,route="independent_alignment",
                                fields=[(ai,a["name"]),(bi,b["name"])])
                            row.update(status="通过",reason="可按共同 UTC 日期范围转为日累计量并保守对齐。未指定目标时比较可行的原生网格。")
                        else:
                            if ar["bytes"]+br["bytes"]>32*1024**2:
                                raise ValueError("独立格点试跑超出读取预算")
                            if products[0]["product"]==products[1]["product"]:
                                row["block_code"]="same_product"
                                raise ValueError("同一产品不作为两个融合来源")
                            prepared[(cid,action)]=dict(action=action,route="independent_grid_search",
                                fields=[(ai,a["name"]),(bi,b["name"])],methods=["spatial.blend"])
                            row.update(status="通过",reason="先做保守网格对齐，缺测可用备用源补位；满足覆盖条件时用原版空间变异搜索比较单源与多源。")
                        continue
            if cf_error and unit_check["status"] not in {"equivalent", "convertible"}:
                raise ValueError(f"降水率与累计量需要明确的单位、时间边界与日界：{cf_error}") from cf_error
            if action=="fusion":
                if not same_quantity:
                    row["block_code"]="quantity_unknown"
                    raise ValueError("融合需先核实两个字段表示同一物理量")
                from . import pair_processing
                reference=reference_plan(scan,(left,right),model_id)
                if reference is not None:
                    prepared[(cid,action)]=reference
                    row.update(status="通过",next_step="alignment_then_fusion",block_code=None,
                               reason="卫星质量标记、合成时间窗和 MCIP 单层网格已匹配。可勾选质量控制与尺度对齐后的原版融合试跑。")
                    continue
                if a.get("observation_contract") or b.get("observation_contract"):
                    from .gap_search import verify
                    task=dict(action=action,route="observation_gap_search",
                        fields=[(ai,a["name"]),(bi,b["name"])],quantity=unit_check["quantity"])
                    if loaded_bytes+ar["bytes"]+br["bytes"]>64*1024**2:
                        raise ValueError("已达自动深读字节上限")
                    loaded_bytes+=ar["bytes"]+br["bytes"]
                    verify(scan,task)
                    prepared[(cid,action)]=task
                    row.update(status="通过",reason="原版观测表、质量标识、缺失位置与背景场契约通过。")
                    continue
                if not ("observations" in ar["roles"] and "background" in br["roles"] or
                        "observations" in br["roles"] and "background" in ar["roles"]):
                    task=dict(action=action,route="pair_processing",fields=[(ai,a["name"]),(bi,b["name"])],
                        quantity=unit_check["quantity"],unit=unit_check["target_unit"],
                        identity_basis=unit_check.get("identity_basis"),field_metadata=[a,b])
                    task["methods"]=pair_processing.methods([read_for_check(ai,a),read_for_check(bi,b)],task)
                    prepared[(cid,action)]=task
                    row.update(status="通过",next_step="alignment_then_fusion",
                        reason="已核对物理量和空间支持。对齐后以原版变异寻优比较单源与多源。",
                        algorithm_candidates=task["methods"]+task.get("weighted_methods",[]))
                    continue
                temporal=a.get("temporal_semantics")
                if not temporal or temporal!=b.get("temporal_semantics"):
                    row["block_code"]="different_time_support" if temporal and b.get("temporal_semantics") else "time_support_unknown"
                    raise ValueError("两来源须声明相同的时间语义")
                selection=[(i,"observations" if "observations" in scan["rows"][i]["roles"] else "primary") for i in (ai,bi)]
                if loaded_bytes+ar["bytes"]+br["bytes"]>64*1024**2:
                    raise ValueError("已达自动深读字节上限")
                loaded_bytes+=ar["bytes"]+br["bytes"]
                primary=a if selection[0][1]=="primary" else b
                state,_=prepare(scan,selection,"fusion",None,"","NetCDF",primary["name"],
                                unit_check["target_unit"],temporal,"{}", physical_quantity=unit_check["quantity"])
                prepared[(cid,action)]=dict(state=state,selection=selection,action=action)
                row.update(status="通过",reason="原版空间融合输入契约通过",
                           algorithm_candidates=state["eligible"])
            elif action in {"align_time","align_space"}:
                if not same_quantity:
                    raise ValueError("未能通过标准名、物理变量声明或原版科学别名确认相同物理量")
                if ar["bytes"]+br["bytes"]>64*1024**2:
                    raise ValueError("核验输入超过 64 MiB，请准备字段子集")
                from .pair_processing import methods
                task=dict(action=action,route="pair_processing",fields=[(ai,a["name"]),(bi,b["name"])],
                    quantity=unit_check["quantity"],unit=unit_check["target_unit"],
                        identity_basis=unit_check.get("identity_basis"),field_metadata=[a,b])
                task["methods"]=methods([read_for_check(ai,a),read_for_check(bi,b)],task)
                prepared[(cid,action)]=task
                detail=("时间重采样采用共同覆盖内较粗的原始采样间隔。" if action=="align_time" else
                        "同坐标系采用共同范围内较粗网格；跨投影映射到第一来源的网格。")
                row.update(status="通过",reason="已核对原始字段和重采样条件。"+detail,
                           algorithm_candidates=task["methods"])
        except Exception as exc:
            row["reason"]=str(exc)
    assert_current(scan)
    target=approved_summary.get('task_requirements',{}).get('target_grid')
    if target:
        from ecodata.spatial.regridding import target_grid_spec
        target=target_grid_spec(target)
        for key,task in prepared.items():
            if task.get('route') not in {'independent_grid_search','independent_alignment'}:
                for row in results:
                    if (row.get('candidate'),row.get('action'))==key:
                        row.update(status='条件不足',next_step=None,block_code='target_grid_unsupported',reason='此任务适配器尚未消费目标网格硬约束，不能按默认网格继续')
            else:
                task['target_grid']=target
                for row in results:
                    if (row.get('candidate'),row.get('action'))==key:
                        row['target_grid']=target
                        row['reason']='已声明目标网格硬约束：纬向 '+str(target['latitude_step'])+'°、经向 '+str(target['longitude_step'])+'°；仅在共同覆盖内执行，所有候选先生成目标网格再评价，不按默认粗网格交付。原始粗参考只能验证回聚合代理误差，不能证明细节准确。'
    return results,prepared


def deferred_quality_check(row,rows):
    """A pending QA obligation is safe to defer only to a bound checked route."""
    return (row.get('action')=='quality_screen' and row.get('next_step')=='runtime_quality_gate'
        and any(other.get('candidate')==row.get('candidate') and other.get('next_step')=='original_workbench' for other in rows))


def blocking_checks(rows):
    return [row for row in rows if row.get('status')!='通过' and row.get('next_step')!='original_workbench'
        and not deferred_quality_check(row,rows)]

def parse_decisions(raw):
    rows=raw.get("decisions")
    if not isinstance(rows,list): raise ValueError("DeepSeek 未返回 decisions 数组")
    out=[]
    for item in rows:
        if not isinstance(item,dict): continue
        candidate,action=item.get("candidate"),item.get("action")
        if not isinstance(candidate,str) or not re.fullmatch(r"C[1-9]\d*",candidate): continue
        if action not in ("fusion","align_time","align_space","compare_station_grid","convert_units","convert_format","preprocess","quality_screen","direct","insufficient","reject","pending_goal"): continue
        goal=item.get("goal")
        if not isinstance(goal,str) or goal not in TASK_GOALS or goal=="auto":
            goal="unknown"
        quantity=item.get("physical_quantity")
        if not isinstance(quantity,str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,71}",quantity) or quantity in {"unknown","uncertain"}:
            quantity=None
        steps=item.get("algorithm_steps",[])
        steps=[list(group) for group in steps if isinstance(group,list) and group and
               all(isinstance(step,str) and len(step)<=100 for step in group)] if isinstance(steps,list) else []
        from .statistical_preprocessing import parse_plan
        plan=parse_plan(item.get("preprocessing_plan"))
        out.append(dict(candidate=candidate,action=action,goal=goal,physical_quantity=quantity,
                        algorithm_steps=steps,algorithm_name=str(item.get("algorithm_name",""))[:100],
                        reason=str(item.get("reason",""))[:1000],**({"preprocessing_plan":plan} if plan else {})))
    return out



