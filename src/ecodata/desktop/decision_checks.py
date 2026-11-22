"""Bounded local evidence for the DeepSeek decision dialogue."""
from contextlib import contextmanager
from ecodata.core.exceptions import EcoDataError

from .scan import assert_current, local_member
from . import sample_values


CHECKS = {
    "numeric_samples": "指定字段的随机小数扰动数值样本",
    "metadata": "物理量、单位换算、网格角色与分辨率",
    "field_context": "字段说明、工程引用、坐标参考及标识或类别声明",
    "time_coverage": "时间支持、间隔和覆盖重叠关系",
    "annual_mean_comparison": "核对同一工程对象的平均年度输出与同年段逐年值的均值",
    "preprocessing_plan": "核验指定统计窗口、对象和面积依据，试算时间聚合与单位换算方案",
    "data_quality": "抽样有效率和质量标记登记",
    "spatial_coverage": "地理坐标样本的外包范围重叠",
    "position_agreement": "可对应位置的数值差异或类别一致率",
}
MAX_ROUNDS = 2
MAX_QUESTIONS = 12
MAX_SAMPLES = 4096
MAX_GRID_PROBE = 16384
MIN_SAMPLES = 10
CATEGORICAL = {"dominant_land_use_category", "land_water_mask"}


def consent():
    return dict(checks=CHECKS, max_rounds=MAX_ROUNDS, max_questions=MAX_QUESTIONS,
                scope="可以按已审核字段 ID 请求数值样本。平台在读取的真值上加随机小数，再仅回传样本和核验结果。"
                      "不发送原始数值、位置、精确时刻、文件路径或完整观测序列。")


def parse_reply(raw):
    from .discovery import parse_decisions

    if not isinstance(raw, dict):
        raise ValueError("DeepSeek 判断返回格式错误")
    questions = raw.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    return dict(decisions=parse_decisions({"decisions":raw.get("decisions", [])}),
                questions=[q for q in questions[:MAX_QUESTIONS] if isinstance(q, dict)
                           and isinstance(q.get("candidate"),str) and isinstance(q.get("check"),str)])


class LocalChecks:
    def __init__(self, scan, candidates, *, documents=()):
        self.scan = scan
        self.candidates = candidates
        self.cache = {}
        self.fields = {}
        self.values = {}
        self.exported = {}
        self.quantities = {}
        self.station_matches = {}
        self.documents = [doc for doc in documents if doc.get("text")]
        self.contexts = {}
        self.preprocessing_plans={}

    def __call__(self, candidate, check, approved):
        pair = self.candidates.get(candidate)
        if check not in CHECKS or not pair or approved.get("fields") != [pair[0][1],pair[1][1]]:
            return {"status":"outside_scope"}
        key = candidate, check
        if key not in self.cache:
            assert_current(self.scan)
            try:
                result = self.inspect(pair,check,self.quantities.get(candidate))
            except (EcoDataError,OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,RuntimeError,ImportError):
                result = {"status":"unavailable","basis":"local_reader_or_metadata_unavailable"}
            assert_current(self.scan)
            self.cache[key] = result
        return self.cache[key]

    @contextmanager
    def dataset(self, row):
        from netCDF4 import Dataset

        with local_member(row) as path, Dataset(str(path)) as ds:
            yield ds

    def numbers(self, item):
        index,fid,field=item
        if fid not in self.values:
            self.values[fid]=sample_values.field_sample(self.scan["rows"][index],field)
        return self.values[fid]

    def field_question(self, item):
        fid=item[1]
        if item[2].get("axis"):
            return dict(field=fid,status="outside_scope",basis="coordinate_or_time_axis")
        if fid not in self.exported:
            if item[2].get("noaa_compound"):
                source=self.scan["rows"][item[0]]["fields"]
                parts=[self.field_question((item[0],f"{fid.split('.')[0]}.F{index}",field))
                       for index,field in enumerate(source,1) if field.get("noaa_component")
                       and field.get("column")==item[2].get("column")]
                return dict(field=fid,status="checked" if any(part["status"]=="checked" for part in parts) else "unavailable",
                            basis="noaa_decoded_components",sample_fields=parts)
            assert_current(self.scan)
            try:
                row = self.scan["rows"][item[0]]
                from .sampling_plan import export_field
                result = export_field(row, item[2])
            except (EcoDataError,OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,RuntimeError,ImportError):
                result=dict(status="unavailable",basis="local_reader_or_metadata_unavailable")
            assert_current(self.scan)
            self.exported[fid]=dict(field=fid,**result)
        return self.exported[fid]

    def field_context(self, item):
        import re
        from .discovery import field_identity, field_references, public_unit

        index,fid,field=item
        if fid in self.contexts:
            return self.contexts[fid]
        source=fid.partition(".")[0]
        row=self.scan["rows"][index]
        quantity,basis=field_identity(row,field)
        declarations={key:field[key] for key in ("standard_name","physical_variable","long_name","var_desc","metadata_role")
                      if field.get(key)}
        pattern=re.compile(r"(?<!\w)"+re.escape(field["name"])+r"(?!\w)",re.I)
        snippets=[]
        for document in self.documents:
            linked=source in document.get("source_ids",[])
            if not linked and document.get("scope")!="project_context":
                continue
            text=document["text"]
            if match:=pattern.search(text):
                start=text.rfind("\n",max(0,match.start()-120),match.start())
                end=text.find("\n",match.end(),match.end()+180)
                snippets.append(dict(document=document["id"],source_linked=linked,
                    text=text[start+1 if start!=-1 else max(0,match.start()-120):end if end!=-1 else match.end()+180].strip()))
        snippets=sorted(snippets,key=lambda record:not record["source_linked"])[:4]
        self.contexts[fid]=dict(field=fid,format=row["format"],name=field["name"],datatype=field["type"],
                                unit=public_unit(field["unit"]),physical_quantity=quantity,
                                identity_basis=basis,coordinate_reference=row.get("coordinate_reference"),
                                declarations=declarations,engineering_mentions=snippets,
                                source_role=self.scan.get("candidate_native_roles",{}).get(row["name"]),
                                **field_references(field,fid))
        return self.contexts[fid]

    def inspect(self,pair,check,quantity=None):
        from .discovery import field_identity, field_references, overlap, pair_units, public_unit_report, resolution, spacing_comparison

        left,right,_=pair
        rows=[self.scan["rows"][item[0]] for item in (left,right)]
        fields=[left[2],right[2]]
        if check=="preprocessing_plan":
            from .statistical_preprocessing import prepare,PlanError
            plan=self.preprocessing_plans.get(tuple(sorted([left[1],right[1]])))
            if not plan:
                return dict(status="unavailable",basis="preprocessing_plan_missing")
            try:
                task=prepare(self.scan,plan,[left[1],right[1]],self.documents)
            except PlanError as error:
                return dict(status="unavailable",basis="preprocessing_evidence_missing",reason=str(error))
            except EcoDataError:
                return dict(status="unavailable",basis="preprocessing_evidence_missing",
                            reason="原版单位或时间引擎未认可换算条件，请核实单位维度、完整区间及统计方法")
            return dict(status="checked",basis="statistical_preprocessing_verified",plan=task["plan"],
                        methods=task["recipe_steps"][0],output_records=[len(frame) for frame in task["frames"]])
        if check=="annual_mean_comparison":
            from .annual_output_check import compare
            return compare(rows,fields)
        if check=="field_context":
            return dict(status="checked",basis="local_field_context",
                        sources=[self.field_context(item) for item in (left,right)])
        if check=="numeric_samples":
            sources=[self.field_question(item) for item in (left,right)]
            return dict(status="checked" if any(source["status"]=="checked" for source in sources) else "unavailable",
                        basis="random_decimal_perturbation",sample_fields=sources)
        durations=[resolution(row)[0] for row in rows]
        if check=="time_coverage":
            relationship=overlap(*rows)
            return dict(status="checked" if relationship!="unknown" else "unavailable",
                        relationship=relationship, sampling_interval_seconds=durations,
                        coverage_basis=[row.get("time_coverage",(0,0,"unknown"))[2] for row in rows])
        units=pair_units(rows[0],fields[0],rows[1],fields[1],quantity)
        grids=[resolution(row)[1] for row in rows]
        if check=="metadata":
            safe=public_unit_report(units)
            metadata=[]
            for row,field,item in zip(rows,fields,(left,right)):
                keys=("standard_name","physical_variable","long_name","var_desc","temporal_semantics","dimensions","shape")
                detail={k:field[k] for k in keys if field.get(k)}
                detail.update(field_references(field,item[1]))
                detail["variable_name"]=field["name"]
                detail["unit"]=field["unit"]
                if field.get("soil_depth"):
                    detail["soil_depth"]=field["soil_depth"]
                if row.get("header_only") and row["format"]=="NetCDF":
                    try:
                        with self.dataset(row) as dataset:
                            variable=dataset.variables[field["name"]]
                            detail.update((k,getattr(variable,k)) for k in keys if isinstance(getattr(variable,k,None),str))
                            detail["unit"]=str(getattr(variable,"units",detail["unit"]))
                            detail["dimensions"]=list(variable.dimensions)
                    except (OSError,ValueError,RuntimeError,ImportError,KeyError):
                        pass
                metadata.append(detail)
            depth={}
            if any(field.get("soil_depth") for field in fields):
                from .soil_depth import compare
                depth["soil_depth_comparison"]=compare(*(field.get("soil_depth") for field in fields))
            return dict(status="checked", unit_status=safe["status"], metadata=metadata,**depth,
                        field_identity_basis=[field_identity(row,field)[1] for row,field in zip(rows,fields)],
                        mapping_basis=units.get("identity_basis","local_metadata"),
                        grid_spacing=spacing_comparison(*grids),
                        grid_roles=[row.get("ioapi_grid_role") for row in rows])
        if check=="data_quality" and any(row["format"]!="NetCDF" for row in rows):
            stats=[]
            for item in (left,right):
                values=self.numbers(item)
                quality=dict(sampled=len(values or []),quality_flag_metadata_present=False)
                if values and len(values)>=MIN_SAMPLES:
                    quality["finite_percent"]=int(round(100*sum(value is not None for value in values)/len(values)/5)*5)
                stats.append(quality)
            return dict(status="checked" if any(row["sampled"]>=MIN_SAMPLES for row in stats) else "unavailable",
                        basis="numeric_field_sample",sources=stats)
        if any(row["format"]!="NetCDF" for row in rows):
            from .station_grid import EXACT, NEAREST, load, task_for
            if units["status"] not in {"equivalent","convertible"}:
                return dict(status="unavailable",basis="physical_quantity_or_unit_unconfirmed")
            task=task_for(self.scan,pair,"compare_station_grid",units)
            if task is None:
                return dict(status="unavailable",basis="position_verification_needs_supported_coordinates")
            key=tuple(item[1] for item in (left,right)),units["quantity"]
            if key not in self.station_matches:
                self.station_matches[key]=load(self.scan,task)
            frames=self.station_matches[key]
            if check=="spatial_coverage":
                return dict(status="checked",relationship="overlap",basis="station_points_inside_native_grid")
            frame=frames[EXACT] if len(frames[EXACT])>=MIN_SAMPLES else frames[NEAREST]
            if len(frame)<MIN_SAMPLES:
                return dict(status="unavailable",basis="too_few_matching_sample_locations")
            residual=frame.background-frame.observed
            return dict(status="checked",basis="station_grid_collocation",sampled_pairs=len(frame),
                        rmse=round(float((residual.pow(2).mean())**.5),3),
                        mae=round(float(residual.abs().mean()),3))
        for item,row in zip((left,right),rows):
            if item[1] not in self.fields:
                with self.dataset(row) as dataset:
                    self.fields[item[1]]=sample(dataset,item[2]["name"])
        a,b=(self.fields[item[1]] for item in (left,right))
        if check=="data_quality":
            stats=[v["quality"] for v in (a,b)]
            available=any(row["sampled"]>=MIN_SAMPLES for row in stats)
            return dict(status="checked" if available else "unavailable",basis="strided_netcdf_sample",
                        sources=stats)
        if check=="spatial_coverage":
            return spatial_coverage(a,b)
        if check=="position_agreement":
            if a.get("time_key") is None or a["time_key"]!=b.get("time_key"):
                return dict(status="unavailable",basis="sampled_time_records_not_verified_equal")
            if (a.get("vertical_key") is None or a["vertical_key"]!=b.get("vertical_key")
                    or a.get("temporal_semantics")!=b.get("temporal_semantics")):
                return dict(status="unavailable",basis="sampled_time_records_not_verified_equal")
            return position_agreement(a,b,units)
        return dict(status="outside_scope")


def sample(ds,name):
    import numpy as np

    var=ds.variables[name]
    dimensions=var.dimensions
    shape=var.shape
    geography={}
    for axis, aliases in (("lat",{"lat","latitude"}),("lon",{"lon","longitude"})):
        for key,candidate in ds.variables.items():
            if (key.casefold() in aliases or getattr(candidate,"standard_name","").casefold() in aliases):
                if candidate.ndim and set(candidate.dimensions)<=set(dimensions):
                    geography[axis]=candidate
                    break
    spatial=set(d for coord in geography.values() for d in coord.dimensions
                if d.casefold() not in {"time","tstep","lay","level","lev"})
    if not spatial:
        spatial={d for d in dimensions if d.casefold() not in {"time","tstep","lay","level","lev"}}
    from math import prod
    spatial_size=prod(n for d,n in zip(dimensions,shape) if d in spatial)
    width=max(1,int(MAX_SAMPLES**(1/max(len(spatial),1))))
    selection={d:slice(None,None,1 if spatial_size<=MAX_GRID_PROBE else max(1,(n+width-1)//width))
               if d in spatial else slice(0,1)
               for d,n in zip(dimensions,shape)}
    def array(v):
        return np.ma.asarray(v[tuple(selection[d] for d in v.dimensions)],dtype=float).filled(np.nan)
    values=array(var)
    finite=np.isfinite(values)
    flags=any(k in var.ncattrs() for k in ("ancillary_variables","quality_note")) or any(
        "flag_meanings" in v.ncattrs() or "qc" in k.casefold().split("_") for k,v in ds.variables.items()
    )
    quality_sample=finite.ravel()[::max(1,(finite.size+MAX_SAMPLES-1)//MAX_SAMPLES)]
    quality=dict(sampled=int(quality_sample.size),quality_flag_metadata_present=flags)
    if quality_sample.size>=MIN_SAMPLES:
        quality["finite_percent"]=int(round(100*quality_sample.mean()/5)*5)
    time_key=None
    flag=ds.variables.get("TFLAG")
    if flag is not None and flag.dimensions==("TSTEP","VAR","DATE-TIME") and "TSTEP" in dimensions:
        names=str(getattr(ds,"VAR-LIST",""))
        members=[names[i:i+16].strip() for i in range(0,len(names),16)]
        if name in members and members.index(name)<flag.shape[1] and shape[dimensions.index("TSTEP")]:
            date,clock=map(int,flag[0,members.index(name),:])
            # I/O API static fields have one record and TSTEP=0.
            step=int(getattr(ds,"TSTEP",-1))
            if step==0 and shape[dimensions.index("TSTEP")]==1:
                time_key=("ioapi_static",)
            elif step>0 and date>0 and clock>=0:
                time_key=("ioapi",date,clock,step)
    elif "time" in dimensions and "time" in ds.variables:
        clock=ds.variables["time"]
        if clock.dimensions==("time",) and getattr(clock,"units",None):
            time_key=("coordinate",float(clock[0]),clock.units,getattr(clock,"calendar","standard"))
    vertical=[d for d in dimensions if d.casefold() in {"lay","level","lev","layer","z"}]
    vertical_key=("surface",) if not vertical else None
    if vertical==["LAY"] and shape[dimensions.index("LAY")]==1 and flag is not None:
        vertical_key=("ioapi_single_layer",tuple(getattr(ds,"VGLVLS",())),getattr(ds,"VGTYP",None))
    if not geography and flag is not None and var.dimensions[-2:]==("ROW","COL"):
        if spatial_size<=MAX_GRID_PROBE:
            import xarray as xr
            from ecodata.models.engineering.ports import grid
            attrs={key:ds.getncattr(key) for key in ds.ncattrs()}
            if (int(attrs.get("NROWS",-1)),int(attrs.get("NCOLS",-1)))==var.shape[-2:]:
                try:
                    longitude,latitude=grid(xr.Dataset(attrs=attrs))
                    result_coordinates=np.column_stack((latitude.ravel(),longitude.ravel()))
                except (ValueError, KeyError, TypeError):
                    result_coordinates=None
            else:
                result_coordinates=None
        else:
            result_coordinates=None
    else:
        result_coordinates=None
    geodetic_basis=None
    if flag is not None and "GDTYP" in ds.ncattrs():
        geodetic_basis=("ioapi",*(str(getattr(ds,k,"")) for k in
                    ("GDTYP","P_ALP","P_BET","P_GAM","XCENT","YCENT")))
    elif {"lat","lon"}<=geography.keys():
        mapping=ds.variables.get(str(getattr(var,"grid_mapping","")))
        if mapping is not None:
            geodetic_basis=("cf",*((key,str(mapping.getncattr(key))) for key in sorted(mapping.ncattrs())))
        elif all(str(getattr(geography[axis],"units","")).casefold() in names for axis,names in (
                ("lat",{"degrees_north","degree_north"}),("lon",{"degrees_east","degree_east"}))):
            geodetic_basis=("cf_geographic",str(getattr(ds,"crs","")))
    result=dict(values=values.reshape(-1),quality=quality,time_key=time_key,vertical_key=vertical_key,
                temporal_semantics=str(getattr(var,"temporal_semantics","")) or str(getattr(var,"cell_methods","")),
                geodetic_basis=geodetic_basis)
    if result_coordinates is not None and len(result_coordinates)==values.size:
        result["coordinates"]=result_coordinates
    if set(geography)=={"lat","lon"} and values.size:
        coords=[]
        for coord in (geography["lat"],geography["lon"]):
            order=[coord.dimensions.index(d) for d in dimensions if d in coord.dimensions]
            axes=array(coord).transpose(order)
            coords.append(np.broadcast_to(axes.reshape([values.shape[i] if d in coord.dimensions else 1
                                                        for i,d in enumerate(dimensions)]),values.shape).ravel())
        latitude,longitude=coords
        if np.all(np.abs(latitude[np.isfinite(latitude)])<=90) and np.all(np.abs(longitude[np.isfinite(longitude)])<=360):
            result["coordinates"]=np.column_stack((latitude,longitude))
    return result


def spatial_coverage(a,b):
    import numpy as np

    if "coordinates" not in a or "coordinates" not in b:
        return dict(status="unavailable",basis="verified_geographic_coordinates_missing")
    def envelope(points):
        points=points[np.isfinite(points).all(axis=1)]
        if len(points)<MIN_SAMPLES or np.ptp(points[:,1])>180:
            raise ValueError("unsupported geographic extent")
        return points.min(axis=0), points.max(axis=0)
    a0,a1=envelope(a["coordinates"])
    b0,b1=envelope(b["coordinates"])
    return dict(status="checked",basis="sampled_geographic_envelopes_not_cell_overlap",
                relationship="overlap" if np.all(np.maximum(a0,b0)<=np.minimum(a1,b1)) else "disjoint")


def position_agreement(a,b,units):
    import numpy as np
    from scipy.spatial import cKDTree

    if units["status"] not in {"equivalent","convertible"}:
        return dict(status="unavailable",basis="physical_quantity_or_unit_unconfirmed")
    if "coordinates" not in a or "coordinates" not in b:
        return dict(status="unavailable",basis="verified_geographic_coordinates_missing")
    if a.get("geodetic_basis") is None or a["geodetic_basis"]!=b.get("geodetic_basis"):
        return dict(status="unavailable",basis="coordinate_reference_not_verified_equal")
    xy=[]
    vals=[]
    for source in (a,b):
        mask=np.isfinite(source["values"]) & np.isfinite(source["coordinates"]).all(axis=1)
        xy.append(source["coordinates"][mask])
        vals.append(source["values"][mask])
    if not len(xy[0]) or not len(xy[1]):
        return dict(status="unavailable",basis="no_valid_sample_coordinates")
    distance,indices=cKDTree(xy[1]).query(xy[0],distance_upper_bound=1e-5)
    keep=np.isfinite(distance)
    matches=indices[keep]
    _,first,frequency=np.unique(matches,return_index=True,return_counts=True)
    first=first[frequency==1]
    first=first[::max(1,(len(first)+MAX_SAMPLES-1)//MAX_SAMPLES)]
    denominator=len(first)
    if denominator<MIN_SAMPLES:
        return dict(status="unavailable",basis="too_few_matching_sample_locations",matched_under_ten=True)
    left=vals[0][keep][first]
    right=vals[1][matches[first]]*units["scale"]+units["offset"]
    diff=left-right
    if not np.isfinite(diff).all():
        return dict(status="unavailable",basis="nonfinite_converted_differences")
    result=dict(status="checked",basis="sampled_same_time_position_pairs_no_interpolation",sampled_pairs=denominator)
    if units["quantity"] in CATEGORICAL:
        result["agreement_percent"]=int(round(100*np.mean(diff==0)/5)*5)
    else:
        result["mae"]=float(f"{np.mean(abs(diff)):.2g}")
        maximum=float(np.max(abs(diff)))
        rmse=maximum*np.sqrt(np.mean((diff/maximum)**2)) if maximum else 0.
        result["rmse"]=float(f"{rmse:.2g}")
        result["difference_unit"]=units["target_unit"] if units["target_unit"] in {"m","1","m2 m-2"} else "target_unit"
    return result
