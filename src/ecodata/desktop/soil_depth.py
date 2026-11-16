"""Read declared soil-depth support without treating ordinal layers as depths."""
import re

SOIL_DIMS={"soil_layer","soil_layers","soil_layers_stag","soil_depth","soil_level","soil_lev"}
DEPTH_NAMES={"depth","soil_depth","depth_below_surface","depth_to_soil_layer","soil_layer_depth","zs"}
LAYER_NAMES={"layer","layers","lay","soil_layer","soil_layers","soil_layers_stag","soil_level","soil_lev"}
LENGTH={"mm","millimeter","millimeters","cm","centimeter","centimeters","m","meter","meters","metre","metres"}
INTERVAL=re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:[-–—~～]|至|to)\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b",re.I)
ORDINAL=re.compile(r"\b(?:soil[_ -]*)?layer[_ -]*(\d+)\b|第\s*(\d+)\s*层",re.I)


def text_support(*values):
    text=" ".join(str(value or "") for value in values)
    match=INTERVAL.search(text)
    if match and re.search(r"soil|depth|layer|bulk.?dens|土|深度",text,re.I):
        a,b=map(float,match.group(1,2))
        if a!=b:
            return dict(kind="depth_interval",source="declared_field_text",units=match[3].casefold(),
                        positive="down",bounds=[[min(a,b),max(a,b)]])
    ordinal=ORDINAL.search(text)
    if ordinal:
        return dict(kind="layer_index",source="declared_field_text",index=int(ordinal[1] or ordinal[2]))
    return None


def read_profile(variable,axis,*,maximum_values=16384):
    import numpy as np
    if axis not in variable.dimensions or variable.size>maximum_values:
        return None
    arr=np.ma.asarray(variable[:],dtype=float).filled(np.nan)
    if not arr.size or not np.isfinite(arr).all():
        return None
    position=variable.dimensions.index(axis)
    arr=np.moveaxis(arr,position,0)
    # A 1-D profile or a coordinate explicitly constant along other axes.
    return arr


def netcdf_depth_axes(dataset):
    return {name:var for name,var in dataset.variables.items()
            if name.casefold() in DEPTH_NAMES or str(getattr(var,"standard_name","")).casefold() in DEPTH_NAMES}


def netcdf_support(dataset,variable,axes,cache):
    named=set(variable.dimensions)|set(str(getattr(variable,"coordinates","")).split())
    text=text_support(variable.name,getattr(variable,"long_name",""),getattr(variable,"var_desc",""))
    for name,coordinate in axes.items():
        name_key=name.casefold()
        if name not in named and not (name_key in DEPTH_NAMES and set(coordinate.dimensions)<=set(variable.dimensions)):
            continue
        axis=next((dim for dim in coordinate.dimensions if dim.casefold() in SOIL_DIMS|LAYER_NAMES|DEPTH_NAMES),None)
        if axis is None and coordinate.ndim==1:
            axis=coordinate.dimensions[0]
        if axis is None or axis not in variable.dimensions:
            continue
        if name not in cache:
            cache[name]=describe_axis(dataset,coordinate,axis)
        return cache[name]
    if text and text["kind"]=="depth_interval":
        return text
    dimension=next((dim for dim in variable.dimensions if dim.casefold() in SOIL_DIMS),None)
    if dimension:
        return dict(kind="layer_index",source="soil_dimension",dimension=dimension,
                    layer_count=len(dataset.dimensions[dimension]))
    if text:
        return text
    if re.search(r"soil|bulk.?dens|土壤",f"{variable.name} {getattr(variable,'long_name','')} {getattr(variable,'var_desc','')}",re.I):
        dim=next((dim for dim in variable.dimensions if dim.casefold() in LAYER_NAMES),None)
        if dim:
            return dict(kind="layer_index",source="soil_dimension",dimension=dim,layer_count=len(dataset.dimensions[dim]))
    return None


def describe_axis(dataset,coordinate,axis):
    import numpy as np
    units=str(getattr(coordinate,"units","")).strip()
    positive=str(getattr(coordinate,"positive","down")).casefold()
    bounds_name=str(getattr(coordinate,"bounds","")).strip()
    result=dict(kind="depth_coordinate",source="netcdf_coordinate",coordinate=coordinate.name,dimension=axis,
                units=units or None,positive=positive if positive in {"up","down"} else None,
                bounds_variable=bounds_name or None,dimensions=list(coordinate.dimensions))
    if units.casefold() not in LENGTH:
        result["kind"]="layer_index"
        result["status"]="depth_unit_missing_or_unrecognized"
        return result
    values=read_profile(coordinate,axis)
    if values is not None:
        layers=values.reshape(values.shape[0],-1)
        if np.allclose(layers,layers[:,0,None],rtol=1e-7,atol=1e-9):
            result["centers"]=layers[:,0].tolist()
        else:
            result["status"]="depth_varies_over_other_dimensions"
    else:
        result["status"]="depth_coordinates_not_readable_as_profile"
    if bounds_name and bounds_name in dataset.variables:
        bound_var=dataset.variables[bounds_name]
        declared_units=str(getattr(bound_var,"units",units)).strip()
        limits=read_profile(bound_var,axis)
        other_axes=[dim for dim in bound_var.dimensions if dim!=axis]
        endpoint=next((i for i,dim in enumerate(other_axes,1) if dim not in coordinate.dimensions
                       and bound_var.shape[bound_var.dimensions.index(dim)]==2),None)
        if limits is not None and endpoint is not None and declared_units.casefold() in LENGTH:
            limits=np.moveaxis(limits,endpoint,-1).reshape(limits.shape[0],-1,2)
            if np.allclose(limits,limits[:,0:1,:],rtol=1e-7,atol=1e-9):
                result.update(kind="depth_interval",bounds=limits[:,0,:].tolist(),bounds_units=declared_units)
                result.pop("status",None)
            else:
                result["status"]="bounds_vary_over_other_dimensions"
    elif bounds_name:
        result["status"]="declared_bounds_variable_unavailable"
    return result


def compare(left,right):
    import numpy as np
    from ecodata.core.exceptions import SchemaError
    from ecodata.engines.units import UnitEngine
    if not left or not right or any(item.get("kind")!="depth_interval" or not item.get("bounds")
                                    or not item.get("positive") for item in (left,right)):
        return "bounds_unconfirmed"
    supports=[]
    for item in (left,right):
        try:
            scale,offset=UnitEngine().affine(item.get("bounds_units") or item["units"],"m")
        except (SchemaError,TypeError,ValueError):
            return "units_unconfirmed"
        values=np.asarray(item["bounds"],dtype=float)*scale+offset
        if item["positive"]=="up":
            values=-values
        supports.append(np.sort(values,axis=1))
    return ("identical_depth_intervals" if supports[0].shape==supports[1].shape and
            np.allclose(*supports,rtol=1e-6,atol=1e-6) else "different_depth_intervals")

