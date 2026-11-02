"""Minimum field-level preparation contracts for legacy requirement indexes.

These contracts standardise supplied data. They are not native model file specs.
"""
from copy import deepcopy


def _f(name, unit="", kind="continuous", aliases=(), required=True):
    return {"name": name, "standard_name": name, "unit": unit,
            "data_type": kind, "aliases": list(aliases), "required": required}


TEMPLATES = {
    "source": [_f("source_id", kind="string_identifier"), _f("x", "m"), _f("y", "m"),
               _f("emission_rate", "g/s"), _f("source_height", "m"),
               _f("source_type", kind="categorical")],
    "receptor": [_f("receptor_id", kind="string_identifier"), _f("x", "m"), _f("y", "m"),
                 _f("elevation", "m", aliases=("terrain_height",))],
    "meteorology": [_f("air_temperature", "K", aliases=("temperature", "T")),
                    _f("wind_speed", "m/s"), _f("wind_direction", "degree"),
                    _f("surface_air_pressure", "Pa", aliases=("pressure",)),
                    _f("relative_humidity", "%", required=False),
                    _f("precipitation", "mm/day", required=False)],
    "terrain": [_f("elevation", "m", aliases=("dem", "topography"))],
    "bathymetry": [_f("segment_id", kind="string_identifier"), _f("bottom_elevation", "m"),
                   _f("width", "m"), _f("length", "m")],
    "flow": [_f("location_id", kind="string_identifier", aliases=("segment_id", "branch_id")),
             _f("flow", "m^3/s", aliases=("discharge",))],
    "water_state": [_f("location_id", kind="string_identifier", aliases=("segment_id", "node_id")),
                    _f("variable", kind="string_identifier"),
                    _f("value", "variable-specific"), _f("quality_flag", kind="categorical", required=False)],
    "domain": [_f("longitude", "degree_east", aliases=("lon",)),
               _f("latitude", "degree_north", aliases=("lat",)),
               _f("area", "m^2"), _f("land_mask", "1", kind="categorical")],
    "surface": [_f("land_cover", kind="categorical"), _f("soil_type", kind="categorical"),
                _f("elevation", "m"), _f("vegetation_fraction", "1", required=False)],
    "grid": [_f("cell_id", kind="string_identifier"), _f("x", "m"), _f("y", "m"),
             _f("elevation", "m", required=False), _f("active", "1", kind="categorical", required=False)],
    "node": [_f("node_id", kind="string_identifier"), _f("elevation", "m"),
             _f("x", "m", required=False), _f("y", "m", required=False)],
    "link": [_f("link_id", kind="string_identifier"), _f("from_node", kind="string_identifier"),
             _f("to_node", kind="string_identifier"), _f("length", "m"),
             _f("diameter", "m", required=False)],
    "precipitation": [_f("precipitation", "mm/day", aliases=("rainfall", "prcp"))],
    "temperature": [_f("air_temperature_max", "degC", aliases=("tmax",)),
                    _f("air_temperature_min", "degC", aliases=("tmin",))],
    "soil": [_f("soil_id", kind="string_identifier"), _f("layer_depth", "m"),
             _f("sand_fraction", "1"), _f("clay_fraction", "1"),
             _f("saturated_hydraulic_conductivity", "m/day")],
    "landuse": [_f("landuse_id", kind="string_identifier"), _f("land_cover", kind="categorical"),
                _f("fraction", "1", required=False)],
    "hru": [_f("hru_id", kind="string_identifier"), _f("area", "m^2"),
            _f("soil_id", kind="string_identifier"), _f("landuse_id", kind="string_identifier")],
    "forcing": [_f("air_temperature", "K"), _f("precipitation", "kg/m^2/s"),
                _f("surface_air_pressure", "Pa"), _f("wind_speed", "m/s"),
                _f("shortwave_radiation", "W/m^2", required=False),
                _f("longwave_radiation", "W/m^2", required=False)],
    "concentration": [_f("location_id", kind="string_identifier"),
                      _f("species", kind="string_identifier"), _f("concentration", "kg/m^3")],
    "transport": [_f("porosity", "1"), _f("longitudinal_dispersivity", "m"),
                  _f("transverse_dispersivity", "m", required=False)],
    "emission": [_f("source_id", kind="string_identifier"), _f("species", kind="string_identifier"),
                 _f("emission_rate", "kg/s"), _f("x", "m", required=False), _f("y", "m", required=False)],
    "profile": [_f("profile_id", kind="string_identifier"), _f("time_fraction", "1"),
                _f("value", "1")],
    "mapping": [_f("source_id", kind="string_identifier"),
                _f("target_id", kind="string_identifier"), _f("fraction", "1")],
    "configuration": [_f("parameter", kind="string_identifier"),
                      _f("value", kind="string_identifier"), _f("unit", kind="string_identifier", required=False)],
    "location": [_f("longitude", "degree_east", aliases=("lon",)),
                 _f("latitude", "degree_north", aliases=("lat",)), _f("height", "m")],
    "time": [_f("start_time", kind="datetime"), _f("end_time", kind="datetime")],
}


ROLE_TEMPLATE = {
    "source_parameters": "source", "receptors": "receptor", "meteorology": "meteorology",
    "terrain": "terrain", "bathymetry": "bathymetry", "inflows": "flow",
    "initial_conditions": "water_state", "boundary_conditions": "water_state",
    "domain": "domain", "surface_dataset": "surface", "atmospheric_forcing": "forcing",
    "grid": "grid", "junctions": "node", "reservoirs": "node", "tanks": "node",
    "pipes": "link", "demands": "flow", "simulation_times": "time",
    "basin_model": "hru", "meteorologic_model": "configuration",
    "control_specifications": "configuration", "precipitation": "precipitation",
    "geometry": "grid", "flow_boundary_conditions": "flow", "plan": "configuration",
    "release_location": "location", "release_time": "time", "release_height": "location",
    "flow_transport_link": "mapping", "initial_concentration": "concentration",
    "transport_parameters": "transport", "sources_sinks": "concentration",
    "control_parameters": "configuration", "hru_parameters": "hru", "temperature": "temperature",
    "flow_model": "configuration", "transport_model": "configuration",
    "density_parameters": "configuration", "initial_head": "water_state",
    "emission_inventory": "emission", "temporal_profiles": "profile",
    "spatial_surrogates": "mapping", "chemical_speciation": "profile", "grid_definition": "grid",
    "weather": "meteorology", "soil": "soil", "landuse": "landuse", "topography": "terrain",
    "hydrologic_units": "hru", "simulation_control": "configuration",
    "options": "configuration", "raingages": "precipitation", "subcatchments": "hru",
    "nodes": "node", "links": "link", "air_temperature": "forcing", "pressure": "forcing",
    "shortwave": "forcing", "longwave": "forcing", "vapor_pressure": "forcing", "wind": "forcing",
    "soil_parameters": "soil", "vegetation_parameters": "landuse",
    "wrf_initial_boundary": "forcing", "anthropogenic_emissions": "emission",
    "biogenic_emissions": "emission", "chemical_initial_boundary": "concentration",
    "control": "configuration",
}


def apply_preparation_catalog(profile):
    """Promote a legacy index only when every role has a reviewed field template."""
    if profile.get("preparation_scenario"):
        if (profile.get("status") == "requirements_index"
                and all(item.get("fields") for item in profile["preparation_scenario"].get("inputs", []))):
            profile["status"] = profile["adapter_status"] = "standardized_preprocessing"
        return profile
    roles = []
    for item in profile["inputs"]:
        template = ROLE_TEMPLATE.get(item["name"])
        if not template:
            return profile
        roles.append({**deepcopy(item), "role": item["name"], "name": item["name"],
                      "fields": deepcopy(TEMPLATES[template]), "dimensions": [],
                      "crs": "source CRS retained; confirmation required when absent",
                      "time": item.get("temporal_resolution"), "space": item.get("spatial_resolution_km"),
                      "output_format": "NetCDF", "processing": {"structure_operation": "preserve"}})
    profile["preparation_scenario"] = {
        "id": profile["id"] + "_minimum_standardized_preparation",
        "name": profile["name"] + " 最小标准化数据准备", "status": "standardized_preprocessing",
        "adapter": "standardized", "inputs": roles, "target_crs": None,
        "output_format": "NetCDF", "native_model_input": False,
        "native_gaps": ["仅生成可审计的标准化数据包；模型控制文件、工程拓扑和未提供参数不自动生成",
                        "尚未通过对应官方模型程序读取和运行验证"],
    }
    profile["status"] = profile["adapter_status"] = "standardized_preprocessing"
    return profile
