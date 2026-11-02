"""Configuration-like engineering profiles; core logic contains no model branches."""
from .general import ModelEngineeringProfile


_COMMON = ("observation", "forcing", "initial_condition", "boundary_condition",
           "source", "load", "static_field", "terrain", "landuse", "hydrology",
           "meteorology", "calibration", "validation", "quality_auxiliary", "emission")

PROFILES = {
    "wasp": ModelEngineeringProfile("WASP", ("POLYGON", "NETWORK_EDGE"), ("POLYGON",), _COMMON,
        {"max_time_gap": "7D"}, {"evidence": "河流入口与 segment 拓扑"}, {},
        ({"source_type": "POINT", "target_type": "POLYGON", "method": "point_to_polygon"},),
        validation_status="real_case_partial"),
    "calpuff": ModelEngineeringProfile("CALPUFF", ("GRID", "POINT"), ("GRID",), _COMMON,
        {}, {"evidence": "model domain edge"}, {},
        ({"source_type": "POINT", "target_type": "GRID", "method": "point_to_grid"},)),
    "swat_plus": ModelEngineeringProfile("SWAT+", ("POLYGON", "NETWORK_EDGE", "POINT"), ("POLYGON", "NETWORK_EDGE"), _COMMON,
        {}, {"evidence": "watershed/reach inflow"}, {},
        ({"source_type": "POINT", "target_type": "POLYGON", "method": "point_to_polygon"},)),
    "wrf": ModelEngineeringProfile("WRF", ("GRID", "LAYER"), ("GRID", "LAYER"), _COMMON,
        {}, {"evidence": "domain boundary / initial field"}, {},
        ({"source_type": "POINT", "target_type": "GRID", "method": "point_to_grid"},)),
    "camx": ModelEngineeringProfile("CAMx", ("GRID", "LAYER"), ("GRID", "LAYER"), _COMMON,
        {}, {"evidence": "lateral domain edge"}, {},
        ({"source_type": "POINT", "target_type": "GRID", "method": "point_to_grid"},)),
    "cmaq": ModelEngineeringProfile("CMAQ", ("GRID", "LAYER"), ("GRID", "LAYER"), _COMMON,
        {}, {"evidence": "lateral domain edge"}, {},
        ({"source_type": "POINT", "target_type": "GRID", "method": "point_to_grid"},),
        validation_status="real_case_partial"),
}


def get_engineering_profile(model):
    key = str(model).casefold().replace("+", "_plus").replace("-", "_")
    if key not in PROFILES:
        raise KeyError(f"尚未定义 {model} 的 ModelEngineeringProfile")
    return PROFILES[key]
