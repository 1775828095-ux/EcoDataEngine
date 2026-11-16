"""Decode the NOAA Global Hourly mandatory compound CSV fields."""
from dataclasses import dataclass
from math import isfinite

SIGNATURE = {"STATION","DATE","REPORT_TYPE","LATITUDE","LONGITUDE","ELEVATION"}
VALID_QUALITY = {"0","1","4","5","9"}


@dataclass(frozen=True)
class Component:
    name: str
    quantity: str
    unit: str
    offset: int
    quality_offset: int
    divisor: int
    missing: int


COLUMNS = {
    "TMP": (2,(Component("TMP.air_temperature","air_temperature","degC",0,1,10,9999),)),
    "SLP": (2,(Component("SLP.air_pressure_at_mean_sea_level","air_pressure_at_mean_sea_level","hPa",0,1,10,99999),)),
    "VIS": (4,(Component("VIS.visibility_in_air","visibility_in_air","m",0,1,1,999999),)),
    "WND": (5,(Component("WND.wind_from_direction","wind_from_direction","degree",0,1,1,999),
               Component("WND.wind_speed","wind_speed","m/s",3,4,10,9999))),
}
COMPONENTS = {component.name:(code,width,component) for code,(width,components) in COLUMNS.items()
              for component in components}


def is_global_hourly(headers):
    names=set(headers)
    return SIGNATURE <= names and bool(names & COLUMNS.keys())


def decode(cell, width, component):
    parts=str(cell).split(",")
    if len(parts)!=width:
        return None
    token=parts[component.offset].strip()
    if parts[component.quality_offset].strip() not in VALID_QUALITY:
        return None
    try:
        value=int(token)
    except ValueError:
        return None
    if value==component.missing:
        return None
    return value/component.divisor


def add_variables(dataset):
    if not is_global_hourly(dataset.variables):
        return
    for name,(source,width,component) in COMPONENTS.items():
        if source not in dataset or name in dataset:
            continue
        original=dataset[source]
        values=[decode(cell,width,component) for cell in original.values]
        dataset[name]=(original.dims,[value if value is not None and isfinite(value) else float("nan") for value in values],
                       {"units":component.unit,"standard_name":component.quantity,"source_variable":source,
                        "temporal_semantics":"instantaneous","quality_note":"NOAA ISD component quality code"})

