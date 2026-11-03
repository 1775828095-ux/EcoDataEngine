#!/usr/bin/env python3
"""Add conservative standardized-preparation scenarios to selected models."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "src" / "ecodata" / "models" / "profiles"


def field(name: str, standard_name: str, unit: str, data_type: str = "continuous",
          aliases: list[str] | None = None, required: bool = True) -> dict:
    value: dict = {
        "name": name,
        "standard_name": standard_name,
        "unit": unit,
        "data_type": data_type,
        "aliases": aliases or [],
        "required": required,
    }
    return value


SCENARIOS = {
    "swat_plus": {
        "id": "swat_plus_standardized_v1",
        "name": "SWAT+ 标准化准备（非原生文本输出）",
        "adapter": "standardized",
        "status": "standardized_preprocessing",
        "output_format": "NetCDF",
        "native_model_input": False,
        "format_scope": "standardized_only",
        "target_crs": "EPSG:4326",
        "record_only_roles": [],
        "derivable_fields": [],
        "native_gaps": [
            "未实现 SWAT+ 固定/自由格式原生写出器",
            "未实现 file.cio、HRU、channel、weather .cli 等原生索引文件",
            "未使用对应版本 SWAT+ 可执行文件读取平台输出",
        ],
        "inputs": [
            {
                "role": "simulation_control", "name": "模拟控制", "required": True,
                "formats": ["JSON", "CSV", "Excel"], "dimensions": ["record"],
                "crs": "不适用", "time": "模拟起止时间", "space": "项目域",
                "fields": [
                    field("simulation_start", "simulation_start_time", "ISO8601", "datetime", ["start"]),
                    field("simulation_end", "simulation_end_time", "ISO8601", "datetime", ["end"]),
                    field("time_step", "time_step", "day", "string_identifier", ["dt"], False),
                ],
            },
            {
                "role": "topography", "name": "地形", "required": True,
                "formats": ["GeoTIFF", "NetCDF"], "dimensions": ["y", "x"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "静态", "space": "覆盖流域",
                "fields": [field("elevation", "surface_altitude", "m", aliases=["dem", "height", "value"])],
            },
            {
                "role": "landuse", "name": "土地利用", "required": True,
                "formats": ["GeoTIFF", "NetCDF"], "dimensions": ["y", "x"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "静态", "space": "覆盖流域",
                "fields": [field("land_cover_class", "land_cover_class", "class_id", "categorical", ["land_cover", "landuse", "land_class"])],
            },
            {
                "role": "soil", "name": "土壤", "required": True,
                "formats": ["GeoTIFF", "NetCDF"], "dimensions": ["y", "x"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "静态", "space": "覆盖流域",
                "fields": [
                    field("sand", "soil_sand_content", "g/kg", aliases=["sand_content"]),
                    field("silt", "soil_silt_content", "g/kg", aliases=["silt_content"]),
                    field("clay", "soil_clay_content", "g/kg", aliases=["clay_content"]),
                    field("bulk_density", "soil_bulk_density", "kg/dm3", aliases=["bdod"], required=False),
                ],
            },
            {
                "role": "weather", "name": "气象驱动", "required": True,
                "formats": ["NetCDF", "CSV"], "dimensions": ["time"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "日或子日", "space": "覆盖流域",
                "fields": [
                    field("precipitation", "precipitation_amount", "mm/day", aliases=["prcp"]),
                    field("air_temperature_max", "air_temperature_max", "degC", aliases=["tmax"]),
                    field("air_temperature_min", "air_temperature_min", "degC", aliases=["tmin"]),
                    field("solar_radiation", "surface_downwelling_shortwave_flux", "W/m2", aliases=["srad"], required=False),
                    field("relative_humidity", "relative_humidity", "%", aliases=["rh"], required=False),
                    field("wind_speed", "wind_speed", "m/s", aliases=["wind"], required=False),
                ],
            },
            {
                "role": "hydrologic_units", "name": "流域对象", "required": True,
                "formats": ["GeoJSON", "Shapefile ZIP", "CSV"], "dimensions": ["record"],
                "crs": "EPSG:4326 或可靠投影", "time": "静态", "space": "流域拓扑",
                "fields": [
                    field("object_id", "watershed_object_id", "", "string_identifier", ["id", "hru_id"]),
                    field("longitude", "longitude", "degree_east", aliases=["lon"]),
                    field("latitude", "latitude", "degree_north", aliases=["lat"]),
                ],
            },
        ],
    },
    "wrf": {
        "id": "wrf_standardized_v1",
        "name": "WRF 标准化气象准备（非 WPS intermediate）",
        "adapter": "standardized",
        "status": "standardized_preprocessing",
        "output_format": "NetCDF",
        "native_model_input": False,
        "format_scope": "standardized_only",
        "target_crs": "EPSG:4326",
        "record_only_roles": ["surface_pressure", "temperature", "humidity", "wind"],
        "derivable_fields": [],
        "native_gaps": [
            "未实现 WPS intermediate/GRIB 原生写出",
            "未实现 namelist 生成和 real.exe 输入验证",
            "官方原始再分析场未下载，仅使用官方配置文本和小型真实气象样本",
        ],
        "inputs": [
            {
                "role": "static_geography", "name": "静态地理场", "required": True,
                "formats": ["GeoTIFF", "NetCDF"], "dimensions": ["y", "x"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "静态", "space": "覆盖 WRF 域",
                "fields": [
                    field("elevation", "surface_altitude", "m", aliases=["dem", "height", "value"]),
                    field("land_cover_class", "land_cover_class", "class_id", "categorical", ["land_cover", "landuse"], False),
                ],
            },
            {
                "role": "meteorological_analysis", "name": "气象分析场", "required": True,
                "formats": ["NetCDF", "CSV"], "dimensions": ["time", "y", "x"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "边界覆盖模拟窗口", "space": "覆盖 WRF 域",
                "fields": [
                    field("air_temperature", "air_temperature", "degC", aliases=["temperature", "t2m"]),
                    field("relative_humidity", "relative_humidity", "%", aliases=["humidity", "rh"]),
                    field("wind_speed", "wind_speed", "m/s", aliases=["wind"]),
                    field("surface_pressure", "surface_air_pressure", "hPa", aliases=["pressure"], required=False),
                    field("precipitation", "precipitation_amount", "mm", aliases=["prcp"], required=False),
                ],
            },
            {"role": "surface_pressure", "name": "地面气压", "required": False, "formats": ["NetCDF", "GRIB"], "dimensions": ["time", "y", "x"], "crs": "覆盖域", "time": "驱动场", "space": "覆盖域", "fields": [field("surface_pressure", "surface_air_pressure", "hPa")]},
            {"role": "temperature", "name": "温度", "required": False, "formats": ["NetCDF", "GRIB"], "dimensions": ["time", "level", "y", "x"], "crs": "覆盖域", "time": "驱动场", "space": "覆盖域", "fields": [field("air_temperature", "air_temperature", "K", aliases=["t"])]},
            {"role": "humidity", "name": "湿度", "required": False, "formats": ["NetCDF", "GRIB"], "dimensions": ["time", "level", "y", "x"], "crs": "覆盖域", "time": "驱动场", "space": "覆盖域", "fields": [field("relative_humidity", "relative_humidity", "%", aliases=["rh"])]},
            {"role": "wind", "name": "风场", "required": False, "formats": ["NetCDF", "GRIB"], "dimensions": ["time", "level", "y", "x"], "crs": "覆盖域", "time": "驱动场", "space": "覆盖域", "fields": [field("wind_speed", "wind_speed", "m/s", aliases=["ws"])]},
        ],
    },
    "camx": {
        "id": "camx_standardized_v1",
        "name": "CAMx 标准化输入准备（非 Fortran binary）",
        "adapter": "standardized",
        "status": "standardized_preprocessing",
        "output_format": "NetCDF",
        "native_model_input": False,
        "format_scope": "standardized_only",
        "target_crs": "EPSG:4326",
        "record_only_roles": ["boundary_concentrations", "chemistry_parameters"],
        "derivable_fields": [],
        "native_gaps": [
            "未实现 CAMx Fortran binary 原生写出器",
            "未下载超过 3 GB 的官方完整 2 天测试输入和输出",
            "未使用官方 CAMx 可执行文件读取平台输出",
        ],
        "inputs": [
            {
                "role": "meteorology", "name": "气象场", "required": True,
                "formats": ["NetCDF", "CSV"], "dimensions": ["time", "y", "x"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "小时", "space": "覆盖 CAMx 域",
                "fields": [
                    field("wind_speed", "wind_speed", "m/s", aliases=["wind"]),
                    field("wind_direction", "wind_from_direction", "degree", aliases=["wind_dir"], required=False),
                    field("air_temperature", "air_temperature", "degC", aliases=["temperature", "t2m"]),
                    field("relative_humidity", "relative_humidity", "%", aliases=["humidity", "rh"]),
                    field("surface_pressure", "surface_air_pressure", "hPa", aliases=["pressure"], required=False),
                ],
            },
            {
                "role": "emissions", "name": "排放", "required": True,
                "formats": ["CSV", "Excel", "JSON"], "dimensions": ["record"],
                "crs": "EPSG:4326", "time": "小时或配置周期", "space": "点源或网格排放",
                "fields": [
                    field("source_id", "source_id", "", "string_identifier", ["id"]),
                    field("longitude", "longitude", "degree_east", aliases=["lon"]),
                    field("latitude", "latitude", "degree_north", aliases=["lat"]),
                    field("emission_rate", "emission_rate", "source-specific", aliases=["rate", "value"]),
                    field("species", "chemical_species", "", "string_identifier", ["pollutant"], False),
                ],
            },
            {
                "role": "initial_concentrations", "name": "初始浓度", "required": True,
                "formats": ["NetCDF"], "dimensions": ["time", "y", "x"],
                "crs": "EPSG:4326 或可靠投影网格", "time": "初始时刻", "space": "覆盖 CAMx 域",
                "fields": [field("PM25", "pm25_mass_concentration", "ug/m3", aliases=["PM2.5", "pm25"])],
            },
            {"role": "boundary_concentrations", "name": "边界浓度", "required": False, "required_for_native": True, "formats": ["Fortran binary"], "dimensions": ["time", "layer", "perimeter"], "crs": "目标CAMx网格一致", "time": "模拟边界时序", "space": "侧边界", "fields": [field("species_concentration", "chemical_species_concentration", "species-specific")]},
            {"role": "chemistry_parameters", "name": "化学参数", "required": False, "required_for_native": True, "formats": ["ASCII"], "dimensions": ["species"], "crs": "不适用", "time": "静态", "space": "化学机制", "fields": [field("species_name", "chemical_species", "", "string_identifier")]},
        ],
    },
}


def main() -> None:
    for model_id, scenario in SCENARIOS.items():
        path = PROFILES / f"{model_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile["preparation_scenario"] = scenario
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"updated {path.name}")


if __name__ == "__main__":
    main()
