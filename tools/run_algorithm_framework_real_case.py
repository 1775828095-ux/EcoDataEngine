"""Run one lightweight real-data path through the unified algorithm framework."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xarray as xr

from ecodata.algorithms import AlgorithmContext, AlgorithmPlan, AlgorithmSelector, UnifiedAlgorithmPipeline, create_default_registry
from ecodata.core.schemas import EcoDataBundle


PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\codex\数据\京津冀_PM25融合测试_201908")
GEOS = DATA / "02_GEOS_CF模拟背景场" / "geos_cf_pm25_201908_bth.nc"
GROUND = DATA / "05_对齐后的测试数据" / "diagnostic_ground_labels_Asia_Shanghai_UTC_plus_8" / "three_source_station_hourly.parquet"
OUTPUT = PROJECT / "output" / "algorithm_framework_real_case_20261013"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(GEOS, engine="h5netcdf") as opened:
        stamp = pd.Timestamp(opened.time.values[24])
        source = opened[["pm25_rh35_gcc"]].isel(time=24, lev=0, drop=True).load()
    source = source.rename(pm25_rh35_gcc="pm25")
    source.attrs["crs"] = source.attrs.get("crs", "EPSG:4326")
    source.pm25.attrs.update(units="ug m-3", data_kind="continuous", physical_variable="PM25", spatial_semantics="cell_mean")

    ground = pd.read_parquet(GROUND)
    frame = ground[pd.to_datetime(ground.geos_average_center_utc).eq(stamp)].copy()
    frame = frame.rename(columns={"longitude_as_stored": "longitude", "latitude_as_stored": "latitude", "pm25_as_stored": "value"})
    frame["unit"] = "ug m-3"
    observations = frame[["station_id", "longitude", "latitude", "value", "unit", "split"]].dropna()

    context = AlgorithmContext(
        sources=(EcoDataBundle(source),), variable="pm25", semantic_type="continuous",
        observations=observations, model_constraints={"nonnegative": True}, parameters={"power": 2.0},
    )
    registry = create_default_registry()
    screening = {key: value.__dict__ for key, value in AlgorithmSelector(registry).screen(context).items()}
    result = UnifiedAlgorithmPipeline(registry).run(context, AlgorithmPlan(
        "real_pm25_idw", ("fusion.idw",), 1.0,
        ("真实训练站观测插值到现有GEOS-CF网格；仅验证框架端到端，不声称高分辨率精度",),
    ))
    target = OUTPUT / "pm25_idw_framework_output.nc"
    result.data.dataset.to_netcdf(target, engine="h5netcdf")
    with xr.open_dataset(target, engine="h5netcdf") as check:
        if not check.pm25.notnull().any() or check.pm25.attrs.get("units") != "ug m-3":
            raise RuntimeError("real-case output reread validation failed")
    (OUTPUT / "algorithm_screening.json").write_text(json.dumps(screening, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUTPUT / "execution_report.json").write_text(json.dumps(result.report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUTPUT / "algorithm_catalog.json").write_text(json.dumps([item.as_dict() for item in registry.specs()], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "time": str(stamp), "training_stations": int((observations.split == 'train').sum()),
                      "output": str(target), "accuracy_claimed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
