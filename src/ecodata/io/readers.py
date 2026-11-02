from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle


REQUIRED_COLUMNS = (
    "station_id",
    "time",
    "latitude",
    "longitude",
    "variable",
    "value",
    "unit",
    "qc_flag",
)


@dataclass(frozen=True)
class CsvReadConfig:
    dataset_id: str
    source_name: str
    time_zone: str = "UTC"
    temporal_semantics: str = "instantaneous"
    data_kind: str = "continuous"
    crs: str = "EPSG:4326"


class StationCsvReader:
    def read(self, source: Path, config: CsvReadConfig) -> EcoDataBundle:
        frame = pd.read_csv(source)
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise SchemaError(f"Missing required CSV columns: {', '.join(missing)}")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        if frame.empty:
            raise SchemaError("CSV contains no observations")

        parsed = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        if parsed.isna().any():
            raise SchemaError("Column 'time' contains invalid timestamps")
        frame["time"] = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

        numeric = ("latitude", "longitude", "value")
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[list(numeric)].isna().any().any():
            raise SchemaError("Latitude, longitude and value must be numeric and non-missing")
        if not frame["latitude"].between(-90, 90).all():
            raise SchemaError("Latitude is outside [-90, 90]")
        if not frame["longitude"].between(-180, 180).all():
            raise SchemaError("Longitude is outside [-180, 180]")

        text_columns = ("station_id", "variable", "unit", "qc_flag")
        if frame[list(text_columns)].isna().any().any():
            raise SchemaError("Station, variable, unit and quality flag must be non-missing")
        for column in text_columns:
            frame[column] = frame[column].astype(str)

        keys = ["station_id", "variable", "time"]
        if frame.duplicated(keys).any():
            raise SchemaError("Duplicate station/variable/time observations are not allowed")
        ordered = frame.groupby(["station_id", "variable"], sort=False)["time"].apply(
            lambda values: values.is_monotonic_increasing
        )
        if not ordered.all():
            raise SchemaError(
                "Observations must be chronological within each station/variable group; "
                "out-of-order input is rejected instead of silently sorted"
            )
        unit_counts = frame.groupby(["station_id", "variable"])["unit"].nunique()
        if (unit_counts != 1).any():
            raise SchemaError("A station/variable group contains inconsistent units")
        coord_counts = frame.groupby("station_id")[["latitude", "longitude"]].nunique()
        if (coord_counts > 1).any().any():
            raise SchemaError("A fixed station_id contains changing coordinates")

        frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
        dataset = _frame_to_dataset(frame)
        dataset.attrs.update(
            {
                "dataset_id": config.dataset_id,
                "source_name": config.source_name,
                "source_format": "CSV",
                "crs": config.crs,
                "time_zone": "UTC",
                "source_time_zone": config.time_zone,
                "temporal_semantics": config.temporal_semantics,
                "data_kind": config.data_kind,
                "processing_stage": "read",
                "Conventions": "CF-1.10",
            }
        )
        dataset["value"].attrs.update(
            {
                "long_name": "environmental observation",
                "temporal_semantics": config.temporal_semantics,
                "data_kind": config.data_kind,
                "units": "stored per observation in variable 'unit'",
                "ancillary_variables": "qc_flag",
            }
        )
        dataset["qc_flag"].attrs["long_name"] = "source or processing quality flag"
        return EcoDataBundle(dataset=dataset)


class NetCdfReader:
    def read(self, source: Path) -> EcoDataBundle:
        with xr.open_dataset(source, engine="h5netcdf") as opened:
            dataset = opened.load()
        return EcoDataBundle(dataset=dataset)


def _frame_to_dataset(frame: pd.DataFrame) -> xr.Dataset:
    observation = np.arange(len(frame), dtype=np.int64)
    return xr.Dataset(
        data_vars={
            "station_id": ("observation", frame["station_id"].to_numpy(dtype=str)),
            "time": ("observation", frame["time"].to_numpy(dtype="datetime64[ns]")),
            "latitude": ("observation", frame["latitude"].to_numpy(dtype=float)),
            "longitude": ("observation", frame["longitude"].to_numpy(dtype=float)),
            "variable": ("observation", frame["variable"].to_numpy(dtype=str)),
            "value": ("observation", frame["value"].to_numpy(dtype=float)),
            "unit": ("observation", frame["unit"].to_numpy(dtype=str)),
            "qc_flag": ("observation", frame["qc_flag"].to_numpy(dtype=str)),
        },
        coords={"observation": observation},
    )


def dataset_to_frame(dataset: xr.Dataset) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataset.data_vars))
    if missing:
        raise SchemaError(f"Dataset is missing required variables: {', '.join(missing)}")
    return pd.DataFrame({name: dataset[name].values for name in REQUIRED_COLUMNS})

