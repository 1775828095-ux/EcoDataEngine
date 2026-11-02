from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult
from ecodata.io.readers import _frame_to_dataset, dataset_to_frame


@dataclass(frozen=True)
class LinearTimeConfig:
    frequency: str


class LinearTimeInterpolator:
    name = "temporal.linear"

    def __init__(self, config: LinearTimeConfig):
        self.config = config

    def validate_input(self, data: EcoDataBundle) -> None:
        attrs = data.dataset.attrs
        if attrs.get("temporal_semantics") != "instantaneous":
            raise SemanticError("Linear interpolation requires instantaneous time semantics")
        if attrs.get("data_kind") != "continuous":
            raise SemanticError("Linear interpolation requires continuous data")
        try:
            pd.tseries.frequencies.to_offset(self.config.frequency)
        except ValueError as exc:
            raise SchemaError(f"Invalid interpolation frequency: {self.config.frequency}") from exc

    def apply(self, data: EcoDataBundle) -> TransformResult:
        self.validate_input(data)
        source = dataset_to_frame(data.dataset)
        output_groups: list[pd.DataFrame] = []
        for (_, _), group in source.groupby(["station_id", "variable"], sort=True):
            output_groups.append(self._interpolate_group(group))

        output = pd.concat(output_groups, ignore_index=True)
        output = output.sort_values(["station_id", "variable", "time"], kind="stable")
        dataset = _frame_to_dataset(output.reset_index(drop=True))
        dataset.attrs.update(data.dataset.attrs)
        dataset.attrs["processing_stage"] = "time_interpolated"
        dataset.attrs["time_interpolation_method"] = "linear"
        dataset.attrs["time_frequency"] = self.config.frequency
        dataset["value"].attrs.update(data.dataset["value"].attrs)
        dataset["qc_flag"].attrs.update(data.dataset["qc_flag"].attrs)

        record = TransformRecord(
            method=self.name,
            parameters={"frequency": self.config.frequency, "extrapolation": False},
            input_rows=len(source),
            output_rows=len(output),
        )
        result_bundle = EcoDataBundle(dataset=dataset, history=data.history + (record,))
        return TransformResult(data=result_bundle, record=record)

    def _interpolate_group(self, group: pd.DataFrame) -> pd.DataFrame:
        if not group["time"].is_monotonic_increasing or group["time"].duplicated().any():
            raise SchemaError(
                "Time must be unique and strictly increasing within each station/variable group; "
                "out-of-order input is rejected instead of silently sorted"
            )
        group = group.sort_values("time", kind="stable").reset_index(drop=True)
        start = group["time"].iloc[0].ceil(self.config.frequency)
        end = group["time"].iloc[-1].floor(self.config.frequency)
        if start > end:
            raise SchemaError("Observation range contains no target timestamps")
        target = pd.date_range(start=start, end=end, freq=self.config.frequency)

        source_ns = group["time"].astype("int64").to_numpy()
        target_ns = target.astype("int64").to_numpy()
        values = np.interp(target_ns, source_ns, group["value"].to_numpy(dtype=float))
        original_by_time = group.set_index("time")
        flags = [
            str(original_by_time.loc[time, "qc_flag"])
            if time in original_by_time.index
            else "INTERPOLATED"
            for time in target
        ]

        first = group.iloc[0]
        return pd.DataFrame(
            {
                "station_id": first["station_id"],
                "time": target,
                "latitude": float(first["latitude"]),
                "longitude": float(first["longitude"]),
                "variable": first["variable"],
                "value": values,
                "unit": first["unit"],
                "qc_flag": flags,
            }
        )

