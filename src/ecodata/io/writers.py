from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ecodata.core.schemas import EcoDataBundle


@dataclass(frozen=True)
class WriteResult:
    path: Path
    rows: int


class NetCdfWriter:
    def write(self, data: EcoDataBundle, target: Path) -> WriteResult:
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset = data.dataset.copy(deep=False)
        dataset.attrs["transform_history"] = json.dumps(
            [record.as_dict() for record in data.history],
            sort_keys=True,
            ensure_ascii=True,
        )
        dataset.to_netcdf(target, engine="h5netcdf")
        return WriteResult(path=target, rows=int(dataset.sizes["observation"]))

