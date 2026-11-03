"""Existing algorithm DAGs with the frozen common metre-coordinate adapter."""
import numpy as np
import pandas as pd
import xarray as xr

import run_taiwan_preparation as task
from taiwan_working_contract import CONTRACT, PROJECTION

original_factory = task.graph_executor
original_field = task.field


def projected_idw(train, query, values=None):
    table = train.copy()
    source = task.xy(train)
    points = task.xy(query)
    table["longitude"], table["latitude"] = source[:, 0], source[:, 1]
    table["value"] = table.value if values is None else values
    # Geometry only: no fabricated background or target values.
    grid = xr.Dataset(coords={"lat": np.unique(points[:, 1]), "lon": np.unique(points[:, 0])}, attrs={"crs":"EPSG:32651"})
    context = task.AlgorithmContext((task.EcoDataBundle(grid),), "pm25", "continuous",
        observations=table.assign(unit=task.UNIT), parameters={"observation_crs":"EPSG:32651", "power":2.0})
    result = task.IDWPlugin().run(context)
    assert result.report["distance_crs"] == "EPSG:32651"
    return result.data.dataset.pm25.sel(lat=xr.DataArray(points[:,1],dims="point"), lon=xr.DataArray(points[:,0],dims="point")).values


def working_field(table, values):
    ds = original_field(table, values)
    ds.attrs.update(contract_version=CONTRACT["version"], metric_crs="EPSG:32651",
        crs_evidence="USER_WORKING_ASSUMPTION_NOT_AUTHOR_DATUM_PROOF")
    if "time" in table and len(table):
        ds.attrs["hour_interval_start_utc"] = str(table.time.iloc[0])
    return ds


def working_factory():
    executor = original_factory()
    adapter = executor.adapters[("quality", "run")]
    def quality(inputs, params, external):
        result = adapter.execute(inputs, params, external)
        result["quality"].value["result"].update(provenance_status="FROZEN_WORKING_CONTRACT; AUTHOR_TIME_DATUM_UNVERIFIED; ARCHIVE_MD5_VERIFIED",
            contract_version=CONTRACT["version"])
        return result
    executor.adapters[("quality", "run")] = task.NodeAdapter(adapter.inputs, adapter.outputs, quality)
    return executor


def configure():
    task.TRANSFORM = PROJECTION
    task.idw = projected_idw
    task.field = working_field
    task.graph_executor = working_factory


def qualified(table, target_role):
    train = table.loc[table.split.eq("train") & table.value.ge(0)].copy()
    target = table.loc[table.split.eq(target_role) & table.value.ge(0)].copy()
    if len(train) < CONTRACT["evaluation"]["min_training_stations"]:
        return train, target.iloc[:0], "INSUFFICIENT_TRAIN"
    inside = task.Delaunay(task.xy(train)).find_simplex(task.xy(target)) >= 0 if len(target) else np.array([], dtype=bool)
    target = target.loc[inside].copy()
    if len(target) < CONTRACT["evaluation"]["min_heldout_stations"]:
        return train, target.iloc[:0], "INSUFFICIENT_IN_DOMAIN_TARGET"
    return train, target, "ELIGIBLE"
