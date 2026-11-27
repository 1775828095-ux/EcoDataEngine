"""Compare and prepare two compatible grids using existing numerical kernels."""
from dataclasses import asdict

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.series_fusion import SeriesMissingFallback
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.metrics import fidelity_metrics
from ecodata.spatial.regridding import GridRegridder, RegridConfig, contained_cell_grid


def daily_contract(dataset, field):
    """Check the declared precipitation contract without reading observations."""
    from pyproj import CRS
    from ecodata.engines.units import UnitEngine
    from ecodata.spatial.regridding import normalize_grid_axes

    if field not in dataset or set(dataset[field].dims) != {"time", "lat", "lon"}:
        raise ValueError("降水补位需要带时间轴的规则经纬网格")
    if not dataset.attrs.get("crs") or not CRS.from_user_input(dataset.attrs["crs"]).equals(CRS.from_epsg(4326)):
        raise ValueError("通用降水补位需要明确的 EPSG:4326 坐标基准")
    dataset, _ = normalize_grid_axes(dataset)
    data = dataset[field].transpose("time", "lat", "lon")
    if data.attrs.get("spatial_semantics", dataset.attrs.get("spatial_semantics")) != "cell_mean":
        raise ValueError("需要明确的格内平均降水语义")
    if str(data.attrs.get("time_basis", dataset.attrs.get("time_basis", ""))).upper() != "UTC":
        raise ValueError("需要明确的 UTC 时间基准")
    time = dataset.time
    bounds_name = time.attrs.get("bounds", "time_bounds")
    bounds = dataset.get(bounds_name)
    if bounds is None or "time" not in bounds.dims or bounds.ndim != 2:
        raise ValueError("需要 CF 时间区间")
    bounds = bounds.transpose("time", ...)
    engine = UnitEngine()
    semantic = data.attrs.get("temporal_semantics")
    if semantic in {"accumulation", "interval_total"}:
        mode = "amount"
        engine.affine(data.attrs.get("units"), "mm")
    elif semantic in {"mean", "interval_mean", "mean_rate"} or "time: mean" in data.attrs.get("cell_methods", ""):
        mode = "rate"
        engine.affine(data.attrs.get("units"), "mm/s")
    else:
        raise ValueError("需要已声明的区间平均降水率或区间累计量")
    return data, bounds, mode


def daily_precipitation(dataset, field):
    """Integrate only complete UTC days; expose incomplete support as missing."""
    import pandas as pd
    from ecodata.engines.units import UnitEngine

    data, bounds, mode = daily_contract(dataset, field)
    time = data.time
    edges = bounds.values
    if edges.shape != (time.size, 2) or not np.issubdtype(edges.dtype, np.datetime64):
        raise ValueError("时间区间须有有效日期")
    begin, end = edges[:, 0], edges[:, 1]
    stamps = time.values
    if (not len(edges) or np.isnat(edges).any() or not np.issubdtype(stamps.dtype, np.datetime64)
            or np.isnat(stamps).any() or (np.diff(stamps) <= np.timedelta64(0, "s")).any()
            or (stamps < begin).any() or (stamps > end).any()
            or (end <= begin).any() or (begin[1:] < end[:-1]).any()):
        raise ValueError("时间坐标无效、倒序或区间重叠")
    day = np.timedelta64(1, "D")
    dates = begin.astype("datetime64[D]")
    if (end > dates + day).any():
        raise ValueError("降水区间跨 UTC 日，缺少日内分配依据")
    engine = UnitEngine()
    if mode == "amount":
        amount = engine.convert(data, "mm")
    else:
        seconds = (end - begin) / np.timedelta64(1, "s")
        amount = engine.integrate_rate(data, seconds, "mm")
    if bool((amount < 0).any()):
        raise ValueError("源降水包含负值")
    labels = pd.date_range(dates[0], dates[-1], freq="D")
    empty = xr.full_like(amount.isel(time=0, drop=True), np.nan)
    daily = []
    observed_dates, offsets = np.unique(dates, return_index=True)
    groups = dict(zip(observed_dates, zip(offsets, np.r_[offsets[1:], len(dates)])))
    for date in labels.to_numpy(dtype="datetime64[D]"):
        low, high = groups.get(date, (0, 0))
        complete = (high > low and begin[low] == date and end[high - 1] == date + day
                    and np.array_equal(end[low:high - 1], begin[low + 1:high]))
        daily.append(amount.isel(time=slice(low, high)).sum("time", skipna=False) if complete else empty)
    out = xr.concat(daily, dim=xr.IndexVariable("time", labels)).rename("precipitation").to_dataset()
    out.precipitation.attrs = dict(units="mm", physical_variable="precipitation", data_kind="continuous",
                                   temporal_semantics="accumulation", spatial_semantics="cell_mean")
    out["time_bounds"] = (("time", "nv"), np.column_stack((labels.values, labels.values + day)))
    out.time.attrs["bounds"] = "time_bounds"
    out.attrs.update(crs="EPSG:4326", time_basis="UTC")
    return out


def remap(dataset, target, variable, method):
    if all(np.array_equal(dataset[axis], target[axis]) for axis in ("lat", "lon")):
        return dataset
    config = RegridConfig(method, tuple(target.lat.values), tuple(target.lon.values), variable)
    engine = GridRegridder(config)
    result = engine.apply(EcoDataBundle(dataset)).data.dataset
    if method == "conservative":
        field = dataset[variable]
        # A partial valid area is insufficient evidence for a full cell value.
        mask = xr.ones_like(field, dtype=float).where(np.isfinite(field), 0.)
        mask.attrs = dict(field.attrs)
        support = engine.apply(EcoDataBundle(dataset.assign({variable: mask}))).data.dataset
        result[variable] = result[variable].where(support[variable] >= 1. - 1e-8)
        result[variable].attrs = dict(field.attrs)
    if "time_bounds" in dataset:
        if not result.indexes["time"].equals(dataset.indexes["time"]):
            raise ValueError("重网格后的日期与已核验的日累计区间不一致")
        result["time_bounds"] = dataset.time_bounds
        result.time.attrs["bounds"] = "time_bounds"
    return result


def grid_candidates(sources, method):
    candidates = []
    for index, native in enumerate(sources):
        if method == "conservative":
            target = native
            try:
                for other in sources:
                    target = contained_cell_grid(target, other)
            except ValueError:
                continue
        else:
            target = native
            for axis in ("lat", "lon"):
                low, high = max(float(ds[axis][0]) for ds in sources), min(float(ds[axis][-1]) for ds in sources)
                target = target.sel({axis: slice(low, high)})
        if min(target.sizes["lat"], target.sizes["lon"]) < 2:
            continue
        if any(all(np.array_equal(target[axis], old[axis]) for axis in ("lat", "lon")) for _, old in candidates):
            continue
        candidates.append((index, target))
    if not candidates:
        raise ValueError("共同覆盖内没有满足对齐要求的目标网格")
    return candidates


def choose_grid(sources, primary, variable, method, *, comparison_times=()):
    """Compare target grids on one fixed native-source reconstruction proxy."""
    candidates = grid_candidates(sources, method)
    reference = sources[primary]
    scores = []
    if len(candidates) > 1 and len(comparison_times):
        try:
            for _, candidate in candidates:
                if method == "conservative":
                    reference = contained_cell_grid(reference, candidate)
                else:
                    reference = reference.sel(lat=slice(float(candidate.lat[0]), float(candidate.lat[-1])),
                                              lon=slice(float(candidate.lon[0]), float(candidate.lon[-1])))
            times = np.asarray(comparison_times)[np.linspace(
                0, len(comparison_times) - 1, min(4, len(comparison_times)), dtype=int)]
            reference = reference.sel(time=times)
            backup = sources[1 - primary].sel(time=times)
            predictions = [remap(remap(backup, target, variable, method), reference, variable, method)[variable]
                           for _, target in candidates]
            mask = np.isfinite(reference[variable])
            for prediction in predictions:
                mask = mask & np.isfinite(prediction)
            if bool(mask.any()):
                scores = [dict(source_grid=index + 1, n=int(mask.sum()),
                               latitude_step=float(target.lat[1] - target.lat[0]),
                               longitude_step=float(target.lon[1] - target.lon[0]),
                               **asdict(fidelity_metrics(prediction.where(mask), reference[variable].where(mask))))
                          for (index, target), prediction in zip(candidates, predictions)]
        except (RuntimeError, ValueError):
            # Geometry with no comparable support cannot be ranked by error.
            scores = []
    if scores:
        chosen = min(range(len(scores)), key=lambda i: (scores[i]["rmse"], scores[i]["mae"],
                                                          candidates[i][1].sizes["lat"] * candidates[i][1].sizes["lon"]))
    else:
        chosen = min(range(len(candidates)), key=lambda i: candidates[i][1].sizes["lat"] * candidates[i][1].sizes["lon"])
    index, target = candidates[chosen]
    return index, target, dict(selected_source_grid=index + 1, candidates=scores,
                               considered_source_grids=[i + 1 for i, _ in candidates],
                               latitude_step=float(target.lat[1] - target.lat[0]),
                               longitude_step=float(target.lon[1] - target.lon[0]),
                               selection_basis="HELDOUT_NATIVE_SOURCE_PROXY" if scores else "GEOMETRY_BASELINE",
                               independent_accuracy_verified=False)


def fill(aligned, primary, variable):
    original = [ds[variable] for ds in aligned]
    context = AlgorithmContext(tuple(EcoDataBundle(ds) for ds in aligned), variable, "continuous",
                               parameters={"primary_index": primary})
    output = SeriesMissingFallback().run(context)
    availability = (np.isfinite(original[0]).astype("int8") + 2 * np.isfinite(original[1]).astype("int8"))
    dataset = output.data.dataset.assign(source_availability=availability)
    dataset.source_availability.attrs.update(flag_values=[0, 1, 2, 3],
                                             flag_meanings="missing source_1 source_2 both_original_sources")
    return dataset, dict(output.report, unresolved=int((availability == 0).sum()), fused=False)


def trial_recipe(aligned, primary, steps, variable="precipitation", *, scoring_request=None, native=None,
                 mask_phase=None, mask_seed=0):
    """Run the frozen recipe and score a fixed reduced-grid reconstruction proxy."""
    from ecodata.algorithms.spatial_fusion_task import blend_fields
    from .recipe_trials import reference_positions, score

    output, evidence = fill(aligned, primary, variable)
    if evidence["unresolved"]:
        return output, evidence, None, "both_sources_missing_on_holdout"
    # The original spatial search seeds its blend with equal weights.
    left, right = (ds[variable] for ds in aligned)
    paired = np.isfinite(left) & np.isfinite(right)
    if not bool(paired.any()):
        return output, evidence, None, "insufficient_score_support"
    if "spatial.blend" in steps:
        combined = blend_fields(left, right, .5)
        output[variable] = combined.where(paired, output[variable])
        output[variable].attrs = dict(left.attrs)
    else:
        # Hide measured primary cells from the fill operation and compare its
        # backup predictions with the median of the remaining primary cells.
        observed = np.asarray(aligned[primary][variable].values, dtype=float)
        available = np.flatnonzero(paired.values)
        if len(available) < 2:
            return output, evidence, None, "insufficient_score_support"
        withheld = available[::2] if mask_phase is None else available[
            reference_positions(len(available), mask_phase, mask_seed)]
        if not len(withheld):
            return output, evidence, None, "insufficient_score_support"
        masked = aligned[primary].copy()
        values = observed.copy()
        values.flat[withheld] = np.nan
        masked[variable] = masked[variable].copy(data=values)
        inputs = list(aligned)
        inputs[primary] = masked
        reconstructed, _ = fill(inputs, primary, variable)
        expected = observed.flat[withheld]
        baseline = np.full(expected.shape, np.nanmedian(values))
        metrics, failure = score(reconstructed[variable].values.flat[withheld], baseline, expected,
                                 "MASKED_PRIMARY_GAP_RECONSTRUCTION_PROXY", retained=values,
                                 request=scoring_request)
        return output, evidence, metrics, failure

    from ecodata.evaluation.fusion_protocol import fixed_recipe_cohort
    protocol, cohort = fixed_recipe_cohort(native if native is not None else aligned, aligned[primary], variable)
    if cohort is None:
        evidence["evaluation_protocol"] = protocol
        return output, evidence, None, "insufficient_score_support"
    estimates = cohort["inputs"]
    truth = cohort["reference"]
    # Stable fine-reference positions are assigned to disjoint assessment phases.
    if mask_phase is not None:
        positions = reference_positions(truth.shape[1], mask_phase, mask_seed)
        estimates = [values[:, positions] for values in estimates]
        truth = truth[:, positions]
    mask = np.isfinite(truth) & np.isfinite(estimates[0]) & np.isfinite(estimates[1])
    if not bool(mask.any()):
        return output, evidence, None, "insufficient_score_support"
    truth = np.where(mask, truth, np.nan)
    baseline = min(estimates, key=lambda estimate: np.nansum((estimate-truth)**2))
    metrics, failure = score(blend_fields(*estimates, .5), baseline, truth,
                             "REDUCED_SCALE_PRODUCT_RECONSTRUCTION_PROXY", retained=np.stack(estimates),
                             request=scoring_request)
    if metrics:
        metrics["evaluation_protocol"] = dict(protocol, assessment_phase=mask_phase,
                                              selection_count=int(mask.sum()))
    return output, evidence, metrics, failure
