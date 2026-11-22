"""Compare candidate and baseline on the same explicitly designated local truth."""
from dataclasses import asdict
import json
from pathlib import Path

from .scan import local_member, safe

MANIFEST = "sampling-references.json"
METHODS = {"raster_task.spatial.nearest": "nearest", "raster_task.spatial.bilinear": "bilinear",
           "spatial.regrid.conservative": "conservative"}


def references(root):
    root = Path(root)
    manifest = root / MANIFEST
    if not root.is_dir() or not manifest.is_file():
        return {}
    if manifest.is_symlink() or manifest.stat().st_size > 65536:
        raise ValueError("独立参考栅格清单无效")
    sources = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(sources, dict):
        raise ValueError("独立参考栅格清单必须是按源文件路径索引的对象")
    result = {}
    for source, record in sources.items():
        if not isinstance(source, str) or not safe(source) or not isinstance(record, dict):
            raise ValueError("独立参考栅格记录无效")
        reference = record.get("reference", "")
        band = record.get("band", 1)
        baseline = record.get("baseline", "nearest")
        semantics = record.get("spatial_semantics")
        if (not isinstance(reference, str) or not safe(reference) or
                type(band) is not int or band < 1 or baseline not in METHODS.values()
                or semantics not in {None, "cell_mean", "flux"}):
            raise ValueError("独立参考栅格路径、波段或基线无效")
        source_path = (root / source).resolve(strict=True)
        reference_path = (root / reference).resolve(strict=True)
        if (not source_path.is_relative_to(root) or not reference_path.is_relative_to(root)
                or source_path.samefile(reference_path)):
            raise ValueError("独立参考必须是授权工作目录中的不同文件")
        result[source] = dict(path=reference_path, locator=reference, band=band, baseline=baseline,
                              spatial_semantics=semantics)
    return result


def compare(row, field, held, reference, method):
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.windows import Window, from_bounds
    import xarray as xr

    from ecodata.engines.units import UnitEngine
    from ecodata.evaluation.metrics import fidelity_metrics

    block = held["windows"][0]
    window = Window(block["column"], block["row"], block["width"], block["height"])
    with local_member(row) as path, rasterio.open(path) as source, rasterio.open(reference["path"]) as truth:
        if not source.crs or not truth.crs:
            raise ValueError("独立栅格评分需要两侧已知的坐标系")
        extent = transform_bounds(source.crs, truth.crs, *source.window_bounds(window))
        try:
            target = from_bounds(*extent, transform=truth.transform).round_offsets().round_lengths()
            target = target.intersection(Window(0, 0, truth.width, truth.height))
        except rasterio.errors.WindowError:
            return None
        if not 0 < target.width * target.height <= 4096:
            return None
        observed = truth.read(reference["band"], window=target, masked=True).astype(float).filled(np.nan)
        original = source.read(field["band"], window=window, masked=True).astype(float).filled(np.nan)
        source_unit = field.get("unit") or source.units[field["band"] - 1]
        ref_unit = truth.units[reference["band"] - 1]
        if not source_unit or not ref_unit:
            return None
        if source_unit != ref_unit:
            scale, offset = UnitEngine().affine(source_unit, ref_unit)
            original = original * scale + offset
        outputs = {}
        for operator in {method, reference["baseline"]}:
            if operator == "conservative":
                declared = field.get("spatial_semantics") or source.tags(field["band"]).get("spatial_semantics")
                confirmed = reference.get("spatial_semantics")
                if (source.crs != truth.crs or (declared and confirmed and declared != confirmed)
                        or (declared or confirmed) not in {"cell_mean", "flux"}):
                    return None
                outputs[operator] = conservative(original, source.window_transform(window), observed.shape,
                                                 truth.window_transform(target), source.crs)
                if outputs[operator] is None:
                    return None
                continue
            prediction = np.full(observed.shape, np.nan)
            reproject(original, prediction, src_crs=source.crs, src_transform=source.window_transform(window),
                      dst_crs=truth.crs, dst_transform=truth.window_transform(target),
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=getattr(Resampling, operator))
            outputs[operator] = prediction
        mask = np.isfinite(observed) & np.isfinite(outputs[method]) & np.isfinite(outputs[reference["baseline"]])
        if not mask.any():
            return None
        expected = xr.DataArray(observed[mask], dims=["cell"])
        return dict(reference="declared_independent_raster", n=int(mask.sum()), candidate_method=method,
                    baseline_method=reference["baseline"],
                    candidate=asdict(fidelity_metrics(xr.DataArray(outputs[method][mask], dims=["cell"]), expected)),
                    baseline=asdict(fidelity_metrics(xr.DataArray(outputs[reference["baseline"]][mask], dims=["cell"]), expected)))


def conservative(values, source_transform, target_shape, target_transform, crs):
    """Use the registered grid engine with real raster cell boundaries."""
    import numpy as np
    import xarray as xr
    from ecodata.engines.grid import GridEngine

    if (any(transform.b or transform.d for transform in (source_transform, target_transform))
            or min(*values.shape, *target_shape) < 2):
        return None

    def coordinates(shape, transform):
        lon = transform.c + transform.a * np.arange(shape[1] + 1)
        lat = transform.f + transform.e * np.arange(shape[0] + 1)
        return dict(lat=("lat", (lat[:-1] + lat[1:]) / 2), lon=("lon", (lon[:-1] + lon[1:]) / 2),
                    lat_b=("lat_b", lat), lon_b=("lon_b", lon))

    source = xr.Dataset({"value": (("lat", "lon"), values)},
                        coords=coordinates(values.shape, source_transform), attrs={"crs": str(crs)})
    target = xr.Dataset(coords=coordinates(target_shape, target_transform))
    source = source.sortby(["lat", "lon", "lat_b", "lon_b"])
    ordered = target.sortby(["lat", "lon", "lat_b", "lon_b"])
    engine = GridEngine()
    if crs.is_geographic and engine.availability()["available"]:
        result, _ = engine.regrid(source, ordered, "conservative", backend="xesmf")
    else:
        # The lightweight backend derives bounds from regular centers. Extra
        # bound dimensions would otherwise be mistaken for axes to remap.
        result, _ = engine.regrid(source.drop_vars(["lat_b", "lon_b"]), ordered.drop_vars(["lat_b", "lon_b"]),
                                  "conservative", backend="fallback")
    return result["value"].sel(lat=target.lat, lon=target.lon).transpose("lat", "lon").values
