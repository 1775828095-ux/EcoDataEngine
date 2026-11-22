"""Compare candidate and baseline on the same explicitly designated local truth."""
from dataclasses import asdict
import json
from pathlib import Path

from .scan import local_member, safe

MANIFEST = "sampling-references.json"


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
        if (not isinstance(reference, str) or not safe(reference) or
                type(band) is not int or band < 1 or baseline not in {"nearest", "bilinear"}):
            raise ValueError("独立参考栅格路径、波段或基线无效")
        source_path = (root / source).resolve(strict=True)
        reference_path = (root / reference).resolve(strict=True)
        if (not source_path.is_relative_to(root) or not reference_path.is_relative_to(root)
                or source_path.samefile(reference_path)):
            raise ValueError("独立参考必须是授权工作目录中的不同文件")
        result[source] = dict(path=reference_path, locator=reference, band=band, baseline=baseline)
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
