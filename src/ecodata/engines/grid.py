"""Prefer xESMF on geographic grids; retain the existing lightweight backend."""
import xarray as xr
from pyproj import CRS
from ecodata.core.exceptions import SchemaError


class GridEngine:
    @staticmethod
    def availability():
        try:
            import xesmf
            return {"available": True, "version": xesmf.__version__}
        except ImportError as exc:
            return {"available": False, "reason": str(exc), "requirements": "ESMPy/ESMF; Windows 建议独立 WSL/conda 环境"}

    def regrid(self, source, target, method, *, backend="auto"):
        status = self.availability()
        geographic = bool(source.attrs.get("crs")) and CRS.from_user_input(source.attrs["crs"]).is_geographic
        if backend == "xesmf" and (not status["available"] or not geographic):
            raise SchemaError("xESMF 不可用或源坐标不是明确的地理坐标：" + status.get("reason", "需要 lat/lon 与地理 CRS"))
        if backend != "fallback" and status["available"] and geographic:
            import xesmf as xe
            if method == "conservative":
                # cf-xarray owns bounds inference; complex grids require explicit bounds.
                for ds in (source, target):
                    for axis in ("lat", "lon"):
                        if ds[axis].ndim != 1 and not ds[axis].attrs.get("bounds"):
                            raise SchemaError("曲线网格保守重网格需要明确的角点/bounds")
                source = source.cf.add_bounds([a for a in ("lat", "lon") if source[a].ndim == 1 and not source[a].attrs.get("bounds")])
                target = target.cf.add_bounds([a for a in ("lat", "lon") if target[a].ndim == 1 and not target[a].attrs.get("bounds")])
            regridder = xe.Regridder(source, target, "nearest_s2d" if method == "nearest" else method,
                                     unmapped_to_nan=True)
            output = regridder(source, keep_attrs=True)
            return output, {"backend": "xESMF", "version": status["version"], "method": method}
        if any(source[a].ndim != 1 for a in ("lat", "lon")):
            raise SchemaError("曲线网格需要可用的 xESMF；轻量 fallback 不支持曲线网格")
        import xarray_regrid  # noqa: F401
        if method == "nearest":
            output = source.regrid.nearest(target)
        elif method == "bilinear":
            output = source.regrid.linear(target)
        elif method == "conservative":
            if source.attrs.get("crs") and CRS.from_user_input(source.attrs["crs"]).is_projected:
                rename = {"lat": "__northing", "lon": "__easting"}
                output = source.rename(rename).regrid.conservative(target.rename(rename), skipna=True, nan_threshold=0.)
                output = output.rename({v: k for k, v in rename.items()})
            else:
                output = source.regrid.conservative(target, latitude_coord="lat", skipna=True, nan_threshold=0.)
        else:
            raise SchemaError(f"未知重网格方法 {method}")
        # Installing sparse changes xarray-regrid's output backend. The existing
        # EcoData Dataset/NetCDF contract uses dense arrays, so convert explicitly.
        for name in output.data_vars:
            if hasattr(output[name].data, "todense"):
                output[name].data = output[name].data.todense()
        return output, {"backend": "xarray-regrid", "method": method,
                        "fallback_reason": status.get("reason") or "用户选择轻量后端或投影坐标"}
