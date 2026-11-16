"""Pint dimensional checks and pint-xarray conversions, not range inference."""
import json
from importlib.metadata import version

import numpy as np
import pint_xarray  # noqa: F401
import xarray as xr
from cf_xarray.units import units as registry

from ecodata.core.exceptions import SchemaError

registry.force_ndarray_like = True
# Published CALMET surface-format units. These are units, not data estimates.
registry.define('hundred_ft = 100 * foot')
registry.define('tenths = 0.1')


class UnitEngine:
    def integrate_rate(self, field, seconds, target='mm'):
        """Integrate an evidenced interval mean; never infer duration from a label."""
        seconds=np.asarray(seconds,dtype=float)
        if seconds.shape!=(field.sizes.get('time',0),) or not np.isfinite(seconds).all() or np.any(seconds<=0):
            raise SchemaError('降水率积分必须提供每条记录的有效区间秒数')
        backend='cf-xarray/Pint explicit interval'
        output=None
        times=np.asarray(field.time.values)
        if len(times)>=3 and np.all(np.diff(times)/np.timedelta64(1,'s')==seconds[:-1]) and np.all(seconds==seconds[0]):
            try:
                from xclim.core.units import rate2amount
            except ImportError:pass
            else:
                output=rate2amount(field,out_units=target);backend='xclim.core.units.rate2amount'
        if output is None:
            scale=registry.Quantity(1,self.parse(field.attrs.get('units','')))*registry.Quantity(seconds,'s')
            factors=scale.to(self.parse(target)).magnitude
            output=field*xr.DataArray(factors,dims='time',coords={'time':field.time})
        output.attrs=dict(field.attrs,units=target,source_units=field.attrs.get('units',''),
            interval_integration=json.dumps(dict(backend=backend,seconds=seconds.tolist(),target=target)))
        bounds={k:output.attrs.pop(k) for k in ('valid_min','valid_max','valid_range') if k in output.attrs}
        if bounds:output.attrs['source_rate_valid_bounds']=json.dumps(bounds,default=lambda v:np.asarray(v).tolist())
        return output

    @staticmethod
    def parse(unit):
        unit = str(unit).strip() if unit is not None else ""
        if not unit or unit.casefold() in {"unknown", "未知"}:
            raise SchemaError("源单位未知，不能自动换算")
        aliases = {"℃": "degC", "°C": "degC", "Celsius": "degC", "degrees_Celsius": "degC",
                   "µ": "u", "μ": "u", "UG/M3": "ug/m3",
                   "fraction": "1", "unitless": "1", "code": "1"}
        value = aliases.get(str(unit), str(unit)).replace("µ", "u").replace("μ", "u")
        try:
            return registry.parse_units(value)
        except Exception as exc:
            raise SchemaError(f"无法解析单位 {unit}：{exc}") from exc

    def affine(self, source, target):
        try:
            converted = registry.Quantity(np.array([0., 1., 2.]), self.parse(source)).to(self.parse(target)).magnitude
        except SchemaError:
            raise
        except Exception as exc:
            raise SchemaError(f"物理含义不兼容的单位换算：{source} → {target}：{exc}") from exc
        scale, offset = float(converted[1]-converted[0]), float(converted[0])
        if not np.isclose(converted[2], 2*scale+offset, rtol=1e-10, atol=1e-12):
            raise SchemaError(f"当前仿射接口不能安全处理非线性单位换算：{source} → {target}")
        return scale, offset

    def convert(self, field, target):
        source = field.attrs.get("units", "")
        scale, offset = self.affine(source, target)
        # Quantify data only: coordinate units belong to their own CRS/axis handling.
        bare = xr.DataArray(field.data, dims=field.dims, name=field.name)
        output = bare.pint.quantify(self.parse(source), unit_registry=registry).pint.to(self.parse(target)).pint.dequantify()
        output = xr.DataArray(output.data, dims=field.dims, coords=field.coords, name=field.name, attrs=dict(field.attrs))
        output.attrs.update(units=target, source_units=source, unit_conversion=json.dumps({
            "backend": "pint-xarray", "version": version("pint-xarray"), "source": source,
            "target": target, "scale": scale, "offset": offset}))
        return output
