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
    @staticmethod
    def parse(unit):
        if not unit or str(unit).casefold() in {"unknown", "未知"}:
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
