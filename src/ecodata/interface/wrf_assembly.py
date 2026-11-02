"""Join pressure updates to complete WPS input without interpolating or falling back.

This is input assembly, not proof that metgrid, real or WRF has succeeded.
The first supported update scope is earth-relative TT/UU/VV/GHT on a regular
latitude/longitude grid. Official humidity and surface records are preserved.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ecodata.core.exceptions import SchemaError

UNITS = {"TT": "K", "UU": "m s-1", "VV": "m s-1", "GHT": "m"}


def _frame(value: bytes) -> bytes:
    size = struct.pack(">i", len(value))
    return size + value + size


@dataclass
class Field:
    parts: list[bytes]
    time: str
    name: str
    unit: str
    level: float
    nx: int
    ny: int
    grid: tuple
    wind: int

    @property
    def key(self):
        return (self.time, "GHT" if self.name == "HGT" else self.name, self.level)

    @property
    def raw(self):
        return b"".join(_frame(p) for p in self.parts)


def read_fields(raw: bytes) -> list[Field]:
    if not raw or len(raw) > 512 * 1024**2:
        raise SchemaError("WPS文件为空或超过512 MiB。")
    parts, offset = [], 0
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise SchemaError("WPS记录被截断。")
        size = struct.unpack_from(">i", raw, offset)[0]
        if size < 0 or offset + size + 8 > len(raw):
            raise SchemaError("WPS记录长度无效。")
        if struct.unpack_from(">i", raw, offset + size + 4)[0] != size:
            raise SchemaError("WPS记录尾标记不匹配。")
        parts.append(raw[offset + 4:offset + 4 + size])
        offset += size + 8
    if len(parts) % 5:
        raise SchemaError("每个WPS场必须有5条完整记录。")
    fields = []
    for i in range(0, len(parts), 5):
        group = parts[i:i + 5]
        version, header, projection, wind, slab = group
        if version != struct.pack(">i", 5) or len(header) != 156 or len(wind) != 4:
            raise SchemaError("仅支持WPS intermediate-v5。")
        level, nx, ny, iproj = struct.unpack(">fiii", header[140:])
        if iproj != 0 or len(projection) != 28 or projection[:8] != b"SWCORNER":
            raise SchemaError("首版接入仅支持规则经纬度SWCORNER网格。")
        grid = struct.unpack(">5f", projection[8:])
        if (nx < 2 or ny < 2 or nx * ny > 10_000_000 or len(slab) != nx * ny * 4
                or not np.isfinite([level, *grid]).all() or not grid[2] or not grid[3]
                or grid[4] <= 0):
            raise SchemaError("WPS网格、层次或数据长度无效。")
        try:
            time = header[:24].decode("ascii").strip()
            datetime.strptime(time, "%Y-%m-%d_%H:%M:%S")
            name = header[60:69].decode("ascii").strip()
            unit = header[69:94].decode("ascii").strip()
        except (ValueError, UnicodeError) as exc:
            raise SchemaError("WPS时间或字段文本无效。") from exc
        flag = struct.unpack(">i", wind)[0]
        if flag not in (0, 1):
            raise SchemaError("WPS风方向标记无效。")
        fields.append(Field(group, time, name, unit, level, nx, ny, grid, flag))
    keys = [f.key for f in fields]
    if len(keys) != len(set(keys)):
        raise SchemaError("WPS有重复时间、字段、层次。")
    if len({f.time for f in fields}) != 1:
        raise SchemaError("每个WPS文件只允许一个时刻。")
    return fields


def _indices(source, target, longitude=False):
    source_index = np.arange(len(source))
    if longitude:
        source, target = source % 360, target % 360
        source = np.concatenate((source - 360, source, source + 360))
        source_index = np.tile(source_index, 3)
    order = np.argsort(source)
    ordered = source[order]
    left = np.searchsorted(ordered, target - 1e-5, side='left')
    right = np.searchsorted(ordered, target + 1e-5, side='right')
    if not np.all(right - left == 1):
        raise SchemaError("配套资料不能通过精确坐标选择匹配平台网格；不自动插值。")
    return source_index[order[left]]


def _subset(field: Field, target: Field) -> bytes:
    lat, lon, dy, dx, radius = field.grid
    tlat, tlon, tdy, tdx, tradius = target.grid
    if abs(radius - tradius) > 1e-3:
        raise SchemaError("配套资料与平台场的地球半径不一致。")
    iy = _indices(lat + np.arange(field.ny) * dy, tlat + np.arange(target.ny) * tdy)
    ix = _indices(lon + np.arange(field.nx) * dx, tlon + np.arange(target.nx) * tdx, True)
    values = np.frombuffer(field.parts[4], dtype=">f4").reshape(field.ny, field.nx)
    parts = list(field.parts)
    header = bytearray(parts[1])
    struct.pack_into(">ii", header, 144, target.nx, target.ny)
    parts[1], parts[2] = bytes(header), target.parts[2]
    parts[4] = values[np.ix_(iy, ix)].astype(">f4").tobytes()
    return b"".join(_frame(p) for p in parts)


def assemble_pressure_update(base: bytes, update: bytes) -> tuple[bytes, dict]:
    """Replace all four pressure fields at all existing levels; keep complements."""
    original, incoming = read_fields(base), read_fields(update)
    template = incoming[0]
    if original[0].time != template.time:
        raise SchemaError("平台场与配套资料时刻不同。")
    expected = {f.key for f in original if f.key[1] in UNITS and 0 < f.level < 200000}
    actual = {f.key for f in incoming}
    if not expected or actual != expected or {f.name for f in incoming} != set(UNITS):
        raise SchemaError("必须完整替换TT、UU、VV、GHT的全部原有气压层；不允许静默回退。")
    for f in incoming:
        if (f.unit != UNITS[f.name] or f.wind != 0 or f.grid != template.grid
                or (f.nx, f.ny) != (template.nx, template.ny)):
            raise SchemaError("平台场单位、风方向或网格不一致。")
        values = np.frombuffer(f.parts[4], dtype=">f4")
        lower, upper = {"TT": (100, 400), "UU": (-500, 500), "VV": (-500, 500), "GHT": (-2000, 100000)}[f.name]
        if not np.isfinite(values).all() or np.any((values < lower) | (values > upper)):
            raise SchemaError("平台场有缺测或数值超出支持范围。")
    complements = [f for f in original if f.key not in expected]
    if not complements:
        raise SchemaError("缺少湿度、地表、土壤等配套记录；四个气压场不是完整WRF输入。")
    merged = b"".join(_subset(f, template) for f in complements) + b"".join(f.raw for f in incoming)
    read_fields(merged)
    return merged, {
        "time": template.time, "replaced_pressure_records": len(incoming),
        "preserved_complement_records": len(complements),
        "base_sha256": hashlib.sha256(base).hexdigest(),
        "platform_sha256": hashlib.sha256(update).hexdigest(),
        "assembled_sha256": hashlib.sha256(merged).hexdigest(),
        "spatial_operation": "exact coordinate selection; no interpolation",
        "pressure_fallback": False, "native_model_executed": False,
    }
