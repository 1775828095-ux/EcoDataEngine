"""Normalize explicitly declared table-unit segments through the shared unit engine."""
import numpy as np
import pandas as pd
import xarray as xr

from ecodata.engines.units import UnitEngine

from .sampling_plan import normalized


def unit_column(labels, field):
    stem = normalized(field["name"])
    effective = [index for index, label in enumerate(labels) if normalized(label) == f"{stem}_effective_unit"]
    if len(effective) == 1:
        return effective[0]
    if len(effective) > 1:
        raise ValueError("输出有效单位列有歧义")
    preferred = {f"{stem}_unit", f"{stem}_units", f"{stem}_单位"}
    exact = [index for index, label in enumerate(labels) if normalized(label) in preferred]
    common = [index for index, label in enumerate(labels) if normalized(label) in {"unit", "units", "单位"}]
    choices = exact or common
    if len(choices) > 1:
        raise ValueError("存在多个同等优先级的单位列，请明确所选变量的单位来源")
    return choices[0] if choices else None


def convert(dataset, variable, field, target):
    values = dataset[variable]
    if values.dtype.kind not in "iuf":
        raw = pd.to_numeric(values.values.reshape(-1), errors="coerce")
        values = values.copy(data=np.asarray(raw, dtype=float).reshape(values.shape))
    if not np.isfinite(values.values).any():
        raise ValueError("单位转换没有有效数值")
    source = field.get("unit") or values.attrs.get("units")
    labels = list(dataset.data_vars)
    index = unit_column(labels, field) if dataset.attrs.get("row_dimension") else None
    engine = UnitEngine()
    if index is None:
        values.attrs = dict(values.attrs, units=source)
        return dataset.assign({variable: engine.convert(values, target)}), [{"unit": source, "records": int(values.size)}]

    column = dataset[labels[index]]
    if column.dims != values.dims or values.ndim != 1:
        raise ValueError("逐条单位列与观测字段不能按同一行对应")
    explicit = [str(unit).strip() if not pd.isna(unit) else "" for unit in column.values]
    effective = np.asarray([unit or source or "" for unit in explicit], dtype=object)
    valid = np.isfinite(values.values)
    if not all(effective[valid]):
        raise ValueError("有效记录缺少可验证的逐条单位")
    result = np.full(values.shape, np.nan)
    runs = np.flatnonzero(np.r_[True, effective[1:] != effective[:-1], True])
    factors = {}
    report = []
    for low, high in zip(runs, runs[1:]):
        unit = effective[low]
        if unit and valid[low:high].any():
            if unit not in factors:
                factors[unit] = engine.affine(unit, target)
            scale, offset = factors[unit]
            result[low:high] = values.values[low:high] * scale + offset
        if len(report) < 64:
            report.append(dict(start=int(low + 1), stop=int(high), unit=unit, target=target))
    if not np.isfinite(result[valid]).all():
        raise ValueError("逐段转换出现非有限值")
    converted = values.copy(data=result)
    converted.attrs = dict(values.attrs, units=target, source_unit_column=labels[index],
                           unit_segments=int(len(runs) - 1))
    # Preserve the original row-unit column for other fields and provenance.
    output_column = f"{field['name']}_effective_unit"
    if output_column in dataset and labels[index] != output_column:
        raise ValueError("输出有效单位列与现有列冲突")
    effective_output = xr.DataArray(np.full(values.shape, target), dims=values.dims)
    return dataset.assign({variable: converted, output_column: effective_output}), report


def convert_holdout(row, field, holdout, target):
    """Reproduce row-specific conversion on the already reserved validation scope."""
    from itertools import islice
    from .csv_index import month
    from .sampling_plan import IDENTIFIERS, TIME, records, station_reserved
    from .sample_values import numeric

    partition = holdout.get("partition") or {}
    basis = holdout["basis"]
    reserved_sheet = partition.get("held_sheet") if basis == "worksheet_groups" else None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with records(row, field, encoding, reserved_sheet) as (rows, _, __):
                header = list(next(rows, ()))
                label = field.get("source_column") or field.get("original_header") or field["name"]
                index = header.index(label)
                units = unit_column(header, field)
                if units is None:
                    return None
                normalized_names = [normalized(name) for name in header]
                station = next((i for i, name in enumerate(normalized_names) if name in IDENTIFIERS - {"id"}), None)
                clock = next((i for i, name in enumerate(normalized_names) if name in TIME), None)
                result = []
                codes = set()
                coefficients = {}
                engine = UnitEngine()
                for position, record in enumerate(islice(rows, 16384)):
                    inside = (reserved_sheet is not None or
                              (basis == "station_groups" and station is not None and len(record)>station
                               and station_reserved(str(record[station]).strip())) or
                              (basis == "calendar_months" and clock is not None and len(record)>clock
                               and partition.get("held_month") is not None and month(record[clock]) == partition["held_month"]) or
                              (basis in {"uniform_heldout_row_blocks", "row_blocks"} and position // 8 % 4 == 3))
                    if not inside or len(record) <= index:
                        continue
                    value = next(numeric([record[index]]))
                    if value is None:
                        continue
                    unit = str(record[units]).strip() if len(record) > units and record[units] else field.get("unit")
                    if unit not in coefficients:
                        coefficients[unit] = engine.affine(unit, target)
                    scale, offset = coefficients[unit]
                    result.append(value * scale + offset)
                    codes.add(unit)
                    if len(result) >= 64:
                        break
                return xr.DataArray(result, dims=["sample"], attrs={"units": target}), sorted(codes)
        except UnicodeDecodeError:
            if encoding == "gb18030":
                raise
    return None
