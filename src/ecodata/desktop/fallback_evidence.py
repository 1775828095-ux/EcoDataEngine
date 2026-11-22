"""Build and explicitly transmit bounded parts of a scoped numeric evidence copy."""
import csv
from itertools import product
import io
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from .scan import AXES, assert_current, local_member, safe
from .sampling_plan import IDENTIFIERS, TIME, normalized, records

PART_BYTES = 48 * 1024
SUPPORTED = {"NetCDF", "GeoTIFF", "CSV", "TSV", "Excel"}


def approved_fields(scan, recommendation):
    from .release_gate import resolve_field, task_fields

    selected = []
    for index, name, fid in task_fields({"fields": recommendation.get("execution_fields", ())}):
        row = scan["rows"][index]
        field = resolve_field(row, index, name, fid)
        if not field or row["format"] not in SUPPORTED or row.get("original_reader"):
            continue
        label = normalized(field.get("source_column") or name)
        if (field.get("axis") or field.get("flag_values") or field.get("flag_meanings") or field.get("flag_masks")
                or label in AXES | IDENTIFIERS | TIME
                or label in {"qc", "qa", "landuse", "land_cover"}
                or any(word in label for word in ("category", "class", "flag", "mask", "quality", "identifier"))
                or field.get("data_kind") in {"categorical", "binary"}):
            continue
        unit = str(field.get("unit") or "").strip()
        if unit and unit != "1":
            selected.append((index, field))
    return selected


def eligible(recommendation):
    release = recommendation.get("execution_release", {})
    if release.get("status") != "pending":
        return False
    coverage_reasons = {"insufficient_sample_coverage", "no_disjoint_partition",
                        "bounded_reader_unavailable", "holdout_not_reviewed", "station_groups",
                        "calendar_months", "spatial_blocks", "row_blocks", "time_windows"}
    return bool(release.get("blockers")) and all(
        row.get("field") and row.get("reason") in coverage_reasons for row in release["blockers"])


def metadata(row, field):
    import math
    info = dict(format=row["format"], variable=field["name"], unit=field["unit"])
    if row["format"] == "GeoTIFF":
        import rasterio
        with local_member(row) as path, rasterio.open(path) as dataset:
            missing = dataset.nodata
            info.update(shape=[dataset.height, dataset.width], crs=str(dataset.crs),
                        transform=list(dataset.transform)[:6], nodata=missing if missing is None or math.isfinite(missing) else str(missing),
                        band=field["band"], columns=["row", "column", "perturbed_value"])
    elif row["format"] == "NetCDF":
        import xarray as xr
        with local_member(row) as path, xr.open_dataset(path, decode_times=False, cache=False) as dataset:
            value = dataset[field["name"]]
            info.update(shape=list(value.shape), dimensions=list(value.dims))
            info["coordinates"] = {dim: {
                key: dataset[dim].attrs[key] for key in ("units", "calendar", "bounds")
                if key in dataset[dim].attrs}
                for dim in value.dims if dim in dataset.coords}
            clock = next((dim for dim in value.dims if dim.casefold() in TIME and dim in dataset.coords), None)
            bound = dataset[clock].attrs.get("bounds") if clock else None
            columns = list(value.dims)
            if bound in dataset and clock in dataset[bound].dims:
                other_dims = [dim for dim in dataset[bound].dims if dim != clock]
                if len(other_dims) == 1 and dataset.sizes[other_dims[0]] == 2:
                    columns.extend(["interval_start", "interval_end"])
            info["columns"] = columns + ["perturbed_value"]
    elif field.get("sheet"):
        info["sheet"] = field["sheet"]
    if "columns" not in info:
        info["columns"] = ["source_record", "perturbed_value"]
    return info


def numerical_rows(row, field):
    """Yield positional rows. Missing values retain their mask; identifiers stay local."""
    from .sample_values import numeric, perturb

    def value(item):
        number = next(numeric([item]))
        return perturb([number])[0] if number is not None else ""

    if row["format"] == "GeoTIFF":
        import rasterio
        from .sampling_plan import raster_windows
        with local_member(row) as path, rasterio.open(path) as dataset:
            band = field["band"]
            for window in raster_windows(dataset, band):
                block = dataset.read(band, window=window, masked=True)
                for i, j in product(range(block.shape[0]), range(block.shape[1])):
                    cell = block[i, j]
                    yield [int(window.row_off) + i, int(window.col_off) + j,
                           value(cell) if not getattr(cell, "mask", False) else ""]
    elif row["format"] == "NetCDF":
        import numpy as np
        import xarray as xr
        with local_member(row) as path, xr.open_dataset(path, decode_times=False, cache=False) as dataset:
            array = dataset[field["name"]]
            if not array.ndim:
                yield [value(array.item())]
                return
            tile = [1] * array.ndim
            remaining = 8192
            for dim in range(array.ndim - 1, -1, -1):
                tile[dim] = max(1, min(array.shape[dim], remaining))
                remaining = max(1, remaining // tile[dim])
            clock = next((dim for dim in array.dims if dim.casefold() in TIME and dim in dataset.coords), None)
            bounds = dataset[clock].attrs.get("bounds") if clock else None
            for starts in product(*(range(0, n, width) for n, width in zip(array.shape, tile))):
                sub = {dim: slice(start, start + width) for dim, start, width in zip(array.dims, starts, tile)}
                block = array.isel(sub).values
                edges = (dataset[bounds].isel({clock: sub[clock]}).transpose(clock, ...).values
                         if bounds in dataset and clock in dataset[bounds].dims else None)
                for position in np.ndindex(block.shape):
                    indexes = [i + start for i, start in zip(position, starts)]
                    if edges is not None and edges.ndim == 2 and edges.shape[1] == 2:
                        indexes.extend(edges[position[array.dims.index(clock)]].tolist())
                    yield [*indexes, value(block[position])]
    else:
        # Inspect before yielding, so an encoding retry can never duplicate emitted rows.
        with local_member(row) as path:
            if row["format"] == "Excel":
                encoding = "utf-8-sig"
            else:
                from codecs import getincrementaldecoder
                encoding = "utf-8-sig"
                decoder = getincrementaldecoder(encoding)()
                try:
                    with path.open("rb") as stream:
                        while block := stream.read(65536):
                            decoder.decode(block)
                        decoder.decode(b"", final=True)
                except UnicodeDecodeError:
                    encoding = "gb18030"
        with records(row, field, encoding) as (rows, _, __):
            header = list(next(rows, ()))
            title = field.get("source_column") or field.get("original_header") or field["name"]
            column = header.index(title)
            for index, row_values in enumerate(rows, 1):
                yield [index, value(row_values[column] if len(row_values) > column else None)]


def build(scan, fields, destination, progress=None):
    """Create the complete authorized field copy before any outbound action."""
    assert_current(scan)
    destination = Path(destination)
    entries = []
    with TemporaryDirectory(dir=destination.parent) as folder:
        temporary = Path(folder) / "evidence.zip"
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for ordinal, (index, field) in enumerate(fields, 1):
                row = scan["rows"][index]
                relative = row["name"].replace("!/", "/")
                if not safe(relative):
                    raise ValueError("外发成员路径无法安全保留")
                info = metadata(row, field)
                prefix = f"fields/{PurePosixPath(relative).as_posix()}/field-{ordinal}"
                total = count = 0
                parts = []
                buffer = bytearray()
                for record in numerical_rows(row, field):
                    line = io.StringIO(newline="")
                    csv.writer(line).writerow(record)
                    encoded = line.getvalue().encode("utf-8")
                    if len(encoded) > PART_BYTES:
                        raise ValueError("单行结构超出扰动证据分片容量")
                    if buffer and len(buffer) + len(encoded) > PART_BYTES:
                        part = f"{prefix}/part-{len(parts)+1:05}.csv"
                        archive.writestr(part, buffer)
                        parts.append(part)
                        total += len(buffer)
                        buffer.clear()
                    buffer.extend(encoded)
                    count += 1
                if buffer:
                    part = f"{prefix}/part-{len(parts)+1:05}.csv"
                    archive.writestr(part, buffer)
                    parts.append(part)
                    total += len(buffer)
                if not count:
                    raise ValueError("获准字段没有可生成副本的记录")
                entries.append(dict(info, id=f"F{ordinal}", bytes=total, records=count, parts=parts))
                if progress:
                    progress(f"已生成扰动副本 {ordinal}/{len(fields)}")
            archive.writestr("manifest.json", json.dumps(entries, ensure_ascii=False, default=str))
        assert_current(scan)
        temporary.replace(destination)
    return entries


def supplement(scan, fields):
    """Read bounded uncovered support without consuming the locked holdout."""
    from .sample_values import SAMPLE_SIZE, perturb
    from .sampling_plan import sample

    assert_current(scan)
    results = []
    for index, field in fields:
        row = scan["rows"][index]
        record = sample(row, field)
        values = [value for value in record["values"] if value is not None]
        extra, basis = uncovered_window(row, field, record)
        if not extra:
            extra, basis = values[SAMPLE_SIZE:SAMPLE_SIZE * 2], "remaining_recommendation_records"
        results.append(dict(variable=field["name"], unit=field["unit"], basis=basis,
                            perturbed_samples=perturb(extra[:SAMPLE_SIZE])))
    assert_current(scan)
    return results


def uncovered_window(row, field, record):
    from .sample_values import numeric

    partition = record.get("partition")
    windows = record.get("windows")
    if not partition or not windows:
        return [], None
    boundary = partition["holdout_start"]
    if row["format"] == "NetCDF":
        import xarray as xr
        from .sampling_plan import time_dimension

        axis = partition["axis"]
        ranges = [(0, 0), *((item["start_index"], item["stop_index"]) for item in windows),
                  (boundary, boundary)]
        gaps = [(right[0] - left[1], left[1]) for left, right in zip(ranges, ranges[1:])]
        size, start = max(gaps)
        if size <= 0:
            return [], None
        length = min(size, 16)
        start += (size - length) // 2
        with local_member(row) as path, xr.open_dataset(path, decode_times=False, cache=False) as dataset:
            array = dataset[field["name"]]
            clock = time_dimension(array, dataset)
            spatial = next((dim for dim in array.dims if dim != clock and array.sizes[dim] > 1), None)
            selector = {}
            for dim in array.dims:
                if dim == axis:
                    continue
                limit = array.sizes[dim]
                if dim == spatial:
                    limit -= max(1, limit // 5)
                selector[dim] = (limit - 1) // 2
            selector[axis] = slice(start, start + length)
            values = array.isel(selector).values.reshape(-1)
        return [value for value in numeric(values) if value is not None], "uncovered_netcdf_window"
    if row["format"] == "GeoTIFF":
        import rasterio
        from rasterio.windows import Window

        with local_member(row) as path, rasterio.open(path) as dataset:
            limit_w = boundary if partition["axis"] == "column" else dataset.width
            limit_h = boundary if partition["axis"] == "row" else dataset.height
            width, height = min(4, limit_w), min(4, limit_h)
            x, y = (limit_w - width) // 2, (limit_h - height) // 2
            if any(x < w["column"] + w["width"] and x + width > w["column"]
                   and y < w["row"] + w["height"] and y + height > w["row"] for w in windows):
                return [], None
            values = dataset.read(field["band"], window=Window(x, y, width, height), masked=True).compressed()
        return [value for value in numeric(values) if value is not None], "uncovered_raster_window"
    return [], None


def digest_file(path):
    from hashlib import sha256

    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def transmit(archive_path, manifest, approved_digest, *, api_key, progress=None, send=None):
    from hashlib import sha256
    from .service import tool_turn

    send = send or tool_turn
    receipt_path = Path(archive_path).with_suffix(".receipts.jsonl")
    total = sum(len(entry["parts"]) for entry in manifest)
    received = 0
    with Path(archive_path).open("rb") as source:
        digest = sha256()
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != approved_digest:
            raise ValueError("扰动副本已变化，请重新审核后发送")
        source.seek(0)
        with ZipFile(source) as archive, receipt_path.open("x", encoding="utf-8") as log:
            if not total or json.loads(archive.read("manifest.json")) != manifest:
                raise ValueError("待发清单与获准副本不一致")
            for entry in manifest:
                meta = {key: value for key, value in entry.items() if key != "parts"}
                for number, part in enumerate(entry["parts"], 1):
                    if archive.getinfo(part).file_size > PART_BYTES:
                        raise ValueError("分片超过批准的单次请求大小")
                    messages = [
                        dict(role="system", content="仅分析当前获准的扰动数据分片并返回简洁 JSON。只说明观察到的证据、可能的处理与缺项。不要把本分片结论解释为全数据验证。"),
                        dict(role="user", content=json.dumps(
                            dict(field=meta, part=number, total_parts=len(entry["parts"]),
                                 csv=archive.read(part).decode("utf-8")), ensure_ascii=False)),
                    ]
                    try:
                        message, receipt = send(messages, [], api_key, progress)
                    except (OSError, TimeoutError, ValueError, RuntimeError) as error:
                        log.write(json.dumps(dict(field=entry["id"], part=number, status="unconfirmed"), ensure_ascii=False) + "\n")
                        raise ValueError(f"全量分片发送未完成，已确认 {received}/{total} 批。回执：{receipt_path}") from error
                    received += 1
                    log.write(json.dumps(dict(field=entry["id"], part=number, receipt=receipt,
                                              advisory=message.get("content")), ensure_ascii=False) + "\n")
                    log.flush()
                    if progress:
                        progress(f"DeepSeek 扰动证据分片 {received}/{total} 已响应")
    return dict(received=received, total=total, receipts=receipt_path)
