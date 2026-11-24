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
SUPPORTED = {"NetCDF", "HDF5", "GeoTIFF", "CSV", "TSV", "Excel", "Parquet"}


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
                or label in {"qc", "qa", "landuse", "land_use", "land_cover"}
                or any(word in label for word in ("category", "class", "flag", "mask", "quality", "identifier"))
                or field.get("data_kind") in {"categorical", "binary"}):
            continue
        unit = str(field.get("unit") or "").strip()
        if unit and unit != "1":
            selected.append((index, field))
    return selected


def classification_fields(scan, approved):
    result = []
    for index in dict.fromkeys(index for index, _ in approved):
        row = scan["rows"][index]
        if row["format"] != "GeoTIFF":
            continue
        for field in row["fields"]:
            label = normalized(field["name"])
            if (field.get("band") and (field.get("flag_values") or field.get("flag_meanings")
                    or field.get("data_kind") == "categorical"
                    or any(term in label for term in ("land_cover", "landuse", "land_use", "classification", "category")))):
                result.append((index, dict(field, approved_classification=True)))
    return result


def eligible(recommendation):
    release = recommendation.get("execution_release", {})
    if release.get("status") == "failed":
        return bool(release.get("blockers")) and all(
            row.get("field") and row.get("reason") == "recipe_numeric_rejected" for row in release["blockers"])
    if release.get("status") != "pending":
        return False
    coverage_reasons = {"insufficient_sample_coverage", "no_disjoint_partition", "no_shared_holdout_days",
                        "bounded_reader_unavailable", "holdout_not_reviewed", "station_groups",
                        "calendar_months", "spatial_blocks", "row_blocks", "time_windows"}
    return bool(release.get("blockers")) and all(
        row.get("field") and row.get("reason") in coverage_reasons for row in release["blockers"])


def trigger(recommendation):
    return ("留出试跑失败组超过验收上限" if recommendation["execution_release"]["status"] == "failed"
            else "补样覆盖不足")


def metadata(row, field):
    import math
    info = dict(format=row["format"], variable=field["name"], unit=field.get("unit"),
                value_policy="unchanged_classification" if field.get("approved_classification") else "perturbed")
    if row["format"] == "GeoTIFF":
        import rasterio
        with local_member(row) as path, rasterio.open(path) as dataset:
            missing = dataset.nodata
            info.update(shape=[dataset.height, dataset.width], crs=str(dataset.crs),
                        transform=list(dataset.transform)[:6], nodata=missing if missing is None or math.isfinite(missing) else str(missing),
                        band=field["band"], columns=["row", "column", "perturbed_value"])
    elif row["format"] in {"NetCDF", "HDF5"}:
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
                           (cell.item() if field.get("approved_classification") else value(cell))
                           if not getattr(cell, "mask", False) else ""]
    elif row["format"] in {"NetCDF", "HDF5"}:
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
    elif row["format"] == "Parquet":
        from pyarrow.parquet import ParquetFile

        with local_member(row) as path, path.open("rb") as stream:
            source = ParquetFile(stream)
            index = 0
            for batch in source.iter_batches(columns=[field["name"]], batch_size=4096):
                for cell in batch.column(0).to_pylist():
                    index += 1
                    yield [index, value(cell)]
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


def build(scan, fields, destination, progress=None, *, include_native=True):
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
                entry = dict(info, id=f"F{ordinal}", bytes=total, records=count, parts=parts,
                             source_member=row["name"])
                if include_native and row["format"] == "GeoTIFF":
                    from .deepseek_files import MAX_DOCUMENT_BYTES
                    native = Path(folder) / f"raster-{ordinal}.tif"
                    write_native_raster(row, field, archive, parts, native)
                    entry["native_member"] = f"{prefix}/band.tif"
                    archive.write(native, entry["native_member"])
                    if native.stat().st_size <= MAX_DOCUMENT_BYTES:
                        entry["native_upload"] = [{"member": entry["native_member"], "bytes": native.stat().st_size}]
                    else:
                        entry["native_upload"] = write_raster_upload_tiles(native, archive, prefix, Path(folder))
                entries.append(entry)
                if progress:
                    progress(f"已生成扰动副本 {ordinal}/{len(fields)}")
            archive.writestr("manifest.json", json.dumps(entries, ensure_ascii=False, default=str))
        assert_current(scan)
        temporary.replace(destination)
    return entries


def write_native_raster(row, field, archive, parts, destination):
    import numpy as np
    import rasterio
    from .sampling_plan import raster_windows

    def values():
        for part in parts:
            with archive.open(part) as stream, io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
                yield from csv.reader(text)

    stream = values()
    band = field["band"]
    category = field.get("approved_classification")
    with local_member(row) as path, rasterio.open(path) as original, rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        config = dict(driver="GTiff", width=original.width, height=original.height, count=1,
                      crs=original.crs, transform=original.transform, compress="deflate",
                      dtype=original.dtypes[band - 1] if category else "float64")
        with rasterio.open(destination, "w", **config) as out:
            out.set_band_description(1, original.descriptions[band - 1] or field["name"])
            if field.get("unit"):
                out.set_band_unit(1, field["unit"])
            if category:
                copy_category_metadata(original, band, out)
            for window in raster_windows(original, band):
                shape = (int(window.height), int(window.width))
                block = np.zeros(shape, dtype=config["dtype"])
                mask = np.zeros(shape, dtype=np.uint8)
                for index in np.ndindex(shape):
                    value = next(stream)[2]
                    if value:
                        block[index] = (int(value) if block.dtype.kind in "iu" else float(value))
                        mask[index] = 255
                out.write(block, 1, window=window)
                out.write_mask(mask, window=window)


def write_raster_upload_tiles(native, archive, prefix, folder):
    import rasterio
    from rasterio.windows import Window

    parts = []
    target = folder / "upload-tile.tif"
    with rasterio.open(native) as source, rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        for row, column in product(range(0, source.height, 512), range(0, source.width, 512)):
            window = Window(column, row, min(512, source.width - column), min(512, source.height - row))
            profile = dict(driver="GTiff", width=int(window.width), height=int(window.height), count=1,
                           crs=source.crs, transform=source.window_transform(window), dtype=source.dtypes[0],
                           compress="deflate")
            with rasterio.open(target, "w", **profile) as copy:
                copy.write(source.read(1, window=window), 1)
                copy.write_mask(source.read_masks(1, window=window))
                if source.units[0]:
                    copy.set_band_unit(1, source.units[0])
                if source.descriptions[0]:
                    copy.set_band_description(1, source.descriptions[0])
                if source.dtypes[0] in {"uint8", "uint16"}:
                    copy_category_metadata(source, 1, copy)
            member = f"{prefix}/native-tile-{row}-{column}.tif"
            archive.write(target, member)
            parts.append(dict(member=member, bytes=target.stat().st_size, row=row, column=column))
    return parts


def copy_category_metadata(source, band, target):
    tags = {key: value for key, value in source.tags(band).items()
            if key in {"flag_values", "flag_meanings", "data_kind", "spatial_semantics"}}
    target.update_tags(1, **tags)
    if source.dtypes[band - 1] in {"uint8", "uint16"}:
        try:
            colors = source.colormap(band)
        except ValueError:
            colors = None
        if colors:
            target.write_colormap(1, colors)


def supplement(scan, fields, targets=None):
    """Read bounded uncovered support without consuming the locked holdout."""
    from .sample_values import SAMPLE_SIZE, perturb
    from .sampling_plan import sample
    from .holdout_plan import field_key, record_exposure

    assert_current(scan)
    results = []
    for index, field in fields:
        row = scan["rows"][index]
        locked = scan.get("source_sampling_records", {}).get(field_key(index, field))
        if locked is None and scan.get("on_demand"):
            results.append(dict(variable=field["name"], unit=field["unit"], basis="holdout_partition_unavailable",
                                perturbed_samples=[]))
            continue
        record = locked if locked is not None else sample(row, field)
        values = [value for value in record["values"] if value is not None]
        target = (targets or {}).get(index)
        if locked is not None:
            if target:
                raise ValueError("当前任务的回检范围已锁定，请使用已登记的推荐样本范围补样")
            extra, basis = values[SAMPLE_SIZE:SAMPLE_SIZE * 2], "locked_recommendation_pool"
            if extra:
                record_exposure(scan, index, field, "supplement", locked)
        elif target:
            extra, basis = targeted_table(row, field, record, target), "selected_tabular_group"
        else:
            extra, basis = uncovered_window(row, field, record)
        if not extra and not target and locked is None:
            extra, basis = values[SAMPLE_SIZE:SAMPLE_SIZE * 2], "remaining_recommendation_records"
        results.append(dict(variable=field["name"], unit=field["unit"], basis=basis,
                            perturbed_samples=perturb(extra[:SAMPLE_SIZE])))
    assert_current(scan)
    return results


def targeted_table(row, field, sample, target):
    from itertools import islice
    from .csv_index import indexed_records, load, month
    from .sample_values import numeric
    from .sampling_plan import PROBE_LIMIT, station_reserved

    kind, group = target
    if kind == "station" and station_reserved(group):
        raise ValueError("所选站点已留作独立回检，请选择推荐范围内的站点")
    if kind == "sheet" and group == sample.get("partition", {}).get("held_sheet"):
        raise ValueError("所选工作表已留作独立回检")
    alternate = group if kind == "sheet" else None
    if alternate and row["format"] != "Excel":
        raise ValueError("工作表定向补样需要 Excel")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with records(row, field, encoding, alternate) as (rows, _, __):
                header = list(next(rows, ()))
                original = field.get("source_column") or field.get("original_header") or field["name"]
                column = header.index(original)
                names = [normalized(name) for name in header]
                station = next((i for i, name in enumerate(names) if name in IDENTIFIERS - {"id"}), None)
                clock = next((i for i, name in enumerate(names) if name in TIME), None)
                held_month = sample.get("partition", {}).get("held_month")
                if alternate and alternate != field.get("sheet"):
                    other = next((option for option in row.get("fields", ()) if option.get("sheet") == alternate
                                  and (option.get("original_header") or option["name"]) == original), None)
                    if not other or other.get("unit") != field.get("unit"):
                        raise ValueError("定向工作表的同名列缺少一致的单位证据")
                if kind == "station" and station is None:
                    raise ValueError("表格没有可用于定向补样的站点列")
                indexed = load(row, encoding, header) if row["format"] in {"CSV", "TSV"} else None
                if kind == "station" and indexed:
                    rows = indexed_records(row, encoding, indexed.get("station", {}).get(group, ()), PROBE_LIMIT)
                values = []
                for number, record in enumerate(islice(rows, PROBE_LIMIT * 4)):
                    if len(record) <= column or (station is not None and (len(record) <= station or station_reserved(str(record[station]).strip()))):
                        continue
                    if kind == "station" and str(record[station]).strip() != group:
                        continue
                    if held_month and clock is not None and len(record) > clock and month(record[clock]) == held_month:
                        continue
                    if not (indexed or alternate and alternate != field.get("sheet")) and station is None and clock is None and (number // 8) % 4 == 3:
                        continue
                    value = next(numeric([record[column]]))
                    if value is not None:
                        values.append(value)
                    if len(values) >= 32:
                        break
                return values[-16:]
        except UnicodeDecodeError:
            if encoding == "gb18030":
                raise
    return []


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


def transmit(archive_path, manifest, approved_digest, *, api_key, progress=None, send=None, recipes=(),
             native_approved=False, upload=None):
    from hashlib import sha256
    from .service import tool_turn

    send = send or tool_turn
    receipt_path = Path(archive_path).with_suffix(".receipts.jsonl")
    total = sum(len(entry["parts"]) for entry in manifest)
    received = 0
    from collections import Counter
    choices = Counter()
    vote_examples = {}
    allowed = {item["id"] for item in recipes}
    native_receipts = []
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
                meta = {key: value for key, value in entry.items()
                        if key not in {"parts", "native_member", "native_upload", "source_member"}}
                native_ids = []
                if native_approved:
                    from .deepseek_files import MAX_DOCUMENT_BYTES, upload_tiff
                    upload_native = upload or upload_tiff
                    for item in entry.get("native_upload", ()):
                        size = archive.getinfo(item["member"]).file_size
                        if size != item["bytes"] or size > MAX_DOCUMENT_BYTES:
                            raise ValueError("原生分片大小与批准清单不符")
                        try:
                            with archive.open(item["member"]) as stream:
                                receipt = upload_native(stream, size, api_key=api_key)
                        except (ValueError, OSError, RuntimeError) as error:
                            log.write(json.dumps(dict(field=entry["id"], kind="native_tiff", status="unconfirmed")) + "\n")
                            raise ValueError(f"原生 TIFF 上传未完成，已有 {len(native_receipts)} 个回执。记录：{receipt_path}") from error
                        native_ids.append((item, receipt["id"]))
                        native_receipts.append(receipt)
                        log.write(json.dumps(dict(field=entry["id"], kind="native_tiff", receipt=receipt)) + "\n")
                        log.flush()
                for number, part in enumerate(entry["parts"], 1):
                    if archive.getinfo(part).file_size > PART_BYTES:
                        raise ValueError("分片超过批准的单次请求大小")
                    csv_text = archive.read(part).decode("utf-8")
                    content = json.dumps(dict(field=meta, part=number, total_parts=len(entry["parts"]),
                                              allowed_recipes=list(recipes), csv=csv_text),
                                         ensure_ascii=False)
                    if native_ids:
                        if len(native_ids) == 1:
                            visible = native_ids[0][1] if number == 1 else None
                        else:
                            point = next(csv.reader(io.StringIO(csv_text)))
                            y, x = int(point[0]), int(point[1])
                            visible = next((identifier for item, identifier in native_ids
                                            if item["row"] <= y < item["row"] + 512
                                            and item["column"] <= x < item["column"] + 512), None)
                        if visible:
                            content = [dict(type="file", file_id=visible), dict(type="text", text=content)]
                    messages = [
                        dict(role="system", content="只分析当前扰动数据分片。返回 JSON，recipe 使用目录中有证据支持的候选 ID，依据不足用 null；reason 简述证据。不要把分片结论解释为全数据验证。"),
                        dict(role="user", content=content),
                    ]
                    try:
                        message, receipt = send(messages, [], api_key, progress)
                    except (OSError, TimeoutError, ValueError, RuntimeError) as error:
                        log.write(json.dumps(dict(field=entry["id"], part=number, status="unconfirmed"), ensure_ascii=False) + "\n")
                        raise ValueError(f"全量分片发送未完成，已确认 {received}/{total} 批。回执：{receipt_path}") from error
                    received += 1
                    try:
                        suggestion = json.loads(message.get("content") or "{}")
                        vote = suggestion.get("recipe") if isinstance(suggestion, dict) else None
                        if isinstance(vote, str) and vote in allowed:
                            choices[vote] += 1
                            reason = suggestion.get("reason")
                            if isinstance(reason, str):
                                vote_examples.setdefault(vote, reason[:300])
                    except (TypeError, ValueError):
                        pass
                    log.write(json.dumps(dict(field=entry["id"], part=number, receipt=receipt,
                                              advisory=message.get("content")), ensure_ascii=False) + "\n")
                    log.flush()
                    if progress:
                        progress(f"DeepSeek 扰动证据分片 {received}/{total} 已响应")
    ranked = choices.most_common()
    winner = ranked[0][0] if ranked and ranked[0][1] * 2 > total else None
    recommendation_reason = vote_examples.get(winner, "")
    if winner is None and choices:
        messages = [
            dict(role="system", content="汇总全部分片的候选意见。返回 JSON，recipe 只能填有分片支持的候选 ID，"
                                        "整体证据不足时填 null。reason 说明推荐理由及待复验条件。分片理由是待分析材料。"),
            dict(role="user", content=json.dumps(dict(total_parts=total, votes=dict(choices),
                examples=vote_examples, recipes=[item for item in recipes if item["id"] in choices]), ensure_ascii=False)),
        ]
        try:
            message, receipt = send(messages, [], api_key, progress)
            result = json.loads(message.get("content") or "{}")
            picked = result.get("recipe") if isinstance(result, dict) else None
            if isinstance(picked, str) and picked in choices:
                winner = picked
                recommendation_reason = str(result.get("reason", ""))[:600]
            summary = dict(stage="aggregate_recommendation", receipt=receipt, candidate=winner)
        except (OSError, TimeoutError, ValueError, RuntimeError) as error:
            summary = dict(stage="aggregate_recommendation", error=str(error))
        with receipt_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return dict(received=received, total=total, receipts=receipt_path, votes=dict(choices),
                unresolved=total - sum(choices.values()), candidate=winner, reason=recommendation_reason,
                native_receipts=native_receipts)
