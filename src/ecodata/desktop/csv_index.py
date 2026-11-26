"""Local CSV record-offset index. Build explicitly to avoid slowing first use."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

SUFFIX = ".ecodata-index.json"


def month(value):
    match = re.match(r"^(\d{4})[-/](\d{1,2})(?=[- /T]|$)", str(value or "").strip())
    return f"{match[1]}-{int(match[2]):02d}" if match and 1 <= int(match[2]) <= 12 else None


def identity(path):
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def lines(stream, encoding):
    for line in stream:
        yield line.decode(encoding)


def scan_index(path, *, encoding="utf-8-sig"):
    from .sampling_plan import IDENTIFIERS, TIME, normalized
    from .sample_values import PROBE_ROWS

    path = Path(path).resolve(strict=True)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    stamp = identity(path)
    with path.open("rb") as stream:
        reader = csv.reader(lines(stream, encoding), delimiter=delimiter)
        headers = next(reader, [])
        names = [normalized(value) for value in headers]
        station = next((i for i, name in enumerate(names) if name in IDENTIFIERS - {"id"}), None)
        time = next((i for i, name in enumerate(names) if name in TIME), None)
        groups = {name: defaultdict(list) for name in ("station", "month")}
        prefix_months = set()
        record_index = 0
        while True:
            offset = stream.tell()
            try:
                row = next(reader)
            except StopIteration:
                break
            end = stream.tell()
            record_index += 1
            for category, index in (("station", station), ("month", time)):
                if index is None or index >= len(row):
                    continue
                key = month(row[index]) if category == "month" else row[index].strip()
                if not key:
                    continue
                if category == "month" and record_index <= PROBE_ROWS:
                    prefix_months.add(key)
                spans = groups[category][key]
                if spans and spans[-1][1] == offset:
                    spans[-1][1] = end
                else:
                    spans.append([offset, end])
    if identity(path) != stamp:
        raise ValueError("CSV 在建索引期间已变化")
    return dict(source=stamp, encoding=encoding, headers=headers, groups=groups,
                prefix_months=sorted(prefix_months))


def build(path, *, encoding="utf-8-sig"):
    path = Path(path).resolve(strict=True)
    document = scan_index(path, encoding=encoding)
    target = path.with_name(path.name + SUFFIX)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(document, temporary,
                  ensure_ascii=False, separators=(",", ":"))
    Path(temporary.name).replace(target)
    return target


def load(row, encoding, headers):
    path, archive, *_ = row["ref"]
    if archive or row.get("containers"):
        return None
    path = Path(path)
    index = path.with_name(path.name + SUFFIX)
    if index.is_symlink() or not index.is_file() or index.stat().st_size > 32 * 1024**2:
        return None
    try:
        document = json.loads(index.read_text(encoding="utf-8"))
        if (document["source"] == identity(path) and document["encoding"] == encoding
                and document["headers"] == headers):
            groups = document["groups"]
            # Older sidecars did not index the USGS site_no spelling.
            if not groups.get("station") and any(str(h).strip().casefold() == "site_no" for h in headers):
                return None
            return groups
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def indexed_records(row, encoding, spans, limit):
    path = Path(row["ref"][0])
    delimiter = "\t" if row["format"] == "TSV" else ","
    with path.open("rb") as stream:
        size = path.stat().st_size
        for start, stop in spans:
            if not (0 <= start < stop <= size):
                raise ValueError("CSV 索引偏移超出原文件")
            stream.seek(start)
            reader = csv.reader(lines(stream, encoding), delimiter=delimiter)
            while (limit is None or limit > 0) and stream.tell() < stop:
                record = next(reader, None)
                if record is None or stream.tell() > stop:
                    raise ValueError("CSV 索引与原记录边界不符")
                yield record
                if limit is not None:
                    limit -= 1
            if limit == 0:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="在本地为 CSV/TSV 建立站点、月份的记录偏移索引。")
    parser.add_argument("input")
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()
    print(build(args.input, encoding=args.encoding))
