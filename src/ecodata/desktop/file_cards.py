"""Collect lightweight structural evidence without assigning a file's purpose."""
import csv
from itertools import islice
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .scan import FORMATS, kind, safe

PAGE_SIZE = 60
PREFIX_BYTES = 4096
MAX_PROBES = 128
TEXT = {".md", ".rst", ".txt"}
TABLE = {".csv", ".tsv", ".pcp", ".tmp", ".hmd", ".slr", ".wnd"}
TEXT_PROBES = TEXT | TABLE
MAGIC_PROBES = {".nc", ".nc4", ".netcdf", ".cdf", ".hdf", ".hdf5", ".h5", ".tif", ".tiff", ".pdf"}
ORDER = ("engineering", "mixed", "data", "archive", "uncertain")


def describe(path, size, prefix=b""):
    suffix = Path(path).suffix.casefold()
    card = dict(path=path, bytes=size,
                format=FORMATS.get(suffix, suffix.lstrip(".") or None), basis="extension")
    if prefix.startswith((b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n",
                          b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+", b"%PDF-")):
        card["basis"] = "file_signature"
    elif prefix and suffix in TEXT_PROBES:
        from .scan import decode

        lines = decode(prefix).splitlines()
        if len(prefix) == PREFIX_BYTES and not prefix.endswith((b"\n", b"\r")):
            lines = lines[:-1]
        head = list(islice((line.strip() for line in lines if line.strip()), 4))
        delimiter = "\t" if suffix == ".tsv" or head and "\t" in head[0] else ","
        widths = [len(next(csv.reader([line], delimiter=delimiter))) for line in head[:3]]
        if len(widths) >= 2 and widths[0] >= 2 and len(set(widths)) == 1:
            card.update(basis="tabular_prefix", columns=widths[0])
        elif any(line.startswith("# ") for line in head):
            card["basis"] = "heading_prefix"
    return card


def local_cards(root, signature, *, excluded=()):
    """Index every plain file and first-level ZIP member. Prefix bytes stay local."""
    from .file_tools import entries
    from .precipitation_partitions import shard_family

    cards = []
    probes = 0
    blocked = {path.casefold() for path in excluded}

    def add(locator, name, size, opener=None):
        nonlocal probes
        from .csv_index import SUFFIX
        from .raster_reference import MANIFEST

        if (name.casefold() in blocked or locator.casefold() in blocked
                or Path(name).name == MANIFEST or name.endswith(SUFFIX)):
            return
        prefix = b""
        if opener is not None and probes < MAX_PROBES and Path(name).suffix.casefold() in TEXT_PROBES | MAGIC_PROBES:
            probes += 1
            try:
                with opener() as stream:
                    prefix = stream.read(PREFIX_BYTES)
            except (BadZipFile, OSError, RuntimeError, NotImplementedError):
                pass
        card = describe(name, size, prefix)
        card["path"] = locator
        if family := shard_family(name):
            card["possible_daily_group"] = f"{family[0]}/{family[1]}"
        cards.append(card)

    for path, size, *_ in signature:
        locator = path.relative_to(root).as_posix() if root.is_dir() else ""
        if locator or not kind(path):
            add(locator, locator or path.name, size, lambda: path.open("rb"))
        if kind(path) == "zip":
            try:
                with ZipFile(path) as archive:
                    for info in archive.infolist():
                        if not info.is_dir() and safe(info.filename):
                            member = "!/".join(filter(None, (locator, info.filename)))
                            add(member, member, info.file_size, lambda: archive.open(info))
            except (BadZipFile, OSError):
                continue
        elif not locator and kind(path):
            for item in entries(path, ""):
                if not item["directory"]:
                    add(item["name"], item["name"], item["bytes"])
    return cards


def page(cards, offset=0):
    from itertools import zip_longest

    groups = ([card for card in cards if card["kind"] == category] for category in ORDER)
    ordered = [card for group in zip_longest(*groups) for card in group if card is not None]
    stop = offset + PAGE_SIZE
    return dict(entries=ordered[offset:stop], total=len(cards), next_offset=stop if stop < len(cards) else None)
