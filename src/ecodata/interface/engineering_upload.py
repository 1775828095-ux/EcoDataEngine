"""Safe ingestion of a WASP engineering directory or archive."""
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import hashlib
import io
import zipfile


SUPPORTED_SUFFIXES = {".csv", ".txt", ".rdb", ".gz", ".isd", ".xlsx",
                      ".parquet", ".nc", ".nc4", ".tif", ".tiff", ".h5"}
WASP_GROUPS = {"usgs", "nbc", "noaa", "ri_dem", "wwtp"}
IGNORED_GROUPS = {"references", "reference", "official_original", "official_wasp_reference"}
MAX_FILE_BYTES = 250 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


@dataclass
class UploadScan:
    files: list[tuple[str, bytes]] = field(default_factory=list)
    accepted: list[dict] = field(default_factory=list)
    ignored: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)

    @property
    def group_counts(self):
        result = {name: 0 for name in sorted(WASP_GROUPS)}
        for item in self.accepted:
            result[item["group"]] = result.get(item["group"], 0) + 1
        return result


def _normalise(name):
    return str(name).replace("\\", "/").lstrip("/")


def _safe_name(name):
    normal = _normalise(name)
    path = PurePosixPath(normal)
    return bool(normal) and not path.is_absolute() and ".." not in path.parts and not (
        len(path.parts[0]) >= 2 and path.parts[0][1] == ":")


def _group(name):
    parts = {part.casefold() for part in PurePosixPath(_normalise(name)).parts[:-1]}
    ignored = parts & IGNORED_GROUPS
    if ignored:
        return None, f"参考资料目录不作为平台输入：{sorted(ignored)[0]}"
    matched = parts & WASP_GROUPS
    return (sorted(matched)[0] if matched else "unclassified"), None


def _members(upload, *, expand_archives=True):
    name, content = _normalise(upload.name), upload.getvalue()
    if PurePosixPath(name).suffix.casefold() != ".zip" or not expand_archives:
        return [(name, content, None)]
    members = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        total = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            member_name = _normalise(info.filename)
            if not _safe_name(member_name):
                members.append((member_name, b"", "ZIP中包含不安全路径"))
                continue
            if info.file_size > MAX_FILE_BYTES:
                members.append((member_name, b"", "文件超过250 MB限制"))
                continue
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                members.append((member_name, b"", "ZIP解压后总量超过2 GB限制"))
                continue
            members.append((member_name, archive.read(info), None))
    return members


def scan_model_uploads(uploads, *, classify_wasp=False, expand_archives=True):
    """Expand model folders/ZIPs, filter unsafe files and deduplicate content."""
    result, seen = UploadScan(), {}
    for upload in uploads or []:
        try:
            members = _members(upload, expand_archives=expand_archives)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            result.ignored.append({"file": upload.name, "reason": f"ZIP无法读取：{exc}"})
            continue
        for name, content, error in members:
            if error:
                result.ignored.append({"file": name, "reason": error})
                continue
            if not _safe_name(name):
                result.ignored.append({"file": name, "reason": "不安全路径"})
                continue
            group, reason = _group(name)
            if not classify_wasp and reason:
                # Reference material is never a model input, regardless of model.
                group = None
            suffix = PurePosixPath(name).suffix.casefold()
            if reason or suffix not in SUPPORTED_SUFFIXES:
                result.ignored.append({"file": name, "reason": reason or f"暂不读取 {suffix or '无扩展名'} 文件"})
                continue
            digest = hashlib.sha256(content).hexdigest()
            identity = (_normalise(name).casefold(), digest)
            if identity in seen:
                result.duplicates.append({"file": name, "same_as": seen[identity],
                                          "reason": "同一路径和内容被浏览器重复提交"})
                continue
            seen[identity] = name
            result.files.append((name, content))
            result.accepted.append({"file": name, "group": group, "size": len(content), "sha256": digest})
    return result


def scan_wasp_uploads(uploads):
    """Backward-compatible WASP scan with engineering-folder classification."""
    return scan_model_uploads(uploads, classify_wasp=True)
