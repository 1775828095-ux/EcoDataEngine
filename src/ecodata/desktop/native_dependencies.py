"""Run original model dependency readers on a complete ZIP member catalogue."""
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from .scan import safe

CONTROL_BYTES = 128 * 1024
NESTED_BYTES = 128 * 1024**2


def review(scan, model):
    from ..models.native_source_inspection import inspect_sources, probe_sources
    from .file_tools import selected

    root = Path(scan["path"])
    if model not in {"swat_plus", "wrf", "cmaq", "camx", "calpuff"}:
        return []
    if not root.is_file() or root.suffix.casefold() != ".zip":
        return []
    checks = []
    def inspect_archive(locator, display):
        with selected(root, locator) as (path, _, __), ZipFile(path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir() and safe(info.filename)]
            names = [item.filename.casefold() for item in members]
            if len(set(names)) != len(names):
                checks.append(dict(document=display, status="pending", check="native_project_scope",
                                   basis="native_dependency_unresolved", reference="同名 ZIP 成员无法唯一绑定"))
                return
            payloads, records = [], []
            next_level = []
            for info in members:
                name = f"{display}!/{info.filename}"
                basename = PurePosixPath(info.filename).name.casefold()
                reads_control = {
                    "swat_plus": basename == "file.cio" or basename.endswith((".cli", ".con")),
                    "wrf": basename in {"namelist.input", "namelist.wps"},
                    "cmaq": basename.endswith((".csh", ".json")),
                    "camx": basename.endswith((".inp", ".nml", ".txt")),
                    "calpuff": basename.endswith(".inp"),
                }[model]
                raw = archive.read(info) if reads_control and info.file_size <= CONTROL_BYTES else b""
                payloads.append((name, raw))
                records.append(dict(file=name, payload_read=bool(raw)))
                if basename.endswith(".zip"):
                    child = f"{locator}!/{info.filename}" if locator else info.filename
                    if info.file_size <= NESTED_BYTES and locator.count("!/") < 2:
                        next_level.append((child, name))
                    else:
                        checks.append(dict(document=name, status="pending", check="native_project_scope",
                                           basis="nested_native_inventory_not_confirmed"))
        report = inspect_sources(model, payloads, records)
        prefix = probe_sources(model, [(name, data) for name, data in payloads if data])
        for claim in report["claims"]:
            if claim["role"] in {"engineering_configuration", "engineering_dependency"}:
                checks.append(dict(document=claim["file"], data=claim["file"], status="checked",
                                   check="native_project_member", basis="native_control_reader"))
        checks.extend(dict(document=issue.get("file") or display, status="pending",
                           check="native_project_dependency", basis="native_dependency_unresolved",
                           reference=issue["message"]) for issue in [*report["issues"], *prefix["issues"]])
        for child, name in next_level:
            try:
                inspect_archive(child, name)
            except (BadZipFile, ValueError, RuntimeError, OSError) as error:
                checks.append(dict(document=name, status="pending", check="native_project_scope",
                                   basis="nested_native_inventory_not_confirmed", reference=str(error)))

    inspect_archive("", root.name)
    return checks
