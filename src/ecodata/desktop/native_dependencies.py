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
                           runtime_manifest_eligible=issue.get("code") == "UNSUPPORTED_NATIVE_CONTROL_READER",
                           reference=issue["message"]) for issue in [*report["issues"], *prefix["issues"]])
        for child, name in next_level:
            try:
                inspect_archive(child, name)
            except (BadZipFile, ValueError, RuntimeError, OSError) as error:
                checks.append(dict(document=name, status="pending", check="native_project_scope",
                                   basis="nested_native_inventory_not_confirmed", reference=str(error)))

    inspect_archive("", root.name)
    return checks


def verify_runtime_manifest(scan, model, manifest):
    """Bind explicitly observed runtime inputs to the exact current archive."""
    from hashlib import sha256
    from .fallback_evidence import digest_file
    from .file_tools import selected
    from .scan import assert_current

    assert_current(scan)
    root = Path(scan["path"])
    if not isinstance(manifest, dict) or manifest.get("schema") != "ecodata.native-runtime-bindings.v1":
        raise ValueError("不是可识别的原生运行依赖清单")
    if manifest.get("model") != model or manifest.get("archive_sha256") != digest_file(root):
        raise ValueError("运行依赖清单与本次模型或 ZIP 原件不匹配")
    prefix = root.name + "!/"
    pending = {item["document"][len(prefix):]: item
               for item in scan.get("document_evidence_checks", ())
               if item.get("runtime_manifest_eligible") and item["status"] == "pending"
               and item["document"].startswith(prefix)}
    groups = manifest.get("controls")
    if not isinstance(groups, list) or not groups or not pending:
        raise ValueError("本次没有可通过运行依赖清单复核的原生控制")
    updates = []
    visited = set()
    for control in groups:
        if not isinstance(control, dict):
            raise ValueError("运行控制条目无效")
        locator = control.get("path")
        evidence = control.get("files")
        if (not isinstance(locator, str) or locator not in pending or locator in visited
                or not isinstance(evidence, list) or not evidence):
            raise ValueError("运行依赖清单需要对应本次待核对的控制文件及其实际输入")
        visited.add(locator)
        members = set()
        for binding in evidence:
            reference = binding.get("path") if isinstance(binding, dict) else None
            if (not isinstance(reference, str) or not all(safe(part) for part in reference.split("!/"))
                    or reference in members or reference == locator):
                raise ValueError("运行依赖路径不明确或重复")
            members.add(reference)
            parent, separator, member = reference.rpartition("!/")
            if not separator:
                member = reference
            with selected(root, parent) as (container, _, __), ZipFile(container) as archive:
                matches = [item for item in archive.infolist() if item.filename == member and not item.is_dir()]
                if len(matches) != 1:
                    raise ValueError(f"运行依赖没有唯一对应的 ZIP 成员：{reference}")
                digest = sha256()
                with archive.open(matches[0]) as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
            if digest.hexdigest() != binding.get("sha256"):
                raise ValueError(f"运行依赖的内容与清单不匹配：{reference}")
            updates.append(dict(document=pending[locator]["document"], data=f"{root.name}!/{reference}",
                                status="checked", check="verified_runtime_member", basis="archive_member_hash",
                                verified_sha256=digest.hexdigest()))
        updates.append(dict(document=pending[locator]["document"], status="checked",
                            check="verified_runtime_bindings", basis="operator_runtime_manifest_bound",
                            verified_members=len(members)))
    assert_current(scan)
    return updates
