"""Package completed exports without changing their conversion logic."""
import io
import json
import zipfile
from pathlib import PurePosixPath

from ecodata.core.exceptions import SchemaError


def package_results(files, conversion_report, recognition_report, *, failures=None):
    """Validate every artifact before creating an in-memory ZIP archive."""
    entries = list(files)
    if len(entries) < 2 and not failures:
        raise SchemaError("至少生成两个转换结果文件才需要打包")
    entries += [
        ("conversion_report.json", json.dumps(conversion_report, ensure_ascii=False, indent=2, default=str).encode("utf-8")),
        ("data_recognition_report.json", json.dumps(recognition_report, ensure_ascii=False, indent=2, default=str).encode("utf-8")),
    ]
    if failures is not None:
        from ecodata.evaluation.self_check import validation_files
        for index, item in enumerate(conversion_report, 1):
            report = item.get("报告", {}).get("validation_report")
            if report:
                entries.extend((f"{index:02d}_{name}", content) for name, content in validation_files(report).items())
        entries += [("failures.json", json.dumps(failures, ensure_ascii=False, indent=2).encode("utf-8")),
                    ("processing_plan.json", json.dumps([item.get("报告", {}).get("自动处理流程", []) for item in conversion_report], ensure_ascii=False, indent=2, default=str).encode("utf-8")),
                    ("quality_report.json", json.dumps([item.get("报告", {}).get("自动质量检查", {}) for item in conversion_report], ensure_ascii=False, indent=2, default=str).encode("utf-8"))]
    names = set()
    for name, content in entries:
        path = PurePosixPath(name)
        if not name or "\\" in name or path.is_absolute() or ".." in path.parts or len(path.parts) != 1 or name in names:
            raise SchemaError(f"结果文件名无效或重名：{name}")
        if not isinstance(content, bytes) or not content:
            raise SchemaError(f"结果文件未生成或为空：{name}")
        names.add(name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()
