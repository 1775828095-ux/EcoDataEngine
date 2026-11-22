"""Bounded local document extraction with traceable passages."""
from base64 import b64encode
import os
from pathlib import Path, PurePosixPath
import re
from shutil import which
import subprocess
from tempfile import TemporaryDirectory

KEYWORDS = re.compile(r"\b(?:units?|utc|time\s*zone|variable|calendar|precipitation|daily|references?)\b"
                      r"|单位|时区|日界|变量|累计|降水|引用", re.I)
REFERENCE = re.compile(r"(?<![\w.])([\w./-]+\.(?:nc4?|tiff?|csv|xlsx|txt|md|cio|cli|json|pcp))\b", re.I)
IDENTITY = re.compile(r"(?:variable|变量)\s*[:：=]?\s*[`'\x22]?([\w-]+)[`'\x22]?.{0,80}?"
                      r"(?:units?|单位)\s*[:：=]?\s*[`'\x22]?([^\s，。,;`'\x22]+)", re.I)


def passages(items):
    evidence = []
    for location, text in items:
        # Windows OCR can insert blanks between adjacent Chinese characters.
        text = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text or "")
        if KEYWORDS.search(text):
            evidence.append(dict(location=location, excerpt=text.strip()[:500],
                                 references=REFERENCE.findall(text)[:10]))
        if len(evidence) == 10:
            break
    return evidence


def text_passages(text, markdown=False):
    if markdown:
        from markdown_it import MarkdownIt
        title = "开头"
        result = []
        tokens = MarkdownIt("commonmark").parse(text)
        for index, token in enumerate(tokens):
            if token.type != "inline":
                continue
            if index and tokens[index - 1].type == "heading_open":
                title = token.content
            result.append((f"{title} · 第 {(token.map or [0])[0] + 1} 行", token.content))
        return passages(result)
    return passages((f"第 {number} 行", line) for number, line in enumerate(text.splitlines(), 1))


def extract(path, suffix, offset):
    if suffix == ".docx":
        from docx import Document
        from docx.table import Table
        document = Document(path)
        title = "开头"
        items = []
        for index, block in enumerate(document.iter_inner_content()):
            if isinstance(block, Table):
                text = "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
            else:
                text = block.text
                if block.style and block.style.name.startswith("Heading"):
                    title = text
            items.append((f"{title} · 段落或表格 {index + 1}", text))
        return document_result(items[offset:offset + 20], offset + 20 if offset + 20 < len(items) else None)

    from pypdf import PdfReader
    reader = PdfReader(path)
    if not reader.pages:
        return document_result([], None, pending="PDF 无可读取页面")
    pages = []
    if not offset:
        for entry in reader.outline:
            if not isinstance(entry, list) and KEYWORDS.search(str(entry.get("/Title", ""))):
                page = reader.get_destination_page_number(entry)
                if 0 <= page < len(reader.pages) and page not in pages:
                    pages.append(page)
    pages = pages[:2] or list(range(offset, min(offset + 2, len(reader.pages))))
    items, unresolved = [], []
    for page in pages:
        text = reader.pages[page].extract_text() or ""
        if len(text.strip()) < 20:
            text = ocr(path, page)
            if not text.strip():
                unresolved.append(page + 1)
        items.extend((f"第 {page + 1} 页 · 第 {line + 1} 行", part)
                     for line, part in enumerate(text.splitlines()))
    return document_result(items, offset + 2 if offset + 2 < len(reader.pages) else None,
                           pending=f"以下页面需 OCR 或回查原页：{unresolved}" if unresolved else None)


def ocr(path, page):
    executable = which("tesseract")
    if not executable and os.name != "nt":
        return ""
    import pypdfium2
    with pypdfium2.PdfDocument(path) as pdf, TemporaryDirectory() as folder:
        image = Path(folder) / "page.png"
        pdf[page].render(scale=2).to_pil().save(image)
        if not executable:
            return windows_ocr(image)
        try:
            available = subprocess.run([executable, "--list-langs"], capture_output=True,
                                       encoding="utf-8", errors="replace", timeout=5)
            languages = [language for language in ("chi_sim", "eng") if language in available.stdout.splitlines()]
            command = [executable, str(image), "stdout"]
            if languages:
                command.extend(["-l", "+".join(languages)])
            result = subprocess.run(command, capture_output=True,
                                    encoding="utf-8", errors="replace", timeout=25)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return result.stdout if result.returncode == 0 else ""


WINDOWS_OCR = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
} | Select-Object -First 1
function Await($operation, $type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($operation))
    $task.GetAwaiter().GetResult()
}
$storage = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$streamType = [Windows.Storage.Streams.IRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
$decoderType = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$bitmapType = [Windows.Graphics.Imaging.SoftwareBitmap,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$resultType = [Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]
$engine = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { exit 2 }
$file = Await ($storage::GetFileFromPathAsync($env:ECODATA_OCR_IMAGE)) $storage
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode,Windows.Storage,ContentType=WindowsRuntime]::Read)) $streamType
try {
    $decoder = Await ($decoderType::CreateAsync($stream)) $decoderType
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) $bitmapType
    try {
        $result = Await ($engine.RecognizeAsync($bitmap)) $resultType
        $result.Lines | ForEach-Object { $_.Text }
    } finally { $bitmap.Dispose() }
} finally { $stream.Dispose() }
"""


def windows_ocr(image):
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    encoded = b64encode(WINDOWS_OCR.encode("utf-16le")).decode("ascii")
    try:
        result = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                env=dict(os.environ, ECODATA_OCR_IMAGE=str(image)), timeout=25,
                                capture_output=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def document_result(items, next_offset, pending=None):
    evidence = passages(items)
    text = "\n".join(f"[{item['location']}] {item['excerpt']}" for item in evidence)
    return dict(text=text, evidence=evidence, next_offset=next_offset,
                pending=pending or (None if evidence else "该范围没有可定位的相关声明，需回查原文"))


def relative_reference(scan, name, reference):
    """Resolve within the same archive or input directory, never across its root."""
    from .file_tools import selected
    from zipfile import BadZipFile, ZipFile

    outer, _, member = name.rpartition("!/")
    parts = list(PurePosixPath(member if outer else name).parent.parts)
    relative = PurePosixPath(reference)
    if relative.is_absolute():
        return None, False
    for part in relative.parts:
        if part == "..":
            if not parts:
                return None, False
            parts.pop()
        elif part not in {"", "."}:
            parts.append(part)
    target = "/".join(parts)
    if not target or not scan.get("path"):
        return (f"{outer}!/" if outer else "") + target, False
    root = Path(scan["path"])
    try:
        if outer:
            locator = outer
            if root.is_file():
                locator = locator.removeprefix(root.name).removeprefix("!/")
            with selected(root, locator) as (path, _, __), ZipFile(path) as archive:
                exists = not archive.getinfo(target).is_dir()
            return f"{outer}!/{target}", exists
        if root.is_dir():
            candidate = (root / target).resolve()
            return target, candidate.is_relative_to(root.resolve()) and candidate.is_file()
    except (OSError, ValueError, KeyError, BadZipFile):
        pass
    return (f"{outer}!/" if outer else "") + target, False


def cross_check(scan):
    from .sampling_plan import sample
    known = {name.replace("!/", "/") for name in scan.get("directory_members", {})}
    known.update(row["name"].replace("!/", "/") for row in scan["rows"])
    results = []
    for name, text in scan.get("engineering_docs", ()):
        references = list(dict.fromkeys(REFERENCE.findall(text)))[:20]
        resolved = set()
        for reference in references:
            target, exists = relative_reference(scan, name, reference)
            checked = bool(target and (exists or target.replace("!/", "/") in known))
            if checked:
                resolved.add(target)
            results.append(dict(document=name, reference=reference, check="member_reference",
                                status="checked" if checked else "pending"))
        linked = [row for row in scan["rows"] if row["name"] in resolved]
        for variable, unit in IDENTITY.findall(text):
            for row in linked:
                field = next((item for item in row.get("fields", ())
                              if item["name"].casefold() == variable.casefold()), None)
                actual = field.get("unit") if field else None
                result = dict(document=name, data=row["name"], check="declared_variable_unit",
                              variable=variable, declared_unit=unit, observed_unit=actual, status="pending")
                if actual:
                    from ecodata.models.scientific_schema import compare_units
                    comparison = compare_units(variable, unit, variable, actual)
                    if comparison.get("status") == "equivalent":
                        result["status"] = "checked"
                    elif comparison.get("status") == "convertible":
                        result["basis"] = "convertible_but_declared_units_differ"
                    elif comparison.get("status") == "unit_not_convertible":
                        result["status"] = "conflict"
                results.append(result)
        if re.search(r"\bUTC\b", text, re.I) and re.search(r"\bdail(?:y|ies)\b|日界|日累计", text, re.I):
            for row in scan["rows"]:
                if row["format"] != "NetCDF":
                    continue
                if references and not any(PurePosixPath(ref).name == PurePosixPath(row["name"]).name
                                          for ref in references):
                    continue
                field = next((field for field in row.get("fields", ()) if not field.get("axis")), None)
                if not field:
                    continue
                status, basis = "pending", "requires_time_bounds_confirmation"
                try:
                    observed = sample(row, field, holdout=True)
                    clocks = [window["time"] for window in observed.get("windows", ()) if "time" in window]
                    for clock in clocks:
                        if "bounds" not in clock or not clock.get("units"):
                            continue
                        from netCDF4 import num2date
                        dates = num2date(clock["bounds"], clock["units"], calendar=clock.get("calendar") or "standard")
                        is_daily = bool(clock.get("intervals_valid") and all(
                            start.hour == stop.hour == start.minute == stop.minute == start.second == stop.second == 0
                            and (stop - start).total_seconds() == 86400 for start, stop in dates))
                        if not is_daily:
                            status, basis = "conflict", "heldout_bounds_disagree_with_utc_daily"
                            break
                        else:
                            status, basis = "checked", "heldout_cf_bounds_at_daily_utc_boundaries"
                except (OSError, ValueError, TypeError, RuntimeError, ImportError):
                    pass
                results.append(dict(document=name, data=row["name"], check="utc_day_boundary",
                                    status=status, basis=basis))
    return results
