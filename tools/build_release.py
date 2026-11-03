"""Create reproducible source and Windows project bundles."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXCLUDES = {".git", ".venv", ".venv-pnc", "__pycache__", ".pytest_cache", ".pytest_tmp",
            ".cache", ".streamlit", "output", "logs", "runtime", "temp", "dist", "build",
            ".idea", ".vscode", "老师演示数据"}
REAL_CASE = Path("examples/real_wasp_narragansett")
REAL_KEEP_NAMES = {"source_manifest.json", "01113895_daily_2009.rdb",
                   "01113895_instantaneous_2009.rdb", "2009-NBC-River-and-Bay-Nutrients-Data.xlsx",
                   "725070-14765-2009.gz", "B12.gsoHB060209.TXT", "F4.nbc.PDB040709.RPT.txt",
                   "NARTBWQ.csv", "B2.gsoNB052709.TXT", "B2.gsoNS061709.TXT",
                   "build_wasp_data_package.py", "README.md", "raw_to_wasp_gap_report.md"}


def allowed(relative: Path, portable: bool) -> bool:
    if relative.parts[0] in EXCLUDES or "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}:
        return False
    if portable and relative.parts[0] in {"tests", "tools"}:
        return False
    if relative.is_relative_to(REAL_CASE):
        return relative.name in REAL_KEEP_NAMES
    return True


def copy_tree(target: Path, portable: bool):
    for source in ROOT.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        if not allowed(relative, portable):
            continue
        destination = target / relative
        destination_parent = Path("\\\\?\\" + str(destination.parent.resolve()))
        destination_parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2("\\\\?\\" + str(source.resolve()), "\\\\?\\" + str(destination.resolve()))


def manifest(target: Path, kind: str):
    version = (target / "VERSION").read_text(encoding="utf-8").strip()
    files = []
    for root, dirs, names in os.walk("\\\\?\\" + str(target.resolve())):
        root_path = Path(root.replace("\\\\?\\", ""))
        for name in names:
            path = root_path / name
            if "_runtime" in path.relative_to(target).parts:
                continue
            extended_path = Path("\\\\?\\" + str(path))
            files.append({"path": path.relative_to(target).as_posix(), "size": extended_path.stat().st_size,
                          "sha256": hashlib.sha256(extended_path.read_bytes()).hexdigest()})
    data = {"product": "EcoData Engine", "version": version, "package_type": kind,
            "built_at_utc": datetime.now(timezone.utc).isoformat(), "files": files}
    if kind == "portable":
        data["bundled_python"] = sys.version.split()[0]
    (target / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return version


def zip_portable(target: Path, archive: Path):
    """Zip a directory without tripping Windows' 260-character path limit."""
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as archive_file:
        for root, dirs, names in os.walk("\\\\?\\" + str(target.resolve())):
            root_path = Path(root.replace("\\\\?\\", ""))
            for name in names:
                path = root_path / name
                archive_file.write("\\\\?\\" + str(path), arcname=path.relative_to(target).as_posix())


def build(kind: str):
    DIST.mkdir(exist_ok=True)
    name = "EcoDataEngine_Source" if kind == "source" else "EcoDataEngine_Windows_Portable"
    target = DIST / name
    if target.exists():
        shutil.rmtree("\\\\?\\" + str(target.resolve()))
    target.mkdir()
    copy_tree(target, portable=kind == "portable")
    if kind == "portable":
        runtime = target / "_runtime"
        shutil.copytree(Path(sys.base_prefix), runtime, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        source_packages = Path(sys.prefix) / "Lib" / "site-packages"
        shutil.copytree(source_packages, runtime / "Lib" / "site-packages", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        launcher = """@echo off
setlocal
cd /d "%~dp0"
if not exist logs mkdir logs
set "PYTHONPATH=%CD%\\src"
start "EcoData Engine" /b "_runtime\\python.exe" -m streamlit run "src\\ecodata\\interface\\app.py" --server.headless true --server.port 8510 --browser.gatherUsageStats false >>logs\\portable_startup.log 2>&1
powershell -NoProfile -Command "$ok=$false; 1..60 | ForEach-Object { try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8510 -TimeoutSec 1; if($r.StatusCode -eq 200){$ok=$true;break} } catch {}; Start-Sleep -Seconds 1 }; if($ok){Start-Process 'http://127.0.0.1:8510'; exit 0}else{exit 1}"
if errorlevel 1 (echo [FAILED] See logs\\portable_startup.log & pause & exit /b 1)
echo EcoData Engine is ready. This window can be closed.
pause
"""
        (target / "启动 EcoData Engine.bat").write_text(launcher, encoding="utf-8-sig")
    version = manifest(target, kind)
    archive = DIST / f"{name}.zip"
    if archive.exists():
        archive.unlink()
    zip_portable(target, archive)
    print(f"{name} {version} | {archive.stat().st_size} bytes | {datetime.now().astimezone().isoformat()}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"source", "portable"}:
        raise SystemExit("usage: build_release.py source|portable")
    build(sys.argv[1])
