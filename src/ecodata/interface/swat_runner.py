"""Version-checked local SWAT+ runner. Uploads are data, never executables."""
from __future__ import annotations

import calendar
import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import stat
import subprocess
import threading
import time
import uuid
import zipfile

from ecodata.core.exceptions import SchemaError

MAX_BYTES = 100 * 1024 * 1024
MAX_FILES = 3000
FORBIDDEN_INPUT_SUFFIXES={".exe", ".dll", ".bat", ".cmd", ".ps1", ".py", ".sh", ".com", ".scr", ".lnk"}
ENGINE_SHA256 = "e757daf64a597ca2da80f12ff6a0075c4e8260c6fcf871f8da979aef21a37db3"
ROOT = Path(__file__).resolve().parents[3]
_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}


def read_project(payload: bytes) -> dict[str, bytes]:
    """Read one portable TxtInOut, reject ambiguous roots and unsafe paths."""
    if len(payload) > MAX_BYTES:
        raise SchemaError("ZIP超过100 MiB；请使用较小的完整测试工程。")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES or sum(i.file_size for i in entries) > MAX_BYTES:
                raise SchemaError("工程解压大小或文件数超限（100 MiB / 3000项）。")
            files = {}
            for info in entries:
                name = info.filename.replace("\\", "/")
                path = PurePosixPath(name)
                reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
                if (path.is_absolute() or ".." in path.parts or any(c in name for c in ':<>"|?*')
                        or any(ord(c) < 32 for c in name)
                        or stat.S_ISLNK(info.external_attr >> 16)
                        or any(p.endswith((".", " ")) or p.split('.')[0].lower() in reserved for p in path.parts)):
                    raise SchemaError(f"ZIP含不安全路径：{name}")
                if info.is_dir():
                    continue
                if path.suffix.lower() in FORBIDDEN_INPUT_SUFFIXES:
                    raise SchemaError(f"工程仅接收数据；请移除程序或脚本：{name}")
                if name.casefold() in {n.casefold() for n in files}:
                    raise SchemaError(f"ZIP有重名文件：{name}")
                files[name] = archive.read(info)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise SchemaError(f"无法读取工程ZIP：{exc}") from exc
    roots = [PurePosixPath(n).parent for n in files if PurePosixPath(n).name == "file.cio"]
    if len(roots) != 1:
        raise SchemaError("必须且只能有一个file.cio；天气下载包不是完整工程。")
    root = roots[0]
    result = {}
    for name, data in files.items():
        p = PurePosixPath(name)
        if p.is_relative_to(root):
            relative = p.relative_to(root).as_posix()
            if len(PurePosixPath(relative).parts) != 1:
                raise SchemaError("首版只接收平铺TxtInOut目录；请把实际引用的数据文件放在同一目录。")
            if relative in {"runner_status.json", "runner.log", "runner_manifest.json"}:
                raise SchemaError(f"工程含运行器保留文件名：{relative}")
            result[relative] = data
    return result


def _lines(files, name):
    try:
        return files[name].decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{name}不是UTF-8/ASCII文本；请明确转换编码。") from exc


def _reference(files, name, source, errors):
    if name.lower() in {"null", "sim", "0"}:
        return
    if name != PurePosixPath(name).name or "\\" in name or ":" in name:
        errors.append(f"{source}引用外部路径{name}；请制作平铺完整工程。")
    elif name not in files:
        errors.append(f"{source}引用缺失文件：{name}")
    elif not files[name].strip():
        errors.append(f"{source}引用空文件：{name}")


def check_project(files, engine_id=None):
    errors, warnings = [], []
    if "file.cio" not in files:
        return {"errors": ["缺少file.cio；仅天气文件不能作为完整工程运行。"], "warnings": []}
    header = _lines(files, "file.cio")[0] if files["file.cio"].strip() else ""
    declared = re.search(r"\brev\.?\s*(\d+)(?:\.\d+)*", header, re.IGNORECASE)
    if declared is None:
        declared = re.search(r'\bfor\s+SWAT\+\s*(\d+)(?:\.\d+)*', header, re.IGNORECASE)
    declared_major = int(declared.group(1)) if declared else None
    exact_revision = re.search(r'\b(?:rev\.?\s*|for\s+SWAT\+\s*)(\d+(?:\.\d+)*)', header, re.I)
    declared_version = exact_revision.group(1) if exact_revision else None
    if engine_id is not None and engine_id not in ENGINE_LABELS:
        raise SchemaError("未知的SWAT+运行程序。")
    selected_major = 60 if engine_id == 'official60' else 61 if engine_id == "official61" else 62 if engine_id else declared_major or 62
    if declared_major is not None and declared_major not in (60, 61, 62):
        errors.append(f"工程声明rev.{declared_major}，当前仅登记60.5.7/61/62系列；请使用匹配的程序与输入格式。")
    if selected_major == 60 and not re.search(r'\brev\.?\s*60\.5\.7\b', header, re.I):
        errors.append('rev60程序仅登记60.5.7，不将其他60系列补丁版本视为兼容。')
    if engine_id and declared_major is not None and declared_major != selected_major:
        errors.append(f"版本不匹配：工程声明rev.{declared_major}，所选程序为rev.{selected_major}；平台不会自动迁移参数。")
    if declared_major is None:
        warnings.append(f"file.cio未声明版本；按所选程序rev.{selected_major}检查，用户仍须核实输入格式兼容性。")
    groups = {}
    for line in _lines(files, "file.cio")[1:]:
        tokens = line.split("!", 1)[0].split()
        if not tokens:
            continue
        if tokens[0] in groups:
            errors.append(f"file.cio分组重复：{tokens[0]}")
        groups[tokens[0]] = tokens[1:]
    generated_indexes = set()
    climate = groups.get('climate', [])
    if selected_major == 61 and climate and climate[0] in files:
        stations = [row.split() for row in _lines(files, climate[0])[2:] if row.strip()]
        # rev.61 cli_hmeas/cli_wmeas first INQUIRE optional index files.
        # Permit an absent index only when every station explicitly uses WGN
        # (sim/null), with no measured-file reference to hide. No input is edited.
        for index_pos, station_pos in ((6, 5), (7, 6)):
            if len(climate) > index_pos and climate[index_pos] not in files and stations and all(
                    len(row) > station_pos and row[station_pos].lower() in {'sim', 'null', '0'} for row in stations):
                generated_indexes.add(climate[index_pos])
                warnings.append(f"rev.61可选索引{climate[index_pos]}缺失，但所有站点显式使用sim/null；保留原WGN生成设置，不创建观测文件。")
    for group, refs in groups.items():
        for ref in refs:
            if group == 'climate' and ref in generated_indexes:
                continue
            _reference(files, ref, f"file.cio/{group}", errors)
    # SWAT+ 62 readcio_read reads derived types sequentially. A short basin
    # record consumes tokens from the following climate record and misreads
    # weather filenames; file existence alone cannot detect that error.
    expected_basin = 2 if selected_major in (60, 61) else 3
    if "basin" in groups and len(groups["basin"]) != expected_basin:
        errors.append(f"SWAT+rev.{selected_major}的file.cio/basin需要{expected_basin}项。项数不符会使后续天气引用读错位；请用匹配版本重新导出，平台不会自动补参数。")
    simulation = groups.get("simulation", [])
    if not simulation or simulation[0] in {"null", "sim", "0"}:
        errors.append("file.cio缺少有效simulation时间文件引用。")
    time_name = simulation[0] if simulation else "time.sim"
    dates = None
    if time_name in files:
        try:
            rows = [x.split() for x in _lines(files, time_name)[2:] if x.strip()]
            day0, year0, day1, year1, step = map(int, rows[0])
            if not 1 <= year0 <= year1 <= 9999:
                raise ValueError()
            if not (0 <= day0 <= 365 + calendar.isleap(year0) and 0 <= day1 <= 365 + calendar.isleap(year1)):
                raise ValueError()
            if year0 == year1 and (day0 or 1) > (day1 or 365 + calendar.isleap(year1)):
                raise ValueError()
            if step != 0:
                errors.append("首版运行页面只接入逐日step=0工程；不静默改变亚日时间步。")
            dates = dict(day_start=day0, yrc_start=year0, day_end=day1, yrc_end=year1, step=step)
        except (ValueError, IndexError):
            errors.append(f"{time_name}需两行表头及有效的5个整数日期/步长字段。")
    climate = groups.get("climate", [])
    # cli_pmeas/tmeas/smeas/hmeas/wmeas: file names below two header lines.
    for name in climate[3:8]:
        if name in files:
            for row in _lines(files, name)[2:]:
                if row.strip():
                    _reference(files, row.strip(), name, errors)
    if climate and climate[0] in files:
        for row in _lines(files, climate[0])[2:]:
            values = row.split()
            if values:
                if len(values) < 7:
                    errors.append(f"{climate[0]}天气站记录字段不足。")
                else:
                    for ref in values[2:7]:
                        _reference(files, ref, climate[0], errors)
                    if (values[1].lower() != "null" or "sim" in values[2:7]) and (len(climate) < 2 or climate[1] not in files):
                        errors.append("天气站要求WGN，但file.cio没有有效的WGN文件。")
    from ecodata.models.swat_project_dependencies import weather_dependencies
    dependencies = {}
    try:
        dependencies = weather_dependencies(files)
        errors.extend(dependencies['errors'])
        warnings.extend(dependencies['warnings'])
    except (ValueError, UnicodeError) as exc:
        errors.append(f'天气依赖无法读取：{exc}')
    warnings.append("检查file.cio直接引用、天气文件引用、显式对象天气站关联、具名WGN月结构及日期；尚不检查全部数据库深层引用、完整拓扑和所有天气覆盖，检查通过不等于模型接受。")
    result = dict(errors=sorted(set(errors)), warnings=warnings, time_file=time_name, dates=dates,
                files=len(files), target_version=str(selected_major), declared_major=declared_major,
                declared_version=declared_version,
                weather_dependencies=dependencies)
    from ecodata.interface.engineering_completeness import swat
    result['completeness']=swat(files,result)
    return result


def set_dates(files, start, end, engine_id=None):
    if start > end:
        raise SchemaError("开始日期不能晚于结束日期。")
    report = check_project(files, engine_id=engine_id)
    if report["errors"]:
        raise SchemaError("；".join(report["errors"]))
    result = dict(files)
    name = report["time_file"]
    lines = _lines(files, name)
    lines[2:] = [f"{start.timetuple().tm_yday} {start.year} {end.timetuple().tm_yday} {end.year} 0"]
    result[name] = ("\n".join(lines) + "\n").encode("ascii")
    return result


ENGINE_LABELS = {
    "gnu62": "62.0.0 · GNU构建（原有程序）",
    "editor62": "rev.62 · 官方Editor 4.0.2随附Intel构建",
    "official61": "rev.61.0.2 · 官方Intel构建（61系列工程，需原生核验）",
    "official60": "rev.60.5.7 · 官方Editor历史存档（原版本工程）",
}
EDITOR_ENGINE_SHA256 = "d680b142b6029762d9a9deeade50494ed0ec5a7e7f68fbace193b2c760f163d0"
EDITOR_ENGINE = Path(r"D:\EcoDataEngine\tools\SWATPlusEditor402\resources\app.asar.unpacked\static\swat_exe\swatplus-62-ifo-win_amd64-Rel.exe")
LEGACY_ENGINE_SHA256 = "f9148a6ffc5227c7cbefdda7f2a8ac868a669048dabde1d85793c33a43690ef5"
LEGACY_ENGINE = ROOT / "runtime" / "swatplus-61.0.2-ifx-win_amd64-Rel.exe"
LEGACY_RUNTIME = ROOT / 'runtime' / 'intel2024'
LEGACY_RUNTIME_PINS = {
    'libifcoremd.dll': '1745bc152f595033082c22f573ecc02a9f2adc8c9bebb306bf3a86a6342c42b3',
    'libifportmd.dll': 'ed3b99abc22c506fe3b97eb1b1f9bbf3d38dffc35029b490e44f13f7a254d0e5',
    'libmmd.dll': '2b8609039193a1a8749e8c78724aa1bf080f9f2811ecb8df15582309fcf7011c',
    'svml_dispmd.dll': '351a7e6b6daf78ed5a87730e7cde1ecd98faf0b52f26a64bd6fc507e7ead7325',
    'libiomp5md.dll': '7f37e24b28383b21a06a198927c3adda2f0e3a780d553922b3819220e84f7615',
}


def engine_path(engine_id="gnu62"):
    if engine_id not in ENGINE_LABELS:
        raise SchemaError("未知的SWAT+运行程序；只能选择管理员登记且校验通过的程序。")
    path = ROOT / "runtime" / "swatplus-62.0.0.exe" if engine_id == "gnu62" else EDITOR_ENGINE
    expected = ENGINE_SHA256 if engine_id == "gnu62" else EDITOR_ENGINE_SHA256
    if engine_id == "official61":
        path, expected = LEGACY_ENGINE, LEGACY_ENGINE_SHA256
    if engine_id == 'official60':
        path = ROOT / 'runtime' / 'rev60.5.7_64rel.exe'
        expected = 'de01a927eebe00b4467ff53424175e247e7808e0bee99790f3d75db1b0a64179'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SchemaError("缺少或校验失败：" + ENGINE_LABELS[engine_id] + "；不接收上传的可执行文件。")
    if engine_id == 'official61':
        for name, pin in LEGACY_RUNTIME_PINS.items():
            dependency = LEGACY_RUNTIME / name
            if not dependency.is_file() or hashlib.sha256(dependency.read_bytes()).hexdigest() != pin:
                raise SchemaError('61系列官方运行库缺少或校验失败：' + name)
    return path


def _save(job):
    state = {k: v for k, v in job.items() if not k.startswith("_")}
    tmp = Path(job["directory"]) / "runner_status.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(tmp.with_name("runner_status.json"))


def _work(job, engine, timeout):
    path = Path(job["directory"])
    try:
        with (path / "runner.log").open("wb") as log:
            environment = os.environ.copy()
            if engine == LEGACY_ENGINE:
                environment['PATH'] = str(LEGACY_RUNTIME) + os.pathsep + environment.get('PATH', '')
            proc = subprocess.Popen([str(engine)], cwd=path, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, shell=False,
                                    env=environment,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            with _LOCK:
                job["_process"] = proc
                job["pid"] = proc.pid
                if job.get("_cancelled"):
                    proc.terminate()
                _save(job)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                job["_timed_out"] = True
                proc.kill()
                code = proc.wait()
        with _LOCK:
            job["returncode"] = code
            fresh_success = (path / "success.fin").is_file() and (path / "success.fin").stat().st_size > 0
            job["state"] = ("stopped" if job.get("_cancelled") else "timeout" if job.get("_timed_out") else
                            "succeeded" if code == 0 and fresh_success else "failed")
            job["success_file"] = fresh_success
    except Exception as exc:
        with _LOCK:
            job.update(state="failed", error=str(exc))
    finally:
        with _LOCK:
            job["finished_at"] = time.time()
            _save(job)


def start_job(files, jobs_root, timeout=1200, engine_id=None):
    selected = engine_id or "gnu62"
    report = check_project(files, engine_id=selected)
    if report["errors"]:
        raise SchemaError("；".join(report["errors"]))
    engine = engine_path() if engine_id is None else engine_path(engine_id)
    with _LOCK:
        if any(j["state"] == "running" for j in _JOBS.values()):
            raise SchemaError("已有SWAT+任务运行中；请先等待或停止，避免争用电脑资源。")
        ident = uuid.uuid4().hex
        directory = Path(jobs_root).resolve() / ident
        directory.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name, data in files.items():
            if name != PurePosixPath(name).name or name in {"runner.log", "runner_status.json", "runner_manifest.json"}:
                raise SchemaError("输入含非法或运行器保留文件名。")
            if name.lower() == "success.fin":
                continue  # Never accept an old success marker as new evidence.
            (directory / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        manifest = dict(target="SWAT+rev." + report["target_version"] + " input format", engine_id=selected,
                        engine_label=ENGINE_LABELS[selected], engine_path=str(engine),
                        engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(), input_sha256=hashes,
                        runtime_dependencies=LEGACY_RUNTIME_PINS if selected == 'official61' else {},
                        checks=report, timeout_seconds=timeout)
        (directory / "runner_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        job = dict(id=ident, directory=str(directory), state="running", started_at=time.time())
        _JOBS[ident] = job
        _save(job)
        threading.Thread(target=_work, args=(job, engine, timeout), daemon=True).start()
        return ident


def job_status(ident):
    with _LOCK:
        if ident in _JOBS:
            return {k: v for k, v in _JOBS[ident].items() if not k.startswith("_")}
    if not isinstance(ident,str) or not re.fullmatch('[0-9a-f]{32}',ident):raise SchemaError('无效SWAT+任务号')
    directory=ROOT/'runs/swat_plus'/ident
    record=json.loads((directory/'runner_status.json').read_text(encoding='utf-8'))
    if record.get('id')!=ident or Path(record.get('directory','')).resolve()!=directory.resolve():raise SchemaError('SWAT+记录与保存位置不匹配')
    return {**record,'process_control_available':False}


def recent_jobs():
    with _LOCK:
        return [(j["id"], j["state"]) for j in reversed(list(_JOBS.values()))]


def stop_job(ident):
    with _LOCK:
        job = _JOBS[ident]
        if job["state"] == "running":
            job["_cancelled"] = True
            proc = job.get("_process")
            if proc and proc.poll() is None:
                proc.terminate()


def log_tail(ident):
    path = Path(job_status(ident)["directory"]) / "runner.log"
    if not path.exists():
        return "程序正在启动……"
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 32000))
        return handle.read().decode("utf-8", errors="replace")


def result_zip(ident):
    status = job_status(ident)
    if status["state"] == "running":
        raise SchemaError("任务尚在运行，请结束后下载。")
    directory = Path(status["directory"])
    if sum(p.stat().st_size for p in directory.iterdir() if p.is_file()) > 250 * 1024 * 1024:
        raise SchemaError("运行目录超过250 MiB；请从工作目录直接取文件，避免网页压缩占用过多内存。")
    manifest = json.loads((directory / "runner_manifest.json").read_text(encoding="utf-8"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                data = path.read_bytes()
                if path.name not in manifest["input_sha256"] or hashlib.sha256(data).hexdigest() != manifest["input_sha256"][path.name]:
                    archive.writestr(path.name, data)
    return buffer.getvalue()
