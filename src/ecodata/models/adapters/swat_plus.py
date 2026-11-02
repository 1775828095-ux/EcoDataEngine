"""Conservative SWAT+ native text writer.

The writer emits only files for which the input data or an explicit official
default template provides evidence. It never invents HRU/channel/aquifer IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.standardized import ModelAdapter


MISSING_WITHOUT_TEMPLATE = [
    "object.cnt",
    "hru-data.hru",
    "topography.hyd",
    "hydrology.hyd",
    "soils.sol",
    "landuse.lum",
    "weather-wgn.cli",
    "channel-lte.cha",
    "aquifer.aqu",
]

TEMPLATE_COPY_FILES = [
    "object.cnt",
    "hru-data.hru",
    "hru.con",
    "topography.hyd",
    "hydrology.hyd",
    "soils.sol",
    "landuse.lum",
    "plant.ini",
    "soil_plant.ini",
    "lum.dtl",
    "cntable.lum",
    "cons_practice.lum",
    "ovn_table.lum",
    "management.sch",
    "management(CTcorn).sch",
    "fertilizer.frt",
    "tillage.til",
    "manure_db.frt",
    "manure_om.frt",
    "snow.sno",
    "carbon.bsn",
    "carbon_lyr.bsn",
    "weather-wgn.cli",
]


ADDITIONAL_WEATHER = {
    "solar_radiation": ("slr", "MJ/m2/day"),
    "relative_humidity": ("hmd", "1"),
    "wind_speed": ("wnd", "m/s"),
}


class SWATPlusWriter:
    """Generate an auditable partial SWAT+ project folder."""

    def write(self, config: dict, template_dir: str | Path | None = None) -> dict[str, bytes]:
        required = {"project_name", "start", "end", "station", "weather"}
        missing = [key for key in required if key not in config]
        if missing:
            raise SchemaError("SWAT+ 配置缺少：" + "、".join(missing))

        start = pd.Timestamp(config["start"])
        end = pd.Timestamp(config["end"])
        if end <= start:
            raise SchemaError("SWAT+ end 必须晚于 start")

        station = config["station"]
        if not {"name", "latitude", "longitude", "elevation"} <= set(station):
            raise SchemaError("SWAT+ 气象站缺少 name/latitude/longitude/elevation")

        additional = config.get("additional_weather", [])
        if not isinstance(additional, list) or any(not isinstance(n, str) or n not in ADDITIONAL_WEATHER for n in additional) or len(set(additional)) != len(additional):
            raise SchemaError("additional_weather 只能选择 solar_radiation、relative_humidity、wind_speed，且不能重复")
        if additional and config.get("target_model_version") != "62.0.0":
            raise SchemaError("新增天气文件需显式 target_model_version=62.0.0")
        if additional and (not isinstance(station['name'], str) or not station['name'] or any(c.isspace() for c in station['name'])):
            raise SchemaError("站点名不能含空格或换行")
        frame = pd.DataFrame(config["weather"])
        required_columns = {"time", "precipitation", "air_temperature_max", "air_temperature_min"}
        if not required_columns <= set(frame.columns):
            raise SchemaError("SWAT+ 气象缺少：" + "、".join(sorted(required_columns - set(frame.columns))))
        frame = frame.copy()
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        if frame["time"].isna().any():
            raise SchemaError("SWAT+ 气象时间缺失或无法解析")
        if frame["time"].dt.tz is not None:
            raise SchemaError("SWAT+ 日气象需明确的无时区 YYYY-MM-DD 日期标签；小时数据不得直接混入")
        if (frame["time"].dt.hour != 0).any() or (frame["time"].dt.minute != 0).any() or (frame["time"].dt.second != 0).any():
            raise SchemaError("SWAT+ 气象要求日标签；小时数据必须先显式聚合并重新标注")
        if frame["time"].duplicated().any():
            raise SchemaError("SWAT+ 气象时间重复；不允许静默删除记录")
        if not frame["time"].is_monotonic_increasing:
            raise SchemaError("SWAT+ 气象时间乱序；不允许静默排序")
        if frame.empty:
            raise SchemaError("SWAT+ 气象数据为空")
        expected = pd.date_range(start, end, freq="D")
        actual_days = frame.time.dt.normalize().to_numpy(dtype="datetime64[ns]")
        expected_days = expected.to_numpy(dtype="datetime64[ns]")
        if len(frame) != len(expected) or not np.array_equal(actual_days, expected_days):
            raise SchemaError("SWAT+ 日气象必须从工程 start 到 end 连续逐日；缺日或超出范围拒绝，不自动补值或外推")
        for column in ("precipitation", "air_temperature_max", "air_temperature_min"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in additional:
            if column not in frame:
                raise SchemaError("SWAT+ 气象缺少：" + column)
            try:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise SchemaError(column + "需数值") from exc
            if not np.isfinite(frame[column]).all():
                raise SchemaError(column + "包含缺失/无效值；没有 WGN，不自动填补或生成 -99")
            if (frame[column] < 0).any() or (column == "relative_humidity" and (frame[column] > 1).any()):
                raise SchemaError(column + "超出允许范围：辐射和风速 >=0，相对湿度 0–1")
        frame["year"] = frame.time.dt.year
        frame["yday"] = frame.time.dt.dayofyear

        files: dict[str, bytes] = {}
        files["time.sim"] = self._time_sim(start, end)
        files["station.pcp"] = self._weather_file(frame, station, "pcp")
        files["station.tem"] = self._weather_file(frame, station, "tem")
        files["pcp.cli"] = self._cli("station.pcp")
        files["tmp.cli"] = self._cli("station.tem")
        for column in additional:
            kind, unit = ADDITIONAL_WEATHER[column]
            files[f"station.{kind}"] = self._additional_file(frame, station, column, kind)
            files[f"{kind}.cli"] = self._cli(f"station.{kind}")
        if additional:
            # Keep all five station headers consistent without changing legacy output.
            metadata = f"{int(frame.year.max()) - int(frame.year.min()) + 1} 0 {float(station['latitude'])!r} {float(station['longitude'])!r} {float(station['elevation'])!r}"
            for name in ["station.pcp", "station.tem"] + [f"station.{ADDITIONAL_WEATHER[n][0]}" for n in additional]:
                records = files[name].decode("utf-8").splitlines()
                records[2] = metadata
                files[name] = ("\n".join(records) + "\n").encode("utf-8")
        files["weather-sta.cli"] = self._weather_sta(station, additional)

        template = Path(template_dir) if template_dir else None
        if template and not template.is_dir():
            raise SchemaError("SWAT+ 官方模板目录不存在：" + str(template))
        files["file.cio"] = self._file_cio(config["project_name"], template)
        if additional:
            lines = files["file.cio"].decode("utf-8").splitlines()
            climate = [i for i, line in enumerate(lines) if line.split() and line.split()[0] == "climate"]
            if len(climate) != 1:
                raise SchemaError("file.cio 模板需唯一 climate 索引行")
            i = climate[0]; slots = lines[i].split()
            if len(slots) != 10:
                raise SchemaError("file.cio climate 索引不符合 62.0.0 的 9 个文件槽位")
            for column in additional:
                kind = ADDITIONAL_WEATHER[column][0]
                slots[{"slr": 6, "hmd": 7, "wnd": 8}[kind]] = f"{kind}.cli"
            lines[i] = "  ".join(slots)
            files["file.cio"] = ("\n".join(lines) + "\n").encode("utf-8")
        for name in ("print.prt", "codes.bsn", "parameters.bsn", "plants.plt"):
            if template and (template / name).exists():
                files[name] = (template / name).read_bytes()
        copied_template_files: list[str] = []
        if config.get("template_mode") == "topology_and_defaults":
            if template is None:
                raise SchemaError("template_mode 要求提供官方 B2 模板目录")
            for name in TEMPLATE_COPY_FILES:
                source = template / name
                if source.exists() and name not in files:
                    files[name] = source.read_bytes()
                    copied_template_files.append(name)

        generated = sorted(files)
        missing_files = [name for name in MISSING_WITHOUT_TEMPLATE if name not in files]
        validation = {
            "status": "WARN",
            "scope": "swat_plus_native_text_candidate",
            "native_model_input_verified": False,
            "runnable_without_template": False,
            "generated_files": generated,
            "missing_without_engineering_template": missing_files,
            "template_mode": config.get("template_mode", "none"),
            "copied_template_files": copied_template_files,
            "warning": (
                "template_mode=topology_and_defaults 时，除 time.sim、station.pcp、station.tem、pcp.cli、tmp.cli、"
                "weather-sta.cli 和 file.cio 的替换部分外，其余文件复制自官方 Ames_sub1 模板，不能视为由 A 推导。"
                if copied_template_files else ""
            ),
            "evidence": {
                "time_from_A": True,
                "precipitation_from_A": True,
                "temperature_from_A": True,
                "object_topology_from_template_or_user": bool(template),
                "official_defaults_copied": [name for name in generated if name in {"print.prt", "codes.bsn", "parameters.bsn", "plants.plt"}],
            },
        }
        manifest = {
            "model": "SWAT+",
            "writer": "SWATPlusWriter",
            "reference_case": "swat-model/swatplus refdata/Ames_sub1",
            "project_name": config["project_name"],
            "template_dir": str(template) if template else None,
            "files": {
                name: {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for name, content in files.items()
            },
            "validation": validation,
        }
        if additional:
            validation["additional_weather"] = additional
            validation["weather_format_target"] = "SWAT+ 62.0.0 / Editor 4.0.2 text weather"
            validation["missing_policy"] = "reject: no WGN or artificial values"
            manifest["weather_format_target"] = "SWAT+ 62.0.0"
        files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        files["validation_report.json"] = json.dumps(validation, ensure_ascii=False, indent=2).encode("utf-8")
        return files

    @staticmethod
    def _cli(filename: str) -> bytes:
        kind = filename.rsplit(".", 1)[-1]
        index = {"pcp": "pcp", "tem": "tmp", "slr": "slr", "hmd": "hmd", "wnd": "wnd"}[kind]
        return f"{index}.cli\nFILENAME\n{filename}\n".encode("utf-8")

    @staticmethod
    def _weather_file(frame: pd.DataFrame, station: dict, kind: str) -> bytes:
        start_year = int(frame.year.min())
        end_year = int(frame.year.max())
        prefix = "STATION.pcp" if kind == "pcp" else "STATION.Tmp"
        lines = [
            prefix,
            " NBYR    TSTEP      LAT     LONG     ELEV",
            f"   {end_year - start_year + 1}        0 {float(station['latitude']):8.4f} {float(station['longitude']):9.4f} {float(station['elevation']):6.0f}",
        ]
        for row in frame.itertuples(index=False):
            if kind == "pcp":
                value = -99.0 if not np.isfinite(row.precipitation) else float(row.precipitation)
                lines.append(f"{int(row.year)} {int(row.yday)} {value}")
            else:
                tmax = -99.0 if not np.isfinite(row.air_temperature_max) else float(row.air_temperature_max)
                tmin = -99.0 if not np.isfinite(row.air_temperature_min) else float(row.air_temperature_min)
                lines.append(f"{int(row.year)}\t{int(row.yday)}\t{tmax}\t{tmin}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    @staticmethod
    def _additional_file(frame: pd.DataFrame, station: dict, column: str, kind: str) -> bytes:
        # Same three header records as cli_smeas/hmeas/wmeas in SWAT+ 62.0.0.
        lines = [f"station.{kind}", " NBYR    TSTEP      LAT     LONG     ELEV",
                 f"{int(frame.year.max()) - int(frame.year.min()) + 1} 0 {station['latitude']:.12g} {station['longitude']:.12g} {station['elevation']:.12g}"]
        for row in frame.to_dict(orient="records"):
            lines.append(f"{int(row['year'])} {int(row['yday'])} {float(row[column])!r}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    @staticmethod
    def _weather_sta(station: dict, additional=()) -> bytes:
        lines = [
            "weather-sta.cli:",
            "name                           wgn               pcp                tmp                slr              hmd             wnd         wnd_dir          atmo_dep",
            f"{station['name']:<24}          null        station.pcp        station.tem              sim              sim             sim            null              null",
        ]
        if additional:
            gages = {ADDITIONAL_WEATHER[n][0]: "station." + ADDITIONAL_WEATHER[n][0] for n in additional}
            lines[2] = "  ".join([station['name'], "null", "station.pcp", "station.tem",
                                  gages.get("slr", "sim"), gages.get("hmd", "sim"), gages.get("wnd", "sim"), "null", "null"])
        return ("\n".join(lines) + "\n").encode("utf-8")

    @staticmethod
    def _time_sim(start: pd.Timestamp, end: pd.Timestamp) -> bytes:
        lines = [
            "time.sim:",
            "day_start  yrc_start   day_end   yrc_end      step",
            f"       0      {start.year}         0      {end.year}         0",
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    @staticmethod
    def _file_cio(project_name: str, template: Path | None) -> bytes:
        if template and (template / "file.cio").exists():
            text = (template / "file.cio").read_text(encoding="utf-8", errors="replace")
            text = text.replace("file.cio: AMES", f"file.cio: {project_name}")
            text = text.replace("ames.pcp", "station.pcp").replace("ames.tem", "station.tem")
            return text.encode("utf-8")
        lines = [
            f"file.cio: {project_name}",
            "simulation        time.sim          print.prt         null              object.cnt        null",
            "basin             codes.bsn         parameters.bsn    carbon.bsn",
            "climate           weather-sta.cli   weather-wgn.cli   null              pcp.cli           tmp.cli           null              null              null              null",
            "connect           hru.con           null              null              null              null              null              null              null              null              null              null              null              null",
            "channel           null              null              null              null              null              null              null              null",
            "reservoir         null              null              null              null              null              null              null              null",
            "routing_unit      null              null              null              null",
            "hru               hru-data.hru      null",
            "exco              null              null              null              null              null              null",
            "recall            null",
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")


class SWATPlusAdapter(ModelAdapter):
    """Adapter boundary for SWAT+ text output; native acceptance remains unverified."""

    supports_partial = True

    def package(self, config: dict, template_dir: str | Path | None = None) -> dict[str, bytes]:
        return SWATPlusWriter().write(config, template_dir)

    def save_package(self, directory: str | Path, config: dict, template_dir: str | Path | None = None) -> Path:
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise SchemaError("SWAT+ 输出目录非空，禁止覆盖旧验收证据")
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in self.package(config, template_dir).items():
            (directory / name).write_bytes(content)
        return directory

    def native_validation(self, result, schema):
        return {"verified": False, "reason": "尚未使用官方 SWAT+ 可执行程序读取或运行候选工程"}
