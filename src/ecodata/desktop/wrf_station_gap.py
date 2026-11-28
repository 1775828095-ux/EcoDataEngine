"""Single-time WRF surface-temperature gap filling with spatial holdouts."""
from contextlib import ExitStack
from hashlib import sha256
from itertools import product
from pathlib import Path, PurePosixPath
import json
import re
import ssl
from zipfile import ZipFile, is_zipfile

import certifi
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from ecodata.algorithms.spatial_selection_protocol import error_metrics
from ecodata.evaluation.blocked_confirmation import block_error_change

TOOL = "wrf_station_gap"
LABEL = "WRF 多站同时段气温补缺"
STAGE = "wrf_station_gap_recommendation"
POLICY_NAME = "wrf_station_gap_policy.json"
DOMAINS = {"neighbors": (4, 8, 16), "power": (1, 2),
           "lapse_k_per_km": (0., 3.25, 6.5), "strength": (.5, .75, 1.)}
CATALOG = {
    "wrf_grid": ("WRF T2 双线性插值（网格单源）", ()),
    "station_idw": ("站点反距离插值（站点单源）", ("neighbors", "power", "lapse_k_per_km")),
    "wrf_height": ("WRF 双线性插值＋海拔订正", ("lapse_k_per_km",)),
    "wrf_residual": ("WRF 双线性插值＋邻站残差", ("neighbors", "power", "strength")),
    "wrf_height_residual": ("WRF 双线性插值＋海拔订正＋邻站偏差订正",
                            ("neighbors", "power", "lapse_k_per_km", "strength")),
}
SINGLE = ("wrf_grid", "station_idw", "wrf_height")
COLUMNS = {"station_id", "report_index", "time_utc", "platform", "latitude", "longitude",
           "elevation_m", "temperature_K", "temperature_K_qc"}
SEED = 42
BLOCK_M = 240_000
MIN_DISTANCE_M = 20_000
MAX_ROUNDS = 4
WEB_POLICY_STAGE = "wrf_station_gap_web_policy"


def discover(scan):
    """Locate a unique portable pair; scientific field checks occur on opening."""
    from .scan import safe

    root = scan.get("path")
    if root is None:
        return None
    root = Path(root)
    locators = {item["path"] for item in scan.get("directory_members", {}).values()}
    paths = [root] if root.is_file() else [Path(item[0]) for item in scan.get("signature", ())]
    for path in paths:
        prefix = path.relative_to(root).as_posix() if root.is_dir() else ""
        if prefix:
            locators.add(prefix)
        if path.suffix.casefold() == ".zip" and is_zipfile(path):
            with ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir() and safe(member.filename):
                        locators.add(f"{prefix}!/{member.filename}" if prefix else member.filename)
    def matches(pattern):
        return sorted(path for path in locators if re.fullmatch(
            pattern, PurePosixPath(path).name, flags=re.IGNORECASE))
    grids = matches(r"WRF_background_surface_.+\.nc")
    stations = matches(r"surface_observations_.+UTC\.csv")
    policies = matches(re.escape(POLICY_NAME))
    if len(grids) != 1 or len(stations) != 1:
        return None
    return dict(grid=grids[0], stations=stations[0], policy_files=policies)


def read_policy(policy):
    if policy is None:
        return None
    required = {"min_mae_gain_k", "max_block_mae_increase_k", "basis"}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("WRF 验收声明须包含 min_mae_gain_k、max_block_mae_increase_k 和 basis。")
    if (not isinstance(policy["basis"], str) or not policy["basis"].strip()
            or len(policy["basis"]) > 800):
        raise ValueError("WRF 温度退化限度需要预先声明业务依据。")
    for key in required - {"basis"}:
        value = policy[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} 须为有限非负数，单位 K。")
    if policy["min_mae_gain_k"] <= 0:
        raise ValueError("整体 MAE 最小改善必须大于零。")
    return dict(policy)


def verify_public_quote(url, quote, api_key):
    from urllib.request import HTTPSHandler, Request, build_opener
    from .service import NoRedirect

    request = Request("https://api.deepseek.com/v1/tools/fetch",
                      data=json.dumps({"url": url}).encode("utf-8"),
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    opener = build_opener(NoRedirect(), HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())))
    with opener.open(request, timeout=15) as response:
        payload = json.loads(response.read(2_000_000))
    content = payload.get("content") if isinstance(payload, dict) else None
    return isinstance(content, str) and " ".join(quote.split()) in " ".join(content.split())


def resolve_policy(declaration, api_key, mock, progress, receipts):
    explicit = read_policy(declaration)
    if explicit is not None:
        return dict(explicit, origin="declared")
    default = dict(min_mae_gain_k=0., max_block_mae_increase_k=None, basis=None,
                   origin="no_applicable_local_limit")
    if not api_key or mock:
        return dict(default, search_status="SKIPPED_SIMULATION" if mock else "SKIPPED_NO_API_KEY")

    from urllib.parse import urlsplit
    from . import service

    request = service.preview(dict(stage=WEB_POLICY_STAGE, variable="2 m air temperature",
        use="retrospective multi-station gap filling, spatial holdout",
        local_metric="spatial-block MAE of fusion minus the strongest single-source comparator",
        required_evidence="A normative numerical limit that directly applies to this use and metric."))
    if progress:
        progress("本次未声明局部限度，正在通过 DeepSeek 联网查询是否有直接适用的规范。")
    try:
        answer = service.send(request, request.digest, api_key=api_key, on_progress=progress,
                              on_receipt=receipts.append)
        url = answer.get("source_url")
        quote = answer.get("quote")
        basis = answer.get("applicability")
        value = answer.get("max_block_mae_increase_k")
        steps = answer.get("server_tool_calls", [])
        opened = {call.get("page", {}).get("url")
                  for step in steps for call in step.get("calls", [])
                  if call.get("status") == "completed" and call.get("page")}
        cited = {row.get("url_citation", {}).get("url") for row in answer.get("citations", [])}
        source = urlsplit(url) if isinstance(url, str) and len(url) <= 8192 else None
        official = bool(source and source.scheme == "https" and not (source.username or source.password)
                        and (source.hostname or "").endswith(
                            (".gov", ".gov.cn", ".gov.uk", ".gc.ca", ".int", ".europa.eu")))
        if (official and url in opened and url in cited and isinstance(quote, str)
                and 0 < len(quote) <= 800 and isinstance(basis, str) and basis.strip()
                and type(value) in (int, float) and np.isfinite(value) and value >= 0):
            numbers = re.findall(r"(\d+(?:\.\d+)?)\s*(?:°\s*[Cc]|℃|K\b)", quote)
            metric = re.search(r"\bmae\b|mean absolute error|平均绝对误差", quote, re.I)
            comparison = re.search(r"increas|degrad|relative|增量|退化|相对", quote, re.I)
            if (metric and comparison and len(numbers) == 1 and float(numbers[0]) == value
                    and verify_public_quote(url, quote, api_key)):
                return dict(default, max_block_mae_increase_k=value, basis=basis[:800],
                            origin="cited_public_limit", source_url=url, quote=quote)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        return dict(default, search_status="UNAVAILABLE", search_reason=str(error))
    return dict(default, search_status="NO_APPLICABLE_CITED_LIMIT")


def align(grid_path, observations_path):
    from netCDF4 import Dataset

    with Dataset(grid_path) as grid:
        names = {"x", "y", "latitude", "longitude", "T2", "HGT"}
        if not names <= grid.variables.keys() or not getattr(grid, "analysis_time_utc", None):
            raise ValueError("WRF 单时刻网格缺少 T2、HGT、空间坐标或 analysis_time_utc。")
        clock = pd.Timestamp(grid.analysis_time_utc)
        if clock.tz is None:
            raise ValueError("WRF 分析时刻须有明确时区。")
        clock = clock.tz_convert("UTC")
        def values(name):
            return np.ma.asarray(grid[name][:], dtype=float).filled(np.nan)
        if any(getattr(grid[k], "units", "").strip() != unit
               for k, unit in {"T2": "K", "HGT": "m", "x": "m", "y": "m"}.items()):
            raise ValueError("本路线需要 T2 为 K、HGT 和投影轴为 m。")
        mapping = getattr(grid["T2"], "grid_mapping", "")
        if mapping not in grid.variables or grid["HGT"].grid_mapping != mapping:
            raise ValueError("T2 和 HGT 需要相同且明确的 CF 投影。")
        projection = CRS.from_cf({key: grid[mapping].getncattr(key) for key in grid[mapping].ncattrs()})
        if not projection.is_projected or any(axis.unit_conversion_factor != 1 for axis in projection.axis_info):
            raise ValueError("WRF 站点距离须在明确的米制投影下计算。")
        transform = Transformer.from_crs("EPSG:4326", projection, always_xy=True)
        x, y = values("x"), values("y")
        if (x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 2
                or any(not np.isfinite(axis).all() or not (np.diff(axis) > 0).all() for axis in (x, y))):
            raise ValueError("WRF 投影网格需要有限且递增的 x/y 轴。")
        if any(grid[key].dimensions != ("y", "x") for key in ("T2", "HGT", "latitude", "longitude")):
            raise ValueError("本路线只接已声明单时刻的二维 T2/HGT 网格。")
        ax, ay = transform.transform(values("longitude"), values("latitude"))
        if not (np.allclose(ax, x[None, :], atol=5., rtol=0)
                and np.allclose(ay, y[:, None], atol=5., rtol=0)):
            raise ValueError("WRF 投影与文件的经纬度不一致。")
        interpolation = {
            name: RegularGridInterpolator((y, x), values(field), bounds_error=False)
            for name, field in (("grid_temperature_k", "T2"), ("grid_elevation_m", "HGT"))
        }

    rows = pd.read_csv(observations_path, dtype={"station_id": "string"}, usecols=lambda name: name in COLUMNS)
    if not COLUMNS <= set(rows):
        raise ValueError("站点表缺字段：" + "、".join(sorted(COLUMNS - set(rows))))
    diagnostics = {"total_reports": len(rows), "analysis_time_utc": clock.isoformat(),
                   "projection_wkt": projection.to_wkt(),
                   "matching_window_seconds": 900, "task_mode": "retrospective_same_time_gap_filling"}
    raw_time = rows["time_utc"].str.replace("_", "T", regex=False)
    times = pd.to_datetime(raw_time, utc=True, errors="coerce", format="mixed")
    rows["time_offset_seconds"] = (times - clock).dt.total_seconds()
    rows["distance_seconds"] = rows["time_offset_seconds"].abs()
    for name in ("longitude", "latitude", "elevation_m", "temperature_K", "temperature_K_qc", "report_index"):
        rows[name] = pd.to_numeric(rows[name], errors="coerce")
    rows = rows[rows.platform.isin(["FM-12 SYNOP", "FM-15 METAR"])
                & rows.distance_seconds.le(900) & rows.station_id.fillna("").str.strip().ne("")
                & rows.latitude.between(-90, 90) & rows.longitude.between(-180, 180)
                & np.isfinite(rows.elevation_m)].copy()
    rows["x"], rows["y"] = transform.transform(rows.longitude.to_numpy(), rows.latitude.to_numpy())
    rows = rows[rows.x.between(x[0], x[-1]) & rows.y.between(y[0], y[-1])].copy()
    for name, interpolator in interpolation.items():
        rows[name] = interpolator(rows[["y", "x"]].to_numpy())
    rows = rows[np.isfinite(rows[list(interpolation)]).all(axis=1)].copy()
    diagnostics["covered_reports"] = len(rows)
    rows["qc_usable"] = rows.temperature_K_qc.eq(0) & rows.temperature_K.between(180, 340)
    diagnostics["covered_qc_usable_reports"] = int(rows.qc_usable.sum())
    rows[["cell_x", "cell_y"]] = (rows[["x", "y"]] / 5_000).round().astype(int)
    rows = (rows.sort_values(["qc_usable", "distance_seconds", "report_index"],
                            ascending=[False, True, True], kind="stable")
            .drop_duplicates("station_id").drop_duplicates(["cell_x", "cell_y"])
            .sort_values(["distance_seconds", "report_index"], kind="stable").reset_index(drop=True))
    return rows, diagnostics


def partition(rows):
    valid = rows.qc_usable.to_numpy()
    block_xy = np.floor(rows[["x", "y"]].to_numpy() / BLOCK_M).astype(int)
    blocks, ids = np.unique(block_xy[valid], axis=0, return_inverse=True)
    if len(blocks) < 15:
        raise ValueError(f"独立空间分块不足：仅 {len(blocks)} 个有实测的块，开发及最终验证各需至少三个。")
    order = np.random.default_rng(SEED).permutation(len(blocks))
    count = len(order) // 5
    roles = np.full(len(rows), "gap", dtype=object)
    roles[valid] = np.select([np.isin(ids, order[:count]), np.isin(ids, order[count:2 * count])],
                             ["final", "development"], default="training")
    block_keys = [f"{x}/{y}" for x, y in block_xy]
    return roles, np.array(block_keys)


def neighbors_for(xy, training):
    tree = cKDTree(xy[training])
    excluded = tree.query_ball_point(xy, MIN_DISTANCE_M, return_length=True)
    pool = len(training) - excluded
    # Extend the query only by the excluded prefix; never build a full distance matrix.
    width = min(len(training), int(excluded.max()) + max(DOMAINS["neighbors"]))
    distance, index = tree.query(xy, k=np.arange(1, width + 1))
    distance[distance <= MIN_DISTANCE_M] = np.inf
    keep = np.argsort(distance, axis=1, kind="stable")[:, :max(DOMAINS["neighbors"])]
    return training[np.take_along_axis(index, keep, axis=1)], np.take_along_axis(distance, keep, axis=1), pool


def recipe(method, **parameters):
    return method, tuple(sorted(parameters.items()))


def fixed_controls(available):
    yield recipe("wrf_grid")
    for lapse in (3.25, 6.5):
        yield recipe("wrf_height", lapse_k_per_km=lapse)
    for k, p, lapse in product(available, DOMAINS["power"], DOMAINS["lapse_k_per_km"]):
        yield recipe("station_idw", neighbors=k, power=p, lapse_k_per_km=lapse)
    if available:
        # Nominal comparison settings. Fusion delivery requires a separate AI nomination.
        shared = dict(neighbors=min(available, key=lambda k: abs(k - 8)), power=2, strength=1.)
        yield recipe("wrf_residual", **shared)
        yield recipe("wrf_height_residual", lapse_k_per_km=6.5, **shared)


def predict(spec, rows, ids, distance, targets):
    method, params = spec
    params = dict(params)
    lapse = params.get("lapse_k_per_km", 0.) / 1_000
    height = rows.elevation_m.to_numpy()
    raw = rows.grid_temperature_k.to_numpy()
    background = raw - lapse * (height - rows.grid_elevation_m.to_numpy())
    if method in {"wrf_grid", "wrf_height"}:
        return background[targets]
    k = params["neighbors"]
    if ids.shape[1] < k:
        return np.full(len(targets), np.nan)
    near, km = ids[targets, :k], distance[targets, :k] / 1_000
    weights = 1 / km ** params["power"]
    supported = np.isfinite(km).all(axis=1)
    totals = weights.sum(axis=1, keepdims=True)
    np.divide(weights, totals, out=weights, where=totals > 0)
    observed = rows.temperature_K.to_numpy()[near]
    if method == "station_idw":
        corrected = observed + lapse * (height[near] - height[targets, None])
        estimate = (weights * corrected).sum(axis=1)
    else:
        residual = observed - background[near]
        estimate = background[targets] + params["strength"] * (weights * residual).sum(axis=1)
    return np.where(supported, estimate, np.nan)


def scored(values, observed, blocks):
    metrics = error_metrics(values, observed)
    entries = {str(key): error_metrics(values[blocks == key], observed[blocks == key])
               for key in np.unique(blocks)}
    return dict(metrics=metrics, blocks=entries)


def compare(score, baseline, policy, *, phase="selection"):
    from ecodata.evaluation.blocked_confirmation import trial_range

    rows = [dict(block=key, candidate=value, baseline=baseline["blocks"][key])
            for key, value in score["blocks"].items()]
    blocks = trial_range(rows, phase=phase)
    for block in blocks["blocks"]:
        block["mae_change_k"] = block["candidate"]["MAE"] - block["baseline"]["MAE"]
    gain = baseline["metrics"]["MAE"] - score["metrics"]["MAE"]
    worst = max(row["mae_change_k"] for row in blocks["blocks"])
    result = dict(mae_gain_k=gain, **block_error_change(score["metrics"], baseline["metrics"]),
                  worst_block_mae_increase_k=worst, range_comparison=blocks)
    minimum = policy["min_mae_gain_k"]
    limit = policy["max_block_mae_increase_k"]
    result.update(overall_ok=gain > 0 and gain >= minimum,
                  local_ok=limit is None or worst <= limit, local_limit_applied=limit is not None)
    return result


def development_feedback(comparison):
    blocks = comparison["range_comparison"]["blocks"]
    ordered = sorted(blocks, key=lambda row: row["mae_change_k"])

    def summarize(row):
        return dict(block=row["block"], sample_count=row["candidate"]["count"],
                    mae_change_k=round(row["mae_change_k"], 4),
                    candidate_bias_k=round(row["candidate"]["Bias"], 4),
                    baseline_bias_k=round(row["baseline"]["Bias"], 4))

    evidence = {key: value for key, value in comparison.items() if key != "range_comparison"}
    evidence["range_comparison"] = dict(
        measured_blocks=len(blocks), degraded_blocks=sum(row["mae_change_k"] > 0 for row in blocks),
        improved_blocks=sum(row["mae_change_k"] < 0 for row in blocks),
        blocks=[summarize(row) for row in [*ordered[:2], *ordered[max(2, len(ordered) - 4):]]])
    return evidence


def report_spec(spec):
    return dict(method=spec[0], parameters=dict(spec[1]))


def sample_training(rows, roles, blocks, pool):
    training = np.flatnonzero(roles == "training")
    group = pd.DataFrame(dict(index=training, block=blocks[training],
                              elevation=rows.elevation_m.iloc[training].to_numpy()))
    group = group.sort_values(["block", "elevation"], kind="stable")
    grouped = [frame.index.to_numpy() for _, frame in group.groupby("block", sort=True)]
    # One spatial block contributes at most one row until the others contribute.
    from .sampling_plan import distributed
    chosen = [group.loc[grouped[i][len(grouped[i]) // 2], "index"]
              for i in distributed(range(len(grouped)), 24)]
    names = ["temperature_K", "grid_temperature_k", "elevation_m",
             "grid_elevation_m", "time_offset_seconds"]
    out = rows.iloc[chosen][names].round(3).copy()
    out["available_training_neighbors"] = pool[chosen]
    return out.to_dict("records")


def training_block_evidence(rows, training, blocks):
    sample = rows.iloc[training]
    bias = sample.grid_temperature_k - sample.temperature_K
    height = sample.elevation_m - sample.grid_elevation_m
    grouped = pd.DataFrame(dict(bias=bias.to_numpy(), height=height.to_numpy(), block=blocks[training]))
    medians = grouped.groupby("block")[["bias", "height"]].median()
    return dict(station_count=len(sample), block_count=len(medians),
                grid_minus_station_bias_k=medians.bias.quantile([.1, .5, .9]).round(3).tolist(),
                station_minus_grid_height_m=medians.height.quantile([.1, .5, .9]).round(1).tolist(),
                quantiles=[.1, .5, .9],
                blocks_with_positive_grid_bias=int((medians.bias > 0).sum()),
                blocks_with_negative_grid_bias=int((medians.bias < 0).sum()))


def offered_catalogue(available):
    return [dict(method=key, description=label,
                 parameter_domains={name: list(available if name == "neighbors" else DOMAINS[name])
                                    for name in fields})
            for key, (label, fields) in CATALOG.items() if available or "neighbors" not in fields]


def proposals(reply, available):
    if not isinstance(reply, dict) or not isinstance(reply.get("proposals"), list):
        raise ValueError("DeepSeek 未提供可执行候选列表。")
    for item in reply["proposals"][:4]:
        if not isinstance(item, dict):
            continue
        method, params = item.get("method"), item.get("parameters")
        if not isinstance(method, str) or method not in CATALOG or not isinstance(params, dict):
            continue
        fields = CATALOG[method][1]
        if set(fields) != set(params):
            continue
        values = []
        for key in fields:
            domain = available if key == "neighbors" else DOMAINS[key]
            options = params[key] if isinstance(params[key], list) else [params[key]]
            if not options or any(type(x) not in (int, float) or x not in domain for x in options):
                break
            values.append(tuple(dict.fromkeys(options)))
        else:
            reason = item.get("reason")
            from itertools import islice
            for combination in islice(product(*values), 12):
                yield recipe(method, **dict(zip(fields, combination))), reason[:800] if isinstance(reason, str) else ""


def select(candidates, methods):
    return min((spec for spec in candidates if spec[0] in methods),
               key=lambda spec: (candidates[spec]["metrics"]["MAE"], spec), default=None)


def qualified(result):
    return result.get("overall_ok") is True and result.get("local_ok") is True


def run(scan, task, stage, progress=None, *, api_key=None, mock=False):
    from .file_tools import selected
    from .scan import assert_current
    from . import service

    assert_current(scan)
    policy_files = task.get("policy_files", [])
    if len(policy_files) > 1 or (policy_files and "policy" in task):
        raise ValueError("请提供唯一的 WRF 验收声明。")
    with ExitStack() as readers:
        def member(locator):
            return readers.enter_context(selected(scan["path"], locator, max_leaf_bytes=64*1024**2))[0]
        declaration = task.get("policy")
        if policy_files:
            path = readers.enter_context(selected(scan["path"], policy_files[0], max_leaf_bytes=8192))[0]
            if path.stat().st_size > 8192:
                raise ValueError("WRF 验收声明超过 8 KiB。")
            declaration = json.loads(path.read_text(encoding="utf-8-sig"))
        receipts = []
        policy = resolve_policy(declaration, api_key, mock, progress, receipts)
        grid, stations = member(task["grid"]), member(task["stations"])
        if any(path.stat().st_size > 64*1024**2 for path in (grid, stations)):
            raise ValueError("单时段 WRF 输入超出每个文件 64 MiB 的读取预算。")
        fingerprints = tuple(sha256(path.read_bytes()).hexdigest() for path in (grid, stations))
        rows, inventory = align(grid, stations)

    roles, blocks = partition(rows)
    train, dev, final, gap = (np.flatnonzero(roles == role)
                              for role in ("training", "development", "final", "gap"))
    ids, distance, pool = neighbors_for(rows[["x", "y"]].to_numpy(), train)
    available = tuple(k for k in DOMAINS["neighbors"] if pool[dev].min() >= k)
    observed = rows.temperature_K.to_numpy()
    inventory.update(usable_stations=int(rows.qc_usable.sum()), gap_targets=len(gap),
        spatial_blocks=len(np.unique(blocks[rows.qc_usable])),
        cohorts={role: dict(stations=int((roles == role).sum()),
                            blocks=len(set(blocks[roles == role])))
                 for role in ("training", "development", "final")},
        development_neighbor_counts=dict(minimum=int(pool[dev].min()), median=float(np.median(pool[dev])),
                                         maximum=int(pool[dev].max())),
        spatial_block_m=BLOCK_M, excluded_neighbor_distance_m=MIN_DISTANCE_M)
    inventory["development_kth_neighbor_distance_km"] = {
        str(k): dict(median=float(np.median(distance[dev, k - 1]) / 1_000),
                     maximum=float(distance[dev, k - 1].max() / 1_000))
        for k in available
    }
    if progress:
        progress(f"WRF 已配对 {inventory['usable_stations']} 个实测站，划分 {inventory['spatial_blocks']} 个空间块。")

    (stage / "validation_plan.json").write_text(service.dump(dict(
        input_sha256=fingerprints, policy=policy, partition_seed=SEED,
        block_roles={block: role for role, block in zip(roles, blocks) if role != "gap"})), encoding="utf-8")

    scores, provenance, unsupported = {}, {}, []
    def evaluate(spec, origin, reason=""):
        if spec in provenance:
            return False
        provenance[spec] = dict(origin=origin, reason=reason)
        estimate = predict(spec, rows, ids, distance, dev)
        if not np.isfinite(estimate).all():
            unsupported.append(report_spec(spec))
            return False
        scores[spec] = scored(estimate, observed[dev], blocks[dev])
        return True

    controls = [spec for spec in fixed_controls(available) if evaluate(spec, "fixed_control")]
    single = select(scores, SINGLE)
    fusion_methods = tuple(key for key in CATALOG if key not in SINGLE)

    development_comparisons = {}
    def development(spec):
        if spec not in development_comparisons:
            development_comparisons[spec] = compare(scores[spec], scores[single], policy)
        return development_comparisons[spec]

    catalogue = offered_catalogue(available)
    payload = dict(stage=STAGE, target="WRF 2 米气温多站同时段补缺", benchmark="MAE on common development sites",
        algorithm_catalogue=catalogue, acceptance_policy=policy,
        data_profile={k: inventory[k] for k in ("total_reports", "usable_stations", "spatial_blocks",
            "development_neighbor_counts", "development_kth_neighbor_distance_km",
            "matching_window_seconds", "spatial_block_m", "excluded_neighbor_distance_m")},
        fields=dict(T2="原网格两米气温 K", HGT="原网格地形高度 m",
                    temperature_K="站点实测 K", elevation_m="站点海拔 m", temperature_K_qc="质量标记 0"),
        paired_training_samples=sample_training(rows, roles, blocks, pool),
        training_spatial_block_summary=training_block_evidence(rows, train, blocks))
    nomination_input = dict(sample_count=len(payload["paired_training_samples"]),
                            training_blocks=payload["training_spatial_block_summary"]["block_count"])
    rounds = []
    nominations = {}
    latest_scored = []
    for index in range(MAX_ROUNDS):
        if not (api_key or mock) or not available:
            break
        if any(qualified(development(spec)) for spec in nominations if spec in scores):
            break
        leaders = [select(scores, (method,)) for method in CATALOG]
        previous = []
        for spec in dict.fromkeys([*leaders, *latest_scored]):
            if spec is None:
                continue
            comparison = development_feedback(development(spec))
            comparison = dict(comparison,
                mae_gain_shortfall_k=max(0., policy["min_mae_gain_k"] - comparison["mae_gain_k"]))
            previous.append(dict(**report_spec(spec), **comparison, metrics=scores[spec]["metrics"], nominated=spec in nominations))
        request = service.preview(dict(payload, round=index + 1, development_scores=previous,
            previous_round_status=rounds[-1]["status"] if rounds else None,
            already_tested=[report_spec(spec) for spec in provenance],
            best_single=report_spec(single)))
        if progress:
            source = "模拟请求" if mock else "发送给 DeepSeek"
            progress(f"第 {index + 1} 轮 · {source} {nomination_input['sample_count']} 条训练配对小样本"
                     f"和 {nomination_input['training_blocks']} 个训练空间块的统计，等待分析和提名。")
        try:
            reply = service.send(request, request.digest, mock, api_key=api_key, on_progress=progress,
                                 on_receipt=receipts.append)
            offered = list(proposals(reply, available))
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
            rounds.append(dict(round=index+1, status="API_OR_RESPONSE_ERROR", reason=str(error)))
            if progress:
                progress(f"第 {index + 1} 轮 · DeepSeek 未返回可用提名：{error}")
            if isinstance(error, (ValueError, KeyError, TypeError)):
                continue
            break
        new, scored_nominations = [], []
        for spec, reason in offered:
            if evaluate(spec, "simulated" if mock else "deepseek", reason):
                new.append(report_spec(spec))
            if spec[0] in fusion_methods:
                nominations.setdefault(spec, dict(round=index + 1, reason=reason))
                if spec in scores:
                    scored_nominations.append(spec)
        latest_scored = sorted(set(scored_nominations), key=lambda spec: (scores[spec]["metrics"]["MAE"], spec))[:4]
        rounds.append(dict(round=index+1, status="EVALUATED" if scored_nominations else "NO_SCORABLE_FUSION_NOMINATION",
                           new_candidates=new, proposed_candidates=[report_spec(spec) for spec, _ in offered],
                           request_sha256=request.digest))
        if progress:
            passed = sum(qualified(development(spec)) for spec in set(scored_nominations))
            progress(f"第 {index + 1} 轮 · 收到 {len(offered)} 个目录内组合，"
                     f"其中 {len(set(scored_nominations))} 个融合组合完成开发评分，{passed} 个达标。")

    nominated_scores = {spec: scores[spec] for spec in nominations if spec in scores}
    eligible = {spec: score for spec, score in nominated_scores.items() if qualified(development(spec))}
    finalist = select(eligible, fusion_methods)
    best_exploratory = select(nominated_scores, fusion_methods)
    fixed_scores = {spec: scores[spec] for spec in controls}
    finalists = list(dict.fromkeys(
        [spec for method in CATALOG if (spec := select(fixed_scores, (method,))) is not None]
        + [spec for spec in (single, finalist) if spec is not None]))
    (stage / "selection_freeze.json").write_text(service.dump(dict(
        best_single=report_spec(single), chosen=report_spec(finalist) if finalist else None,
        controls=[report_spec(spec) for spec in finalists], policy=policy,
        confirmation_used_for_selection=False)), encoding="utf-8")

    if progress and finalist is not None:
        progress(f"已冻结候选 · {CATALOG[finalist[0]][0]}。正在本地核对留出站点。")
    # No request to DeepSeek and no selection runs after opening these labels.
    final_scores = {}
    for spec in finalists:
        prediction = predict(spec, rows, ids, distance, final)
        if np.isfinite(prediction).all():
            final_scores[spec] = scored(prediction, observed[final], blocks[final])
    final_review = (compare(final_scores[finalist], final_scores[single], policy, phase="confirmation")
                    if finalist in final_scores and single in final_scores else None)
    if not nominated_scores:
        status = "NO_FUSION_NOMINATION"
    elif finalist is None:
        status = "DEVELOPMENT_REJECTED"
    elif final_review is None:
        status = "HOLDOUT_COVERAGE_INSUFFICIENT"
    elif not qualified(final_review):
        status = "HOLDOUT_REJECTED"
    else:
        status = "BATCH_RECOMMENDED"
    recommended = status == "BATCH_RECOMMENDED"
    selected_spec = finalist if recommended else single
    if len(gap):
        predicted = predict(selected_spec, rows, ids, distance, gap)
        output = rows.iloc[gap][["station_id", "latitude", "longitude", "elevation_m"]].copy()
        supported = np.isfinite(predicted)
        output["temperature_fill_k"] = np.where(supported, predicted, rows.grid_temperature_k.iloc[gap])
        output["status"] = np.where(supported, "FILLED", "RAW_WRF_FALLBACK")
        output.to_csv(stage / "station_gap_predictions.csv", index=False)
        inventory["gap_filled"] = len(gap)
        inventory["gap_raw_wrf_fallback"] = int((~supported).sum())
    verified = rows.iloc[final][["station_id", "latitude", "longitude", "temperature_K"]].copy()
    verified["block"] = blocks[final]
    verified["best_single_k"] = predict(single, rows, ids, distance, final)
    if finalist:
        verified["fusion_candidate_k"] = predict(finalist, rows, ids, distance, final)
    verified.to_csv(stage / "holdout_predictions.csv", index=False)
    report = dict(status=status, recommended=recommended,
        independent_acceptance=False, inventory=inventory,
        policy=policy, best_single=report_spec(single),
        candidate=report_spec(finalist or best_exploratory) if (finalist or best_exploratory) else None,
        delivered_recipe=report_spec(selected_spec), final_review=final_review,
        recommendation_scope="same_time_spatial_holdout", forecast_verified=False,
        upstream_observation_independence="UNKNOWN",
        development=[dict(**report_spec(spec), **provenance[spec], metrics=score["metrics"],
                          nomination=nominations.get(spec), comparison=development(spec)) for spec, score in scores.items()],
        final_controls=[dict(**report_spec(spec), status="SCORED" if spec in final_scores else "INSUFFICIENT_SUPPORT",
                            metrics=final_scores.get(spec, {}).get("metrics"))
                        for spec in finalists],
        unsupported_development=unsupported, nomination_input=nomination_input,
        recommendation_rounds=rounds, api_receipts=receipts,
        recommendation_status="SIMULATED" if mock else "ONLINE_ATTEMPTED" if rounds else
                              "NO_SUPPORTED_NEIGHBORS" if not available else "NO_API_KEY")
    (stage / "wrf_station_gap_report.json").write_text(service.dump(report), encoding="utf-8")
    return report
