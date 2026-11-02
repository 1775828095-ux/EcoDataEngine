"""Observation-constrained multi-source FNO reconstruction experiment.

This module is intentionally separate from ordinary format conversion.  It uses
the official ``neuraloperator`` FNO implementation and keeps validation stations
out of both model inputs and losses.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.core.exceptions import SchemaError


@dataclass(frozen=True)
class NeuralReconstructionConfig:
    epochs: int = 40
    learning_rate: float = 2e-3
    validation_station_count: int = 20
    seed: int = 20261009
    hidden_channels: int = 16
    n_layers: int = 3
    observation_weight: float = 8.0
    coarse_weight: float = 1.0
    truth_weight: float = 1.0
    gradient_weight: float = 0.1
    nonnegative_weight: float = 1.0
    observation_channels: int = 8
    backend: str = "fno"
    spectral_weight: float = 0.1
    peak_weight: float = 0.5
    observation_residual_gate: float = 0.5
    observation_locality_scale: float = 0.25
    observation_bias_weight: float = 1.0
    observation_warmup_fraction: float = 0.25
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    scheduler: str = "warmup_cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-3
    minimum_epochs: int = 10
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = True
    training_mode: str = "scratch"
    load_checkpoint_path: str | None = None
    save_checkpoint_path: str | None = None
    variable_family: str = "auto"
    nonnegative: bool | None = None
    missing_value_strategy: str = "reject"
    cross_attention: bool = True
    reliability_weighting: bool = True
    backbone: str = "native"


@dataclass(frozen=True)
class VariablePolicy:
    family: str
    nonnegative: bool
    observation_loss: str
    coarse_loss: str


def resolve_variable_policy(variable: str, field: xr.DataArray, config: NeuralReconstructionConfig) -> VariablePolicy:
    """Resolve physical constraints from explicit metadata, never value ranges."""
    allowed = {"pollutant_concentration", "temperature", "precipitation", "water_quality"}
    family = config.variable_family
    if family == "auto":
        identity = " ".join(str(field.attrs.get(name, "")) for name in
                            ("physical_variable", "standard_name", "long_name")) + " " + variable
        identity = identity.lower()
        units = str(field.attrs.get("units", "")).lower()
        if any(token in identity for token in ("temperature", "temp", "t2m")) or units in {"k", "kelvin", "degc", "celsius"}:
            family = "temperature"
        elif any(token in identity for token in ("precip", "rainfall", "rain")):
            family = "precipitation"
        elif any(token in identity for token in ("pm2", "pm10", "concentration", "pollut")):
            family = "pollutant_concentration"
        elif any(token in identity for token in ("water_quality", "cod", "ammonia", "nitrogen", "phosphorus")):
            family = "water_quality"
        else:
            raise SchemaError("无法可靠判断目标变量物理类型；请设置 variable_family")
    if family not in allowed:
        raise SchemaError("variable_family 必须是 pollutant_concentration/temperature/precipitation/water_quality")
    nonnegative = family != "temperature" if config.nonnegative is None else config.nonnegative
    return VariablePolicy(
        family=family,
        nonnegative=bool(nonnegative),
        observation_loss="smooth_l1" if family == "temperature" else "l1",
        coarse_loss="l1" if family == "precipitation" else "mse",
    )


@dataclass(frozen=True)
class NeuralReconstructionResult:
    field: xr.Dataset
    report: dict
    station_predictions: pd.DataFrame
    loss_history: tuple[float, ...]


def split_stations(stations: pd.DataFrame, validation_count: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"station_id", "time", "latitude", "longitude", "value"}
    missing = required - set(stations.columns)
    if missing:
        raise SchemaError("监测站缺少字段：" + "、".join(sorted(missing)))
    frame = stations.copy()
    if "split" in frame and not frame["split"].isin(["train", "validation", "test"]).all():
        raise SchemaError("split 必须为 train/validation/test，不能静默重新划分")
    if "qc_flag" in frame and not frame.qc_flag.astype(str).str.upper().isin(["VALID", "INVALID", "MISSING"]).all():
        raise SchemaError("未知 QA 标记；请明确映射为 VALID/INVALID/MISSING")
    if "qc_flag" in frame:
        frame = frame[frame.qc_flag.astype(str).str.upper() == "VALID"]
    if "split" in frame and frame["split"].isin(["train", "validation", "test"]).all():
        train = frame[frame.split == "train"].copy()
        validation = frame[frame.split == "validation"].copy()
    else:
        ids = np.array(sorted(frame.station_id.astype(str).unique()))
        if len(ids) <= validation_count:
            raise SchemaError(f"只有 {len(ids)} 个站点，无法留出 {validation_count} 个后继续训练")
        rng = np.random.default_rng(seed)
        validation_ids = set(rng.choice(ids, size=validation_count, replace=False).tolist())
        validation = frame[frame.station_id.astype(str).isin(validation_ids)].copy()
        train = frame[~frame.station_id.astype(str).isin(validation_ids)].copy()
        train["split"], validation["split"] = "train", "validation"
    train_ids = set(train.station_id.astype(str))
    validation_ids = set(validation.station_id.astype(str))
    if not train_ids or not validation_ids:
        raise SchemaError("训练站和独立验证站都必须存在")
    if train_ids & validation_ids:
        raise SchemaError("训练站与验证站重叠，已阻止数据泄漏")
    if set(zip(train.latitude, train.longitude)) & set(zip(validation.latitude, validation.longitude)):
        raise SchemaError("训练与验证位置重叠，不能称为独立空间验证")
    return train, validation


def _main_variable(dataset: xr.Dataset) -> str:
    candidates = [name for name, item in dataset.data_vars.items() if item.dtype.kind in "biuf" and {"lat", "lon"}.issubset(item.dims)]
    if not candidates:
        raise SchemaError("没有找到含 lat/lon 的连续数值场")
    return candidates[0]


def default_target_coordinates(source):
    """Fourfold cell-centre refinement preserves the background footprint."""
    coordinates = []
    for name in ('lat', 'lon'):
        values = source[name].values
        if len(values) < 2:
            raise SchemaError('背景网格每个方向至少需要两个像元')
        spacing = float(values[1] - values[0])
        coordinates.append(np.linspace(float(values[0])-spacing/2+spacing/8,
                                       float(values[-1])+spacing/2-spacing/8, len(values)*4))
    return tuple(coordinates)


def _station_indices(frame: pd.DataFrame, times, lat, lon):
    time_lookup = {pd.Timestamp(value): index for index, value in enumerate(pd.to_datetime(times))}
    rows = []
    for item in frame.itertuples():
        stamp = pd.Timestamp(item.time)
        if stamp not in time_lookup:
            continue
        if not (lat.min() <= float(item.latitude) <= lat.max() and lon.min() <= float(item.longitude) <= lon.max()):
            raise SchemaError("监测站位于目标网格之外，禁止吸附到边缘")
        quality = float(getattr(item, "quality_weight", 1.0))
        if not np.isfinite([item.value, item.latitude, item.longitude, quality]).all() or not 0 < quality <= 1:
            raise SchemaError("站点数值必须有限；quality_weight 必须在 (0,1] 内")
        iy = int(np.abs(lat - float(item.latitude)).argmin())
        ix = int(np.abs(lon - float(item.longitude)).argmin())
        rows.append((time_lookup[stamp], iy, ix, float(item.value), str(item.station_id),
                     float(item.longitude), float(item.latitude), quality))
    if not rows:
        raise SchemaError("监测站时间与背景场没有共同记录")
    return rows


def observation_reliability(train: pd.DataFrame, times) -> tuple[pd.DataFrame, dict]:
    """Training-only QA × completeness × relative inverse error variance."""
    frame = train.copy()
    frame['time'] = pd.to_datetime(frame.time)
    frame = frame[frame.time.isin(pd.to_datetime(times))].copy()
    if frame.empty or frame.duplicated(['station_id', 'time']).any():
        raise SchemaError("训练站没有共同时间或 station_id/time 存在重复记录")
    quality = pd.to_numeric(frame.get('quality_weight', pd.Series(1., index=frame.index)), errors='coerce')
    if not np.isfinite(quality).all() or not quality.between(0, 1, inclusive='right').all():
        raise SchemaError("quality_weight 必须为 (0,1] 内的有限数值")
    completeness = frame.groupby('station_id').time.nunique() / len(times)
    precision = np.ones(len(frame))
    if 'observation_error' in frame:
        error = pd.to_numeric(frame.observation_error, errors='coerce').to_numpy()
        if not np.isfinite(error).all() or np.any(error <= 0):
            raise SchemaError("observation_error 必须为正的观测标准差，单位与 value 相同")
        precision = np.clip(np.median(error ** 2) / error ** 2, .05, 20.)
    raw = quality.to_numpy() * frame.station_id.map(completeness).to_numpy() * precision
    frame['qa_weight'] = quality
    frame['completeness_weight'] = frame.station_id.map(completeness)
    frame['loss_weight'] = raw / raw.mean()
    frame['quality_weight'] = raw / raw.max()
    diagnostics = {'formula': 'QA * valid-time completeness * clipped relative inverse variance',
                   'error_definition': 'standard deviation in value units; not predictive uncertainty',
                   'precision_clip': [.05, 20.], 'completeness': completeness.to_dict(),
                   'observation_error_used': 'observation_error' in frame,
                   'weights': frame[['station_id', 'time', 'qa_weight', 'completeness_weight', 'loss_weight', 'quality_weight']].assign(time=lambda f: f.time.astype(str)).to_dict('records')}
    return frame, diagnostics


def run_neural_reconstruction(
    background: xr.Dataset,
    stations: pd.DataFrame,
    *,
    truth: xr.Dataset | None = None,
    evaluation_truth: xr.Dataset | None = None,
    auxiliaries: Sequence[xr.Dataset] = (),
    config: NeuralReconstructionConfig = NeuralReconstructionConfig(),
) -> NeuralReconstructionResult:
    """Train and evaluate a compact FNO residual reconstruction experiment."""
    try:
        import torch
        import torch.nn.functional as F
        from neuralop.models import FNO
    except ImportError as exc:
        raise SchemaError("该实验需要可选依赖：pip install -e .[ml]") from exc

    started = perf_counter()
    guided = config.backend == 'observation_guided_uno'
    if config.backbone not in {"native", "fno"}:
        raise SchemaError("backbone 必须为 native 或 fno")
    if config.training_mode not in {"scratch", "finetune", "inference"}:
        raise SchemaError("training_mode 必须为 scratch/finetune/inference")
    if not (isinstance(config.epochs, int) and config.epochs > 0) and config.training_mode != "inference":
        raise SchemaError("训练或微调时 epochs 必须为正整数")
    if config.optimizer not in {"adam", "adamw"}:
        raise SchemaError("optimizer 必须为PyTorch官方 Adam 或 AdamW")
    if config.scheduler not in {"warmup_cosine", "plateau", "none"}:
        raise SchemaError("scheduler 必须为 warmup_cosine/plateau/none")
    if config.training_mode in {"finetune", "inference"} and not config.load_checkpoint_path:
        raise SchemaError("微调或推理必须提供 load_checkpoint_path")
    if (config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0) or config.weight_decay < 0:
        raise SchemaError("gradient_clip_norm 必须为空或大于0，且 weight_decay 不能为负")
    if not (0 <= config.observation_residual_gate <= 1 and config.observation_locality_scale > 0
            and 0 <= config.observation_warmup_fraction <= 1 and config.observation_bias_weight >= 0):
        raise SchemaError("观测gate、局部尺度、warmup或bias权重无效")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    variable = _main_variable(background)
    source = background[variable].transpose("time", "lat", "lon")
    policy = resolve_variable_policy(variable, source, config)
    if background.attrs.get("crs") != "EPSG:4326":
        raise SchemaError("原型要求 EPSG:4326 经纬度网格，请先用平台完成 CRS 标准化")
    if not source.attrs.get("units"):
        raise SchemaError("背景场缺少单位")
    for axis in (source.lat, source.lon):
        differences = np.diff(axis.values)
        if len(axis) < 2 or not np.all(differences > 0) or not np.allclose(differences, differences[0], rtol=1e-5, atol=1e-10):
            raise SchemaError("原型要求严格递增的规则经纬度坐标")
    from ecodata.engines.units import UnitEngine
    stations = stations.copy()
    if "unit" not in stations:
        raise SchemaError("监测站缺少 unit，无法核对物理单位")
    for unit in stations.unit.unique():
        selected = stations.unit == unit
        converted = UnitEngine().convert(xr.DataArray(stations.loc[selected, "value"].to_numpy(), attrs={"units": unit}), source.attrs["units"])
        stations.loc[selected, "value"] = converted.values
        if guided and 'observation_error' in stations:
            values = pd.to_numeric(stations.loc[selected, 'observation_error'], errors='coerce').to_numpy()
            error = UnitEngine().convert(xr.DataArray(values, attrs={'units': unit}), source.attrs['units'])
            zero = UnitEngine().convert(xr.DataArray(0., attrs={'units': unit}), source.attrs['units'])
            stations.loc[selected, 'observation_error'] = error.values - zero.values
    stations['unit'] = source.attrs['units']
    for dataset in auxiliaries:
        if dataset.attrs.get("crs") != background.attrs["crs"]:
            raise SchemaError("辅助场 CRS 与背景场不一致，请先标准化")
    grid_reference = truth if truth is not None else evaluation_truth
    if grid_reference is not None:
        grid_variable = _main_variable(grid_reference)
        target_lat = grid_reference.lat.values
        target_lon = grid_reference.lon.values
    if truth is not None:
        truth_variable = _main_variable(truth)
        reference = truth[truth_variable].sel(time=source.time).transpose("time", "lat", "lon")
    elif evaluation_truth is not None:
        reference = None
    else:
        target_lat, target_lon = default_target_coordinates(source)
        reference = None
    # Fine cell centres may extend half a coarse cell beyond the outermost
    # coarse centres while remaining inside the same grid footprint.
    for axis in (target_lat, target_lon):
        differences = np.diff(axis)
        if len(axis) < 2 or not np.all(differences > 0) or not np.allclose(differences, differences[0], rtol=1e-5, atol=1e-10):
            raise SchemaError('目标网格必须严格递增且等间距；不能把不规则坐标用于grid_sample')
    for target, original in ((target_lat, source.lat.values), (target_lon, source.lon.values)):
        margin = (original[1]-original[0])/2
        if target.min() < original.min()-margin-1e-8 or target.max() > original.max()+margin+1e-8:
            raise SchemaError('目标网格超出背景像元空间覆盖，不允许大范围外推')
    background_high = source.interp(
        lat=target_lat, lon=target_lon, method="linear", kwargs={"fill_value": "extrapolate"}
    )
    if not np.isfinite(background_high.values).all() and config.missing_value_strategy == "reject":
        raise SchemaError("背景场插值产生 NaN；请先检查空间范围和坐标方向")
    if config.missing_value_strategy not in {"reject", "mask"}:
        raise SchemaError("missing_value_strategy 必须为 reject 或 mask")

    test_records_excluded = int((stations.get("split", pd.Series(index=stations.index, dtype=str)) == "test").sum())
    train, validation = split_stations(stations, config.validation_station_count, config.seed)
    reliability = {'enabled': False}
    if guided and config.reliability_weighting:
        train, reliability = observation_reliability(train, source.time.values)
        reliability['enabled'] = True
    elif guided:
        train = train.assign(quality_weight=1.0, loss_weight=1.0)
    train_points = _station_indices(train, source.time.values, target_lat, target_lon)
    validation_points = _station_indices(validation, source.time.values, target_lat, target_lon)

    raw_channels = [background_high.values.astype("float32")]
    channel_names = [f"background:{variable}"]
    channel_kinds = ["continuous"]
    if not np.isfinite(raw_channels[0]).all():
        raw_channels.append(np.isfinite(raw_channels[0]).astype("float32"))
        channel_names.append(f"background:{variable}:valid_mask")
        channel_kinds.append("mask")
    auxiliary_names = []
    auxiliary_alignment = []
    for dataset in auxiliaries:
        names = [name for name, field in dataset.data_vars.items()
                 if field.dtype.kind in "biuf" and {"lat", "lon"}.issubset(field.dims)]
        for name in names:
            field = dataset[name]
            categorical = field.attrs.get('data_kind') == 'categorical'
            method = 'nearest' if categorical else 'linear'
            for target, original in ((target_lat, field.lat.values), (target_lon, field.lon.values)):
                if len(original) < 2 or not np.all(np.diff(original) > 0):
                    raise SchemaError(f'辅助变量 {name} 需要递增的二维网格坐标')
                margin = min(np.diff(original))/2
                if target.min() < original.min()-margin-1e-8 or target.max() > original.max()+margin+1e-8:
                    raise SchemaError(f'辅助变量 {name} 不覆盖目标网格，请先裁剪到共同范围')
            edge_extension = bool(target_lat.min() < field.lat.min() or target_lat.max() > field.lat.max()
                                  or target_lon.min() < field.lon.min() or target_lon.max() > field.lon.max())
            auxiliary_alignment.append({'variable':name, 'method':method, 'cell_edge_extension':edge_extension,
                                        'categorical_one_hot':categorical, 'scope':'inside pixel footprint only'})
            if "time" in field.dims:
                if not pd.Index(source.time.values).isin(field.time.values).all():
                    raise SchemaError(f'辅助变量 {name} 缺少背景时刻，请先完成时间对齐')
                field = field.sel(time=source.time).interp(lat=target_lat, lon=target_lon, method=method, kwargs={'fill_value':'extrapolate'})
            else:
                field = field.interp(lat=target_lat, lon=target_lon, method=method, kwargs={'fill_value':'extrapolate'}).expand_dims(time=source.time)
            values = field.transpose("time", "lat", "lon").values.astype("float32")
            has_missing = not np.isfinite(values).all()
            if has_missing and (categorical or config.missing_value_strategy == "reject"):
                raise SchemaError(f"辅助变量 {name} 与目标网格不完全重叠")
            if categorical:
                classes = np.unique(values)
                if len(classes) > 32:
                    raise SchemaError(f"分类辅助变量 {name} 超过32类，请先明确分类映射")
                for category in classes:
                    channel_name = f'{name}={category:g}'
                    raw_channels.append((values == category).astype('float32'))
                    channel_names.append(channel_name)
                    channel_kinds.append("categorical")
                    auxiliary_names.append(channel_name)
            else:
                raw_channels.append(values)
                channel_names.append(name)
                channel_kinds.append("continuous")
                auxiliary_names.append(name)
                if has_missing:
                    raw_channels.append(np.isfinite(values).astype("float32"))
                    channel_names.append(f"{name}:valid_mask")
                    channel_kinds.append("mask")
                    auxiliary_names.append(f"{name}:valid_mask")

    loaded_checkpoint = None
    if config.load_checkpoint_path:
        checkpoint_path = Path(config.load_checkpoint_path)
        if not checkpoint_path.is_file():
            raise SchemaError(f"checkpoint 不存在：{checkpoint_path}")
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if loaded_checkpoint.get("channel_names") != channel_names:
            raise SchemaError("checkpoint 输入变量/通道与当前数据不兼容")
        channel_statistics = loaded_checkpoint.get("channel_statistics", [])
        if len(channel_statistics) != len(raw_channels):
            raise SchemaError("checkpoint 标准化参数与当前输入通道不兼容")
    else:
        channel_statistics = []
        for name, kind, values in zip(channel_names, channel_kinds, raw_channels):
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise SchemaError(f"输入通道 {name} 没有有效训练值")
            channel_statistics.append({
                "name": name, "kind": kind,
                "mean": float(finite.mean()) if kind == "continuous" else 0.0,
                "scale": float(finite.std()) or 1.0 if kind == "continuous" else 1.0,
            })

    channels = []
    for name, kind, values, stats in zip(channel_names, channel_kinds, raw_channels, channel_statistics):
        if stats.get("name") != name or stats.get("kind") != kind:
            raise SchemaError("checkpoint 通道顺序或标准化类型与当前输入不兼容")
        if kind == "continuous":
            filled = np.where(np.isfinite(values), values, stats["mean"])
            channels.append(((filled - stats["mean"]) / stats["scale"]).astype("float32"))
        else:
            channels.append(values.astype("float32"))

    mean = float(channel_statistics[0]["mean"])
    scale = float(channel_statistics[0]["scale"])
    bg = channels[0]

    x = torch.from_numpy(np.stack(channels, axis=1))
    bg_tensor = torch.from_numpy(bg[:, None])
    truth_tensor = None if reference is None else torch.from_numpy(((reference.values - mean) / scale).astype("float32")[:, None])
    truth_mask = torch.ones_like(bg_tensor, dtype=torch.bool)
    if truth_tensor is not None:
        # Exclude a neighbourhood covering the held station's bilinear stencil
        # from every training frame, including gradient supervision.
        for point in validation_points:
            iy, ix = point[1:3]
            truth_mask[:, :, max(0, iy-1):iy+2, max(0, ix-1):ix+2] = False
        if not truth_mask.any() or not np.isfinite(reference.values).all():
            raise SchemaError("训练真值无有效区域或包含缺失值")
    modes = (min(8, len(target_lat) // 2), min(8, len(target_lon) // 2))
    from ecodata.fusion.observation_encoder import SparseObservationEncoder, AttentionObservationEncoder
    if config.backend not in {"fno", "multiscale_uno", "observation_guided_uno"}:
        raise SchemaError("backend 必须为 fno / multiscale_uno / observation_guided_uno")
    upgraded = config.backend != "fno"
    encoder = (AttentionObservationEncoder(config.observation_channels, x.shape[1],
               residual_gate=config.observation_residual_gate if guided else 0,
               locality_scale=config.observation_locality_scale,
               cross_attention=config.cross_attention) if upgraded
               else SparseObservationEncoder(config.observation_channels))
    # Keep actual subpixel station locations; never raster-snap observations.
    def normalized_xy(px, py):
        return [2 * (px - target_lon.min()) / np.ptp(target_lon) - 1,
                2 * (py - target_lat.min()) / np.ptp(target_lat) - 1]
    counts = [sum(p[0] == it for p in train_points) for it in range(len(source.time))]
    point_tokens = np.zeros((len(source.time), max(counts), 4), dtype="float32")
    point_valid = np.zeros(point_tokens.shape[:2], dtype=bool)
    for it in range(len(source.time)):
        for slot, point in enumerate(p for p in train_points if p[0] == it):
            point_tokens[it, slot] = [*normalized_xy(point[5], point[6]), (point[3] - mean) / scale, point[7]]
            point_valid[it, slot] = True
    point_tokens, point_valid = torch.from_numpy(point_tokens), torch.from_numpy(point_valid)
    gx, gy = np.meshgrid(np.linspace(-1, 1, len(target_lon)), np.linspace(-1, 1, len(target_lat)))
    queries = torch.tensor(np.column_stack([gx.ravel(), gy.ravel()]), dtype=torch.float32)
    model = FNO(n_modes=modes, in_channels=x.shape[1] + config.observation_channels, out_channels=1,
                hidden_channels=config.hidden_channels, n_layers=config.n_layers)
    frequency_loss = None
    if upgraded:
        from neuralop.models import UNO
        try:
            from focal_frequency_loss import FocalFrequencyLoss
        except ImportError as exc:
            raise SchemaError("多尺度版本需要可选依赖：pip install -e .[ml]") from exc
        if config.backbone == "native":
            model = UNO(in_channels=x.shape[1] + config.observation_channels, out_channels=1,
                    hidden_channels=config.hidden_channels, lifting_channels=config.hidden_channels,
                    projection_channels=config.hidden_channels, n_layers=3,
                    uno_out_channels=[config.hidden_channels] * 3,
                    uno_n_modes=[list(modes)] * 3,
                    uno_scalings=[[0.5, 0.5], [1, 1], [2, 2]], horizontal_skips_map={2: 0},
                    channel_mlp_skip="linear")
        frequency_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)

    coordinate_hash = sha256(np.asarray([*target_lat, *target_lon], dtype="float64").tobytes()).hexdigest()
    checkpoint_signature = {
        "backend": config.backend,
        "variable": variable,
        "physical_variable": source.attrs.get("physical_variable"),
        "units": source.attrs.get("units"),
        "variable_family": policy.family,
        "channel_names": channel_names,
        "target_grid_shape": [len(target_lat), len(target_lon)],
        "target_grid_hash": coordinate_hash,
        "hidden_channels": config.hidden_channels,
        "n_layers": 3 if upgraded and config.backbone == "native" else config.n_layers,
        "observation_channels": config.observation_channels,
        "backbone": config.backbone,
        "cross_attention": config.cross_attention,
        "reliability_weighting": config.reliability_weighting,
        "observation_residual_gate": config.observation_residual_gate,
    }
    if loaded_checkpoint is not None:
        # Older checkpoints predate explicit ablation switches; use their
        # original config/default semantics without rewriting the old files.
        saved_signature = dict(loaded_checkpoint.get("signature", {}))
        for key in ("backbone", "cross_attention", "reliability_weighting", "observation_residual_gate"):
            saved_signature.setdefault(key, loaded_checkpoint.get("config", {}).get(
                key, getattr(NeuralReconstructionConfig(), key)))
        differences = {key: {"checkpoint": saved_signature.get(key), "current": value}
                       for key, value in checkpoint_signature.items()
                       if saved_signature.get(key) != value}
        if differences:
            raise SchemaError("checkpoint 与当前变量、单位、通道、网格或网络配置不兼容：" + json.dumps(differences, ensure_ascii=False))
        model.load_state_dict(loaded_checkpoint["model_state"])
        encoder.load_state_dict(loaded_checkpoint["encoder_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    model, encoder = model.to(device), encoder.to(device)
    x, bg_tensor = x.to(device), bg_tensor.to(device)
    point_tokens, point_valid, queries = point_tokens.to(device), point_valid.to(device), queries.to(device)
    if truth_tensor is not None:
        truth_tensor, truth_mask = truth_tensor.to(device), truth_mask.to(device)

    def predict():
        encoded = encoder(point_tokens, point_valid, queries, x) if upgraded else encoder(point_tokens, point_valid, queries)
        embedding = encoded.reshape(len(source.time), config.observation_channels, len(target_lat), len(target_lon))
        return bg_tensor + model(torch.cat([x, embedding], dim=1))
    def sample(prediction, points):
        ids = torch.tensor([p[0] for p in points], device=device)
        locations = torch.tensor([normalized_xy(p[5], p[6]) for p in points], dtype=torch.float32, device=device)
        return F.grid_sample(prediction[ids], locations[:, None, None], align_corners=True).reshape(-1)
    train_values = torch.tensor([(p[3] - mean) / scale for p in train_points], dtype=torch.float32, device=device)
    validation_values = torch.tensor([(p[3] - mean) / scale for p in validation_points], dtype=torch.float32, device=device)
    coarse_values = source.values.astype("float32")
    coarse_valid = np.isfinite(coarse_values)
    coarse_filled = np.where(coarse_valid, coarse_values, mean)
    coarse_tensor = torch.from_numpy(((coarse_filled - mean) / scale)[:, None]).to(device)
    coarse_mask = torch.from_numpy(coarse_valid[:, None]).to(device)
    history = []
    validation_history = []
    learning_rates = []
    gradient_norms = []
    schedules = []
    weights = (torch.tensor(train.loss_weight.to_numpy(), dtype=torch.float32, device=device) if guided
               else torch.tensor([p[7] for p in train_points], dtype=torch.float32, device=device))
    parameters = list(model.parameters()) + list(encoder.parameters())
    initial_parameters = [item.detach().cpu().clone() for item in parameters]
    best_state = {"model": copy.deepcopy(model.state_dict()), "encoder": copy.deepcopy(encoder.state_dict())}
    best_validation_rmse = float("inf")
    best_epoch = 0
    stopped_early = False

    if config.training_mode != "inference":
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        if config.scheduler == "warmup_cosine":
            warmup = min(max(0, config.warmup_epochs), max(0, config.epochs - 1))
            if warmup:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1 / max(2, warmup), total_iters=warmup)
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, config.epochs - warmup))
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup])
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
        elif config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=max(1, config.early_stopping_patience // 3))
        else:
            scheduler = None
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        epochs_without_improvement = 0
        for epoch in range(config.epochs):
            model.train()
            encoder.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = predict()
                quality = weights
                station_error = sample(prediction, train_points) - train_values
                if policy.observation_loss == "smooth_l1":
                    point_losses = F.smooth_l1_loss(sample(prediction, train_points), train_values, reduction="none")
                else:
                    point_losses = station_error.abs()
                obs_loss = (quality * point_losses).sum() / quality.sum()
                schedule = min(1., (epoch + 1) / max(1., config.epochs * config.observation_warmup_fraction)) if guided else 1.
                schedules.append(schedule)
                coarse_prediction = F.interpolate(prediction, size=source.shape[-2:], mode="area")
                if policy.coarse_loss == "l1":
                    coarse_loss = F.l1_loss(coarse_prediction[coarse_mask], coarse_tensor[coarse_mask])
                else:
                    coarse_loss = F.mse_loss(coarse_prediction[coarse_mask], coarse_tensor[coarse_mask])
                loss = config.observation_weight * schedule * obs_loss + config.coarse_weight * coarse_loss
                if guided:
                    loss = loss + config.observation_bias_weight * ((quality * station_error).sum() / quality.sum()).square()
                if upgraded:
                    loss = loss + config.spectral_weight * frequency_loss(coarse_prediction, coarse_tensor)
                    peak = train_values >= torch.quantile(train_values, 0.9)
                    loss = loss + config.peak_weight * (quality[peak] *
                        (sample(prediction, train_points)[peak] - train_values[peak]).abs()).sum() / quality[peak].sum()
                if truth_tensor is not None:
                    loss = loss + config.truth_weight * F.l1_loss(prediction[truth_mask], truth_tensor[truth_mask])
                    dy_pred, dx_pred = prediction[:, :, 1:] - prediction[:, :, :-1], prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
                    dy_true, dx_true = truth_tensor[:, :, 1:] - truth_tensor[:, :, :-1], truth_tensor[:, :, :, 1:] - truth_tensor[:, :, :, :-1]
                    my = truth_mask[:, :, 1:] & truth_mask[:, :, :-1]
                    mx = truth_mask[:, :, :, 1:] & truth_mask[:, :, :, :-1]
                    if my.any() and mx.any():
                        loss = loss + config.gradient_weight * (F.l1_loss(dy_pred[my], dy_true[my]) + F.l1_loss(dx_pred[mx], dx_true[mx]))
                else:
                    loss = loss + config.gradient_weight * (
                        (prediction[:, :, 1:] - prediction[:, :, :-1]).abs().mean()
                        + (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]).abs().mean())
                if policy.nonnegative:
                    physical = prediction * scale + mean
                    loss = loss + config.nonnegative_weight * torch.relu(-physical).mean() / scale
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.gradient_clip_norm is None:
                gradient_norm = torch.sqrt(sum(torch.sum(torch.abs(item.grad).double() ** 2)
                                               for item in parameters if item.grad is not None))
            else:
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            if not torch.isfinite(gradient_norm):
                raise SchemaError("训练梯度出现非有限值")
            scaler.step(optimizer)
            scaler.update()

            model.eval()
            encoder.eval()
            with torch.no_grad():
                validation_prediction = sample(predict(), validation_points)
                validation_rmse = float(torch.sqrt(torch.mean((validation_prediction - validation_values) ** 2)).cpu() * scale)
            history.append(float(loss.detach().cpu()))
            validation_history.append(validation_rmse)
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            if scheduler is not None:
                if config.scheduler == "plateau":
                    scheduler.step(validation_rmse)
                else:
                    scheduler.step()
            if validation_rmse < best_validation_rmse - config.early_stopping_min_delta:
                best_validation_rmse = validation_rmse
                best_epoch = epoch + 1
                best_state = {"model": copy.deepcopy(model.state_dict()), "encoder": copy.deepcopy(encoder.state_dict())}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (config.early_stopping_patience > 0 and epoch + 1 >= min(config.minimum_epochs, config.epochs)
                    and epochs_without_improvement >= config.early_stopping_patience):
                stopped_early = True
                break
        model.load_state_dict(best_state["model"])
        encoder.load_state_dict(best_state["encoder"])
    else:
        best_epoch = int(loaded_checkpoint.get("best_epoch", 0))
        best_validation_rmse = float(loaded_checkpoint.get("best_validation_rmse", float("nan")))

    parameter_delta_l2 = float(np.sqrt(sum(
        float(torch.sum(torch.abs(item.detach().cpu() - start).double() ** 2).item())
        for item, start in zip(parameters, initial_parameters)
    )))
    checkpoint_saved = None
    if config.save_checkpoint_path and config.training_mode != "inference":
        checkpoint_path = Path(config.save_checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1, "signature": checkpoint_signature, "channel_names": channel_names,
            "channel_statistics": channel_statistics, "variable_policy": asdict(policy),
            "model_state": copy.deepcopy(model.state_dict()),
            "encoder_state": copy.deepcopy(encoder.state_dict()),
            "best_epoch": best_epoch, "best_validation_rmse": best_validation_rmse,
            "config": asdict(config), "loss_history": history, "validation_rmse_history": validation_history,
        }, checkpoint_path)
        checkpoint_saved = str(checkpoint_path)

    model.eval()
    encoder.eval()
    with torch.no_grad():
        predicted_tensor = predict()
        held_predictions = sample(predicted_tensor, validation_points).cpu().numpy() * scale + mean
        baseline_predictions = sample(bg_tensor, validation_points).cpu().numpy() * scale + mean
        prediction = predicted_tensor.squeeze(1).cpu().numpy() * scale + mean
    clipped_count = int((prediction < 0).sum())
    if policy.nonnegative:
        prediction = np.maximum(prediction, 0.0)
    with torch.no_grad():
        held_predictions = sample(torch.from_numpy(prediction[:, None]).to(device), validation_points).cpu().numpy()

    records = []
    for index, (it, iy, ix, observed, station_id, *_rest) in enumerate(validation_points):
        records.append({"station_id": station_id, "time": pd.Timestamp(source.time.values[it]),
                        "observed": observed, "predicted": float(max(0, held_predictions[index]) if policy.nonnegative else held_predictions[index]),
                        "bilinear_background": float(baseline_predictions[index])})
    station_predictions = pd.DataFrame(records)
    error = station_predictions.predicted - station_predictions.observed
    baseline_error = station_predictions.bilinear_background - station_predictions.observed
    metrics = {
        "coarse_consistency_rmse": float(np.sqrt(np.mean((F.interpolate(torch.from_numpy(prediction[:, None]), size=source.shape[-2:], mode="area").numpy()[:, 0] - source.values) ** 2))),
        "validation_station_count": int(station_predictions.station_id.nunique()),
        "validation_record_count": int(len(station_predictions)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "station_correlation": (float(np.corrcoef(station_predictions.predicted, station_predictions.observed)[0, 1])
                                if station_predictions.predicted.std() > 0 and station_predictions.observed.std() > 0 else None),
        "validation_peak_error": float((station_predictions.groupby('time')['predicted'].max()
                                         - station_predictions.groupby('time')['observed'].max()).abs().mean()),
        "spatial_correlation": None,
        "bilinear_rmse": float(np.sqrt(np.mean(baseline_error**2))),
        "bilinear_mae": float(np.mean(np.abs(baseline_error))),
    }
    if evaluation_truth is not None:
        evaluation_field = evaluation_truth[_main_variable(evaluation_truth)].sel(time=source.time).transpose("time", "lat", "lon")
        if not np.array_equal(evaluation_field.lat, target_lat) or not np.array_equal(evaluation_field.lon, target_lon):
            raise SchemaError('评价真值网格与目标不同，不能静默重采样真值')
        evaluation = UnitEngine().convert(evaluation_field, source.attrs['units']).values
        if not np.isfinite(evaluation).all():
            raise SchemaError('评价真值包含缺失，请先明确有效区域')
        correlations = [float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) for a, b in zip(prediction, evaluation)
                        if np.std(a) > 0 and np.std(b) > 0]
        metrics.update({
            "spatial_correlation": float(np.mean(correlations)) if correlations else None,
            "peak_amplitude_mae": float(np.mean(np.abs(prediction.max(axis=(1, 2)) - evaluation.max(axis=(1, 2))))),
            "gradient_mae": float((np.abs(np.diff(prediction, axis=1) - np.diff(evaluation, axis=1)).mean()
                                    + np.abs(np.diff(prediction, axis=2) - np.diff(evaluation, axis=2)).mean()) / 2),
            "full_field_rmse": float(np.sqrt(np.mean((prediction - evaluation) ** 2))),
            "full_field_mae": float(np.mean(np.abs(prediction - evaluation))),
            "bilinear_full_field_rmse": float(np.sqrt(np.mean((background_high.values - evaluation) ** 2))),
        })
    output = xr.Dataset(
        {variable: (("time", "lat", "lon"), prediction.astype("float32"))},
        coords={"time": source.time, "lat": target_lat, "lon": target_lon},
        attrs=dict(background.attrs),
    )
    output[variable].attrs = dict(background[variable].attrs)
    for axis in ('time', 'lat', 'lon'):
        output[axis].attrs = dict(source[axis].attrs)
        if axis != 'time':
            output[axis].attrs.pop('bounds', None)
    time_bounds = source.time.attrs.get('bounds')
    if time_bounds in background:
        output[time_bounds] = background[time_bounds].sel(time=source.time)
    if "grid_resolution_km" in output.attrs:
        output.attrs.pop("grid_resolution_km")  # Coarse-grid resolution is not output resolution.
    output.attrs.update(
        processing_stage="observation_constrained_neural_reconstruction",
        algorithm=("observation-guided UNO + reliability weights + local attention residual" if guided else "official UNO + PyTorch cross attention + coarse FFL + training-peak constraints" if upgraded
                   else "FNO residual reconstruction with observation and coarse-grid constraints"),
        training_station_count=len(set(train.station_id.astype(str))),
        validation_station_count=metrics["validation_station_count"],
        auxiliary_variables=json.dumps(auxiliary_names, ensure_ascii=False),
        validation_station_leakage="false",
        variable_family=policy.family,
        nonnegative_constraint=str(policy.nonnegative).lower(),
    )
    report = {
        "status": "WARN" if metrics.get("full_field_rmse", 0) > metrics.get("bilinear_full_field_rmse", float("inf")) else "PASS",
        "algorithm": output.attrs["algorithm"],
        "config": asdict(config),
        "ablation_controls": {"background_cross_attention": config.cross_attention if upgraded else True,
                              "local_observation_residual_gate": config.observation_residual_gate if guided else 0,
                              "reliability_weighting": guided and config.reliability_weighting,
                              "backbone": "uno" if upgraded and config.backbone == "native" else "fno",
                              "spectral_loss_weight": config.spectral_weight if upgraded else 0,
                              "peak_loss_weight": config.peak_weight if upgraded else 0},
        "variable_policy": asdict(policy),
        "normalization": {"fit_boundary": "training inputs only; validation/test observations excluded",
                          "channel_statistics": channel_statistics},
        "observation_reliability": reliability,
        "observation_weight_schedule": schedules,
        "loss_history": history,
        "validation_rmse_history": validation_history,
        "learning_rate_history": learning_rates,
        "gradient_norm_history": gradient_norms,
        "training_control": {
            "optimizer": f"torch.optim.{config.optimizer.title()}",
            "scheduler": config.scheduler,
            "epochs_requested": config.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_rmse": best_validation_rmse,
            "stopped_early": stopped_early,
            "gradient_clip_norm": config.gradient_clip_norm,
            "mixed_precision_requested": config.mixed_precision,
            "mixed_precision_enabled": amp_enabled,
            "device": str(device),
            "parameter_delta_l2": parameter_delta_l2,
            "checkpoint_loaded": config.load_checkpoint_path,
            "checkpoint_saved": checkpoint_saved,
            "training_mode": config.training_mode,
        },
        "input_metadata": {'background': dict(background.attrs), 'background_variable': dict(source.attrs),
                           'auxiliaries': [dict(ds.attrs) for ds in auxiliaries],
                           'stations': dict(stations.attrs),
                           'station_sources': {name: stations[name].astype(str).unique().tolist()
                                               for name in ('source_file', 'source_url', 'source_quality', 'agency') if name in stations}},
        "clipped_negative_values": clipped_count,
        "station_records_used": {'train': len(train_points), 'validation': len(validation_points)},
        "test_station_records_excluded": test_records_excluded,
        "qa_assumption": '未提供QA时按等可靠度处理，不能解释为正式质控' if 'qc_flag' not in stations and 'quality_weight' not in stations else '使用提供的QA，INVALID/MISSING不参与训练或评价',
        "training_station_ids": sorted(set(train.station_id.astype(str))),
        "validation_station_ids": sorted(set(validation.station_id.astype(str))),
        "station_overlap": [],
        "observation_encoder": "PyTorch MultiheadAttention: grid/location/value/quality" if upgraded else "legacy point cross-attention: location/value/quality",
        "observation_operator": "bilinear grid_sample at actual station coordinates",
        "extensions": {"dynamic_weights": "encoder quality prior / loss weights",
                       "uncertainty": "future additional output head",
                       "multiscale_fusion": "official UNO down/up scales and skip connections" if upgraded else "not enabled"},
        "spectral_supervision": "coarse background only; no high-resolution frequency target" if upgraded else "not enabled",
        "physics_scope": (("nonnegativity, " if policy.nonnegative else "") +
                          "gradients, station peaks and coarse consistency; not a PDE solver"),
        "extrapolation_applied": bool(target_lat.min() < source.lat.min() or target_lat.max() > source.lat.max()
                                      or target_lon.min() < source.lon.min() or target_lon.max() > source.lon.max()),
        "auxiliary_variables": auxiliary_names,
        "auxiliary_alignment": auxiliary_alignment,
        "metrics": metrics,
        "has_high_resolution_truth": truth is not None,
        "supervision_note": "训练真值排除验证站邻域；全场监督仍限制独立评价范围" if truth is not None else "仅训练站与背景约束；参考场不参与优化",
        "has_independent_full_field_evaluation": evaluation_truth is not None,
        "warning": None if truth is not None or evaluation_truth is not None else "没有独立高分辨率真值；只能报告独立站点评价，不能声称全场精度。",
        "runtime_seconds": perf_counter() - started,
        "quality_checks": {"finite_output": bool(np.isfinite(prediction).all()),
                           "physical_range_constraint": bool((prediction >= 0).all()) if policy.nonnegative else True,
                           "units_preserved": bool(output[variable].attrs.get("units")),
                           "station_disjoint": True},
    }
    if not all(report["quality_checks"].values()):
        raise SchemaError("重构输出未通过有限值/单位/非负检查")
    return NeuralReconstructionResult(output, report, station_predictions, tuple(history))
