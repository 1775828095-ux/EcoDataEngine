from __future__ import annotations

from dataclasses import fields
from importlib.util import find_spec
from time import perf_counter

import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist

from ecodata.algorithms.contracts import AlgorithmContext, AlgorithmResult, AlgorithmSpec, Compatibility
from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord


def _basic_compatibility(spec: AlgorithmSpec, context: AlgorithmContext) -> Compatibility:
    reasons = []
    if spec.status != "available":
        reasons.append(f"状态为 {spec.status}，尚未通过本项目执行验证")
    if context.semantic_type not in spec.semantic_types:
        reasons.append(f"不适用于 {context.semantic_type} 语义")
    if not context.sources:
        reasons.append("缺少输入数据")
    if spec.requires_observations and (context.observations is None or context.observations.empty):
        reasons.append("缺少训练观测")
    if spec.requires_auxiliary and not context.auxiliaries:
        reasons.append("缺少辅助变量")
    return Compatibility(not reasons, tuple(reasons))


class CatalogOnlyPlugin:
    def __init__(self, spec: AlgorithmSpec):
        self.spec = spec

    def compatibility(self, context):
        return _basic_compatibility(self.spec, context)

    def run(self, context):
        raise SchemaError(f"{self.spec.name} 尚未接入可执行后端，不能伪装为可用算法")


class RegridPlugin:
    def __init__(self, method: str, semantic_types: tuple[str, ...]):
        from importlib.metadata import version
        self.method = method
        self.spec = AlgorithmSpec(
            id=f"spatial.{method}", name=f"{method} regridding", version="1",
            category="single_source_scale", status="available", input_kinds=("rectilinear_grid",),
            semantic_types=semantic_types, physical_constraints=("coordinate_monotonicity",),
            compute="low", license="BSD-3-Clause", backend=f"xarray-regrid {version('xarray-regrid')}",
            source_url="https://github.com/xarray-contrib/xarray-regrid",
            input_contract=dict(source_arity=1),
        )

    def compatibility(self, context):
        result = _basic_compatibility(self.spec, context)
        reasons = list(result.reasons)
        if context.target_lat is None or context.target_lon is None:
            reasons.append("缺少目标网格")
        elif context.sources and context.variable not in context.sources[0].dataset:
            reasons.append("目标变量不存在")
        elif context.sources:
            from ecodata.spatial.regridding import GridRegridder, RegridConfig, normalize_grid_axes
            try:
                normalized, _ = normalize_grid_axes(context.sources[0].dataset)
                GridRegridder(RegridConfig(self.method, context.target_lat, context.target_lon, context.variable)).validate_input(EcoDataBundle(normalized))
            except (SchemaError, SemanticError) as exc:
                reasons.append(str(exc))
        return Compatibility(not reasons, tuple(reasons), result.warnings)

    def run(self, context):
        from ecodata.spatial.regridding import GridRegridder, RegridConfig
        transform = GridRegridder(RegridConfig(self.method, context.target_lat, context.target_lon, context.variable))
        result = transform.apply(context.sources[0])
        return AlgorithmResult(result.data, self.spec.id, {"backend": self.spec.backend, "status": "PASS"})


class ResidualFusionPlugin:
    spec = AlgorithmSpec(
        id="fusion.residual", name="Background plus observation residual fusion", version="1",
        category="multi_source_fusion", status="available", input_kinds=("grid", "station_table"),
        semantic_types=("continuous",), requires_observations=True,
        physical_constraints=("same_variable", "same_unit", "same_crs"), compute="low",
        license="project", backend="SciPy griddata", source_url="https://scipy.org/",
    )

    def compatibility(self, context):
        result = _basic_compatibility(self.spec, context)
        reasons = list(result.reasons)
        if len(context.sources) < 2:
            reasons.append("需要背景场和观测 EcoDataBundle")
        elif "variable" not in context.sources[1].dataset:
            reasons.append("第二来源必须为标准化观测表，不能把另一网格误当站点观测")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        from ecodata.fusion.residual import ResidualFieldFusion, ResidualFusionConfig
        result = ResidualFieldFusion(ResidualFusionConfig(context.variable, context.parameters.get("interpolation", "linear"),
            context.parameters.get("outside_hull", "nearest"))).apply(context.sources[0], context.sources[1])
        return AlgorithmResult(result.data, self.spec.id, {"status": "PASS", "backend": self.spec.backend})


class WeightedFusionPlugin:
    spec = AlgorithmSpec(
        id="fusion.uncertainty_weighted", name="Inverse-variance weighted fusion", version="1",
        category="multi_source_fusion", status="available", input_kinds=("aligned_grids",),
        semantic_types=("continuous",), physical_constraints=("same_variable", "positive_uncertainty"),
        compute="low", license="project", backend="xarray",
    )

    def compatibility(self, context):
        result = _basic_compatibility(self.spec, context)
        reasons = list(result.reasons)
        if len(context.sources) < 2:
            reasons.append("至少需要两份同变量网格")
        for source in context.sources:
            if context.parameters.get("uncertainty_variable", "uncertainty") not in source.dataset:
                reasons.append("缺少明确的不确定性变量，无法进行逆方差加权")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        from ecodata.fusion.weighted import UncertaintyWeightedFusion, WeightedFusionConfig
        result = UncertaintyWeightedFusion(WeightedFusionConfig(context.variable, context.parameters.get("uncertainty_variable", "uncertainty"))).apply(context.sources)
        return AlgorithmResult(result.data, self.spec.id, {"status": "PASS", "backend": self.spec.backend})


class IDWPlugin:
    spec = AlgorithmSpec(
        id="fusion.idw", name="Inverse-distance weighting", version="1", category="observation_interpolation",
        status="available", input_kinds=("station_table", "target_grid"), semantic_types=("continuous",),
        requires_observations=True, compute="low", license="project", backend="SciPy cdist",
    )

    def compatibility(self, context):
        result = _basic_compatibility(self.spec, context)
        reasons = list(result.reasons)
        required = {"longitude", "latitude", "value", "split"}
        if context.observations is not None and not required.issubset(context.observations):
            reasons.append("观测缺少 longitude/latitude/value/split")
        if not context.sources or not {"lat", "lon"}.issubset(context.sources[0].dataset.coords):
            reasons.append("缺少规则目标网格")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        from pyproj import CRS, Transformer
        started = perf_counter()
        target = context.sources[0].dataset
        if context.target_lat is not None and context.target_lon is not None:
            target = xr.Dataset(coords={"lat": list(context.target_lat), "lon": list(context.target_lon)},
                                attrs=dict(target.attrs))
        train = context.observations[context.observations["split"].eq("train")].dropna(subset=["longitude", "latitude", "value"])
        if len(train) < 1:
            raise SchemaError("IDW 没有有效训练观测")
        lon, lat = np.meshgrid(target.lon.values, target.lat.values)
        source = train[["longitude", "latitude"]].to_numpy(float)
        points = np.column_stack([lon.ravel(), lat.ravel()])
        source_crs = context.parameters.get("observation_crs") or target.attrs.get("crs")
        if not source_crs:
            raise SchemaError("IDW 需要明确观测和目标网格 CRS")
        crs = CRS.from_user_input(source_crs)
        metric_crs = crs
        if crs.is_geographic:
            metric_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={float(np.mean(lat))} +lon_0={float(np.mean(lon))} +datum=WGS84 +units=m")
        transformer = Transformer.from_crs(crs, metric_crs, always_xy=True)
        px, py = transformer.transform(points[:, 0], points[:, 1])
        sx, sy = transformer.transform(source[:, 0], source[:, 1])
        distance = cdist(np.column_stack([px, py]), np.column_stack([sx, sy]))
        from ecodata.fusion.point_weights import inverse_distance_weights
        weights = inverse_distance_weights(distance, float(context.parameters.get("power", 2.0)))
        values = weights @ train.value.to_numpy(float) / weights.sum(axis=1)
        output = target.copy(deep=True)
        output[context.variable] = (("lat", "lon"), values.reshape(lon.shape))
        unit = context.parameters.get("unit") or (str(train.unit.iloc[0]) if "unit" in train and train.unit.nunique() == 1 else None)
        if not unit:
            raise SchemaError("IDW 输出需要明确单位")
        output[context.variable].attrs.update(units=unit, data_kind="continuous", physical_variable=context.variable)
        output.attrs.update(processing_stage="fused", fusion_method="IDW")
        record = TransformRecord("fusion.idw", {"power": context.parameters.get("power", 2.0), "training_count": len(train), "distance_crs": metric_crs.to_string()}, len(train), output[context.variable].size, perf_counter() - started)
        return AlgorithmResult(EcoDataBundle(output, context.sources[0].history + (record,)), self.spec.id, {"status": "PASS", "training_count": len(train), "distance_crs": metric_crs.to_string()})


class QPPlugin:
    spec = AlgorithmSpec(
        id="downscale.qp", name="Constrained quadratic-program downscaling", version="P3.2",
        category="constrained_downscaling", status="available", input_kinds=("grid", "station_table"),
        semantic_types=("continuous",), requires_observations=True,
        physical_constraints=("nonnegative", "coarse_consistency", "optional_peak_constraint"),
        compute="medium", license="Apache-2.0/BSD dependencies", backend="CVXPY + OSQP",
        source_url="https://www.cvxpy.org/",
    )

    def compatibility(self, context):
        result = _basic_compatibility(self.spec, context)
        reasons = list(result.reasons)
        if context.target_lat is None or context.target_lon is None:
            reasons.append("缺少目标网格")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        from ecodata.optimization.downscaling import ConstrainedDownscaler, DownscalingConfig
        allowed = {item.name for item in fields(DownscalingConfig)} - {"target_lat", "target_lon", "variable"}
        options = {key: value for key, value in context.parameters.items() if key in allowed}
        result = ConstrainedDownscaler(DownscalingConfig(context.target_lat, context.target_lon, variable=context.variable, **options)).apply(context.sources[0], context.observations)
        return AlgorithmResult(result.data, self.spec.id, {"status": "PASS", "backend": self.spec.backend})


class TemporalSemanticPlugin:
    spec = AlgorithmSpec(
        id="temporal.semantic_resample", name="Semantic temporal resampling", version="1",
        category="temporal_scale", status="available", input_kinds=("time_grid", "time_table"),
        semantic_types=("continuous", "accumulation"), physical_constraints=("interval_semantics", "interval_conservation"),
        compute="low", license="project", backend="EcoData overlap-weighted temporal resampler",
        capabilities=("temporal_accumulation", "temporal_mean_resampling", "temporal_interpolation"),
        preconditions=("time_coordinate", "temporal_semantics", "target_time_frequency"),
        verification_evidence=("tests/test_p2_temporal.py", "tests/test_post_conversion_validation.py"),
        input_contract=dict(source_arity=1),
    )

    @staticmethod
    def _source(context):
        source=context.sources[0]
        field=source.dataset[context.variable]
        if field.attrs.get('temporal_semantics',source.dataset.attrs.get('temporal_semantics'))!='interval_mean':return source
        ds=source.dataset.copy(deep=False)
        ds[context.variable].attrs=dict(field.attrs,temporal_semantics='mean')
        return EcoDataBundle(ds,source.history)

    def compatibility(self, context):
        result = _basic_compatibility(self.spec, context)
        reasons = list(result.reasons)
        if not context.parameters.get("target_time_frequency"):
            reasons.append("缺少 target_time_frequency")
        if not context.sources or "time" not in context.sources[0].dataset.coords:
            reasons.append("缺少时间坐标")
        if not reasons:
            from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler
            try:SemanticTimeResampler(SemanticTimeConfig(context.parameters['target_time_frequency'],context.variable,context.parameters.get('temporal_interpolation','linear'))).validate_input(self._source(context))
            except (ValueError,KeyError,TypeError,SchemaError,SemanticError) as exc:reasons.append(str(exc))
        return Compatibility(not reasons, tuple(reasons), result.warnings)

    def run(self, context):
        from ecodata.temporal.semantic import SemanticTimeConfig, SemanticTimeResampler
        transformed = SemanticTimeResampler(SemanticTimeConfig(context.parameters["target_time_frequency"], context.variable,context.parameters.get('temporal_interpolation','linear'))).apply(self._source(context))
        original=context.sources[0].dataset[context.variable].attrs.get('temporal_semantics')
        if original=='interval_mean':transformed.data.dataset[context.variable].attrs['temporal_semantics']=original
        return AlgorithmResult(transformed.data, self.spec.id, {"status": "PASS", "backend": self.spec.backend,"semantic_alias":"interval_mean -> mean" if original=='interval_mean' else None})


class NeuralReconstructionPlugin:
    def __init__(self, backend: str):
        self.backend = backend
        installed = find_spec("torch") is not None and find_spec("neuralop") is not None
        self.spec = AlgorithmSpec(
            id=f"neural.{backend}", name=backend.replace("_", " ").upper(), version="existing",
            category="learned_reconstruction", status="available" if installed else "dependency_missing",
            input_kinds=("time_grid", "station_table"), semantic_types=("continuous",),
            requires_training=True, requires_observations=True, physical_constraints=("coarse_consistency", "variable_policy"),
            compute="high", license="MIT dependencies / project wrapper", backend="neuraloperator + PyTorch",
            source_url="https://github.com/neuraloperator/neuraloperator",
        )

    def compatibility(self, context):
        result = _basic_compatibility(self.spec, context)
        reasons = list(result.reasons)
        if context.validation_role not in {"validation", "independent_validation"}:
            reasons.append("神经重构需要明确验证集，最终测试集不能用于选型")
        return Compatibility(not reasons, tuple(reasons))

    def run(self, context):
        from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig, run_neural_reconstruction
        allowed = {item.name for item in fields(NeuralReconstructionConfig)} - {"backend"}
        options = {key: value for key, value in context.parameters.items() if key in allowed}
        result = run_neural_reconstruction(context.sources[0].dataset, context.observations, auxiliaries=context.auxiliaries,
                                           config=NeuralReconstructionConfig(backend=self.backend, **options))
        return AlgorithmResult(EcoDataBundle(result.dataset), self.spec.id, result.report, result.report.get("metrics", {}))


def builtin_plugins():
    from ecodata.algorithms.open_source_spatial import spatial_plugins
    from ecodata.algorithms.open_source_stations import station_plugins
    from ecodata.algorithms.open_source_bias import bias_plugins
    from ecodata.algorithms.galileo import GalileoFeaturePlugin
    from ecodata.algorithms.quality_aware import QualityAwareFusionPlugin
    from ecodata.algorithms.cross_scale import SemanticCrossScaleStackPlugin

    plugins = [
        RegridPlugin("nearest", ("continuous", "categorical", "integer_count")),
        RegridPlugin("bilinear", ("continuous",)),
        RegridPlugin("conservative", ("continuous", "integer_count", "cell_total", "flux", "accumulation")),
        IDWPlugin(), ResidualFusionPlugin(), WeightedFusionPlugin(), QPPlugin(), TemporalSemanticPlugin(),
        NeuralReconstructionPlugin("fno"), NeuralReconstructionPlugin("multiscale_uno"), NeuralReconstructionPlugin("observation_guided_uno"),
        GalileoFeaturePlugin(),
        QualityAwareFusionPlugin(),
        SemanticCrossScaleStackPlugin(),
    ]
    candidates = (
        AlgorithmSpec("super_resolution.zssr", "ZSSR", "official CVPR 2018", "single_source_self_supervised", "research_only", ("image",), ("continuous",), requires_training=True, compute="high", license="WIS/Yeda Academic Non-Commercial", backend="official standalone code", source_url="https://github.com/assafshocher/ZSSR", notes="Commercial use is not granted; official input path is PNG/image oriented and environmental NetCDF semantics are not integrated."),
        AlgorithmSpec("super_resolution.thera", "Thera", "TMLR 2025", "continuous_scale_representation", "gpu_required", ("image",), ("continuous",), requires_training=False, requires_weights=True, compute="gpu", license="Apache-2.0", backend="JAX official code", source_url="https://github.com/prs-eth/thera", notes="GPU_REQUIRED: official environment is Linux, Python 3.10 and NVIDIA GPU; no weights downloaded or inference performed.", capabilities=("arbitrary_scale_image_reconstruction",), input_contract={"training_domain": "natural RGB images", "supported_input": "image", "hardware": "NVIDIA GPU", "environmental_field_validated": False}),
        AlgorithmSpec("super_resolution.liif", "LIIF", "CVPR 2021", "implicit_representation", "research_only", ("image",), ("continuous",), requires_training=True, requires_weights=True, compute="gpu", license="BSD-3-Clause", backend="official PyTorch 1.6 code", source_url="https://github.com/yinboc/liif", notes="DIV2K image checkpoints are not valid environmental-field priors."),
        AlgorithmSpec("super_resolution.diin", "DIIN", "IJCAI 2025", "implicit_representation", "research_only", ("image",), ("continuous",), requires_training=True, requires_weights=True, compute="gpu", license="unverified", backend="official research repository", source_url="https://github.com/Song-1205/DIIN", notes="License and environmental compatibility require verification before redistribution or execution."),
        AlgorithmSpec("spatial.cubic", "Cubic interpolation", "planned", "single_source_scale", "research_only", ("rectilinear_grid",), ("continuous",), compute="low", license="not selected", notes="No verified project backend yet; bilinear is the current safe fallback."),
        AlgorithmSpec("downscale.topopyscale", "TopoPyScale", "8668a5bf6adb", "weather_downscaling", "python_version_blocked", ("climate_grid", "DEM"), ("continuous",), requires_auxiliary=True, license="MIT", backend="official TopoPyScale workflow", source_url="https://github.com/ArcticSnow/TopoPyScale", capabilities=("topography_assisted_climate_downscaling",), preconditions=("Python>=3.13 for checked revision", "pressure-level climate", "DEM", "time/CRS alignment"), notes="BLOCKED on current Python 3.12; local ERA5 single-level subset is insufficient."),
        AlgorithmSpec("downscale.granite_wxc", "Granite / Prithvi WxC", "a6c4f3165660", "weather_downscaling", "checkpoint_missing", ("MERRA2_task_tensor", "ECCC_task_tensor"), ("continuous",), requires_weights=True, compute="gpu", license="Apache-2.0 / Prithvi MIT", backend="official Granite WxC", source_url="https://github.com/IBM/granite-wxc", capabilities=("weather_downscaling",), preconditions=("task-specific weights", "normalization statistics", "official variable/channel schema"), notes="RESOURCE_BLOCKED: no task checkpoint loaded; no MERRA2-temperature official inference executed; not generic environmental SR."),
        AlgorithmSpec("super_resolution.sentinel5p", "Sentinel-5P Super-Resolution", "98ee2c8c357d", "sensor_specific_reconstruction", "license_unverified", ("TROPOMI_hyperspectral",), ("continuous",), requires_weights=True, compute="gpu", license="no explicit license found", backend="official supervised/self-supervised code", source_url="https://github.com/hyamomar/Sentinel-5P-Super-Resolution", capabilities=("sentinel5p_hyperspectral_super_resolution",), preconditions=("TROPOMI sensor data", "authorized code use", "compatible checkpoint"), notes="BLOCKED: checked tree contains no explicit license; no installation/inference."),
        AlgorithmSpec("fusion.primer", "PRIMER", "07166483591c", "precipitation_fusion", "license_unverified", ("precipitation_grid", "gauge"), ("continuous", "accumulation"), requires_weights=True, requires_observations=True, compute="gpu", license="no explicit license found", backend="official coordinate diffusion", source_url="https://github.com/sunmoumou1/PRIMER", capabilities=("multi_source_precipitation_fusion",), preconditions=("precipitation-specific checkpoint", "CUDA-compatible environment", "authorized code use"), notes="RESOURCE_BLOCKED / LICENSE_UNVERIFIED: authors report 2xA100 training environment, not a proven inference minimum; no checkpoint run."),
        AlgorithmSpec("fusion.precipitation_fusion", "PrecipitationFusion PDFN", "60bc16dac9e7", "precipitation_fusion", "checkpoint_missing", ("precipitation_grid", "gauge"), ("continuous", "accumulation"), requires_weights=True, compute="high", license="MIT", backend="official PDFN code", source_url="https://github.com/kgavahi/PrecipitationFusion", capabilities=("multi_source_precipitation_fusion",), preconditions=("trained PDFN weights", "matching time/space source products"), notes="BLOCKED: checked tree has no pretrained weight file or directly runnable inference sample; training not attempted."),
    )
    plugins.extend(CatalogOnlyPlugin(item) for item in candidates)
    plugins.extend(spatial_plugins())
    plugins.extend(station_plugins())
    plugins.extend(bias_plugins())
    from ecodata.algorithms.open_source_time import XclimStatisticsPlugin
    from ecodata.algorithms.fords import ForDsPressurePlugin
    plugins.extend((XclimStatisticsPlugin(), ForDsPressurePlugin()))
    from ecodata.algorithms.expansion_v2 import candidate_specs
    plugins.extend(CatalogOnlyPlugin(spec) for spec in candidate_specs())
    from ecodata.algorithms.dual_times_field import DualTimesFieldPlugin
    plugins.append(DualTimesFieldPlugin())
    from ecodata.algorithms.climatelearn import ClimateLearnTemperaturePlugin
    from ecodata.algorithms.fords_temperature import ForDsTemperaturePlugin
    plugins.extend((ClimateLearnTemperaturePlugin(), ForDsTemperaturePlugin()))
    from ecodata.algorithms.expansion_v5 import v5_plugins
    plugins.extend(v5_plugins())
    from ecodata.algorithms.soil_spline import SoilSplinePlugin
    plugins.append(SoilSplinePlugin())
    from ecodata.algorithms.gp_residual import GPResidualPlugin
    plugins.append(GPResidualPlugin())
    from ecodata.algorithms.station_calibration import StationAffineCalibrationPlugin
    plugins.append(StationAffineCalibrationPlugin())
    from ecodata.algorithms.station_feature_prediction import StationFeatureForest
    plugins.extend(StationFeatureForest(group) for group in ("background", "weather", "joint"))
    from ecodata.algorithms.raster_gap_prediction import plugins as raster_gap_plugins
    plugins.extend(raster_gap_plugins())
    from ecodata.algorithms.raster_complements import plugins as raster_complement_plugins
    plugins.extend(raster_complement_plugins())
    from ecodata.algorithms.advanced_complements import plugins as advanced_complement_plugins
    plugins.extend(advanced_complement_plugins())
    return tuple(plugins)

