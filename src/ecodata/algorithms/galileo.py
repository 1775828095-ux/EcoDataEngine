from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext, AlgorithmResult, AlgorithmSpec, Compatibility
from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle, TransformRecord


class GalileoFeaturePlugin:
    """Thin adapter around NASA Harvest's official Galileo single-file encoder."""

    spec = AlgorithmSpec(
        id="features.galileo_nano",
        name="Galileo Nano remote-sensing feature encoder",
        version="official nano",
        category="multi_source_feature_extraction",
        status="available",
        input_kinds=("aligned_grid",),
        semantic_types=("continuous", "categorical"),
        requires_weights=True,
        compute="medium",
        license="MIT",
        backend="official single_file_galileo.py + nano encoder.pt",
        source_url="https://github.com/nasaharvest/galileo",
        notes=(
            "Produces pretrained spatial features only. It is not a standalone fusion, "
            "downscaling, or environmental-variable prediction algorithm."
        ),
    )

    _supported_groups = {
        "SRTM": ("elevation", "slope"),
        "ERA5": ("temperature_2m", "total_precipitation_sum"),
        "S2_RGB": ("B2", "B3", "B4"),
    }

    @staticmethod
    def _default_repo() -> Path:
        return Path(__file__).resolve().parents[3] / "external" / "galileo"

    def _repo(self, context: AlgorithmContext) -> Path:
        return Path(context.parameters.get("official_repo_path", self._default_repo())).resolve()

    @staticmethod
    def _datasets(context: AlgorithmContext) -> tuple[xr.Dataset, ...]:
        return tuple(source.dataset for source in context.sources) + tuple(context.auxiliaries)

    def _group_sources(self, context: AlgorithmContext) -> dict[str, xr.Dataset]:
        result = {}
        for name, variables in self._supported_groups.items():
            matches = [dataset for dataset in self._datasets(context) if all(variable in dataset for variable in variables)]
            if matches:
                result[name] = matches[0]
        return result

    @staticmethod
    def _spatial_dims(dataset: xr.Dataset) -> tuple[str, str] | None:
        if {"y", "x"}.issubset(dataset.dims):
            return "y", "x"
        if {"lat", "lon"}.issubset(dataset.dims):
            return "lat", "lon"
        return None

    def compatibility(self, context: AlgorithmContext) -> Compatibility:
        reasons: list[str] = []
        warnings: list[str] = []
        if not context.sources:
            reasons.append("缺少输入数据")
            return Compatibility(False, tuple(reasons))
        if context.variable != "galileo_embedding":
            reasons.append("该插件只生成 galileo_embedding 特征，不能冒充目标环境变量预测器")
        groups = self._group_sources(context)
        if not groups:
            expected = "；".join(f"{name}={','.join(values)}" for name, values in self._supported_groups.items())
            reasons.append(f"没有完整的 Galileo 官方输入组；当前接入支持 {expected}")
        spatial_groups = {name: dataset for name, dataset in groups.items() if name in {"SRTM", "S2_RGB"}}
        if not spatial_groups:
            reasons.append("至少需要一个带 y/x 或 lat/lon 网格的 SRTM 或 S2_RGB 输入组")
        elif any(self._spatial_dims(dataset) is None for dataset in spatial_groups.values()):
            reasons.append("SRTM/S2_RGB 输入必须包含 y/x 或 lat/lon 二维网格")
        crs_values = {str(dataset.attrs.get("crs", "")) for dataset in spatial_groups.values()}
        if "" in crs_values:
            reasons.append("所有空间输入必须明确 CRS")
        elif len(crs_values) > 1:
            reasons.append("Galileo 特征提取前必须先统一 CRS；当前输入 CRS 不一致")
        repo = self._repo(context)
        for required in (
            repo / "single_file_galileo.py",
            repo / "data" / "models" / "nano" / "config.json",
            repo / "data" / "models" / "nano" / "encoder.pt",
            repo / "config" / "normalization.json",
        ):
            if not required.exists():
                reasons.append(f"缺少官方 Galileo 文件：{required}")
        try:
            import torch  # noqa: F401
            import einops  # noqa: F401
        except ImportError as exc:
            reasons.append(f"缺少运行依赖：{exc.name}")
        if groups:
            warnings.append("仅生成 Galileo 预训练融合特征；不会直接生成高分辨率环境场")
            if len(groups) > 1:
                warnings.append("不同空间网格会匹配到参考网格；网格加密不代表信息尺度提高")
        return Compatibility(not reasons, tuple(reasons), tuple(warnings))

    @staticmethod
    def _load_official_module(repo: Path):
        path = repo / "single_file_galileo.py"
        spec = importlib.util.spec_from_file_location("ecodata_official_galileo", path)
        if spec is None or spec.loader is None:
            raise SchemaError(f"无法加载 Galileo 官方实现：{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _as_spatial(dataset: xr.Dataset, variable: str, dims: tuple[str, str], target: xr.Dataset) -> tuple[np.ndarray, str]:
        data = dataset[variable]
        source_dims = GalileoFeaturePlugin._spatial_dims(dataset)
        if source_dims is None:
            raise SchemaError(f"{variable} 缺少空间网格")
        extra = [dim for dim in data.dims if dim not in source_dims]
        if extra:
            if extra == ["time"] and data.sizes["time"] == 1:
                data = data.isel(time=0)
            else:
                raise SchemaError(f"{variable} 必须是单时次二维网格；收到维度 {data.dims}")
        if source_dims != dims:
            data = data.rename(dict(zip(source_dims, dims)))
        for dim in dims:
            if np.all(np.diff(data[dim].values) < 0):
                data = data.sortby(dim)
        target_coords = {dim: target[dim] for dim in dims}
        same_grid = all(np.array_equal(data[dim].values, target[dim].values) for dim in dims)
        if not same_grid:
            data = data.interp(target_coords, method="linear")
        values = np.asarray(data.transpose(*dims).values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise SchemaError(f"{variable} 尺度匹配后出现 NaN；请检查空间范围是否真实重叠")
        return values, "none" if same_grid else "bilinear_feature_alignment"

    def run(self, context: AlgorithmContext) -> AlgorithmResult:
        compatibility = self.compatibility(context)
        if not compatibility.eligible:
            raise SchemaError("Galileo 不可执行：" + "；".join(compatibility.reasons))

        import torch

        started = perf_counter()
        source = context.sources[0]
        repo = self._repo(context)
        official = self._load_official_module(repo)
        groups = self._group_sources(context)
        spatial_groups = {name: dataset for name, dataset in groups.items() if name in {"SRTM", "S2_RGB"}}
        requested_reference = context.parameters.get("reference_group")
        if requested_reference:
            if requested_reference not in spatial_groups:
                raise SchemaError(f"reference_group={requested_reference} 不存在或不是空间输入组")
            reference_group = requested_reference
        else:
            reference_group = max(
                spatial_groups,
                key=lambda name: np.prod([spatial_groups[name].sizes[d] for d in self._spatial_dims(spatial_groups[name])]),
            )
        dataset = spatial_groups[reference_group]
        dims = self._spatial_dims(dataset)
        assert dims is not None
        height, width = dataset.sizes[dims[0]], dataset.sizes[dims[1]]
        patch_size = int(context.parameters.get("patch_size", 2))
        if patch_size < 1 or height % patch_size or width % patch_size:
            raise SchemaError(f"网格 {height}x{width} 必须能被 patch_size={patch_size} 整除")

        batch, timesteps = 1, 1
        s_t_x = np.zeros((batch, height, width, timesteps, len(official.SPACE_TIME_BANDS)), np.float32)
        sp_x = np.zeros((batch, height, width, len(official.SPACE_BANDS)), np.float32)
        t_x = np.zeros((batch, timesteps, len(official.TIME_BANDS)), np.float32)
        st_x = np.zeros((batch, len(official.STATIC_BANDS)), np.float32)
        s_t_m = np.ones((batch, height, width, timesteps, len(official.SPACE_TIME_BANDS_GROUPS_IDX)), np.float32)
        sp_m = np.ones((batch, height, width, len(official.SPACE_BAND_GROUPS_IDX)), np.float32)
        t_m = np.ones((batch, timesteps, len(official.TIME_BAND_GROUPS_IDX)), np.float32)
        st_m = np.ones((batch, len(official.STATIC_BAND_GROUPS_IDX)), np.float32)

        scale_matching = []
        if "SRTM" in groups:
            group_dataset = groups["SRTM"]
            for variable in self._supported_groups["SRTM"]:
                values, method = self._as_spatial(group_dataset, variable, dims, dataset)
                sp_x[0, :, :, official.SPACE_BANDS.index(variable)] = values
                scale_matching.append({"variable": variable, "method": method, "target_shape": [height, width]})
            sp_m[..., list(official.SPACE_BAND_GROUPS_IDX).index("SRTM")] = 0
        if "S2_RGB" in groups:
            group_dataset = groups["S2_RGB"]
            for variable in self._supported_groups["S2_RGB"]:
                values, method = self._as_spatial(group_dataset, variable, dims, dataset)
                s_t_x[0, :, :, 0, official.SPACE_TIME_BANDS.index(variable)] = values
                scale_matching.append({"variable": variable, "method": method, "target_shape": [height, width]})
            s_t_m[..., list(official.SPACE_TIME_BANDS_GROUPS_IDX).index("S2_RGB")] = 0
        if "ERA5" in groups:
            group_dataset = groups["ERA5"]
            for variable in self._supported_groups["ERA5"]:
                values = np.asarray(group_dataset[variable].values, dtype=np.float32).reshape(-1)
                if values.size != 1:
                    raise SchemaError("当前 ERA5 多组接入只接受单时次区域代表值；空间气象场不能静默压成标量")
                t_x[0, 0, official.TIME_BANDS.index(variable)] = values[0]
            t_m[..., list(official.TIME_BAND_GROUPS_IDX).index("ERA5")] = 0

        normalization = json.loads((repo / "config" / "normalization.json").read_text(encoding="utf-8"))
        for values in (s_t_x, sp_x, t_x, st_x):
            stats = normalization[str(values.shape[-1])]
            values -= np.asarray(stats["mean"], dtype=np.float32)
            values /= np.asarray(stats["std"], dtype=np.float32)

        tensors = [torch.from_numpy(value) for value in (s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)]
        months = torch.full((batch, timesteps), int(context.parameters.get("month", 5)), dtype=torch.long)
        device = torch.device("cpu")
        encoder = official.Encoder.load_from_folder(repo / "data" / "models" / "nano", device=device).to(device).eval()
        with torch.no_grad():
            outputs = encoder(*tensors, months, patch_size=patch_size,
                              input_resolution_m=int(context.parameters.get("input_resolution_m", 30)))
        feature_tokens = official.Encoder.apply_mask_and_average_tokens_per_patch(*outputs[:8])
        feature = feature_tokens[0].reshape(height // patch_size, width // patch_size, -1).cpu().numpy()

        output = xr.Dataset(
            {"galileo_embedding": (("feature_y", "feature_x", "embedding"), feature.astype(np.float32))},
            coords={
                "feature_y": np.arange(feature.shape[0]),
                "feature_x": np.arange(feature.shape[1]),
                "embedding": np.arange(feature.shape[2]),
            },
            attrs={
                "crs": dataset.attrs.get("crs", ""),
                "processing_stage": "feature_extraction",
                "algorithm": self.spec.id,
                "official_source": self.spec.source_url,
                "official_groups": ",".join(groups),
                "reference_group": reference_group,
                "not_a_resolution_claim": "Feature extraction does not prove improved spatial accuracy.",
            },
        )
        output["galileo_embedding"].attrs.update(units="1", long_name="Galileo pretrained feature embedding")
        elapsed = perf_counter() - started
        record = TransformRecord(
            self.spec.id,
            {"groups": list(groups), "reference_group": reference_group, "scale_matching": scale_matching,
             "patch_size": patch_size, "input_resolution_m": context.parameters.get("input_resolution_m", 30)},
            int(height * width),
            int(feature.shape[0] * feature.shape[1]),
            elapsed,
            compatibility.warnings,
        )
        report = {
            "status": "PASS",
            "backend": self.spec.backend,
            "groups": list(groups),
            "reference_group": reference_group,
            "scale_matching": scale_matching,
            "input_shape": [height, width],
            "feature_shape": list(feature.shape),
            "runtime_seconds": elapsed,
            "scope": "official pretrained feature extraction only",
        }
        return AlgorithmResult(EcoDataBundle(output, source.history + (record,)), self.spec.id, report)
