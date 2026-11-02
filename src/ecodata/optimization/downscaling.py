from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np
import pandas as pd
import xarray as xr

from ecodata.core.exceptions import SchemaError, SemanticError
from ecodata.core.schemas import EcoDataBundle, TransformRecord, TransformResult
from ecodata.optimization.operators import aggregation_operator, difference_operator, station_operator
from ecodata.spatial.regridding import GridRegridder, RegridConfig, normalize_grid_axes


SUPPORTED_QP_SOLVERS = ("OSQP", "CLARABEL", "SCS")


@dataclass(frozen=True)
class QPOptions:
    """User controls shared by the upload workflow and P3 pipeline."""

    initial_method: str = "bilinear"
    solver: str = "OSQP"
    alpha: float = 1.0
    beta: float = 20.0
    gamma: float = 0.8
    peak_threshold: float = 45.0
    observation_tolerance: float = 0.35
    soft_constraint_penalty: float = 80.0
    max_iter: int = 20_000
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5


@dataclass(frozen=True)
class DownscalingConfig:
    target_lat: tuple[float, ...]
    target_lon: tuple[float, ...]
    initial_method: str = "bilinear"
    solver: str = "OSQP"
    alpha: float = 1.0
    beta: float = 20.0
    gamma: float = 0.8
    peak_threshold: float = 45.0
    peak_observation_multiplier: float = 3.0
    peak_smoothing_factor: float = 0.12
    adaptive: bool = True
    adaptive_smoothing: bool = False
    trusted_peak_constraints: bool = False
    observation_tolerance: float = 0.35
    soft_constraint_penalty: float = 80.0
    adaptive_smoothing_strength: float = 0.10
    minimum_smoothing_weight: float = 0.08
    variable: str = "value"
    max_iter: int = 20_000
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5


class ConstrainedDownscaler:
    name = "downscale.peak_aware"

    def __init__(self, config: DownscalingConfig):
        self.config = config

    def validate_input(self, coarse: EcoDataBundle, observations: pd.DataFrame) -> None:
        config = self.config
        if isinstance(config.max_iter, bool) or not isinstance(config.max_iter, int) or config.max_iter <= 0:
            raise SchemaError("QP 最大迭代次数必须是正整数")
        positive = (config.alpha, config.soft_constraint_penalty, config.eps_abs, config.eps_rel)
        nonnegative = (config.beta, config.gamma, config.observation_tolerance, config.peak_threshold)
        if not all(np.isfinite(v) and v > 0 for v in positive) or not all(
            np.isfinite(v) and v >= 0 for v in nonnegative
        ):
            raise SchemaError("QP 权重、容差必须有限且非负；alpha、惩罚系数和求解精度必须大于 0")
        if config.initial_method not in {"nearest", "bilinear", "conservative"}:
            raise SchemaError("QP 初始场仅支持 nearest、bilinear、conservative")
        if config.solver not in SUPPORTED_QP_SOLVERS:
            raise SchemaError("QP 求解器仅支持 OSQP、CLARABEL、SCS")
        ds = coarse.dataset
        var = self.config.variable
        if var not in ds or not {"lat", "lon"}.issubset(ds[var].dims):
            raise SchemaError("Downscaling requires a value field on lat/lon dimensions")
        if ds[var].attrs.get("data_kind", ds.attrs.get("data_kind")) != "continuous":
            raise SemanticError("Constrained downscaling requires continuous data")
        required = {"time", "station_id", "latitude", "longitude", "variable", "value", "unit", "split"}
        missing = sorted(required - set(observations.columns))
        if missing:
            raise SchemaError(f"Observation table missing: {', '.join(missing)}")
        if set(observations["split"]) != {"train"}:
            raise SchemaError("Optimizer accepts training observations only")
        physical = ds[var].attrs.get("physical_variable")
        if set(observations["variable"]) != {physical}:
            raise SemanticError("Coarse field and observations represent different variables")
        if set(observations["unit"]) != {ds[var].attrs.get("units")}:
            raise SemanticError("Coarse field and observations use different units")
        if (self.config.adaptive_smoothing or self.config.trusted_peak_constraints) and "qc_flag" not in observations:
            raise SchemaError("Adaptive P3.2 methods require qc_flag on training observations")
        if self.config.observation_tolerance < 0 or self.config.soft_constraint_penalty <= 0:
            raise SchemaError("Peak tolerance must be non-negative and soft-constraint penalty positive")

    def apply(self, coarse: EcoDataBundle, observations: pd.DataFrame) -> TransformResult:
        self.validate_input(coarse, observations)
        started = perf_counter()
        ds = coarse.dataset
        var = self.config.variable
        coarse_data = ds[var]
        has_time = "time" in coarse_data.dims
        if not has_time:
            coarse_data = coarse_data.expand_dims(time=[pd.Timestamp(observations["time"].iloc[0])])
        times = pd.DatetimeIndex(coarse_data["time"].values)
        observations = observations.copy()
        observations["time"] = pd.to_datetime(observations["time"])

        factor_y = len(self.config.target_lat) / coarse_data.sizes["lat"]
        factor_x = len(self.config.target_lon) / coarse_data.sizes["lon"]
        if factor_y != factor_x or not float(factor_y).is_integer():
            raise SchemaError("Target grid must be an integer uniform refinement of the coarse grid")
        factor = int(factor_y)
        fine_shape = (len(self.config.target_lat), len(self.config.target_lon))
        a_matrix = aggregation_operator((coarse_data.sizes["lat"], coarse_data.sizes["lon"]), factor)
        e_matrix, edge_cells = difference_operator(fine_shape)

        station_meta = observations.drop_duplicates("station_id").sort_values("station_id")
        h_matrix, station_cells = station_operator(
            np.asarray(self.config.target_lat),
            np.asarray(self.config.target_lon),
            station_meta["latitude"].to_numpy(),
            station_meta["longitude"].to_numpy(),
        )
        station_ids = station_meta["station_id"].tolist()
        station_order = {station: index for index, station in enumerate(station_ids)}

        x = cp.Variable(fine_shape[0] * fine_shape[1])
        x0_p = cp.Parameter(x.size)
        y_p = cp.Parameter(a_matrix.shape[0])
        z_p = cp.Parameter(h_matrix.shape[0])
        obs_w_p = cp.Parameter(h_matrix.shape[0], nonneg=True)
        edge_w_p = cp.Parameter(e_matrix.shape[0], nonneg=True)
        tolerance_p = cp.Parameter(h_matrix.shape[0], nonneg=True)
        slack_w_p = cp.Parameter(h_matrix.shape[0], nonneg=True)
        peak_slack = cp.Variable(h_matrix.shape[0], nonneg=True)
        objective = cp.Minimize(
            self.config.alpha * cp.sum_squares(x - x0_p)
            + self.config.beta * cp.sum_squares(cp.multiply(obs_w_p, h_matrix @ x - z_p))
            + self.config.gamma * cp.sum_squares(cp.multiply(edge_w_p, e_matrix @ x))
            + self.config.soft_constraint_penalty * cp.sum(cp.multiply(slack_w_p, peak_slack))
        )
        problem = cp.Problem(
            objective,
            [
                a_matrix @ x == y_p,
                x >= 0,
                h_matrix @ x - z_p <= tolerance_p + peak_slack,
                z_p - h_matrix @ x <= tolerance_p + peak_slack,
            ],
        )

        outputs: list[np.ndarray] = []
        statuses: list[str] = []
        peak_counts: list[int] = []
        trusted_peak_counts: list[int] = []
        slack_maxima: list[float] = []
        smooth_weight_ranges: list[tuple[float, float]] = []
        solver_iterations: list[int] = []
        consistency_errors: list[float] = []
        nonnegative_violations: list[float] = []
        for time in times:
            coarse_slice = coarse_data.sel(time=time)
            initial = supported_regrid(
                EcoDataBundle(coarse_slice.to_dataset(name=var)),
                self.config.initial_method,
                self.config.target_lat,
                self.config.target_lon,
                var,
            ).dataset[var].values
            time_obs = observations.loc[observations["time"].eq(time)].copy()
            if set(time_obs["station_id"]) != set(station_ids):
                raise SchemaError(f"Training observations are incomplete at {time}")
            time_obs["_order"] = time_obs["station_id"].map(station_order)
            time_obs = time_obs.sort_values("_order")
            z = time_obs["value"].to_numpy(dtype=float)
            peak_mask = z >= self.config.peak_threshold if self.config.adaptive else np.zeros_like(z, dtype=bool)
            obs_weights = np.ones_like(z)
            obs_weights[peak_mask] = self.config.peak_observation_multiplier
            edge_weights = np.ones(e_matrix.shape[0])
            if peak_mask.any():
                peak_cells = set(station_cells[peak_mask].tolist())
                affected = np.array([left in peak_cells or right in peak_cells for left, right in edge_cells])
                edge_weights[affected] = self.config.peak_smoothing_factor
            trusted_mask = np.zeros_like(z, dtype=bool)
            if self.config.adaptive_smoothing or self.config.trusted_peak_constraints:
                trusted = time_obs["qc_flag"].astype(str).str.upper().eq("VALID").to_numpy()
                station_initial = np.asarray(h_matrix @ initial.ravel()).ravel()
                residual = z - station_initial
                edge_weights = adaptive_edge_weights(
                    initial,
                    edge_cells,
                    station_cells,
                    residual,
                    trusted.astype(float),
                    strength=self.config.adaptive_smoothing_strength,
                    minimum=self.config.minimum_smoothing_weight,
                )
                trusted_mask = trusted & (z >= self.config.peak_threshold) & (residual > 0)
            trusted_peak_counts.append(int(trusted_mask.sum()))

            x0_p.value = initial.ravel()
            y_p.value = coarse_slice.values.ravel()
            z_p.value = z
            obs_w_p.value = obs_weights
            edge_w_p.value = edge_weights
            tolerance_p.value = np.where(
                trusted_mask & self.config.trusted_peak_constraints,
                self.config.observation_tolerance,
                1.0e6,
            )
            slack_w_p.value = (trusted_mask & self.config.trusted_peak_constraints).astype(float)
            solver_settings = {
                "OSQP": {"eps_abs": self.config.eps_abs, "eps_rel": self.config.eps_rel, "max_iter": self.config.max_iter},
                "SCS": {"eps_abs": self.config.eps_abs, "eps_rel": self.config.eps_rel, "max_iters": self.config.max_iter},
                "CLARABEL": {
                    "tol_gap_abs": self.config.eps_abs, "tol_gap_rel": self.config.eps_rel,
                    "tol_feas": self.config.eps_abs, "max_iter": self.config.max_iter,
                },
            }[self.config.solver]
            try:
                problem.solve(
                    solver=self.config.solver,
                    warm_start=True,
                    verbose=False,
                    ignore_dpp=True,
                    **solver_settings,
                )
            except cp.error.SolverError as exc:
                raise SchemaError(f"{self.config.solver} 求解失败（时间 {time}）；请检查数据或增加最大迭代次数") from exc
            statuses.append(str(problem.status))
            peak_counts.append(int(peak_mask.sum()))
            if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                raise SchemaError(f"{self.config.solver} 未收敛（时间 {time}，状态 {problem.status}）；请检查数据或增加最大迭代次数")
            solver_iterations.append(int(problem.solver_stats.num_iters or 0))
            consistency_errors.append(float(np.max(np.abs(a_matrix @ x.value - y_p.value))))
            nonnegative_violations.append(float(max(0.0, -np.min(x.value))))
            outputs.append(np.asarray(x.value).reshape(fine_shape))
            active_slack = np.asarray(peak_slack.value)[trusted_mask] if trusted_mask.any() else np.asarray([])
            slack_maxima.append(float(active_slack.max()) if active_slack.size else 0.0)
            smooth_weight_ranges.append((float(edge_weights.min()), float(edge_weights.max())))

        result_values = np.stack(outputs)
        output = xr.Dataset(
            {var: (("time", "lat", "lon"), result_values)},
            coords={
                "time": times,
                "lat": np.asarray(self.config.target_lat),
                "lon": np.asarray(self.config.target_lon),
            },
        )
        if not has_time:
            output = output.isel(time=0, drop=True)
        for coordinate in output.coords:
            if coordinate in ds.coords:
                output[coordinate].attrs.update(ds[coordinate].attrs)
        output.attrs.update(ds.attrs)
        if self.config.trusted_peak_constraints:
            method = "adaptive_peak_constraint"
        elif self.config.adaptive_smoothing:
            method = "adaptive_smoothing"
        else:
            method = "peak_aware" if self.config.adaptive else "constrained_baseline"
        output.attrs.update(
            {
                "processing_stage": "constrained_downscaled",
                "downscaling_method": method,
                "solver": self.config.solver,
                "solver_statuses": ",".join(sorted(set(statuses))),
                "training_station_count": len(station_ids),
                "validation_data_used": "false",
                "weights_determined_before_optimization": "true",
                "peak_constraint_type": "explicit_soft" if self.config.trusted_peak_constraints else "none",
                "peak_constraint_max_slack": max(slack_maxima, default=0.0),
                "coarse_consistency_max_abs_error": max(consistency_errors),
                "nonnegativity_max_violation": max(nonnegative_violations),
            }
        )
        output[var].attrs.update(ds[var].attrs)
        record = TransformRecord(
            method=f"downscale.{method}",
            parameters={
                "initial_method": self.config.initial_method,
                "solver": self.config.solver,
                "max_iter": self.config.max_iter,
                "eps_abs": self.config.eps_abs,
                "eps_rel": self.config.eps_rel,
                "solver_iterations": solver_iterations,
                "solver_statuses": statuses,
                "coarse_consistency_max_abs_error": max(consistency_errors),
                "nonnegativity_max_violation": max(nonnegative_violations),
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma,
                "peak_threshold": self.config.peak_threshold,
                "adaptive": self.config.adaptive,
                "peak_observation_counts": peak_counts,
                "trusted_peak_counts": trusted_peak_counts,
                "adaptive_smoothing": self.config.adaptive_smoothing,
                "trusted_peak_constraints": self.config.trusted_peak_constraints,
                "observation_tolerance": self.config.observation_tolerance,
                "soft_constraint_penalty": self.config.soft_constraint_penalty,
                "peak_constraint_max_slack": max(slack_maxima, default=0.0),
                "smoothing_weight_min": min((item[0] for item in smooth_weight_ranges), default=1.0),
                "smoothing_weight_max": max((item[1] for item in smooth_weight_ranges), default=1.0),
            },
            input_rows=int(ds[var].size + len(observations)),
            output_rows=int(output[var].size),
            elapsed_seconds=perf_counter() - started,
            warnings=(f"{self.config.solver} 返回 optimal_inaccurate，请检查约束误差或提高求解精度。",)
            if cp.OPTIMAL_INACCURATE in statuses else (),
        )
        bundle = EcoDataBundle(output, coarse.history + (record,))
        return TransformResult(bundle, record)


def supported_regrid(
    coarse: EcoDataBundle,
    method: str,
    target_lat: tuple[float, ...],
    target_lon: tuple[float, ...],
    variable: str = "value",
) -> EcoDataBundle:
    """Pad one edge cell so target centers spanning full coarse cells are supported."""
    dataset, _ = normalize_grid_axes(coarse.dataset)
    source = dataset[variable].transpose(..., "lat", "lon")
    lat = dataset["lat"].values
    lon = dataset["lon"].values
    if len(lat) < 2 or len(lon) < 2:
        raise SchemaError("P3 regridding requires at least two coarse cells per axis")
    padded_values = np.pad(
        source.values,
        [(0, 0)] * (source.ndim - 2) + [(1, 1), (1, 1)],
        mode="edge",
    )
    padded_lat = np.concatenate([[lat[0] - (lat[1] - lat[0])], lat, [lat[-1] + (lat[-1] - lat[-2])]])
    padded_lon = np.concatenate([[lon[0] - (lon[1] - lon[0])], lon, [lon[-1] + (lon[-1] - lon[-2])]])
    coordinates = {dim: source.coords[dim] for dim in source.dims[:-2]}
    coordinates.update({"lat": padded_lat, "lon": padded_lon})
    padded = xr.Dataset(
        {variable: (source.dims, padded_values)},
        coords=coordinates,
        attrs=dataset.attrs,
    )
    padded[variable].attrs.update(source.attrs)
    for axis in ("lat", "lon"):
        padded[axis].attrs.update(dataset[axis].attrs)
    result = GridRegridder(RegridConfig(method, target_lat, target_lon, variable=variable)).apply(
        EcoDataBundle(padded, coarse.history)
    ).data
    result.dataset.attrs["boundary_support"] = "one nearest-value ghost cell; original coarse data unchanged"
    return result


def adaptive_edge_weights(
    initial: np.ndarray,
    edge_cells: np.ndarray,
    station_cells: np.ndarray,
    station_residuals: np.ndarray,
    confidence: np.ndarray,
    *,
    strength: float,
    minimum: float,
) -> np.ndarray:
    """Compute fixed convex smoothing weights from training-side information only."""
    flat = np.asarray(initial, dtype=float).ravel()
    edge_gradient = np.abs(flat[edge_cells[:, 0]] - flat[edge_cells[:, 1]])
    gradient_scale = max(float(np.quantile(edge_gradient, 0.9)), 1.0e-12)
    gradient_score = np.clip(edge_gradient / gradient_scale, 0.0, 1.0)
    positive = np.maximum(np.asarray(station_residuals, dtype=float), 0.0)
    residual_scale = max(float(np.quantile(positive, 0.9)), 1.0)
    station_score = np.clip(positive / residual_scale, 0.0, 1.0) * np.clip(confidence, 0.0, 1.0)
    cell_score = np.zeros(flat.size, dtype=float)
    np.maximum.at(cell_score, np.asarray(station_cells, dtype=int), station_score)
    peak_score = np.maximum(cell_score[edge_cells[:, 0]], cell_score[edge_cells[:, 1]])
    weights = (1.0 - 0.45 * gradient_score) * (1.0 - strength * peak_score)
    return np.clip(weights, minimum, 1.0)
