"""Thin GSTools adapter: conditioning and drift use training observations only."""
from dataclasses import dataclass
from importlib.metadata import version

import gstools as gs
import numpy as np
from ecodata.core.exceptions import SchemaError


@dataclass(frozen=True)
class KrigingOptions:
    minimum_samples: int = 8
    maximum_samples: int = 1000
    minimum_drift_cv_r2: float = .1
    chunk_size: int = 2048
    bounded_variogram_fit: bool = False


class SpatialFusionEngine:
    def __init__(self, options=None):
        self.options = options or KrigingOptions()

    @staticmethod
    def conditioned_field(krige, targets, *, seed=710, target_drift=None):
        """GSTools conditional simulation, not a forecast/assimilation ensemble."""
        conditional = gs.CondSRF(krige, seed=seed)
        return conditional(targets, ext_drift=target_drift, mesh_type="unstructured")

    def estimate(self, positions, values, targets, *, method="ordinary_kriging", drift=None, target_drift=None,
                 split=None, observation_error=None):
        positions, targets = np.asarray(positions, float), np.asarray(targets, float)
        values = np.asarray(values, float)
        if positions.shape != (2, len(values)) or targets.ndim != 2 or targets.shape[0] != 2:
            raise SchemaError("Kriging 需要二维坐标 (2,n)，且观测值数量一致")
        if split is None or not np.all(np.asarray(split) == "train"):
            raise SchemaError("Kriging 只接受显式 train 样本；验证站不能参与拟合")
        if len(values) < self.options.minimum_samples or not all(np.isfinite(a).all() for a in (positions, values, targets)):
            raise SchemaError("共同有效训练样本不足或坐标/数值存在缺失")
        if len(values) > self.options.maximum_samples:
            raise SchemaError(f"全局 GSTools Kriging 当前上限 {self.options.maximum_samples} 个训练站；请分区域处理，系统不会静默删站")
        if np.unique(positions, axis=1).shape[1] != len(values):
            raise SchemaError("重复训练站坐标需先按观测误差/时间语义显式合并")
        if np.linalg.matrix_rank(positions - positions.mean(axis=1, keepdims=True)) < 2:
            raise SchemaError("训练站共线，无法可靠约束二维空间场")
        if np.var(values) <= np.finfo(float).eps:
            raise SchemaError("训练目标变量没有变化，不能自动拟合可靠 variogram")
        fitting = values
        diagnostics = {"backend": "GSTools", "version": version("gstools"), "method": method,
                       "training_count": len(values), "fit_scope": "training_only", "auxiliary_added_to_target": False}
        drift_args = {}
        if method == "external_drift_kriging":
            if drift is None or target_drift is None:
                raise SchemaError("External Drift 需要训练站与全部目标位置的辅助变量共同样本")
            drift, target_drift = np.atleast_2d(drift).astype(float), np.atleast_2d(target_drift).astype(float)
            p = drift.shape[0]
            if drift.shape != (p, len(values)) or target_drift.shape != (p, targets.shape[1]):
                raise SchemaError("辅助变量的训练/目标空间覆盖不匹配")
            if not np.isfinite(drift).all() or not np.isfinite(target_drift).all():
                raise SchemaError("辅助变量覆盖不完整，禁止外推或强行使用 External Drift")
            if len(values) < max(self.options.minimum_samples, 3*(p+1)):
                raise SchemaError("辅助变量数量相对共同训练样本过多")
            scale = drift.std(axis=1, keepdims=True)
            if np.any(scale <= np.finfo(float).eps):
                raise SchemaError("辅助变量为常数，不能建立可靠关系")
            center = drift.mean(axis=1, keepdims=True)
            drift, target_drift = (drift-center)/scale, (target_drift-center)/scale
            design = np.column_stack([np.ones(len(values)), drift.T])
            if np.linalg.matrix_rank(design) != p+1 or np.linalg.cond(design) > 1e8:
                raise SchemaError("辅助变量之间共线/秩不足，不能可靠估计 External Drift")
            coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
            residuals = values-design@coefficients
            # Training-only leave-one-out linear relation diagnostic; never tune on validation.
            q, _ = np.linalg.qr(design, mode="reduced")
            leverage = np.sum(q*q, axis=1)
            if np.any(1-leverage < 1e-8):
                raise SchemaError("辅助关系受单一样本支配，不能可靠验证")
            cv_r2 = 1 - np.sum((residuals/(1-leverage))**2)/np.sum((values-values.mean())**2)
            if cv_r2 < self.options.minimum_drift_cv_r2:
                raise SchemaError(f"已识别辅助变量，当前无法可靠参与修正：训练集留一关系 R²={cv_r2:.3f}")
            fitting = residuals
            drift_args = {"ext_drift": drift}
            diagnostics.update(drift_cv_r2=float(cv_r2), drift_count=p, drift_center=center.ravel().tolist(),
                               drift_scale=scale.ravel().tolist(), drift_coefficients=coefficients.tolist())
        elif method == "universal_kriging":
            design = np.column_stack([np.ones(len(values)), positions.T])
            fitting = values-design@np.linalg.lstsq(design, values, rcond=None)[0]
            drift_args = {"drift_functions": "linear"}
        elif method != "ordinary_kriging":
            raise SchemaError(f"未知 Kriging 方法：{method}")
        error = np.zeros(len(values)) if observation_error is None else np.broadcast_to(observation_error, values.shape).astype(float)
        if not np.isfinite(error).all() or np.any(error < 0):
            raise SchemaError("observation_error 必须是明确单位下的有限非负方差")
        model = gs.Gaussian(dim=2, var=max(float(np.var(fitting)), 1e-12),
                            len_scale=max(float(np.ptp(positions, axis=1).max())/3, 1e-6))
        if np.var(fitting) > 1e-12:
            bins, gamma, counts = gs.vario_estimate(positions, fitting, return_counts=True)
            usable = (counts > 0) & np.isfinite(gamma)
            if np.count_nonzero(usable) < 3:
                raise SchemaError("有效距离分组不足，无法拟合 variogram")
            try:
                fit_options = {}
                if self.options.bounded_variogram_fit:
                    from scipy.spatial.distance import pdist
                    distances = pdist(positions.T)
                    model.set_arg_bounds(len_scale=[float(distances.min()) / 10, float(distances.max()) * 10])
                    fit_options['max_eval'] = 10000
                    diagnostics['fit_bounds'] = {'len_scale': [float(distances.min()) / 10, float(distances.max()) * 10],
                                                 'max_eval': 10000, 'scope': 'training distances only'}
                _, _, r2 = model.fit_variogram(bins[usable], gamma[usable], nugget=False, return_r2=True, **fit_options)
            except (ValueError, RuntimeError) as exc:
                raise SchemaError(f"GSTools variogram 拟合失败：{exc}") from exc
            diagnostics["variogram_fit_r2"] = float(r2) if np.isfinite(r2) else None
        else:
            diagnostics["variogram_note"] = "训练 drift 残差近零，无可拟合随机结构；使用数值正则协方差"
        classes = {"ordinary_kriging": gs.krige.Ordinary, "universal_kriging": gs.krige.Universal,
                   "external_drift_kriging": gs.krige.ExtDrift}
        krige = classes[method](model, positions, values, cond_err=error, **drift_args)
        field, variance = krige(targets, ext_drift=target_drift if drift_args.get("ext_drift") is not None else None,
                                chunk_size=self.options.chunk_size)
        diagnostics.update(variogram_model="Gaussian", variance=float(model.var), len_scale=float(model.len_scale),
                           nugget=float(model.nugget), observation_error_variance=error.tolist(),
                           observation_error_assumption="未提供测量方差时按无噪声条件观测处理" if observation_error is None else "使用调用方提供的测量方差",
                           conditioned_field="已保留 GSTools krige 对象，可调用 CondSRF；本阶段不生成随机同化集合")
        return field, variance, diagnostics, krige
