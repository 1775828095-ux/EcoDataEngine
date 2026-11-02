"""Small, explicit MODFLOW 6 structured-grid package writer backed by FloPy."""
from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.standardized import ModelAdapter
from ecodata.models.contracts import ScenarioResult


REQUIRED_FIELDS = (
    "top", "bottom", "hydraulic_conductivity", "initial_head",
    "recharge", "constant_head",
)
REQUIRED_ATTRS = ("delr", "delc", "length_units", "time_units", "period_length", "nstp")


class MODFLOW6FloPyAdapter(ModelAdapter):
    """Append a FloPy-authored MODFLOW 6 text package after standardisation."""

    supports_partial = False

    def execute(self, schema, inputs, assignments, analysis, plan, **options):
        prepared = super().execute(schema, inputs, assignments, analysis, plan, **options)
        if not prepared.quality["passed"] or not prepared.output.data_vars:
            return prepared
        try:
            native_files, native_check = build_modflow6_package(prepared.output)
        except (SchemaError, ValueError, TypeError) as exc:
            quality = dict(prepared.quality)
            quality.update(passed=False, native_format={"verified": False, "reason": str(exc)})
            updated_plan = dict(prepared.plan)
            updated_plan.setdefault("failed_inputs", []).append({"role": "modflow6_native_package", "reason": str(exc)})
            return ScenarioResult(prepared.output, prepared.fused, prepared.files, prepared.analysis, updated_plan, quality)

        files = {**prepared.files, **native_files}
        quality = dict(prepared.quality)
        quality["native_format"] = native_check
        quality["completion"] = {
            "standardized_preprocessing": True,
            "native_format_output": True,
            "official_software_read": False,
            "official_model_run": False,
        }
        updated_plan = dict(prepared.plan)
        updated_plan["native_output_files"] = sorted(native_files)
        return ScenarioResult(prepared.output, prepared.fused, files, prepared.analysis, updated_plan, quality)

    def native_validation(self, result, schema):
        check = result.quality.get("native_format", {})
        return {
            "verified": False,
            "reason": "FloPy已回读原生文本结构，但尚未由官方mf6可执行程序读取和运行",
            "native_format_output": bool(check.get("verified")),
            "flopy_reload": bool(check.get("verified")),
        }


def build_modflow6_package(dataset):
    """Write and reload one-layer GWF input; all physical values must be supplied."""
    missing_fields = [name for name in REQUIRED_FIELDS if name not in dataset]
    missing_attrs = [name for name in REQUIRED_ATTRS if dataset.attrs.get(name) in (None, "")]
    if missing_fields or missing_attrs:
        details = []
        if missing_fields:
            details.append("缺少字段：" + ", ".join(missing_fields))
        if missing_attrs:
            details.append("缺少网格/时间参数：" + ", ".join(missing_attrs))
        raise SchemaError("；".join(details))

    arrays = {name: _grid(dataset[name].values, name) for name in REQUIRED_FIELDS}
    shape = arrays["top"].shape
    if any(value.shape != shape for value in arrays.values()):
        raise SchemaError("MODFLOW 6字段必须位于相同的二维结构网格")
    if np.any(arrays["bottom"] >= arrays["top"]):
        raise SchemaError("bottom 必须严格低于 top")
    if np.any(arrays["hydraulic_conductivity"] <= 0):
        raise SchemaError("hydraulic_conductivity 必须为正值")
    if np.any(arrays["recharge"] < 0):
        raise SchemaError("recharge 不能为负值")
    boundary = np.argwhere(np.isfinite(arrays["constant_head"]))
    if not len(boundary):
        raise SchemaError("constant_head 没有任何有效边界单元；平台不会编造边界条件")

    delr, delc = float(dataset.attrs["delr"]), float(dataset.attrs["delc"])
    period_length, nstp = float(dataset.attrs["period_length"]), int(dataset.attrs["nstp"])
    if min(delr, delc, period_length, nstp) <= 0:
        raise SchemaError("delr、delc、period_length 和 nstp 必须为正值")

    try:
        import flopy
    except ImportError as exc:
        raise SchemaError("缺少 FloPy；请安装项目固定依赖后重试") from exc

    with TemporaryDirectory(prefix="ecodata-mf6-") as temp:
        workspace = Path(temp)
        sim = flopy.mf6.MFSimulation(sim_name="ecodata", sim_ws=workspace)
        flopy.mf6.ModflowTdis(sim, time_units=str(dataset.attrs["time_units"]),
                             nper=1, perioddata=[(period_length, nstp, 1.0)])
        flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
        gwf = flopy.mf6.ModflowGwf(sim, modelname="groundwater", save_flows=True)
        nrow, ncol = shape
        flopy.mf6.ModflowGwfdis(
            gwf, length_units=str(dataset.attrs["length_units"]), nlay=1,
            nrow=nrow, ncol=ncol, delr=delr, delc=delc,
            top=arrays["top"], botm=arrays["bottom"][None, :, :],
        )
        flopy.mf6.ModflowGwfic(gwf, strt=arrays["initial_head"][None, :, :])
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=arrays["hydraulic_conductivity"][None, :, :])
        flopy.mf6.ModflowGwfrcha(gwf, recharge=arrays["recharge"])
        stress = [((0, int(row), int(col)), float(arrays["constant_head"][row, col])) for row, col in boundary]
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: stress})
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord="groundwater.hds", budget_filerecord="groundwater.cbc",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )
        sim.write_simulation(silent=True)

        # FloPy's independent read path checks the complete name/package reference graph.
        loaded = flopy.mf6.MFSimulation.load(sim_name="ecodata", sim_ws=workspace, verbosity_level=0)
        model = loaded.get_model("groundwater")
        if model is None or tuple(model.modelgrid.shape) != (1, nrow, ncol):
            raise SchemaError("FloPy回读后的MODFLOW 6网格与输入不一致")
        np.testing.assert_allclose(np.asarray(model.npf.k.array)[0], arrays["hydraulic_conductivity"])
        np.testing.assert_allclose(np.asarray(model.ic.strt.array)[0], arrays["initial_head"])

        files = {path.name: path.read_bytes() for path in workspace.iterdir() if path.is_file()}
        if not files or any(not payload for payload in files.values()):
            raise SchemaError("FloPy生成了空文件，禁止输出")
        manifest = {
            "generator": "FloPy", "flopy_version": flopy.__version__,
            "target": "MODFLOW 6 block-structured text input",
            "grid": {"nlay": 1, "nrow": nrow, "ncol": ncol, "delr": delr, "delc": delc},
            "boundary_cells": len(stress), "files": sorted(files),
            "validation": {
                "standardized_preprocessing": True,
                "native_format_output": True,
                "flopy_reload": True,
                "official_software_read": False,
                "official_model_run": False,
            },
        }
        files["modflow6_manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        return files, {
            "verified": True, "reader": f"FloPy {flopy.__version__}",
            "scope": "MODFLOW 6 input files reload and array comparison",
            "official_software_read": False, "official_model_run": False,
        }


def _grid(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise SchemaError(f"{name} 必须是二维规则网格")
    if name == "constant_head":
        if np.isinf(array).any():
            raise SchemaError("constant_head 不能包含无穷值")
    elif not np.isfinite(array).all():
        raise SchemaError(f"{name} 含缺失或无穷值")
    return array
