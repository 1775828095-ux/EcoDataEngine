"""Acyclic, named-port cooperation over existing executable registry kernels."""
from dataclasses import replace

import numpy as np
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmResult
from ecodata.algorithms.dag import Artifact, DAGError, NodeAdapter, Port
from ecodata.algorithms.station_calibration import background_stations
from ecodata.core.schemas import EcoDataBundle
from ecodata.fusion.residual import residual_difference, residual_addition
from ecodata.algorithms.golem_generation import TypedGolemFactory


class CollaborationFactory(TypedGolemFactory):
    """Both addition branches must refer to the same computed background."""
    def parameters(self, key):
        if key == ("fusion.residual", "add_background"):
            yield (("mode", "add_background"),)
        else:
            yield from super().parameters(key)

    @staticmethod
    def coherent(recipe):
        operation, parameters, edges = recipe
        children = {name: child for name, output, child in edges}
        if dict(parameters).get("mode") == "residual_correction":
            background_operation = children["field"][0]
            # A second correction of a station interpolant or an already
            # corrected field is not a reviewed independent background role.
            if background_operation.startswith("station.") or background_operation in {"fusion.idw", "fusion.gp_residual", "fusion.residual"}:
                return False
        if operation == "fusion.residual" and dict(parameters).get("mode") == "add_background":
            residual_edges = {name: child for name, output, child in children["correction"][2]}
            if children["field"] != residual_edges["field"]:
                return False
        return all(CollaborationFactory.coherent(child) for child in children.values())

    def recipes(self, kind, stack=()):
        generated = super().recipes(kind, stack)
        if getattr(self,"reverse_limit",None) and len(generated)>=self.reverse_limit:
            # Filtering incoherent owners must not hide an earlier enumeration cap.
            self.collaboration_truncated = True
        return [(out, recipe) for out, recipe in generated if self.coherent(recipe)]

    def inspect(self, graph, *, complete=False):
        report = super().inspect(graph, complete=complete)
        for node in graph.nodes:
            if node.name != "fusion.residual" or node.parameters.get("mode") != "add_background":
                continue
            bindings = node.parameters.get("_port_bindings", {})
            if not {"field", "correction"} <= set(bindings):
                continue
            background = node.nodes_from[bindings["field"][0]]
            correction = node.nodes_from[bindings["correction"][0]]
            owner = correction.parameters.get("_port_bindings", {}).get("field")
            if owner is not None and correction.nodes_from[owner[0]] is not background:
                raise DAGError("Residual branches must share the same background node")
        return report


def source_calibration(registry, context, request, adapters, inputs):
    """Offer a genuine per-source choice: skip or calibrate; no self cycles."""
    if request["task"] != "multisource" or not request.get("allow_training"):
        return []
    plugin = registry.get("bias.station_affine")
    if "observations" not in inputs:
        return []
    added = []
    for index, source in enumerate(context.sources):
        local = replace(context, sources=(source,), auxiliaries=())
        if not plugin.compatibility(local).eligible:
            continue
        key = "source" + str(index)
        original = inputs[key]
        choice = Port(original.kind + "_calibration_choice", ("contract",))
        def skip(values, params, external, kind=choice.kind):
            return {"value": Artifact(kind, values["source"].value, dict(values["source"].metadata))}
        adapters["prepare.source_pass." + key, "run"] = NodeAdapter(
            {"source": original}, {"value": choice}, skip,
            decision_contract=dict(operation="validate", input="source", structural_only=True))
        def apply(values, params, external, kind=choice.kind):
            actual = replace(context, sources=(values["source"].value,), auxiliaries=(),
                             observations=values["observations"].value)
            check = plugin.compatibility(actual)
            if not check.eligible:
                raise DAGError("Actual calibration input rejected: " + str(check.reasons))
            result = plugin.run(actual)
            return {"value": Artifact(kind, result.data, dict(contract=values["source"].metadata["contract"],
                algorithm_report=result.report, source_fingerprint=values["source"].fingerprint()))}
        adapters[plugin.spec.id, key] = NodeAdapter({"source": original, "observations": inputs["observations"]},
            {"value": choice}, apply, decision_contract=dict(operation="compute", registered=False,
                algorithm_id=plugin.spec.id, capabilities=list(plugin.spec.capabilities), input="source"))
        inputs[key] = choice
        added.append(plugin.spec.id)
    return added


def residual_stages(registry, context, request, adapters, inputs):
    """Expose alternative residual estimators and background addition separately."""
    if request["task"] != "multisource" or "observations" not in inputs:
        return []
    local = replace(context, sources=context.sources[:1], auxiliaries=())
    try:
        background_stations(local)
    except (ValueError, KeyError, TypeError, IndexError):
        return []
    raw = Port("platform_raw_product", ("contract",))
    correction = Port("platform_residual_correction", ("contract", "background_fingerprint"))
    corrected = Port("platform_corrected_product", ("contract",))
    identifiers = []
    for identifier in ("fusion.gp_residual", "station.pykrige_ordinary", "station.metpy_linear"):
        plugin = registry.get(identifier)
        if not plugin.compatibility(local).eligible:
            continue
        def estimate(values, params, external, identifier=identifier):
            upstream = values["field"].value
            actual = replace(context, sources=(upstream.data,), auxiliaries=(), observations=values["observations"].value)
            plugin = registry.get(identifier)
            if identifier == "fusion.gp_residual":
                result = plugin.run(actual)
                ds = result.data.dataset.copy()
                ds[context.variable] = ds.residual_correction.copy()
                ds[context.variable].attrs = dict(upstream.data.dataset[context.variable].attrs)
                result = replace(result, data=EcoDataBundle(ds, result.data.history))
            else:
                _, train, sampled = background_stations(actual)
                residuals = train.copy()
                residuals["value"] = residual_difference(train.value.to_numpy(float), sampled)
                actual = replace(actual, observations=residuals)
                check = plugin.compatibility(actual)
                if not check.eligible:
                    raise DAGError("Actual residual estimator rejected: " + str(check.reasons))
                result = plugin.run(actual)
            if not np.isfinite(result.data.dataset[context.variable]).all():
                raise DAGError("Residual correction incomplete; no common-cohort deletion")
            fingerprint = Artifact("background", upstream.data).fingerprint()
            return {"correction": Artifact(correction.kind, result, dict(contract=values["field"].metadata["contract"],
                background_fingerprint=fingerprint, estimator=identifier))}
        adapters[identifier, "residual_correction"] = NodeAdapter(
            {"field": raw, "observations": inputs["observations"]}, {"correction": correction}, estimate,
            decision_contract=dict(operation="compute", registered=False, algorithm_id=identifier,
                capabilities=["observation_constraint"], input="field"))
        identifiers.append(identifier)
    if not identifiers:
        return []
    def add(values, params, external):
        background, estimate = values["field"].value, values["correction"].value
        if Artifact("background", background.data).fingerprint() != values["correction"].metadata["background_fingerprint"]:
            raise DAGError("Residual correction belongs to another background")
        field, delta = xr.align(background.data.dataset[context.variable], estimate.data.dataset[context.variable], join="exact")
        ds = background.data.dataset.copy(deep=True)
        ds[context.variable] = residual_addition(field, delta)
        ds[context.variable].attrs = dict(field.attrs)
        ds["background_value"] = field
        ds["residual_correction"] = delta
        if "residual_gp_std" in estimate.data.dataset:
            ds["residual_gp_std"] = estimate.data.dataset.residual_gp_std
        report = dict(background_algorithm=background.algorithm_id, residual_algorithm=estimate.algorithm_id,
                      background_report=background.report, residual_report=estimate.report,
                      state_owner_checked=True, reference_used=False)
        result = AlgorithmResult(EcoDataBundle(ds, background.data.history + estimate.data.history), "fusion.residual", report)
        return {"product": Artifact(corrected.kind, result, dict(contract=values["field"].metadata["contract"]))}
    adapters["fusion.residual", "add_background"] = NodeAdapter({"field": raw, "correction": correction},
        {"product": corrected}, add, decision_contract=dict(operation="compute", registered=False,
            capabilities=["multi_source_fusion", "observation_constraint"], input="field"))
    return identifiers + ["fusion.residual"]
