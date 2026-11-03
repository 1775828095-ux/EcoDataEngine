"""Task-local adapters for the existing temporal baseline computations.

These four minimal baselines are not additions to the frozen 34-component pool.
Interpolation is on reported hourly labels, not an invented interval-mean model.
"""
import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmSpec, Compatibility, AlgorithmResult
from ecodata.algorithms.dag import Artifact, DAGError, NodeAdapter, Port
from ecodata.algorithms.golem_generation import TypedGolemFactory
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate


METHODS=("linear","nearest","ffill","hour_median")
IDS=tuple("baseline.temporal_"+name for name in METHODS)
META=("entity_id","file_id","unit_status","clock_status")


from ecodata.algorithms.temporal_kernels import reconstruct


class TemporalBaseline:
    def __init__(self,method):
        self.method=method
        # These describe the actual kernel, not preferred winners for gap sizes.
        support={"linear":"both","nearest":"either","ffill":"left",
                 "hour_median":"training","calendar_knn":"training"}[method]
        self.spec=AlgorithmSpec("baseline.temporal_"+method,method,"hour-label-v1","temporal_reconstruction",
            "available",("time_series",),("continuous",),backend="pandas/SciPy; existing UCI baseline operations",
            capabilities=("temporal_missing_recovery",),input_contract={"targets":"absent from inputs",
                "time":"unique increasing labels; unknown timezone preserved","units":"native, unconfirmed allowed only for restoration",
                "requires_fitted_state":support=="training"},
            output_contract={"kind":"time_series","support":support,"observation_evidence":("none" if support=="training" else "both" if support in {"both","either"} else support),
                "uses_future":support in {"both","either"},
                "not_supported":"spatial reconstruction or physical unit conversion"})

    def compatibility(self,context):
        eligible=bool(context.sources and context.variable in context.sources[0].dataset and context.semantic_type=="continuous"
            and context.sources[0].dataset[context.variable].dims==("time",))
        return Compatibility(eligible,() if eligible else ("Continuous per-entity time series required",))

    def run(self,context):
        if not self.compatibility(context).eligible:raise DAGError("Incompatible temporal input")
        source=context.sources[0].dataset
        result=source.copy()
        result[context.variable]=("time",reconstruct(source[context.variable].to_series(),self.method).to_numpy())
        result[context.variable].attrs=dict(source[context.variable].attrs)
        return AlgorithmResult(EcoDataBundle(result),self.spec.id,{"method":self.method,"native_units_only":True})


def temporal_setup():
    registry=AlgorithmRegistry()
    plugins={p.spec.id:p for p in [TemporalBaseline(m) for m in METHODS]}
    for plugin in plugins.values():registry.register(plugin)
    def source(inputs,params,external):
        artifact=external["series"]
        series=artifact.value
        targets=pd.DatetimeIndex(external["target_times"])
        if not series.loc[targets].isna().all():raise DAGError("Masked targets leaked into algorithm input")
        ds=xr.Dataset({"value":("time",series.to_numpy())},coords={"time":series.index})
        ds.value.attrs["units"]="1"  # Explicit retained-input scaling; not a claim about original PM units.
        quality=RawInputQualityGate().validate(ds,variable="value",intended_operation="temporal",requires_spatial=False)
        if quality["status"]=="BLOCK":raise DAGError("Raw temporal input blocked")
        return {"series":Artifact("masked_series",series,dict(artifact.metadata,input_quality=quality))}
    def predictor(method,kind):
        def run(inputs,params,external):
            item=inputs["series"]
            return {"prediction":Artifact(kind,reconstruct(item.value,method),dict(item.metadata))}
        return run
    def fallback(method):
        def run(inputs,params,external):
            primary,source=inputs["primary"],inputs["series"]
            if primary.metadata["entity_id"]!=source.metadata["entity_id"] or not primary.value.index.equals(source.value.index):
                raise DAGError("Fallback crossed entity or time axis")
            return {"prediction":Artifact("reconstruction_series",primary.value.fillna(reconstruct(source.value,method)),dict(primary.metadata))}
        return run
    def quality(inputs,params,external):
        item=inputs["prediction"]; target=pd.DatetimeIndex(external["target_times"])
        values=item.value.loc[target]
        ds=xr.Dataset({"value":("time",values.to_numpy())},coords={"time":target})
        ds.value.attrs["units"]="1"
        if not np.isfinite(values).all():raise DAGError("Output target coverage/nonnegative quality BLOCK")
        check=ResultQualityGate().validate(ds,ds,variable="value",semantic_type="continuous",nonnegative=True)
        if check["status"]=="BLOCK":raise DAGError("Output target coverage/nonnegative quality BLOCK")
        check["status"]="WARN";check["unit_status"]="UNCONFIRMED_NATIVE_UNITS_ONLY"
        check["checks"].append(dict(id="source.unit_unconfirmed",status="WARN",
            message="Only dimensionless normalized restoration checked; original physical concentration unit unconfirmed"))
        check["summary"]["WARN"]+=1;check["decision"]="exploratory_native_restoration_only"
        return {"field":Artifact("validated_field",dict(prediction=values.to_numpy(),quality=check),dict(item.metadata))}
    adapters={}
    p=lambda kind:Port(kind,META)
    adapters["source.temporal","run"]=NodeAdapter({}, {"series":p("masked_series")},source)
    for operation,plugin in plugins.items():
        adapters[operation,"direct"]=NodeAdapter({"series":p("masked_series")},{"prediction":p("reconstruction_series")},predictor(plugin.method,"reconstruction_series"))
        if plugin.method in {"linear","ffill"}:
            adapters[operation,"partial"]=NodeAdapter({"series":p("masked_series")},{"prediction":p("partial_series")},predictor(plugin.method,"partial_series"))
        if plugin.method in {"nearest","hour_median"}:
            adapters[operation,"fallback"]=NodeAdapter({"series":p("masked_series"),"primary":p("partial_series")},
                {"prediction":p("reconstruction_series")},fallback(plugin.method))
    adapters["quality.temporal","run"]=NodeAdapter({"prediction":p("reconstruction_series")},{"field":p("validated_field")},quality)
    registry.register_task_adapters("city_pm25_temporal",adapters,algorithm_ids=IDS,boundary_ids=("source.temporal","quality.temporal"))
    return registry,TemporalGraphFactory(adapters,IDS,max_nodes=8,max_depth=6)


class TemporalGraphFactory(TypedGolemFactory):
    SOURCES={"source.temporal":"series"}
    PARAMS={}

    def initial(self):
        from golem.core.optimisers.graph import OptGraph
        _,recipe=self.recipes("validated_field")[0]
        return OptGraph([self.instantiate(recipe,omit_root_ports=("prediction",))])
