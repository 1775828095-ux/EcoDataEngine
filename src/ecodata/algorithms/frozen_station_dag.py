"""Strict frozen-kernel station DAG: PyKrige/MetPy and project residual algebra.

RF and IDW remain historical baselines, never aliases of a frozen component.
"""
import numpy as np
from scipy.spatial import Delaunay
from sklearn.model_selection import GroupKFold

from ecodata.algorithms.dag import Artifact, DAGError, NodeAdapter, Port
from ecodata.algorithms.station_fusion_dag import StationFusionDAG, StationGraphFactory, META, SPATIAL
from ecodata.fusion.residual import residual_difference, residual_addition


BASE_MODELS=("station.pykrige_ordinary","station.pykrige_universal")


class FrozenStationFusionDAG(StationFusionDAG):
    def __init__(self, *, xy, quality, input_quality=None):
        super().__init__(xy=xy,idw=None,rf_fit=None,quality=quality,input_quality=input_quality)

    def estimator(self, operation, table):
        from pykrige.ok import OrdinaryKriging
        from pykrige.uk import UniversalKriging
        a=self.xy(table)
        kwargs=dict(variogram_model="linear",pseudo_inv=True,verbose=False)
        if operation==BASE_MODELS[0]:
            return OrdinaryKriging(a[:,0],a[:,1],table.value.to_numpy(),**kwargs)
        if operation==BASE_MODELS[1]:
            return UniversalKriging(a[:,0],a[:,1],table.value.to_numpy(),drift_terms=["specified"],
                specified_drift=[table.aod.to_numpy()],**kwargs)
        raise DAGError("Unfrozen estimator requested")

    def predict_estimator(self, model, operation, query):
        b=self.xy(query)
        kwargs={"specified_drift_arrays":[query.aod.to_numpy()]} if operation==BASE_MODELS[1] else {}
        values,_=model.execute("points",b[:,0],b[:,1],**kwargs)
        return np.asarray(values,float)

    def fit_model(self, operation):
        def fit(inputs,params,external):
            if params.get("estimator_id")!=operation: raise DAGError("Node/backend identity mismatch")
            a=inputs["train"]
            train=a.value
            if not train.split.eq("train").all(): raise DAGError("Only training stations can fit a state")
            oof=np.full(len(train),np.nan)
            fold_records=[]
            for fit_index,held in GroupKFold(5).split(train,groups=train.station_id):
                fit_table=train.iloc[fit_index]
                held_table=train.iloc[held]
                supported=Delaunay(self.xy(fit_table)).find_simplex(self.xy(held_table))>=0
                if supported.any():
                    model=self.estimator(operation,fit_table)
                    oof[held[supported]]=self.predict_estimator(model,operation,held_table.loc[supported])
                fold_records.append(dict(fit_stations=fit_table.station_id.tolist(),heldout_stations=held_table.station_id.tolist(),
                    supported=int(supported.sum()),excluded_outside_fold_hull=int((~supported).sum())))
            model=self.estimator(operation,train)
            state=dict(model=model,estimator_id=operation,backend_class=type(model).__module__+"."+type(model).__name__,
                residual=residual_difference(train.value.to_numpy(),oof),fit_fingerprint=a.fingerprint(),
                fit_stations=train.station_id.tolist(),crossfit="5 station-disjoint folds; unsupported OOF points remain missing",folds=fold_records)
            artifact=Artifact("background_state",state,dict(a.metadata,estimator_id=operation,
                backend_class=state["backend_class"]))
            artifact.metadata["fit_state_fingerprint"]=artifact.fingerprint()
            return {"state":artifact}
        return fit

    def apply_model(self, operation):
        def predict(inputs,params,external):
            _,query=self.check(inputs)
            state=self.state(inputs)
            if params.get("estimator_id")!=operation or state["estimator_id"]!=operation:
                raise DAGError("Fit/apply estimator identity differs")
            return {"prediction":Artifact("background_prediction",self.predict_estimator(state["model"],operation,query.value),
                dict(query.metadata,fit_state_fingerprint=inputs["state"].metadata["fit_state_fingerprint"]))}
        return predict

    def residual(self,inputs,params,external):
        state=self.state(inputs)
        a=inputs["train"]
        table=a.value.copy()
        table["value"]=state["residual"]
        valid=np.isfinite(table.value)
        return {"table":Artifact("residual_training",table.loc[valid].copy(),dict(a.metadata,
            fit_state_fingerprint=inputs["state"].metadata["fit_state_fingerprint"],
            residual="supported station-disjoint OOF only",excluded_oof=int((~valid).sum()),
            excluded_reason="outside fold training hull or unavailable OOF estimate"))}

    def interpolate(self,operation,residual=False):
        original=super().interpolate(operation,residual)
        if not residual: return original
        def run(inputs,params,external):
            train,query=inputs["train"],inputs["query"]
            if any(train.metadata[k]!=query.metadata[k] for k in META): raise DAGError("Residual context mismatch")
            supported=np.zeros(len(query.value),dtype=bool)
            values=np.full(len(query.value),np.nan)
            if len(train.value)>=4:
                supported=Delaunay(self.xy(train.value)).find_simplex(self.xy(query.value))>=0
                if supported.any():
                    subset=Artifact(query.kind,query.value.loc[supported].copy(),dict(query.metadata))
                    predicted=original(dict(train=train,query=subset),params,external)["prediction"].value
                    values[supported]=predicted
            valid=supported & np.isfinite(values)
            return {"prediction":Artifact("residual_prediction",values,dict(query.metadata,
                fit_state_fingerprint=train.metadata["fit_state_fingerprint"],valid_residual_mask=valid.tolist(),
                supported_residual_stations=len(train.value)))}
        return run

    def combine(self,inputs,params,external):
        a,b=inputs["background"],inputs["residual"]
        if any(a.metadata[k]!=b.metadata[k] for k in META) or a.value.shape!=b.value.shape:
            raise DAGError("Residual/background context mismatch")
        if not a.metadata.get("fit_state_fingerprint") or a.metadata["fit_state_fingerprint"]!=b.metadata.get("fit_state_fingerprint"):
            raise DAGError("Background and residual must use the same fitted state")
        valid=np.asarray(b.metadata["valid_residual_mask"],bool)
        if valid.shape!=a.value.shape or not np.isfinite(a.value).all(): raise DAGError("Invalid base or correction mask")
        values=a.value.copy()
        values[valid]=residual_addition(a.value[valid],b.value[valid])
        return {"prediction":Artifact("raw_prediction",values,dict(a.metadata,residual_used_mask=valid.tolist(),
            fallback="background only where no supported finite residual; never invent a residual observation"))}

    def finish(self,inputs,params,external):
        result=super().finish(inputs,params,external)
        mask=inputs["prediction"].metadata.get("residual_used_mask")
        if mask is not None:
            result["field"].value["residual_used_mask"]=mask
        return result

    def adapters(self):
        adapters=super().adapters()
        for key in list(adapters):
            if key[0]=="fusion.residual" and key[1] not in {"residual","sum"}: del adapters[key]
        p=lambda kind:Port(kind,META)
        state=Port("background_state",META+("fit_state_fingerprint",))
        output=Port("background_prediction",META+("fit_state_fingerprint",))
        for operation in BASE_MODELS:
            adapters[operation,"fit"]=NodeAdapter({"train":p("training")},{"state":state},self.fit_model(operation),version="strict-frozen-1")
            adapters[operation,"background"]=NodeAdapter({"train":p("training"),"query":p("query"),"state":state},
                {"prediction":output},self.apply_model(operation),version="strict-frozen-1")
        return adapters


class FrozenStationGraphFactory(StationGraphFactory):
    def parameters(self,key):
        params={"mode":key[1]} if key[1]!="run" else {}
        if key[0] in self.SOURCES: params["source"]=self.SOURCES[key[0]]
        if key[0] in BASE_MODELS and key[1] in {"fit","background"}: params["estimator_id"]=key[0]
        yield tuple(sorted(params.items()))

    @staticmethod
    def regressors(recipe):
        found={dict(recipe[1])["estimator_id"]} if "estimator_id" in dict(recipe[1]) else set()
        for _,_,child in recipe[2]: found |= FrozenStationGraphFactory.regressors(child)
        return found

    def inspect(self,graph,*,complete=False):
        status=super().inspect(graph,complete=complete)
        owners={n.parameters["estimator_id"] for n in graph.nodes if "estimator_id" in n.parameters}
        if len(owners)>1: raise DAGError("Fit/apply estimator identity differs across branches")
        return status
