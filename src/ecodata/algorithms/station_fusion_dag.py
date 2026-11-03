"""Typed execution modes for existing station/residual components.

No new Registry algorithms: expose the existing Taiwan regression and residual
operations as separately executed, fit-bound ports rather than opaque routes.
"""
import numpy as np
from scipy.spatial import Delaunay
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from ecodata.algorithms.dag import Artifact, DAGError, NodeAdapter, Port
from ecodata.algorithms.golem_generation import TypedGolemFactory


META = ("unit", "metric_crs", "time", "contract")
SPATIAL = ["station.pykrige_ordinary", "station.pykrige_universal",
           "station.metpy_linear", "station.metpy_natural_neighbor",
           "station.metpy_barnes", "station.metpy_cressman"]


class StationFusionDAG:
    def __init__(self, *, xy, idw, rf_fit, quality, input_quality=None):
        self.xy, self.idw, self.rf_fit, self.quality = xy, idw, rf_fit, quality
        self.input_quality = input_quality

    def source(self, inputs, params, external):
        a = external[params["source"]]
        t = a.value
        required = {"station_id", "longitude", "latitude", "time", "unit", "aod"}
        if not required.issubset(t) or t.time.nunique() != 1:
            raise DAGError("Missing station fields or nonunique hour")
        if t.duplicated(["station_id", "time"]).any() or not t.unit.eq("ug m-3").all():
            raise DAGError("Duplicate station/hour or incompatible concentration unit")
        if not np.isfinite(t[["longitude", "latitude", "aod"]]).all().all():
            raise DAGError("Invalid coordinates/AOD; missing mask cannot be discarded")
        if not t.longitude.between(-180,180).all() or not t.latitude.between(-90,90).all():
            raise DAGError("Geographic coordinates out of range")
        if params["source"] == "train":
            if not t.split.eq("train").all() or not np.isfinite(t.value).all() or (t.value < 0).any():
                raise DAGError("Only finite nonnegative training labels may enter fitting")
            if len(t)<15:
                raise DAGError("At least 15 training stations required by the frozen task")
            if self.input_quality is not None:
                report=self.input_quality(t)
                if report["status"]=="BLOCK":
                    raise DAGError("RawInputQualityGate BLOCK before algorithm fitting")
                a=Artifact(a.kind,a.value,dict(a.metadata,input_quality=report))
        elif "value" in t or "split" in t:
            raise DAGError("Query port must not carry held-out labels")
        if a.metadata["unit"] != "ug m-3" or a.metadata["metric_crs"] != "EPSG:32651":
            raise DAGError("Frozen unit/metric CRS mismatch")
        if not t.time.eq(a.metadata["time"]).all():
            raise DAGError("Artifact time differs from row interval start")
        return {"table": a}

    def fit(self, inputs, params, external):
        if params.get("regressor") not in {"linear","rf"}:
            raise DAGError("Unsupported regression parameter; expected linear or rf")
        a = inputs["train"]
        t = a.value
        if not t.split.eq("train").all():
            raise DAGError("Fit rejected: nontraining station")
        if params["regressor"] == "rf":
            state = self.rf_fit(t)
        else:
            model = LinearRegression()
            x, y = t[["aod"]].to_numpy(), t.value.to_numpy()
            oof = cross_val_predict(model, x, y, groups=t.station_id, cv=GroupKFold(5), n_jobs=1)
            model.fit(x, y)
            state = dict(model=model, residual=y-oof, fit_stations=t.station_id.tolist(),
                         crossfit="5-fold station-disjoint")
        state["fit_fingerprint"] = a.fingerprint()
        artifact=Artifact("background_state",state,dict(a.metadata))
        # Assign the owner once, before prediction can touch an estimator copy.
        artifact.metadata["fit_state_fingerprint"]=artifact.fingerprint()
        return {"state":artifact}

    def check(self, inputs):
        train, query = inputs["train"], inputs["query"]
        if any(train.metadata[k] != query.metadata[k] for k in META):
            raise DAGError("Unit/CRS/hour/contract mismatch across branches")
        if set(train.value.station_id) & set(query.value.station_id):
            raise DAGError("Held-out station leakage")
        if (Delaunay(self.xy(train.value)).find_simplex(self.xy(query.value)) < 0).any():
            raise DAGError("Outside training support domain; no extrapolation")
        return train, query

    def state(self, inputs):
        a = inputs["train"]
        state = inputs["state"]
        if state.value["fit_fingerprint"] != a.fingerprint():
            raise DAGError("Fit/apply state belongs to another training dataset")
        return state.value

    def background(self, inputs, params, external):
        train, query = self.check(inputs)
        state = self.state(inputs)
        return {"prediction": Artifact("background_prediction",
            state["model"].predict(query.value[["aod"]].to_numpy()),
            dict(query.metadata,fit_state_fingerprint=inputs["state"].metadata["fit_state_fingerprint"]))}

    def residual(self, inputs, params, external):
        state = self.state(inputs)
        a = inputs["train"]
        t = a.value.copy()
        t["value"] = state["residual"]
        return {"table": Artifact("residual_training", t, dict(a.metadata, residual="OOF_station_5fold",
            fit_state_fingerprint=inputs["state"].metadata["fit_state_fingerprint"]))}

    def interpolate(self, operation, residual=False):
        def run(inputs, params, external):
            train, query = self.check(inputs)
            t, q = train.value, query.value
            a, b = self.xy(t), self.xy(q)
            if operation == "fusion.residual":
                values = self.idw(t, q)
            elif operation.startswith("station.pykrige"):
                from pykrige.ok import OrdinaryKriging
                from pykrige.uk import UniversalKriging
                kw = dict(variogram_model="linear", pseudo_inv=True, verbose=False)
                if operation.endswith("universal"):
                    model = UniversalKriging(a[:, 0], a[:, 1], t.value.to_numpy(),
                        drift_terms=["specified"], specified_drift=[t.aod.to_numpy()], **kw)
                    values, _ = model.execute("points", b[:, 0], b[:, 1], specified_drift_arrays=[q.aod.to_numpy()])
                else:
                    model = OrdinaryKriging(a[:, 0], a[:, 1], t.value.to_numpy(), **kw)
                    values, _ = model.execute("points", b[:, 0], b[:, 1])
            else:
                from metpy.interpolate import interpolate_to_points
                values = interpolate_to_points(a, t.value.to_numpy(), b,
                    interp_type=operation.removeprefix("station.metpy_"))
            kind = "residual_prediction" if residual else "raw_prediction"
            metadata=dict(query.metadata)
            if residual: metadata["fit_state_fingerprint"]=train.metadata["fit_state_fingerprint"]
            return {"prediction": Artifact(kind, np.asarray(values, float), metadata)}
        return run

    def combine(self, inputs, params, external):
        a, b = inputs["background"], inputs["residual"]
        if any(a.metadata[k] != b.metadata[k] for k in META) or a.value.shape != b.value.shape:
            raise DAGError("Residual/background output contract mismatch")
        if not a.metadata.get("fit_state_fingerprint") or a.metadata["fit_state_fingerprint"]!=b.metadata.get("fit_state_fingerprint"):
            raise DAGError("Background and residual must use the same fitted state")
        return {"prediction": Artifact("raw_prediction", a.value+b.value, dict(a.metadata))}

    def finish(self, inputs, params, external):
        self.check(inputs)
        values = inputs["prediction"].value
        if len(values) != len(inputs["query"].value) or not np.isfinite(values).all():
            raise DAGError("BLOCK: incomplete/nonfinite prediction; entire common cohort required")
        result = self.quality(inputs, params, external)
        return {"field": Artifact("validated_field", {
            "prediction": result["prediction"].value,
            "raw_prediction": result["raw_prediction"].value,
            "quality": result["quality"].value}, dict(inputs["query"].metadata))}

    def adapters(self):
        p = lambda k: Port(k, META)
        origin = lambda k: Port(k, META+("fit_state_fingerprint",))
        training, query = p("training"), p("query")
        out = {("source.train", "run"): NodeAdapter({}, {"table": training}, self.source),
               ("source.query", "run"): NodeAdapter({}, {"table": query}, self.source),
               ("fusion.residual", "fit"): NodeAdapter({"train": training}, {"state": origin("background_state")}, self.fit),
               ("fusion.residual", "background"): NodeAdapter({"train": training, "query": query,
                    "state": origin("background_state")}, {"prediction": origin("background_prediction")}, self.background),
               ("fusion.residual", "residual"): NodeAdapter({"train": training, "state": origin("background_state")},
                    {"table": origin("residual_training")}, self.residual),
               ("fusion.residual", "sum"): NodeAdapter({"background": origin("background_prediction"),
                    "residual": origin("residual_prediction")}, {"prediction": p("raw_prediction")}, self.combine),
               ("quality.station", "run"): NodeAdapter({"train": training, "query": query,
                    "prediction": p("raw_prediction")}, {"field": p("validated_field")}, self.finish)}
        for operation in ["fusion.residual"]+SPATIAL:
            for residual in [False, True]:
                mode = "residual_interpolate" if residual else "direct"
                out[operation, mode] = NodeAdapter({"train": origin("residual_training") if residual else training,
                    "query": query}, {"prediction": origin("residual_prediction") if residual else p("raw_prediction")},
                    self.interpolate(operation, residual), version="station-modes-1")
        return out


class StationGraphFactory(TypedGolemFactory):
    SOURCES = {"source.train": "train", "source.query": "query"}

    def initial(self):
        from golem.core.optimisers.graph import OptGraph
        _, recipe = self.recipes("validated_field")[0]
        return OptGraph([self.instantiate(recipe, omit_root_ports=("prediction",))])

    def parameters(self, key):
        params = {"mode": key[1]} if key[1] != "run" else {}
        if key[0] in self.SOURCES:
            params["source"] = self.SOURCES[key[0]]
        for regressor in (["linear", "rf"] if key == ("fusion.residual", "fit") else [None]):
            yield tuple(sorted(dict(params, **({"regressor": regressor} if regressor else {})).items()))

    def recipes(self, kind, stack=()):
        results = super().recipes(kind, stack)
        # Subtrees are generated independently by official add/tree mutations.
        # Enforce state consistency there too, not only at the final sink.
        return [(output, recipe) for output, recipe in results if len(self.regressors(recipe)) <= 1]

    @staticmethod
    def regressors(recipe):
        found = {dict(recipe[1])["regressor"]} if "regressor" in dict(recipe[1]) else set()
        for _, _, child in recipe[2]:
            found |= StationGraphFactory.regressors(child)
        return found

    def inspect(self, graph, *, complete=False):
        result = super().inspect(graph, complete=complete)
        fit = [n for n in graph.nodes if n.parameters.get("mode") == "fit"]
        if len(fit) > 1:
            raise DAGError("Branches must share the same training state")
        requirements=getattr(self,"requirements",None)
        if requirements is not None:
            provided=set()
            if any(n.name.startswith("station.") or (n.name=="fusion.residual" and n.parameters.get("mode") in {"direct","residual_interpolate"}) for n in graph.nodes):
                provided.add("observation_interpolation")
            if any(n.name=="quality.station" for n in graph.nodes):
                provided.add("nonnegative_validation")
            missing=sorted(set(requirements.required)-provided)
            result.update(semantic_required=list(requirements.required),semantic_missing=missing,
                semantic_trace=list(requirements.trace),gap=max(result["gap"],len(missing)))
            if complete and missing:
                raise DAGError(f"Missing semantic capabilities: {missing}")
        return result
