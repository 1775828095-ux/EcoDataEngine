"""BTH task modes of frozen kernels; WGS84 is an explicit user convention."""
import numpy as np
from pyproj import Transformer
from sklearn.model_selection import GroupKFold

from ecodata.algorithms.dag import Artifact, DAGError, NodeAdapter, Port
from ecodata.algorithms.station_fusion_dag import StationFusionDAG, StationGraphFactory, META
from ecodata.algorithms.quality_aware import fit_affine_source
from ecodata.evaluation.quality_gates import RawInputQualityGate,ResultQualityGate
import xarray as xr


FEATURES = {"geos": ["geos"], "era5": ["t2m","d2m","u10","v10","sp","tp"],
            "both": ["geos","t2m","d2m","u10","v10","sp","tp"]}
TX = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)


def xy(table):
    return np.column_stack(TX.transform(table.longitude.to_numpy(), table.latitude.to_numpy()))


class BTHFusionDAG(StationFusionDAG):
    def __init__(self):
        super().__init__(xy=xy, idw=None, rf_fit=None, quality=None)

    def source(self, inputs, params, external):
        artifact = external[params["source"]]
        table = artifact.value
        needed = {"station_id","time","longitude","latitude","unit",*FEATURES["both"]}
        if not needed.issubset(table) or table.time.nunique()!=1 or table.station_id.duplicated().any():
            raise DAGError("Missing feature/time/entity contract")
        if artifact.metadata.get("coordinate_status")!="WGS84_USER_WORKING_ASSUMPTION":
            raise DAGError("Explicit coordinate working assumption required")
        if not table.unit.eq("ug m-3").all() or not np.isfinite(table[["longitude","latitude",*FEATURES["both"]]]).all().all():
            raise DAGError("Units/nonfinite source features")
        if not table.time.eq(artifact.metadata["time"]).all():
            raise DAGError("Time context mismatch")
        if params["source"]=="train":
            if not table.split.eq("train").all() or len(table)<15 or not np.isfinite(table.value).all() or table.value.lt(0).any():
                raise DAGError("Only sufficient finite training labels may enter fitting")
            source=xr.Dataset({"pm25":("station",table.value.to_numpy())},coords={
                "longitude":("station",table.longitude.to_numpy()),"latitude":("station",table.latitude.to_numpy())},
                attrs={"crs":"EPSG:4326 working assumption"})
            source.pm25.attrs["units"]="ug m-3"
            report=RawInputQualityGate().validate(source,variable="pm25",requires_spatial=True,expected_unit="ug m-3")
            if report["status"]=="BLOCK":raise DAGError("Training input quality BLOCK")
            artifact=Artifact(artifact.kind,table,dict(artifact.metadata,input_quality=report))
        elif "value" in table or "split" in table:
            raise DAGError("Query must not contain validation/test labels")
        return {"table":artifact}

    def spatial(self, operation):
        def run(inputs, params, external):
            train,query=self.check(inputs)
            a,b=xy(train.value),xy(query.value)
            from pykrige.ok import OrdinaryKriging
            from pykrige.uk import UniversalKriging
            kwargs=dict(variogram_model="linear",pseudo_inv=True,verbose=False)
            if operation=="station.pykrige_universal":
                names=FEATURES[params["features"]]
                x=train.value[names].to_numpy();q=query.value[names].to_numpy()
                mean=x.mean(0);std=x.std(0);keep=[];design=np.ones((len(x),1))
                # Training-only, label-independent rank check. No target-driven feature selection.
                for j in range(len(names)):
                    if std[j]>1e-12:
                        column=(x[:,j]-mean[j])/std[j]
                        if np.linalg.matrix_rank(np.column_stack([design,column]))>design.shape[1]:
                            keep.append(j);design=np.column_stack([design,column])
                if not keep:raise DAGError("No varying full-rank drift feature")
                model=UniversalKriging(a[:,0],a[:,1],train.value.value.to_numpy(),drift_terms=["specified"],
                    specified_drift=[(x[:,j]-mean[j])/std[j] for j in keep],**kwargs)
                values,_=model.execute("points",b[:,0],b[:,1],
                    specified_drift_arrays=[(q[:,j]-mean[j])/std[j] for j in keep])
                details={"features_used":[names[j] for j in keep],"normalization":"current training stations only"}
            else:
                model=OrdinaryKriging(a[:,0],a[:,1],train.value.value.to_numpy(),**kwargs)
                values,_=model.execute("points",b[:,0],b[:,1]);details={"features_used":[]}
            return {"prediction":Artifact("raw_prediction",np.asarray(values,float),dict(query.metadata,**details))}
        return run

    def fit_calibration(self,inputs,params,external):
        train=inputs["train"];t=train.value
        if not t.split.eq("train").all():raise DAGError("Calibration split leakage")
        slope,intercept=fit_affine_source(t.geos.to_numpy(),t.value.to_numpy())
        oof=np.full(len(t),np.nan)
        folds=[]
        for fit,held in GroupKFold(5).split(t,groups=t.station_id):
            s,i=fit_affine_source(t.geos.iloc[fit].to_numpy(),t.value.iloc[fit].to_numpy())
            oof[held]=s*t.geos.iloc[held].to_numpy()+i
            folds.append({"fit":t.station_id.iloc[fit].tolist(),"held":t.station_id.iloc[held].tolist()})
        state={"slope":slope,"intercept":intercept,"residual":t.value.to_numpy()-oof,
               "fit_fingerprint":train.fingerprint(),"folds":folds,"fit_stations":t.station_id.tolist(),
               "meaning":"empirical RH35 feature -> station target; NOT physical humidity conversion"}
        return {"state":Artifact("calibration_state",state,dict(train.metadata,fit_state_fingerprint=train.fingerprint()))}

    def calibrated(self,inputs,params,external):
        train,query=self.check(inputs);state=inputs["state"].value
        if state["fit_fingerprint"]!=train.fingerprint():raise DAGError("Calibration state owner mismatch")
        return {"prediction":Artifact("background_prediction",state["slope"]*query.value.geos.to_numpy()+state["intercept"],
            dict(query.metadata,fit_state_fingerprint=train.fingerprint()))}

    def residual_table(self,inputs,params,external):
        train=inputs["train"];state=inputs["state"].value
        if state["fit_fingerprint"]!=train.fingerprint():raise DAGError("Residual state owner mismatch")
        table=train.value.copy();table["value"]=state["residual"]
        return {"table":Artifact("residual_training",table,dict(train.metadata,fit_state_fingerprint=train.fingerprint()))}

    def residual_interpolator(self,operation):
        def run(inputs,params,external):
            train,query=self.check(inputs);a,b=xy(train.value),xy(query.value)
            if operation=="station.metpy_linear":
                from metpy.interpolate import interpolate_to_points
                values=interpolate_to_points(a,train.value.value.to_numpy(),b,interp_type="linear")
            else:
                from pykrige.ok import OrdinaryKriging
                model=OrdinaryKriging(a[:,0],a[:,1],train.value.value.to_numpy(),variogram_model="linear",pseudo_inv=True,verbose=False)
                values,_=model.execute("points",b[:,0],b[:,1])
            return {"prediction":Artifact("residual_prediction",np.asarray(values,float),dict(query.metadata,
                fit_state_fingerprint=train.metadata["fit_state_fingerprint"]))}
        return run

    def as_raw(self,inputs,params,external):
        a=inputs["background"]
        return {"prediction":Artifact("raw_prediction",a.value,a.metadata)}

    def finish(self,inputs,params,external):
        self.check(inputs)
        raw=np.asarray(inputs["prediction"].value,float)
        if len(raw)!=len(inputs["query"].value) or not np.isfinite(raw).all():
            raise DAGError("Incomplete common query output")
        values=np.maximum(raw,0)
        ds=xr.Dataset({"pm25":("point",values)},attrs={"crs":"EPSG:4326 working assumption"})
        ds.pm25.attrs["units"]="ug m-3"
        report=ResultQualityGate().validate(ds,ds,variable="pm25",nonnegative=True)
        if report["status"]=="BLOCK":raise DAGError("Result gate BLOCK")
        report.update(status="WARN",assumptions=["station WGS84 not verified","GEOS lev72 retained, not asserted surface"],
            nonnegative_projection_count=int((raw<0).sum()),accuracy_assessed=False)
        return {"field":Artifact("validated_field",{"prediction":values,"raw_prediction":raw,"quality":report},
            dict(inputs["query"].metadata))}

    def adapters(self):
        p=lambda k:Port(k,META)
        out={}
        def add(op,mode,ins,outs,fn):
            out[op,mode]=NodeAdapter({k:p(v) for k,v in ins.items()},{k:p(v) for k,v in outs.items()},fn,version="bth-exploratory-1")
        add("source.train","run",{}, {"table":"training"},self.source)
        add("source.query","run",{}, {"table":"query"},self.source)
        for op in ["station.pykrige_ordinary","station.pykrige_universal"]:
            add(op,"direct",{"train":"training","query":"query"},{"prediction":"raw_prediction"},self.spatial(op))
        add("fusion.quality_aware","fit",{"train":"training"},{"state":"calibration_state"},self.fit_calibration)
        add("fusion.quality_aware","apply",{"train":"training","query":"query","state":"calibration_state"},
            {"prediction":"background_prediction"},self.calibrated)
        add("prepare.calibrated","run",{"background":"background_prediction"},{"prediction":"raw_prediction"},self.as_raw)
        add("fusion.residual","residual",{"train":"training","state":"calibration_state"},{"table":"residual_training"},self.residual_table)
        for op in ["station.pykrige_ordinary","station.metpy_linear"]:
            add(op,"residual_interpolate",{"train":"residual_training","query":"query"},{"prediction":"residual_prediction"},self.residual_interpolator(op))
        add("fusion.residual","sum",{"background":"background_prediction","residual":"residual_prediction"},{"prediction":"raw_prediction"},self.combine)
        add("quality.station","run",{"train":"training","query":"query","prediction":"raw_prediction"},{"field":"validated_field"},self.finish)
        return out


class BTHGraphFactory(StationGraphFactory):
    def parameters(self,key):
        params={"mode":key[1]} if key[1]!="run" else {}
        if key[0] in self.SOURCES:params["source"]=self.SOURCES[key[0]]
        for features in (list(FEATURES) if key==("station.pykrige_universal","direct") else [None]):
            yield tuple(sorted(dict(params,**({"features":features} if features else {})).items()))
