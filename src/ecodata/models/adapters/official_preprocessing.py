"""Explicit model preprocessing APIs; no native-model or generic-search claim."""
import json
from pathlib import Path
import subprocess
import tempfile

import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate

ROOT=Path(__file__).resolve().parents[4]
RUNTIME=ROOT/".venv-capability-v6/Scripts/python.exe"


def _quality(source,output,variables):
    # I/O API uses fixed-width, space-padded unit attributes. Normalize only
    # whitespace in the QA views; preserve the native files and numerical data.
    source=source.copy(deep=False);output=output.copy(deep=False)
    for ds in (source,output):
        for name in variables:
            ds[name].attrs={**ds[name].attrs,"units":str(ds[name].attrs.get("units","")).strip()}
    reports={}
    for name in variables:
        before=RawInputQualityGate().validate(source,variable=name,requires_spatial=False)
        after=ResultQualityGate().validate(source,output,variable=name,semantic_type="continuous",nonnegative=True,
            model_check={"passed":True,"scope":"worker independent structure/numerical assertions, NOT official model run"})
        if before["status"]=="BLOCK" or after["status"]=="BLOCK":raise SchemaError("Model preprocessing quality gate blocked: "+name)
        reports[name]={"input":before,"output":after}
    return reports


class OfficialChemicalSpeciationAdapter:
    def generate(self,gspro,molecular_weights,config,output):
        if config.get("mechanism")!="CB6R5_AE7" or config.get("profile_code")!="0000" or config.get("unit")!="kg" or config.get("input_pollutant")!="TOG" or not config.get("mass_basis_description"):
            raise SchemaError("Confirmed CB6R5_AE7 TOG mass basis in kg required")
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        _execute("run_chemical_speciation_acceptance.py",[str(gspro),str(molecular_weights),str(output)],config)
        ds=xr.load_dataset(output,engine="h5netcdf")
        source=xr.Dataset({"mass":("record",[float(config["input_mass"])])})
        source.mass.attrs["units"]="kg"
        report=json.loads(Path(str(output)+".json").read_text(encoding="utf-8"))
        report["quality_control"]=_quality(source,ds,["mass"])
        Path(str(output)+".validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        return report


class OfficialCmaqProfileAdapter:
    def generate(self,kind,profile,griddesc,config,output):
        if kind not in {"icon","bcon"} or config.get("profile_kind")!="static_CB05_AERO5_reference":
            raise SchemaError("Only the verified legacy static profile contract is accepted")
        for key in ("grid_name","VGTYP","VGTOP","SDATE","STIME"):
            if key not in config:raise SchemaError("Missing explicit project metadata: "+key)
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        _execute("run_cmaq_profile_acceptance.py",[kind,str(profile),str(griddesc),str(output)],config)
        source=xr.load_dataset(str(output)+".source.nc",engine="h5netcdf")
        ds=xr.load_dataset(output,engine="netcdf4",decode_times=False)
        report=json.loads(Path(str(output)+".json").read_text(encoding="utf-8"))
        report["quality_control"]=_quality(source,ds,report["species"])
        Path(str(output)+".validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        return report


class OfficialEDGARAdapter:
    def generate(self,source,config,destination):
        if config.get("temporal_assumption")!="uniform_annual_mean_equivalent_day" or config.get("confirm_assumption") is not True:
            raise SchemaError("Annual-mean equivalent day and uniform profile require explicit confirmation")
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        _execute("run_real_emissions_acceptance.py",[str(source),str(destination)],config)
        before=xr.load_dataset(destination/"edgar_daily_mass_source.nc",engine="h5netcdf")
        report=json.loads((destination/"real_emissions_execution.json").read_text(encoding="utf-8"))
        report["quality_control"]={}
        for name in ("edgar_spatial.nc","edgar_hourly.nc"):
            ds=xr.load_dataset(destination/name,engine="h5netcdf")
            report["quality_control"][name]=_quality(before,ds,["mass"])
        (destination/"real_emissions_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        return report


def _execute(script,arguments,config):
    if not RUNTIME.is_file():raise SchemaError("Optional official preprocessing environment missing")
    with tempfile.TemporaryDirectory(prefix="ecodata_official_model_") as temporary:
        path=Path(temporary)/"config.json"
        path.write_text(json.dumps(config),encoding="utf-8")
        result=subprocess.run([str(RUNTIME),str(ROOT/"tools"/script),*arguments,str(path)],capture_output=True,timeout=120)
        if result.returncode:
            failure=Path(arguments[-1])
            if script=="run_real_emissions_acceptance.py":failure=failure/"preprocessing"
            failure.parent.mkdir(parents=True,exist_ok=True)
            Path(str(failure)+".failure.log").write_bytes(result.stdout+b"\nSTDERR\n"+result.stderr)
            raise SchemaError(result.stderr.decode("utf-8",errors="replace")[-3500:])


def get_preprocessor(identifier):
    adapters={"epa_s2s_emiproc":OfficialChemicalSpeciationAdapter,"pnc_cmaq_profile":OfficialCmaqProfileAdapter,"emiproc_edgar_subset":OfficialEDGARAdapter}
    if identifier not in adapters:raise SchemaError("Unknown official model preprocessor: "+identifier)
    return adapters[identifier]()
