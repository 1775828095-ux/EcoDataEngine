"""Run C26 then restricted C27/C28 through model adapters and shared gates."""
import hashlib
import json
from pathlib import Path
import shutil

from ecodata.models.adapters.official_preprocessing import get_preprocessor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/capability_final_acceptance"


def main():
    records=[]
    s2s=OUT/"s2s_run/output"
    gspro=next(s2s.glob("gspro.CB6R5_AE7_CRITERIA_VOC.CMAQ.*.txt"))
    cfg={"mechanism":"CB6R5_AE7","profile_code":"0000","input_pollutant":"TOG","unit":"kg","input_mass":1.,
        "mass_basis_description":"1 kg reference basis of official SPECIATE profile 0000; not measured inventory mass"}
    report=get_preprocessor("epa_s2s_emiproc").generate(gspro,OUT/"official_s2s/input/mechanism_mw.csv",cfg,OUT/"chemical_speciation.nc")
    records.append({"capability":"C26","output":"chemical_speciation.nc","report":report})
    pnc=ROOT/".venv/Lib/site-packages/PseudoNetCDF/testcase/cmaqfiles"
    sources=OUT/"pnc_official_reference";sources.mkdir(exist_ok=True)
    for name,path in {"icon_profile":pnc/"profiles/test.icon_profile","bcon_profile":pnc/"profiles/test.bcon_profile","GRIDDESC":pnc/"griddesc/GRIDDESC"}.items():
        shutil.copy2(path,sources/name)
    # Explicit official example domain and documented library vertical settings.
    # Not asserted to be a measured MCIP file or a current CB6/CMAQ5.5 setup.
    cfg={"profile_kind":"static_CB05_AERO5_reference","grid_name":"36US3","VGTYP":7,"VGTOP":5000.,"SDATE":2011222,"STIME":0}
    adapter=get_preprocessor("pnc_cmaq_profile")
    for code,kind in (("C27","icon"),("C28","bcon")):
        output=OUT/(kind+"_static_profile.nc")
        report=adapter.generate(kind,sources/(kind+"_profile"),sources/"GRIDDESC",cfg,output)
        records.append({"capability":code,"output":output.name,"report":report})
    for r in records:r["sha256"]=hashlib.sha256((OUT/r["output"]).read_bytes()).hexdigest()
    emissions=get_preprocessor("emiproc_edgar_subset").generate(ROOT/"examples/ai_round1/cmaq/emissions/A_EDGAR_CO_2014_Paris.nc",
        {"temporal_assumption":"uniform_annual_mean_equivalent_day","confirm_assumption":True},OUT)
    records.append({"capability":"C24/C25","report":emissions,"output":"edgar_hourly.nc","sha256":hashlib.sha256((OUT/"edgar_hourly.nc").read_bytes()).hexdigest()})
    (OUT/"model_execution_manifest.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps([{r['capability']:r['output']} for r in records]))


if __name__=="__main__":main()
