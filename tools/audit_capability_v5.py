"""One row per capability, not per algorithm; no model-runtime inference."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from ecodata.algorithms.registry import create_default_registry

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"output/algorithm_library_expansion"
V5 = OUT/"v5"

LIMITS = {
    1:"Continuous numeric fields only; variable/domain contracts still apply",
    2:"Nearest/mode cannot infer missing category meaning", 3:"Fractions describe composition, not new class detail",
    4:"Existing regular-grid conservation only; no verified general curvilinear conservative backend",
    5:"Fixed ClimateLearn and locally trained LIIF temperature only, 1.5..4x queries; not all variables/regions",
    6:"Station geometry, variogram/sample support still required", 7:"Temperature and pressure paths only; full ForDs not executed",
    8:"pyresample nearest/bilinear; NOT conservative; real 2D DEM geolocation plus controlled nonrectilinear tests",
    9:"Time-window semantics must be explicit", 10:"Metric/calendar definitions must match variable",
    11:"Accumulation interval semantics required", 12:"Existing extreme correction component, no general extreme generative model",
    13:"Existing real irregular-time interpolation, not general forecasting", 14:"Explicit vertical coordinate/units required",
    15:"Single observed soil profile, cm support; missing horizons require confirmation; pH is not mass-conserved",
    16:"Structural/physical validity does not prove accuracy", 17:"Conservation depends on variable and geometry",
    18:"Coarse consistency alone does not establish fine-scale accuracy", 19:"Source compatibility/reference observations required",
    20:"Fit only on designated training data", 21:"Concurrent observations must be disclosed",
    22:"Reference-free accuracy ranking prohibited", 23:"Feature extraction is not physical prediction",
    24:"emiproc numerical fixture core verified; real inventory/model package NOT verified",
    25:"emiproc explicit daily-to-hourly fractions verified; annual/DST/model-native paths NOT verified",
    26:"No verified chemical mechanism mapping; table/chemistry dependencies still required",
    27:"Official ICON execution absent; I/O API, csh/compatible build and MCIP/project data missing",
    28:"Official BCON execution absent; I/O API, csh/compatible build and MCIP/project data missing",
    29:"SWAT+62 soil file/reference and landuse subsets only; not arbitrary models or full project runtime",
    30:"Fixed calendar management subset tested on synthetic schedules; automatic management/rotation still missing",
}


def main():
    matrix=OUT/"algorithm_capability_matrix.csv"
    if not (V5/"pre_v5_matrix.csv").exists():
        shutil.copy2(matrix,V5/"pre_v5_matrix.csv")
    with matrix.open(encoding="utf-8-sig",newline="") as stream:
        entries=list(csv.DictReader(stream))
    specs={s.id:s for s in create_default_registry().specs()}
    rows=[]
    for number in range(1,31):
        code=f"C{number}"
        candidates=[r for r in entries if r["CapabilityCode"]==code and r["Algorithm"] not in {"MODEL_SPECIFIC","NO_VERIFIED_CANDIDATE"}]
        names=sorted(set(r["Algorithm"] for r in candidates))
        states={r["Status"] for r in candidates}
        state="VERIFIED" if "VERIFIED" in states else ("PARTIAL" if "PARTIAL" in states else ("BLOCKED" if "BLOCKED" in states else "UNVERIFIED"))
        ran=state in {"VERIFIED","PARTIAL"} and any(r["Runtime Status"]=="available" for r in candidates)
        evidence=sorted({p for n in names if n in specs for p in specs[n].verification_evidence})
        layer="GENERIC_NUMERICAL" if number<=23 else "MODEL_PREPROCESSING"
        algorithm_state=state
        adapter_state="NOT_APPLICABLE" if number<=23 else "UNVERIFIED"
        if number in {24,25}:
            state="PARTIAL" # Broad engineering capability not proven by numerical fixtures.
            adapter_state="UNVERIFIED"
        if number in {26,27,28}:
            names={26:["HERMES_GR; emiproc speciation (not executed)"],27:["EPA ICON"],28:["EPA BCON"]}[number]
            state=algorithm_state="BLOCKED";ran=False
            evidence=["output/algorithm_library_expansion/v5/upstream_blockers.json"]
        if number in {29,30}:
            names=["SWAT+62 existing soil/landuse adapter" if number==29 else "SWAT+62 existing calendar management adapter"]
            state="PARTIAL";algorithm_state="NOT_APPLICABLE";ran=True
            adapter_state="VERIFIED_SCHEMA_SUBSET" if number==29 else "PARTIAL_SYNTHETIC_SUBSET"
            evidence=["output/algorithm_library_expansion/v5/model_audit_tests.xml", "tests/test_swat_plus_soil.py" if number==29 else "tests/test_swat_plus_management.py"]
        rows.append({"Capability":code,"Layer":layer,"Candidates":";".join(names),"ActuallyRun":ran,"Status":state,
            "ALGORITHM_CAPABILITY_VERIFIED":algorithm_state,"MODEL_ADAPTER_VERIFIED":adapter_state,
            "OfficialSoftwareReadVerified":False if number>=24 else "NOT_APPLICABLE",
            "Limitations":LIMITS[number],"Evidence":";".join(evidence)})
    with (V5/"C1_C30_coverage.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    baseline=json.loads((OUT/"v2_delivery_check.json").read_text(encoding="utf-8"))["golem_sha256_unchanged"]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in baseline.items())
    tests={}
    for name in ("targeted_tests.xml","model_audit_tests.xml","full_regression.xml"):
        path=V5/name
        if path.exists():
            suites=ET.parse(path).findall(".//testsuite")
            tests[name]={k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
    report={"coverage":rows,"golem_core_unchanged":True,"tests":tests,
        "note":"Capability role audit, not count of installed algorithms. VERIFIED is contract-scoped, not all environmental data."}
    (V5/"coverage_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    from collections import Counter
    print(json.dumps({"capabilities":len(rows),"states":dict(Counter(r["Status"] for r in rows)),"tests":tests},ensure_ascii=False))


if __name__=="__main__":
    main()
