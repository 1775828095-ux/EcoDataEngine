"""Artifact/source checks for V5; not a model or GOLEM execution."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/algorithm_library_expansion/v5"


def main():
    upstream=ROOT/"output/algorithm_library_expansion/upstream"
    repos={"liif":"yinboc__liif","mpspline2":"obrl-soil__mpspline2","aqp":"ncss-tech__aqp","pyresample":"pytroll__pyresample",
        "emiproc":"C2SM-RCM__emiproc","HERMES_GR":"BSC-ES__HERMES_GR","CMAQ":"USEPA__CMAQ","SWATEditor":"swat-model__swatplus-editor"}
    sources={}
    for key,folder in repos.items():
        info=json.loads((upstream/folder/"source_evidence.json").read_text(encoding="utf-8"))
        sources[key]={"repository":info["repository"],"commit":info["commit"],"license":info["license"],"files":info["files"]}
    # Use actual DESCRIPTION/license files rather than GitHub's sometimes-null classification.
    sources["mpspline2"]["license_verified_from_DESCRIPTION"]="GPL"
    sources["aqp"]["license_verified_from_installed_DESCRIPTION"]="GPL (>=3)"
    sources["HERMES_GR"]["license_verified_from_LICENSE"]="Apache-2.0"
    sources["SWATEditor"]["license_verified_from_license_txt"]="Public-domain/service disclaimer; retain upstream terms and component licenses"
    package_matches={}
    for name,paths in {"emiproc":["emiproc/regrid.py","emiproc/profiles/temporal/operators.py"],
                      "pyresample":["pyresample/bilinear/_numpy_resampler.py"]}.items():
        entries={r["path"]:r for r in sources[name]["files"]}
        for p in paths:
            installed=ROOT/".venv-capability-v5/Lib/site-packages"/p
            actual=hashlib.sha256(installed.read_bytes()).hexdigest()
            package_matches[p]={"byte_identical":actual==entries[p]["sha256"],
                "text_identical_normalized_newlines":installed.read_text(encoding="utf-8")==
                    (upstream/repos[name]/p).read_text(encoding="utf-8"),"installed_sha256":actual}
    assert all(v["text_identical_normalized_newlines"] for v in package_matches.values()), package_matches
    files=[]
    for path in (OUT/"real_honcut_profile.csv",OUT/"aqp_sp2_original.csv",OUT/"aqp_sp2_metadata.txt",OUT/"mpspline_source_match.csv",
        ROOT/"output/algorithm_library_expansion/v4/era5_bth_t2m_reference.nc",ROOT/"examples/engineering_case_air_quality/terrain/dem.tif",
        OUT/"downloads/R-4.5.2-win.exe"):
        assert path.is_file()
        files.append({"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    records=json.loads((OUT/"execution_manifest.json").read_text(encoding="utf-8"))
    for r in records:
        path=OUT/r["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r["sha256"]
        assert r["status"] in {"PASS","WARN"}
        assert r["quality"]["quality_control"]["input"]["status"] in {"PASS","WARN"}
        with xr.open_dataset(path,engine="h5netcdf") as ds:
            assert np.isfinite(ds.value).all() and ds.value.attrs.get("units")
    blockers={
        "C5b":{"replacement":"LIIF locally trained on ERA5; actual coordinate-query inference at 1.5,2,3,4x", "scope":"Only confirmed regional temperature task"},
        "C8":{"replacement":"pyresample nearest and bilinear; no conservative replacement claim"},
        "C15":{"resolved":"Independent R 4.5.2 + mpspline2 0.1.9 + aqp 2.3.2", "input":"Honcut field pH; USGS Bulletin 1590-G provenance"},
        "C24_C25":{"replacement":"emiproc official spatial weights and daily profile APIs executed on controlled official test-cell subset", "native_model_verified":False},
        "HERMES_GR":{"status":"BLOCKED","executed":False,"evidence":"Official environment.yml requires Python3.10 and MPI-enabled NetCDF/HDF5, eccodes; model/inventory/speciation config absent; full environment not installed"},
        "C26":{"status":"BLOCKED","candidates_checked":["HERMES_GR","emiproc.speciation"],"reason":"No executed matching chemical-mechanism table; official CO2 source partition table is not a mechanism conversion"},
        "C27_C28":{"status":"BLOCKED","executed":False,"reason":"Windows gfortran exists, but official csh/I-O API compatible runtime and MCIP/project input set not present; no fake ICON/BCON output generated"},
        "C29_C30":{"scope":"Existing SWAT+62 writers preserved and re-executed; 61 existing tests passed; Editor source evaluated, not its database backend executed", "native_software_run_verified":False},
        "C7":{"scope":"V4 real temperature and pressure paths retained; no additional large terrain environment needed for current bounded coverage"},
    }
    (OUT/"upstream_blockers.json").write_text(json.dumps(blockers,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"source_manifest.json").write_text(json.dumps({"sources":sources,"local_input_files":files,"original_data_modified":False,
        "installed_python_core_matches_official_source":package_matches,
        "R_installer_url":"https://cran.r-project.org/bin/windows/base/old/4.5.2/R-4.5.2-win.exe",
        "R_MD5_reference":"https://cran.r-project.org/bin/windows/base/old/4.5.2/md5sum.R-4.5.2.txt",
        "R_binary_packages_source":"https://cloud.r-project.org/bin/windows/contrib/4.5/"},ensure_ascii=False,indent=2),encoding="utf-8")
    tests={}
    for name in ("targeted_tests.xml","model_audit_tests.xml","full_regression.xml"):
        path=OUT/name
        assert path.exists(), name
        suites=ET.parse(path).findall(".//testsuite")
        item={k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
        assert item["tests"] and not item["failures"] and not item["errors"], item
        tests[name]=item
    baseline=json.loads((ROOT/"output/algorithm_library_expansion/v2_delivery_check.json").read_text(encoding="utf-8"))["golem_sha256_unchanged"]
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in baseline.items())
    result={"checked_outputs":len(records),"tests":tests,"golem_core_unchanged":True,"formal_golem_experiment_started":False,
        "ALL_C1_C30_FULLY_VERIFIED":False,"scope":"Core algorithm execution != native model adapter/official model execution"}
    (OUT/"delivery_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":main()
