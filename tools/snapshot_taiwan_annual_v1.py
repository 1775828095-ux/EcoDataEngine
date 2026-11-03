"""Preserve final reproduction sources and reuse sensitivity evidence without fits."""
import importlib.metadata
import json
import shutil
import sys

import pandas as pd

from taiwan_working_contract import CONFIG, OUT, ROOT, save, sha


def main():
    folder = OUT/"code_snapshot"
    folder.mkdir(exist_ok=False)
    sources = ["configs/taiwan_2017_working_v1.json", "tools/taiwan_working_contract.py",
        "tools/taiwan_annual_adapter.py", "tools/prepare_taiwan_annual_v1.py", "tools/run_taiwan_annual_v1.py",
        "tools/audit_taiwan_annual_v1.py", "tools/verify_taiwan_annual_v1.py", "tools/enumerate_taiwan_objectives_v1.py",
        "tools/export_taiwan_coordinate_metadata_v1.py", "tools/run_taiwan_preparation.py", "tools/run_three_source_ready_smoke.py",
        "src/ecodata/algorithms/dag.py", "src/ecodata/algorithms/plugins.py", "src/ecodata/algorithms/contracts.py",
        "src/ecodata/evaluation/quality_gates.py", "tests/test_taiwan_working_contract.py", "tests/test_taiwan_annual_execution.py"]
    rows = []
    for name in sources:
        output = folder/name
        output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,output)
        assert sha(ROOT/name) == sha(output)
        rows.append(dict(source=name,snapshot=str(output.relative_to(OUT)),sha256=sha(output)))
    sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
    versions = {}
    for package in ["numpy","pandas","pyproj","h5py","xarray","scikit-learn","PyKrige","mgwr","thegolem"]:
        versions[package] = importlib.metadata.version(package)
    save(OUT/"source_and_environment.json",dict(sources=rows,python=sys.version,packages=versions,
        reproduction_scope="final working-contract pipeline; full project still needed",third_party_optimizer_modified=False,
        contract_sha256=sha(CONFIG),new_code_snapshot_not_a_new_algorithm_library=True))
    previous = ROOT/"output/taiwan_contract_sensitivity_paired_20261019"
    pairs = pd.read_csv(previous/"paired_rank_changes.csv")
    pairs = pairs.loc[pairs.hypothesis_a.str.endswith("4326") & pairs.hypothesis_b.str.endswith("4326")]
    save(OUT/"reused_sensitivity.json",dict(previous_source=str(previous.relative_to(ROOT)),previous_real_fits=576,
        alternatives_rerun=False,comparable_time_pairs=len(pairs),ranking_changed=int(pairs.rank_changed.sum()),
        first_place_changed=int((pairs.a_best != pairs.b_best).sum()),
        main_time_rule="g8/s8/start frozen by user, not selected from these scores",
        limitation="Earlier distances used EPSG3826 / IDW AEQD and four-date samples; not a controlled full-year EPSG32651 sensitivity comparison",
        claim="Uncertainty disclosed; main-rule annual scores do not establish robust historical-definition truth"))
    print("Final source snapshot",len(rows),"files; previous sensitivity evidence reused without retraining",flush=True)


if __name__ == "__main__":
    main()
