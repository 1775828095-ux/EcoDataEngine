"""Readback, parity and bounded strategy wiring evidence; not a research study."""
from pathlib import Path
import hashlib,io,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tools"))
import numpy as np,xarray as xr
from ecodata.algorithms.numerical_search import save
from ecodata.algorithms.platform_search import submit_file_request
from accept_platform_integration import OUT,BUDGET,run_one


def source_snapshot(name="source_delivery"):
    destination=OUT/name
    files=list((ROOT/"src/ecodata").rglob("*.py"))+list((ROOT/"src/ecodata").rglob("*.json"))
    files += [ROOT/"tools"/name for name in ("accept_platform_integration.py","audit_platform_library.py","verify_platform_integration.py")]
    files += [ROOT/"tests/test_platform_integration.py"]
    manifest={}
    for file in files:
        relative=file.relative_to(ROOT);target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=file.read_bytes():raise ValueError("Frozen source delivery differs; do not overwrite")
        if not target.exists():shutil.copy2(file,target)
        manifest[str(relative)]=hashlib.sha256(file.read_bytes()).hexdigest()
    save(OUT/(name+"_manifest.json"),manifest)


def parity(suffix="final",spatial_suffix=None,output_name="entry_parity.json"):
    checks={}
    for task in ("spatial","temporal","multisource"):
        actual_suffix=spatial_suffix if task=="spatial" and spatial_suffix else suffix
        a=OUT/"runs"/(task+"_toolkit_"+actual_suffix);b=OUT/"runs"/(task+"_page_"+actual_suffix)
        left=json.loads((a/"summary.json").read_text(encoding="utf-8"));right=json.loads((b/"summary.json").read_text(encoding="utf-8"))
        decision=[]
        for path in (a,b):
            value=json.loads((path/"decision.json").read_text(encoding="utf-8"));value.pop("seconds",None);decision.append(value)
        projection=lambda r:[(v["key"],v["status"],v.get("objective"),v["graph"]) for v in r["rows"]]
        weights=lambda r:[{k:v[k] for k in ("weights","probabilities","chosen_index","keys") if k in v} for v in r["proposals"] if v.get("status")=="SHARED_DECISION"]
        for x,y in zip([decision[0],projection(left),weights(left),left["candidate_domain"],left["evaluation_protocol"],left["best"]["graph"]],
                       [decision[1],projection(right),weights(right),right["candidate_domain"],right["evaluation_protocol"],right["best"]["graph"]]):
            if x!=y:raise ValueError("Entry parity mismatch: "+task)
        for l,r in zip(left["rows"],right["rows"]):
            if l["status"]!="PASS":continue
            with xr.open_dataset(l["product"],engine="h5netcdf") as x,xr.open_dataset(r["product"],engine="h5netcdf") as y:xr.testing.assert_identical(x,y)
        checks[task]=dict(status="PASS",semantic_decision=True,candidate_domain=True,weights_and_actual_draw=True,selected_graph=True,numeric_products=True,
            toolkit=str(a),page_handler=str(b),counts=left["counts"],unique_valid=left["unique_valid"],unique_behaviors=left["unique_behaviors"],
            objective=left["best"]["objective"],metric=left["evaluation_protocol"],seconds=left["total_service_seconds"],
            setup_seconds=left["platform_setup_seconds"],search_seconds=left["seconds"],discrimination=left["discrimination"])
    save(OUT/output_name,checks)
    return checks


def strategy_wiring(name="strategy_wiring"):
    # Fixed before execution, only one real spatial input, no ranking study.
    inputs=OUT/"inputs"
    request=json.loads((inputs/"spatial_request.json").read_text(encoding="utf-8"))
    budget=dict(objective_limit=3,proposal_limit=20,callback_limit=80,wall_seconds=10)
    config=dict(inputs=[dict(path=str(inputs/"spatial.nc"),role="primary")],request=request,
                evaluation=dict(kind="declared_task_metric",metric="coverage"),budget=budget,seed=11)
    save(OUT/(name+"_protocol.json"),dict(config=config,modes=["A","B","C","C-gap","R"],purpose="configuration/real service wiring, not superiority"))
    results={}
    for mode in ("A","B","C","C-gap","R"):
        result=submit_file_request(dict(config,mode=mode),OUT/name/mode)
        results[mode]=dict(counts=result["counts"],selected=bool(result["best"]),stop_reason=result["stop_reason"],optimizer_error=result["optimizer_error"],candidate_domain=result["candidate_domain"])
    if len({json.dumps(r["candidate_domain"],sort_keys=True) for r in results.values()})!=1:raise ValueError("Strategy domains differ")
    save(OUT/(name+"_results.json"),results)


def protected_inputs():
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    rows=[]
    for file,sha in protocol["sources"].items():
        actual=hashlib.sha256(Path(file).read_bytes()).hexdigest()
        rows.append(dict(file=file,expected=sha,actual=actual,unchanged=actual==sha))
    # The previous delivery manifest is evidence, never rewritten.
    previous=ROOT/"output/platform_search_service_20261021/delivery_manifest.json"
    if previous.is_file():
        manifest=json.loads(previous.read_text(encoding="utf-8"))
        save(OUT/"prior_manifest_readback.json",dict(path=str(previous),sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),schema=list(manifest) if isinstance(manifest,dict) else "list",not_modified=True))
    if not all(row["unchanged"] for row in rows):raise ValueError("Protected input changed")
    save(OUT/"protected_inputs.json",rows)


def browser_readback():
    source=Path("C:/Users/LX/Downloads/platform_search.zip")
    target=OUT/"browser_download.zip"
    if not target.exists():shutil.copy2(source,target)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert {"summary.json","selected_graph.json","request.json","decision.json"}<=set(archive.namelist())
        summary=json.loads(archive.read("summary.json"));files=[n for n in archive.namelist() if n.endswith("product.nc")]
        products=[]
        for file in files:
            with xr.open_dataset(io.BytesIO(archive.read(file)),engine="h5netcdf") as ds:
                assert np.isfinite(ds.aod).all();products.append(dict(file=file,shape=dict(ds.sizes),unit=ds.aod.attrs.get("units")))
        save(OUT/"browser_acceptance.json",dict(status="PASS",browser="Actual in-app Chromium localhost upload/submit/download, not AppTest",url="http://127.0.0.1:8502/",
             screenshot=str(OUT/"browser_result.png"),download_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),products=products,
             counts=summary["counts"],selected=bool(summary["best"]),refresh="Visible Rerun retained result and counters; no new submission",scope="spatial case only; temporal/multisource paired handlers tested separately"))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["snapshot","parity","strategies","protection","browser"]);parser.add_argument("--suffix",default="final");parser.add_argument("--snapshot-name",default="source_delivery")
    parser.add_argument("--spatial-suffix");parser.add_argument("--output-name",default="entry_parity.json");parser.add_argument("--run-name",default="strategy_wiring")
    args=parser.parse_args()
    if args.action=="snapshot":source_snapshot(args.snapshot_name)
    elif args.action=="parity":parity(args.suffix,args.spatial_suffix,args.output_name)
    elif args.action=="strategies":strategy_wiring(args.run_name)
    elif args.action=="protection":protected_inputs()
    else:browser_readback()
