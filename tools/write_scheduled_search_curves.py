"""Export actual execution trajectories; no synthetic interpolation points."""
from pathlib import Path
import csv,json,hashlib,shutil,sys
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/scheduled_fusion_20261025'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    protocol=load(OUT/'frozen_protocol.json')
    with (OUT/'search_curves.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case','method','seed','unique_execution','origin','status','validation_rmse','best_so_far','oracle_rmse','gap','hit_oracle','completed_seconds','kernel_calls'])
        writer.writeheader()
        for case,_ in protocol['validation_test_pairs']:
            optimum=load(OUT/'oracles'/str(case)/'none/11/summary.json')['final_validation_rmse']
            for method in protocol['methods']:
                for seed in protocol['seeds']:
                    report=load(OUT/'search'/str(case)/method/str(seed)/'summary.json');best=None
                    for i,row in enumerate(report['rows'],1):
                        value=row.get('search_validation',{}).get('RMSE')
                        if row['status']=='PASS':best=value if best is None else min(best,value)
                        writer.writerow(dict(case=case,method=method,seed=seed,unique_execution=i,origin=row['origin'],status=row['status'],validation_rmse=value,best_so_far=best,oracle_rmse=optimum,gap=None if best is None else best-optimum,hit_oracle=best is not None and abs(best-optimum)<=1e-9,completed_seconds=row['completed_seconds'],kernel_calls=sum(t.get('kernel_calls',0)+int('roles' in t) for t in row.get('trace',[]))))
    paths=[ROOT/'src/ecodata/algorithms'/name for name in ('composition_decision.py','composition_optimizers.py','platform_search.py','spatial_median.py','spatiotemporal_detail.py')]
    paths += [ROOT/'src/ecodata/interface/search_page.py']
    paths += [ROOT/'tools'/name for name in ('scheduled_fusion_data.py','run_scheduled_fusion_study.py','audit_scheduled_fusion_study.py','accept_scheduled_delivery.py','verify_boundary_release.py')]
    manifest=[]
    for path in paths:
        target=OUT/'source_release'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==path.read_bytes()
        else:shutil.copy2(path,target)
        manifest.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'release_manifest.json').write_text(json.dumps(dict(source=manifest,python=sys.version,packages={p:version(p) for p in ('numpy','scipy','xarray','h5netcdf','alns','pymoo')}),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',curve=str(OUT/'search_curves.csv'),sources=len(manifest))))
if __name__=='__main__':main()
