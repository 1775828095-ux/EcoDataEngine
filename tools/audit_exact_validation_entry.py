"""Supplementary full-coverage CLI certificate; excluded from efficiency runs."""
from pathlib import Path
import json,hashlib
import numpy as np
import xarray as xr
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/scheduled_fusion_20261025'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    folder=OUT/'delivery/exact_validation_entry';report=load(folder/'summary.json');request=load(folder/'request.json')
    assert report['counts']['evaluated']==report['counts']['usable_observed']==1296
    assert report['certification']=='EXACT_VALIDATION_RMSE_OPTIMUM_IN_DECLARED_DOMAIN'
    assert report['native_optimizer']['package']=='alns' and report['exact_audit_additional_executions']>0
    path=Path(request['search_validation']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==request['search_validation']['sha256']
    with xr.open_dataset(path,engine='h5netcdf') as ds:truth=ds.air_temperature.load()
    assert truth.attrs['split']=='search_validation';errors={}
    for row in report['rows']:
        path=Path(row['product']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['product_sha256']
        with xr.open_dataset(path,engine='h5netcdf') as ds:
            assert ds.air_temperature.shape==truth.shape
            for axis in truth.coords:np.testing.assert_array_equal(ds[axis],truth[axis])
            error=float(np.sqrt(np.mean((ds.air_temperature.values-truth.values)**2)))
        assert abs(error-row['search_validation']['RMSE'])<1e-9
        errors[row['key']]=error
    assert abs(errors[report['recommended_key']]-min(errors.values()))<1e-9
    oracle=load(OUT/'oracles/1510/none/11/summary.json')
    assert abs(report['final_validation_rmse']-oracle['final_validation_rmse'])<1e-9
    receipt=dict(status='PASS',real_products_independently_checked=len(errors),validation_minimum_K=min(errors.values()),
        certification=report['certification'],native_optimizer='alns',exact_audit_additional_executions=report['exact_audit_additional_executions'],
        budget_search_comparison_unchanged=True,scope='Supplementary certificate acceptance only; not an extra efficiency replicate or final-test optimum')
    (OUT/'delivery/exact_validation_acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    assurance=load(OUT/'final_assurance.json');assurance['full_coverage_cli_verified']=receipt
    (OUT/'final_assurance.json').write_text(json.dumps(assurance,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
