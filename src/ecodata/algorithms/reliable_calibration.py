"""Development-only practical-tolerance calibration from real diagnostic banks."""
from pathlib import Path
import json
import re

import numpy as np
import xarray as xr

from ecodata.algorithms import reliable_selection as selection

ROOT = Path(__file__).resolve().parents[3] / 'output/reliable_selection_calibrations'
VERSION = 'reliable_tolerance_calibration_v1'


def _load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()


def _json(value):return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)


def _write_immutable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=_json(value)
    if path.exists():
        if path.read_text(encoding='utf-8')!=content:
            raise ValueError('Immutable calibration artifact already exists with different content: '+str(path))
    else:path.write_text(content,encoding='utf-8')


def _read_run(item):
    folder=Path(item['folder']);group=item.get('group')
    request_record=_load(folder/'request.json');request=request_record.get('request',{})
    if request.get('evidence_role')!='development' or not group or request.get('group')!=group:
        raise ValueError('Calibration requires matching explicit development group declarations')
    if request.get('variable')!='ndvi' or request.get('unit')!='1':
        raise ValueError('Current tolerance publication is scoped to NDVI unit1')
    summary=_load(folder/'summary.json');policy=summary.get('reliable_selection')
    if not isinstance(policy,dict) or not policy.get('weights') or not policy.get('scales'):
        raise ValueError('Diagnostic bank needs a recorded normalized reliable-selection loss')
    reference_path=Path(item['reference']);variable=request['variable']
    with xr.open_dataset(reference_path) as f:
        truth=f[variable].load()
    roles={truth.attrs.get(name) for name in ('split','evidence_role') if truth.attrs.get(name) is not None}
    if not roles or roles!={'development'}:
        raise ValueError('Reference must be explicitly development, never final-test or diagnostic labels')
    if truth.dims!=('time','lat','lon'):
        raise ValueError('Timed raster development reference required')
    support=np.isfinite(truth.values)
    if not support.reshape(len(truth.time),-1).any(1).all():
        raise ValueError('Every development reference date needs finite common support')
    rows=[];seen=set()
    for original in summary.get('bank_rows',[]):
        identifier=original.get('bank_id')
        if not identifier or identifier in seen:raise ValueError('Unique diagnostic bank IDs required')
        seen.add(identifier)
        record={name:original.get(name) for name in ('bank_id','key','status','validation_status','selection_eligible','algorithm_nodes','error')}
        if original.get('status')=='PASS':
            product=Path(original['product'])
            digest=_sha(product)
            if original.get('product_sha256') not in {None,digest}:
                raise ValueError('Diagnostic product fingerprint mismatch')
            with xr.open_dataset(product) as f:
                predicted=f[variable].sel(time=truth.time,lat=truth.lat,lon=truth.lon).load()
            if predicted.dims!=truth.dims or predicted.attrs.get('units')!=truth.attrs.get('units'):
                raise ValueError('Development prediction/reference physical support differs')
            values=predicted.values.astype(float);expected=truth.values.astype(float)
            if not np.isfinite(values[support]).all():
                raise ValueError('Incomplete development prediction; no successful reference subset')
            per_date=[]
            for index,(pred,ref,mask) in enumerate(zip(values,expected,support)):
                error=pred[mask]-ref[mask]
                per_date.append(dict(fold_id=str(index),status='PASS',metrics=dict(
                    RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.mean(abs(error))),
                    Bias=float(np.mean(error)),denominator=int(mask.sum()))))
            target_policy=dict(policy);target_policy.pop('objective_fold_ids',None)
            actual=selection.aggregate(per_date,target_policy,[str(i) for i in range(len(per_date))])
            record.update(product=str(product),product_sha256=digest,actual_loss=actual['objective'],
                actual_metrics={name:actual[name] for name in ('RMSE','MAE','Bias','valid_n')})
        metrics=original.get('metrics')
        if original.get('selection_eligible') and isinstance(metrics,dict):
            loss=metrics.get('objective')
            if not isinstance(loss,(int,float)) or isinstance(loss,bool) or not np.isfinite(loss) or loss<0:
                raise ValueError('Eligible bank graph lacks a finite measured validation loss')
            record['metrics']=dict(objective=float(loss))
        rows.append(record)
    if len(rows)<2 or not any(r.get('actual_loss') is not None for r in rows):
        raise ValueError('Comparable real multi-graph bank required')
    if not any(r.get('metrics') and r.get('actual_loss') is not None for r in rows):
        raise ValueError('No selection-eligible graph deploys on this development case')
    return dict(folder=str(folder),group=group,rows=rows,policy=policy,
        reference=str(reference_path),reference_sha256=_sha(reference_path),
        request_sha256=_sha(folder/'request.json'),summary_sha256=_sha(folder/'summary.json'),
        oracle_loss=min(r['actual_loss'] for r in rows if r.get('actual_loss') is not None))


def _grid(runs):
    differences=[]
    for run in runs:
        losses=[r['metrics']['objective'] for r in run['rows'] if r.get('metrics')]
        minimum=min(losses);differences.extend(value-minimum for value in losses)
    return sorted({0.,*[float(np.quantile(differences,q)) for q in (.1,.25,.5)]})


def _evaluate(runs,delta):
    cases=[]
    for run in runs:
        policy=dict(run['policy'],delta=float(delta),parsimony_enabled=True)
        candidates=[dict(r,status='ELIGIBLE_OUTPUT') for r in run['rows'] if r.get('metrics')]
        ranked=selection.ordered(candidates,policy)
        rejected=[];chosen=None
        for row in ranked:
            if row.get('actual_loss') is not None:
                chosen=row;break
            rejected.append(row['bank_id'])
        if chosen is None:raise ValueError('No deployable development graph in measured ranking')
        cases.append(dict(folder=run['folder'],group=run['group'],bank_id=chosen['bank_id'],
            actual_loss=chosen['actual_loss'],oracle_loss=run['oracle_loss'],
            regret=max(0.,chosen['actual_loss']-run['oracle_loss']),algorithm_nodes=chosen['algorithm_nodes'],
            rejected_deployments=rejected))
    groups={group:float(np.mean([r['regret'] for r in cases if r['group']==group])) for group in sorted({r['group'] for r in cases})}
    return dict(delta=float(delta),mean_group_regret=float(np.mean(list(groups.values()))),
        worst_case_regret=max(r['regret'] for r in cases),
        mean_nodes=float(np.mean([r['algorithm_nodes'] for r in cases])),groups=groups,cases=cases)


def _fit(runs):
    trials=[_evaluate(runs,delta) for delta in _grid(runs)]
    return min(trials,key=lambda r:(r['mean_group_regret'],r['mean_nodes'],r['delta'])),trials


def calibrate(development_runs,calibration_id,destination=None):
    """Publish a development-derived tolerance only with held-group support.

    All numerical products are read back.  Unsuccessful bank graphs remain in
    the audit; deployment failures are simulated in the original ranked order.
    A nonzero tolerance needs lower held-group mean regret without worsening
    any held group's regret beyond the existing numerical roundoff tolerance.
    """
    if not isinstance(calibration_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',calibration_id):
        raise ValueError('Safe calibration ID required')
    destination=Path(destination) if destination is not None else selection.POLICY
    if destination.exists() and _load(destination).get('calibration_id')!=calibration_id:
        raise ValueError('Cannot overwrite another published calibration ID')
    runs=[_read_run(item) for item in development_runs]
    if not runs:raise ValueError('Development runs required')
    signatures={selection._digest(run['policy']['weights']) for run in runs}
    if len(signatures)!=1:raise ValueError('Different model/application objectives cannot share a tolerance')
    refs={}
    for run in runs:
        prior=refs.setdefault(run['reference_sha256'],run['group'])
        if prior!=run['group']:raise ValueError('Same development reference cannot cross independent groups')
    groups=sorted({run['group'] for run in runs});held=[]
    for group in groups if len(groups)>1 else []:
        train=[r for r in runs if r['group']!=group];test=[r for r in runs if r['group']==group]
        fitted,trials=_fit(train);baseline=_evaluate(test,0.);outcome=_evaluate(test,fitted['delta'])
        held.append(dict(group=group,training_groups=sorted({r['group'] for r in train}),
            training_trials=trials,selected_delta=fitted['delta'],baseline=baseline,outcome=outcome,
            regret_improvement=baseline['mean_group_regret']-outcome['mean_group_regret']))
    roundoff=max(float(run['policy'].get('roundoff_tolerance',0.)) for run in runs)
    supported=bool(held) and all(r['regret_improvement']>=-roundoff for r in held) and any(r['regret_improvement']>roundoff for r in held)
    fitted,trials=_fit(runs)
    delta=fitted['delta'] if supported else 0.
    status='LIMITED_DEVELOPMENT_IMPROVEMENT' if supported and delta>0 else 'NO_HELD_GROUP_GAIN_ZERO_TOLERANCE'
    if len(groups)<2:status='INSUFFICIENT_GROUPS_ZERO_TOLERANCE'
    common_ids=set.intersection(*[{r['bank_id'] for r in run['rows'] if r.get('actual_loss') is not None} for run in runs])
    common=None
    if common_ids:
        def baseline_rank(identifier):
            group_losses=[float(np.mean([next(r['actual_loss'] for r in run['rows'] if r['bank_id']==identifier)
                for run in runs if run['group']==group])) for group in groups]
            return float(np.mean(group_losses)),identifier
        identifier=min(common_ids,key=baseline_rank)
        chosen=next(r for r in runs[0]['rows'] if r['bank_id']==identifier)
        records=_load(Path(runs[0]['folder'])/'graphs.json');subset={}
        def collect(key):
            if key in subset:return
            for parent in records[key]['parents']:collect(parent)
            subset[key]=records[key]
        collect(chosen['key'])
        common=dict(records=subset,key=chosen['key'],bank_id=identifier,
            evidence='Frozen strongest common deployable development-bank graph by equal-group normalized loss; never injected into search candidates')
    report=dict(version=VERSION,calibration_id=calibration_id,evidence_role='development',status=status,
        scope=dict(variable='ndvi',unit='1'),metric_weights=runs[0]['policy']['weights'],
        groups=groups,held_group_audit=held,full_fit_trials=trials,published_delta=delta,
        baseline=_evaluate(runs,0.),published_selection=_evaluate(runs,delta),development_runs=runs,
        limitations=['Small grouped development evidence, not a confidence bound or generalization guarantee',
            'Oracle is only the actual successful diagnostic bank, never open-space global optimum',
            'Raw model losses and failed bank rows are retained; test/diagnostic labels are rejected'])
    report_root=ROOT if destination==selection.POLICY else destination.parent/'calibration_reports'
    report_path=report_root/(calibration_id+'.json')
    _write_immutable(report_path,report)
    policy=dict(version=VERSION,calibration_id=calibration_id,scope=report['scope'],metric_weights=report['metric_weights'],
        evidence_role='development',evidence='Grouped real development-bank readback; '+status,
        delta=delta,status=status,report=str(report_path),report_sha256=_sha(report_path),common_reference_graph=common)
    _write_immutable(destination,policy)
    return dict(policy=policy,report=report,policy_path=str(destination),report_path=str(report_path))
