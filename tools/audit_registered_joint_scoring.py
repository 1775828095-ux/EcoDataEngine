"""Independent replay of saved rule arithmetic, delivery policy and products."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


def audit(roots):
    checked=[];failures=[];products=set()
    for path in sorted({p for root in roots for p in Path(root).rglob('summary.json')}):
        try:
            report=json.loads(path.read_text(encoding='utf-8'))
            if report.get('scoring_policy')!='joint_portrait_v1':raise ValueError('Wrong selection policy')
            tables=report['portraits'];model=tables['model'];registry=model['attribute_registry']
            snapshot={k:v for k,v in registry.items() if k!='sha256'}
            digest=hashlib.sha256(json.dumps(snapshot,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            if digest!=registry['sha256']:raise ValueError('Registry content differs from its saved digest')
            definitions={a['id']:a for a in registry['attributes']}
            ranked=json.loads((path.parent/'candidate_ranking.json').read_text(encoding='utf-8'))['candidates']
            ordering=[]
            for row in ranked:
                lower=upper=0.;has_unknown=False
                for term in row['matching']['terms']:
                    key=term['dimension']
                    if term['definition']!=definitions[key]:raise ValueError('Scoring term definition mismatch')
                    if term['weight']!=model['matching_requirements']['priorities'][key]:raise ValueError('User priority silently changed')
                    settings=model['scoring_parameters'].get(key,{})
                    a=term['raw_activation'];weight=term['weight']*settings.get('multiplier',1.)
                    if a is not None and a<settings.get('activation_threshold',0.):a=0.
                    if not definitions[key]['scoring_enabled'] or weight==0 or a==0:
                        lo=hi=0.
                    elif a is None or (term['guarantee'] is None and term.get('utility') is None):
                        lo=-weight*(1 if a is None else a);hi=-lo;has_unknown=True
                        if term['contribution'] is not None:raise ValueError('Unknown attribute has fabricated point contribution')
                    else:
                        utility=term.get('utility')
                        if utility is None:utility=1 if term['guarantee'] else -1
                        lo=hi=weight*a*utility
                    np.testing.assert_allclose(term['interval'],[lo,hi],atol=1e-12,rtol=0)
                    lower+=lo;upper+=hi
                np.testing.assert_allclose(row['matching']['score_interval'],[lower,upper],atol=1e-12,rtol=0)
                if has_unknown and row['matching']['score'] is not None:raise ValueError('Unknown total score fabricated')
                for hard in model['matching_requirements']['hard']:
                    if row['matching']['properties'].get(definitions[hard]['property']) is not True:
                        raise ValueError('Unknown/failed hard requirement admitted')
                ordering.append((-lower,row['declared_cost'],len(row['steps']),row['key']))
            if ordering!=sorted(ordering):raise ValueError('Another sorter displaced joint score ordering')
            if ranked and report['recommended_key']!=ranked[0]['key']:raise ValueError('Recommendation does not match joint ranking')
            variable=tables['data'][0]['semantics']['variable']
            valid=[]
            for row in report['rows']:
                if row['status']!='PASS':continue
                product=Path(row['product'])
                if hashlib.sha256(product.read_bytes()).hexdigest()!=row['product_sha256']:raise ValueError('Product fingerprint mismatch')
                with xr.open_dataset(product) as data:
                    if not np.isfinite(data[variable]).all() or not data[variable].attrs.get('units'):raise ValueError('Invalid numeric output')
                products.add(str(product))
                actual=json.loads((product.parent/'result.json').read_text(encoding='utf-8'))
                if any(actual.get(k)!=row.get(k) for k in ('objective','RMSE','MAE','Bias')):raise ValueError('Evaluation record mismatch')
                if row.get('task_loss') is not None:valid.append(row)
            if valid and report['best']['key']!=valid[0]['key']:raise ValueError('Delivery substituted hindsight minimum for the rule choice')
            if report['learning_status']['affects_selection'] is not False:raise ValueError('Legacy learner affects selection')
            checked.append(dict(summary=str(path),candidates=len(ranked),products=len(valid),recommended_key=report['recommended_key']))
        except Exception as exc:
            failures.append(dict(summary=str(path),error=str(exc)))
    return dict(status='PASS' if checked and not failures else 'FAIL',summaries_verified=len(checked),products_verified=len(products),
                failures=failures,records=checked,scope='synthetic functional acceptance only; no independent environmental benefit claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('roots',nargs='+');parser.add_argument('--out',required=True)
    args=parser.parse_args();report=audit(args.roots)
    Path(args.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))
    raise SystemExit(0 if report['status']=='PASS' else 1)
