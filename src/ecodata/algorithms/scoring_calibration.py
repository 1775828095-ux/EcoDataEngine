"""Calibrate the existing joint-rule parameters from verified development runs.

No new selector, graph execution, or test-label access occurs during fitting.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np

from ecodata.algorithms import attribute_registry as attributes
from ecodata.algorithms.portrait_rules import parameterized_terms


ROOT=Path(__file__).resolve().parents[3]/"output/score_calibrations"


def scope(tables):
    model=tables['model']
    value=dict(model={k:model[k] for k in ('model_id','version','selected_role','selected_field')},
        requirements=model['matching_requirements'],preferences=model['application_preferences'],
        task_context=tables['task_context'],pool=tables['candidate_pool']['fingerprint'],
        registry_sha256=model['attribute_registry']['sha256'])
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def resolve(calibration_id,tables):
    calibration_id=attributes.identifier(calibration_id)
    envelope=json.loads((ROOT/(calibration_id+'.json')).read_text(encoding='utf-8'))
    if set(envelope)!={'report','sha256'} or attributes.digest(envelope['report'])!=envelope['sha256']:
        raise ValueError('Scoring calibration fingerprint mismatch')
    report=envelope['report']
    if report.get('schema_version')!=1 or report.get('calibration_id')!=calibration_id:
        raise ValueError('Scoring calibration version mismatch')
    if report['scope']!=scope(tables):
        raise ValueError('Calibration context differs: freeze registry, pool, model requirements and metric preferences')
    params=attributes.parameters(report['parameters'],tables['model']['attribute_registry'])
    return params,dict(calibration_id=calibration_id,sha256=envelope['sha256'],status='LIMITED_DEVELOPMENT',
        affects='joint rule parameters only; hard constraints and user preferences unchanged',
        holdout=report['holdout'],scope='same declared context; no statistical generalization guarantee')


def _runs(folders,registry):
    from ecodata.algorithms.profile_matching import fingerprint_bytes,task_loss,objective_semantics,VERSION
    runs=[];source_groups={};seen=set();expected=None;preferences=None;semantics=None;base=None
    for folder in folders:
        path=Path(folder)/'summary.json'
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('profile_version')!=VERSION or report.get('evidence_role')!='development' or not report.get('group') or report.get('stop_reason')!='CANDIDATE_POOL_COMPLETED':
            raise ValueError('Calibration needs explicit complete grouped development runs, never final-test evidence')
        tables=deepcopy(report['portraits'])
        recorded=tables['model'].get('attribute_registry')
        if recorded is not None and recorded['sha256']!=registry['sha256']:
            raise ValueError('Development attribute registry differs')
        if recorded is None and registry['version']!='builtin-v1':
            raise ValueError('Legacy evidence cannot fill new attributes')
        tables['model']['attribute_registry']=registry
        rows=report['rows']
        if len(rows)<2 or len(rows)!=report['counts']['candidate_count']:
            raise ValueError('Complete comparable candidate coverage required')
        this_scope=scope(tables)
        this_preferences=tables['model']['application_preferences']
        if 'objective' in this_preferences['weights']:
            raise ValueError('Calibration requires explicit RMSE/MAE/abs_bias preferences; generic objective is ambiguous online')
        this_semantics=objective_semantics(report['evaluation_protocol'])
        if this_semantics is None:raise ValueError('No executable numerical loss semantics')
        this_base=tables['model'].get('scoring_parameters',{})
        if expected is not None and (this_scope!=expected or this_semantics!=semantics or this_base!=base):
            raise ValueError('Incompatible registry, candidate pool, objective, requirements or base parameters')
        expected=this_scope;preferences=this_preferences;semantics=this_semantics;base=this_base
        identities=[s.get('scientific_content_hash',s['content_hash']) for s in tables['data']]
        for identity in identities:
            if identity in source_groups and source_groups[identity]!=report['group']:
                raise ValueError('Same source content cannot cross development groups')
            source_groups[identity]=report['group']
        identity=attributes.digest(dict(data=identities,scope=this_scope,candidates=sorted(r['key'] for r in rows)))
        if identity in seen:continue
        seen.add(identity)
        definitions={a['id']:a for a in registry['attributes']}
        costs={a['id']:a['declared_cost'] for a in tables['algorithms']}
        for row in rows:
            if row['status']!='PASS' or row.get('file_readback')!='PASS' or not row.get('numerical_execution_attempted'):
                raise ValueError('Calibration requires successful real execution/readback for every candidate')
            product_path=Path(row['product'])
            if fingerprint_bytes(product_path)!=row.get('product_sha256'):
                raise ValueError('Calibration product fingerprint mismatch')
            executed=json.loads((product_path.parent/'result.json').read_text(encoding='utf-8'))
            if any(row.get(k)!=executed.get(k) for k in ('objective','RMSE','MAE','Bias')):
                raise ValueError('Calibration labels differ from numerical execution')
            if row.get('task_loss') is None or task_loss(row,preferences)!=row['task_loss']:
                raise ValueError('Calibration loss differs from declared metric preferences')
            if row['matching'].get('hard_status')!='PASS':raise ValueError('Calibration cannot waive hard constraints')
            terms=row['matching']['terms']
            for term in terms:
                if term['dimension'] not in definitions:raise ValueError('Unregistered calibration term')
                if 'definition' not in term:
                    term['definition']=definitions[term['dimension']]
                    term['raw_activation']=term['activation'];term['utility']=None
                if term['definition']!=definitions[term['dimension']]:raise ValueError('Attribute definition differs from recorded term')
            row['calibration_cost']=sum(costs[s] for s in row['steps'])
        runs.append(dict(group=report['group'],rows=rows,summary=str(path),summary_sha256=fingerprint_bytes(path)))
    return runs,expected,base


def _evaluate(runs,parameters):
    groups={};chosen=[]
    for run in runs:
        ranked=[]
        for row in run['rows']:
            _,lower,upper=parameterized_terms(row['matching']['terms'],parameters)
            ranked.append(((-lower,row['calibration_cost'],len(row['steps']),row['key']),row))
        winner=min(ranked,key=lambda item:item[0])[1]
        regret=winner['task_loss']-min(r['task_loss'] for r in run['rows'])
        groups.setdefault(run['group'],[]).append(regret)
        chosen.append(dict(summary=run['summary'],key=winner['key'],regret=regret))
    return dict(mean_group_regret=float(np.mean([np.mean(values) for values in groups.values()])),
                groups=sorted(groups),runs=len(runs),choices=chosen)


def calibrate(folders,calibration_id,*,holdout_groups,parameter_grid,registry_version=None):
    calibration_id=attributes.identifier(calibration_id)
    destination=ROOT/(calibration_id+'.json')
    if destination.exists():raise ValueError('Scoring calibration is immutable')
    registry=attributes.load(registry_version)
    runs,context,base=_runs(folders,registry)
    held=set(holdout_groups)
    train=[r for r in runs if r['group'] not in held];test=[r for r in runs if r['group'] in held]
    if not held or len({r['group'] for r in train})<2 or {r['group'] for r in test}!=held:
        raise ValueError('Need two development training groups and disjoint holdout groups')
    if not isinstance(parameter_grid,dict) or not 1<=len(parameter_grid)<=6:raise ValueError('Use 1..6 bounded, interpretable parameter axes')
    names=sorted(parameter_grid);domains=[];combinations=1
    for name in names:
        parts=name.split('.')
        if len(parts)!=2:raise ValueError('Parameter axes use attribute.parameter')
        key,param=parts
        if context['requirements']['priorities'].get(key,0)<=0:raise ValueError('Cannot tune an inactive/unrequested score term')
        definition=next(a for a in registry['attributes'] if a['id']==key)
        if not definition['scoring_enabled'] or definition['provider']=='unavailable':raise ValueError('Cannot learn missing or disabled measurements')
        observed=[t for run in train for row in run['rows'] for t in row['matching']['terms']
                  if t['dimension']==key and t['raw_activation'] is not None and (t['guarantee'] is not None or t.get('utility') is not None)]
        if not observed:raise ValueError('Cannot calibrate an attribute without established/observed support')
        values=parameter_grid[name]
        if not isinstance(values,list) or not 1<=len(values)<=16:raise ValueError('Bounded parameter lists required')
        for value in values:attributes.parameters({key:{param:value}},registry)
        domains.append(values);combinations*=len(values)
    if combinations>256:raise ValueError('Calibration trial budget exceeds 256; narrow parameter grid')
    candidates=[deepcopy(base)]
    for values in product(*domains):
        candidate=deepcopy(base)
        for name,value in zip(names,values):
            key,param=name.split('.');candidate.setdefault(key,{})[param]=float(value)
        candidates.append(candidate)
    trials=[];seen=set()
    for candidate in candidates:
        identity=attributes.digest(candidate)
        if identity in seen:continue
        seen.add(identity)
        measured=_evaluate(train,candidate)
        distance=sum((candidate.get(name.split('.')[0],{}).get(name.split('.')[1],attributes.PARAMETERS[name.split('.')[1]][2])-
                      base.get(name.split('.')[0],{}).get(name.split('.')[1],attributes.PARAMETERS[name.split('.')[1]][2]))**2 for name in names)
        trials.append(dict(parameters=candidate,training=measured,distance_from_initial=distance))
    selected=min(trials,key=lambda t:(t['training']['mean_group_regret'],t['distance_from_initial'],attributes.digest(t['parameters'])))
    diagnostics={}
    for key in context['requirements']['priorities']:
        terms=[t for run in train for row in run['rows'] for t in row['matching']['terms'] if t['dimension']==key]
        measured=[t['raw_activation'] for t in terms if t['raw_activation'] is not None]
        variances={}
        for label,settings in [('initial',base),('calibrated',selected['parameters'])]:
            values=[]
            for run in train:
                points=[t['contribution'] for row in run['rows'] for t in parameterized_terms(row['matching']['terms'],settings)[0]
                        if t['dimension']==key and t['contribution'] is not None]
                if len(points)>=2:values.append(float(np.var(points)))
            variances[label]=float(np.mean(values)) if values else None
        diagnostics[key]=dict(observations=len(terms),unknown_guarantees=sum(t['guarantee'] is None and t.get('utility') is None for t in terms),
                              activation_variance=float(np.var(measured)) if measured else None,
                              within_task_contribution_variance=variances,
                              action='diagnostic only; no automatic attribute deletion or hard-rule changes')
    report=dict(schema_version=1,calibration_id=calibration_id,status='LIMITED_DEVELOPMENT',scope=context,
        parameters=selected['parameters'],initial_parameters=base,parameters_changed=selected['parameters']!=base,parameter_grid=parameter_grid,trials=trials,
        training=selected['training'],holdout=_evaluate(test,selected['parameters']),initial_holdout=_evaluate(test,base),
        attribute_diagnostics=diagnostics,
        evidence=[{k:r[k] for k in ('group','summary','summary_sha256')} for r in runs],
        selection='training group mean regret; smallest parameter change breaks ties; holdout never selects parameters',
        real_executions_during_calibration=0,hard_constraints_changed=False,user_preferences_changed=False)
    ROOT.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(dict(report=report,sha256=attributes.digest(report)),stream,ensure_ascii=False,indent=2,allow_nan=False)
    return report
