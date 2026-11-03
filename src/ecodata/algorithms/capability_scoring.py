"""Empirical optional-capability evidence, not hard safety or numerical fitness.

Only the service writes/loads its own local model store. Training records contain
public input features and independently evaluated development losses, not labels
for the inputs being searched. An unknown score is None, never zero benefit.
"""
from pathlib import Path
from time import perf_counter
import hashlib,json,re
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'output/capability_models'
FEATURES=('mean','std','range','gradient_mean','gradient_p95','flat_fraction','edge_concentration',
          'curvature_mean','extrema_fraction','max_median_ratio','unique_fraction','scale')
CONTEXT=('task_family','variable','unit','metric','metric_kind','capability','implementation','base_algorithm','support')
COMPARE=('input_fingerprint','queue_fingerprint','reference_fingerprint','information_fingerprint',
         'parameters_fingerprint','metric','metric_kind','loss_unit')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def paired_evidence(records, *, epsilon, loss_unit):
    """Conservative label gate. Changed stages must be explicitly attributable."""
    if not np.isfinite(epsilon) or epsilon<=0:raise ValueError('A positive unit-bearing epsilon must be frozen')
    accepted=[];excluded=[];seen=set()
    for r in records:
        reason=None;a=r.get('before',{});b=r.get('after',{})
        if r.get('split')!='development':reason='Only exposed development records are admitted'
        elif not r.get('group') or not r.get('comparison_id'):reason='Independent source group and comparison identity required'
        elif any(a.get(k)!=b.get(k) or a.get(k) is None for k in COMPARE):reason='Unmatched input/queue/reference/information/parameters/metric'
        elif a.get('loss_unit')!=loss_unit:reason='Loss unit does not match epsilon'
        elif a.get('metric_kind') not in {'reference_error','engineering'}:reason='Unknown metric family'
        elif any(x.get('status')!='PASS' or x.get('actual_execution') is not True or x.get('quality_pass') is not True for x in (a,b)):reason='Failure/incomplete/UNKNOWN/cache is not numerical evidence'
        elif r.get('attribution') not in {'single_capability','legal_alternative'}:reason='Composite or unproven attribution'
        elif len(r.get('changed_stages',[]))!=1:reason='Not a one-stage controlled contrast'
        elif not a.get('steps') or not b.get('steps') or a['steps'][:-1]!=b['steps'][:-1]:reason='Actual upstream algorithm paths differ'
        elif a['steps'][-1]!=r['changed_stages'][0].get('before') or b['steps'][-1]!=r['changed_stages'][0].get('after'):reason='Claimed change differs from executed steps'
        elif any(not isinstance(x.get('loss'),(int,float)) or not np.isfinite(x['loss']) or x['loss']<0 for x in (a,b)):reason='Loss must be finite and nonnegative, not a penalty'
        elif a['loss']<=epsilon:reason='Near-zero baseline loss; relative reward withheld'
        identity=digest({k:a.get(k) for k in COMPARE}|dict(stage=r.get('changed_stages')))
        if not reason and identity in seen:reason='Duplicate reference/input/queue/contrast'
        if reason:excluded.append(dict(id=r.get('id'),reason=reason));continue
        seen.add(identity)
        accepted.append(dict(r,reward=float((a['loss']-b['loss'])/max(a['loss'],epsilon)),epsilon=epsilon,epsilon_unit=loss_unit))
    return accepted,excluded


def _vector(features,context):
    if set(features)!=set(FEATURES) or any(not np.isfinite(features[k]) for k in FEATURES):
        raise ValueError('Missing/nonfinite observable score features')
    if any(k not in context for k in CONTEXT):raise ValueError('Incomplete benefit implementation/task context')
    return dict(features,**{k:str(context[k]) for k in CONTEXT})


class CapabilityScoringService:
    """Small local RF service; group-disjoint holdout and explicit scope gate."""
    @classmethod
    def train(cls,records,protocol,model_id):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        start=perf_counter()
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',model_id):raise ValueError('Invalid model ID')
        folder=ROOT/model_id
        if folder.exists():raise ValueError('Immutable model version already exists')
        accepted,excluded=paired_evidence(records,epsilon=protocol['epsilon'],loss_unit=protocol['loss_unit'])
        groups=sorted({r['group'] for r in accepted});held=set(protocol['holdout_groups'])
        train=[r for r in accepted if r['group'] not in held];test=[r for r in accepted if r['group'] in held]
        folder.mkdir(parents=True)
        report=dict(model_id=model_id,protocol=protocol,excluded=excluded,accepted=len(accepted),groups=groups,
                    training_groups=sorted({r['group'] for r in train}),holdout_groups=sorted({r['group'] for r in test}),
                    independent_unit='declared region group, repeated fields/scales/algorithms are clustered',
                    reliability='LIMITED_DEVELOPMENT_HOLDOUT; not calibrated, no confidence interval')
        if len(report['training_groups'])<2 or not test or not held<=set(groups):
            report.update(status='INSUFFICIENT_EVIDENCE',reason='Need >=2 training groups and >=1 disjoint held-out group')
        else:
            if any(r['context']['metric_kind']!=protocol['metric_kind'] or r['context']['metric']!=protocol['metric'] for r in accepted):raise ValueError('Metric families must be trained separately')
            vectorizer=DictVectorizer(sparse=False)
            x=vectorizer.fit_transform([_vector(r['features'],r['context']) for r in train]);y=np.array([r['reward'] for r in train])
            model=RandomForestRegressor(n_estimators=64,max_depth=4,min_samples_leaf=2,random_state=protocol['seed'],n_jobs=1).fit(x,y)
            predictions=model.predict(vectorizer.transform([_vector(r['features'],r['context']) for r in test]))
            mean=float(y.mean());ranking={}
            for r in train:ranking.setdefault(digest(r['context']),[]).append(r['reward'])
            ranked=np.array([np.mean(ranking.get(digest(r['context']),[mean])) for r in test])
            truth=np.array([r['reward'] for r in test]);regrets=[];global_regrets=[];constant_regrets=[];ranks=[]
            for source in sorted({r['comparison_id'] for r in test}):
                indices=[i for i,r in enumerate(test) if r['comparison_id']==source]
                actual=truth[indices];p=predictions[indices];g=ranked[indices]
                regrets.append(float(max(actual)-actual[np.argmax(p)]));global_regrets.append(float(max(actual)-actual[np.argmax(g)]))
                constant_regrets.append(float(max(actual)-actual[0]))
                ranks.append(dict(comparison_id=source,implementations=[test[i]['context']['base_algorithm'] for i in indices],
                    predicted_order=np.argsort(-p).tolist(),actual_order=np.argsort(-actual).tolist()))
            report.update(status='TRAINED_LIMITED',train_records=len(train),holdout_records=len(test),
                reward_MAE=float(np.abs(predictions-truth).mean()),constant_MAE=float(np.abs(mean-truth).mean()),
                global_rank_MAE=float(np.abs(ranked-truth).mean()),mean_selection_regret=float(np.mean(regrets)),
                global_rank_regret=float(np.mean(global_regrets)),constant_regret=float(np.mean(constant_regrets)),ranks=ranks,
                heldout=[dict(id=r['id'],group=r['group'],actual=r['reward'],prediction=float(p)) for r,p in zip(test,predictions)])
            payload=dict(model=model,vectorizer=vectorizer,contexts=[dict(t) for t in {tuple(sorted(r['context'].items())) for r in train}],
                bounds={k:[min(r['features'][k] for r in train),max(r['features'][k] for r in train)] for k in FEATURES})
            in_support=[r['context'] in payload['contexts'] and all(lo<=r['features'][k]<=hi for k,(lo,hi) in payload['bounds'].items()) for r in test]
            report.update(heldout_in_support=sum(in_support),heldout_support_coverage=float(np.mean(in_support)),
                heldout_metric_scope='raw RF diagnostic includes out-of-support rows; deployment abstains there',
                supported_reward_MAE=float(np.abs(predictions-truth)[in_support].mean()) if any(in_support) else None)
            path=folder/'model.joblib';joblib.dump(payload,path)
            report.update(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),training_records_fingerprint=digest(train))
        report.update(training_seconds=perf_counter()-start)
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/'evidence.json').write_text(json.dumps(dict(accepted=accepted,excluded=excluded),ensure_ascii=False,indent=2),encoding='utf-8')
        return report

    def __init__(self,model_id):
        import joblib
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',model_id):raise ValueError('Only a platform-trained model ID is allowed')
        self.report=json.loads((ROOT/model_id/'report.json').read_text(encoding='utf-8'));self.payload=None
        if self.report['status']=='TRAINED_LIMITED':
            path=ROOT/model_id/'model.joblib'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.report['model_sha256']:raise ValueError('Model fingerprint mismatch')
            self.payload=joblib.load(path)

    def score(self,features,context):
        start=perf_counter();base=dict(capability=context.get('capability'),implementation=context.get('implementation'),
            implementation_context=dict(context),
            estimated_relative_benefit=None,status='INSUFFICIENT_EVIDENCE',model_id=self.report['model_id'],
            training_support={k:self.report.get(k) for k in ('training_groups','holdout_groups','train_records','holdout_records','reliability')})
        reason='No trained model'
        if self.payload:
            if context not in self.payload['contexts']:reason='Unsupported capability/algorithm/task/metric context'
            elif set(features)!=set(FEATURES):reason='Missing observable features'
            elif any(not np.isfinite(features[k]) or not lo-1e-12<=features[k]<=hi+1e-12 for k,(lo,hi) in self.payload['bounds'].items()):reason='Out of observed training support; fallback, no extrapolated reward'
            else:
                prediction=self.payload['model'].predict(self.payload['vectorizer'].transform([_vector(features,context)]))[0]
                base.update(status='EVIDENCE_LIMITED',estimated_relative_benefit=float(prediction),model_sha256=self.report['model_sha256'])
                reason='Actual paired development losses; experimental, reliability not certified'
        return dict(base,reason=reason,inference_seconds=perf_counter()-start)


def optional_capability_matches(decision):
    """Public two-class ledger, also returned when no empirical model is selected."""
    ledger=dict(required=[dict(capability=c,status='REQUIRED',acceptance='output-path require_graph',
            local_requirements=[dict(id=p['id'],rule=p['rule'],port=p.get('binding',p.get('port')),source_ids=p.get('source_ids',p.get('source_roles'))) for p in decision.requirements.problems if c in p['capabilities']]) for c in decision.requirements.required],
        optional=[dict(capability=c,status='INSUFFICIENT_EVIDENCE',estimated_relative_benefit=None,
            problems=[p['id'] for p in decision.requirements.problems if c in p['capabilities']],
            candidates=[dict(algorithm=k,eligible=m.get('eligible'),resources_or_reasons=m.get('reasons',[]))
                        for k,m in sorted(decision.matches.items()) if c in m.get('capabilities',[])],
            reason='No empirical evidence for this particular input/implementation context yet')
            for c in decision.requirements.optional_capabilities if c not in decision.requirements.required])
    for item in ledger['optional']:
        scores=[s for s in getattr(decision,'benefit_manifest',{}).get('scores',[]) if s.get('capability')==item['capability']]
        if scores:
            item.update(implementation_scores=scores,status='EVIDENCE_LIMITED' if any(s.get('estimated_relative_benefit') is not None for s in scores) else 'INSUFFICIENT_EVIDENCE',
                reason='Implementation-conditional development estimates; no intrinsic capability-wide reward or certified reliability')
    return ledger


def configure_optional_scoring(decision,request,evaluation_protocol,context):
    """Public model loading and scope check; evaluation values never enter it."""
    spec=request.get('optional_scoring',{})
    if set(spec)-{'model_id','enabled','acknowledge_limited_development'}:raise ValueError('Unsupported scoring configuration')
    if not spec.get('enabled'):return
    if not spec.get('acknowledge_limited_development'):raise ValueError('Development-only benefit model requires explicit acknowledgement')
    if context is None or request['task']!='spatial' or len(context.sources)!=1:
        decision.benefit_manifest=dict(status='NOT_APPLICABLE',estimated_relative_benefit=None,reason='No supported single-raster context')
        return
    service=CapabilityScoringService(spec['model_id'])
    size=service.report['protocol'].get('fine_size')
    if size and (not request.get('nonnegative') or len(context.target_lat or ())!=size or len(context.target_lon or ())!=size):
        decision.benefit_manifest=dict(status='NOT_APPLICABLE',estimated_relative_benefit=None,
            reason='Outside recorded nonnegative square target-grid support',model_id=spec['model_id'])
        return
    decision.benefit_service=service
    from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
    decision.benefit_features=SemanticFeatureExtractor.raster_features(context.sources[0].dataset,context.variable,len(context.target_lat or ()))
    decision.benefit_context=dict(task_family='spatial_raster',variable=context.variable,unit=request['unit'],
        metric=evaluation_protocol.get('objective','UNKNOWN'),metric_kind=evaluation_protocol.get('kind','UNKNOWN'),
        capability='coarse_scale_consistency',implementation='raster_task.coarse_consistency',support=context.sources[0].dataset[context.variable].attrs.get('spatial_semantics','UNKNOWN'))
    decision.benefit_base_algorithms=tuple(sorted({c['base_algorithm'] for c in service.payload['contexts']})) if service.payload else ()
    decision.benefit_manifest=dict(status=service.report['status'],model_id=spec['model_id'],context=decision.benefit_context,
        features=decision.benefit_features,training_support={k:service.report.get(k) for k in ('training_groups','holdout_groups','train_records','holdout_records','reliability')},
        scores=[service.score(decision.benefit_features,dict(decision.benefit_context,base_algorithm=b)) for b in decision.benefit_base_algorithms],
        final_fitness='independent real execution only',evaluator_values_received=False)


def empirical_proposal_weights(decision,factory,candidates,weights,trace):
    """Only C-local opts in. Missing/OOD evidence leaves old weights untouched."""
    service=getattr(decision,'benefit_service',None);context=getattr(decision,'benefit_context',{})
    details=[];factors=[]
    for graph in candidates:
        nodes=list(graph.nodes);bases=[n.name for n in nodes if n.name in getattr(decision,'benefit_base_algorithms',())]
        has_projection=any(n.name=='raster_task.coarse_consistency' for n in nodes)
        score=dict(status='INSUFFICIENT_EVIDENCE',estimated_relative_benefit=None,reason='No single supported implementation context')
        if service and len(bases)==1 and context.get('capability') in decision.requirements.optional_capabilities and context.get('capability') not in decision.requirements.required:
            score=service.score(getattr(decision,'benefit_features',{}),dict(context,base_algorithm=bases[0]))
        # Absence is the paired baseline, not a fabricated zero-benefit label.
        factor=float(np.exp(np.clip(score['estimated_relative_benefit'],-1,1))) if has_projection and score['estimated_relative_benefit'] is not None else 1.
        details.append(dict(score,applied_to_optional_stage=has_projection,factor=factor));factors.append(factor)
    adjusted=np.asarray(weights)*factors
    # Explicit mixture preserves exploration, independent of predicted winner.
    p=adjusted/adjusted.sum();p=.9*p+.1/len(p)
    if not service or all(f==1 for f in factors):p=np.asarray(weights)/sum(weights)
    trace.update(empirical_optional=dict(enabled=service is not None,formula='0.9*normalize(w*exp(clip(r,-1,1))*I_optional_or_1)+0.1/n',
        before_weights=list(weights),details=details),weights=p.tolist(),probabilities=p.tolist())
    return p.tolist(),trace
