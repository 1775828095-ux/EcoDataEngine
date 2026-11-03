"""Model-independent evidence assessment; never changes algorithms or scores.

Graph membership is structural evidence, not proof of numerical influence or
accuracy. Unknown graph protocols and missing reference evidence stay unknown.
"""
from pathlib import Path,PurePosixPath
import hashlib,json

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def graph_sources(graph, inputs, self_reference):
    records=graph.get('records',{}) if isinstance(graph,dict) else {}
    root=graph.get('key') if isinstance(graph,dict) else None
    if not root or root not in records:return dict(status='UNKNOWN_GRAPH_PROTOCOL',indices=[],operations=[])
    seen=set();active=set();indices=set();operations=[];issues=[]
    fields=[(k,i) for k,i in enumerate(inputs) if i.get('role') in {'primary','background'} and i.get('_spatial_support')!='point']
    def walk(key):
        if key in active:raise ValueError('Selected graph contains a cycle')
        if key in seen:return
        if key not in records:raise ValueError('Selected graph parent is missing')
        active.add(key);node=records[key];op=node.get('operation');operations.append(op)
        for parent in node.get('parents',[]):walk(parent)
        if op=='spatial.source':
            index=node.get('parameters',{}).get('index')
            if type(index)!=int or not 0<=index<len(fields):raise ValueError('Invalid source index')
            if fields[index][1].get('_spatial_support') not in {'grid_estimate','cell_mean'}:
                issues.append('Grid-source support could not be verified')
            indices.add(fields[index][0])
        elif op=='spatial.observations':
            if self_reference is True:
                primary=[k for k,s in fields if s.get('role')=='primary']
                if len(primary)!=1:raise ValueError('Self reference requires one identified primary')
                indices.add(primary[0])
            elif self_reference is False:
                anchors=[k for k,i in enumerate(inputs) if i.get('role')=='observations' or i.get('_spatial_support')=='point']
                if not anchors:issues.append('Conditioning observations are not traced to input files')
                indices.update(anchors)
            else:issues.append('Observation origin is not identified')
        elif op=='spatial.auxiliary':
            auxiliary=[k for k,i in enumerate(inputs) if i.get('role')=='auxiliary']
            index=node.get('parameters',{}).get('index')
            if type(index)!=int or not 0<=index<len(auxiliary):raise ValueError('Invalid auxiliary index')
            indices.add(auxiliary[index])
        elif not isinstance(op,str) or not op.startswith('spatial.'):
            issues.append('Unrecognised source semantics: '+str(op))
        active.remove(key);seen.add(key)
    try:walk(root)
    except (ValueError,TypeError,AttributeError) as e:return dict(status='INVALID_GRAPH',indices=[],operations=operations,issues=[str(e)])
    return dict(status='UNKNOWN_SOURCE_SEMANTICS' if issues else 'TRACED',indices=sorted(indices),
        operations=operations,issues=issues,reachable_nodes=sorted(seen))

def _input_file(root, search, name):
    p=PurePosixPath(str(name).replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or ':' in str(p):raise ValueError('Unsafe source path')
    for parent in [search,*search.parents]:
        if not parent.is_relative_to(root):break
        for base in (parent,parent/'alignment'):
            candidate=(base/str(p)).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():return candidate
    return None

def assess_task(folder,task,product):
    root=Path(folder).resolve();product=Path(product).resolve()
    if not product.is_relative_to(root) or not product.is_file():raise ValueError('Current product missing or outside this job')
    result=task.get('result',{});report=dict(field=task.get('field'),role=task.get('role'),
        product_path=product.relative_to(root).as_posix(),product_sha256=sha(product),
        numerical_source_influence='NOT_TESTED',accuracy_improvement='NOT_ESTABLISHED',
        independent_reference_verified=False,algorithm_implementation_changed=False)
    if result.get('status')=='DERIVED_FROM_COMPONENT_SEARCH':
        components=result.get('component_tasks',[])
        if not components:raise ValueError('Derived product has no upstream component searches')
        reports=[assess_task(root,c,c['product']) for c in components]
        if any(c['outcome'] in {'NOT_DETERMINABLE','SINGLE_SOURCE_PRESERVED'} for c in reports):
            outcome='NOT_DETERMINABLE'
        elif any(c['outcome']=='MULTISOURCE_GRAPH_EFFECT_UNVERIFIED' for c in reports):
            outcome='MULTISOURCE_COMPONENT_GRAPH_EFFECT_UNVERIFIED'
        else:outcome='SINGLE_SOURCE_COMPONENTS_SELECTED'
        return dict(report,outcome=outcome,reference_basis='COMPONENT_SEARCH_EVIDENCE',
            transform=result.get('transform'),components=reports,
            warning='Vector component selection is traced separately; graph participation alone does not establish accuracy improvement')
    if result.get('status')=='UNCHANGED_COMPLETE_INPUT':
        return dict(report,outcome='SINGLE_SOURCE_PRESERVED',reference_basis='NO_COMPARATIVE_VALIDATION',inputs=[])
    search_product=Path(result.get('product',product)).resolve()
    if not search_product.is_relative_to(root):raise ValueError('Search product outside current job')
    request_path=search_product.parent/'request.json'
    if not request_path.is_file():
        return dict(report,outcome='NOT_DETERMINABLE',reference_basis='MISSING_SEARCH_EVIDENCE',inputs=[])
    request=json.loads(request_path.read_text(encoding='utf8'));inputs=request.get('inputs',[])
    ledger=[]
    for item in inputs:
        source=_input_file(root,request_path.parent,item.get('source_filename',''))
        expected=item.get('sha256');actual=sha(source) if source else None
        if source and actual!=expected:raise ValueError('Current source hash disagrees with algorithm input audit')
        support=None
        if source and item.get('role') in {'primary','background'}:
            import xarray as xr
            try:
                with xr.open_dataset(source) as ds:support=ds[task.get('field')].attrs.get('spatial_semantics')
            except (ValueError,KeyError,OSError):pass
        item['_spatial_support']=support
        ledger.append(dict(role=item.get('role'),source_filename=item.get('source_filename'),spatial_support=support,
            sha256=expected,path=source.relative_to(root).as_posix() if source else None,
            hash_verified=bool(source and actual==expected)))
    protocol=result.get('self_validation_protocol',{})
    comparison=result.get('selection_comparison',{})
    assessment=result.get('post_selection_assessment',{})
    if comparison and comparison.get('selected_key')!=result.get('selected_graph',{}).get('key'):
        raise ValueError('Scored winner differs from deployed graph')
    report.update(selection_protocol=comparison.get('protocol','LEGACY_OR_NOT_APPLICABLE'),
        candidate_comparison=comparison,post_selection_assessment=assessment,
        held_out_assessment_completed=assessment.get('status')=='PASS',
        test_used_for_selection=protocol.get('test_used_for_selection'))
    self_ref=protocol.get('self_gridded_reference')
    reference=('SELF_GRID_REFERENCE' if self_ref is True else 'STATION_REFERENCE_INDEPENDENCE_UNVERIFIED' if self_ref is False else 'UNKNOWN_REFERENCE')
    trace=graph_sources(result.get('selected_graph'),inputs,self_ref)
    used=[ledger[i] for i in trace['indices']]
    fingerprints={i['sha256'] for i in used if i.get('sha256')}
    verified=all(i['hash_verified'] for i in used) and bool(used)
    if trace['status']!='TRACED' or not verified:outcome='NOT_DETERMINABLE'
    elif len(fingerprints)==1:outcome='SINGLE_SOURCE_SELECTED'
    elif len(fingerprints)>1:outcome='MULTISOURCE_GRAPH_EFFECT_UNVERIFIED'
    else:outcome='NOT_DETERMINABLE'
    return dict(report,outcome=outcome,reference_basis=reference,inputs=ledger,selected_sources=used,
        graph_trace=trace,distinct_selected_file_hashes=len(fingerprints),
        source_file_identity_is_not_statistical_independence=True,
        warnings=['Self-reference scores do not establish improvement against independent truth'] if self_ref is True else
            ['Independent reference origin has not been verified'],
        reference_proof_required=['Reference origin and upstream independence','Excluded from fitting and candidate selection',
            'Same held-out support for both single sources and fusion','Uncertainty of paired improvement comparison'])

def assess_preparation(folder,receipt,products):
    import numpy as np
    tasks=receipt.get('tasks',[])
    reports=[assess_task(folder,t,p) for t,p in zip(tasks,products)]
    states={r['outcome'] for r in reports}
    overall=next(iter(states)) if len(states)==1 else 'MIXED_OR_NOT_DETERMINABLE'
    if not reports:overall='NOT_DETERMINABLE'
    stage=receipt.get('fusion_stage',{})
    selections=[]
    if stage.get('status')=='COMPLETE' and stage.get('algorithm_search_executed') is True:
        for search in stage.get('searches',[]):
            comparison=search.get('selection_comparison',{});graph=search.get('selected_graph',{})
            family=comparison.get('selected_family');key=comparison.get('selected_key')
            families=comparison.get('families',{});winner=families.get(family,{})
            metrics=winner.get('metrics',{})
            valid=(comparison.get('decision_after_scoring') is True
                and family in {'SINGLE_SOURCE','MULTISOURCE'} and isinstance(key,str)
                and graph.get('key')==key and winner.get('best_key')==key
                and int(winner.get('evaluated',0))>0 and int(metrics.get('count',0))>0
                and np.isfinite([metrics.get('RMSE',np.nan),metrics.get('MAE',np.nan)]).all())
            if not valid:
                selections=[];break
            selections.append(dict(component=search.get('component'),selected_family=family,
                selection_metric=comparison.get('selection_metric'),selected_key=key,
                evaluated=int(winner['evaluated']),scores=metrics,
                single_source_scores=families.get('SINGLE_SOURCE',{}).get('metrics'),
                multisource_scores=families.get('MULTISOURCE',{}).get('metrics'),
                independent_location_test=search.get('post_selection_assessment',{}).get('status')))
        if selections and len(selections)==len(stage.get('searches',[])):
            overall='PLATFORM_SELECTION_COMPLETED'
    effect_verified=bool(selections) and all(s['selected_family']=='MULTISOURCE' for s in selections) \
        and int(stage.get('updated_source_records',0))>0 and float(stage.get('max_wind_speed_change_m_s',0))>0
    return dict(schema='ecodata.fusion-assessment.v1',outcome=overall,tasks=reports,
        platform_selection=dict(status='PASS' if selections else 'NOT_VERIFIED',algorithm_search_executed=bool(selections),
            components=selections,scoring_is_not_independent_accuracy_validation=True),
        numerical_multisource_contribution_verified=effect_verified,
        numerical_effect=dict(updated_records=stage.get('updated_source_records'),
            max_wind_speed_change_m_s=stage.get('max_wind_speed_change_m_s')) if selections else None,
        accuracy_improvement='NOT_ESTABLISHED',
        scope='Read-only evidence gate shared by all converter models, not a replacement scorer or fusion algorithm',
        warning='Multiple inputs or graph edges do not prove effective fusion; native conversion success does not establish accuracy')
