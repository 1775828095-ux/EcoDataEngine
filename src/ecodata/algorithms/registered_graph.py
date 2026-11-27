"""Shared Registry discovery and admission of externally proposed field DAGs.

Discovery is exhaustive; compatibility on roots is evidence, not a claim that an
algorithm will remain incompatible after a legal upstream transformation.
"""
from dataclasses import replace
from itertools import permutations
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from pathlib import Path
import hashlib
import json

VERSION = 'registered_graph_v1'
_recommendation_session=ContextVar('ecodata_prepared_recommendation',default=None)


@contextmanager
def recommendation_session(provider,*,variables=None):
    """Request-local transport, inherited by legacy adapters calling public search."""
    token=_recommendation_session.set(dict(provider=provider,cache={},variables=set(variables) if variables else None))
    try:yield
    finally:_recommendation_session.reset(token)


def has_recommender():
    return _recommendation_session.get() is not None


def apply_prepared_recommendation(registry,context,request,folder,*,components=None,resources=()):
    """The single late-binding consumer, after preparation and label isolation.

    Adapters declare ports. They do not choose algorithms, weights or search
    loops. Failure to obtain a valid proposal retains the lawful open search.
    """
    session=_recommendation_session.get()
    if session is None:return request,0.
    if session['variables'] is not None and context.variable not in session['variables']:return request,0.
    from ecodata.algorithms.numerical_search import save
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    started=perf_counter();payload=proposal_request(registry,context,request)
    if components is not None:
        payload['components']=components
        payload['resources']['named_ports']=list(resources)
        payload['graph_schema']['parents']='source:N, declared named resource, or preceding node'
        payload['binding_scope']='Paired spatial estimates; only declared component ports can execute here'
    save(folder/'prepared_recommendation_request.json',payload)
    key=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    result=dict(request,initial_algorithms=[],recommended_recipes=[],recommended_graphs=[])
    receipt=dict(protocol='prepared-registry-recommendation-v2',request_sha256=key,attempts=[],cache_hit=key in session['cache'])
    try:
        if receipt['cache_hit']:
            proposed=session['cache'][key]
            graph=validate(proposed.get('graph'),registry,len(context.sources),components=components,resources=resources)
        else:
            for attempt in range(2):
                proposed=session['provider'](payload)
                save(folder/f'prepared_recommendation_response_{attempt+1}.json',proposed)
                try:
                    graph=validate(proposed.get('graph'),registry,len(context.sources),components=components,resources=resources)
                    receipt['attempts'].append(dict(status='STRUCTURE_ADMITTED'))
                    session['cache'][key]=proposed
                    break
                except (ValueError,KeyError,TypeError) as exc:
                    receipt['attempts'].append(dict(status='REJECTED',reason=str(exc)))
                    if attempt:raise
                    payload=dict(payload,previous_proposal=proposed,contract_failure=str(exc))
        result['recommended_graphs']=[dict(id='prepared-ds-graph',graph=graph,scope=dict(variable=context.variable),reason=str(proposed.get('reason','')))]
        receipt.update(status='SEARCH_PROPOSAL',numerically_verified=False)
    except (ValueError,KeyError,TypeError,OSError,TimeoutError,RuntimeError) as exc:
        receipt.update(status='RECOMMENDATION_FAILED_OPEN_SEARCH_CONTINUES',reason=str(exc))
    seconds=perf_counter()-started
    receipt['seconds']=seconds
    save(folder/'prepared_recommendation.json',receipt)
    return result,seconds


def catalogue(registry):
    return {s.id: dict(s.as_dict(), execution='compatibility_then_actual_parent_execution',
                      candidate_role='automatic_quality_gate' if s.category=='validation' else 'processing')
            for s in registry.specs()}


def discover(registry, context, *, max_bindings=16):
    """Inspect every registration without executing kernels or seeing labels.

    Parent trials are bounded per algorithm, not by a global first-N cutoff.
    UNKNOWN means unproven applicability, never an executable acceptance.
    """
    context = replace(context, validation_reference=None, validation_role=None)
    rows = []
    for spec in registry.specs():
        row = dict(id=spec.id, status=spec.status, trials=[], truncated=False,
                   root_eligible=False, candidate_status='CONDITIONAL',
                   reference_used=False)
        if spec.status != 'available' or spec.category == 'validation':
            row.update(candidate_status='EXCLUDED', reason=spec.notes or spec.status)
            rows.append(row)
            continue
        arity = spec.input_contract.get('source_arity')
        counts = [arity] if arity is not None else range(1, len(context.sources) + 1)
        for count in counts:
            if not isinstance(count, int) or count < 1 or count > len(context.sources):
                continue
            for parents in permutations(range(len(context.sources)), count):
                if len(row['trials']) >= max_bindings:
                    row['truncated'] = True
                    break
                local = replace(context, sources=tuple(context.sources[i] for i in parents))
                try:
                    check = registry.get(spec.id).compatibility(local)
                    row['trials'].append(dict(parents=list(parents), eligible=check.eligible,
                                              reasons=list(check.reasons), warnings=list(check.warnings)))
                    row['root_eligible'] |= check.eligible
                except (ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
                    row['trials'].append(dict(parents=list(parents), eligible=None, reason=str(exc)))
        row['candidate_status'] = 'ROOT_APPLICABLE' if row['root_eligible'] else 'CONDITIONAL'
        row['reason'] = ('Actual root compatibility passed; numerical and final gates still required'
                         if row['root_eligible'] else
                         'No proved root binding; upstream products/parameters/resources may be needed')
        rows.append(row)
    return dict(version=VERSION, algorithms=rows, inspected=len(rows),
                complete_registry_scan=True, complete_parent_space=False,
                binding_limit_per_algorithm=max_bindings, reference_used=False)


def validate(graph, registry, source_count, *, components=None, resources=()):
    """Validate a small explicit DAG, including every node's real source parents.

    Sources use source:0, source:1, ... within the prepared public request. No
    file names, Python expressions, arbitrary resources or evaluation handles.
    """
    if not isinstance(graph, dict) or set(graph) != {'nodes', 'output'}:
        raise ValueError('Graph requires nodes and output only')
    nodes = graph['nodes']
    if not isinstance(nodes, list) or not 0 <= len(nodes) <= 8:
        raise ValueError('Recommended graph requires 0..8 algorithm nodes')
    known = {f'source:{i}' for i in range(source_count)} | set(resources)
    normalized = []
    for node in nodes:
        if not isinstance(node, dict) or set(node) - {'id', 'algorithm', 'parents', 'parameters'}:
            raise ValueError('Unknown graph node fields')
        identifier = node.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in known:
            raise ValueError('Duplicate, absent or reserved node identity')
        from types import SimpleNamespace
        algorithm=node.get('algorithm')
        spec=(SimpleNamespace(id=algorithm,status='available',category='task_component',input_contract=components[algorithm])
              if components and algorithm in components else registry.get(algorithm).spec)
        if spec.status != 'available' or spec.category == 'validation':
            raise ValueError('Algorithm unavailable for processing: ' + spec.id)
        parents = node.get('parents')
        if not isinstance(parents, list) or not parents or any(p not in known for p in parents):
            raise ValueError('Explicit topological source/parent bindings required')
        arity = spec.input_contract.get('source_arity')
        if arity is not None and len(parents) != arity:
            raise ValueError('Parent arity conflict: ' + spec.id)
        parameters = node.get('parameters', {})
        if not isinstance(parameters, dict):
            raise ValueError('Node parameters must be an object')
        # State, targets, training and references belong to the public request,
        # never to a language model's parameter dictionary.
        allowed = set(spec.input_contract.get('parameter_domains', {})) | set(spec.input_contract.get('requires_parameters', []))
        if set(parameters) - allowed:
            raise ValueError('Undeclared recommended parameters: ' + spec.id)
        normalized.append(dict(id=identifier, algorithm=spec.id, parents=list(parents), parameters=parameters))
        known.add(identifier)
    by_id = {n['id']: n for n in normalized}
    if graph['output'] not in by_id and graph['output'] not in {f'source:{i}' for i in range(source_count)}:
        raise ValueError('Graph output must be an algorithm node or unchanged field source')
    ancestors = set()
    def visit(key):
        if key in by_id and key not in ancestors:
            ancestors.add(key)
            for parent in by_id[key]['parents']:
                visit(parent)
    visit(graph['output'])
    if ancestors != set(by_id):
        raise ValueError('Unused branches cannot lend capabilities to output')
    return dict(nodes=normalized, output=graph['output'])


def execute_recommendation(search, recommendation):
    """Consume explicit edges using the existing compatibility, executor and ledger."""
    graph = validate(recommendation['graph'], search.registry, len(search.roots))
    bindings = {f'source:{i}': root for i, root in enumerate(search.roots)}
    for node in graph['nodes']:
        if search.exhausted():
            raise ValueError('Recommendation stopped by shared search budget')
        parents = [bindings[p] for p in node['parents']]
        spec = search.registry.get(node['algorithm']).spec
        options = [node['parameters']] if node['parameters'] else list(search.options(spec, parents))
        if not options:
            raise ValueError('No legal parameter binding for ' + spec.id)
        key = search.intern(spec.id, parents, options[0])
        cached = search.evaluated.get(key)
        if not cached:
            trial = search.candidate(None, None, spec, parents, options[0], 'registered_graph_recommendation')
            if not trial:
                raise ValueError('Actual parent compatibility/redundancy rejected: ' + spec.id)
            cached = search.execute(trial['child'], 'registered_graph_recommendation')
        if key not in search.states:
            raise ValueError('Recommended upstream produced no usable field: ' + spec.id)
        bindings[node['id']] = key
    sink = bindings[graph['output']]
    if search.evaluated.get(sink, {}).get('status') != 'ELIGIBLE_OUTPUT':
        raise ValueError('Recommended output failed final requirements or numerical evaluation')
    return sink


def proposal_request(registry, context, request):
    """Reference-free public decision payload for any recommendation transport."""
    from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
    from ecodata.algorithms.router import AlgorithmSelector
    context=replace(context,validation_reference=None,validation_role=None,
        parameters=dict(context.parameters,**({'raster_target_times':request['target_times']} if request.get('target_times') else {})))
    extractor=SemanticFeatureExtractor()
    features=[extractor.extract(source.dataset,context.variable,
        confirmed=dict(nonnegative=bool(context.model_constraints.get('nonnegative')))).as_dict() for source in context.sources]
    target=extractor.task_target(context,request)
    if context.sources:
        semantic_input=context.sources[0].dataset
    elif context.observations is not None and not context.observations.empty:
        import xarray as xr
        semantic_input=xr.Dataset({context.variable:('record',context.observations.value.to_numpy())})
        semantic_input[context.variable].attrs=dict(units=request['unit'],data_kind=context.semantic_type,
            spatial_semantics='point',temporal_semantics=request.get('temporal_semantics','UNKNOWN'))
    else:raise ValueError('No actual data for a recommendation context')
    decision=AlgorithmSelector(registry).decide(context,target,
        semantics=extractor.extract(semantic_input,context.variable,
            confirmed=dict(nonnegative=bool(context.model_constraints.get('nonnegative'))))).as_dict()
    decision.pop('seconds',None)
    observation_summary=None
    if context.observations is not None:
        table=context.observations
        observation_summary=dict(records=len(table),
            positions=len(table[['longitude','latitude']].drop_duplicates()) if {'longitude','latitude'}<=set(table) else None,
            splits=sorted(table['split'].astype(str).unique().tolist()) if 'split' in table else [],
            scope='Actual allowed conditioning/training resource only')
    return dict(stage='registered_graph',protocol=VERSION,task=request['task'],
        variable=context.variable,semantic_type=context.semantic_type,
        sources=[dict(id=f'source:{i}',features=value,
            provenance=context.sources[i].dataset.attrs.get('feature_scope','Actual parsed/isolated input')) for i,value in enumerate(features)],
        requirements=decision,discovery=discover(registry,context),algorithms=catalogue(registry),
        target=dict(latitude=list(context.target_lat) if context.target_lat is not None else None,
            longitude=list(context.target_lon) if context.target_lon is not None else None,
            times=request.get('target_times'),unit=context.model_constraints.get('unit'),
            temporal_semantics=request.get('temporal_semantics'),nonnegative=context.model_constraints.get('nonnegative'),
            complete_coverage=target.get('require_complete_coverage'),grid_change=target.get('grid_change'),
            time_frequency=context.parameters.get('target_time_frequency'),
            paired_positions=context.parameters.get('paired_target_positions')),
        graph_schema=dict(nodes='0..8 topological nodes: id,algorithm,parents,parameters',
            parents='source:N or preceding node id',output='algorithm node id or unchanged source:N'),
        resources=dict(observations=context.observations is not None,auxiliary_count=len(context.auxiliaries),
            observations_summary=observation_summary,
            fitted_state='fitted_state' in context.parameters,allow_training=context.parameters.get('allow_training',False)),
        reference_used=False)
