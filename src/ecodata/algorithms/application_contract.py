"""Explicit reconstruction-task contracts, without inventing simulator profiles."""
from copy import deepcopy
import numpy as np
from ecodata.algorithms.attribute_registry import configuration,digest
from ecodata.algorithms.portrait_rules import model_requirements


def prepare(request):
    if request.get('model_profile') or request.get('portrait_policy'):
        raise ValueError('Application contract and simulator profile are mutually exclusive')
    declaration=request['application_contract']
    required={'id','version','field','unit','semantic_type','valid_range','evidence','issues'}
    if set(declaration)!=required or not declaration['evidence'] or not declaration['issues']:
        raise ValueError('Complete evidence-bearing application contract required')
    if (declaration['field']!=request['variable'] or declaration['unit']!=request['unit'] or
            declaration['semantic_type']!=request['semantic_type'] or declaration['unit'] in {'','UNKNOWN'}):
        raise ValueError('Application target identity, unit or semantic type differs')
    bounds=declaration['valid_range']
    if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
        raise ValueError('Finite ordered application range required')
    registry,parameters=configuration(request)
    definitions={a['id']:a for a in registry['attributes']}
    for issue in declaration['issues']:
        if set(issue)!={'attribute','metric','acceptable','bad','importance','evidence'}:
            raise ValueError('Explicit state issue and evidence required')
        attribute=definitions.get(issue['attribute'])
        if (attribute is None or attribute['provider']!='state_metric' or attribute['property']!=issue['metric']
                or not attribute['scoring_enabled'] or not issue['evidence']
                or not np.isfinite([issue['acceptable'],issue['bad'],issue['importance']]).all()
                or issue['bad']<=issue['acceptable'] or issue['importance']<=0):
            raise ValueError('Issue must use a registered finite state measurement')
    priorities={i['attribute']:i['importance'] for i in declaration['issues']}
    matching=model_requirements(dict(priorities=priorities,evidence=declaration['evidence']),registry)
    scenario=dict(id=declaration['id'],task=request['task'],role='reconstruction',field=request['variable'],
        issues=deepcopy(declaration['issues']),coupled_constraints=[],evidence=declaration['evidence'])
    model=dict(model_id='application:'+declaration['id'],version=declaration['version'],
        selected_role='reconstruction',selected_field=dict(name=request['variable'],unit=request['unit'],data_type=request['semantic_type']),
        matching_requirements=matching,attribute_registry=registry,scoring_parameters=parameters,
        state_policy=dict(policy_version=declaration['version'],policy_sha256=digest(declaration),scenario=scenario),
        application_preferences={},sensitivity=None,sensitivity_status='UNKNOWN',full_model_ready=False,
        native_acceptance='NOT_APPLICABLE_RECONSTRUCTION_TASK',scope='Explicit task contract; not a simulator response model')
    result=dict(request,parameters=dict(request.get('parameters',{}),target_valid_range=list(bounds)))
    return result,model
