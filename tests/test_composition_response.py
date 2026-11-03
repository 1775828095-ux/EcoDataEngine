"""The response portrait is scoped, empirical and independent of test truth."""
from copy import deepcopy
import hashlib,json
import numpy as np
import pytest
from ecodata.algorithms.composition_response import ResponsePortrait,family,scope
from ecodata.algorithms.composition_decision import Composition
from ecodata.algorithms import platform_search as api
from ecodata.algorithms import attribute_registry as ar
from ecodata.models import portrait_policy as policy
from test_full_screen_spatiotemporal import prepare_case,publish,stores


def response_fixture(tmp_path):
    publish();inputs,req,_=prepare_case(tmp_path)
    parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
    profile=dict(schema_version=1,version='test-response',training_partition='development',training_groups=[0,1],
        scope=scope(engine.sources,req,req['composition']['reconstruction']),families={})
    # Correlated, biased error moments; the best convex weight is not a prewritten algorithm id.
    for index in range(engine.count):
        chosen=[dict(field=s[0],window=s[1],branch=s[2],choice=engine.axes[i][g]) for i,(s,g) in enumerate(zip(engine.slots,engine.decode(index))) ]
        key,a=family(chosen,'fine_slow')
        profile['families'][key]=dict(second_moments=[[[1.,-.5],[-.5,1.]]]*2)
    path=tmp_path/'response.json';path.write_text(json.dumps(profile),encoding='utf-8')
    declaration=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return engine,req,profile,declaration


def test_error_second_moments_and_unknown_scope(tmp_path):
    engine,req,profile,declaration=response_fixture(tmp_path)
    portrait=ResponsePortrait(declaration,engine.sources,req,req['composition']['reconstruction'])
    for index in range(engine.count):
        plan=[dict(field=s[0],window=s[1],branch=s[2],choice=engine.axes[i][g]) for i,(s,g) in enumerate(zip(engine.slots,engine.decode(index)))]
        _,a=family(plan,'fine_slow')
        assert portrait.estimate(plan)==pytest.approx(np.sqrt(a*a+(1-a)**2-a*(1-a)))
    changed=deepcopy(req);changed['unit']='degC'
    unsupported=ResponsePortrait(declaration,engine.sources,changed,req['composition']['reconstruction'])
    assert not unsupported.supported and unsupported.estimate(plan) is None
    with pytest.raises(ValueError,match='digest'):
        ResponsePortrait(dict(declaration,sha256='wrong'),engine.sources,req,req['composition']['reconstruction'])


def test_unsupported_post_not_given_invented_response(tmp_path):
    engine,req,_,declaration=response_fixture(tmp_path)
    portrait=ResponsePortrait(declaration,engine.sources,req,req['composition']['reconstruction'])
    plan=[dict(field=s[0],window=s[1],branch=s[2],choice=engine.axes[i][g]) for i,(s,g) in enumerate(zip(engine.slots,engine.decode(0)))]
    next(p for p in plan if p['branch']=='post')['choice']=[dict(algorithm='raster_task.nonnegative',parameters={})]
    assert portrait.estimate(plan) is None


def test_model_weight_changes_real_public_recommendation_without_reference(tmp_path):
    _,req,profile,declaration=response_fixture(tmp_path)
    attrs=deepcopy(ar.load('complementary-state-v1')['attributes'])
    attrs.append(dict(id='expected_test',definition='Empirical normalized error',tables=['data','algorithm','model'],unit='dimensionless',role='scoring',scoring_enabled=True,hard_allowed=False,provider='state_metric',property='expected_error',activation='constant_one',evidence='Explicit synthetic response portrait mechanism check',history=['Not scientific training data']))
    ar.publish('response-test',attrs,evidence='Synthetic mechanism test')
    inputs,_,_=prepare_case(tmp_path)
    chosen=[]
    for weight in (0.,100.):
        scenario=deepcopy(policy.load('complementary-wrf-v1')['scenarios'][0])
        scenario['issues'].append(dict(attribute='expected_test',metric='expected_error',acceptable=0.,bad=1.,importance=weight,evidence='Synthetic model preference change'))
        version='response-model-'+str(int(weight))
        policy.publish(version,'wrf',[scenario],evidence='Synthetic preference mechanism test')
        request=deepcopy(req);request['scoring']['registry_version']='response-test';request['portrait_policy']['version']=version
        request['composition']['response_profile']=declaration
        request['composition']['search'].update(backend='none',generations=0)
        result=api.submit_search(inputs,request,None,tmp_path/('weight_'+str(weight)),mode='Profile',budget=dict(objective_limit=4,wall_seconds=120))
        assert result['best'] is not None
        choice=next(p['choice'] for p in result['best']['plan'] if p['branch']=='fusion')
        chosen.append(choice)
    assert chosen[0]!=chosen[1]
    assert chosen[1]['algorithm']=='fusion.spatiotemporal_detail' and chosen[1]['detail_strength']==.5
