from ecodata.interface.input_result_binding import bind_result

def test_changing_material_inputs_clears_previous_success():
    state={}
    bind_result(state,'job','binding',b'old data','target1')
    state['job']='old_success'
    bind_result(state,'job','binding',b'new data','target1')
    assert 'job' not in state

def test_same_input_keeps_running_job_and_choice_change_clears_it():
    state={}
    bind_result(state,'job','binding',b'data','target1')
    state['job']='current_running'
    bind_result(state,'job','binding',b'data','target1')
    assert state['job']=='current_running'
    bind_result(state,'job','binding',b'data','target2')
    assert 'job' not in state

def test_length_delimited_hash_not_ambiguous():
    state={};a=bind_result(state,'job','binding',b'ab',b'c')
    b=bind_result(state,'job','binding',b'a',b'bc')
    assert a!=b


def test_preexisting_unbound_success_is_not_assumed_to_match_current_input():
    state={'job':'legacy_success'}
    bind_result(state,'job','new_binding',b'new data')
    assert 'job' not in state
