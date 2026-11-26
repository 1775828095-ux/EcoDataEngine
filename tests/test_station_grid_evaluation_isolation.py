"""Synthetic protocol boundaries; not environmental improvement evidence."""
import numpy as np
import pandas as pd
import pytest
from ecodata.desktop import station_grid as sg


def frame():
    times=pd.date_range('2000-01-01',periods=24,freq='h')
    return pd.DataFrame(dict(station='A',grid_time=times,time=times,observed=np.arange(24.)+2,
        background=np.arange(24.),grid_lon=110.,grid_lat=30.,matched_observations=1))


def test_test_values_cannot_change_bias_or_frozen_selection():
    a=frame();first,report=sg.evaluate_correction(a)
    b=a.copy();b.loc[b.grid_time.isin(pd.to_datetime(report['test_times'])),'observed']+=1000
    second,changed=sg.evaluate_correction(b)
    assert report['selected']==changed['selected']=='corrected'
    assert report['training_bias']==changed['training_bias']
    np.testing.assert_array_equal(first.selected,second.selected)
    assert report['heldout_correction']['rmse']==0
    assert changed['heldout_correction']['rmse']>900


def test_validation_selects_background_when_correction_does_not_transfer():
    data=frame();data.loc[12:,'observed']=data.loc[12:,'background']
    output,report=sg.evaluate_correction(data)
    assert report['selected']=='background'
    assert report['selection_baseline']['rmse']==0
    assert report['selection_correction']['rmse']==2
    assert report['heldout_baseline']['count']==6
    assert set(report['training_times']).isdisjoint(report['validation_times'])
    assert set(report['test_times']).isdisjoint(report['validation_times'])


def test_insufficient_time_support_not_invented_and_ties_keep_simple_source():
    with pytest.raises(ValueError,match='9'):sg.evaluate_correction(frame().iloc[:8])
    data=frame();data.observed=data.background
    _,report=sg.evaluate_correction(data)
    assert report['selected']=='background'


def test_common_queue_never_compares_different_targets_or_hides_failures():
    a=frame();b=a.iloc[2:].copy()
    aligned,report=sg.common_queue({'a':a,'b':b})
    assert len(aligned['a'])==len(aligned['b'])==22
    assert report['a']['raw_matched_records']==24
    b.observed+=1
    with pytest.raises(ValueError,match='reference'):sg.common_queue({'a':a,'b':b})
    a.loc[0,'background']=np.nan
    with pytest.raises(ValueError,match='finite'):sg.evaluate_correction(a)


def test_tolerance_uses_same_actual_duration_for_us_and_ns():
    times=pd.date_range('2000-01-01',periods=4,freq='h')
    assert sg.matching_tolerance(times.as_unit('us'))==pd.Timedelta(minutes=30)
    assert sg.matching_tolerance(times.as_unit('ns'))==pd.Timedelta(minutes=30)


def test_result_card_ranking_uses_validation_and_exposes_failed_confirmation():
    from ecodata.desktop.output_review import cards
    data=frame();data.loc[18:,'observed']=data.loc[18:,'background']
    _,report=sg.evaluate_correction(data)
    assert report['selected']=='corrected' and report['effect_confirmation']=='NOT_CONFIRMED'
    report.update(methods=[sg.UNIT_STEP,sg.FORMAT_STEP,sg.EXACT,sg.MATCH,sg.BIAS],matched_records=24,stations=1,unit='K')
    result=dict(mode='selected_operations',original_report=[dict(task='temperature',backend='station_grid',report=[report])])
    items=cards(result,dict(source_formats=['NetCDF','CSV'],target_format='Parquet',conversion_reason='Synthetic display check'))
    selected=next(x for x in items if x['selected'])
    assert selected['rank']==1 and selected['metrics']['RMSE']==2
    assert any('NOT_CONFIRMED' in line for line in selected['audit'])
