from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ecodata.interface.calmet_observations import decode_surface,decode_upper,decode_precipitation
from ecodata.interface.calpuff_results import records,_expand
from ecodata.core.exceptions import SchemaError

BASE=Path('D:/EcoDataEngine/cases/CALPUFF_20261021')

@pytest.mark.skipif(not BASE.exists(),reason='Local registered public archives not installed')
def test_original_surface_matches_independent_extraction():
    frame,_,_=decode_surface(BASE/'public_case_search/SMERGE_v5.57.zip')
    reference=pd.read_csv(BASE/'complete_official_case/raw_surface/A_from_original_CD144_SAMSON.csv')
    pd.testing.assert_frame_equal(frame.reset_index(drop=True),reference.reset_index(drop=True),check_dtype=False,atol=1e-9,rtol=1e-10)

@pytest.mark.skipif(not BASE.exists(),reason='Local registered public archives not installed')
@pytest.mark.parametrize('station',['PWM','CHH'])
def test_upper_matches_independent_extraction(station):
    frame,_,_=decode_upper(Path('D:/EcoDataEngine/downloads/READ62_v5.54.zip'),station)
    reference=pd.read_csv(BASE/f'complete_official_case/raw_upper_preserve_verified/{station}/A.csv')
    pd.testing.assert_frame_equal(frame.reset_index(drop=True),reference.reset_index(drop=True),check_dtype=False,atol=1e-9,rtol=1e-10)

@pytest.mark.skipif(not BASE.exists(),reason='Local registered public archives not installed')
def test_precipitation_dry_case_coverage():
    frame,_,source=decode_precipitation(BASE/'public_case_search/PXTRACT_v4.25.zip')
    assert len(frame)==64 and frame.station_id.nunique()==16 and frame.time.nunique()==4
    assert (frame.hourly_mm==0).all() and len(source['records'])==64

def test_fortran_truncation_rejected():
    with pytest.raises(SchemaError):list(records(b'\x04\0\0\0abcd'))

def test_compression_expands_and_rejects_overflow():
    assert _expand(np.array([-3.0001,2.],dtype='<f4').tobytes(),4,True).tolist()==[0,0,0,2]
    with pytest.raises(SchemaError):_expand(np.array([-5.0001],dtype='<f4').tobytes(),4,True)
