from pathlib import Path
import pandas as pd
import pytest
from ecodata.interface.camx_runner import decode_table,directory

def test_native_table_matches_previous_decoding():
    case=Path(__file__).resolve().parents[1]/'examples/ai_round1/camx'
    actual=decode_table(case/'raw/tuv.do_CB6r5.20160610')
    expected=pd.read_csv(case/'A_photolysis_table.csv')
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    assert len(actual)==189750

@pytest.mark.parametrize('ident',['../foo','a'*33,'','ABCDEF0123456789abcdef0123456789ab'])
def test_bad_task_id_rejected(ident):
    with pytest.raises(ValueError):directory(ident)
