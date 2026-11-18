import pytest

from ecodata.core.exceptions import SchemaError
from ecodata.io.raw_tables import read_monitoring_text


def test_monitoring_fields_with_units_and_separate_annotations():
    raw=b'date,temperature,conductivity\nISO8601,degC,mS/cm\n2020-01-01,18,0.3,"Original Simulation",0.000\n'
    table,evidence=read_monitoring_text(raw)
    assert table.columns.tolist()==['date','temperature','conductivity']
    assert table.iloc[0].tolist()==['2020-01-01','18','0.3']
    assert evidence['trailing_annotations']==[dict(line=3,extra_tokens=2,basis='trailing_text_annotation')]


@pytest.mark.parametrize('record',['2020-01-01,18,0.3,99',
                                  '2020-01-01,Original Simulation,18,0.3'])
def test_numeric_extra_columns_or_shifted_observations_require_evidence(record):
    with pytest.raises(SchemaError):
        read_monitoring_text(('date,temperature,conductivity\nISO8601,degC,mS/cm\n'+record+'\n').encode())
