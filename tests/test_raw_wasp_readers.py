import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import gzip
import pytest

from ecodata.recognition import recognise_file
from ecodata.core.exceptions import SchemaError
from ecodata.io.raw_tables import locate_header
from ecodata.models import RequirementStore
from ecodata.models.scenario import recognise_inputs, suggest_roles, plan_scenario

RAW=Path(__file__).resolve().parents[1]/'examples/real_wasp_narragansett/raw_source_data'


def test_noaa_reader_skips_library_parse_error_instead_of_aborting_file():
    from ecodata.io.raw_readers import NOAAISDReader
    import pytest
    from ecodata.core.exceptions import SchemaError
    with pytest.raises(SchemaError, match="没有可读取"):
        NOAAISDReader().read_bytes(gzip.compress(b"0091 malformed ISD record\n"))


def test_excel_selects_actual_data_and_keeps_censoring():
    p=RAW/'nbc/2009-NBC-River-and-Bay-Nutrients-Data.xlsx'
    ds,info=recognise_file(p.name,p.read_bytes())
    assert json.loads(ds.attrs['reader_evidence'])['selected_sheet']=='2009'
    assert ds.sizes['__row__']==448 and 'Station' in ds
    assert ds['Total Nitrogen'].values.tolist()==['']*448
    assert any(str(x).startswith('<') for x in ds['NH3'].values)
    assert ds['NH3__numeric'].attrs['mass_basis']=='N'
    assert ds['NH3__numeric'].attrs['units']=='ug/L'
    assert info.time_coordinate=='time'


def test_sheet_tie_requires_confirmation():
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        for name in ['a','b']:pd.DataFrame({'value':[1,2]}).to_excel(writer,sheet_name=name,index=False)
    with pytest.raises(SchemaError,match='sheet'):
        recognise_file('ambiguous.xlsx',buffer.getvalue())


def test_usgs_preserves_identifiers_daily_semantics_and_quality():
    p=RAW/'usgs/01113895_daily_2009.rdb'
    ds,_=recognise_file(p.name,p.read_bytes())
    assert len(ds.value)==365
    assert ds.station_id.values[0]=='01113895' and ds.parameter_code.values[0]=='00060'
    assert ds.value.values[0]==2000 and ds.quality_code.values[0]=='A'
    assert ds.value.attrs['units']=='ft^3/s'
    assert ds.attrs['temporal_semantics']=='mean'
    assert ds.attrs['time_zone']=='local_calendar_day_unconfirmed'


def test_usgs_explicit_est_edt_conversion_and_metadata():
    p=RAW/'usgs/01113895_instantaneous_2009.rdb'
    ds,_=recognise_file(p.name,p.read_bytes())
    assert len(ds.value)==35021
    assert ds.time.values[0]==np.datetime64('2009-01-01T05:00')
    assert ds.datetime.values[0]=='2009-01-01 00:00' and ds.timezone.values[0]=='EST'
    assert 'quality' in ds.attrs['source_metadata'].lower() or 'qualification' in ds.attrs['source_metadata']


@pytest.mark.parametrize('fragment',['B12.gsoHB060209.TXT','F4.nbc.PDB040709.RPT.txt','NARTBWQ.csv'])
def test_text_header_numeric_units_and_no_footer_rows(fragment):
    p=next((RAW/'ri_dem').rglob(fragment))
    ds,info=recognise_file(p.name,p.read_bytes())
    assert info.time_coordinate=='time'
    assert json.loads(ds.attrs['reader_evidence'])['source_data_rows']==ds.sizes['__row__']
    do=next(x for x in ds if ds[x].attrs.get('physical_variable')=='dissolved_oxygen')
    assert ds[do].dtype.kind=='f' and ds[do].attrs['units']=='mg/L'
    assert not np.isnat(ds.time.values).any()


def test_raw_wrong_year_is_not_corrected():
    p=next((RAW/'ri_dem').rglob('B2.gsoNB052709.TXT'))
    ds,_=recognise_file(p.name,p.read_bytes())
    assert str(ds.time.values[0]).startswith('2000-')


def test_duplicate_measurement_names_disambiguated_by_source_unit():
    p=next((RAW/'ri_dem').rglob('B2.gsoNS061709.TXT'))
    ds,_=recognise_file(p.name,p.read_bytes())
    oxygen=[x for x in ds if ds[x].attrs.get('physical_variable')=='dissolved_oxygen']
    saturation=[x for x in ds if ds[x].attrs.get('physical_variable')=='oxygen_saturation']
    assert len(oxygen)==len(saturation)==1
    assert ds[oxygen[0]].values[0]==14.185
    assert ds[saturation[0]].values[0]==174.898
    assert ds[oxygen[0]].attrs['source_field_name']=='Optical DO'


def test_noaa_real_native_parser_and_missing_fields_not_generated():
    p=RAW/'noaa/725070-14765-2009.gz'
    ds,_=recognise_file(p.name,p.read_bytes())
    assert ds.sizes['__row__']==13390
    assert ds.temperature.values[0]==-6.7 and ds.wind_speed.values[0]==8.2
    assert ds.pressure.values[0]==1004.7 and ds.precipitation_AA1.values[0]==4.1
    assert ds.precipitation_AA1_hours.values[0]==6
    assert 'solar_radiation' not in ds and 'humidity' not in ds
    assert ds.attrs['reader_backend']=='ish-parser 0.0.25'


def test_header_locator_does_not_guess_unstructured_text():
    with pytest.raises(SchemaError):locate_header('hello\nworld\n')


def test_wasp_generic_plan_does_not_repurpose_observations():
    p=RAW/'nbc/2009-NBC-River-and-Bay-Nutrients-Data.xlsx'
    profile=RequirementStore().get('wasp')
    ins=recognise_inputs([(p.name,p.read_bytes())])
    roles=suggest_roles(profile,ins)
    assert not roles['initial_conditions'] and not roles['validation_observations']
    roles['initial_conditions']=p.name
    _,plan=plan_scenario(profile,ins,roles)
    assert not plan['can_process']
    assert next(x for x in plan['matches'] if x['role']=='initial_conditions')['missing']==['role_evidence']


def test_wasp_flow_plan_reuses_unit_conversion():
    p=RAW/'usgs/01113895_daily_2009.rdb'
    profile=RequirementStore().get('wasp')
    ins=recognise_inputs([(p.name,p.read_bytes())])
    roles=suggest_roles(profile,ins)
    assert roles['tributary_inflows']==p.name
    _,plan=plan_scenario(profile,ins,roles)
    assert plan['can_process_partial'] and not plan['complete_model_input']
    assert any(x['operation']=='unit_conversion' for x in plan['steps'])
