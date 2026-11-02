import io,zipfile
import pytest
from ecodata.models.swat_inputs_project import attach_inputs
from ecodata.core.exceptions import SchemaError


def bundle(name,raw):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:z.writestr(name,raw)
    return b.getvalue()


def project():
    return {'file.cio':b'test\nsimulation time.sim\nhru hru-data.hru\n','time.sim':b'test\nday yr day yr step\n1 2018 3 2018 0\n','hru-data.hru':b'test\nid name attribute\n1 hru1 2.0\n2 hru2 4.0\n'}


def test_updates_independent_copy_preserving_ids():
    original=project();updated=b'converted\nid name attribute\n1 hru1 3.0\n2 hru2 5.0\n'
    files,report=attach_inputs(original,bundle('hru-data.hru',updated),'hru_attributes')
    assert files['hru-data.hru']==updated
    assert original['hru-data.hru']!=updated and files['file.cio']==original['file.cio']
    assert report['replacements'][0]['before_sha256']!=report['replacements'][0]['after_sha256']


def test_reordered_ids_rejected_to_protect_numeric_references():
    raw=b'converted\nid name attribute\n2 hru2 5.0\n1 hru1 3.0\n'
    with pytest.raises(SchemaError,match='记录标识'):attach_inputs(project(),bundle('hru-data.hru',raw),'hru_attributes')


def test_new_unreferenced_target_rejected():
    with pytest.raises(SchemaError,match='没有同名'):attach_inputs(project(),bundle('soils.sol',b'test\nheader\nsoil1 2\n'),'soil')


def test_unsafe_files_are_never_executed_or_extracted():
    with pytest.raises(SchemaError,match='安全平铺'):attach_inputs(project(),bundle('../hru-data.hru',b'bad'),'hru_attributes')


def test_wgn_station_formatting_preserves_identity_but_coordinate_change_rejected():
    body=b'header\n'+b'1 '*14+b'\n'
    rows=b'columns\n'+(b'1 '*14+b'\n')*12
    original=project();original['weather-wgn.cli']=b'title\nstation 11.60000 37.38000 1770.00000 10\n'+rows
    equivalent=b'title\nstation 11.6 37.38 1770.0 10.0\n'+rows
    actual,_=attach_inputs(original,bundle('weather-wgn.cli',equivalent),'wgn')
    assert actual['weather-wgn.cli']==equivalent
    with pytest.raises(SchemaError,match='记录标识'):
        attach_inputs(original,bundle('weather-wgn.cli',equivalent.replace(b'37.38',b'37.39')),'wgn')
