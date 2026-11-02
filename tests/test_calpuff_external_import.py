"""Synthetic structural fixtures only; no fabricated scientific acceptance."""
import hashlib,io,json,zipfile
import pytest
from test_calpuff7_project import control
from ecodata.interface.calpuff_project import unpack
from ecodata.interface.calpuff_update_contract import validate_update

def pack(data):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for name,value in data.items():z.writestr(name,value)
    return out.getvalue()

def sample():
    text=control().replace('CALMET.DAT','MY_MET.DAT').encode()
    return {'author/run/MY_PUFF.INP':text,'author/run/my_met.dat':b'fixture met bytes',
            'author/run/my_met.dat.aux':b'fixture aux bytes','author/notes.txt':b'notes'}

def test_nested_original_names_and_bytes_preserved():
    source=sample();data,provenance=unpack(pack(source),'existing')
    assert data['CALPUFF.INP']==source['author/run/MY_PUFF.INP']
    assert data['MY_MET.DAT']==source['author/run/my_met.dat']
    report=json.loads(data['import_report.json'])
    assert report['control_bytes_preserved'] and report['weather_bytes_preserved']
    assert not report['scientific_parameters_changed']
    assert report['generated_provenance_declaration']
    assert '未提供来源' in provenance['sources'][0]

def test_missing_auxiliary_never_filled():
    source=sample();del source['author/run/my_met.dat.aux']
    with pytest.raises(ValueError,match='实际引用'):unpack(pack(source),'existing')

def test_unsupported_version_not_relabelled():
    source=sample();source['author/run/MY_PUFF.INP']=source['author/run/MY_PUFF.INP'].replace(b'7.0',b'5.8')
    with pytest.raises(ValueError,match='其他版本不自动改头'):unpack(pack(source),'existing')

def test_multiple_controls_require_selection():
    source=sample();source['author/run/OTHER.INP']=source['author/run/MY_PUFF.INP']
    with pytest.raises(ValueError,match='多个工程需选择'):unpack(pack(source),'existing')
    data,_=unpack(pack(source),'existing','author/run/OTHER.INP')
    assert json.loads(data['import_report.json'])['control_source']=='author/run/OTHER.INP'

BASE=b'''CALPUFF.INP     7.0
! NX = 9 !
! METDAT = CALMET.DAT !
! NPT1 = 1 ! ! NPT2 = 0 ! ! IPTU = 1 ! ! NSPT1 = 0 ! ! END !
! SRCNAM = S1 ! ! X = 1, 2, 3 ! ! END !
! NAR1 = 0 ! ! END !
! NREC = 1 ! ! NRGRP = 0 ! ! END !
! X = 1, 2, 3, 0 ! ! END !
'''

@pytest.mark.parametrize('mode',['constant_points','receptors'])
def test_only_requested_group_may_change(mode):
    changed=BASE.replace(b'X = 1, 2, 3 !',b'X = 4, 5, 6 !') if mode=='constant_points' else BASE.replace(b'X = 1, 2, 3, 0',b'X = 4, 5, 6, 0')
    assert validate_update(BASE,changed,mode)['unrelated_active_controls_preserved']
    with pytest.raises(ValueError,match='目标组以外'):
        validate_update(BASE,changed.replace(b'NX = 9',b'NX = 10'),mode)
    with pytest.raises(ValueError,match='目标组以外'):
        validate_update(BASE,changed.replace(b'CALMET.DAT',b'OTHER.DAT'),mode)

def test_receptor_update_cannot_change_point_sources():
    with pytest.raises(ValueError,match='目标组以外'):
        validate_update(BASE,BASE.replace(b'SRCNAM = S1',b'SRCNAM = S2'),'receptors')
