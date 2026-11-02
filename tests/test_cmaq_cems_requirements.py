import csv
from ecodata.interface.cmaq_cems_import import check_point_requirements

def write(tmp_path, **changes):
    # Synthetic input only for validating rejection logic; not model evidence.
    row=dict(region_cd='A',facility_id='B',unit_id='C',rel_point_id='D',process_id='E',erptype='2',longitude='10',latitude='20',stkhgt='30',stkdiam='2',stktemp='100',stkvel='5')
    row.update(changes);p=tmp_path/'input.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
    return p

def test_complete_point_fields_pass_without_rewriting(tmp_path):
    p=write(tmp_path);before=p.read_bytes()
    assert check_point_requirements(p)['ready_for_point_route']
    assert p.read_bytes()==before

def test_missing_fields_listed_without_filling(tmp_path):
    p=write(tmp_path,stkdiam='',stkvel='-9');before=p.read_bytes()
    r=check_point_requirements(p)
    assert not r['ready_for_point_route']
    assert len(r['missing_information'][0]['required'])==2
    assert not r['unsupported_sources']
    assert p.read_bytes()==before

def test_fugitive_is_unsupported_not_missing_chimney(tmp_path):
    p=write(tmp_path,erptype='1',stkhgt='',stkdiam='',stktemp='',stkvel='')
    r=check_point_requirements(p)
    assert not r['ready_for_point_route']
    assert not r['missing_information']
    assert len(r['unsupported_sources'])==1
