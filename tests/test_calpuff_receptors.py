import csv,io,json,zipfile
from pathlib import Path
import numpy as np
import pytest
from ecodata.models.adapters.calpuff_receptors import export_receptors
from ecodata.core.exceptions import SchemaError

BASE=Path(__file__).parents[1]/'examples/ai_round1/calpuff/receptors'


def case(alternate=False):
    template=(BASE/'official_CPUF.INP').read_text(encoding='ascii')
    columns=['id','x','y','elevation','height_agl']
    rows=[['r1',310250,4820500,123.5,2.5],['r2',310750,4821250,125.25,0]]
    expected=np.array([[310.25,4820.5,123.5,2.5],[310.75,4821.25,125.25,0]])
    cfg=dict(mode='receptors',version='CALPUFF-5.8.4-group17',table='receptors.csv',control_file='CPUF.INP',
        fields=dict(zip(columns,columns)),units=dict(x='m',y='m',elevation='m',height_agl='m'),source_crs='EPSG:26719',
        elevation_vertical_reference='MSL',missing_values=[-9999],max_float32_error_m=.1,
        source=dict(kind='synthetic_with_official_template',description='Synthetic receptors in official control domain; not real receptors or a full engineering test',license='Synthetic CC0; official template used for interface testing'))
    if alternate:
        rows=[['s1',310.25,4820.5,.1235,.0025],['s2',310.75,4821.25,.12525,0]]
        columns=['site','east','north','base','above'];cfg['fields']=dict(zip(['id','x','y','elevation','height_agl'],columns))
        cfg['units']=dict(x='km',y='km',elevation='km',height_agl='km');cfg['source_crs']='EPSG:32619'
        template=template.replace('DATUM = NAS-C','DATUM = WGS-84')
    return bundle(rows,columns,template),cfg,expected,template


def bundle(rows,columns,template):
    b=io.StringIO();w=csv.writer(b);w.writerow(columns);w.writerows(rows);out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('receptors.csv',b.getvalue());z.writestr('CPUF.INP',template)
    return out.getvalue()


@pytest.mark.parametrize('alternate',[False,True])
def test_independent_group17_parse_and_preserved_control(alternate):
    data,cfg,expected,original=case(alternate);files,report=export_receptors(data,cfg)
    text=files['CALPUFF.INP'].decode('ascii')
    # Every odd exclamation-delimited substring is an active input token.
    tokens=[v.strip() for v in text.split('!')[1::2]]
    pos=next(i for i,v in enumerate(tokens) if v.startswith('NREC'))
    assert tokens[pos]=='NREC = 2' and tokens[pos+1]=='END'
    records=[]
    for i in range(pos+2,len(tokens),2):
        assert tokens[i].startswith('X =') and tokens[i+1]=='END'
        records.append([float(v) for v in tokens[i].split('=',1)[1].split(',')])
    np.testing.assert_allclose(records,expected,rtol=0,atol=1e-12)
    old_prefix=original[:original.index('!  NREC')]
    assert text[:len(old_prefix)]==old_prefix
    assert report['model_run_verified'] is False and report['receptors']==2


@pytest.mark.parametrize('error',['field','units','crs','missing','outdomain','height','oldcount','unknown_record','datum','elevation_reference'])
def test_receptor_rejection(error):
    data,cfg,_,template=case()
    if error=='field':cfg['fields'].pop('x')
    if error=='units':cfg['units']['x']='K'
    if error=='crs':cfg['source_crs']='EPSG:32619'
    if error=='missing':cfg['missing_values']=[310250]
    if error=='outdomain':data=bundle([['r',999000,4820500,1,0]],list(cfg['fields']),template)
    if error=='height':data=bundle([['r',310250,4820500,1,-1]],list(cfg['fields']),template)
    if error=='oldcount':data=bundle([['r',310250,4820500,1,0]],list(cfg['fields']),template.replace('NREC =  0','NREC =  1'))
    if error=='unknown_record':data=bundle([['r',310250,4820500,1,0]],list(cfg['fields']),template+'\n! UNKNOWN = 1 !\n! END !')
    if error=='datum':data=bundle([['r',310250,4820500,1,0]],list(cfg['fields']),template.replace('DATUM = NAS-C','DATUM = UNKNOWN'))
    if error=='elevation_reference':cfg['elevation_vertical_reference']='ellipsoid'
    with pytest.raises(SchemaError):export_receptors(data,cfg)


def save_examples(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    for alternate in (False,True):
        data,cfg,_,_=case(alternate);name='synthetic_mapped_WGS84' if alternate else 'synthetic_NAD27'
        (path/f'A_{name}.zip').write_bytes(data)
        (path/f'config_{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')


def test_group20_preserves_explicit_no_groups_and_latin1_comments():
    data,cfg,expected,original=case()
    original=original.replace('!  NREC =  0   !','!  NREC =  0   !\n! NRGRP = 0 !')
    assert 'NRGRP = 0' in original
    original='author\x92s comment\n'+original
    cfg.update(version='CALPUFF-7.2.1-group20',control_encoding='latin1')
    rows=[['r1',310250,4820500,123.5,2.5]]
    raw=bundle(rows,list(cfg['fields']),original)
    # bundle encodes strings asUTF8; write original control bytes explicitly.
    b=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as z,zipfile.ZipFile(b,'w') as out:
        out.writestr('receptors.csv',z.read('receptors.csv'));out.writestr('CPUF.INP',original.encode('latin1'))
    files,report=export_receptors(b.getvalue(),cfg)
    text=files['CALPUFF.INP'].decode('latin1')
    assert text.startswith('author\x92s comment') and '! NRGRP = 0 !' in text and 'group20c' in text
    assert report['receptors']==1 and report['version']==cfg['version']
    cfg['control_encoding']='utf8'
    with pytest.raises(SchemaError):export_receptors(b.getvalue(),cfg)
