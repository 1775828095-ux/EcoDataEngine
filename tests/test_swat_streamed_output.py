"""Small parser cases; actual large native output is verified separately."""
from datetime import date
import pytest
from ecodata.interface import swat_results as results


def fixture(tmp_path,rows):
    path=tmp_path/'channel_sd_day.txt'
    path.write_text('native structure fixture\njday mon day yr unit gis_id name flo_out\nm^3/s\n'+'\n'.join(rows)+'\n')
    return path


def test_selected_object_matches_full_reader(tmp_path):
    path=fixture(tmp_path,['1 1 1 2018 1 1 cha1 0.3','1 1 1 2018 2 2 cha2 0.7','2 1 2 2018 1 1 cha1 0.4','2 1 2 2018 2 2 cha2 0.8'])
    identities=results.output_objects(path)
    assert identities==[(1,1,'cha1'),(2,2,'cha2')]
    full=results.read_output(path)
    selected=results.read_output(path,selected_identity=identities[1])
    assert selected['objects']==[identities[1]]
    assert selected['validation_scope']=='selected_object'
    assert results.series(full,'flo_out',identities[1])==results.series(selected,'flo_out',identities[1])==([date(2018,1,1),date(2018,1,2)],[.7,.8])


def test_selected_nonfinite_reports_original_native_line(tmp_path):
    path=fixture(tmp_path,['1 1 1 2018 1 1 cha1 0.3','1 1 1 2018 2 2 cha2 nan'])
    assert results.output_objects(path)==[(1,1,'cha1'),(2,2,'cha2')]
    with pytest.raises(ValueError,match='第5行'):results.read_output(path,selected_identity=(2,2,'cha2'))


@pytest.mark.parametrize('rows',[
    ['1 1 1 2018 1 1 cha1 0.3','1 1 1 2018 1 1 cha1 0.4'],
    ['1 1 1 2018 1 1 cha1 0.3','3 1 3 2018 1 1 cha1 0.4'],
])
def test_selected_dates_still_reject_duplicate_or_gap(tmp_path,rows):
    with pytest.raises(ValueError):results.read_output(fixture(tmp_path,rows),selected_identity=(1,1,'cha1'))
