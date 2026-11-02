import io
import zipfile
import pytest
from ecodata.models.swat_weather_project import _read_zip
from ecodata.interface import swat_results,swat_runner

def archive(extra):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('author/file.cio',b'original control bytes')
        z.writestr('author/time.sim',b'original dates')
        for name,data in extra.items():z.writestr(name,data)
    return buf.getvalue()

def test_macos_metadata_does_not_require_user_repacking():
    raw=archive({'__MACOSX/author/._file.cio':b'finder','author/.DS_Store':b'finder'})
    assert _read_zip(raw)=={'file.cio':b'original control bytes','time.sim':b'original dates'}

def test_other_scientific_directory_is_not_silently_discarded():
    with pytest.raises(ValueError,match='平铺'):
        _read_zip(archive({'another/soil.sol':b'actual data'}))

def test_unsafe_metadata_is_still_rejected():
    with pytest.raises(ValueError,match='不安全'):
        _read_zip(archive({'__MACOSX/../outside':b'unsafe'}))

def test_chart_validation_uses_explicit_selected_engine(monkeypatch):
    received=[]
    def checked(files,engine_id=None):
        received.append(engine_id)
        return {'errors':['sentinel validation failure']}
    monkeypatch.setattr(swat_runner,'check_project',checked)
    with pytest.raises(ValueError,match='sentinel'):
        swat_results.enable_charts({},engine_id='official61')
    assert received==['official61']

def test_date_validation_uses_explicit_selected_engine(monkeypatch):
    import datetime
    received=[]
    def checked(files,engine_id=None):
        received.append(engine_id)
        return {'errors':['sentinel date failure']}
    monkeypatch.setattr(swat_runner,'check_project',checked)
    with pytest.raises(Exception,match='sentinel date'):
        swat_runner.set_dates({},datetime.date(2021,1,1),datetime.date(2021,12,31),engine_id='official61')
    assert received==['official61']
