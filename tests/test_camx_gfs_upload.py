from unittest.mock import patch
import pytest
from ecodata.interface.camx_meteorology import validate_gfs_uploads,PINS
from ecodata.interface import camx_runner

def test_missing_times_refused_before_native():
    with patch.object(camx_runner.threading,'Thread') as native:
        with pytest.raises(ValueError,match='三份'):camx_runner.start('rebuild',new_meteorology=True,raw_gfs_files={})
        native.assert_not_called()

def test_renamed_wrong_content_never_falls_back():
    with pytest.raises(ValueError,match='不回退'):validate_gfs_uploads({n:b'GRIB-wrong' for n in PINS})

def test_uploads_cannot_silently_change_existing_route():
    with pytest.raises(ValueError,match='上传原始GFS'):camx_runner.start('existing',raw_gfs_files={})
