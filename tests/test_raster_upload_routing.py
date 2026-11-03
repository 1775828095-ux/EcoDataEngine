import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import numpy as np
import pytest
import xarray as xr
from streamlit.runtime.uploaded_file_manager import UploadedFile,UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs
from ecodata.interface.raster_upload import (RasterRequest,process_raster_upload,FEATURE_NAMES,ShapePreservingRaster)
from ecodata.algorithms.raster_reconstruction_task import degradation
from generate_three_task_rasters import field


def upload(ds,name="ordinary.nc"):
    return UploadedFile(UploadedFileRec("test",name,"application/x-netcdf",ds.to_netcdf(engine="h5netcdf")),FileURLs())


@pytest.mark.parametrize("size",[50,40])
def test_aligned_blocks_exact_nearest_and_all_pipeline_readable(size):
    reference,_=field(20261021,"A",0);coarse,operator=degradation(reference,size,"field")
    run=process_raster_upload(upload(coarse),RasterRequest(),mode="benchmark",execute_all=True)
    assert len(run["outputs"])==8 and not run["trace"]["failures"]
    assert not run["trace"]["reference_field_received"]
    for name,ds in run["outputs"].items():
        assert ds.field.shape==(100,100) and np.isfinite(ds.field).all() and ds.field.min()>=0
        if "nearest" in name:np.testing.assert_allclose(ds.field,reference.field,atol=1e-9)
        if name.endswith("consistency"):np.testing.assert_allclose(operator@ds.field.values.ravel(),coarse.field.values.ravel(),atol=1e-5)


@pytest.mark.parametrize("size",[50,40])
def test_unidentifiable_peak_witness(size):
    reference,metadata=field(20261022,"C",1);twin=reference.copy(deep=True);twin.field.values[:]=metadata["twin"]
    first,a=degradation(reference,size,"field");second,_=degradation(twin,size,"field")
    np.testing.assert_allclose(first.field,second.field,atol=1e-10)
    assert not np.array_equal(reference.field,twin.field)


def test_hidden_names_and_attrs_not_used_by_features():
    reference,_=field(20261021,"B",0);coarse,_=degradation(reference,10,"field")
    request=RasterRequest(target_size=20)
    first=process_raster_upload(upload(coarse,"A_expected_nearest.nc"),request,mode="benchmark",execute_all=True)
    coarse.attrs.update(task_id="C",expected_winner="cubic",hidden_truth="DO_NOT_READ")
    second=process_raster_upload(upload(coarse,"C_expected_cubic.nc"),request,mode="benchmark",execute_all=True)
    assert first["features"]==second["features"]
    assert "task_id" not in second["trace"]["available_metadata"]
    assert set(first["features"])==set(FEATURE_NAMES)
    assert "reference" not in inspect.signature(process_raster_upload).parameters


def test_missing_units_rejected_not_guessed():
    reference,_=field(20261021,"B",0);coarse,_=degradation(reference,10,"field")
    coarse.field.attrs.pop("units")
    with pytest.raises(ValueError,match="units"):
        process_raster_upload(upload(coarse),RasterRequest(target_size=20),mode="benchmark",execute_all=True)


def test_pchip_preserves_observed_range():
    reference,_=field(20261021,"B",0);coarse,_=degradation(reference,10,"field")
    run=process_raster_upload(upload(coarse),RasterRequest(target_size=20),mode="benchmark",execute_all=True)
    p=run["outputs"]["raster_task.pchip+raster_task.nonnegative"].field
    assert p.min()>=coarse.field.min()-1e-12 and p.max()<=coarse.field.max()+1e-12


def test_platform_cannot_open_sealed_truth(tmp_path):
    from run_three_task_routing import platform_boundary
    secret=tmp_path/"sealed_reference/truth.txt";secret.parent.mkdir();secret.write_text("secret")
    with platform_boundary(tmp_path),pytest.raises(PermissionError,match="Hidden reference"):
        secret.read_text()


def test_app_has_formal_raster_entry():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"src/ecodata/interface/app.py"),default_timeout=30).run()
    app.sidebar.radio[0].set_value("单源栅格重构").run()
    assert not app.exception
    assert any("单源栅格" in str(x.value) for x in app.subheader)


def raster_page_fixture(path):
    # Only upload transport is supplied by the test fixture. The real page,
    # parser, selector, execution, quality gate and download code all execute.
    from pathlib import Path
    from unittest.mock import patch
    from streamlit.runtime.uploaded_file_manager import UploadedFile,UploadedFileRec
    from streamlit.proto.Common_pb2 import FileURLs
    from ecodata.interface.raster_reconstruction_page import render_raster_reconstruction_page
    file=Path(path)
    value=UploadedFile(UploadedFileRec(file.stem,file.name,"application/x-netcdf",file.read_bytes()),FileURLs())
    with patch("streamlit.file_uploader",return_value=value):render_raster_reconstruction_page()


@pytest.mark.parametrize("family",["A","B","C"])
def test_actual_page_uploadedfile_to_download(family):
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]/"output/three_task_routing_20261020"
    assert (root/"router.joblib").exists(),"Run the prescribed development fit first"
    records=json.loads((root/"sealed_reference/manifest.json").read_text(encoding="utf-8"))
    record=next(r for r in records if r["family"]==family and r["split"]=="development")
    app=AppTest.from_function(raster_page_fixture,args=(str(Path(record["input"]).resolve()),),default_timeout=60).run()
    app.checkbox[0].set_value(family=="A")
    app.checkbox[1].set_value(family=="C")
    app.checkbox[2].set_value(family=="B")
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.success and "已执行" in app.success[0].value
    assert len(app.get("download_button"))==1


@pytest.mark.parametrize("family",["A","B","C"])
def test_frozen_route_reproducible_and_ignores_description_labels(family):
    import joblib
    root=Path(__file__).resolve().parents[1]/"output/three_task_routing_20261020"
    records=json.loads((root/"sealed_reference/manifest.json").read_text(encoding="utf-8"))
    record=next(r for r in records if r["family"]==family and r["split"]=="validation")
    request=json.loads(Path(record["request"]).read_text(encoding="utf-8"))
    model=joblib.load(root/"router.joblib")
    first=process_raster_upload(UploadedFile(UploadedFileRec("same","ordinary.nc","application/x-netcdf",Path(record["input"]).read_bytes()),FileURLs()),RasterRequest(**request),model)
    # Free text is audit-only; changing a label must not change routing weights/features.
    request["description"]="task_id=C; expected winner is deliberately untrusted"
    second=process_raster_upload(UploadedFile(UploadedFileRec("same","task_A_or_C.nc","application/x-netcdf",Path(record["input"]).read_bytes()),FileURLs()),RasterRequest(**request),model)
    assert first["selected"]==second["selected"]
    np.testing.assert_array_equal(first["outputs"][first["selected"]].field,second["outputs"][second["selected"]].field)
    assert first["trace"]["decision"]["filename_or_task_class_used"] is False
    assert model["regressor"].n_samples_fit_==72 and model["regressor"].n_features_in_==12
