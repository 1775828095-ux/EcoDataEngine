"""Numerical and UI guards only; not counted as real model experiments."""
from unittest.mock import patch
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from ecodata.interface.camx_vertical import interpolation_weights,interpolate
from ecodata.interface import camx_runner
from netCDF4 import Dataset


def test_original_gfs_disallowed_for_existing_route():
    with patch.object(camx_runner.threading,'Thread') as worker:
        with pytest.raises(ValueError,match='新建路线'):
            camx_runner.start('existing',new_meteorology=True)
        worker.assert_not_called()


def test_agl_interpolation_preserves_linear_profile_and_constant():
    source=np.array([10.,100.,400.])[:,None,None]
    target=np.array([20.,70.,200.])[:,None,None]
    weights=interpolation_weights(source,target)
    np.testing.assert_allclose(interpolate(2*source+3,weights),2*target+3,rtol=0,atol=1e-5)
    np.testing.assert_array_equal(interpolate(np.full_like(source,7.),weights),np.full_like(target,7.))


@pytest.mark.parametrize('target',[[9.,50.],[50.,401.],[50.,np.nan],[80.,50.]])
def test_agl_refuses_extrapolation_nonfinite_and_reversed_layers(target):
    source=np.array([10.,100.,400.])[:,None,None]
    with pytest.raises(ValueError):interpolation_weights(source,np.array(target)[:,None,None])


def test_new_meteorology_page_passes_explicit_flag():
    app=AppTest.from_string('from ecodata.interface.camx_run_page import render_camx_run_page\nrender_camx_run_page(b"current_zip",None)')
    app.run();assert not app.exception and not app.checkbox
    app.radio[0].set_value('用配套资料重新组装登记工程').run()
    app.checkbox[0].check().run()
    with patch('ecodata.interface.camx_run_page.start',side_effect=ValueError('test-no-native-launch')) as start:
        app.button[0].click().run()
    assert not app.exception and app.error
    start.assert_called_once_with('rebuild',b'current_zip',None,True)


def test_existing_new_met_project_page_passes_task_id():
    app=AppTest.from_string('from ecodata.interface.camx_run_page import render_camx_run_page\nrender_camx_run_page(b"current_zip","initial_concentrations")')
    app.run();app.text_input[0].set_value('a'*32).run()
    with patch('ecodata.interface.camx_run_page.start',side_effect=ValueError('test-no-native-launch')) as start:
        app.button[0].click().run()
    assert not app.exception and app.error
    start.assert_called_once_with('existing',b'current_zip','initial_concentrations',meteorology_project='a'*32)


@pytest.mark.parametrize('route,new,project',[('rebuild',False,'a'*32),('existing',True,'a'*32),('existing',False,'../bad')])
def test_project_id_and_route_refused_before_native_start(route,new,project):
    with patch.object(camx_runner.threading,'Thread') as worker:
        with pytest.raises(ValueError):camx_runner.start(route,new_meteorology=new,meteorology_project=project)
        worker.assert_not_called()


def test_failed_new_met_project_never_falls_back_to_official_case(tmp_path,monkeypatch):
    runner=camx_runner;monkeypatch.setattr(runner,'RUNS',tmp_path)
    monkeypatch.setattr(runner,'sha',lambda p:runner.PIN)
    monkeypatch.setattr(runner,'assemble',lambda *a,**kw:{})
    ident='b'*32;project='c'*32
    for n,data in [(ident,dict(id=ident,route='existing',status='running')),(project,dict(id=project,route='rebuild',status='failed',new_meteorology_requested=True))]:
        folder=runner.directory(n);folder.mkdir();runner.save(folder,data)
    assert runner.LOCK.acquire(blocking=False)
    with patch.object(runner.subprocess,'run') as native:
        runner.execute(ident,'existing',meteorology_project=project)
        native.assert_not_called()
    assert runner.status(ident)['status']=='failed'
    assert '已成功' in runner.status(ident)['error'] and not runner.LOCK.locked()


@pytest.mark.parametrize('bad',[np.nan,np.inf,-9999.,'partial_mask'])
def test_non_o3_invalid_output_cannot_be_hidden_by_valid_o3_plot(tmp_path,bad):
    output=tmp_path/'outputs';output.mkdir()
    d=Dataset('fixture','w',memory=65536,format='NETCDF3_CLASSIC')
    try:
        for key,size in [('TSTEP',1),('LAY',1),('ROW',2),('COL',3),('VAR',1),('DATE-TIME',2)]:d.createDimension(key,size)
        d.TSTEP=500
        d.createVariable('TFLAG','i4',('TSTEP','VAR','DATE-TIME'))[:]=[[[2016162,0]]]
        o3=d.createVariable('O3','f4',('TSTEP','LAY','ROW','COL'));o3.units='ppmv';o3[:]=.04
        no=d.createVariable('NO','f4',('TSTEP','LAY','ROW','COL'),fill_value=-9999.)
        if isinstance(bad,str) and bad=='partial_mask':
            no[:]=.001
            no[0,0,0,0]=-9999.
        else:no[:]=bad
        (output/'CAMx.20160610.avrg.grd01.nc').write_bytes(bytes(d.close()));d=None
    finally:
        if d is not None:d.close()
    from matplotlib import pyplot as plt
    try:
        with pytest.raises(ValueError,match='NO'):camx_runner.plot_outputs(tmp_path,1)
    finally:plt.close('all')
