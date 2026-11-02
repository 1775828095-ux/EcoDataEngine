"""Verify current-data handoff and absence of implicit demo execution."""
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def page(model, mode, signature='first'):
    return AppTest.from_string(
        'from ecodata.interface.round1_page import render_project_continuation\n'
        f'render_project_continuation({model!r}, {mode!r}, b"current-zip", '
        f'b"current-source", b"current-config", {signature!r})'
    ).run()


def test_swat_receives_current_weather_and_original_inputs():
    with patch('ecodata.interface.swat_run_page.render_swat_run_page') as render:
        app = page('swat_plus', None)
        assert not render.called
        app.checkbox[0].check().run()
        assert not app.exception
        render.assert_called_once_with(converted_weather=b'current-zip',
                                       weather_inputs=(b'current-source', b'current-config'))


def test_wrf_receives_current_zip_without_registered_source_fallback():
    with patch('ecodata.interface.wrf_run_page.render_wrf_run_page') as render:
        app = page('wrf', 'pressure_fields')
        app.checkbox[0].check().run()
        assert not app.exception
        render.assert_called_once_with(converted_zip=b'current-zip')


def test_profile_categories_handoff_current_zip_without_fallback():
    for model, mode in [('cmaq', None), ('cmaq', 'bcon_profile')]:
        with patch('ecodata.interface.cmaq_run_page.render_cmaq_run_page') as render:
            app=page(model,mode);app.checkbox[0].check().run()
            assert not app.exception
            render.assert_called_once_with(converted_zip=b'current-zip',mode=mode)


def test_current_zip_handoff_to_new_registered_category_routes():
    cases=[('camx',None),('camx','gridded_emissions'),('camx','initial_concentrations'),
           ('camx','boundary_concentrations'),('cmaq','gridded_emissions'),('cmaq','boundary_concentrations')]
    cases += [('calpuff',m) for m in (None,'upper_air','precipitation','geography','constant_points','receptors')]
    for model,mode in cases:
        module='calmet' if model=='calpuff' else model
        with patch(f'ecodata.interface.{module}_run_page.render_{module}_run_page') as render:
            app=page(model,mode)
            assert not render.called
            app.checkbox[0].check().run()
            assert not app.exception
            render.assert_called_once_with(converted_zip=b'current-zip',mode=mode)


def test_wrf_package_uses_current_zip():
    with patch('ecodata.interface.wrf_run_page.render_wrf_run_page') as render:
        app=page('wrf','input_package');app.checkbox[0].check().run()
        assert not app.exception
        render.assert_called_once_with(converted_zip=b'current-zip')


def test_swat_parameter_types_pass_current_zip():
    for mode in ('wgn','soil','hru_attributes','landuse_database','aquifer_parameters','connections','spatial_landuse','management_schedule'):
        with patch('ecodata.interface.swat_run_page.render_swat_run_page') as render:
            app=page('swat_plus',mode);app.checkbox[0].check().run()
            assert not app.exception
            render.assert_called_once_with(converted_inputs=b'current-zip',input_mode=mode)


def test_wrf_terrain_and_sst_use_current_zip():
    for mode,actual in [('geogrid_tiles','geogrid_tiles'),(None,'sst')]:
        with patch('ecodata.interface.wrf_run_page.render_wrf_run_page') as render:
            app=page('wrf',mode);app.checkbox[0].check().run()
            assert not app.exception
            render.assert_called_once_with(converted_zip=b'current-zip',input_mode=actual)


def test_changed_conversion_clears_previous_job_and_station_confirmation():
    app = page('wrf', 'pressure_fields')
    app.session_state['wrf_job'] = 'old-job'
    app.session_state['run_weather_confirm'] = True
    app.session_state['continuation_fingerprint'] = 'older'
    app.run()
    assert 'wrf_job' not in app.session_state
    assert 'run_weather_confirm' not in app.session_state
    assert not app.exception
