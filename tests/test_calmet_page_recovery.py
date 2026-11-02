from unittest.mock import patch
from streamlit.testing.v1 import AppTest

SCRIPT='from ecodata.interface.calmet_run_page import render_calmet_run_page\nrender_calmet_run_page()'

def test_refresh_restores_route_and_result():
    app=AppTest.from_string(SCRIPT)
    app.query_params['calmet_job']='c2b0bba4fc2240a78557e902f4d5ad92'
    app.run()
    assert not app.exception
    assert app.radio[0].value=='从原始观测重建登记工程'
    assert app.success and app.metric[0].value=='36.220268 µg/m³'

def test_new_failure_does_not_show_previous_success():
    app=AppTest.from_string(SCRIPT)
    app.query_params['calmet_job']='c2b0bba4fc2240a78557e902f4d5ad92'
    app.run()
    with patch('ecodata.interface.calmet_run_page.execute',side_effect=RuntimeError('injected verification failure')):
        app.button[0].click().run()
    assert not app.exception
    assert app.error and 'injected verification failure' in app.error[0].value
    assert not app.success and not app.metric
