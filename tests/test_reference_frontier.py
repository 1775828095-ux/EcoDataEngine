"""Library-equivalence and actual decision-consumer tests; synthetic fixtures."""
import numpy as np
import pytest
import xarray as xr
from ecodata.evaluation.spatial_structure import compare_fields
from ecodata.algorithms.reliable_selection import reference_frontier


def field(values):
 return xr.DataArray(values,dims=('lat','lon'),coords={'lat':np.arange(values.shape[0]),'lon':np.arange(values.shape[1])},attrs={'units':'mm'},name='rain')


def test_calls_official_scores_and_matches_it(monkeypatch):
 import scores.spatial
 actual=scores.spatial.fss_2d;calls=[]
 def spy(*args,**kwargs):
  calls.append(kwargs['window_size']);return actual(*args,**kwargs)
 monkeypatch.setattr(scores.spatial,'fss_2d',spy)
 a=field(np.indices((8,8)).sum(0)%2);b=field(np.roll(a.values,1,axis=0))
 result=compare_fields(a,b,threshold=.5)
 assert result['backend']['FSS']['package']=='scores'
 assert calls==[(1,1),(3,3),(5,5)]
 for row in result['FSS']:
  expected=actual(a,b,event_threshold=.5,window_size=(row['scale_pixels'],)*2,spatial_dims=('lat','lon'),zero_padding=False)
  assert row['value']==pytest.approx(float(expected))


def candidates():
 y=field((np.indices((8,8)).sum(0)%2).astype(float))
 forecasts={'smooth':y*0+.5,'detail':y+1,'bad':y+2}
 rows=[]
 for key,prediction in forecasts.items():
  rmse=float(np.sqrt(np.mean((prediction.values-y.values)**2)))
  rows.append(dict(key=key,status='PASS',algorithm_nodes=1,metrics=dict(objective=rmse,RMSE=rmse,
   spatial_structure=compare_fields(prediction,y))))
 return rows


def test_structure_changes_selected_set_without_arbitrary_weights():
 rows=candidates();result=reference_frontier(rows)
 assert set(result['keys'])=={'smooth','detail'} and result['tradeoff']
 assert result['minimum_error_key']=='smooth' and result['minimum_gradient_error_key']=='detail'
 assert result['weights'] is None and not result['universal_winner']
 assert all(r['metrics']['spatial_structure']['used_for_selection'] for r in rows)
 assert reference_frontier(list(reversed(rows)))==result


def test_unknown_and_different_reference_queue_cannot_win_by_zero_score():
 rows=candidates();rows[0]['metrics']['spatial_structure']={}
 assert reference_frontier(rows)['status']=='STRUCTURE_EVIDENCE_INCOMPLETE'
 rows=candidates();rows[0]['metrics']['spatial_structure']['support_sha256']='different'
 assert reference_frontier(rows)['status']=='STRUCTURE_QUEUES_DIFFER'


def test_measured_structure_option_is_actual_mutation_parent():
 from ecodata.algorithms.spatial_fusion_task import SpatialSearch
 search=SpatialSearch.__new__(SpatialSearch)
 search.rows=candidates();search.request={};search.recommended_parents=[];search.resolution=1e-12
 assert search.best()['key']=='smooth'
 assert search.recommendation_parent()=='detail'
 assert search.recommendation_parent() is None
 search.request={'spatial_objective_mode':'rmse_v1'}
 assert search.recommendation_parent() is None

def test_scope_reports_are_packaged_as_audit_not_input_tables():
 import io,json,zipfile
 from ecodata.desktop.service import bundle
 result=dict(search=None,files=[('product.nc',b'actual artifact placeholder for archive boundary only')],
  original_report=[],local={'artifacts':[]},mode='fusion',source_formats=['NetCDF'],target_format='NetCDF',
  grid_searches={'task':dict(evidence=dict(scope_report={'comparisons':[]}))})
 with zipfile.ZipFile(io.BytesIO(bundle(result))) as archive:
  assert json.loads(archive.read('scope_reports.json'))['task']=={'comparisons':[]}
