"""Synthetic mechanism boundaries, not environmental accuracy evidence."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.evaluation.spatial_structure import compare_fields,paired_grid
from ecodata.algorithms.spatial_selection_protocol import split_dates
from ecodata.algorithms.platform_search import DatasetReference


def field(values):
 return xr.DataArray(values,dims=('lat','lon'),coords=dict(lat=np.arange(values.shape[0]),lon=np.arange(values.shape[1])),attrs=dict(units='mm'),name='p')


def test_identical_and_smoothing_are_distinguished():
 a=field(np.indices((8,8)).sum(0)%2)
 good=compare_fields(a,a,threshold=.5)
 bad=compare_fields(a*0+.5,a,threshold=.5)
 assert good['gradient_RMSE_per_cell']==0 and good['normalized_spectrum_L1']==0
 assert good['FSS'][0]['value']==1
 assert bad['gradient_RMSE_per_cell']>0 and bad['normalized_spectrum_L1']>0
 assert bad['FSS'][0]['value']<1


def test_no_reference_or_threshold_is_not_invented():
 a=field(np.ones((8,8)))
 assert compare_fields(a,a)['FSS_status']=='THRESHOLD_NOT_DECLARED'
 a.values[3,3]=np.nan
 assert compare_fields(a,a)['spectrum_status']=='MISSING_VALUES_NO_ZERO_FILL'
 with pytest.raises(ValueError,match='unit'):compare_fields(a.assign_attrs(units='K'),a)


def test_public_reference_really_consumes_structure():
 a=field(np.indices((8,8)).sum(0)%2)
 result=DatasetReference(a,'p')((a*.5).to_dataset())
 assert result['spatial_structure']['gradient_RMSE_per_cell']>0
 assert result['RMSE']>0


def test_whole_dates_and_unchanged_metadata_split():
 clock=pd.date_range('2000-01-01',periods=12*4,freq='6h')
 a,b,receipt=split_dates(clock)
 assert not set(a.normalize())&set(b.normalize())
 assert receipt['sufficient'] and receipt['confirmation_dates']>=3
 assert min(b)>max(a)
 assert split_dates(clock.as_unit('us'))[2]==receipt
 _,_,short=split_dates(clock[:4])
 assert not short['sufficient'] and short['confirmation_dates']==0


def test_hidden_outer_values_cannot_enter_structure_metrics():
 q=np.column_stack([np.tile(np.arange(8),8),np.repeat(np.arange(8),8)])
 y=np.arange(64,dtype=float)[None,:];p=y+1
 selected=np.arange(32)
 first=paired_grid(p,y,q,[pd.Timestamp('2000')],selected,'K')
 y[:,32:]+=1000
 assert paired_grid(p,y,q,[pd.Timestamp('2000')],selected,'K')==first

def test_desktop_displays_scope_comparison(monkeypatch):
 monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
 from PySide6.QtWidgets import QApplication,QLabel
 from ecodata.desktop.output_review import selection_summaries
 from ecodata.desktop.result_cards import ResultCards
 app=QApplication.instance() or QApplication([])
 comparison=dict(candidate='ds_recommendation_0',phase='confirmation',status='EVALUATED',candidate_better=1,baseline_better=1,equivalent=0,
  blocks=[dict(block='day1|region1',lower_error='candidate'),dict(block='day2|region2',lower_error='baseline')])
 rows=selection_summaries([],[],[],result=dict(grid_searches={'task':dict(scope_report=dict(comparisons=[comparison]))}))
 widget=ResultCards();widget.render_selection(rows)
 text=' '.join(label.text() for label in widget.findChildren(QLabel))
 assert 'DS推荐与基线' in text and '独立复检' in text and '基线较好 1 块' in text
 widget.close();app.processEvents()
