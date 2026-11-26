"""Synthetic daily fields exercise the real regrid, search and deployment path."""

import io

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.platform_search import submit_search
from ecodata.desktop import discovery, independent_alignment, local_search
from ecodata.models.engineering import precipitation_adaptation, precipitation_products


def test_preserved_daily_intervals_reach_real_seeded_mutation(tmp_path, monkeypatch):
    time = pd.date_range('2000-01-01', periods=12)
    bounds = np.column_stack((time.values, (time + pd.Timedelta(days=1)).values))
    products = []
    for index, size in enumerate((9, 17)):
        axis = np.linspace(0, 2, size)
        xx, yy = np.meshgrid(axis, axis)
        values = np.stack([10 + np.sin(xx * 2) + yy + day * 0.2 + index * 0.3 for day in range(len(time))])
        ds = xr.Dataset(dict(precipitation=(('time', 'lat', 'lon'), values), time_bounds=(('time', 'nv'), bounds)),
                        coords=dict(time=time, lat=30 + axis, lon=110 + axis), attrs=dict(crs='EPSG:4326'))
        ds.time.attrs['bounds'] = 'time_bounds'
        ds.precipitation.attrs.update(units='mm', spatial_semantics='cell_mean', data_kind='continuous',
                                      temporal_semantics='accumulation', interval_seconds=86400)
        products.append(dict(product=f'source-{index}', dataset=ds))

    monkeypatch.setattr(independent_alignment, 'assert_current', lambda scan: None)
    monkeypatch.setattr(independent_alignment, 'read_selected', lambda row, **kwargs: row['name'].encode())
    source_files = dict(zip(('a.nc', 'b.nc'), products, strict=True))
    monkeypatch.setattr('ecodata.recognition.recognise_file',
                        lambda name, *args, **kwargs: (source_files[name]['dataset'], None))
    monkeypatch.setattr(precipitation_products, 'inspect_product', lambda ds: dict(
        fields=[dict(field='precipitation')], source_id='fixture', product='fixture'))
    monkeypatch.setattr(discovery, 'audit_precipitation', lambda *args: dict(time_basis='UTC', blockers=[]))
    monkeypatch.setattr(precipitation_adaptation, 'adapt_products',
                        lambda *args: (products, [dict(product=row['product']) for row in products]))

    def actual_search(inputs, request, folder, **kwargs):
        for item in inputs:
            with xr.open_dataset(io.BytesIO(item.content)) as prepared:
                np.testing.assert_array_equal(prepared.time_bounds.values, bounds)
        return submit_search(inputs, request, None, folder,
                             budget=dict(objective_limit=130, total_call_limit=180, proposal_limit=20, wall_seconds=45))

    monkeypatch.setattr(local_search, 'submit', actual_search)
    task = dict(fields=[(0, 'precipitation'), (1, 'precipitation')], route='independent_grid_search',
                seeds=['spatial.blend'], recipes=[['spatial.regrid.conservative', 'spatial.blend']])
    result = independent_alignment.run(dict(rows=[dict(name='a.nc'), dict(name='b.nc')]),
                                       [('C2:fusion', task)], tmp_path / 'result')
    record = result['grid_searches']['C2:fusion']
    assert record['seed_usage']['scored'] == 1
    assert record['seed_usage']['mutations'] > 0
    assert record['rankings'][0]['selected'] and record['rankings'][0]['metrics']['count'] > 0
