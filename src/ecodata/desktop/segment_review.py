"""Recheck evidenced cadence segments; keep statistical change points advisory."""
import numpy as np


def change_candidates(values):
    import ruptures

    values = np.asarray(values, dtype=float)
    if len(values) < 12 or not np.isfinite(values).all():
        return []
    scale = float(np.std(values))
    if not scale:
        return []
    signal = (values - np.median(values)) / scale
    positions = ruptures.Pelt(model="l2", min_size=4, jump=1).fit(signal).predict(pen=3 * np.log(len(values)))
    return positions[:-1]


def review_netcdf(row, field, audit, *, precipitation=False):
    import xarray as xr

    from .release_gate import precipitation_trial
    from .sampling_plan import time_dimension, numerics
    from .scan import local_member

    if (audit["interval_changes"] > len(audit["change_examples"]) or
            audit.get("interval_width_changes", 0) > len(audit.get("interval_width_examples", ()))):
        return dict(status="pending", reason="cadence_changes_exceed_review_index", segments=[])
    with local_member(row) as path, xr.open_dataset(path, decode_times=False, cache=False) as dataset:
        array = dataset[field["name"]]
        time = time_dimension(array, dataset)
        if time not in dataset.coords:
            return dict(status="pending", reason="time_coordinate_missing", segments=[])
        clock = dataset[time]
        bounds = clock.attrs.get("bounds")
        size = clock.size
        interval_edges = audit.get("interval_width_examples", ())
        transitions = [*interval_edges, *(index for index in audit["change_examples"]
                                          if all(abs(index - edge) > 1 for edge in interval_edges))]
        ends = [0, *sorted({index for index in transitions if 0 < index < size}), size]
        fixed = {dim: (array.sizes[dim] - 1) // 2 for dim in array.dims if dim != time}
        reports = []
        for start, stop in zip(ends, ends[1:]):
            positions = sorted({start, max(start, (start + stop - 8) // 2), max(start, stop - 8)})
            sampled = []
            for lower in positions:
                window = {time: slice(lower, min(lower + 8, stop))}
                values = numerics(array.isel({**fixed, **window}).values.reshape(-1))
                if precipitation and bounds in dataset and time in dataset[bounds].dims:
                    edges = np.asarray(dataset[bounds].isel(window).transpose(time, ...).values)
                    sampled.append(dict(local_values=values, time=dict(
                        bounds=edges.tolist(), units=clock.attrs.get("units"), calendar=clock.attrs.get("calendar"),
                        intervals_valid=bool(edges.ndim == 2 and edges.shape[1] == 2 and
                                             np.isfinite(edges).all() and (edges[:, 1] > edges[:, 0]).all()))))
                else:
                    sampled.append(dict(local_values=values))
            size_valid = sum(value is not None for item in sampled for value in item["local_values"])
            report = dict(start=start, stop=stop, inspected=size_valid, status="checked" if size_valid else "failed")
            if precipitation:
                trial = precipitation_trial(dict(unit=array.attrs.get("units") or field.get("unit"), windows=sampled), field)
                if trial:
                    report["precipitation_trial"] = trial
                else:
                    report["status"] = "failed"
            reports.append(report)

        stride = max(1, size // 128)
        summaries, locations = [], []
        for index in range(0, size, stride):
            observed = array.isel({**fixed, time: slice(index, min(index + 4, size))}).values
            finite = observed[np.isfinite(observed)]
            if finite.size:
                summaries.append(float(finite.mean()))
                locations.append(index)
        changes = [locations[index] for index in change_candidates(summaries)]
        return dict(status="checked" if all(item["status"] == "checked" for item in reports) else "failed",
                    segments=reports, statistical_change_candidates=changes,
                    statistical_scope="a_bounded_series_of_observation_window_means")
