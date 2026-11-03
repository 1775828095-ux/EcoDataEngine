"""Inspect source headers, without inferring timezone or datum from values."""
import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/taiwan_2017_zenodo4420335"
OUT = ROOT / "output/taiwan_final_20261019"


def serial(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value.tolist() if hasattr(value, "tolist") else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-archive", action="store_true")
    args = parser.parse_args()
    if args.verified_archive:
        source = json.loads((DATA / "verified_taiwan_source_manifest.json").read_text(encoding="utf-8"))
        members = source["taiwan_members"]
        target = OUT / "header_audit_verified_archive.json"
    else:
        members = json.loads((DATA / "source_manifest.json").read_text(encoding="utf-8"))["members"]
        target = OUT / "header_audit_prefix.json"
    if target.exists():
        raise ValueError("Preserve existing header audit rather than overwrite")
    rows = []
    for member in members:
        if not member["member"].lower().endswith(".nc"):
            continue
        path = DATA / member["file"] if "file" in member else DATA / "provisional_members" / member["member"]
        with h5py.File(path) as source:
            attrs = {key:serial(value) for key, value in source.attrs.items() if not key.startswith("_NC")}
            clock = [key for key in source if any(term in key.lower() for term in ["time", "date", "utc"])]
            clock_attrs = {key:value for key, value in attrs.items() if any(term in key.lower() for term in ["time", "date", "utc"])}
            datum = {key:value for key, value in attrs.items() if any(term in key.lower() for term in ["datum", "epsg", "projection", "spatial_ref", "crs"])}
            rows.append(dict(member=member["member"], keys=list(source), global_attributes=attrs,
                clock_variables=clock, clock_attributes=clock_attrs, datum_attributes=datum,
                variable_georeferencing={name:{key:serial(value) for key,value in source[name].attrs.items()
                    if key in {"grid_mapping", "spatial_ref", "crs_wkt", "standard_name", "units"}}
                    for name in source if hasattr(source[name], "attrs")}))
            if args.verified_archive:
                aod = source["AOD550"]
                raw = aod[...]
                values = raw * aod.attrs.get("scale_factor", 1) + aod.attrs.get("add_offset", 0)
                qa = source["QCAll"][...]
                low, high = aod.attrs["valid_range"]
                valid = np.isfinite(values) & (raw >= low) & (raw <= high) & np.isin(qa, [0, 1])
                for key in ["_FillValue", "missing_value"]:
                    if key in aod.attrs:
                        valid &= raw != aod.attrs[key]
                lon, lat = source["Longitude"][...], source["Latitude"][...]
                coordinates_valid = np.isfinite(lon) & np.isfinite(lat) & (abs(lon) <= 180) & (abs(lat) <= 90)
                valid &= coordinates_valid
                if not (values.shape == qa.shape == lon.shape == lat.shape):
                    raise ValueError(f"Inconsistent source array shapes: {member['member']}")
                rows[-1]["array_quality"] = dict(shape=list(values.shape), pixels=int(values.size),
                    high_medium_valid_pixels=int(valid.sum()), valid_fraction=float(valid.mean()),
                    valid_aod_min=float(values[valid].min()) if valid.any() else None,
                    valid_aod_max=float(values[valid].max()) if valid.any() else None,
                    valid_negative_aod_pixels=int((valid & (values < 0)).sum()),
                    qa_counts={str(int(code)):int(count) for code, count in zip(*np.unique(qa, return_counts=True))},
                    coordinate_valid_fraction=float(coordinates_valid.mean()),
                    longitude_range=[float(lon[coordinates_valid].min()), float(lon[coordinates_valid].max())],
                    latitude_range=[float(lat[coordinates_valid].min()), float(lat[coordinates_valid].max())])
        if len(rows) % 500 == 0:
            print(f"Source files actually inspected: {len(rows)}", flush=True)
    result = dict(files=len(rows), rows=rows, verified_archive=args.verified_archive,
        files_with_clock_variables=sum(bool(row["clock_variables"]) for row in rows),
        files_with_clock_attributes=sum(bool(row["clock_attributes"]) for row in rows),
        files_with_explicit_global_datum=sum(bool(row["datum_attributes"]) for row in rows),
        note="Algorithm release dates, template System strings and CF declaration do not establish observation timestamps or datum")
    if args.verified_archive:
        result["filename_dates_not_verified_observation_times"] = dict(Counter(Path(row["member"]).parent.name for row in rows))
        result["all_arrays_read"] = True
        result["mean_valid_pixel_fraction"] = float(np.mean([row["array_quality"]["valid_fraction"] for row in rows]))
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
