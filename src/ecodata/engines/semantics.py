"""CF evidence, without guessing standard names or coordinate systems."""
import cf_xarray  # noqa: F401 - registers the accessor
from pyproj import CRS


class SemanticEngine:
    def inspect(self, dataset):
        result = {"backend": "cf-xarray", "axes": dataset.cf.axes,
                  "coordinates": dataset.cf.coordinates, "bounds": dataset.cf.bounds,
                  "standard_names": dataset.cf.standard_names, "ambiguities": []}
        for role, candidates in {**result["axes"], **result["coordinates"]}.items():
            if len(candidates) > 1:
                result["ambiguities"].append({"role": role, "candidates": candidates})
        mappings = [name for name in dataset.variables if "grid_mapping_name" in dataset[name].attrs]
        result["grid_mappings"] = mappings
        result["crs"] = None
        if len(mappings) == 1:
            try:
                result["crs"] = CRS.from_cf(dataset[mappings[0]].attrs).to_wkt()
            except (ValueError, KeyError) as exc:
                result["crs_error"] = str(exc)
        return result

    @staticmethod
    def coordinate(evidence, *roles):
        for role in roles:
            candidates = set(evidence["axes"].get(role, ())) | set(evidence["coordinates"].get(role, ()))
            if candidates:
                return next(iter(candidates)) if len(candidates) == 1 else None
        return None
