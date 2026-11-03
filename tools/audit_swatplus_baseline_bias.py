"""Read-only diagnosis of the pinned LREW baseline; never changes model inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "output/swatplus_baseline_20261026"
    out = root / "bias_audit"
    out.mkdir(exist_ok=True)
    original = root / "lrew_extracted/plus_rev59_project/swatplus_rev59_demo"
    run = root / "runs/lrew_water_audit"
    baseline = root / "runs/lrew_verified"
    for file in original.iterdir():
        # The upstream archive also bundles one historical output, not an input.
        if file.is_file() and file.name not in {"print.prt", "channel_yr.txt"}:
            assert file.read_bytes() == (run / file.name).read_bytes()
    assert (run / "channel_day.txt").read_bytes() == (baseline / "channel_day.txt").read_bytes()
    routing = []
    for line in (run / "rout_unit.con").read_text().splitlines()[2:]:
        fields = line.split()
        if not fields:
            continue
        links = [dict(destination=fields[i], object_id=int(fields[i + 1]),
                      hydrograph=fields[i + 2], fraction=float(fields[i + 3]))
                 for i in range(13, 13 + int(fields[12]) * 4, 4)]
        routing.append(dict(routing_unit=fields[1], links=links,
                            total_flow_fraction=sum(x["fraction"] for x in links
                                                    if x["hydrograph"] == "tot")))
    assert len(routing) == 3 and all(x["total_flow_fraction"] == 2 for x in routing)

    def read_output(name: str) -> pd.DataFrame:
        return pd.read_csv(run / name, sep=r"\s+", skiprows=[0, 2])

    hru = pd.read_csv(run / "hru.con", sep=r"\s+", skiprows=1)
    weights = hru.groupby("wst").area.sum() / hru.area.sum()
    aq = read_output("aquifer_yr.txt")
    areas = pd.read_csv(run / "aquifer.con", sep=r"\s+", skiprows=1)[["id", "area"]]
    aq = aq.merge(areas, left_on="unit", right_on="id", validate="many_to_one")
    aq["recharge_m3"] = aq.rchrg * aq.area * 10
    aq["groundwater_out_m3"] = aq.flo * aq.area * 10
    aq["deep_seep_m3"] = aq.seep * aq.area * 10
    budget = aq.groupby("yr")[["recharge_m3", "groundwater_out_m3", "deep_seep_m3"]].sum()
    wb = read_output("basin_wb_yr.txt").set_index("yr")
    budget["land_yield_m3"] = wb.wateryld * hru.area.sum() * 10
    budget["land_perc_mm"] = wb.perc
    daily = pd.read_csv(baseline / "outlet_daily_comparison.csv", parse_dates=["date"])
    daily["yr"] = daily.date.dt.year
    annual = daily.groupby("yr")[["simulation_m3_s", "discharge"]].sum() * 86400
    budget["outlet_m3"] = annual.simulation_m3_s
    budget["observed_m3"] = annual.discharge
    budget["bias_m3"] = budget.outlet_m3 - budget.observed_m3
    budget["outlet_minus_land_and_aquifer"] = (
        budget.outlet_m3 - budget.land_yield_m3 - budget.groundwater_out_m3)
    budget.to_csv(out / "annual_water_budget.csv")
    assert (budget.land_perc_mm == 0).all()
    assert ((budget.recharge_m3 / budget.land_yield_m3 - 1).abs() < 0.0001).all()

    # Trace actual model precipitation on missing days using unchanged station-area weights.
    precip = {}
    for name, file in {"sta01": "lrew146.pcp", "sta02": "lrew137.pcp",
                       "sta03": "lrew138.pcp"}.items():
        frame = pd.read_csv(run / file, sep=r"\s+", skiprows=3, header=None).iloc[:, :3]
        frame["date"] = (pd.to_datetime(frame[0].astype(str), format="%Y")
                         + pd.to_timedelta(frame[1] - 1, unit="D"))
        precip[name] = frame.set_index("date")[2]
    wb_day = read_output("basin_wb_day.txt")
    wb_day["date"] = pd.to_datetime(dict(year=wb_day.yr, month=wb_day.mon, day=wb_day.day))
    actual = wb_day.set_index("date").precip
    supplied = pd.DataFrame(precip).reindex(actual.index)
    missing = supplied == -99
    known = supplied.where(~missing, 0).mul(weights).sum(axis=1)
    missing_fraction = missing.mul(weights).sum(axis=1)
    trace = pd.DataFrame({"basin_precip_mm": actual, "known_weighted_rain_mm": known,
                          "generated_contribution_mm": actual - known,
                          "missing_area_fraction": missing_fraction})
    # Small negative inferred values are output rounding, not negative generated rain.
    trace["inferred_generated_station_mm"] = (trace.generated_contribution_mm
                                               / missing_fraction.replace(0, np.nan))
    bad = missing.any(axis=1)
    trace.loc[bad].to_csv(out / "missing_precipitation_trace.csv")
    total = budget.sum()
    source = out / "source59"
    assert 'Rev 2019.59.3' in (source / "main.f90").read_text()
    assert 'do ihyd = 3, 5' in (source / "hru_hyds.f90").read_text()
    assert 'ch_d(jrch)%flo_out = rtwtr_d / 10000.' in (source / "channel_control.f90").read_text()
    summary = dict(
        inputs_changed_only_print=True, diagnostic_channel_output_identical=True,
        upstream_cached_output_excluded_from_input_audit="channel_yr.txt",
        routing=routing, budget_totals_m3=total.drop("land_perc_mm").to_dict(),
        groundwater_to_total_bias_ratio=float(total.groundwater_out_m3 / total.bias_m3),
        recharge_to_land_yield_ratio=float(total.recharge_m3 / total.land_yield_m3),
        generated_precipitation_area_mean_mm=float(trace.loc[bad, "generated_contribution_mm"].sum()),
        generated_precipitation_volume_m3=float(trace.loc[bad, "generated_contribution_mm"].sum()
                                               * hru.area.sum() * 10),
        generated_precipitation_fraction=float(trace.loc[bad, "generated_contribution_mm"].sum()
                                               / actual.sum()),
        missing_days=int(bad.sum()),
        nonmissing_precipitation_max_rounding_error_mm=float((actual - known).loc[~bad].abs().max()),
        best_lag_in_minus2_to_plus2=max(range(-2, 3),
                                      key=lambda lag: daily.simulation_m3_s.corr(daily.discharge.shift(lag))),
        flow_correlation=float(daily.simulation_m3_s.corr(daily.discharge)),
        source_build_provenance="Same revision/date archival source; binary not rebuilt from source",
        calibration_performed=False, fusion_performed=False, physical_inputs_modified=False,
    )
    (out / "bias_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    records = [dict(file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    url="https://raw.githubusercontent.com/dhardestylewis/swatplus_predocker/"
                        "7076625db0a2c9fb88efaf60d55fa3b296c4a3a1/source_codes/" + file.name)
               for file in sorted(source.glob("*.f90"))]
    (out / "source59_provenance.json").write_text(json.dumps(dict(
        repository="dhardestylewis/swatplus_predocker",
        commit="7076625db0a2c9fb88efaf60d55fa3b296c4a3a1", files=records), indent=2), encoding="utf-8")
    assert json.loads((out / "bias_audit_summary.json").read_text(encoding="utf-8"))["missing_days"] == 11
    saved = pd.read_csv(out / "annual_water_budget.csv")
    assert abs(saved.outlet_m3.sum() - total.outlet_m3) < 1e-6
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
