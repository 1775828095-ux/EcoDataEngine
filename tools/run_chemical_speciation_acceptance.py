"""Apply true EPA S2S mechanism mass fractions with official emiproc.speciate."""
import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from emiproc.inventories import Inventory
from emiproc.speciation import speciate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("gspro");p.add_argument("mw");p.add_argument("output");p.add_argument("config")
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert cfg["mechanism"]=="CB6R5_AE7" and cfg["input_pollutant"]=="TOG" and cfg["unit"]=="kg"
    text=Path(args.gspro).read_text()
    assert "#S2S_MECH_BASIS      CB6R5_AE7" in text
    table=pd.read_csv(args.gspro,sep=r"\s+",comment="#",header=None,dtype={0:str})
    rows=table[(table[0]==cfg["profile_code"])&(table[1]=="TOG")].copy()
    # NMOG is an overlapping diagnostic quantity, not an additional conserved species.
    diagnostics=rows.loc[rows[2]=="NMOG",[2,5]].values.tolist()
    rows=rows[rows[2]!="NMOG"]
    assert len(rows)>1 and rows[2].is_unique and np.isfinite(rows[[3,4,5]]).all().all()
    mw=pd.read_csv(args.mw); mw=mw[mw.Mechanism==cfg["mechanism"]].set_index("Species").SPEC_MW
    np.testing.assert_allclose(rows[4].values,mw.loc[rows[2]].values,rtol=1e-6)
    fractions=rows[5].to_numpy()
    assert np.all(fractions>=0) and abs(fractions.sum()-1)<1e-6
    mass=float(cfg["input_mass"])
    assert mass>0 and np.isfinite(mass)
    inv=Inventory.from_gdf(gpd.GeoDataFrame({("profile","TOG"):[mass]},geometry=[None]))
    ratios=xr.DataArray(fractions[None],dims=("speciation","substance"),coords={"speciation":[0],"substance":rows[2].values})
    output=speciate(inv,"TOG",ratios)
    values=np.array([output.gdf[("profile",key)].iloc[0] for key in rows[2]])
    np.testing.assert_allclose(values,mass*fractions,rtol=1e-12)
    error=float(abs(values.sum()-mass)/mass)
    assert error<1e-6
    ds=xr.Dataset({"mass":("species",values),"mass_fraction":("species",fractions),"molecular_weight":("species",rows[4].values),
        "amount":("species",values*1000/rows[4].values)},coords={"species":rows[2].values},
        attrs={"mechanism":cfg["mechanism"],"profile_code":cfg["profile_code"],"source":"EPA S2S/SPECIATE official CB6R5 GSPRO",
            "source_mass_basis":cfg["mass_basis_description"],"native_model_verified":"false"})
    ds.mass.attrs["units"]="kg";ds.mass_fraction.attrs["units"]="1"
    ds.molecular_weight.attrs["units"]="g/mol";ds.amount.attrs["units"]="mol"
    ds.to_netcdf(args.output,engine="h5netcdf")
    report={"mechanism":cfg["mechanism"],"profile_code":cfg["profile_code"],"official_entry_point":"emiproc.speciation.speciate",
        "source_sha256":hashlib.sha256(Path(args.gspro).read_bytes()).hexdigest(),"species":rows[2].tolist(),
        "input_mass_kg":mass,"output_mass_kg":float(values.sum()),"mass_relative_error":error,
        "coefficient_sum":float(fractions.sum()),"renormalized":False,"excluded_overlapping_diagnostics":diagnostics,
        "mole_check":"moles = mass_kg * 1000 / MW_g_per_mol; total molecule count is not constrained across lumped species",
        "MODEL_NATIVE_VERIFIED":False,"source_is_measured_inventory":False}
    Path(args.output+".json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":main()
