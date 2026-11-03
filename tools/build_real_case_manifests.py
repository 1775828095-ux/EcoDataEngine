#!/usr/bin/env python3
"""Create provenance manifests for the five-model real-case output folder."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_ROOT = Path(r"C:\Users\ASUS\Documents\Codex\2026-10-02\wo\outputs")
REAL_CASES = OUTPUT_ROOT / "real_cases"

SOURCES = {
    "CALPUFF_Update_Memo_12032013.pdf": {
        "url": "https://gaftp.epa.gov/Air/aqmg/SCRAM/models/other/calpuff/CALPUFF_Update_Memo_12032013.pdf",
        "license": "EPA public distribution as posted; CALPUFF software has separate EULA",
        "version": "EPA SCRAM posting, 2013 memo",
    },
    "Previous_SCRAM_CALPUFF_Posting_Reference.pdf": {
        "url": "https://gaftp.epa.gov/Air/aqmg/SCRAM/models/other/calpuff/Previous_SCRAM_CALPUFF_Posting_Reference.pdf",
        "license": "EPA public distribution as posted",
        "version": "EPA SCRAM posting reference",
    },
    "calpuff_eula.htm": {
        "url": "https://calpuff.org/calpuff/calpuff_eula.htm",
        "license": "CALPUFF EULA, see source file",
        "version": "CALPUFF 7.2.1 reference",
    },
    "v7.40.specific.inputs.260801.tgz": {
        "url": "https://www.camx.com/getmedia/v7.40.specific.inputs.260801.tgz",
        "license": "CAMx distribution terms; see source package LICENSE/README",
        "version": "CAMx 7.40",
    },
    "CAMx_v7.40.src.260801.tgz": {
        "url": "https://www.camx.com/getmedia/CAMx_v7.40.src.260801.tgz",
        "license": "CAMx distribution terms; see source package LICENSE",
        "version": "CAMx 7.40",
    },
    "CAMxUsersGuide_v7.40.pdf": {
        "url": "https://www.camx.com/Files/CAMxUsersGuide_v7.40.pdf",
        "license": "CAMx distribution terms",
        "version": "CAMx 7.40",
    },
    "namelist.input": {
        "url": "https://raw.githubusercontent.com/wrf-model/WRF/master/test/em_real/namelist.input",
        "license": "WRF open-source repository terms",
        "version": "official WRF repository master",
    },
    "examples.namelist": {
        "url": "https://raw.githubusercontent.com/wrf-model/WRF/master/test/em_real/examples.namelist",
        "license": "WRF open-source repository terms",
        "version": "official WRF repository master",
    },
    "namelist.wps": {
        "url": "https://raw.githubusercontent.com/wrf-model/WPS/master/namelist.wps",
        "license": "WPS open-source repository terms",
        "version": "official WPS repository master",
    },
    "GEOGRID.TBL.ARW": {
        "url": "https://raw.githubusercontent.com/wrf-model/WPS/master/geogrid/GEOGRID.TBL.ARW",
        "license": "WPS open-source repository terms",
        "version": "official WPS repository master",
    },
    "METGRID.TBL.ARW": {
        "url": "https://raw.githubusercontent.com/wrf-model/WPS/master/metgrid/METGRID.TBL.ARW",
        "license": "WPS open-source repository terms",
        "version": "official WPS repository master",
    },
    "Vtable.GFS": {
        "url": "https://raw.githubusercontent.com/wrf-model/WPS/master/ungrib/Variable_Tables/Vtable.GFS",
        "license": "WPS open-source repository terms",
        "version": "official WPS repository master",
    },
    "icon_profile_cb6r3m_ae7_kmtbr.csv": {
        "url": "https://raw.githubusercontent.com/USEPA/CMAQ/main/PREP/icon/src/profile/avprofile_cb6r3m_ae7_kmtbr_hemi2016_v53beta2_m3dry_col051_row068.csv",
        "license": "CMAQ repository license",
        "version": "CMAQ repository main",
    },
    "bcon_profile_cb6r3m_ae7_kmtbr.csv": {
        "url": "https://raw.githubusercontent.com/USEPA/CMAQ/main/PREP/bcon/src/profile/avprofile_cb6r3m_ae7_kmtbr_hemi2016_v53beta2_m3dry_col051_row068.csv",
        "license": "CMAQ repository license",
        "version": "CMAQ repository main",
    },
    "ICON_README.md": {
        "url": "https://raw.githubusercontent.com/USEPA/CMAQ/main/PREP/icon/README.md",
        "license": "CMAQ repository license",
        "version": "CMAQ repository main",
    },
    "BCON_README.md": {
        "url": "https://raw.githubusercontent.com/USEPA/CMAQ/main/PREP/bcon/README.md",
        "license": "CMAQ repository license",
        "version": "CMAQ repository main",
    },
    "CMAQ_UG_tutorial_benchmark.md": {
        "url": "https://raw.githubusercontent.com/USEPA/CMAQ/main/DOCS/Users_Guide/Tutorials/CMAQ_UG_tutorial_benchmark.md",
        "license": "CMAQ repository license",
        "version": "CMAQ repository main",
    },
}

SWAT_BASE = "https://raw.githubusercontent.com/swat-model/swatplus/main/refdata/Ames_sub1/"


def entry(path: Path) -> dict:
    data = path.read_bytes()
    source = SOURCES.get(path.name)
    if source is None and path.parent.parent.name == "swat_plus":
        source = {
            "url": SWAT_BASE + path.name,
            "license": "SWAT+ repository license",
            "version": "SWAT+ official Ames_sub1 reference",
        }
    if source is None:
        source = {"url": "derived/organized from the model-specific source package", "license": "see sibling source", "version": "see sibling source"}
    return {
        "path": str(path.relative_to(REAL_CASES)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        **source,
    }


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    for model_dir in sorted(path for path in REAL_CASES.iterdir() if path.is_dir()):
        files = [path for path in model_dir.rglob("*") if path.is_file()]
        manifest = {
            "model": model_dir.name,
            "generated_utc": now,
            "files": [entry(path) for path in files],
            "notes": {
                "calpuff": "No license-free official native test project was found; only official documentation/EULA is stored.",
                "camx": "The full official 2-day test-case tarballs exceed 3 GB and were intentionally not downloaded.",
                "cmaq": "The full CMAS benchmark package was not downloaded; small official ICON/BCON profiles were used.",
                "wrf": "Large reanalysis forcing was not downloaded; official namelist/WPS tables were retained.",
                "swat_plus": "Official Ames_sub1 input and selected output reference files were retained.",
            }.get(model_dir.name, ""),
        }
        (model_dir / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(model_dir.name, len(manifest["files"]))


if __name__ == "__main__":
    main()
