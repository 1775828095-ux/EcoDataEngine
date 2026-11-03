# Third-party licenses

EcoData Engine does not relicense third-party packages. The authoritative
license text is installed with each Python distribution. The current direct
dependencies and their verified versions/licenses are recorded in
`docs/dependencies.md`.

Important families include BSD-3-Clause (NumPy, pandas, SciPy, GeoPandas),
Apache-2.0 (Xarray, Streamlit, CVXPY, cf-xarray, pint-xarray,
xarray-regrid, compliance-checker, xskillscore), LGPL-3.0 (GSTools), MIT
(OSQP and several utilities), and the licenses shipped with GDAL/PROJ-backed
Python wheels. Review the installed package metadata before redistribution.

FloPy is distributed as a United States Geological Survey public-domain work
unless a bundled component states otherwise. Preserve its notices and verify
the installed distribution metadata when redistributing a portable build.
# Optional neural reconstruction dependencies

- PyTorch — https://github.com/pytorch/pytorch — BSD-3-Clause — tensor training backend.
- NeuralOperator — https://github.com/neuraloperator/neuraloperator — MIT — official FNO implementation used by the optional fusion reconstruction experiment.
- Galileo — https://github.com/nasaharvest/galileo — MIT — optional official pretrained remote-sensing feature encoder. EcoData Engine calls the upstream `single_file_galileo.py`; it does not copy or rename the model implementation. The checked-out upstream license must remain with any redistributed weights/source.
- GOLEM / `thegolem` — https://github.com/aimclub/GOLEM — BSD-3-Clause — optional graph-optimisation backend. EcoData Engine installs version 0.5.0 in an isolated environment and calls its public optimiser, graph, objective, constraint and mutation APIs; the upstream implementation is not copied into this project.
- PyKrige 1.7.3 — https://github.com/GeoStat-Framework/PyKrige — BSD-3-Clause — optional station interpolation adapters.
- MetPy 1.7.1 — https://github.com/Unidata/MetPy — BSD-3-Clause — optional point and vertical interpolation adapters.
- xsdba 0.5.0 — https://github.com/Ouranosinc/xsdba — Apache-2.0 — optional statistical bias correction adapters.
- xclim 0.59.1 — https://github.com/Ouranosinc/xclim — Apache-2.0 — optional daily semantic statistics, isolated subprocess.
- ForDs pressure component — https://github.com/leelew/ForDs — MIT — unchanged `offline/topo_factory.py` and `offline/utils.py`, pinned c533dfa4aa65089da2dcc0035aeda536094bfe53. Upstream LICENSE and SHA256 manifest retained in `external/fords/`. This is not the complete ForDs/CoLM workflow.
- DualTimesField — https://github.com/WisdomTogether/DualTimesField — upstream README declares MIT License at 6a2263fd6b7cee4f0a662406df0eb850543fe086; standalone LICENSE not present in inspected tree. Original README, source and provenance retained in `external/dualtimesfield/`. Review complete notices before redistribution. Official interpolation model is called without modifications; README publication claims are not local accuracy evidence.
- scikit-learn 1.9.1 — https://github.com/scikit-learn/scikit-learn — BSD-3-Clause — optional ForDs import dependency.
- xarray-regrid 0.4.2 — https://github.com/xarray-contrib/xarray-regrid — Apache-2.0 — existing grid kernels, with new aggregation adapters.
- ESA WorldCover 2021 v200 subset — https://esa-worldcover.org/en/data-access — CC-BY-4.0 — small actual land-cover test window; cite ESA WorldCover when redistributing this test data.

## Algorithm Library Expansion V4

- ClimateLearn — https://github.com/aditya-grover/climate-learn — MIT, commit b48fb0242acc47e365af86bfbd9dd86e9dcbd6d2. Unmodified ResNet, CNN blocks and LitModule are retained with LICENSE and provenance under external/climatelearn. V4 weights are locally trained, not official pretrained weights.
- PyTorch Lightning 2.5.6 — https://github.com/Lightning-AI/pytorch-lightning — Apache-2.0; TorchMetrics 1.8.2 — https://github.com/Lightning-AI/torchmetrics — Apache-2.0. Installed only in the optional V4 environment; preserve distribution notices when redistributing.
- ForDs temperature component — same MIT source and pinned commit as above. Original lapse-rate/temperature functions are called; only the CDO file/interpolation hook is replaced with xarray interpolation. This does not verify the complete ForDs workflow.
- ClimateDiffuse — https://github.com/robbiewatt1/ClimateDiffuse — license unresolved in inspected commit 7b8ee181fa3baeacf08a2bf9dc275989048b4030; LFS pointers inspected, neither model weight downloaded. Not licensed here for unrestricted deployment.
- ERA5 subsets retain ECMWF/Copernicus attribution and Google ARCO source metadata. USGS DEM source attribution remains in the existing source manifest. Data attribution does not imply model accuracy or exact vertical-datum equivalence.

## Algorithm Library Expansion V5

- LIIF — https://github.com/yinboc/liif — BSD-3-Clause, commit 4c833f8f9d5d038df3ee2b50ebe649c30e628c0f. Original source and LICENSE retained in external/liif. Runtime-only device adaptation changes one CUDA placement to the input tensor device; no core network rewrite. Weights are locally trained ERA5 weights, not the DIV2K pretrained model.
- pyresample 1.35.0 — https://github.com/pytroll/pyresample — LGPL-3.0-or-later; commit/source evidence under output/algorithm_library_expansion/upstream/pytroll__pyresample. Called as an installed package in optional V5 environment; retain LGPL notices/source obligations when redistributing.
- emiproc 2.10.0 — https://github.com/C2SM-RCM/emiproc — BSD-3-Clause; official spatial allocation and temporal profile APIs, not a verified native air-quality-model package.
- mpspline2 0.1.9 — https://github.com/obrl-soil/mpspline2 — GPL as declared in DESCRIPTION. Called through separate R process; installed core function bodies checked against official pinned source. R 4.5.2 from CRAN and aqp 2.3.2 (GPL >=3) remain independent runtime packages with their original licenses. Do not omit GPL notices/source obligations from redistributed runtime copies.
- aqp::sp2 — Honcut Creek soil morphologic/field pH data, documented source USGS Bulletin 1590-G, Busacca/Singer/Verosub (1989). Retain original package/data metadata; field pH is not a conserved mass variable.
- HERMES_GR source audit — Apache-2.0; not installed or executed here. CMAQ official PREP audit — MIT; ICON/BCON not executed. SWAT+ Editor source audit retains its public-domain/service disclaimer and any component-specific licenses; Editor backend was not executed or added to the main environment.

## Final C1-C30 acceptance

- EPA S2S-Tool — https://github.com/USEPA/S2S-Tool — MIT, commit ba5005aab36832248832012ae388b7ae897c2564. Original source, license and official input/output tables retained under output/capability_final_acceptance/official_s2s. Official code executed with only the documented mechanism setting changed to CB6R5_AE7. Preserve EPA/SPECIATE data provenance; the 1kg profile basis is not a measured emissions inventory.
- PseudoNetCDF 3.4.1 — https://github.com/barronh/pseudonetcdf — BSD-3-Clause; exact wheel retained with hash. Calls public icon_profile/toGriddedFile and bcon_profile/toPerim APIs using its original reference profiles and GRIDDESC. Adapter supplies explicit project time metadata; no library source altered. These legacy static-profile outputs are not official CMAQ native-run verification.
- aqmbc 0.5.0 — https://github.com/barronh/aqmbc — GPL-3.0, commit 9da3eb028817fc3a454fb296c9c0f1cff93c6d21, installed only in the optional V6 environment. Three upstream metadata/expression tests executed; no claim of a dynamic 3D chemistry boundary run. Retain GPL notices/source obligations when redistributing.
- Real EDGAR v8.1 CO 2014 Paris subset remains attributed to European Commission/JRC, https://edgar.jrc.ec.europa.eu/dataset_ap81 . Read the dataset's usage conditions before redistribution. The explicit uniform hourly profile is a modelling assumption applied to an equivalent mean day, not observed hourly emissions.


## Complementary-resolution composition search (2026-10-24)

- alns 7.0.0 — https://github.com/N-Wouda/ALNS — MIT. Official ALNS, AlphaUCB operator selection and HillClimbing acceptance; domain operators remain project adapters.
- pymoo 0.6.2 — https://github.com/anyoptimization/pymoo — Apache-2.0. Official MixedVariableGA used as a controlled optimiser comparison.
- License copies and inspected source versions: output/complementary_20261024/upstream and frozen_protocol.json. STARFM/cuESTARFM were reviewed only, not installed, copied or claimed as reproduced.

## Complementary raster operators (2026-10-29)

- scikit-image 0.26.0 — https://github.com/scikit-image/scikit-image — BSD-3-Clause. Calls restoration.denoise_tv_chambolle, denoise_bilateral and estimate_sigma; upstream numerical kernels are unchanged. The adapter rescales bilateral-filter intensities and restores physical units.
- PyWavelets 1.10.0 — https://github.com/PyWavelets/pywt — MIT AND BSD-3-Clause as declared by the installed distribution. Calls swt2/iswt2 with db2; the two-source detail-transfer formula is a local composition, not a claim of reproducing STARFM or another named fusion model.
- xsdba 0.5.0 — https://github.com/Ouranosinc/xsdba — Apache-2.0. Reuses the existing isolated official adjustment runner and adds Scaling.train/adjust; reference data are historical training pairs only.
- Installed-package license texts and version evidence are retained under output/complementary_library_20261029/licenses. Preserve the complete upstream notices when redistributing runtime dependencies. These library licenses do not replace input satellite-data attribution.

## Five complementary algorithm classes (2026-10-29)

- VAM Whittaker 2.0.6 — https://pypi.org/project/vam.whittaker/2.0.6/ — MIT. Official Windows wheel and source distribution retained and checked against PyPI SHA256. Its NumPy1 ABI is loaded only in an isolated process/package path.
- sinkr 0.8.0 — https://github.com/marchtaylor/sinkr — GPL-3.0, commit dd3f9e119ee6a41e4e44f4be2051914c223f2bb6. Unchanged R/dineof.R is called through the existing R 4.5.2 runtime with base SVD. LICENSE and DESCRIPTION retained in external/sinkr. The separate Fortran DINEOF source was inspected, not claimed as the executed backend.
- OpenCV contrib headless 5.0.0.93 — https://github.com/opencv/opencv_contrib — Apache-2.0, with packaged third-party notices retained. Calls ximgproc.guidedFilter.
- AROSICS 1.13.2 — https://github.com/GFZ/arosics — Apache-2.0. Actual conda-forge runtime uses GDAL 3.13.1 and GeoArray; exact package URLs retained in output/complementary_five_20261029/geo_runtime_explicit.txt. The separately inspected newer repository snapshot is not misrepresented as the executed version. Preserve all GDAL and transitive component notices on redistribution.
- starfm4py — https://github.com/nmileva/starfm4py — GPL-3.0, commit b1a917f68f88fe9f274d99d29a8e0eaac24892b3. Original source and LICENSE retained in external/starfm4py; predict kernel unchanged. Local adapter provides streamed windows and isolated documented parameters.
- Dependency license copies, original source archives, wheel hashes and acceptance evidence are retained in output/complementary_five_20261029. No upstream publication claim is treated as local accuracy evidence.
