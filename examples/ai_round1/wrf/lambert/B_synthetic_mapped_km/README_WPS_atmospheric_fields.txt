WPS 4.6 intermediate v5 / IPROJ=3.
On WPS host rename ATM_YYYY-MM-DD_HH.wps to ATM:YYYY-MM-DD_HH.
Append ATM to existing metgrid fg_name and retain all required sources.
Dates, original levels and valid physical values are preserved; no time/vertical/spatial interpolation.
Earth-relative winds remain earth-relative (is_wind_grid_rel=false).
These mapped fields are not a complete WRF input chain. Required terrain/geogrid,
land use, compatible METGRID.TBL, surface/soil fields not supplied in this conversion and real.exe configuration
must still be supplied. Availability depends on selected physics and domain.
Official metgrid, real.exe and wrf.exe execution is not established by file export.
