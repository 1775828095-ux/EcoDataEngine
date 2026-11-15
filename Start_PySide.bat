@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
py -3.12 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --no-cache-dir -e ".[test,terrain-components,composition-search]" PySide6-Essentials "netCDF4>=1.6,<2" certifi
if errorlevel 1 goto failed
:run
if not exist ".venv\Scripts\python.exe" goto failed
set "TEMP=%CD%\.desktop-tmp"
set "TMP=%TEMP%"
set "PYTHONPATH=%CD%\integrations\existing_air;%CD%\integrations\local_swat;%PYTHONPATH%"
if not exist "%TEMP%" mkdir "%TEMP%"
".venv\Scripts\python.exe" -B -m ecodata.desktop %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Launch failed. Install Python 3.12 x64 and run this file again.
pause
exit /b 1

