@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_CMD=py -3.12"
%PYTHON_CMD% -c "import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8" >nul 2>nul
if errorlevel 1 set "PYTHON_CMD=python"
%PYTHON_CMD% -c "import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8" >nul 2>nul
if errorlevel 1 goto :failed
if not exist ".venv\Scripts\python.exe" %PYTHON_CMD% -m venv .venv
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -c requirements-handoff-constraints.txt -e ".[test,ml,galileo,verification,composition-search,environment-algorithms,raster-complements,advanced-complements,climate-statistics,terrain-components]"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --no-deps -e integrations/existing_air -e integrations/local_swat
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" verify_handoff.py
if errorlevel 1 goto :failed
echo [OK] Use start_handoff.bat to open EcoData.
pause
exit /b 0
:failed
echo [FAILED] Python 3.12 x64 and internet access are required. Review the error above.
pause
exit /b 1
