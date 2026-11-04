@echo off
setlocal
cd /d "%~dp0"
echo [EcoData Engine] Preparing Python 3.12 development environment...
set "PYTHON_CMD=python"
py -3.12 -c "import sys,struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8" >nul 2>nul && set "PYTHON_CMD=py -3.12"
%PYTHON_CMD% -c "import sys,struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8" >nul 2>nul || (echo [FAILED] Python 3.12 x64 is required. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" %PYTHON_CMD% -m venv .venv || goto :failed
call ".venv\Scripts\activate.bat" || goto :failed
python -m pip install --upgrade pip || goto :failed
python -m pip install -r requirements-lock.txt || goto :failed
python -m pip install -r requirements-verification.txt || goto :failed
python -m pip install -e . --no-deps || goto :failed
python -c "import ecodata, streamlit, xarray, geopandas, rasterio, cvxpy, torch, neuralop; print('[OK] Key dependencies imported successfully')" || goto :failed
echo [SUCCESS] Development environment is ready.
pause
exit /b 0
:failed
echo [FAILED] Setup did not complete. Review the messages above.
pause
exit /b 1
