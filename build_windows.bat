@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo [FAILED] Run setup_dev.bat first. & pause & exit /b 1)
call ".venv\Scripts\activate.bat"
python -c "import streamlit,xarray,geopandas,rasterio,cvxpy" || (echo [FAILED] Dependencies are incomplete. & pause & exit /b 1)
python -m pytest || (echo [FAILED] Tests failed. Packaging stopped. & pause & exit /b 1)
python tools\build_release.py portable || (echo [FAILED] Portable build failed. & pause & exit /b 1)
echo [SUCCESS] dist\EcoDataEngine_Windows_Portable.zip created.
pause
