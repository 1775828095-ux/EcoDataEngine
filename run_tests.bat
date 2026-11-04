@echo off
setlocal
cd /d "%~dp0"
set "TEST_PYTHON=.venv\Scripts\python.exe"
if not exist "%TEST_PYTHON%" set "TEST_PYTHON=_runtime\python.exe"
if not exist "%TEST_PYTHON%" (echo [FAILED] Run setup_dev.bat first. & pause & exit /b 1)
"%TEST_PYTHON%" tools\run_regression.py
set code=%errorlevel%
if not "%code%"=="0" (echo [FAILED] Tests failed or errored. Exit code: %code% & pause & exit /b %code%)
echo [SUCCESS] All required tests passed. The pytest summary above lists passed and skipped tests.
pause
exit /b 0
