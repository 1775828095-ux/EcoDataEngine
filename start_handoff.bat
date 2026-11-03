@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :failed
echo Open http://127.0.0.1:8501 after the server starts. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m streamlit run src/ecodata/interface/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --server.fileWatcherType poll --browser.gatherUsageStats false
pause
exit /b 0
:failed
echo [FAILED] Run setup_handoff.bat first.
pause
exit /b 1
