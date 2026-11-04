@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (echo [FAILED] Run setup_dev.bat first. & pause & exit /b 1)
if not exist logs mkdir logs
echo [%date% %time%] Starting EcoData Engine>logs\dev_startup.log
start "EcoData Engine" /b ".venv\Scripts\streamlit.exe" run "src\ecodata\interface\app.py" --server.headless true --server.port 8501 --browser.gatherUsageStats false >>logs\dev_startup.log 2>&1
powershell -NoProfile -Command "$ok=$false; 1..60 | ForEach-Object { try { $r=Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501 -TimeoutSec 1; if($r.StatusCode -eq 200){$ok=$true;break} } catch {}; Start-Sleep -Seconds 1 }; if(-not $ok){exit 1}" || (echo [FAILED] Service did not become ready. See logs\dev_startup.log & pause & exit /b 1)
start "" "http://127.0.0.1:8501"
echo [SUCCESS] EcoData Engine is ready at http://127.0.0.1:8501
echo Startup log: logs\dev_startup.log
pause
