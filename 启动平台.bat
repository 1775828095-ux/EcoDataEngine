@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0src
python -m streamlit run src/ecodata/interface/app.py --server.address 127.0.0.1 --server.port 8524
pause
