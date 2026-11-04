param([string]$Python='')
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
if(-not $Python){
 if(Test-Path -LiteralPath '_runtime/python.exe'){$Python=(Join-Path $PSScriptRoot '_runtime/python.exe')}
 elseif(Test-Path -LiteralPath '.venv/Scripts/python.exe'){$Python=(Join-Path $PSScriptRoot '.venv/Scripts/python.exe')}
 else{$Python='python'}
}
$env:PYTHONPATH="$PSScriptRoot/src;$PSScriptRoot"
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'logs/round1_temp') | Out-Null
$env:TEMP=(Join-Path $PSScriptRoot 'logs/round1_temp')
$env:TMP=$env:TEMP
& $Python -X utf8 -u -m examples.ai_round1.run_round1
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& $Python -X utf8 -m pytest tests/test_round1_formats.py tests/test_round1_cmaq_regressions.py tests/test_round1_ui.py tests/test_round1_ioapi_errors.py -q --basetemp=.pytest_tmp_round1 --junitxml=logs/round1_regression.xml
exit $LASTEXITCODE
