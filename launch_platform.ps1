param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$python = Join-Path $appRoot '.venv\Scripts\python.exe'
$application = Join-Path $appRoot 'src\ecodata\interface\app.py'
$url = 'http://127.0.0.1:8502'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing runtime: $python" }
$listeners = @(Get-NetTCPConnection -LocalPort 8502 -State Listen -ErrorAction SilentlyContinue)
$ready = $false
if ($listeners.Count -gt 0) {
    $server = Get-CimInstance Win32_Process -Filter "ProcessId=$($listeners[0].OwningProcess)"
    if (-not $server -or $server.CommandLine -notlike "*$application*") {
        throw 'Port 8502 is occupied by another application. Close it before starting EcoData Engine.'
    }
    try { $ready = (Invoke-WebRequest "$url/_stcore/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch {}
}
if (-not $ready) {
    if ($listeners.Count -gt 0) { throw 'EcoData Engine is running but its health check failed. Check outputs\launcher.' }
    $env:PYTHONPATH = "$appRoot\src;$appRoot\integrations\existing_air"
    $logRoot = Join-Path $appRoot 'outputs\launcher'
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    Start-Process -FilePath $python -ArgumentList @(
        '-m', 'streamlit', 'run', $application,
        '--server.address', '127.0.0.1', '--server.port', '8502',
        '--server.headless', 'true', '--server.fileWatcherType', 'poll',
        '--browser.gatherUsageStats', 'false'
    ) -WorkingDirectory $appRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') -RedirectStandardError (Join-Path $logRoot 'stderr.log') | Out-Null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try { $ready = (Invoke-WebRequest "$url/_stcore/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch {}
        if ($ready) { break }
    }
    if (-not $ready) { throw 'EcoData Engine did not become ready. Check outputs\launcher.' }
}
if (-not $NoBrowser) { Start-Process ($url + '/?fresh=' + [DateTimeOffset]::Now.ToUnixTimeMilliseconds()) }
