param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
foreach($name in @('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')) {
    if (-not [Environment]::GetEnvironmentVariable($name,'Process')) { [Environment]::SetEnvironmentVariable($name,'1','Process') }
}
$env:PYTHONPATH="$PSScriptRoot\src;$PSScriptRoot\integrations\existing_air;$PSScriptRoot\integrations\local_swat"
$key=[Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY','User')
if($key){$env:DEEPSEEK_API_KEY=$key}
Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\pythonw.exe" -ArgumentList @('-B','-m','ecodata.desktop') -WorkingDirectory $PSScriptRoot -WindowStyle Normal
