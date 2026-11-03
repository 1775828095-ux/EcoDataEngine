$ErrorActionPreference = 'Stop'
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$rawDirectory = Join-Path $projectDirectory 'output\three_experiments_20261019\anonymous\original'
$auditDirectory = Join-Path $projectDirectory 'output\taiwan_2017_preparation_20261019'
$rows = Import-Csv -LiteralPath (Join-Path $auditDirectory 'cleanup_candidates.csv')
$deleted = @()
foreach ($row in $rows) {
    if ($row.eligible -ne 'True') { continue }
    $target = [IO.Path]::GetFullPath($row.path)
    if ([IO.Path]::GetDirectoryName($target) -ne $rawDirectory) { throw 'Unexpected deletion directory' }
    if (-not $target.StartsWith($projectDirectory + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Outside project' }
    if (-not (Test-Path -LiteralPath $target)) { continue }
    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $row.sha256) { throw 'Checksum changed' }
    Remove-Item -LiteralPath $target -Force
    $deleted += [pscustomobject]@{path=$target;bytes=[long]$row.bytes;sha256=$actual;deleted=$true}
}
$deleted | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $auditDirectory 'cleanup_execution.json') -Encoding UTF8
$deleted | Measure-Object -Property bytes -Sum
