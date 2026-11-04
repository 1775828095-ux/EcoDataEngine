$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$source = Join-Path $repo 'external/swatplus-editor'
$commit = '3606ac4c3b54585072a557382b8e3b60b44dd354'
if (-not (Test-Path -LiteralPath $source)) {
    git clone --filter=blob:none --sparse https://github.com/swat-model/swatplus-editor.git $source
    if ($LASTEXITCODE -ne 0) { throw 'Official source clone failed' }
    git -C $source sparse-checkout set src/api
    git -C $source fetch --depth 1 origin $commit
    git -C $source checkout --detach $commit
}
$revision = git -C $source rev-parse HEAD
if ($revision -ne $commit) { throw 'Existing Editor source differs; preserved without overwrite' }
$runtime = Join-Path $repo '.venv-swat-editor/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    & (Join-Path $repo '.venv/Scripts/python.exe') -m venv (Join-Path $repo '.venv-swat-editor')
    if ($LASTEXITCODE -ne 0) { throw 'Editor environment creation failed' }
}
& $runtime -m pip install 'peewee==3.18.3'
if ($LASTEXITCODE -ne 0) { throw 'Editor dependency installation failed' }
Write-Output "Official Editor4.0.0 input exporter ready; no SWAT+ executable required: $runtime"
