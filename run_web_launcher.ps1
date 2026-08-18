param(
    [string]$BindHost = '127.0.0.1',
    [int]$Port = 8770,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$python = 'C:\Program Files\Python312\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python 3.12 not found: $python"
}

$arguments = @(
    (Join-Path $PSScriptRoot 'web_launcher.py'),
    '--host', $BindHost,
    '--port', $Port
)
if ($NoBrowser) { $arguments += '--no-browser' }

Write-Host "Starting RoboMaster Control Center web launcher on http://${BindHost}:$Port ..."
Write-Host 'Keep this window open; press Ctrl+C to stop the launcher.'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
