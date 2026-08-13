param(
    [switch]$Upgrade
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonCommand = Join-Path $workspaceRoot '.venv-robomaster\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonCommand)) {
    throw "Python environment not found: $pythonCommand"
}

$arguments = @('-m', 'pip', 'install', '-r', (Join-Path $PSScriptRoot 'requirements-yolo.txt'))
if ($Upgrade) { $arguments += '--upgrade' }
& $pythonCommand @arguments
if ($LASTEXITCODE -ne 0) {
    throw "YOLO dependency installation failed with exit code $LASTEXITCODE"
}
Write-Host 'YOLO dependencies installed. Model weights download on first use.'
