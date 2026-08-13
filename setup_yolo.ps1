param(
    [switch]$Upgrade,
    [switch]$SkipModels
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
if (-not $SkipModels) {
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & $pythonCommand -c "from ultralytics import YOLO, YOLOE; YOLO('yolo11n.pt'); YOLOE('yoloe-26n-seg-pf.pt')"
        if ($LASTEXITCODE -ne 0) {
            throw "YOLO model download failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    Write-Host 'YOLO dependencies and default models are ready.'
} else {
    Write-Host 'YOLO dependencies installed; model download was skipped.'
}
