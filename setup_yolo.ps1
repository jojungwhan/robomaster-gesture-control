param(
    [switch]$Upgrade,
    [switch]$SkipModels,
    # Install a CUDA build of torch/torchvision for GPU inference. Requires an
    # NVIDIA GPU with a recent driver. Without this, YOLO runs on the CPU.
    [switch]$Gpu,
    # PyTorch CUDA wheel channel (see https://pytorch.org for the right one for
    # your driver). cu124 = CUDA 12.4; use cu121 for older drivers.
    [string]$CudaVersion = 'cu124'
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

if ($Gpu) {
    # ultralytics pulls in the CPU build of torch by default; replace it with the
    # matching CUDA wheels so Ultralytics uses the GPU (device=auto picks it up).
    Write-Host "Installing CUDA ($CudaVersion) build of torch/torchvision..."
    $torchArguments = @(
        '-m', 'pip', 'install', '--upgrade',
        '--index-url', "https://download.pytorch.org/whl/$CudaVersion",
        'torch', 'torchvision'
    )
    & $pythonCommand @torchArguments
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA torch installation failed with exit code $LASTEXITCODE"
    }
    & $pythonCommand -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'))"
    if ($LASTEXITCODE -ne 0) {
        throw "torch could not be imported after the CUDA install."
    }
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

if ($Gpu) {
    Write-Host 'GPU build installed. YOLO uses the GPU automatically (device auto).'
} else {
    Write-Host 'CPU build. Re-run with -Gpu on a machine with an NVIDIA GPU for real-time detection.'
}
