param(
    [switch]$Stop,
    [double]$Duration = 0,
    [int]$X = -1,
    [int]$Y = 72,
    [double]$Opacity = 0.90
)

$ErrorActionPreference = 'Stop'
$statePath = Join-Path $PSScriptRoot 'logs\leap_visualizer_state.json'
$pythonRoot = 'C:\Program Files\Python312'
$pythonWindowed = Join-Path $pythonRoot 'pythonw.exe'

function Get-VisualizerProcess {
    if (-not (Test-Path -LiteralPath $statePath)) {
        return $null
    }
    try {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        $process = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
        if (-not $process) {
            return $null
        }
        $details = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
        if ($details.CommandLine -notlike '*robomaster_gesture.leap_visualizer*') {
            return $null
        }
        return $process
    } catch {
        return $null
    }
}

$existing = Get-VisualizerProcess
if ($Stop) {
    if ($existing) {
        Stop-Process -Id $existing.Id
        Write-Host "Stopped Leap hand overlay (PID $($existing.Id))."
    } else {
        Write-Host 'Leap hand overlay is not running.'
    }
    exit 0
}

if ($existing) {
    Write-Host "Leap hand overlay is already running (PID $($existing.Id))."
    exit 0
}
if (-not (Test-Path -LiteralPath $pythonWindowed)) {
    throw "Python 3.12 windowed executable not found: $pythonWindowed"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $statePath) -Force | Out-Null
$arguments = @(
    '-m', 'robomaster_gesture.leap_visualizer',
    '--x', $X,
    '--y', $Y,
    '--opacity', $Opacity
)
if ($Duration -gt 0) {
    $arguments += @('--duration', $Duration)
}

$process = Start-Process `
    -FilePath $pythonWindowed `
    -ArgumentList $arguments `
    -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden `
    -PassThru

[ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString('o')
} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8

Start-Sleep -Milliseconds 700
if ($process.HasExited) {
    throw 'Leap hand overlay exited during startup.'
}
Write-Host "Leap hand overlay started (PID $($process.Id))."
Write-Host 'It is always-on-top, click-through, and cannot take RoboMaster keyboard focus.'
Write-Host 'Stop it with: .\run_leap_visualizer.ps1 -Stop'
