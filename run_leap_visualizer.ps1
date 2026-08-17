param(
    [switch]$Stop,
    [double]$Duration = 0,
    [int]$X = -1,
    [int]$Y = 72,
    [int]$Width = 620,
    [int]$Height = 680,
    [double]$Opacity = 0.90
)

$ErrorActionPreference = 'Stop'
$statePath = Join-Path $PSScriptRoot 'logs\leap_visualizer_state.json'
$shutdownPath = Join-Path $PSScriptRoot 'logs\control_center_shutdown.request'
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
        Set-Content -LiteralPath $shutdownPath -Value 'stop' -Encoding ASCII
        try {
            Wait-Process -Id $existing.Id -Timeout 8 -ErrorAction Stop
            Write-Host "Safely stopped RoboMaster Control Center (PID $($existing.Id))."
        } catch {
            Stop-Process -Id $existing.Id -Force -ErrorAction SilentlyContinue
            throw 'Control Center did not complete its safe shutdown within 8 seconds and was forced closed.'
        }
    } else {
        Write-Host 'RoboMaster Control Center is not running.'
    }
    Remove-Item -LiteralPath $shutdownPath -Force -ErrorAction SilentlyContinue
    exit 0
}

if ($existing) {
    Write-Host "RoboMaster Control Center is already running (PID $($existing.Id))."
    exit 0
}
if (-not (Test-Path -LiteralPath $pythonWindowed)) {
    throw "Python 3.12 windowed executable not found: $pythonWindowed"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $statePath) -Force | Out-Null
Remove-Item -LiteralPath $shutdownPath -Force -ErrorAction SilentlyContinue
$arguments = @(
    '-m', 'robomaster_gesture.leap_visualizer',
    '--x', $X,
    '--y', $Y,
    '--width', $Width,
    '--height', $Height,
    '--opacity', $Opacity,
    '--shutdown-file', $shutdownPath
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
    throw 'RoboMaster Control Center exited during startup.'
}
Write-Host "RoboMaster Control Center started (PID $($process.Id))."
Write-Host 'It opens without stealing focus; live mode returns focus to RoboMaster before movement.'
Write-Host 'Stop it with: .\run_control_center.ps1 -Stop'
